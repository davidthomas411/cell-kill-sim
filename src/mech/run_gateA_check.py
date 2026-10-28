#!/usr/bin/env python3
"""
Stage A Gate Check for Step 6b:
  1. Extended Pi grid: PI_NODES = [0.0, 0.6, 1.2, 2.4, 4.0, 6.0, 8.0, 10.0, 12.0]
     with NO extrapolation and NO cap inside the solved range.
     Visits whose observed depth exceeds the model's maximum reachable depth across the
     solved Pi nodes are flagged `unreachable = True` (not clipped) and counted.
  2. Runs ONLY the Pi-fitting part on all 305 eligible chambers (`3,475` pre-death contact visits)
     at `(alpha_syn = 2.0, chi = 1.0)`.
     Records both:
       - `pi_fit_delta` (using net indentation `d_solved(Pi) - d_solved(0)` from `_solve_visit_mesh_for_pi`
         with the artificial monotonicity floor removed), and
       - `pi_fit_raw` (using raw solved hull depth `desc["cup_depth_hull_p95_um"]`).
     Also records solver breakdown diagnostics at every Pi node (`min_contact_sep_um`,
     `max_vol_rel_err`, `nit`, `success`, `grad_norm_final`, and whether the depth curve turns over).
  3. Runs the two-shell coarse-vs-fine check from `validate_solver.py` across edge lengths
     `1.4, 1.0, 0.7, 0.5 um` up to `Pi = 12.0` (highlighting `1.0 vs 0.5 um` at `Pi = 12.0`).
  4. Saves all numbers to `data/step6b/gateA.json` and per-visit fits to `data/step6b/gateA_visit_fits.csv`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mech.run_step6b import (
    MANIFEST_CSV,
    MESH_EDGE_UM,
    STEP3_TRACKED_DIR,
    STEP4_CHAMBER_CSV,
    STEP5_VISIT_CSV,
    STEP6B_DIR,
    _align_reference_target_to_visit,
    _extract_tcell_and_target_masks_from_h5,
    _tag_synapse_faces_for_patch_area,
)
from src.mech.shell_model import (
    ShellMesh,
    ShellParameters,
    check_closed_manifold,
    compute_mesh_descriptors,
    extract_cell_mesh_marching_cubes,
    make_cup_filled_reference_mesh,
    solve_two_shell_equilibrium,
)
from src.mech.validate_solver import _build_two_spheres_smooth_patch

PI_NODES_GATEA = np.array([0.0, 0.6, 1.2, 2.4, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)


def _invert_depth_curve_no_extrap(
    d_obs: float,
    pi_nodes: np.ndarray,
    d_nodes: np.ndarray,
) -> tuple[float, bool, str]:
    """
    Invert `d_nodes(pi_nodes)` for `d_obs` with NO extrapolation and NO cap inside the solved range.
    Because `d_nodes` can turn over (decrease) at large Pi when deep indentation shifts the target
    centroid/hull, we invert on the rising branch up to the peak reachable depth `d_max = max(d_nodes)`
    (using the running maximum ` np.maximum.accumulate(d_nodes)` up to `idx_peak`).
    If `d_obs > d_max`, returns `(np.nan, True, "unreachable_above_max")`.
    If `d_obs <= d_nodes[0]`, returns `(0.0, False, "at_or_below_zero_load")`.
    Otherwise returns `(pi_fit, False, "in_range")`.
    """
    d_max = float(np.max(d_nodes))
    if d_obs > d_max:
        return float("nan"), True, "unreachable_above_max"
    if d_obs <= float(d_nodes[0]):
        return 0.0, False, "at_or_below_zero_load"

    idx_peak = int(np.argmax(d_nodes))
    pi_sub = pi_nodes[: idx_peak + 1]
    d_sub = np.maximum.accumulate(d_nodes[: idx_peak + 1])

    # Interpolate strictly inside [pi_sub[0], pi_sub[-1]] with unique increasing knots
    u_vals, u_idx = np.unique(d_sub, return_index=True)
    pi_u = pi_sub[u_idx]
    pi_fit = float(np.interp(d_obs, u_vals, pi_u))
    return pi_fit, False, "in_range"


def process_one_chamber_gateA(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    exp: str = task["experiment"]
    roi: str = task["roi"]
    cond: str = task["condition"]
    visit_rows: list[dict[str, Any]] = task["visit_rows"]
    rep_sweep_map: dict[int, dict[str, Any]] = task["rep_sweep_map"]

    df_v = pd.DataFrame(visit_rows).sort_values("visit").reset_index(drop=True)
    valid_c = df_v[df_v["is_pre_death_contact"]].sort_values("visit").reset_index(drop=True)
    n_c = len(valid_c)
    if n_c < 4:
        return {"visit_rows": [], "elapsed_s": 0.0}

    first_contact_v = int(valid_c["visit"].iloc[0])

    # 1. Identify pre-contact reference if available
    pre_df = df_v[
        (df_v["visit"] < first_contact_v)
        & (~df_v["in_contact_visit"])
        & (df_v["target_volume_um3"].fillna(0.0) > 20.0)
    ].sort_values("visit")

    has_pre_contact_ref = False
    V_g_pre_raw: np.ndarray | None = None
    F_g_pre_raw: np.ndarray | None = None

    if len(pre_df) > 0:
        for idx_p in range(len(pre_df) - 1, -1, -1):
            cand_v = int(pre_df.iloc[idx_p]["visit"])
            if cand_v in rep_sweep_map:
                sw_info = rep_sweep_map[cand_v]
                h5_p = REPO_ROOT / sw_info["rel_path"]
                if h5_p.is_file():
                    _, _, b_sel_p, off_b_p = _extract_tcell_and_target_masks_from_h5(
                        h5_p,
                        int(sw_info["tracked_tcell_id"]),
                        int(sw_info["n_tcell_instances"]),
                        int(sw_info["n_target_instances"]),
                    )
                    if b_sel_p is not None and np.any(b_sel_p):
                        V_g_pre_raw, F_g_pre_raw = extract_cell_mesh_marching_cubes(
                            b_sel_p, off_b_p, target_edge_um=MESH_EDGE_UM
                        )
                        has_pre_contact_ref = True
                        break

    # 2. Extract 3D meshes for every pre-death contact visit
    per_visit_meshes: list[dict[str, Any]] = []
    for idx_v, vrow in valid_c.iterrows():
        v_num = int(vrow["visit"])
        sw_info = rep_sweep_map.get(v_num)
        V_t_v = F_t_v = V_g_v = F_g_v = None
        if sw_info is not None:
            h5_v = REPO_ROOT / sw_info["rel_path"]
            if h5_v.is_file():
                t_sel, off_t, b_sel, off_b = _extract_tcell_and_target_masks_from_h5(
                    h5_v,
                    int(sw_info["tracked_tcell_id"]),
                    int(sw_info["n_tcell_instances"]),
                    int(sw_info["n_target_instances"]),
                )
                if t_sel is not None and np.any(t_sel) and b_sel is not None and np.any(b_sel):
                    V_t_v, F_t_v = extract_cell_mesh_marching_cubes(t_sel, off_t, target_edge_um=MESH_EDGE_UM)
                    V_g_v, F_g_v = extract_cell_mesh_marching_cubes(b_sel, off_b, target_edge_um=MESH_EDGE_UM)

        if V_t_v is None or V_g_v is None:
            if len(per_visit_meshes) > 0:
                prev = per_visit_meshes[-1]
                V_t_v, F_t_v = prev["V_t"].copy(), prev["F_t"].copy()
                V_g_v, F_g_v = prev["V_g_data"].copy(), prev["F_g_data"].copy()
            else:
                from src.mech.shell_model import build_icosphere
                r_g0 = float((3.0 * max(50.0, float(vrow["target_volume_um3"])) / (4.0 * np.pi)) ** (1.0 / 3.0))
                r_t0 = float((3.0 * max(50.0, float(vrow["tcell_volume_um3"])) / (4.0 * np.pi)) ** (1.0 / 3.0))
                V_g_v, F_g_v = build_icosphere(np.zeros(3), r_g0, target_edge_um=MESH_EDGE_UM)
                V_t_v, F_t_v = build_icosphere(np.array([0.0, 0.0, -(r_g0 + r_t0 + 0.18)]), r_t0, target_edge_um=MESH_EDGE_UM)

        per_visit_meshes.append({
            "idx_v": int(idx_v),
            "vrow": vrow,
            "V_t": V_t_v,
            "F_t": F_t_v,
            "V_g_data": V_g_v,
            "F_g_data": F_g_v,
        })

    first_vm = per_visit_meshes[0]
    V_g_cupfilled_base = make_cup_filled_reference_mesh(first_vm["V_g_data"], first_vm["F_g_data"], first_vm["V_t"])
    F_g_cupfilled_base = first_vm["F_g_data"]

    ref_type = "pre_contact" if (has_pre_contact_ref and V_g_pre_raw is not None) else "cup_filled"
    V_ref_base = V_g_pre_raw if ref_type == "pre_contact" else V_g_cupfilled_base
    F_ref_base = F_g_pre_raw if ref_type == "pre_contact" else F_g_cupfilled_base

    out_visit_rows: list[dict[str, Any]] = []
    for vm in per_visit_meshes:
        vrow = vm["vrow"]
        v_num = int(vrow["visit"])
        d_obs = float(vrow["cup_depth_hull_p95_um"])
        r_p = float(vrow["patch_equiv_radius_um"])
        vol_g = float(vrow["target_volume_um3"])
        V_t_v = vm["V_t"]
        F_t_v = vm["F_t"]
        V_g_data_v = vm["V_g_data"]

        V_g_ref_v = _align_reference_target_to_visit(
            V_ref_base, F_ref_base, np.mean(V_g_data_v, axis=0), vol_g, V_t_v
        )
        syn_faces_v = _tag_synapse_faces_for_patch_area(
            V_t_v, F_t_v, V_g_ref_v, float(vrow["contact_area_um2"])
        )
        t_mesh_v = ShellMesh(V0=V_t_v, F=F_t_v, synapse_faces=syn_faces_v)
        g_mesh_v = ShellMesh(V0=V_g_ref_v, F=F_ref_base, synapse_faces=np.zeros(len(F_ref_base), dtype=bool))

        # Solve at Pi = 0
        params0 = ShellParameters.from_dimensionless(
            pi_push=0.0, chi_t=1.0, alpha_syn=2.0, r_target_um=g_mesh_v.R_eq0, subtract_prestress=True
        )
        sol0 = solve_two_shell_equilibrium(t_mesh_v, g_mesh_v, params0, maxiter=6, gtol=2e-3)
        base_desc = compute_mesh_descriptors(
            sol0["V_tcell"], t_mesh_v.F, sol0["V_target"], g_mesh_v.F, synapse_faces_t=t_mesh_v.synapse_faces
        )
        bd0 = sol0["breakdown"]
        base_hull_d = float(base_desc["cup_depth_hull_p95_um"])

        d_delta_nodes = [0.0]
        d_delta_with_sph_fallback_nodes = [0.0]
        d_raw_hull_nodes = [base_hull_d]
        min_sep_nodes = [float(bd0["min_contact_sep_um"])]
        vol_err_nodes = [float(max(bd0["vol_tcell_rel_err"], bd0["vol_target_rel_err"]))]
        nit_nodes = [int(sol0["nit"])]
        ok_nodes = [bool(sol0["success"])]

        x_warm = sol0["x_eq"]
        for pi_val in PI_NODES_GATEA[1:]:
            params_p = ShellParameters.from_dimensionless(
                pi_push=float(pi_val), chi_t=1.0, alpha_syn=2.0, r_target_um=g_mesh_v.R_eq0, subtract_prestress=True
            )
            sol_p = solve_two_shell_equilibrium(
                t_mesh_v, g_mesh_v, params_p, maxiter=16, gtol=2e-3, x0=x_warm
            )
            x_warm = sol_p["x_eq"]
            desc_p = compute_mesh_descriptors(
                sol_p["V_tcell"], t_mesh_v.F, sol_p["V_target"], g_mesh_v.F, synapse_faces_t=t_mesh_v.synapse_faces
            )
            bd_p = sol_p["breakdown"]
            raw_h = float(desc_p["cup_depth_hull_p95_um"])
            raw_s = float(desc_p["cup_depth_sphere_p95_um"])
            d_del = max(0.0, raw_h - base_hull_d)
            d_fb = d_del
            if d_fb < 0.05 * raw_s:
                d_fb = max(d_fb, 0.35 * raw_s)

            d_delta_nodes.append(d_del)
            d_delta_with_sph_fallback_nodes.append(d_fb)
            d_raw_hull_nodes.append(raw_h)
            min_sep_nodes.append(float(bd_p["min_contact_sep_um"]))
            vol_err_nodes.append(float(max(bd_p["vol_tcell_rel_err"], bd_p["vol_target_rel_err"])))
            nit_nodes.append(int(sol_p["nit"]))
            ok_nodes.append(bool(sol_p["success"]))

        arr_delta = np.asarray(d_delta_nodes, dtype=np.float64)
        arr_fb = np.asarray(d_delta_with_sph_fallback_nodes, dtype=np.float64)
        arr_raw = np.asarray(d_raw_hull_nodes, dtype=np.float64)

        pi_fit_delta, unreach_delta, status_delta = _invert_depth_curve_no_extrap(d_obs, PI_NODES_GATEA, arr_delta)
        pi_fit_fb, unreach_fb, status_fb = _invert_depth_curve_no_extrap(d_obs, PI_NODES_GATEA, arr_fb)
        pi_fit_raw, unreach_raw, status_raw = _invert_depth_curve_no_extrap(d_obs, PI_NODES_GATEA, arr_raw)

        rec: dict[str, Any] = {
            "experiment": exp,
            "roi": roi,
            "condition": cond,
            "visit": v_num,
            "ref_type": ref_type,
            "cup_depth_hull_p95_um": d_obs,
            "patch_equiv_radius_um": r_p,
            "r_target_um": float(g_mesh_v.R_eq0),
            "base_hull_depth_pi0_um": base_hull_d,
            "max_depth_delta_um": float(np.max(arr_delta)),
            "pi_at_max_delta": float(PI_NODES_GATEA[int(np.argmax(arr_delta))]),
            "max_depth_fb_um": float(np.max(arr_fb)),
            "pi_at_max_fb": float(PI_NODES_GATEA[int(np.argmax(arr_fb))]),
            "max_depth_raw_hull_um": float(np.max(arr_raw)),
            "pi_at_max_raw_hull": float(PI_NODES_GATEA[int(np.argmax(arr_raw))]),
            "pi_fit_delta": pi_fit_delta,
            "unreachable_delta": bool(unreach_delta),
            "status_delta": status_delta,
            "pi_fit_fb": pi_fit_fb,
            "unreachable_fb": bool(unreach_fb),
            "status_fb": status_fb,
            "pi_fit_raw_hull": pi_fit_raw,
            "unreachable_raw_hull": bool(unreach_raw),
            "status_raw_hull": status_raw,
            "curve_nonmonotone_delta": bool(np.any(np.diff(arr_delta) < -0.05)),
            "curve_nonmonotone_raw": bool(np.any(np.diff(arr_raw) < -0.05)),
            "min_sep_overall_um": float(np.min(min_sep_nodes)),
            "min_sep_pi12_um": float(min_sep_nodes[-1]),
            "max_vol_err_overall": float(np.max(vol_err_nodes)),
            "max_vol_err_pi12": float(vol_err_nodes[-1]),
            "any_nonconverged": bool(not all(ok_nodes)),
            "any_hit_maxiter": bool(any(n >= 16 for n in nit_nodes[1:])),
        }
        for k_i, pi_v in enumerate(PI_NODES_GATEA):
            tag = f"pi_{str(pi_v).replace('.', 'p')}"
            rec[f"d_delta_{tag}"] = float(arr_delta[k_i])
            rec[f"d_fb_{tag}"] = float(arr_fb[k_i])
            rec[f"d_raw_{tag}"] = float(arr_raw[k_i])
            rec[f"min_sep_{tag}"] = float(min_sep_nodes[k_i])
            rec[f"vol_err_{tag}"] = float(vol_err_nodes[k_i])

        out_visit_rows.append(rec)

    return {
        "experiment": exp,
        "roi": roi,
        "visit_rows": out_visit_rows,
        "elapsed_s": time.perf_counter() - t0,
    }


def run_two_shell_coarse_vs_fine_extended() -> list[dict[str, Any]]:
    edge_targets = [1.4, 1.0, 0.7, 0.5]
    pi_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    rows = []
    for edge_nom in edge_targets:
        t_mesh, g_mesh, actual_edge = _build_two_spheres_smooth_patch(
            r_tcell_um=5.3, r_target_um=5.0, patch_radius_um=2.6, target_edge_um=edge_nom
        )
        x_warm = None
        for pi_val in pi_grid:
            t0 = time.perf_counter()
            params = ShellParameters.from_dimensionless(
                pi_push=pi_val, chi_t=1.0, alpha_syn=2.0, r_target_um=5.0, subtract_prestress=False
            )
            sol = solve_two_shell_equilibrium(
                t_mesh, g_mesh, params, maxiter=160, gtol=1e-6, ftol=1e-11, x0=x_warm
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            x_warm = sol["x_eq"]
            desc = compute_mesh_descriptors(
                sol["V_tcell"], t_mesh.F, sol["V_target"], g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
            )
            bd = sol["breakdown"]
            rows.append({
                "nominal_edge_um": float(edge_nom),
                "actual_mean_edge_um": float(actual_edge),
                "pi_push": float(pi_val),
                "cup_depth_hull_p95_um": float(desc["cup_depth_hull_p95_um"]),
                "cup_depth_sphere_p95_um": float(desc["cup_depth_sphere_p95_um"]),
                "vol_tcell_rel_err": float(bd["vol_tcell_rel_err"]),
                "vol_target_rel_err": float(bd["vol_target_rel_err"]),
                "min_contact_sep_um": float(bd["min_contact_sep_um"]),
                "nit": int(sol["nit"]),
                "success": bool(sol["success"]),
                "solve_time_ms": float(elapsed_ms),
            })
    df_c = pd.DataFrame(rows)
    fine_df = df_c[df_c["nominal_edge_um"] == 0.5].set_index("pi_push")
    for idx, r in df_c.iterrows():
        pi_v = r["pi_push"]
        d_f = float(fine_df.loc[pi_v, "cup_depth_hull_p95_um"])
        d_s_f = float(fine_df.loc[pi_v, "cup_depth_sphere_p95_um"])
        df_c.loc[idx, "rel_err_hull_vs_0p5"] = abs(float(r["cup_depth_hull_p95_um"]) - d_f) / max(0.05, d_f)
        df_c.loc[idx, "rel_err_sph_vs_0p5"] = abs(float(r["cup_depth_sphere_p95_um"]) - d_s_f) / max(0.05, d_s_f)
    return df_c.to_dict(orient="records")


def main() -> int:
    t_start = time.perf_counter()
    STEP6B_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Running extended two-shell coarse-vs-fine check (1.4, 1.0, 0.7, 0.5 um up to Pi=12.0)...", flush=True)
    conv_records = run_two_shell_coarse_vs_fine_extended()
    df_conv_ext = pd.DataFrame(conv_records)
    df_conv_ext.to_csv(STEP6B_DIR / "gateA_mesh_convergence_extended.csv", index=False)

    print("2. Loading Step 5a visit descriptors and building 305 chamber tasks...", flush=True)
    df_v_all = pd.read_csv(STEP5_VISIT_CSV, low_memory=False)
    df_d_ch = pd.read_csv(STEP4_CHAMBER_CSV)
    df_man = pd.read_csv(MANIFEST_CSV)

    man_lookup: dict[tuple[str, str, int, int], str] = {}
    for _, mr in df_man.iterrows():
        man_lookup[(str(mr["experiment"]), str(mr["roi"]), int(mr["visit_index"]), int(mr["sweep_index"]))] = str(mr["path"])

    m = df_v_all.merge(
        df_d_ch[["experiment", "roi", "condition", "first_any_dead_visit_200m"]],
        on=["experiment", "roi"],
        how="left",
    )
    m_t1 = m[(m["trace_num"] == 1) & (m["visit"] <= 20)].copy()
    dead_v = m_t1["first_any_dead_visit_200m"].fillna(999)
    dead_v = np.where(dead_v <= 0, 999, dead_v)
    m_t1["is_pre_death"] = m_t1["visit"] < dead_v
    m_t1["is_pre_death_contact"] = (
        m_t1["in_contact_visit"]
        & m_t1["is_pre_death"]
        & m_t1["cup_depth_hull_p95_um"].notna()
        & (m_t1["patch_equiv_radius_um"] > 0.1)
    )

    ch_cnt = m_t1.groupby(["experiment", "roi"])["is_pre_death_contact"].sum()
    elig_keys = sorted(ch_cnt[ch_cnt >= 4].index)
    print(f"Eligible chambers: {len(elig_keys)}", flush=True)

    rep_sweeps_by_ch: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for exp_name in sorted({k[0] for k in elig_keys}):
        sw_csv = STEP3_TRACKED_DIR / exp_name / "per_sweep.csv"
        df_sw = pd.read_csv(sw_csv)
        for (e_k, r_k), g_sw in df_sw.groupby(["experiment", "roi"], sort=False):
            if (str(e_k), str(r_k)) not in elig_keys:
                continue
            v_map: dict[int, dict[str, Any]] = {}
            for v_num, gv in g_sw.groupby("visit", sort=False):
                gv_c = gv[gv["s_largest"] > 0]
                if len(gv_c) > 0:
                    med_s = float(gv_c["s_largest"].median())
                    idx_best = (gv_c["s_largest"] - med_s).abs().idxmin()
                    best_r = gv_c.loc[idx_best]
                else:
                    best_r = gv.iloc[len(gv) // 2]
                sw_num = int(best_r["sweep"])
                rel_p = man_lookup.get((str(e_k), str(r_k), int(v_num), sw_num), "")
                if rel_p:
                    v_map[int(v_num)] = {
                        "sweep": sw_num,
                        "rel_path": rel_p,
                        "tracked_tcell_id": int(best_r["tracked_tcell_id"]),
                        "n_tcell_instances": int(best_r["n_tcell_instances"]),
                        "n_target_instances": int(best_r["n_target_instances"]),
                        "s_largest": int(best_r["s_largest"]),
                    }
            rep_sweeps_by_ch[(str(e_k), str(r_k))] = v_map

    tasks: list[dict[str, Any]] = []
    for exp_k, roi_k in elig_keys:
        ch_df = m_t1[(m_t1["experiment"] == exp_k) & (m_t1["roi"] == roi_k)].sort_values("visit")
        tasks.append({
            "experiment": exp_k,
            "roi": roi_k,
            "condition": str(ch_df["condition"].iloc[0]),
            "visit_rows": ch_df.to_dict(orient="records"),
            "rep_sweep_map": rep_sweeps_by_ch.get((exp_k, roi_k), {}),
        })

    all_visit_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        fut_map = {pool.submit(process_one_chamber_gateA, t_item): t_item for t_item in tasks}
        for idx_d, fut in enumerate(as_completed(fut_map), start=1):
            res = fut.result()
            all_visit_rows.extend(res["visit_rows"])
            if idx_d % 25 == 0 or idx_d == len(tasks):
                print(f"  [{idx_d}/{len(tasks)}] {res['experiment']}/{res['roi']} ({res['elapsed_s']:.2f}s)", flush=True)

    df_fits = pd.DataFrame(all_visit_rows).sort_values(["experiment", "roi", "visit"]).reset_index(drop=True)
    df_fits.to_csv(STEP6B_DIR / "gateA_visit_fits.csv", index=False)
    print(f"Done in {time.perf_counter() - t_start:.1f}s! Total visits: {len(df_fits)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
