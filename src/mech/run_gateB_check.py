#!/usr/bin/env python3
"""
Stage B Gate Check for Step 6b (`src/mech/run_gateB_check.py`):
  1. Runs Pi fitting on `PI_NODES = [0.0, 0.6, 1.2, 2.4, 4.0, 6.0, 8.0, 10.0, 12.0]`
     (no extrapolation, no cap, alpha_syn=2.0, chi=1.0) ONLY on the 20 in-depth chambers
     in `results/step5_indepth_chambers.csv` (207 pre-death contact visits, 1,863 solves).
  2. Resolves initial contact overlap at Pi=0 explicitly via `resolve_initial_contact_overlap`
     before loading, recording initial separation before push-apart, push-apart distance (um),
     and separation after push-apart.
  3. Runs each solve to force tolerance `gtol = 5e-2 pN` with `maxiter = 450` and outer contact
     pair rebuilds; `success` = strictly converged to `gtol`.
  4. Reports:
     - Fraction converged (`success == True`), iterations (`nit`), outer rebuilds, wall time per solve
     - Push-apart distance distribution at Pi=0 and solved `min_contact_sep_um` distribution
     - Volume error distribution (`max_vol_rel_err < 1%`)
     - Inverted triangles count (`n_inverted_total`)
     - Fraction of visits in-range vs unreachable (`net_delta` and `raw_hull`)
     - Whether solved `depth(Pi)` is monotone
     - 5 example curves across `Pi in [0..12]`
     - Wall time per chamber and projected time for all 305 chambers
  5. Checks `1.0 vs 0.5 um` mesh on 3 real visits up to the largest fitted Pi (and across `0..12`).
  6. Reconciles Gate A `.md` (204, 16) vs `gateA.json` (226, 2) from `data/step6b/gateA_visit_fits.csv`.
  7. Writes `data/step6b/gateB.json` and `data/step6b/gateB_visit_fits.csv`.
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
    TwoShellSystem,
    build_icosphere,
    check_closed_manifold,
    check_inverted_triangles,
    compute_mesh_descriptors,
    extract_cell_mesh_marching_cubes,
    make_cup_filled_reference_mesh,
    resolve_initial_contact_overlap,
    solve_two_shell_equilibrium,
)
PI_NODES_GATEB = np.array([0.0, 0.6, 1.2, 2.4, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
INDEPTH_CSV = REPO_ROOT / "results" / "step5_indepth_chambers.csv"


def _invert_depth_curve_no_extrap(
    d_obs: float,
    pi_nodes: np.ndarray,
    d_nodes: np.ndarray,
) -> tuple[float, bool, str]:
    """
    Invert `d_nodes(pi_nodes)` for `d_obs` with NO extrapolation and NO cap inside the solved range.
    If `d_obs > max(d_nodes)`, returns `(np.nan, True, "unreachable_above_max")`.
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

    u_vals, u_idx = np.unique(d_sub, return_index=True)
    pi_u = pi_sub[u_idx]
    pi_fit = float(np.interp(d_obs, u_vals, pi_u))
    return pi_fit, False, "in_range"


def _prepare_visit_meshes_at_edge(
    vrow: dict[str, Any] | pd.Series,
    sw_info: dict[str, Any] | None,
    V_ref_base_raw: np.ndarray | None,
    F_ref_base_raw: np.ndarray | None,
    has_pre_contact_ref: bool,
    first_t_mask_info: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    target_edge_um: float = 1.0,
) -> dict[str, Any]:
    """Extract and align T-cell and reference target meshes at a specified `target_edge_um`."""
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
                V_t_v, F_t_v = extract_cell_mesh_marching_cubes(t_sel, off_t, target_edge_um=target_edge_um)
                V_g_v, F_g_v = extract_cell_mesh_marching_cubes(b_sel, off_b, target_edge_um=target_edge_um)

    if V_t_v is None or V_g_v is None:
        r_g0 = float((3.0 * max(50.0, float(vrow["target_volume_um3"])) / (4.0 * np.pi)) ** (1.0 / 3.0))
        r_t0 = float((3.0 * max(50.0, float(vrow["tcell_volume_um3"])) / (4.0 * np.pi)) ** (1.0 / 3.0))
        V_g_v, F_g_v = build_icosphere(np.zeros(3), r_g0, target_edge_um=target_edge_um)
        V_t_v, F_t_v = build_icosphere(np.array([0.0, 0.0, -(r_g0 + r_t0 + 0.18)]), r_t0, target_edge_um=target_edge_um)

    if has_pre_contact_ref and V_ref_base_raw is not None and F_ref_base_raw is not None:
        V_ref_use, F_ref_use = V_ref_base_raw, F_ref_base_raw
    elif first_t_mask_info is not None:
        t_sel0, off_t0, b_sel0, off_b0 = first_t_mask_info
        V_t0, _ = extract_cell_mesh_marching_cubes(t_sel0, off_t0, target_edge_um=target_edge_um)
        V_g0, F_g0 = extract_cell_mesh_marching_cubes(b_sel0, off_b0, target_edge_um=target_edge_um)
        V_ref_use = make_cup_filled_reference_mesh(V_g0, F_g0, V_t0)
        F_ref_use = F_g0
    else:
        V_ref_use = make_cup_filled_reference_mesh(V_g_v, F_g_v, V_t_v)
        F_ref_use = F_g_v

    vol_g = float(vrow["target_volume_um3"])
    V_g_ref_v = _align_reference_target_to_visit(
        V_ref_use, F_ref_use, np.mean(V_g_v, axis=0), vol_g, V_t_v
    )
    syn_faces_v = _tag_synapse_faces_for_patch_area(
        V_t_v, F_t_v, V_g_ref_v, float(vrow["contact_area_um2"])
    )

    # Measure minimum contact separation BEFORE explicit overlap resolution
    t_tmp = ShellMesh(V0=V_t_v, F=F_t_v, synapse_faces=syn_faces_v)
    g_tmp = ShellMesh(V0=V_g_ref_v, F=F_ref_use, synapse_faces=np.zeros(len(F_ref_use), dtype=bool))
    p_tmp = ShellParameters.from_dimensionless(pi_push=0.0, chi_t=1.0, alpha_syn=2.0, r_target_um=g_tmp.R_eq0)
    sys_before = TwoShellSystem(t_tmp, g_tmp, p_tmp)
    _, _, _, min_sep_before = sys_before._contact_energy_and_grad(V_t_v, V_g_ref_v)

    # Explicitly resolve initial overlap at Pi = 0 before loading
    V_t_clean, V_g_clean, push_apart_um, min_sep_after_push = resolve_initial_contact_overlap(
        V_t_v, F_t_v, V_g_ref_v, F_ref_use, gap_eq_um=0.15, syn_faces_t=syn_faces_v
    )
    t_mesh = ShellMesh(V0=V_t_clean, F=F_t_v, synapse_faces=syn_faces_v)
    g_mesh = ShellMesh(V0=V_g_clean, F=F_ref_use, synapse_faces=np.zeros(len(F_ref_use), dtype=bool))
    sys_after = TwoShellSystem(t_mesh, g_mesh, p_tmp)
    _, _, _, min_sep_after = sys_after._contact_energy_and_grad(V_t_clean, V_g_clean)

    return {
        "t_mesh": t_mesh,
        "g_mesh": g_mesh,
        "min_sep_before_push_um": float(min_sep_before),
        "push_apart_um": float(push_apart_um),
        "min_sep_after_push_um": float(min_sep_after),
    }


def process_one_chamber_gateB(task: dict[str, Any]) -> dict[str, Any]:
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
        return {"experiment": exp, "roi": roi, "visit_rows": [], "elapsed_s": 0.0}

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

        # Measure initial overlap before explicit push-apart
        t_tmp = ShellMesh(V0=V_t_v, F=F_t_v, synapse_faces=syn_faces_v)
        g_tmp = ShellMesh(V0=V_g_ref_v, F=F_ref_base, synapse_faces=np.zeros(len(F_ref_base), dtype=bool))
        p_tmp = ShellParameters.from_dimensionless(pi_push=0.0, chi_t=1.0, alpha_syn=2.0, r_target_um=g_tmp.R_eq0)
        sys_before = TwoShellSystem(t_tmp, g_tmp, p_tmp)
        _, _, _, min_sep_before = sys_before._contact_energy_and_grad(V_t_v, V_g_ref_v)

        # Explicitly resolve initial overlap at Pi = 0
        V_t_clean, V_g_clean, push_apart_um, _ = resolve_initial_contact_overlap(
            V_t_v, F_t_v, V_g_ref_v, F_ref_base, gap_eq_um=0.15, syn_faces_t=syn_faces_v
        )

        t_mesh_v = ShellMesh(V0=V_t_clean, F=F_t_v, synapse_faces=syn_faces_v)
        g_mesh_v = ShellMesh(V0=V_g_clean, F=F_ref_base, synapse_faces=np.zeros(len(F_ref_base), dtype=bool))

        sys_after = TwoShellSystem(t_mesh_v, g_mesh_v, p_tmp)
        _, _, _, min_sep_after_push = sys_after._contact_energy_and_grad(V_t_clean, V_g_clean)

        d_delta_nodes: list[float] = []
        d_raw_hull_nodes: list[float] = []
        d_sph_nodes: list[float] = []
        min_sep_nodes: list[float] = []
        vol_err_nodes: list[float] = []
        n_inv_nodes: list[int] = []
        nit_nodes: list[int] = []
        rebuilds_nodes: list[int] = []
        time_ms_nodes: list[float] = []
        grad_nodes: list[float] = []
        ok_nodes: list[bool] = []

        x_warm: np.ndarray | None = None
        base_hull_d = 0.0
        for idx_pi, pi_val in enumerate(PI_NODES_GATEB):
            params_p = ShellParameters.from_dimensionless(
                pi_push=float(pi_val), chi_t=1.0, alpha_syn=2.0, r_target_um=g_mesh_v.R_eq0, subtract_prestress=True
            )
            sol_p = solve_two_shell_equilibrium(
                t_mesh_v, g_mesh_v, params_p, maxiter=450, gtol=5e-2, x0=x_warm
            )
            x_warm = sol_p["x_eq"]
            desc_p = compute_mesh_descriptors(
                sol_p["V_tcell"], t_mesh_v.F, sol_p["V_target"], g_mesh_v.F, synapse_faces_t=t_mesh_v.synapse_faces
            )
            bd_p = sol_p["breakdown"]
            raw_h = float(desc_p["cup_depth_hull_p95_um"])
            raw_s = float(desc_p["cup_depth_sphere_p95_um"])
            if idx_pi == 0:
                base_hull_d = raw_h
                d_del = 0.0
            else:
                d_del = max(0.0, raw_h - base_hull_d)

            d_delta_nodes.append(d_del)
            d_raw_hull_nodes.append(raw_h)
            d_sph_nodes.append(raw_s)
            min_sep_nodes.append(float(bd_p["min_contact_sep_um"]))
            vol_err_nodes.append(float(max(bd_p["vol_tcell_rel_err"], bd_p["vol_target_rel_err"])))
            n_inv_nodes.append(int(bd_p["n_inverted_total"]))
            nit_nodes.append(int(sol_p["nit"]))
            rebuilds_nodes.append(int(sol_p["n_outer_rebuilds"]))
            time_ms_nodes.append(float(sol_p["solve_time_ms"]))
            grad_nodes.append(float(sol_p["grad_norm_final"]))
            ok_nodes.append(bool(sol_p["success"]))

        arr_delta = np.asarray(d_delta_nodes, dtype=np.float64)
        arr_raw = np.asarray(d_raw_hull_nodes, dtype=np.float64)
        arr_sph = np.asarray(d_sph_nodes, dtype=np.float64)

        pi_fit_delta, unreach_delta, status_delta = _invert_depth_curve_no_extrap(d_obs, PI_NODES_GATEB, arr_delta)
        pi_fit_raw, unreach_raw, status_raw = _invert_depth_curve_no_extrap(d_obs, PI_NODES_GATEB, arr_raw)

        rec: dict[str, Any] = {
            "experiment": exp,
            "roi": roi,
            "condition": cond,
            "visit": v_num,
            "ref_type": ref_type,
            "cup_depth_hull_p95_um": d_obs,
            "patch_equiv_radius_um": r_p,
            "r_target_um": float(g_mesh_v.R_eq0),
            "min_sep_before_push_um": float(min_sep_before),
            "push_apart_um": float(push_apart_um),
            "min_sep_after_push_um": float(min_sep_after_push),
            "base_hull_depth_pi0_um": base_hull_d,
            "max_depth_delta_um": float(np.max(arr_delta)),
            "pi_at_max_delta": float(PI_NODES_GATEB[int(np.argmax(arr_delta))]),
            "max_depth_raw_hull_um": float(np.max(arr_raw)),
            "pi_at_max_raw_hull": float(PI_NODES_GATEB[int(np.argmax(arr_raw))]),
            "pi_fit_delta": pi_fit_delta,
            "unreachable_delta": bool(unreach_delta),
            "status_delta": status_delta,
            "pi_fit_raw_hull": pi_fit_raw,
            "unreachable_raw_hull": bool(unreach_raw),
            "status_raw_hull": status_raw,
            "curve_nonmonotone_delta_gt_0p05": bool(np.any(np.diff(arr_delta) < -0.05)),
            "curve_nonmonotone_raw_gt_0p05": bool(np.any(np.diff(arr_raw) < -0.05)),
            "min_diff_delta_um": float(np.min(np.diff(arr_delta))),
            "min_diff_raw_um": float(np.min(np.diff(arr_raw))),
            "min_sep_overall_um": float(np.min(min_sep_nodes)),
            "min_sep_pi0_um": float(min_sep_nodes[0]),
            "min_sep_pi12_um": float(min_sep_nodes[-1]),
            "max_vol_err_overall": float(np.max(vol_err_nodes)),
            "max_vol_err_pi12": float(vol_err_nodes[-1]),
            "max_n_inverted_overall": int(np.max(n_inv_nodes)),
            "total_inverted_across_nodes": int(np.sum(n_inv_nodes)),
            "n_converged_nodes": int(np.sum(ok_nodes)),
            "all_nodes_converged": bool(all(ok_nodes)),
            "mean_nit_per_solve": float(np.mean(nit_nodes)),
            "max_nit_per_solve": int(np.max(nit_nodes)),
            "mean_solve_time_ms": float(np.mean(time_ms_nodes)),
            "total_visit_solve_time_s": float(np.sum(time_ms_nodes) / 1000.0),
            "max_grad_norm_final_pN": float(np.max(grad_nodes)),
        }
        for k_i, pi_v in enumerate(PI_NODES_GATEB):
            tag = f"pi_{str(pi_v).replace('.', 'p')}"
            rec[f"d_delta_{tag}"] = float(arr_delta[k_i])
            rec[f"d_raw_{tag}"] = float(arr_raw[k_i])
            rec[f"d_sph_{tag}"] = float(arr_sph[k_i])
            rec[f"min_sep_{tag}"] = float(min_sep_nodes[k_i])
            rec[f"vol_err_{tag}"] = float(vol_err_nodes[k_i])
            rec[f"n_inv_{tag}"] = int(n_inv_nodes[k_i])
            rec[f"nit_{tag}"] = int(nit_nodes[k_i])
            rec[f"rebuilds_{tag}"] = int(rebuilds_nodes[k_i])
            rec[f"time_ms_{tag}"] = float(time_ms_nodes[k_i])
            rec[f"grad_{tag}"] = float(grad_nodes[k_i])
            rec[f"ok_{tag}"] = bool(ok_nodes[k_i])

        out_visit_rows.append(rec)

    return {
        "experiment": exp,
        "roi": roi,
        "visit_rows": out_visit_rows,
        "elapsed_s": time.perf_counter() - t0,
    }


def run_real_visit_1p0_vs_0p5_check(
    selected_visits: list[dict[str, Any]],
    m_t1: pd.DataFrame,
    rep_sweeps_by_ch: dict[tuple[str, str], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Compare `1.0 um` vs `0.5 um` mesh on 3 real visits up to the largest fitted Pi node
    (and across all grid nodes up to at least `pi_ceil`).
    """
    results: list[dict[str, Any]] = []
    for item in selected_visits:
        exp = str(item["experiment"])
        roi = str(item["roi"])
        v_num = int(item["visit"])
        pi_fit_max = float(item["pi_fit_max"])

        ch_df = m_t1[(m_t1["experiment"] == exp) & (m_t1["roi"] == roi)].sort_values("visit").reset_index(drop=True)
        valid_c = ch_df[ch_df["is_pre_death_contact"]].sort_values("visit").reset_index(drop=True)
        first_contact_v = int(valid_c["visit"].iloc[0])
        vrow = valid_c[valid_c["visit"] == v_num].iloc[0]
        rep_map = rep_sweeps_by_ch.get((exp, roi), {})

        # Pre-contact reference mask info or first contact mask info
        pre_df = ch_df[
            (ch_df["visit"] < first_contact_v)
            & (~ch_df["in_contact_visit"])
            & (ch_df["target_volume_um3"].fillna(0.0) > 20.0)
        ].sort_values("visit")

        pre_mask_info = None
        if len(pre_df) > 0:
            for idx_p in range(len(pre_df) - 1, -1, -1):
                cand_v = int(pre_df.iloc[idx_p]["visit"])
                if cand_v in rep_map:
                    sw_p = rep_map[cand_v]
                    h5_p = REPO_ROOT / sw_p["rel_path"]
                    if h5_p.is_file():
                        _, _, b_sel_p, off_b_p = _extract_tcell_and_target_masks_from_h5(
                            h5_p, int(sw_p["tracked_tcell_id"]), int(sw_p["n_tcell_instances"]), int(sw_p["n_target_instances"])
                        )
                        if b_sel_p is not None and np.any(b_sel_p):
                            pre_mask_info = (b_sel_p, off_b_p)
                            break

        first_sw = rep_map.get(first_contact_v)
        first_t_mask_info = None
        if first_sw is not None:
            h5_0 = REPO_ROOT / first_sw["rel_path"]
            if h5_0.is_file():
                t0, ot0, b0, ob0 = _extract_tcell_and_target_masks_from_h5(
                    h5_0, int(first_sw["tracked_tcell_id"]), int(first_sw["n_tcell_instances"]), int(first_sw["n_target_instances"])
                )
                if t0 is not None and b0 is not None:
                    first_t_mask_info = (t0, ot0, b0, ob0)

        # Determine Pi nodes up to the ceiling grid node covering `pi_fit_max`
        idx_ceil = int(np.searchsorted(PI_NODES_GATEB, pi_fit_max, side="left"))
        idx_ceil = min(len(PI_NODES_GATEB) - 1, max(3, idx_ceil))
        pi_sub_nodes = PI_NODES_GATEB[: idx_ceil + 1]

        edge_curves: dict[float, dict[str, Any]] = {}
        for edge_um in [1.0, 0.5]:
            V_ref_raw = F_ref_raw = None
            has_pre = False
            if pre_mask_info is not None:
                V_ref_raw, F_ref_raw = extract_cell_mesh_marching_cubes(
                    pre_mask_info[0], pre_mask_info[1], target_edge_um=edge_um
                )
                has_pre = True

            prep = _prepare_visit_meshes_at_edge(
                vrow=vrow,
                sw_info=rep_map.get(v_num),
                V_ref_base_raw=V_ref_raw,
                F_ref_base_raw=F_ref_raw,
                has_pre_contact_ref=has_pre,
                first_t_mask_info=first_t_mask_info,
                target_edge_um=edge_um,
            )
            t_mesh = prep["t_mesh"]
            g_mesh = prep["g_mesh"]

            x_warm = None
            base_h = 0.0
            node_rows = []
            for k_idx, pi_val in enumerate(pi_sub_nodes):
                params_p = ShellParameters.from_dimensionless(
                    pi_push=float(pi_val), chi_t=1.0, alpha_syn=2.0, r_target_um=g_mesh.R_eq0, subtract_prestress=True
                )
                sol_p = solve_two_shell_equilibrium(
                    t_mesh, g_mesh, params_p, maxiter=450, gtol=5e-2, x0=x_warm
                )
                x_warm = sol_p["x_eq"]
                desc_p = compute_mesh_descriptors(
                    sol_p["V_tcell"], t_mesh.F, sol_p["V_target"], g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
                )
                bd_p = sol_p["breakdown"]
                raw_h = float(desc_p["cup_depth_hull_p95_um"])
                raw_s = float(desc_p["cup_depth_sphere_p95_um"])
                if k_idx == 0:
                    base_h = raw_h
                    d_del = 0.0
                else:
                    d_del = max(0.0, raw_h - base_h)
                node_rows.append({
                    "pi_push": float(pi_val),
                    "d_delta_um": d_del,
                    "d_raw_hull_um": raw_h,
                    "d_sphere_um": raw_s,
                    "min_contact_sep_um": float(bd_p["min_contact_sep_um"]),
                    "max_vol_rel_err": float(max(bd_p["vol_tcell_rel_err"], bd_p["vol_target_rel_err"])),
                    "n_inverted_total": int(bd_p["n_inverted_total"]),
                    "nit": int(sol_p["nit"]),
                    "solve_time_ms": float(sol_p["solve_time_ms"]),
                    "success": bool(sol_p["success"]),
                })

            d_del_arr = np.array([r["d_delta_um"] for r in node_rows], dtype=np.float64)
            d_raw_arr = np.array([r["d_raw_hull_um"] for r in node_rows], dtype=np.float64)
            pi_fit_d, _, _ = _invert_depth_curve_no_extrap(float(vrow["cup_depth_hull_p95_um"]), pi_sub_nodes, d_del_arr)
            pi_fit_r, _, _ = _invert_depth_curve_no_extrap(float(vrow["cup_depth_hull_p95_um"]), pi_sub_nodes, d_raw_arr)

            edge_curves[edge_um] = {
                "n_verts_tcell": int(len(t_mesh.V0)),
                "n_faces_tcell": int(len(t_mesh.F)),
                "n_verts_target": int(len(g_mesh.V0)),
                "n_faces_target": int(len(g_mesh.F)),
                "push_apart_um": float(prep["push_apart_um"]),
                "pi_fit_delta": float(pi_fit_d) if np.isfinite(pi_fit_d) else None,
                "pi_fit_raw_hull": float(pi_fit_r) if np.isfinite(pi_fit_r) else None,
                "nodes": node_rows,
            }

        nodes_10 = edge_curves[1.0]["nodes"]
        nodes_05 = edge_curves[0.5]["nodes"]
        abs_diffs_raw = [abs(a["d_raw_hull_um"] - b["d_raw_hull_um"]) for a, b in zip(nodes_10, nodes_05)]
        rel_diffs_raw = [
            abs(a["d_raw_hull_um"] - b["d_raw_hull_um"]) / max(0.10, b["d_raw_hull_um"])
            for a, b in zip(nodes_10, nodes_05)
        ]
        abs_diffs_del = [abs(a["d_delta_um"] - b["d_delta_um"]) for a, b in zip(nodes_10[1:], nodes_05[1:])]
        rel_diffs_del = [
            abs(a["d_delta_um"] - b["d_delta_um"]) / max(0.10, b["d_delta_um"])
            for a, b in zip(nodes_10[1:], nodes_05[1:])
        ]

        results.append({
            "experiment": exp,
            "roi": roi,
            "visit": v_num,
            "observed_cup_depth_hull_p95_um": float(vrow["cup_depth_hull_p95_um"]),
            "largest_fitted_pi_tested": float(pi_sub_nodes[-1]),
            "pi_nodes_tested": [float(x) for x in pi_sub_nodes],
            "median_rel_diff_raw_hull_1p0_vs_0p5": float(np.median(rel_diffs_raw)),
            "max_rel_diff_raw_hull_1p0_vs_0p5": float(np.max(rel_diffs_raw)),
            "median_abs_diff_raw_hull_um": float(np.median(abs_diffs_raw)),
            "median_rel_diff_delta_1p0_vs_0p5": float(np.median(rel_diffs_del)),
            "at_largest_pi_node": {
                "pi_push": float(pi_sub_nodes[-1]),
                "d_raw_1p0_um": float(nodes_10[-1]["d_raw_hull_um"]),
                "d_raw_0p5_um": float(nodes_05[-1]["d_raw_hull_um"]),
                "rel_diff_raw_at_max_pi": float(rel_diffs_raw[-1]),
                "d_delta_1p0_um": float(nodes_10[-1]["d_delta_um"]),
                "d_delta_0p5_um": float(nodes_05[-1]["d_delta_um"]),
                "rel_diff_delta_at_max_pi": float(rel_diffs_del[-1]),
                "min_sep_1p0_um": float(nodes_10[-1]["min_contact_sep_um"]),
                "min_sep_0p5_um": float(nodes_05[-1]["min_contact_sep_um"]),
                "vol_err_1p0": float(nodes_10[-1]["max_vol_rel_err"]),
                "vol_err_0p5": float(nodes_05[-1]["max_vol_rel_err"]),
                "n_inv_1p0": int(nodes_10[-1]["n_inverted_total"]),
                "n_inv_0p5": int(nodes_05[-1]["n_inverted_total"]),
            },
            "edge_1p0": edge_curves[1.0],
            "edge_0p5": edge_curves[0.5],
        })
    return results


def main() -> int:
    t_start = time.perf_counter()
    STEP6B_DIR.mkdir(parents=True, exist_ok=True)

    df_indepth = pd.read_csv(INDEPTH_CSV)
    indepth_keys = sorted({(str(r["experiment"]), str(r["roi"])) for _, r in df_indepth.iterrows()})
    print(f"Loaded {len(indepth_keys)} in-depth chambers from {INDEPTH_CSV.name}", flush=True)

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

    rep_sweeps_by_ch: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for exp_name in sorted({k[0] for k in indepth_keys}):
        sw_csv = STEP3_TRACKED_DIR / exp_name / "per_sweep.csv"
        df_sw = pd.read_csv(sw_csv)
        for (e_k, r_k), g_sw in df_sw.groupby(["experiment", "roi"], sort=False):
            if (str(e_k), str(r_k)) not in indepth_keys:
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
    for exp_k, roi_k in indepth_keys:
        ch_df = m_t1[(m_t1["experiment"] == exp_k) & (m_t1["roi"] == roi_k)].sort_values("visit")
        tasks.append({
            "experiment": exp_k,
            "roi": roi_k,
            "condition": str(ch_df["condition"].iloc[0]),
            "visit_rows": ch_df.to_dict(orient="records"),
            "rep_sweep_map": rep_sweeps_by_ch.get((exp_k, roi_k), {}),
        })

    print(f"Running Stage B Pi-fitting on {len(tasks)} in-depth chambers...", flush=True)
    t_pool_start = time.perf_counter()
    all_visit_rows: list[dict[str, Any]] = []
    chamber_times_s: list[float] = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        fut_map = {pool.submit(process_one_chamber_gateB, t_item): t_item for t_item in tasks}
        for idx_d, fut in enumerate(as_completed(fut_map), start=1):
            res = fut.result()
            all_visit_rows.extend(res["visit_rows"])
            chamber_times_s.append(float(res["elapsed_s"]))
            print(
                f"  [{idx_d:02d}/{len(tasks)}] {res['experiment']}/{res['roi']} "
                f"({len(res['visit_rows'])} visits, {res['elapsed_s']:.2f}s)",
                flush=True,
            )
    pool_wall_s = time.perf_counter() - t_pool_start

    df_fits = pd.DataFrame(all_visit_rows).sort_values(["experiment", "roi", "visit"]).reset_index(drop=True)
    df_fits.to_csv(STEP6B_DIR / "gateB_visit_fits.csv", index=False)
    print(f"Saved {len(df_fits)} visits to data/step6b/gateB_visit_fits.csv in {pool_wall_s:.2f}s", flush=True)

    # Select 3 real visits spanning moderate to largest fitted Pi for the 1.0 vs 0.5 um mesh check
    in_range_df = df_fits[df_fits["pi_fit_raw_hull"].notna() & (df_fits["pi_fit_raw_hull"] > 0.5)].sort_values("pi_fit_raw_hull")
    if len(in_range_df) >= 3:
        picks = [
            in_range_df.iloc[len(in_range_df) // 3],
            in_range_df.iloc[(2 * len(in_range_df)) // 3],
            in_range_df.iloc[-1],
        ]
    else:
        picks = [df_fits.iloc[0], df_fits.iloc[len(df_fits) // 2], df_fits.iloc[-1]]

    selected_3_visits = []
    for r_pick in picks:
        pi_m = float(np.nanmax([
            float(r_pick["pi_fit_delta"]) if pd.notna(r_pick["pi_fit_delta"]) else 2.4,
            float(r_pick["pi_fit_raw_hull"]) if pd.notna(r_pick["pi_fit_raw_hull"]) else 2.4,
        ]))
        selected_3_visits.append({
            "experiment": str(r_pick["experiment"]),
            "roi": str(r_pick["roi"]),
            "visit": int(r_pick["visit"]),
            "pi_fit_max": max(2.4, pi_m),
        })

    print("Running 1.0 vs 0.5 um mesh check on 3 real visits up to largest fitted Pi...", flush=True)
    real_mesh_check = run_real_visit_1p0_vs_0p5_check(selected_3_visits, m_t1, rep_sweeps_by_ch)

    # Aggregate all node-level solver metrics across all 207 * 9 = 1,863 solves
    all_seps = []
    all_vol_errs = []
    all_n_invs = []
    all_nits = []
    all_rebuilds = []
    all_times_ms = []
    all_grads = []
    all_oks = []
    for pi_v in PI_NODES_GATEB:
        tag = f"pi_{str(pi_v).replace('.', 'p')}"
        all_seps.extend(df_fits[f"min_sep_{tag}"].tolist())
        all_vol_errs.extend(df_fits[f"vol_err_{tag}"].tolist())
        all_n_invs.extend(df_fits[f"n_inv_{tag}"].tolist())
        all_nits.extend(df_fits[f"nit_{tag}"].tolist())
        all_rebuilds.extend(df_fits[f"rebuilds_{tag}"].tolist())
        all_times_ms.extend(df_fits[f"time_ms_{tag}"].tolist())
        all_grads.extend(df_fits[f"grad_{tag}"].tolist())
        all_oks.extend(df_fits[f"ok_{tag}"].tolist())

    arr_seps = np.asarray(all_seps, dtype=np.float64)
    arr_vol_errs = np.asarray(all_vol_errs, dtype=np.float64)
    arr_n_invs = np.asarray(all_n_invs, dtype=np.int32)
    arr_nits = np.asarray(all_nits, dtype=np.int32)
    arr_rebuilds = np.asarray(all_rebuilds, dtype=np.int32)
    arr_times_ms = np.asarray(all_times_ms, dtype=np.float64)
    arr_grads = np.asarray(all_grads, dtype=np.float64)
    arr_oks = np.asarray(all_oks, dtype=bool)

    # 5 example curves across Pi in [0..12]
    example_indices = np.linspace(0, len(df_fits) - 1, 5, dtype=int)
    example_curves = []
    for idx_ex in example_indices:
        r_ex = df_fits.iloc[int(idx_ex)]
        c_delta = {}
        c_raw = {}
        c_sep = {}
        for pi_v in PI_NODES_GATEB:
            tag = f"pi_{str(pi_v).replace('.', 'p')}"
            c_delta[str(pi_v)] = round(float(r_ex[f"d_delta_{tag}"]), 4)
            c_raw[str(pi_v)] = round(float(r_ex[f"d_raw_{tag}"]), 4)
            c_sep[str(pi_v)] = round(float(r_ex[f"min_sep_{tag}"]), 4)
        example_curves.append({
            "experiment": str(r_ex["experiment"]),
            "roi": str(r_ex["roi"]),
            "visit": int(r_ex["visit"]),
            "ref_type": str(r_ex["ref_type"]),
            "observed_depth_um": round(float(r_ex["cup_depth_hull_p95_um"]), 4),
            "pi_fit_delta": round(float(r_ex["pi_fit_delta"]), 4) if pd.notna(r_ex["pi_fit_delta"]) else None,
            "pi_fit_raw_hull": round(float(r_ex["pi_fit_raw_hull"]), 4) if pd.notna(r_ex["pi_fit_raw_hull"]) else None,
            "d_delta_curve_um": c_delta,
            "d_raw_hull_curve_um": c_raw,
            "min_sep_curve_um": c_sep,
        })

    # Reconcile Gate A discrepancy from data/step6b/gateA_visit_fits.csv
    gateA_reconciliation = {}
    gateA_csv = STEP6B_DIR / "gateA_visit_fits.csv"
    if gateA_csv.is_file():
        df_ga = pd.read_csv(gateA_csv)
        ch_ga_del = df_ga.groupby(["experiment", "roi"])["unreachable_delta"].agg(["any", "all"])
        ch_ga_raw = df_ga.groupby(["experiment", "roi"])["unreachable_raw_hull"].agg(["any", "all"])
        gateA_reconciliation = {
            "total_chambers_gateA": int(len(ch_ga_del)),
            "total_visits_gateA": int(len(df_ga)),
            "exact_from_gateA_visit_fits_csv": {
                "net_delta_chambers_any_unreachable": int(ch_ga_del["any"].sum()),
                "net_delta_chambers_all_unreachable": int(ch_ga_del["all"].sum()),
                "net_delta_chambers_with_in_range_visits": int((~ch_ga_del["all"]).sum()),
                "raw_hull_chambers_any_unreachable": int(ch_ga_raw["any"].sum()),
                "raw_hull_chambers_all_unreachable": int(ch_ga_raw["all"].sum()),
                "raw_hull_chambers_with_in_range_visits": int((~ch_ga_raw["all"]).sum()),
            },
            "explanation": (
                "gateA.json (226 chambers with >=1 unreachable visit, 2 with all visits unreachable, "
                "303 with >=1 in-range visit for net_delta; 191 any, 2 all for raw_hull) was computed "
                "directly from data/step6b/gateA_visit_fits.csv, whereas .agy_step6b_gateA.md contained "
                "stale hand-typed numbers (204 any, 16 all, N=289)."
            ),
        }

    ch_gb_del = df_fits.groupby(["experiment", "roi"])["unreachable_delta"].agg(["any", "all"])
    ch_gb_raw = df_fits.groupby(["experiment", "roi"])["unreachable_raw_hull"].agg(["any", "all"])

    mean_ch_s = float(np.mean(chamber_times_s))
    wall_per_ch_parallel_s = float(pool_wall_s / len(tasks))
    proj_305_parallel_s = float(wall_per_ch_parallel_s * 305.0)
    proj_305_serial_cpu_s = float(mean_ch_s * 305.0)

    gateB_summary = {
        "n_chambers": int(len(tasks)),
        "n_visits": int(len(df_fits)),
        "n_total_solves": int(len(arr_oks)),
        "pi_nodes": [float(x) for x in PI_NODES_GATEB],
        "convergence": {
            "gtol_force_pN": 5e-2,
            "maxiter": 450,
            "n_converged_solves": int(np.sum(arr_oks)),
            "n_total_solves": int(len(arr_oks)),
            "fraction_solves_converged": float(np.mean(arr_oks)),
            "fraction_visits_all_nodes_converged": float(df_fits["all_nodes_converged"].mean()),
            "nit_median": float(np.median(arr_nits)),
            "nit_mean": float(np.mean(arr_nits)),
            "nit_p95": float(np.percentile(arr_nits, 95)),
            "nit_max": int(np.max(arr_nits)),
            "outer_rebuilds_median": float(np.median(arr_rebuilds)),
            "outer_rebuilds_max": int(np.max(arr_rebuilds)),
            "grad_inf_pN_median": float(np.median(arr_grads)),
            "grad_inf_pN_p95": float(np.percentile(arr_grads, 95)),
            "grad_inf_pN_max": float(np.max(arr_grads)),
            "solve_time_ms_median": float(np.median(arr_times_ms)),
            "solve_time_ms_mean": float(np.mean(arr_times_ms)),
            "solve_time_ms_p95": float(np.percentile(arr_times_ms, 95)),
        },
        "contact_and_initial_overlap": {
            "n_visits_initial_overlap_before_push": int((df_fits["min_sep_before_push_um"] < 0.0).sum()),
            "fraction_visits_initial_overlap_before_push": float((df_fits["min_sep_before_push_um"] < 0.0).mean()),
            "min_sep_before_push_um_quantiles": {
                "min": float(df_fits["min_sep_before_push_um"].min()),
                "p10": float(np.percentile(df_fits["min_sep_before_push_um"], 10)),
                "median": float(df_fits["min_sep_before_push_um"].median()),
                "p90": float(np.percentile(df_fits["min_sep_before_push_um"], 90)),
                "max": float(df_fits["min_sep_before_push_um"].max()),
            },
            "push_apart_distance_um_quantiles": {
                "min": float(df_fits["push_apart_um"].min()),
                "p10": float(np.percentile(df_fits["push_apart_um"], 10)),
                "median": float(df_fits["push_apart_um"].median()),
                "mean": float(df_fits["push_apart_um"].mean()),
                "p90": float(np.percentile(df_fits["push_apart_um"], 90)),
                "max": float(df_fits["push_apart_um"].max()),
            },
            "min_sep_after_push_um_min": float(df_fits["min_sep_after_push_um"].min()),
            "solved_min_contact_sep_um_all_solves": {
                "min": float(np.min(arr_seps)),
                "p05": float(np.percentile(arr_seps, 5)),
                "p25": float(np.percentile(arr_seps, 25)),
                "median": float(np.median(arr_seps)),
                "p75": float(np.percentile(arr_seps, 75)),
                "p95": float(np.percentile(arr_seps, 95)),
                "max": float(np.max(arr_seps)),
                "n_interpenetrating_solves_lt_0": int(np.sum(arr_seps < 0.0)),
                "fraction_interpenetrating_solves_lt_0": float(np.mean(arr_seps < 0.0)),
                "n_visits_any_interpenetration_lt_0": int((df_fits["min_sep_overall_um"] < 0.0).sum()),
            },
            "solved_min_contact_sep_pi0_median_um": float(df_fits["min_sep_pi0_um"].median()),
            "solved_min_contact_sep_pi12_median_um": float(df_fits["min_sep_pi12_um"].median()),
        },
        "volume_and_inverted_triangles": {
            "max_vol_rel_err_all_solves": {
                "median": float(np.median(arr_vol_errs)),
                "p95": float(np.percentile(arr_vol_errs, 95)),
                "p99": float(np.percentile(arr_vol_errs, 99)),
                "max": float(np.max(arr_vol_errs)),
                "n_solves_gt_1pct": int(np.sum(arr_vol_errs > 0.01)),
                "n_visits_gt_1pct": int((df_fits["max_vol_err_overall"] > 0.01).sum()),
            },
            "inverted_triangles": {
                "total_inverted_across_all_1863_solves": int(np.sum(arr_n_invs)),
                "max_inverted_in_any_solve": int(np.max(arr_n_invs)),
                "n_solves_with_any_inverted": int(np.sum(arr_n_invs > 0)),
            },
        },
        "reachability_and_monotonicity": {
            "net_delta_d_Pi_minus_d_0": {
                "n_in_range": int((df_fits["status_delta"] == "in_range").sum()),
                "fraction_in_range": float((df_fits["status_delta"] == "in_range").mean()),
                "n_at_or_below_zero_load": int((df_fits["status_delta"] == "at_or_below_zero_load").sum()),
                "fraction_at_or_below_zero_load": float((df_fits["status_delta"] == "at_or_below_zero_load").mean()),
                "n_total_reachable": int((~df_fits["unreachable_delta"]).sum()),
                "fraction_total_reachable": float((~df_fits["unreachable_delta"]).mean()),
                "n_unreachable": int(df_fits["unreachable_delta"].sum()),
                "fraction_unreachable": float(df_fits["unreachable_delta"].mean()),
                "chambers_with_any_unreachable": int(ch_gb_del["any"].sum()),
                "chambers_with_all_unreachable": int(ch_gb_del["all"].sum()),
                "fitted_pi_median": float(df_fits["pi_fit_delta"].dropna().median()),
                "fitted_pi_iqr": [
                    float(np.percentile(df_fits["pi_fit_delta"].dropna(), 25)),
                    float(np.percentile(df_fits["pi_fit_delta"].dropna(), 75)),
                ],
                "max_reachable_depth_median_um": float(df_fits["max_depth_delta_um"].median()),
            },
            "raw_hull_depth_d_Pi": {
                "n_in_range": int((df_fits["status_raw_hull"] == "in_range").sum()),
                "fraction_in_range": float((df_fits["status_raw_hull"] == "in_range").mean()),
                "n_at_or_below_zero_load": int((df_fits["status_raw_hull"] == "at_or_below_zero_load").sum()),
                "fraction_at_or_below_zero_load": float((df_fits["status_raw_hull"] == "at_or_below_zero_load").mean()),
                "n_total_reachable": int((~df_fits["unreachable_raw_hull"]).sum()),
                "fraction_total_reachable": float((~df_fits["unreachable_raw_hull"]).mean()),
                "n_unreachable": int(df_fits["unreachable_raw_hull"].sum()),
                "fraction_unreachable": float(df_fits["unreachable_raw_hull"].mean()),
                "chambers_with_any_unreachable": int(ch_gb_raw["any"].sum()),
                "chambers_with_all_unreachable": int(ch_gb_raw["all"].sum()),
                "fitted_pi_median": float(df_fits["pi_fit_raw_hull"].dropna().median()),
                "fitted_pi_iqr": [
                    float(np.percentile(df_fits["pi_fit_raw_hull"].dropna(), 25)),
                    float(np.percentile(df_fits["pi_fit_raw_hull"].dropna(), 75)),
                ],
                "max_reachable_depth_median_um": float(df_fits["max_depth_raw_hull_um"].median()),
            },
            "monotonicity": {
                "n_visits_nonmonotone_delta_gt_0p05um": int(df_fits["curve_nonmonotone_delta_gt_0p05"].sum()),
                "fraction_visits_monotone_delta_within_0p05um": float((~df_fits["curve_nonmonotone_delta_gt_0p05"]).mean()),
                "n_visits_strictly_monotone_delta_gt_neg1e4": int((df_fits["min_diff_delta_um"] >= -1e-4).sum()),
                "worst_step_drop_delta_um": float(df_fits["min_diff_delta_um"].min()),
                "median_pi_at_max_depth_delta": float(df_fits["pi_at_max_delta"].median()),
                "fraction_visits_peak_at_pi12": float((df_fits["pi_at_max_delta"] == 12.0).mean()),
            },
        },
        "timing_and_projection": {
            "wall_time_20_chambers_parallel_6workers_s": float(pool_wall_s),
            "wall_time_per_chamber_parallel_s": wall_per_ch_parallel_s,
            "cpu_time_per_chamber_serial_mean_s": mean_ch_s,
            "cpu_time_per_chamber_serial_median_s": float(np.median(chamber_times_s)),
            "projected_wall_time_305_chambers_parallel_6workers_s": proj_305_parallel_s,
            "projected_wall_time_305_chambers_parallel_6workers_min": float(proj_305_parallel_s / 60.0),
            "projected_cpu_time_305_chambers_serial_min": float(proj_305_serial_cpu_s / 60.0),
        },
        "real_visits_1p0_vs_0p5_mesh_check": real_mesh_check,
        "five_example_curves": example_curves,
        "gateA_reconciliation": gateA_reconciliation,
    }

    (STEP6B_DIR / "gateB.json").write_text(json.dumps(gateB_summary, indent=2))
    print("=== Stage B Gate Summary Saved to data/step6b/gateB.json ===", flush=True)
    print(f"Total runtime: {time.perf_counter() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
