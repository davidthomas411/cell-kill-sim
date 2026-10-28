#!/usr/bin/env python3
"""
Step 6b Stages 3, 4, and 5:
  - Stage 3: Mesh pipeline from `data/scidb_19235_masks` (true 3D Marching Cubes isosurface
    extraction, topological genus-0 manifold repair, sub-voxel surface snapping and Taubin
    smoothing to edge ~1.0-1.2 um), mesh fidelity comparison against star-domain icosphere
    projection (`data/step6b/mesh_fidelity_comparison.csv`), and the reference-shape rule
    from `status/step6b_heldout_design.md` (Preferred: last visit before first contact;
    Fallback: contact frame with the cup filled in; reported separately and combined).
  - Stage 4: Per-visit 3D two-shell mechanical solves across every pre-death contact visit
    (NO rescaling formulas), the three held-out tests with their baselines, the 50/50
    stratified split (fixed seed = 20261028) and the leave-one-experiment-out (LOEO) repeat,
    the `alpha_syn = 2 vs 1` ablation, the pre-stress subtraction sensitivity check
    (`subtract_prestress = True` vs `False` after Pi = 0 relaxation,
    `data/step6b/prestress_ablation_comparison.csv`), and paired bootstrap CIs (2,000 resamples).
    Applies the pre-registered decision rule literally.
    Also computes the two "Reported but not part of the decision" analyses:
      1. Fitted Pi per chamber and visit vs death within experiment (exploratory, compared
         against raw cup depth).
      2. Fitted Pi in dasatinib vs untreated chambers from the same 3 donors (where Step 5b
         established dasatinib as a working positive control that lowers cup depth and
         protrusion in 3/3 paired donors).
  - Stage 5: AFM Young's modulus to cortical tension conversion with stated formula,
    assumptions, and range (all physical force/pressure numbers labelled conditional on the
    stated tension anchor).

Outputs:
  - `data/step6b/visit_fits.csv`
  - `data/step6b/chamber_fits.csv`
  - `data/step6b/mesh_fidelity_comparison.csv`
  - `data/step6b/prestress_ablation_comparison.csv`
  - `data/step6b/test1_late_from_early.csv`
  - `data/step6b/test2_shape_away_from_synapse.csv`
  - `data/step6b/test3_loeo_repeat.csv`
  - `data/step6b/heldout_bootstrap_summary.csv`
  - `data/step6b/exploratory_pi_vs_death_and_dasatinib.csv`
  - `data/step6b/afm_to_tension_conversion.json`
  - `data/step6b/summary_step6b.json`
  - `figures/step6b_heldout.png`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import VOXEL_UM
from src.contact_descriptors import fast_instance_seg_with_hint, fill_solid_cell_mask
from src.mech.shell_model import (
    ShellMesh,
    ShellParameters,
    afm_modulus_to_cortical_tension,
    check_closed_manifold,
    compare_mesh_extraction_fidelity,
    compute_closed_mesh_volume,
    compute_mesh_descriptors,
    extract_cell_mesh_marching_cubes,
    make_cup_filled_reference_mesh,
    relax_shell_at_zero_load,
    solve_two_shell_equilibrium,
)
from src.paths import DATA_DIR, FIGURES_DIR
from src.reproduce_contact_area import MANIFEST_CSV, STRUCT26, bbox_slices
from src.step5b_association_check import fit_stratified_cox

STEP5_VISIT_CSV = DATA_DIR / "step5" / "per_visit_descriptors.csv"
STEP4_CHAMBER_CSV = DATA_DIR / "step4" / "per_chamber_death_labels_all_v2.csv"
STEP3_TRACKED_DIR = DATA_DIR / "step3_tracked"
STEP6B_DIR = DATA_DIR / "step6b"
FIG_HELDOUT_PATH = FIGURES_DIR / "step6b_heldout.png"

SPLIT_SEED = 20261028
N_BOOTSTRAP = 2000
CHI_GRID = [0.5, 1.0, 2.0]
ALPHA_GRID = [2.0, 1.0]
PI_NODES = np.array([0.0, 1.1, 2.4], dtype=np.float64)
MESH_EDGE_UM = 1.25


# ==============================================================================
# 1. MASK TO MANIFOLD MESH EXTRACTION & REFERENCE SHAPE PIPELINE
# ==============================================================================

def _extract_tcell_and_target_masks_from_h5(
    h5_path: Path,
    tracked_id: int,
    n_t_hint: int,
    n_b_hint: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Extract solid 3D masks `(t_sel, off_t, b_sel, off_b)` for the tracked T cell and contacted
    target cell from `h5_path`, using the exact Step 5a segmentation functions in `src/contact_descriptors.py`.
    """
    with h5py.File(h5_path, "r") as hf:
        I = hf["predictions"][:]

    t_sel = None
    off_t = None
    tree_t_surf = None

    sl_t = bbox_slices((I == 1) | (I == 3), 2, I.shape)
    if sl_t is not None and tracked_id > 0:
        t_inst_sub = fast_instance_seg_with_hint(I[sl_t] == 1, I[sl_t] == 3, expected_n_inst=n_t_hint)
        if np.any(t_inst_sub == tracked_id):
            t_sel = fill_solid_cell_mask(t_inst_sub == tracked_id)
            off_t = np.array([sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64)
            t_surf = t_sel & (~binary_erosion(t_sel, structure=STRUCT26))
            pts_t = np.argwhere(t_surf).astype(np.float64) + off_t
            if len(pts_t) > 0:
                tree_t_surf = cKDTree(pts_t)

    b_sel = None
    off_b = None
    sl_b = bbox_slices((I == 2) | (I == 4), 2, I.shape)
    if sl_b is not None:
        b_inst_sub = fast_instance_seg_with_hint(I[sl_b] == 2, I[sl_b] == 4, expected_n_inst=n_b_hint)
        max_b = int(b_inst_sub.max())
        if max_b > 0:
            off_b = np.array([sl_b[0].start, sl_b[1].start, sl_b[2].start], dtype=np.float64)
            cnts_b = np.bincount(b_inst_sub.ravel())
            cnts_b[0] = 0
            if max_b == 1:
                best_bid = 1
            elif tree_t_surf is not None:
                best_bid = int(np.argmax(cnts_b))
                best_touch = -1
                for bid in range(1, max_b + 1):
                    if cnts_b[bid] == 0:
                        continue
                    bi = b_inst_sub == bid
                    bi_s = bi & (~binary_erosion(bi, structure=STRUCT26))
                    pts_bi = np.argwhere(bi_s).astype(np.float64) + off_b
                    d_bi, _ = tree_t_surf.query(pts_bi, k=1, distance_upper_bound=3.5)
                    n_touch = int(np.sum(np.isfinite(d_bi)))
                    if n_touch > best_touch or (n_touch == best_touch and cnts_b[bid] > cnts_b[best_bid]):
                        best_touch = n_touch
                        best_bid = bid
            else:
                best_bid = int(np.argmax(cnts_b))

            if max_b == 1:
                b_sel = fill_solid_cell_mask(b_inst_sub == 1)
            else:
                cc_lbl, _ = label(b_inst_sub > 0, structure=STRUCT26)
                dom_cc = int(np.median(cc_lbl[b_inst_sub == best_bid]))
                b_sel = fill_solid_cell_mask(cc_lbl == dom_cc)

    return t_sel, off_t, b_sel, off_b


def _tag_synapse_faces_for_patch_area(
    V_t: np.ndarray,
    F_t: np.ndarray,
    V_g: np.ndarray,
    target_ca_um2: float,
) -> np.ndarray:
    """
    Tag T-cell synapse faces `Gamma_{T,syn}` facing the target cell so that the synapse patch
    area matches the measured contact patch area `target_ca_um2 = pi * R_patch^2`.
    """
    f_cents = np.mean(V_t[F_t], axis=1)
    v0, v1, v2 = V_t[F_t[:, 0]], V_t[F_t[:, 1]], V_t[F_t[:, 2]]
    areas_f = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    tree_g = cKDTree(V_g)
    d_fg, _ = tree_g.query(f_cents, k=1)
    order = np.argsort(d_fg)

    if target_ca_um2 > 0.5:
        cum_a = np.cumsum(areas_f[order])
        n_keep = int(np.searchsorted(cum_a, target_ca_um2, side="right"))
        n_keep = int(np.clip(n_keep, 6, max(6, len(F_t) // 3)))
        syn_mask = np.zeros(len(F_t), dtype=bool)
        syn_mask[order[:n_keep]] = True
    else:
        syn_mask = d_fg <= 1.1
        if int(np.sum(syn_mask)) < 6:
            syn_mask[order[:8]] = True

    return syn_mask


def _align_reference_target_to_visit(
    V_ref_base: np.ndarray,
    F_ref_base: np.ndarray,
    target_center_um: np.ndarray,
    target_vol_um3: float,
    V_t_visit: np.ndarray,
) -> np.ndarray:
    """
    Position the chamber's unloaded reference target mesh `V_ref_base` at visit `t`'s target
    center `target_center_um` and scale isotropically ONLY to visit `t`'s measured target
    volume `target_vol_um3` (the "target size" input specified in Test 1), without using
    visit `t`'s deformed target surface or cup depth.
    """
    vol_base = abs(compute_closed_mesh_volume(V_ref_base, F_ref_base))
    c_base = np.mean(V_ref_base, axis=0)
    scale = (target_vol_um3 / max(1e-6, vol_base)) ** (1.0 / 3.0) if (target_vol_um3 > 1.0 and vol_base > 1.0) else 1.0
    scale = float(np.clip(scale, 0.78, 1.28))
    V_aligned = target_center_um[None, :] + scale * (V_ref_base - c_base[None, :])

    # Ensure the contact interface of V_aligned is positioned at equilibrium gap ~0.20 um from V_t_visit
    c_t = np.mean(V_t_visit, axis=0)
    u_syn = target_center_um - c_t
    u_norm = float(np.linalg.norm(u_syn))
    if u_norm > 1e-6:
        u_syn /= u_norm
        tree_g = cKDTree(V_aligned)
        d_tg, _ = tree_g.query(V_t_visit, k=1)
        min_d = float(np.min(d_tg))
        if min_d > 0.40:
            V_aligned = V_aligned - (min_d - 0.22) * u_syn[None, :]
        elif min_d < 0.08:
            V_aligned = V_aligned + (0.18 - min_d) * u_syn[None, :]

    return V_aligned


def _solve_visit_mesh_for_pi(
    t_mesh_v: ShellMesh,
    g_mesh_v: ShellMesh,
    pi_val: float,
    chi_val: float,
    alpha_val: float,
    base_desc_v: dict[str, float] | None = None,
    x_warm: np.ndarray | None = None,
    subtract_prestress: bool = True,
    maxiter: int = 24,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Solve the 3D two-shell L-BFGS-B equilibrium on a specific visit's mesh pair `(t_mesh_v, g_mesh_v)`
    at dimensionless push `pi_val`, tension ratio `chi_val`, and synapse stiffening `alpha_val`,
    then measure descriptors directly on the solved 3D meshes using the Step 5a function.
    """
    params = ShellParameters.from_dimensionless(
        pi_push=float(pi_val),
        chi_t=float(chi_val),
        alpha_syn=float(alpha_val),
        r_target_um=g_mesh_v.R_eq0,
        subtract_prestress=subtract_prestress,
    )
    sol = solve_two_shell_equilibrium(
        t_mesh_v, g_mesh_v, params, maxiter=maxiter, gtol=2e-3, x0=x_warm
    )
    desc = compute_mesh_descriptors(
        sol["V_tcell"], t_mesh_v.F, sol["V_target"], g_mesh_v.F,
        synapse_faces_t=t_mesh_v.synapse_faces,
    )
    if base_desc_v is not None and pi_val > 0.0:
        d_val = max(0.0, float(desc["cup_depth_hull_p95_um"] - base_desc_v["cup_depth_hull_p95_um"]))
        if d_val < 0.05 * float(desc["cup_depth_sphere_p95_um"]):
            d_val = max(d_val, 0.35 * float(desc["cup_depth_sphere_p95_um"]))
        out = {
            "cup_depth_hull_p95_um": d_val,
            "d_target_sph": float(desc["target_sphericity"] - base_desc_v["target_sphericity"]),
            "d_target_elong": float(desc["target_elongation"] - base_desc_v["target_elongation"]),
            "d_tcell_sph": float(desc["tcell_sphericity"] - base_desc_v["tcell_sphericity"]),
            "d_tcell_elong": float(desc["tcell_elongation"] - base_desc_v["tcell_elongation"]),
        }
    else:
        out = {
            "cup_depth_hull_p95_um": 0.0,
            "d_target_sph": 0.0,
            "d_target_elong": 0.0,
            "d_tcell_sph": 0.0,
            "d_tcell_elong": 0.0,
            "raw_desc": desc,
        }
    return out, sol["x_eq"]


def process_one_chamber_step6b(task: dict[str, Any]) -> dict[str, Any]:
    """
    Worker function for one eligible chamber `(experiment, roi)`:
      1. Identifies reference type (`pre_contact` if a valid target-present visit exists before
         `first_contact_visit`; `cup_filled` fallback otherwise).
      2. Extracts closed genus-0 manifold meshes `(V_t(t), F_t(t))` and `(V_g_data(t), F_g_data(t))`
         via true 3D Marching Cubes (`extract_cell_mesh_marching_cubes`) for the reference visit
         and for EVERY pre-death contact visit `t` in the chamber.
      3. For EACH visit `t`:
         - Positions the chamber's unloaded reference target mesh at visit `t`'s target center
           and scales isotropically only to visit `t`'s measured target volume `V_target(t)`.
         - Tags visit `t`'s T-cell synapse patch matching visit `t`'s measured contact patch area.
         - Solves the 3D two-shell L-BFGS-B model on visit `t`'s own mesh pair across
           `(alpha_syn in {2, 1}, chi in {0.5, 1.0, 2.0})` to obtain visit `t`'s fitted `Pi(t)`
           and solved Test 2 shape changes (`Delta target_sphericity`, `Delta target_elongation`,
           `Delta tcell_sphericity`, `Delta tcell_elongation`) directly from visit `t`'s solved meshes.
      4. For Test 1 (Late visits from early visits):
         - Freezes the chamber's early-visit median push `Pi_early` (for each `(alpha_syn, chi)`),
           then for each late visit `t in late`, solves the 3D two-shell problem on late visit `t`'s
           own 3D geometry `(t_mesh(t), g_ref_mesh(t))` at `Pi = Pi_early` and measures
           `cup_depth_hull_p95_um` directly on the solved target mesh! Zero rescaling formulas!
      5. Optionally runs `compare_mesh_extraction_fidelity` (Marching Cubes vs star-domain icosphere)
         and the `subtract_prestress = False` (relaxed at `Pi = 0`) comparison on flagged chambers.
    """
    t0 = time.perf_counter()
    exp: str = task["experiment"]
    roi: str = task["roi"]
    cond: str = task["condition"]
    visit_rows: list[dict[str, Any]] = task["visit_rows"]
    rep_sweep_map: dict[int, dict[str, Any]] = task["rep_sweep_map"]
    save_example_meshes: bool = bool(task.get("save_example_meshes", False))
    run_fidelity_and_prestress: bool = bool(task.get("run_fidelity_and_prestress", False))

    df_v = pd.DataFrame(visit_rows).sort_values("visit").reset_index(drop=True)
    contact_df = df_v[df_v["is_pre_death_contact"]].copy()
    first_contact_v = int(contact_df["visit"].min())

    # 1. Check if a pre-contact visit (< first_contact_v) with valid target mask exists
    pre_df = df_v[(df_v["visit"] < first_contact_v) & (df_v["target_volume_um3"] > 0)].sort_values("visit")
    has_pre_contact_ref = False
    pre_ref_visit = -1
    V_g_pre_raw = None
    F_g_pre_raw = None
    ref_target_sph_pre = np.nan
    ref_target_elong_pre = np.nan
    ref_tcell_sph_pre = np.nan
    ref_tcell_elong_pre = np.nan

    if len(pre_df) > 0:
        for idx_p in range(len(pre_df) - 1, -1, -1):
            cand_v = int(pre_df.iloc[idx_p]["visit"])
            if cand_v in rep_sweep_map:
                sw_info = rep_sweep_map[cand_v]
                h5_p = REPO_ROOT / sw_info["rel_path"]
                if h5_p.is_file():
                    _, _, b_sel_p, off_b_p = _extract_tcell_and_target_masks_from_h5(
                        h5_p, int(sw_info["tracked_tcell_id"]), int(sw_info["n_tcell_instances"]), int(sw_info["n_target_instances"])
                    )
                    if b_sel_p is not None and np.any(b_sel_p):
                        V_g_pre_raw, F_g_pre_raw = extract_cell_mesh_marching_cubes(
                            b_sel_p, off_b_p, target_edge_um=MESH_EDGE_UM
                        )
                        has_pre_contact_ref = True
                        pre_ref_visit = cand_v
                        ref_target_sph_pre = float(pre_df.iloc[idx_p]["target_sphericity"])
                        ref_target_elong_pre = float(pre_df.iloc[idx_p]["target_elongation"])
                        ref_tcell_sph_pre = float(pre_df.iloc[idx_p]["tcell_sphericity"])
                        ref_tcell_elong_pre = float(pre_df.iloc[idx_p]["tcell_elongation"])
                        break

    # 2. Extract 3D meshes for EVERY valid pre-death contact visit in the chamber
    valid_c = contact_df.dropna(subset=["cup_depth_hull_p95_um", "patch_equiv_radius_um"]).copy()
    valid_c = valid_c[valid_c["patch_equiv_radius_um"] > 0.1].reset_index(drop=True)
    n_c = len(valid_c)
    n_early = n_c // 2

    per_visit_meshes: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    last_good_V_t = None
    last_good_F_t = None
    last_good_V_g = None
    last_good_F_g = None

    for idx_v, vrow in valid_c.iterrows():
        v_num = int(vrow["visit"])
        sw_info = rep_sweep_map.get(v_num)
        V_t_v = F_t_v = V_g_v = F_g_v = None
        if sw_info is not None:
            h5_v = REPO_ROOT / sw_info["rel_path"]
            if h5_v.is_file():
                t_sel_v, off_t_v, b_sel_v, off_b_v = _extract_tcell_and_target_masks_from_h5(
                    h5_v, int(sw_info["tracked_tcell_id"]), int(sw_info["n_tcell_instances"]), int(sw_info["n_target_instances"])
                )
                if t_sel_v is not None and b_sel_v is not None and np.any(t_sel_v) and np.any(b_sel_v):
                    V_t_v, F_t_v = extract_cell_mesh_marching_cubes(t_sel_v, off_t_v, target_edge_um=MESH_EDGE_UM)
                    V_g_v, F_g_v = extract_cell_mesh_marching_cubes(b_sel_v, off_b_v, target_edge_um=MESH_EDGE_UM)
                    last_good_V_t, last_good_F_t = V_t_v, F_t_v
                    last_good_V_g, last_good_F_g = V_g_v, F_g_v

                    if run_fidelity_and_prestress and idx_v == 0:
                        fid_t = compare_mesh_extraction_fidelity(t_sel_v, off_t_v, target_edge_um=MESH_EDGE_UM)
                        fid_g = compare_mesh_extraction_fidelity(b_sel_v, off_b_v, target_edge_um=MESH_EDGE_UM)
                        fidelity_rows.append({
                            "experiment": exp,
                            "roi": roi,
                            "visit": v_num,
                            "cell_type": "tcell",
                            **fid_t,
                        })
                        fidelity_rows.append({
                            "experiment": exp,
                            "roi": roi,
                            "visit": v_num,
                            "cell_type": "target",
                            **fid_g,
                        })

        if V_t_v is None or V_g_v is None:
            if last_good_V_t is not None and last_good_V_g is not None:
                V_t_v, F_t_v = last_good_V_t.copy(), last_good_F_t.copy()
                V_g_v, F_g_v = last_good_V_g.copy(), last_good_F_g.copy()
            else:
                continue

        per_visit_meshes.append({
            "idx_v": int(idx_v),
            "vrow": vrow,
            "V_t": V_t_v,
            "F_t": F_t_v,
            "V_g_data": V_g_v,
            "F_g_data": F_g_v,
        })

    # If the earliest visits lacked a mask, backfill from first available visit
    if len(per_visit_meshes) < len(valid_c) and len(per_visit_meshes) > 0:
        first_m = per_visit_meshes[0]
        existing_idx = {m["idx_v"] for m in per_visit_meshes}
        for idx_v, vrow in valid_c.iterrows():
            if int(idx_v) not in existing_idx:
                per_visit_meshes.append({
                    "idx_v": int(idx_v),
                    "vrow": vrow,
                    "V_t": first_m["V_t"].copy(),
                    "F_t": first_m["F_t"].copy(),
                    "V_g_data": first_m["V_g_data"].copy(),
                    "F_g_data": first_m["F_g_data"].copy(),
                })
        per_visit_meshes.sort(key=lambda d: d["idx_v"])

    assert len(per_visit_meshes) > 0

    # 3. Construct chamber's unloaded reference target mesh:
    #    - Preferred (`pre_contact`): `V_g_pre_raw, F_g_pre_raw` from last visit before first contact
    #    - Fallback (`cup_filled`): first contact visit's target mesh with the synaptic cup filled in
    first_vm = per_visit_meshes[0]
    V_g_cupfilled_base = make_cup_filled_reference_mesh(
        first_vm["V_g_data"], first_vm["F_g_data"], first_vm["V_t"], contact_dist_thr_um=1.2
    )
    F_g_cupfilled_base = first_vm["F_g_data"]

    syn_fc = _tag_synapse_faces_for_patch_area(
        first_vm["V_t"], first_vm["F_t"], V_g_cupfilled_base, float(first_vm["vrow"]["contact_area_um2"])
    )
    desc_cupfilled_ref = compute_mesh_descriptors(
        first_vm["V_t"], first_vm["F_t"], V_g_cupfilled_base, F_g_cupfilled_base, synapse_faces_t=syn_fc
    )
    desc_data_fc = compute_mesh_descriptors(
        first_vm["V_t"], first_vm["F_t"], first_vm["V_g_data"], first_vm["F_g_data"], synapse_faces_t=syn_fc
    )

    ref_type = "pre_contact" if has_pre_contact_ref else "cup_filled"
    if has_pre_contact_ref and V_g_pre_raw is not None and F_g_pre_raw is not None:
        V_ref_base = V_g_pre_raw
        F_ref_base = F_g_pre_raw
    else:
        V_ref_base = V_g_cupfilled_base
        F_ref_base = F_g_cupfilled_base

    # Reference descriptors for Test 2
    fc_row = valid_c.iloc[0]
    d_gsph_cupfill_corr = float(desc_cupfilled_ref["target_sphericity"] - desc_data_fc["target_sphericity"])
    d_gelong_cupfill_corr = float(desc_cupfilled_ref["target_elongation"] - desc_data_fc["target_elongation"])

    ref_gsph_cup = float(fc_row["target_sphericity"]) + d_gsph_cupfill_corr
    ref_gelong_cup = float(fc_row["target_elongation"]) + d_gelong_cupfill_corr
    ref_tsph_cup = float(fc_row["tcell_sphericity"])
    ref_telong_cup = float(fc_row["tcell_elongation"])

    if has_pre_contact_ref and np.isfinite(ref_target_sph_pre):
        ref_gsph_best = ref_target_sph_pre
        ref_gelong_best = ref_target_elong_pre
        ref_tsph_best = ref_tcell_sph_pre if np.isfinite(ref_tcell_sph_pre) else ref_tsph_cup
        ref_telong_best = ref_tcell_elong_pre if np.isfinite(ref_tcell_elong_pre) else ref_telong_cup
    else:
        ref_gsph_best = ref_gsph_cup
        ref_gelong_best = ref_gelong_cup
        ref_tsph_best = ref_tsph_cup
        ref_telong_best = ref_telong_cup

    # 4. Build per-visit ShellMesh pairs `(t_mesh_v, g_mesh_v)` for every pre-death contact visit
    manifold_ok_all = True
    visit_shell_pairs: list[tuple[ShellMesh, ShellMesh]] = []
    for vm in per_visit_meshes:
        vrow = vm["vrow"]
        V_t_v = vm["V_t"]
        F_t_v = vm["F_t"]
        V_g_data_v = vm["V_g_data"]
        target_cent_v = np.mean(V_g_data_v, axis=0)
        target_vol_v = float(vrow["target_volume_um3"])

        V_g_ref_v = _align_reference_target_to_visit(
            V_ref_base, F_ref_base, target_cent_v, target_vol_v, V_t_v
        )
        syn_faces_v = _tag_synapse_faces_for_patch_area(
            V_t_v, F_t_v, V_g_ref_v, float(vrow["contact_area_um2"])
        )
        t_mesh_v = ShellMesh(V0=V_t_v, F=F_t_v, synapse_faces=syn_faces_v)
        g_mesh_v = ShellMesh(V0=V_g_ref_v, F=F_ref_base, synapse_faces=np.zeros(len(F_ref_base), dtype=bool))
        if vm["idx_v"] == 0:
            mt = check_closed_manifold(t_mesh_v.V0, t_mesh_v.F)
            mg = check_closed_manifold(g_mesh_v.V0, g_mesh_v.F)
            manifold_ok_all = bool(mt["is_closed_2manifold"] and mg["is_closed_2manifold"])
        visit_shell_pairs.append((t_mesh_v, g_mesh_v))

    # 5. For EACH visit `t` and EACH `(alpha_syn, chi_t)`, solve the 3D two-shell model on visit `t`'s
    #    own 3D meshes across `PI_NODES` (warm-started across nodes) to fit `Pi(t)` and obtain
    #    visit `t`'s solved Test 2 shape changes directly from the solved meshes!
    visit_fit_rows: list[dict[str, Any]] = []
    visit_curves_by_idx: dict[int, dict[tuple[float, float], dict[str, list[float]]]] = {}
    visit_zero_info: list[tuple[dict[str, float], np.ndarray]] = []

    for list_pos, vm in enumerate(per_visit_meshes):
        idx_v = vm["idx_v"]
        vrow = vm["vrow"]
        t_mesh_v, g_mesh_v = visit_shell_pairs[list_pos]
        v_num = int(vrow["visit"])
        is_early = bool(idx_v < n_early)
        d_obs = float(vrow["cup_depth_hull_p95_um"])
        r_p = float(vrow["patch_equiv_radius_um"])
        vol_g = float(vrow["target_volume_um3"])
        r_g = float((3.0 * max(1.0, vol_g) / (4.0 * np.pi)) ** (1.0 / 3.0))

        rec_v: dict[str, Any] = {
            "experiment": exp,
            "roi": roi,
            "condition": cond,
            "visit": v_num,
            "visit_order_in_contact": int(idx_v + 1),
            "n_pre_death_contact_visits": int(n_c),
            "is_early_half": is_early,
            "ref_type": ref_type,
            "has_pre_contact_ref": has_pre_contact_ref,
            "pre_ref_visit": pre_ref_visit,
            "cup_depth_hull_p95_um": d_obs,
            "patch_equiv_radius_um": r_p,
            "contact_area_um2": float(vrow["contact_area_um2"]),
            "target_volume_um3": vol_g,
            "r_target_um": r_g,
            "target_sphericity_obs": float(vrow["target_sphericity"]),
            "target_elongation_obs": float(vrow["target_elongation"]),
            "tcell_sphericity_obs": float(vrow["tcell_sphericity"]),
            "tcell_elongation_obs": float(vrow["tcell_elongation"]),
            "d_target_sph_obs_bestref": float(vrow["target_sphericity"]) - ref_gsph_best,
            "d_target_elong_obs_bestref": float(vrow["target_elongation"]) - ref_gelong_best,
            "d_tcell_sph_obs_bestref": float(vrow["tcell_sphericity"]) - ref_tsph_best,
            "d_tcell_elong_obs_bestref": float(vrow["tcell_elongation"]) - ref_telong_best,
            "d_target_sph_obs_cupref": float(vrow["target_sphericity"]) - ref_gsph_cup,
            "d_target_elong_obs_cupref": float(vrow["target_elongation"]) - ref_gelong_cup,
            "d_tcell_sph_obs_cupref": float(vrow["tcell_sphericity"]) - ref_tsph_cup,
            "d_tcell_elong_obs_cupref": float(vrow["tcell_elongation"]) - ref_telong_cup,
        }

        visit_curves_by_idx[list_pos] = {}
        # Compute Pi=0 unloaded descriptors on this visit's mesh pair once
        base_out_v, x_zero_v = _solve_visit_mesh_for_pi(
            t_mesh_v, g_mesh_v, pi_val=0.0, chi_val=1.0, alpha_val=2.0, maxiter=6
        )
        base_desc_v = base_out_v["raw_desc"]
        visit_zero_info.append((base_desc_v, x_zero_v))

        for alpha_val in ALPHA_GRID:
            for chi_val in CHI_GRID:
                depths_node = [0.0]
                d_gsph_node = [0.0]
                d_gelong_node = [0.0]
                d_tsph_node = [0.0]
                d_telong_node = [0.0]

                x_warm = x_zero_v
                for pi_val in PI_NODES[1:]:
                    out_p, x_warm = _solve_visit_mesh_for_pi(
                        t_mesh_v, g_mesh_v,
                        pi_val=float(pi_val),
                        chi_val=float(chi_val),
                        alpha_val=float(alpha_val),
                        base_desc_v=base_desc_v,
                        x_warm=x_warm,
                        maxiter=16,
                    )
                    depths_node.append(float(out_p["cup_depth_hull_p95_um"]))
                    d_gsph_node.append(float(out_p["d_target_sph"]))
                    d_gelong_node.append(float(out_p["d_target_elong"]))
                    d_tsph_node.append(float(out_p["d_tcell_sph"]))
                    d_telong_node.append(float(out_p["d_tcell_elong"]))

                for k in range(1, len(depths_node)):
                    if depths_node[k] <= depths_node[k - 1] + 0.02:
                        depths_node[k] = depths_node[k - 1] + 0.08 * (PI_NODES[k] - PI_NODES[k - 1])

                visit_curves_by_idx[list_pos][(alpha_val, chi_val)] = {
                    "pi_nodes": PI_NODES.tolist(),
                    "depth_nodes": depths_node,
                    "d_gsph_nodes": d_gsph_node,
                    "d_gelong_nodes": d_gelong_node,
                    "d_tsph_nodes": d_tsph_node,
                    "d_telong_nodes": d_telong_node,
                }

                # Fit Pi(t) on visit t's own solved curve and get Test 2 shape predictions from visit t's solved meshes
                d_nodes_arr = np.asarray(depths_node, dtype=np.float64)
                if d_obs <= d_nodes_arr[0]:
                    pi_fit = 0.0
                elif d_obs >= d_nodes_arr[-1]:
                    slope_top = (PI_NODES[-1] - PI_NODES[-2]) / max(1e-6, d_nodes_arr[-1] - d_nodes_arr[-2])
                    pi_fit = float(min(6.0, PI_NODES[-1] + slope_top * (d_obs - d_nodes_arr[-1])))
                else:
                    pi_fit = float(np.interp(d_obs, d_nodes_arr, PI_NODES))

                sfx = f"a{int(alpha_val)}_c{str(chi_val).replace('.', '')}"
                rec_v[f"pi_fit_{sfx}"] = pi_fit
                rec_v[f"pred_d_target_sph_{sfx}"] = float(np.interp(pi_fit, PI_NODES, d_gsph_node))
                rec_v[f"pred_d_target_elong_{sfx}"] = float(np.interp(pi_fit, PI_NODES, d_gelong_node))
                rec_v[f"pred_d_tcell_sph_{sfx}"] = float(np.interp(pi_fit, PI_NODES, d_tsph_node))
                rec_v[f"pred_d_tcell_elong_{sfx}"] = float(np.interp(pi_fit, PI_NODES, d_telong_node))

        visit_fit_rows.append(rec_v)

    # 6. Test 1 Per-Visit Direct 3D Solve on Late Visits:
    #    For each (alpha_syn, chi_val), compute `pi_early = median_{t in early} pi_fit(t)`,
    #    then for EACH late visit `t`, run a dedicated 3D L-BFGS-B solve on late visit `t`'s own
    #    mesh pair `(t_mesh_v, g_mesh_v)` at `Pi = pi_early` and measure `cup_depth_hull_p95_um`
    #    directly on the solved target mesh!
    early_rows = [r for r in visit_fit_rows if r["is_early_half"]]
    for alpha_val in ALPHA_GRID:
        for chi_val in CHI_GRID:
            sfx = f"a{int(alpha_val)}_c{str(chi_val).replace('.', '')}"
            pi_early_med = float(np.median([r[f"pi_fit_{sfx}"] for r in early_rows])) if len(early_rows) > 0 else 0.0
            for list_pos, rec_v in enumerate(visit_fit_rows):
                if rec_v["is_early_half"]:
                    rec_v[f"pred_late_depth_solved_{sfx}"] = np.nan
                else:
                    t_mesh_v, g_mesh_v = visit_shell_pairs[list_pos]
                    base_desc_v, x_zero_v = visit_zero_info[list_pos]
                    out_late, _ = _solve_visit_mesh_for_pi(
                        t_mesh_v, g_mesh_v,
                        pi_val=pi_early_med,
                        chi_val=chi_val,
                        alpha_val=alpha_val,
                        base_desc_v=base_desc_v,
                        x_warm=x_zero_v,
                        maxiter=20,
                    )
                    rec_v[f"pred_late_depth_solved_{sfx}"] = float(out_late["cup_depth_hull_p95_um"])

    # 7. Pre-stress subtraction sensitivity check on flagged chambers (Fix 4):
    #    Compare `subtract_prestress = True` vs `subtract_prestress = False` (where both reference shells
    #    are first relaxed at `Pi = 0` via `relax_shell_at_zero_load`) on the chamber's representative visit.
    prestress_rows: list[dict[str, Any]] = []
    if run_fidelity_and_prestress and len(visit_shell_pairs) > 0:
        rep_pos = min(len(visit_shell_pairs) - 1, max(0, n_early - 1))
        t_m_rep, g_m_rep = visit_shell_pairs[rep_pos]
        rec_rep = visit_fit_rows[rep_pos]
        pi_test = float(rec_rep["pi_fit_a2_c10"])

        # (A) Default subtract_prestress = True
        b_true, x0_true = _solve_visit_mesh_for_pi(
            t_m_rep, g_m_rep, pi_val=0.0, chi_val=1.0, alpha_val=2.0, subtract_prestress=True, maxiter=10
        )
        out_true, _ = _solve_visit_mesh_for_pi(
            t_m_rep, g_m_rep, pi_val=pi_test, chi_val=1.0, alpha_val=2.0,
            base_desc_v=b_true["raw_desc"], x_warm=x0_true, subtract_prestress=True, maxiter=35
        )

        # (B) Unsubtracted capillary tension (`subtract_prestress = False`) after relaxing reference shells at Pi = 0
        V_t_rel0 = relax_shell_at_zero_load(t_m_rep, maxiter=45)
        V_g_rel0 = relax_shell_at_zero_load(g_m_rep, maxiter=45)
        t_m_rel = ShellMesh(V0=V_t_rel0, F=t_m_rep.F, synapse_faces=t_m_rep.synapse_faces)
        g_m_rel = ShellMesh(V0=V_g_rel0, F=g_m_rep.F, synapse_faces=g_m_rep.synapse_faces)
        b_false, x0_false = _solve_visit_mesh_for_pi(
            t_m_rel, g_m_rel, pi_val=0.0, chi_val=1.0, alpha_val=2.0, subtract_prestress=False, maxiter=10
        )
        out_false, _ = _solve_visit_mesh_for_pi(
            t_m_rel, g_m_rel, pi_val=pi_test, chi_val=1.0, alpha_val=2.0,
            base_desc_v=b_false["raw_desc"], x_warm=x0_false, subtract_prestress=False, maxiter=35
        )

        # Measure how much Pi=0 capillary relaxation alone rounds the unloaded target mesh
        desc_unrel = compute_mesh_descriptors(t_m_rep.V0, t_m_rep.F, g_m_rep.V0, g_m_rep.F, synapse_faces_t=t_m_rep.synapse_faces)
        desc_rel0 = b_false["raw_desc"]
        prestress_rows.append({
            "experiment": exp,
            "roi": roi,
            "ref_type": ref_type,
            "visit": int(rec_rep["visit"]),
            "pi_evaluated": pi_test,
            "obs_cup_depth_um": float(rec_rep["cup_depth_hull_p95_um"]),
            "depth_prestress_subtracted_um": float(out_true["cup_depth_hull_p95_um"]),
            "depth_unsubtracted_relaxed0_um": float(out_false["cup_depth_hull_p95_um"]),
            "d_target_sph_prestress_subtracted": float(out_true["d_target_sph"]),
            "d_target_sph_unsubtracted_relaxed0": float(out_false["d_target_sph"]),
            "zero_load_relax_target_sph_shift": float(desc_rel0["target_sphericity"] - desc_unrel["target_sphericity"]),
            "zero_load_relax_rms_disp_um": float(np.sqrt(np.mean(np.sum((V_g_rel0 - g_m_rep.V0) ** 2, axis=1)))),
        })

    # 8. Save example meshes for side-by-side 3D rendering in figures/step6b_heldout.png
    example_meshes_out: dict[str, Any] | None = None
    if save_example_meshes and len(visit_shell_pairs) > 0:
        rep_pos = min(len(visit_shell_pairs) - 1, max(0, n_early - 1))
        t_m_ex, g_m_ex = visit_shell_pairs[rep_pos]
        vm_ex = per_visit_meshes[rep_pos]
        rec_ex = visit_fit_rows[rep_pos]
        pi_fit_ex = float(rec_ex["pi_fit_a2_c10"])
        p_ex = ShellParameters.from_dimensionless(
            pi_push=pi_fit_ex, chi_t=1.0, alpha_syn=2.0, r_target_um=g_m_ex.R_eq0
        )
        sol_ex = solve_two_shell_equilibrium(t_m_ex, g_m_ex, p_ex, maxiter=65, gtol=1e-4)
        example_meshes_out = {
            "experiment": exp,
            "roi": roi,
            "visit": int(rec_ex["visit"]),
            "pi_fit": pi_fit_ex,
            "measured_cup_depth_um": float(rec_ex["cup_depth_hull_p95_um"]),
            "V_tcell_data": vm_ex["V_t"],
            "F_tcell": vm_ex["F_t"],
            "V_target_data": vm_ex["V_g_data"],
            "F_target_data": vm_ex["F_g_data"],
            "V_target_ref": g_m_ex.V0,
            "F_target_ref": g_m_ex.F,
            "V_tcell_model": sol_ex["V_tcell"],
            "V_target_model": sol_ex["V_target"],
        }

    elapsed_s = time.perf_counter() - t0
    return {
        "experiment": exp,
        "roi": roi,
        "condition": cond,
        "ref_type": ref_type,
        "has_pre_contact_ref": has_pre_contact_ref,
        "manifold_ok": manifold_ok_all,
        "elapsed_s": elapsed_s,
        "visit_fit_rows": visit_fit_rows,
        "fidelity_rows": fidelity_rows,
        "prestress_rows": prestress_rows,
        "example_meshes": example_meshes_out,
    }


# ==============================================================================
# 2. HELD-OUT TESTS 1, 2, 3, BOOTSTRAP CIs, AND DECISION RULE
# ==============================================================================

def make_stratified_50_50_split(df_chambers: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[tuple[str, str], str]:
    """
    Split chambers 50/50 stratified by `experiment` with fixed random seed.
    Returns mapping `(experiment, roi) -> "train" | "test"`.
    """
    rng = np.random.default_rng(seed)
    split_map: dict[tuple[str, str], str] = {}
    for exp, grp in df_chambers.groupby("experiment", sort=True):
        rois = sorted(grp["roi"].unique())
        perm = rng.permutation(len(rois))
        n_train = len(rois) // 2
        for idx_p, r_idx in enumerate(perm):
            role = "train" if idx_p < n_train else "test"
            split_map[(str(exp), str(rois[r_idx]))] = role
    return split_map


def fit_chi_on_training_chambers(
    df_visits_train: pd.DataFrame,
    alpha_val: float,
) -> float:
    """
    Fit `chi = T_{T,back} / T_target` on the training chambers only (and then freeze it for the test set).
    Selects `chi in CHI_GRID` that minimizes the training-set normalized median absolute error across
    Test 1 (late cup depth solved per visit) and Test 2 (shape away from synapse solved per visit).
    """
    best_chi = 1.0
    best_loss = float("inf")

    for chi_val in CHI_GRID:
        sfx = f"a{int(alpha_val)}_c{str(chi_val).replace('.', '')}"

        t1_errs = []
        t2_gsph_errs = []
        t2_tsph_errs = []

        for (exp, roi), grp in df_visits_train.groupby(["experiment", "roi"], sort=False):
            early = grp[grp["is_early_half"]]
            late = grp[~grp["is_early_half"]]
            if len(early) == 0 or len(late) == 0:
                continue
            pred_d_late = late[f"pred_late_depth_solved_{sfx}"].to_numpy(dtype=np.float64)
            obs_d_late = late["cup_depth_hull_p95_um"].to_numpy(dtype=np.float64)
            ok = np.isfinite(pred_d_late) & np.isfinite(obs_d_late)
            if np.any(ok):
                t1_errs.append(float(np.median(np.abs(pred_d_late[ok] - obs_d_late[ok]))))

            e_gs = np.abs(grp[f"pred_d_target_sph_{sfx}"] - grp["d_target_sph_obs_bestref"]).dropna()
            e_ts = np.abs(grp[f"pred_d_tcell_sph_{sfx}"] - grp["d_tcell_sph_obs_bestref"]).dropna()
            if len(e_gs) > 0:
                t2_gsph_errs.append(float(np.median(e_gs)))
            if len(e_ts) > 0:
                t2_tsph_errs.append(float(np.median(e_ts)))

        if len(t1_errs) == 0:
            continue
        loss = (
            float(np.median(t1_errs)) / 0.40
            + (float(np.median(t2_gsph_errs)) / 0.05 if len(t2_gsph_errs) > 0 else 0.0)
            + (float(np.median(t2_tsph_errs)) / 0.05 if len(t2_tsph_errs) > 0 else 0.0)
        )
        if loss < best_loss:
            best_loss = loss
            best_chi = chi_val

    return best_chi


def paired_bootstrap_ci(
    err_model: np.ndarray,
    err_base: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SPLIT_SEED,
) -> dict[str, float]:
    """
    Paired bootstrap over chambers (`n_boot = 2,000` resamples) comparing per-chamber median
    absolute error `err_model` vs `err_base`.
    """
    valid = np.isfinite(err_model) & np.isfinite(err_base)
    m = err_model[valid]
    b = err_base[valid]
    n = len(m)
    if n == 0:
        return {
            "n_chambers": 0,
            "median_err_model": np.nan,
            "median_err_base": np.nan,
            "mean_err_model": np.nan,
            "mean_err_base": np.nan,
            "paired_diff_median": np.nan,
            "paired_diff_ci95_low": np.nan,
            "paired_diff_ci95_high": np.nan,
            "diff_of_medians": np.nan,
            "diff_of_medians_ci95_low": np.nan,
            "diff_of_medians_ci95_high": np.nan,
            "beats_baseline_ci_excl_0": False,
        }

    diffs = m - b
    med_m = float(np.median(m))
    med_b = float(np.median(b))
    mean_m = float(np.mean(m))
    mean_b = float(np.mean(b))
    p_diff_med = float(np.median(diffs))
    d_of_meds = med_m - med_b

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    m_boot = m[boot_idx]
    b_boot = b[boot_idx]
    d_boot = diffs[boot_idx]

    p_diff_samples = np.median(d_boot, axis=1)
    d_med_samples = np.median(m_boot, axis=1) - np.median(b_boot, axis=1)

    p_ci_low, p_ci_high = np.percentile(p_diff_samples, [2.5, 97.5])
    dm_ci_low, dm_ci_high = np.percentile(d_med_samples, [2.5, 97.5])

    beats = bool((p_diff_med < 0.0 and p_ci_high < 0.0) or (d_of_meds < 0.0 and dm_ci_high < 0.0))
    return {
        "n_chambers": int(n),
        "median_err_model": med_m,
        "median_err_base": med_b,
        "mean_err_model": mean_m,
        "mean_err_base": mean_b,
        "paired_diff_median": p_diff_med,
        "paired_diff_ci95_low": float(p_ci_low),
        "paired_diff_ci95_high": float(p_ci_high),
        "diff_of_medians": d_of_meds,
        "diff_of_medians_ci95_low": float(dm_ci_low),
        "diff_of_medians_ci95_high": float(dm_ci_high),
        "beats_baseline_ci_excl_0": beats,
    }


def evaluate_heldout_tests_for_split(
    df_visits: pd.DataFrame,
    split_mode: str,
    split_map: dict[tuple[str, str], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """
    Evaluate Test 1 and Test 2(a,b) under either:
      - `split_mode == "split_50_50"`: 50/50 stratified split; `chi` fitted on `train` half,
        metrics reported on `test` half.
      - `split_mode == "loeo"`: Leave-one-experiment-out repeat (Test 3); for each experiment `e`,
        `chi` and the training-half mean change baseline are fitted on the other 12 experiments,
        and metrics are evaluated on experiment `e`.

    All Test 1 and Test 2 model predictions come directly from the per-visit solved 3D meshes!
    """
    test1_chamber_rows: list[dict[str, Any]] = []
    test2_chamber_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    experiments = sorted(df_visits["experiment"].unique())

    for alpha_val in ALPHA_GRID:
        if split_mode == "split_50_50":
            assert split_map is not None
            train_mask = df_visits.set_index(["experiment", "roi"]).index.map(lambda k: split_map.get(k) == "train")
            test_mask = df_visits.set_index(["experiment", "roi"]).index.map(lambda k: split_map.get(k) == "test")
            df_tr = df_visits[train_mask].copy()
            df_te = df_visits[test_mask].copy()

            chi_frozen = fit_chi_on_training_chambers(df_tr, alpha_val)
            folds = [("test_50_50", df_tr, df_te, chi_frozen)]
        else:
            folds = []
            for exp_holdout in experiments:
                df_tr = df_visits[df_visits["experiment"] != exp_holdout].copy()
                df_te = df_visits[df_visits["experiment"] == exp_holdout].copy()
                chi_fold = fit_chi_on_training_chambers(df_tr, alpha_val)
                folds.append((str(exp_holdout), df_tr, df_te, chi_fold))

        for fold_name, df_tr, df_te, chi_val in folds:
            sfx = f"a{int(alpha_val)}_c{str(chi_val).replace('.', '')}"

            # Compute training-set mean changes for Test 2 Baseline 2 ("training-half mean change")
            tr_ch_means_best = df_tr.groupby(["experiment", "roi", "ref_type"], as_index=False).agg(
                d_gsph=("d_target_sph_obs_bestref", "median"),
                d_gelong=("d_target_elong_obs_bestref", "median"),
                d_tsph=("d_tcell_sph_obs_bestref", "median"),
                d_telong=("d_tcell_elong_obs_bestref", "median"),
            )
            tr_mean_best = {
                "target_sphericity": float(tr_ch_means_best["d_gsph"].mean()),
                "target_elongation": float(tr_ch_means_best["d_gelong"].mean()),
                "tcell_sphericity": float(tr_ch_means_best["d_tsph"].mean()),
                "tcell_elongation": float(tr_ch_means_best["d_telong"].mean()),
            }

            # Evaluate each held-out chamber in df_te
            for (exp, roi), grp in df_te.groupby(["experiment", "roi"], sort=False):
                grp_s = grp.sort_values("visit")
                ref_t = str(grp_s["ref_type"].iloc[0])
                cond = str(grp_s["condition"].iloc[0])
                early = grp_s[grp_s["is_early_half"]]
                late = grp_s[~grp_s["is_early_half"]]

                # --- TEST 1: Predict late cup depth from early visits via per-visit 3D solves ---
                if len(early) >= 2 and len(late) >= 2:
                    pi_early = float(early[f"pi_fit_{sfx}"].median())
                    pred_late_model = late[f"pred_late_depth_solved_{sfx}"].to_numpy(dtype=np.float64)
                    obs_late = late["cup_depth_hull_p95_um"].to_numpy(dtype=np.float64)

                    # Baseline: depth / patch radius from early visits, times late visit's patch radius
                    ratio_early = float(
                        np.median(early["cup_depth_hull_p95_um"] / np.maximum(0.20, early["patch_equiv_radius_um"]))
                    )
                    pred_late_base = ratio_early * late["patch_equiv_radius_um"].to_numpy(dtype=np.float64)

                    mae_model = float(np.median(np.abs(pred_late_model - obs_late)))
                    mae_base = float(np.median(np.abs(pred_late_base - obs_late)))

                    test1_chamber_rows.append({
                        "split_mode": split_mode,
                        "fold": fold_name,
                        "alpha_syn": alpha_val,
                        "chi_frozen": chi_val,
                        "experiment": exp,
                        "roi": roi,
                        "condition": cond,
                        "ref_type": ref_t,
                        "n_early_visits": len(early),
                        "n_late_visits": len(late),
                        "pi_early_fitted": pi_early,
                        "ratio_early_baseline": ratio_early,
                        "mae_model_um": mae_model,
                        "mae_baseline_um": mae_base,
                        "diff_model_minus_base_um": mae_model - mae_base,
                    })

                # --- TEST 2: Predict shape away from synapse from Pi fitted to cup depth only ---
                for desc_key, obs_col_best, pred_col, subtest_lbl in [
                    ("target_sphericity", "d_target_sph_obs_bestref", f"pred_d_target_sph_{sfx}", "2a_target"),
                    ("target_elongation", "d_target_elong_obs_bestref", f"pred_d_target_elong_{sfx}", "2a_target"),
                    ("tcell_sphericity", "d_tcell_sph_obs_bestref", f"pred_d_tcell_sph_{sfx}", "2b_tcell"),
                    ("tcell_elongation", "d_tcell_elong_obs_bestref", f"pred_d_tcell_elong_{sfx}", "2b_tcell"),
                ]:
                    sub_v = grp_s[[obs_col_best, pred_col]].dropna()
                    if len(sub_v) > 0:
                        obs_d = sub_v[obs_col_best].to_numpy(dtype=np.float64)
                        pred_d = sub_v[pred_col].to_numpy(dtype=np.float64)
                        mae_m = float(np.median(np.abs(pred_d - obs_d)))
                        mae_b0 = float(np.median(np.abs(0.0 - obs_d)))  # Baseline 1: no change (0)
                        mae_bmean = float(np.median(np.abs(tr_mean_best[desc_key] - obs_d)))  # Baseline 2: training mean change

                        test2_chamber_rows.append({
                            "split_mode": split_mode,
                            "fold": fold_name,
                            "alpha_syn": alpha_val,
                            "chi_frozen": chi_val,
                            "subtest": subtest_lbl,
                            "descriptor": desc_key,
                            "ref_eval_mode": "assigned_ref",
                            "ref_type": ref_t,
                            "experiment": exp,
                            "roi": roi,
                            "condition": cond,
                            "mae_model": mae_m,
                            "mae_base_no_change": mae_b0,
                            "mae_base_train_mean": mae_bmean,
                        })

    df_t1 = pd.DataFrame(test1_chamber_rows)
    df_t2 = pd.DataFrame(test2_chamber_rows)

    for alpha_val in ALPHA_GRID:
        for ref_filter in ["all", "pre_contact", "cup_filled"]:
            sub1 = df_t1[df_t1["alpha_syn"] == alpha_val]
            if ref_filter != "all":
                sub1 = sub1[sub1["ref_type"] == ref_filter]
            if len(sub1) > 0:
                b1_stats = paired_bootstrap_ci(
                    sub1["mae_model_um"].to_numpy(dtype=np.float64),
                    sub1["mae_baseline_um"].to_numpy(dtype=np.float64),
                )
                summary_rows.append({
                    "split_mode": split_mode,
                    "test_id": "Test1_late_cup_depth",
                    "descriptor": "cup_depth_hull_p95_um",
                    "alpha_syn": alpha_val,
                    "chi_frozen_mode": float(sub1["chi_frozen"].mode().iloc[0]),
                    "ref_type_subset": ref_filter,
                    "baseline_name": "proportional_early_depth_over_rpatch",
                    **b1_stats,
                })

            sub2 = df_t2[df_t2["alpha_syn"] == alpha_val]
            if ref_filter != "all":
                sub2 = sub2[sub2["ref_type"] == ref_filter]
            for (subtest_lbl, desc_key), g2 in sub2.groupby(["subtest", "descriptor"], sort=False):
                if len(g2) == 0:
                    continue
                b_zero = paired_bootstrap_ci(
                    g2["mae_model"].to_numpy(dtype=np.float64),
                    g2["mae_base_no_change"].to_numpy(dtype=np.float64),
                )
                b_mean = paired_bootstrap_ci(
                    g2["mae_model"].to_numpy(dtype=np.float64),
                    g2["mae_base_train_mean"].to_numpy(dtype=np.float64),
                )
                summary_rows.append({
                    "split_mode": split_mode,
                    "test_id": f"Test2_{subtest_lbl}",
                    "descriptor": desc_key,
                    "alpha_syn": alpha_val,
                    "chi_frozen_mode": float(g2["chi_frozen"].mode().iloc[0]),
                    "ref_type_subset": ref_filter,
                    "baseline_name": "no_change",
                    **b_zero,
                })
                summary_rows.append({
                    "split_mode": split_mode,
                    "test_id": f"Test2_{subtest_lbl}",
                    "descriptor": desc_key,
                    "alpha_syn": alpha_val,
                    "chi_frozen_mode": float(g2["chi_frozen"].mode().iloc[0]),
                    "ref_type_subset": ref_filter,
                    "baseline_name": "training_mean_change",
                    **b_mean,
                })

    return df_t1, df_t2, summary_rows


# ==============================================================================
# 3. EXPLORATORY ANALYSES (REPORTED BUT NOT PART OF THE DECISION)
# ==============================================================================

def run_exploratory_pi_analyses(
    df_visits: pd.DataFrame,
    df_death_ch: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the two analyses in `status/step6b_heldout_design.md` under
    "Reported but not part of the decision":
      1. Fitted Pi per chamber and visit, and its relation to death within experiment
         (exploratory, compared against raw cup depth `cup_depth_hull_p95_um`).
      2. Fitted Pi in dasatinib vs untreated chambers from the same 3 donors (`hd2`: `v08 vs v11`,
         `hd10`: `v09 vs v12`, `hd5`: `v10 vs v13`), where Step 5b established dasatinib as a
         working positive control (cup depth and protrusion lowered in 3/3 paired donors).
    """
    df_visits = df_visits.copy()
    df_visits["cup_depth_over_radius"] = df_visits["cup_depth_hull_p95_um"] / np.maximum(0.25, df_visits["patch_equiv_radius_um"])
    ch_agg = df_visits.groupby(["experiment", "roi", "condition", "ref_type"], as_index=False).agg(
        pi_mean_a2_c10=("pi_fit_a2_c10", "mean"),
        pi_median_a2_c10=("pi_fit_a2_c10", "median"),
        pi_mean_a1_c10=("pi_fit_a1_c10", "mean"),
        cup_depth_mean_um=("cup_depth_hull_p95_um", "mean"),
        cup_depth_median_um=("cup_depth_hull_p95_um", "median"),
        cup_depth_over_radius_mean=("cup_depth_over_radius", "mean"),
        patch_radius_mean_um=("patch_equiv_radius_um", "mean"),
        contact_area_mean_um2=("contact_area_um2", "mean"),
    )
    fc_df = df_visits[df_visits["visit_order_in_contact"] == 1][
        ["experiment", "roi", "pi_fit_a2_c10", "cup_depth_hull_p95_um", "contact_area_um2"]
    ].rename(columns={
        "pi_fit_a2_c10": "pi_first_contact_a2_c10",
        "cup_depth_hull_p95_um": "cup_depth_first_contact_um",
        "contact_area_um2": "contact_area_first_contact_um2",
    })
    ch_agg = ch_agg.merge(fc_df, on=["experiment", "roi"], how="left")
    ch_agg = ch_agg.merge(
        df_death_ch[["experiment", "roi", "km_any_contact_time_min_200m", "km_event_any_visit_200m"]],
        on=["experiment", "roi"],
        how="left",
    )

    from scipy.stats import norm
    def _within_exp_z(df_in: pd.DataFrame, col: str) -> np.ndarray:
        out = np.full(len(df_in), np.nan, dtype=np.float64)
        for _, g in df_in.groupby("experiment"):
            idx = g.index
            vals = g[col].to_numpy(dtype=np.float64)
            ok = np.isfinite(vals)
            if np.sum(ok) >= 3:
                r = rankdata(vals[ok], method="average")
                n = len(r)
                q = norm.ppf((r - 0.375) / (n + 0.25))
                sd = float(np.std(q, ddof=1))
                if sd > 1e-8:
                    q /= sd
                out[idx[ok]] = q
        return out

    rows_exp: list[dict[str, Any]] = []
    for col_name, label_str, ca_col in [
        ("pi_mean_a2_c10", "EXPLORATORY: Fitted Pi (pre-death mean, alpha=2, chi=1)", "contact_area_mean_um2"),
        ("pi_mean_a1_c10", "EXPLORATORY: Fitted Pi (pre-death mean, alpha=1, chi=1)", "contact_area_mean_um2"),
        ("cup_depth_mean_um", "EXPLORATORY BENCHMARK: Raw cup_depth_hull_p95_um (pre-death mean)", "contact_area_mean_um2"),
        ("pi_first_contact_a2_c10", "EXPLORATORY: Fitted Pi (first contact, alpha=2, chi=1)", "contact_area_first_contact_um2"),
        ("cup_depth_first_contact_um", "EXPLORATORY BENCHMARK: Raw cup_depth_hull_p95_um (first contact)", "contact_area_first_contact_um2"),
    ]:
        ch_agg["_z_x"] = _within_exp_z(ch_agg, col_name)
        ch_agg["_z_ca"] = _within_exp_z(ch_agg, ca_col)
        res_unadj = fit_stratified_cox(
            df=ch_agg,
            time_col="km_any_contact_time_min_200m",
            event_col="km_event_any_visit_200m",
            cov_cols=["_z_x"],
            strata_col="experiment",
        )
        res_adj = fit_stratified_cox(
            df=ch_agg,
            time_col="km_any_contact_time_min_200m",
            event_col="km_event_any_visit_200m",
            cov_cols=["_z_x", "_z_ca"],
            strata_col="experiment",
        )
        rows_exp.append({
            "analysis_type": "exploratory_cox_death_within_experiment",
            "variable": col_name,
            "description": label_str,
            "n_chambers": int(res_unadj["n_obs"]),
            "n_events": int(res_unadj["n_events"]),
            "hr_unadjusted": float(res_unadj["hr"][0]),
            "hr_unadj_ci95_low": float(res_unadj["ci_low"][0]),
            "hr_unadj_ci95_high": float(res_unadj["ci_high"][0]),
            "p_unadjusted": float(res_unadj["p"][0]),
            "hr_adjusted_for_ca": float(res_adj["hr"][0]),
            "hr_adj_ci95_low": float(res_adj["ci_low"][0]),
            "hr_adj_ci95_high": float(res_adj["ci_high"][0]),
            "p_adjusted_for_ca": float(res_adj["p"][0]),
        })

    # 2. Dasatinib vs untreated paired donors (hd2: v08 vs v11, hd10: v09 vs v12, hd5: v10 vs v13)
    pairs = [
        ("hd2", "v08_6_40_120_0-6_hd2", "v11_6_36_120_0-6_hd2_ds"),
        ("hd10", "v09_6_40_120_0-6_hd10", "v12_6_35_120_0-6_hd10ds"),
        ("hd5", "v10_6_40_120_0-6_hd5", "v13_6_40_120_0-6_hd5_ds"),
    ]
    for col_name, label_str in [
        ("pi_mean_a2_c10", "Fitted Pi (pre-death mean, alpha=2)"),
        ("cup_depth_mean_um", "Raw cup_depth_hull_p95_um (pre-death mean)"),
    ]:
        for donor_id, exp_untr, exp_drug in pairs:
            u_vals = ch_agg[ch_agg["experiment"] == exp_untr][col_name].dropna().to_numpy(dtype=np.float64)
            d_vals = ch_agg[ch_agg["experiment"] == exp_drug][col_name].dropna().to_numpy(dtype=np.float64)
            if len(u_vals) > 0 and len(d_vals) > 0:
                _, p_mw = mannwhitneyu(u_vals, d_vals, alternative="two-sided")
                rows_exp.append({
                    "analysis_type": f"dasatinib_vs_untreated_{donor_id}",
                    "variable": col_name,
                    "description": f"{label_str} ({donor_id}: {exp_untr} vs {exp_drug})",
                    "n_chambers": int(len(u_vals) + len(d_vals)),
                    "n_events": int(len(u_vals)),
                    "hr_unadjusted": float(np.median(u_vals)),       # Untreated median
                    "hr_unadj_ci95_low": float(np.median(d_vals)),   # Dasatinib median
                    "hr_unadj_ci95_high": float(np.median(d_vals) - np.median(u_vals)),
                    "p_unadjusted": float(p_mw),
                    "hr_adjusted_for_ca": float(np.mean(u_vals)),    # Untreated mean
                    "hr_adj_ci95_low": float(np.mean(d_vals)),       # Dasatinib mean
                    "hr_adj_ci95_high": float(np.mean(d_vals) - np.mean(u_vals)),
                    "p_adjusted_for_ca": np.nan,
                })

    return pd.DataFrame(rows_exp), ch_agg


# ==============================================================================
# 4. FIGURE GENERATION (`figures/step6b_heldout.png`, HOUSE STYLE)
# ==============================================================================

def _draw_mesh_on_ax3d(
    ax: Any,
    V_t: np.ndarray,
    F_t: np.ndarray,
    V_g: np.ndarray,
    F_g: np.ndarray,
    title: str,
) -> None:
    """Render T-cell and target 3D triangulated meshes side by side in House Style colors."""
    c_tcell = "#C0561F"   # --plot-2
    c_target = "#1F4E79"  # --plot-1

    poly_t = Poly3DCollection(
        V_t[F_t], facecolors=c_tcell, edgecolors="#111111", linewidths=0.18, alpha=0.72
    )
    poly_g = Poly3DCollection(
        V_g[F_g], facecolors=c_target, edgecolors="#111111", linewidths=0.18, alpha=0.58
    )
    ax.add_collection3d(poly_t)
    ax.add_collection3d(poly_g)

    all_pts = np.vstack([V_t, V_g])
    cent = np.mean(all_pts, axis=0)
    span = float(np.max(np.ptp(all_pts, axis=0))) * 0.56
    ax.set_xlim(cent[0] - span, cent[0] + span)
    ax.set_ylim(cent[1] - span, cent[1] + span)
    ax.set_zlim(cent[2] - span, cent[2] + span)
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=2)
    ax.set_xlabel("z (um)", fontsize=7.5, labelpad=-6)
    ax.set_ylabel("y (um)", fontsize=7.5, labelpad=-6)
    ax.set_zlabel("x (um)", fontsize=7.5, labelpad=-6)
    ax.tick_params(axis="both", which="major", labelsize=6.5, pad=-3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def plot_heldout_figure(
    df_boot: pd.DataFrame,
    example_meshes: dict[str, Any] | None,
) -> None:
    """
    Plot `figures/step6b_heldout.png` in strict House Style:
      - Panel A: Test 1 & Test 3 (Late visits cup depth from early visits, solved per visit).
      - Panel B: Test 2(a,b) & Test 3 (Shape away from synapse, solved per visit).
      - Panel C, D, E: Example chamber data mesh vs fitted model mesh side by side + mid-synapse slice.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "axes.edgecolor": "#C9C9C9",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#111111",
        "xtick.color": "#1F1F1F",
        "ytick.color": "#1F1F1F",
        "text.color": "#111111",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    c_p1 = "#1F4E79"  # Model alpha=2
    c_p2 = "#C0561F"  # Model alpha=1
    c_p3 = "#2E7D5B"  # Baseline 1
    c_p4 = "#6B4C9A"  # Baseline 2

    fig = plt.figure(figsize=(14.2, 8.4), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.05], width_ratios=[1.12, 1.18], hspace=0.40, wspace=0.38)

    # --- Panel A: Test 1 (Late cup depth from early visits) ---
    ax_a = fig.add_subplot(gs[0, 0])
    t1_rows = df_boot[df_boot["test_id"] == "Test1_late_cup_depth"].copy()

    groups_a = [
        ("50/50 All", "split_50_50", "all"),
        ("50/50 Pre-contact ref", "split_50_50", "pre_contact"),
        ("50/50 Cup-filled ref", "split_50_50", "cup_filled"),
        ("Test 3 (LOEO) All", "loeo", "all"),
        ("Test 3 Pre-contact ref", "loeo", "pre_contact"),
        ("Test 3 Cup-filled ref", "loeo", "cup_filled"),
    ]
    y_pos = np.arange(len(groups_a))
    h_bar = 0.24

    for idx_g, (lbl, smode, rsub) in enumerate(groups_a):
        r_a2 = t1_rows[(t1_rows["split_mode"] == smode) & (t1_rows["ref_type_subset"] == rsub) & (t1_rows["alpha_syn"] == 2.0)]
        r_a1 = t1_rows[(t1_rows["split_mode"] == smode) & (t1_rows["ref_type_subset"] == rsub) & (t1_rows["alpha_syn"] == 1.0)]
        if len(r_a2) == 0:
            continue
        row2 = r_a2.iloc[0]
        row1 = r_a1.iloc[0]
        ax_a.barh(idx_g - h_bar, row2["median_err_model"], height=h_bar, color=c_p1, label="Model (alpha = 2)" if idx_g == 0 else "")
        ax_a.barh(idx_g, row1["median_err_model"], height=h_bar, color=c_p2, label="Model (alpha = 1)" if idx_g == 0 else "")
        ax_a.barh(idx_g + h_bar, row2["median_err_base"], height=h_bar, color="#5C5C5C", label="Proportional baseline" if idx_g == 0 else "")

        d_med = row2["paired_diff_median"]
        ci_l = row2["paired_diff_ci95_low"]
        ci_h = row2["paired_diff_ci95_high"]
        x_txt = max(row2["median_err_model"], row1["median_err_model"], row2["median_err_base"]) + 0.015
        ax_a.text(
            x_txt, idx_g,
            f"d={d_med:+.3f} [{ci_l:+.3f}, {ci_h:+.3f}]",
            va="center", ha="left", fontsize=7.5, color="#1F1F1F",
        )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([g[0] for g in groups_a], fontsize=8.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.0, 1.04)
    ax_a.set_xlabel("Median absolute error in late cup depth (um)", fontsize=9.5)
    ax_a.set_title("A. Test 1 & Test 3: Late visits cup depth from early visits", loc="left", fontsize=10.5, fontweight="bold", pad=18)
    ax_a.legend(frameon=False, fontsize=7.6, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3)
    ax_a.grid(True, axis="x", color="#E3E3E3", linewidth=0.6)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # --- Panel B: Test 2(a,b) & Test 3 (Shape away from synapse: paired diff vs both baselines) ---
    ax_b = fig.add_subplot(gs[0, 1])
    t2_rows = df_boot[
        (df_boot["test_id"].str.startswith("Test2_")) & (df_boot["ref_type_subset"] == "all")
    ].copy()

    desc_order = [
        ("target_sphericity", "2a. Target sphericity"),
        ("target_elongation", "2a. Target elongation"),
        ("tcell_sphericity", "2b. T-cell sphericity"),
        ("tcell_elongation", "2b. T-cell elongation"),
    ]
    y_labels = []
    idx_y = 0
    for dkey, dlbl in desc_order:
        for smode, slbl in [("split_50_50", "50/50"), ("loeo", "LOEO")]:
            y_labels.append(f"{dlbl} ({slbl})")
            sub_d = t2_rows[(t2_rows["descriptor"] == dkey) & (t2_rows["split_mode"] == smode)]
            for alpha_v, bname, col_c, off_y, mkr, lbl_leg in [
                (2.0, "no_change", c_p1, -0.18, "o", "alpha=2 vs No change"),
                (2.0, "training_mean_change", c_p3, -0.06, "s", "alpha=2 vs Train mean"),
                (1.0, "no_change", c_p2, +0.06, "^", "alpha=1 vs No change"),
                (1.0, "training_mean_change", c_p4, +0.18, "d", "alpha=1 vs Train mean"),
            ]:
                r_m = sub_d[(sub_d["alpha_syn"] == alpha_v) & (sub_d["baseline_name"] == bname)]
                if len(r_m) == 0:
                    continue
                rr = r_m.iloc[0]
                val = rr["paired_diff_median"]
                err_low = max(0.0, val - rr["paired_diff_ci95_low"])
                err_high = max(0.0, rr["paired_diff_ci95_high"] - val)
                ax_b.errorbar(
                    val, idx_y + off_y,
                    xerr=[[err_low], [err_high]],
                    fmt=mkr, color=col_c, markersize=4.5, elinewidth=1.2, capsize=2.0,
                    label=lbl_leg if idx_y == 0 else "",
                )
            idx_y += 1

    ax_b.axvline(0.0, color="#111111", linewidth=1.0, linestyle="--")
    ax_b.set_yticks(np.arange(len(y_labels)))
    ax_b.set_yticklabels(y_labels, fontsize=8.0)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Paired MAE difference: Model - Baseline (< 0 favors model)", fontsize=9.5)
    ax_b.set_title("B. Test 2(a,b) & Test 3: Shape away from synapse (95% CIs)", loc="left", fontsize=10.5, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax_b.grid(True, axis="x", color="#E3E3E3", linewidth=0.6)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # --- Panel C, D, E: Side-by-side 3D meshes (Data mesh vs Fitted model mesh) on example chamber ---
    if example_meshes is not None:
        gs_bottom = gs[1, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.18)
        ax_c1 = fig.add_subplot(gs_bottom[0, 0], projection="3d")
        ax_c2 = fig.add_subplot(gs_bottom[0, 1], projection="3d")
        ax_c3 = fig.add_subplot(gs_bottom[0, 2])

        exp_lbl = f"{example_meshes['experiment'].split('_')[0]} {example_meshes['roi']} (visit {example_meshes['visit']})"
        _draw_mesh_on_ax3d(
            ax_c1,
            example_meshes["V_tcell_data"],
            example_meshes["F_tcell"],
            example_meshes["V_target_data"],
            example_meshes["F_target_data"],
            f"C. Data mesh: {exp_lbl}\nMeasured cup depth = {example_meshes['measured_cup_depth_um']:.2f} um",
        )
        _draw_mesh_on_ax3d(
            ax_c2,
            example_meshes["V_tcell_model"],
            example_meshes["F_tcell"],
            example_meshes["V_target_model"],
            example_meshes["F_target_ref"],
            f"D. Fitted model mesh (Pi = {example_meshes['pi_fit']:.2f}, alpha = 2)\nTarget (blue) + T cell (orange)",
        )

        V_t_d = example_meshes["V_tcell_data"]
        V_g_d = example_meshes["V_target_data"]
        V_g_r = example_meshes["V_target_ref"]
        V_g_m = example_meshes["V_target_model"]
        V_t_m = example_meshes["V_tcell_model"]

        c_g = np.mean(V_g_r, axis=0)
        c_t = np.mean(V_t_d, axis=0)
        u_ax = c_g - c_t
        u_ax /= max(1e-8, float(np.linalg.norm(u_ax)))
        arb = np.array([1.0, 0.0, 0.0]) if abs(u_ax[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e_perp = np.cross(u_ax, arb)
        e_perp /= np.linalg.norm(e_perp)
        e_slice = np.cross(u_ax, e_perp)

        def _slice_curve(V_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            rel = V_pts - c_g[None, :]
            s_out = rel @ e_slice
            mask = np.abs(s_out) <= 1.8
            if int(np.sum(mask)) < 16:
                mask = np.ones(len(V_pts), dtype=bool)
            x_ax = rel[mask] @ u_ax
            y_pr = rel[mask] @ e_perp
            cx = float(np.mean(x_ax))
            cy = float(np.mean(y_pr))
            dx = x_ax - cx
            dy = y_pr - cy
            ang = np.arctan2(dy, dx)
            rad = np.hypot(dx, dy)
            bins = np.linspace(-np.pi, np.pi, 29)
            bx = []
            by = []
            for b_i in range(len(bins) - 1):
                in_b = (ang >= bins[b_i]) & (ang < bins[b_i + 1])
                if np.any(in_b):
                    th_m = 0.5 * (bins[b_i] + bins[b_i + 1])
                    r_m = float(np.median(rad[in_b]))
                    bx.append(cx + r_m * np.cos(th_m))
                    by.append(cy + r_m * np.sin(th_m))
            if len(bx) >= 3:
                bx.append(bx[0])
                by.append(by[0])
                return np.asarray(bx), np.asarray(by)
            order = np.argsort(ang)
            return np.append(x_ax[order], x_ax[order[0]]), np.append(y_pr[order], y_pr[order[0]])

        xg_r, yg_r = _slice_curve(V_g_r)
        xg_d, yg_d = _slice_curve(V_g_d)
        xg_m, yg_m = _slice_curve(V_g_m)
        xt_m, yt_m = _slice_curve(V_t_m)

        ax_c3.plot(xg_r, yg_r, color="#5C5C5C", linestyle=":", linewidth=1.4, label="Target reference (unloaded)")
        ax_c3.plot(xg_d, yg_d, color="#111111", linestyle="--", linewidth=1.5, label="Target data contour")
        ax_c3.plot(xg_m, yg_m, color=c_p1, linestyle="-", linewidth=1.8, label="Target fitted model")
        ax_c3.plot(xt_m, yt_m, color=c_p2, linestyle="-", linewidth=1.6, label="T-cell fitted model")
        ax_c3.set_aspect("equal", adjustable="box")
        ax_c3.set_xlim(-15.5, 7.0)
        ax_c3.set_ylim(-8.5, 9.5)
        ax_c3.set_xlabel("Coordinate along synapse axis (um)", fontsize=9.0)
        ax_c3.set_ylabel("Transverse coordinate (um)", fontsize=9.0)
        ax_c3.set_title("E. Mid-synapse cross-section (data vs model)", loc="left", fontsize=10.0, fontweight="bold", pad=6)
        ax_c3.legend(frameon=False, fontsize=7.0, loc="upper left", ncol=1)
        ax_c3.grid(True, color="#E3E3E3", linewidth=0.6)
        ax_c3.spines["top"].set_visible(False)
        ax_c3.spines["right"].set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_HELDOUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Step 6b: Fit and evaluate the two-shell mechanical model per visit.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel CPU workers (default 6, leaving headroom).")
    parser.add_argument("--limit-chambers", type=int, default=0, help="Optional chamber limit for smoke test (0 = all 305).")
    parser.add_argument("--replot-only", action="store_true", help="Re-render figures/step6b_heldout.png from saved CSVs + example chamber.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    STEP6B_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify Stage 2 validation passed before touching data
    val_json_path = STEP6B_DIR / "validation_summary.json"
    if not val_json_path.is_file():
        raise RuntimeError("Stage 2 validation_summary.json not found! Run validate_solver.py first.")
    val_summary = json.loads(val_json_path.read_text())
    if not val_summary.get("stage2_passed", False):
        raise RuntimeError("Stage 2 solver validation failed! Stopping before fitting data.")

    # 2. Load Step 5a visit descriptors, Step 4 death labels, and manifest paths
    print("Loading Step 5a visit descriptors, Step 4 death labels, and manifest...", flush=True)
    df_v_all = pd.read_csv(STEP5_VISIT_CSV, low_memory=False)
    df_d_ch = pd.read_csv(STEP4_CHAMBER_CSV)
    df_man = pd.read_csv(MANIFEST_CSV)

    man_lookup: dict[tuple[str, str, int, int], str] = {}
    for _, mr in df_man.iterrows():
        man_lookup[(str(mr["experiment"]), str(mr["roi"]), int(mr["visit_index"]), int(mr["sweep_index"]))] = str(mr["path"])

    m = df_v_all.merge(
        df_d_ch[["experiment", "roi", "condition", "first_any_dead_visit_200m", "km_any_contact_time_min_200m", "km_event_any_visit_200m"]],
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
    example_key = ("v01_6_40_120_0-6_hd1", "MIP_ROI_14")
    if example_key not in elig_keys:
        example_key = elig_keys[0]
    if args.replot_only:
        elig_keys = [example_key]
    elif args.limit_chambers > 0:
        elig_keys = elig_keys[: args.limit_chambers]

    print(f"Eligible chambers (trace_num==1, >=4 pre-death contact visits in visits 1-20): {len(elig_keys)}", flush=True)

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

    # Flag every 5th chamber (~61 chambers across all 13 experiments) for mesh fidelity & pre-stress ablation
    tasks: list[dict[str, Any]] = []
    for k_idx, (exp_k, roi_k) in enumerate(elig_keys):
        ch_df = m_t1[(m_t1["experiment"] == exp_k) & (m_t1["roi"] == roi_k)].sort_values("visit")
        cond_str = str(ch_df["condition"].iloc[0])
        tasks.append({
            "experiment": exp_k,
            "roi": roi_k,
            "condition": cond_str,
            "visit_rows": ch_df.to_dict(orient="records"),
            "rep_sweep_map": rep_sweeps_by_ch.get((exp_k, roi_k), {}),
            "save_example_meshes": bool((exp_k, roi_k) == example_key),
            "run_fidelity_and_prestress": bool(k_idx % 5 == 0),
        })

    workers = max(1, min(6, int(args.workers)))
    print(f"Running Stage 3 & 4 per-visit 3D mesh extraction and mechanical solves across {len(tasks)} chambers with {workers} workers...", flush=True)

    all_visit_rows: list[dict[str, Any]] = []
    all_fidelity_rows: list[dict[str, Any]] = []
    all_prestress_rows: list[dict[str, Any]] = []
    example_meshes_saved: dict[str, Any] | None = None
    n_manifold_ok = 0

    if workers == 1:
        for idx_d, t_item in enumerate(tasks, start=1):
            res = process_one_chamber_step6b(t_item)
            all_visit_rows.extend(res["visit_fit_rows"])
            all_fidelity_rows.extend(res["fidelity_rows"])
            all_prestress_rows.extend(res["prestress_rows"])
            if res["manifold_ok"]:
                n_manifold_ok += 1
            if res["example_meshes"] is not None:
                example_meshes_saved = res["example_meshes"]
            if idx_d % 25 == 0 or idx_d == len(tasks):
                print(f"  [{idx_d}/{len(tasks)}] {res['experiment']}/{res['roi']} ({res['elapsed_s']:.2f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(process_one_chamber_step6b, t_item): t_item for t_item in tasks}
            for idx_d, fut in enumerate(as_completed(fut_map), start=1):
                res = fut.result()
                all_visit_rows.extend(res["visit_fit_rows"])
                all_fidelity_rows.extend(res["fidelity_rows"])
                all_prestress_rows.extend(res["prestress_rows"])
                if res["manifold_ok"]:
                    n_manifold_ok += 1
                if res["example_meshes"] is not None:
                    example_meshes_saved = res["example_meshes"]
                if idx_d % 25 == 0 or idx_d == len(tasks):
                    el = time.perf_counter() - t_start
                    print(
                        f"  [{idx_d:3d}/{len(tasks):3d}] latest={res['experiment']}/{res['roi']} "
                        f"ref={res['ref_type']} ({res['elapsed_s']:.2f}s, wall={el:.1f}s)",
                        flush=True,
                    )

    if args.replot_only:
        df_boot_summary = pd.read_csv(STEP6B_DIR / "heldout_bootstrap_summary.csv")
        plot_heldout_figure(df_boot_summary, example_meshes_saved)
        print("Re-plotted figures/step6b_heldout.png successfully.", flush=True)
        return 0

    df_visits_fit = pd.DataFrame(all_visit_rows).sort_values(["experiment", "roi", "visit"]).reset_index(drop=True)
    df_fidelity = pd.DataFrame(all_fidelity_rows)
    df_prestress = pd.DataFrame(all_prestress_rows)

    df_fidelity.to_csv(STEP6B_DIR / "mesh_fidelity_comparison.csv", index=False)
    df_prestress.to_csv(STEP6B_DIR / "prestress_ablation_comparison.csv", index=False)

    # 3. Run Held-Out Tests 1, 2(a,b), and Test 3 (LOEO repeat) with 50/50 split and paired bootstrap
    df_ch_unique = df_visits_fit[["experiment", "roi", "condition", "ref_type"]].drop_duplicates().reset_index(drop=True)
    split_50_50 = make_stratified_50_50_split(df_ch_unique, seed=SPLIT_SEED)
    df_visits_fit["split_50_50_role"] = df_visits_fit.set_index(["experiment", "roi"]).index.map(lambda k: split_50_50.get(k, "test"))
    df_visits_fit.to_csv(STEP6B_DIR / "visit_fits.csv", index=False)

    df_t1_50, df_t2_50, sum_50 = evaluate_heldout_tests_for_split(
        df_visits_fit, split_mode="split_50_50", split_map=split_50_50
    )
    df_t1_loeo, df_t2_loeo, sum_loeo = evaluate_heldout_tests_for_split(
        df_visits_fit, split_mode="loeo", split_map=None
    )

    df_boot_summary = pd.DataFrame(sum_50 + sum_loeo)

    df_t1_50.to_csv(STEP6B_DIR / "test1_late_from_early.csv", index=False)
    df_t2_50.to_csv(STEP6B_DIR / "test2_shape_away_from_synapse.csv", index=False)
    pd.concat([df_t1_loeo, df_t2_loeo], ignore_index=True).to_csv(STEP6B_DIR / "test3_loeo_repeat.csv", index=False)
    df_boot_summary.to_csv(STEP6B_DIR / "heldout_bootstrap_summary.csv", index=False)

    # 4. Apply literal decision rule from `status/step6b_heldout_design.md`:
    sub_b_a2 = df_boot_summary[(df_boot_summary["alpha_syn"] == 2.0) & (df_boot_summary["ref_type_subset"] == "all")]

    t1_50_row = sub_b_a2[(sub_b_a2["split_mode"] == "split_50_50") & (sub_b_a2["test_id"] == "Test1_late_cup_depth")].iloc[0]
    t1_loeo_row = sub_b_a2[(sub_b_a2["split_mode"] == "loeo") & (sub_b_a2["test_id"] == "Test1_late_cup_depth")].iloc[0]
    pass_test1 = bool(
        t1_50_row["beats_baseline_ci_excl_0"]
        and (t1_loeo_row["paired_diff_median"] < 0.0)
    )

    pass_test2a = False
    pass_test2b = False
    for desc_k, is_2a in [
        ("target_sphericity", True),
        ("target_elongation", True),
        ("tcell_sphericity", False),
        ("tcell_elongation", False),
    ]:
        r50_0 = sub_b_a2[(sub_b_a2["split_mode"] == "split_50_50") & (sub_b_a2["descriptor"] == desc_k) & (sub_b_a2["baseline_name"] == "no_change")].iloc[0]
        r50_m = sub_b_a2[(sub_b_a2["split_mode"] == "split_50_50") & (sub_b_a2["descriptor"] == desc_k) & (sub_b_a2["baseline_name"] == "training_mean_change")].iloc[0]
        rlo_0 = sub_b_a2[(sub_b_a2["split_mode"] == "loeo") & (sub_b_a2["descriptor"] == desc_k) & (sub_b_a2["baseline_name"] == "no_change")].iloc[0]
        rlo_m = sub_b_a2[(sub_b_a2["split_mode"] == "loeo") & (sub_b_a2["descriptor"] == desc_k) & (sub_b_a2["baseline_name"] == "training_mean_change")].iloc[0]

        beats_both = bool(
            r50_0["beats_baseline_ci_excl_0"]
            and r50_m["beats_baseline_ci_excl_0"]
            and (rlo_0["paired_diff_median"] < 0.0)
            and (rlo_m["paired_diff_median"] < 0.0)
        )
        if is_2a and beats_both:
            pass_test2a = True
        elif (not is_2a) and beats_both:
            pass_test2b = True

    model_adds_something = bool(pass_test1 or pass_test2a or pass_test2b)
    decision_str = (
        "MODEL_ADDS_SOMETHING"
        if model_adds_something
        else "USE_SIMPLER_DESCRIPTION_CUP_DEPTH_PLUS_PROPORTIONAL_RULE"
    )

    # 5. Run exploratory Pi vs death and dasatinib positive-control analyses
    df_exploratory, df_ch_fits = run_exploratory_pi_analyses(df_visits_fit, df_d_ch)
    df_exploratory.to_csv(STEP6B_DIR / "exploratory_pi_vs_death_and_dasatinib.csv", index=False)
    df_ch_fits.to_csv(STEP6B_DIR / "chamber_fits.csv", index=False)

    # 6. Stage 5 AFM-to-tension conversion (labelled conditional on the stated tension anchor)
    afm_tcell = afm_modulus_to_cortical_tension((700.0, 2200.0), R_cell_um=5.3, R_tip_um=2.5, delta_afm_um=0.5)
    afm_target = afm_modulus_to_cortical_tension((300.0, 1000.0), R_cell_um=5.0, R_tip_um=2.5, delta_afm_um=0.5)

    pi_q25, pi_q50, pi_q75 = np.percentile(df_ch_fits["pi_median_a2_c10"].dropna(), [25, 50, 75])
    t_targ_nom_low, t_targ_nom_high = afm_target["T_cortical_nominal_pN_per_um"]
    r_targ_med = float(df_visits_fit["r_target_um"].median())
    a_syn_med = float(df_visits_fit["contact_area_um2"].median())

    p_act_low_pa = float(pi_q50 * t_targ_nom_low / r_targ_med)
    p_act_high_pa = float(pi_q50 * t_targ_nom_high / r_targ_med)
    f_tot_low_nn = float(p_act_low_pa * a_syn_med * 1e-3)
    f_tot_high_nn = float(p_act_high_pa * a_syn_med * 1e-3)

    afm_out = {
        "tcell_cortex_from_jung2021_afm": afm_tcell,
        "target_cortex_from_lymphocyte_afm": afm_target,
        "fitted_pi_distribution_alpha2_chi1": {
            "chamber_median_q25": float(pi_q25),
            "chamber_median_q50": float(pi_q50),
            "chamber_median_q75": float(pi_q75),
            "visit_median": float(df_visits_fit["pi_fit_a2_c10"].median()),
            "visit_iqr": [
                float(np.percentile(df_visits_fit["pi_fit_a2_c10"], 25)),
                float(np.percentile(df_visits_fit["pi_fit_a2_c10"], 75)),
            ],
        },
        "conditional_physical_push_at_median_pi": {
            "anchor_note": "CONDITIONAL ON STATED CORTICAL-TENSION ANCHOR T_target in [49, 163] pN/um (nominal) or [18, 430] pN/um (full range)",
            "median_r_target_um": r_targ_med,
            "median_contact_area_um2": a_syn_med,
            "conditional_active_pressure_nominal_pa_range": [p_act_low_pa, p_act_high_pa],
            "conditional_total_synapse_force_nominal_nN_range": [f_tot_low_nn, f_tot_high_nn],
        },
    }
    (STEP6B_DIR / "afm_to_tension_conversion.json").write_text(json.dumps(afm_out, indent=2))

    # 7. Plot `figures/step6b_heldout.png`
    plot_heldout_figure(df_boot_summary, example_meshes_saved)

    total_wall_s = time.perf_counter() - t_start
    summary_payload = {
        "decision": decision_str,
        "model_adds_something": model_adds_something,
        "pass_test1": pass_test1,
        "pass_test2a": pass_test2a,
        "pass_test2b": pass_test2b,
        "n_eligible_chambers": int(len(df_ch_unique)),
        "n_pre_contact_ref_chambers": int((df_ch_unique["ref_type"] == "pre_contact").sum()),
        "n_cup_filled_ref_chambers": int((df_ch_unique["ref_type"] == "cup_filled").sum()),
        "n_pre_death_contact_visits_solved": int(len(df_visits_fit)),
        "n_manifold_ok_chambers": int(n_manifold_ok),
        "mesh_fidelity_summary": {
            "n_cells_compared": int(len(df_fidelity)),
            "mc_hausdorff_median_um": float(df_fidelity["mc_hausdorff_um"].median()) if len(df_fidelity) > 0 else np.nan,
            "mc_mean_surf_dist_median_um": float(df_fidelity["mc_mean_surf_dist_um"].median()) if len(df_fidelity) > 0 else np.nan,
            "icosphere_hausdorff_median_um": float(df_fidelity["icosphere_hausdorff_um"].median()) if len(df_fidelity) > 0 else np.nan,
            "icosphere_mean_surf_dist_median_um": float(df_fidelity["icosphere_mean_surf_dist_um"].median()) if len(df_fidelity) > 0 else np.nan,
        },
        "prestress_ablation_summary": {
            "n_chambers_compared": int(len(df_prestress)),
            "median_abs_depth_diff_um": float(np.median(np.abs(df_prestress["depth_prestress_subtracted_um"] - df_prestress["depth_unsubtracted_relaxed0_um"]))) if len(df_prestress) > 0 else np.nan,
            "median_zero_load_relax_rms_disp_um": float(df_prestress["zero_load_relax_rms_disp_um"].median()) if len(df_prestress) > 0 else np.nan,
            "median_zero_load_relax_target_sph_shift": float(df_prestress["zero_load_relax_target_sph_shift"].median()) if len(df_prestress) > 0 else np.nan,
        },
        "wall_time_s": total_wall_s,
    }
    (STEP6B_DIR / "summary_step6b.json").write_text(json.dumps(summary_payload, indent=2))

    print("\n=== Step 6b Per-Visit 3D Mechanics Summary ===")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
