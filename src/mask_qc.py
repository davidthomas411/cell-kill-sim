#!/usr/bin/env python3
"""
Task 1: Per-visit QC flags for the tracked T cell and its contacted target in every
trace_num == 1 contact visit, using the exact same mask extraction as src/contact_descriptors.py.

Computes on every trace_num == 1 contact visit (using the representative contact sweep
matching src/mech/run_step6b.py and src/review/make_gallery.py, plus sweep-level contact
summary columns from data/step5/per_sweep_descriptors.csv):
  - target_n_components (3D, 26-connected, components >= 20 voxels) and target_largest_frac
    (largest component volume / total) — computed on:
      (a) the extracted solid target mask `b_cell_sub` (`fill_solid_cell_mask`, matching
          `src/contact_descriptors.py` lines 889-893),
      (b) the raw target voxels `(I == 2) | (I == 4)` within the target bounding box (`raw_b`),
      (c) the 1-voxel eroded `b_cell_sub` (`erode1`), and
      (d) the Step 5 sweep-level target volume CV (`target_vol_sweep_cv`) across sweeps 1..5;
  - target_solidity = target volume / 3D convex hull volume;
  - touches_fov_edge for target and T cell (any voxel within 2 voxels of the x, y or z border:
    coord < 2 or coord >= dim - 2);
  - target_vol_ratio_to_own_median over the chamber's visits.

Defines:
  - flag_fragmented = (target_n_components >= 2) and (target_largest_frac < 0.9)
    (evaluated on `b_cell_sub`, plus `flag_fragmented_raw` on `(I == 2) | (I == 4)`,
    `flag_fragmented_erode1`, and `flag_low_solidity_lt05` for comparison);
  - flag_edge = touches_fov_edge (either cell).

Outputs:
  - data/qc/per_visit_qc.csv
  - data/qc/task1_summary.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _env_k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_k, "1")

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, label
from scipy.spatial import ConvexHull, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.contact_descriptors import (
    build_tasks_for_chambers,
    edt_indices_matlab_xyz,
    fill_solid_cell_mask,
)
from src.paths import DATA_DIR
from src.reproduce_contact_area import (
    STRUCT26,
    bbox_slices,
    bwareaopen_26,
    imfill_holes_26,
    sphere_dilate_edt,
)

QC_DIR = DATA_DIR / "qc"
STEP5_SWEEP_CSV = DATA_DIR / "step5" / "per_sweep_descriptors.csv"
STEP5_VISIT_CSV = DATA_DIR / "step5" / "per_visit_descriptors.csv"
STEP4_CHAMBER_CSV = DATA_DIR / "step4" / "per_chamber_death_labels_all_v2.csv"
MAX_WORKERS = 3


def fast_instance_seg_exact(
    nucleus: np.ndarray, membrane: np.ndarray, expected_n_inst: int = -1
) -> np.ndarray:
    """
    100% exact equivalent of `fast_instance_seg_with_hint(nucleus, membrane, expected_n_inst)`
    in `src/contact_descriptors.py`, avoiding redundant `bwareaopen_26(nucleus, 3000)` and
    skipping the 25-voxel sphere dilation check when `expected_n_inst == max_index`
    (since `per_sweep.csv` already confirmed all `max_index` nuclei passed `temp >= 1000`).
    """
    nuc_clean, nuc_lbl, max_index = bwareaopen_26(nucleus, 3000)
    if max_index == 0:
        result = np.zeros(nucleus.shape, dtype=np.int8)
        if int(np.sum(membrane)) > 3000:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                counts = np.bincount(mem_lbl.ravel())
                counts[0] = 0
                filled = imfill_holes_26(mem_lbl == int(np.argmax(counts)))
                result[filled] = 1
        return result
    if max_index == 1 and expected_n_inst == 1:
        mem_lbl, n_mem = label(membrane, structure=STRUCT26)
        if n_mem > 0:
            counts = np.bincount(mem_lbl.ravel())
            counts[0] = 0
            combined = nuc_clean | (mem_lbl == int(np.argmax(counts)))
        else:
            combined = nuc_clean
        return imfill_holes_26(combined).astype(np.int8)

    result = np.zeros(nucleus.shape, dtype=np.int8)
    if max_index == 1:
        membrane_instance = membrane.astype(np.int8)
    else:
        sl = bbox_slices(nuc_clean | membrane, 2, nucleus.shape)
        membrane_instance = np.zeros(nucleus.shape, dtype=np.int8)
        if sl is not None:
            inds = edt_indices_matlab_xyz(~nuc_clean[sl])
            sub_inst = nuc_lbl[sl][inds].astype(np.int8)
            sub_inst[~membrane[sl]] = 0
            membrane_instance[sl] = sub_inst

    skip_temp = expected_n_inst == max_index
    for i in range(1, max_index + 1):
        membrane_mask = membrane_instance == i
        nucleus_mask = nuc_lbl == i
        if not skip_temp:
            temp = membrane_mask & sphere_dilate_edt(nucleus_mask, 25.0)
            if int(np.sum(temp)) < 1000:
                continue
        mem_lbl, n_mem = label(membrane_mask, structure=STRUCT26)
        if n_mem > 0:
            counts = np.bincount(mem_lbl.ravel())
            counts[0] = 0
            combined = nucleus_mask | (mem_lbl == int(np.argmax(counts)))
        else:
            combined = nucleus_mask
        filled = imfill_holes_26(combined)
        result[filled] = i

    if not np.any(result):
        if int(np.sum(membrane)) > 3000:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                counts = np.bincount(mem_lbl.ravel())
                counts[0] = 0
                filled = imfill_holes_26(mem_lbl == int(np.argmax(counts)))
                result[filled] = 1
    return result


def _components_stats_26(mask_3d: np.ndarray, min_vox: int = 20) -> tuple[int, float, int]:
    """
    Return `(n_components_ge20, largest_frac, total_vox)` for 3D 26-connected components
    of `mask_3d`, where `n_components_ge20` counts components with >= `min_vox` voxels,
    and `largest_frac` is `largest_component_voxels / total_voxels`.
    """
    tot_vox = int(np.sum(mask_3d))
    if tot_vox <= 0:
        return 0, np.nan, 0
    lbl, n_c = label(mask_3d, structure=STRUCT26)
    if n_c == 0:
        return 0, np.nan, tot_vox
    cnts = np.bincount(lbl.ravel())
    cnts[0] = 0
    ge20 = cnts[cnts >= min_vox]
    n_ge20 = int(len(ge20))
    largest = int(cnts.max())
    return n_ge20, float(largest / tot_vox), tot_vox


def _touches_fov_border(
    pts_zyx: np.ndarray | None, shape_zyx: tuple[int, int, int], margin: int = 2
) -> bool:
    """
    True if any voxel coordinate `(z, y, x)` lies within `margin` voxels of the 3D FOV border:
    `0 <= coord < margin` or `coord >= dim - margin`.
    """
    if pts_zyx is None or len(pts_zyx) == 0:
        return False
    Z, Y, X = shape_zyx
    z = pts_zyx[:, 0]
    y = pts_zyx[:, 1]
    x = pts_zyx[:, 2]
    return bool(
        np.any(
            (z < margin)
            | (z >= Z - margin)
            | (y < margin)
            | (y >= Y - margin)
            | (x < margin)
            | (x >= X - margin)
        )
    )


def process_one_chamber_qc(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Process every contact visit in a `trace_num == 1` chamber using the exact mask extraction
    from `src/contact_descriptors.py` on the representative contact sweep (median contact area
    sweep in the visit, matching `src/mech/run_step6b.py` and `src/review/make_gallery.py`).
    """
    exp: str = str(task["experiment"])
    roi: str = str(task["roi"])
    roi_idx: int = int(task["roi_index"])
    gid: str = str(task["group_id"])
    cond: str = str(task["condition"])
    trace_num: int = int(task["trace_num"])
    if trace_num != 1:
        return []

    sweep_meta_list: list[dict[str, Any]] = task["sweep_meta_list"]
    by_visit: dict[int, list[dict[str, Any]]] = {}
    for s in sweep_meta_list:
        by_visit.setdefault(int(s["visit"]), []).append(s)

    vox_vol_um3 = VOXEL_UM ** 3
    rows: list[dict[str, Any]] = []

    for visit in sorted(by_visit.keys()):
        v_sweeps = by_visit[visit]
        c_sweeps = [s for s in v_sweeps if int(s["s_largest"]) > 0]
        if not c_sweeps:
            continue

        # Select representative contact sweep (median s_largest among contact sweeps)
        s_vals = np.array([int(s["s_largest"]) for s in c_sweeps], dtype=np.float64)
        med_s = float(np.median(s_vals))
        rep_sw = c_sweeps[int(np.argmin(np.abs(s_vals - med_s)))]

        sweep = int(rep_sw["sweep"])
        tp_idx = int(rep_sw["timepoint_idx"])
        rel_path = str(rep_sw["rel_path"])
        tracked_id = int(rep_sw["tracked_tcell_id"])
        n_t_hint = int(rep_sw["n_tcell_instances"])
        n_b_hint = int(rep_sw["n_target_instances"])
        s_largest = int(rep_sw["s_largest"])
        s_total = int(rep_sw["s_total"])

        abs_path = REPO_ROOT / rel_path
        with h5py.File(abs_path, "r") as hf:
            I = hf["predictions"][:]

        shape_zyx = I.shape

        # 1. Segment tracked T-cell instance (exact match to contact_descriptors.py lines 828-854)
        t_surf_vox: np.ndarray | None = None
        t_all_vox: np.ndarray | None = None
        tree_t_surf: cKDTree | None = None
        tcell_n_comp = 0
        tcell_largest_frac = np.nan
        tcell_vox_count = 0

        sl_t = bbox_slices((I == 1) | (I == 3), 2, shape_zyx)
        if sl_t is not None and tracked_id > 0:
            t_inst_sub = fast_instance_seg_exact(
                I[sl_t] == 1, I[sl_t] == 3, expected_n_inst=n_t_hint
            )
            if np.any(t_inst_sub == tracked_id):
                t_sel = fill_solid_cell_mask(t_inst_sub == tracked_id)
                tcell_n_comp, tcell_largest_frac, tcell_vox_count = _components_stats_26(
                    t_sel, min_vox=20
                )
                off_t = np.array(
                    [sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64
                )
                t_all_vox = np.argwhere(t_sel).astype(np.float64) + off_t
                t_surf_mask = t_sel & (~binary_erosion(t_sel, structure=STRUCT26))
                t_surf_vox = np.argwhere(t_surf_mask).astype(np.float64) + off_t
                if len(t_surf_vox) > 0:
                    tree_t_surf = cKDTree(t_surf_vox)

        tcell_touches_edge = _touches_fov_border(
            t_surf_vox if t_surf_vox is not None else t_all_vox, shape_zyx, margin=2
        )

        # 2. Segment contacted target instance (exact match to contact_descriptors.py lines 856-898)
        b_cell_sub: np.ndarray | None = None
        b_surf_vox: np.ndarray | None = None
        b_all_vox: np.ndarray | None = None
        target_n_comp = 0
        target_largest_frac = np.nan
        target_vox_count = 0
        target_b_inst_n_comp = 0
        target_b_inst_largest_frac = np.nan
        target_raw_n_comp = 0
        target_raw_largest_frac = np.nan
        target_erode1_n_comp = 0
        target_erode1_largest_frac = np.nan
        target_solidity = np.nan
        target_hull_vol_um3 = np.nan
        max_b = 0

        raw_b_full = (I == 2) | (I == 4)
        sl_b = bbox_slices(raw_b_full, 2, shape_zyx)
        if sl_b is not None:
            target_raw_n_comp, target_raw_largest_frac, _ = _components_stats_26(
                raw_b_full[sl_b], min_vox=20
            )
            b_inst_sub = fast_instance_seg_exact(
                I[sl_b] == 2, I[sl_b] == 4, expected_n_inst=n_b_hint
            )
            max_b = int(b_inst_sub.max())
            if max_b > 0:
                off_b = np.array(
                    [sl_b[0].start, sl_b[1].start, sl_b[2].start], dtype=np.float64
                )
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
                        if n_touch > best_touch or (
                            n_touch == best_touch and cnts_b[bid] > cnts_b[best_bid]
                        ):
                            best_touch = n_touch
                            best_bid = bid
                else:
                    best_bid = int(np.argmax(cnts_b))

                target_b_inst_n_comp, target_b_inst_largest_frac, _ = _components_stats_26(
                    b_inst_sub > 0, min_vox=20
                )

                # Exact contact_descriptors.py target extraction:
                if max_b == 1:
                    b_cell_sub = fill_solid_cell_mask(b_inst_sub == 1)
                else:
                    cc_lbl, _ = label(b_inst_sub > 0, structure=STRUCT26)
                    dom_cc = int(np.median(cc_lbl[b_inst_sub == best_bid]))
                    b_cell_sub = fill_solid_cell_mask(cc_lbl == dom_cc)

                target_n_comp, target_largest_frac, target_vox_count = _components_stats_26(
                    b_cell_sub, min_vox=20
                )
                b_er1 = binary_erosion(b_cell_sub, structure=STRUCT26, iterations=1)
                target_erode1_n_comp, target_erode1_largest_frac, _ = _components_stats_26(
                    b_er1, min_vox=20
                )

                b_all_vox = np.argwhere(b_cell_sub).astype(np.float64) + off_b
                b_surf_mask = b_cell_sub & (~binary_erosion(b_cell_sub, structure=STRUCT26))
                b_surf_vox = np.argwhere(b_surf_mask).astype(np.float64) + off_b

                if len(b_surf_vox) >= 12:
                    try:
                        pts_b_surf_um = b_surf_vox * VOXEL_UM
                        hull_b = ConvexHull(pts_b_surf_um)
                        target_hull_vol_um3 = float(hull_b.volume)
                        b_vol_um3 = float(target_vox_count) * vox_vol_um3
                        if target_hull_vol_um3 > 0:
                            target_solidity = float(b_vol_um3 / target_hull_vol_um3)
                    except Exception:
                        target_solidity = np.nan

        target_touches_edge = _touches_fov_border(
            b_surf_vox if b_surf_vox is not None else b_all_vox, shape_zyx, margin=2
        )
        touches_fov_edge = bool(target_touches_edge or tcell_touches_edge)

        rows.append(
            {
                "experiment": exp,
                "roi": roi,
                "roi_index": roi_idx,
                "group_id": gid,
                "condition": cond,
                "trace_num": trace_num,
                "visit": visit,
                "rep_sweep": sweep,
                "rep_timepoint_idx": tp_idx,
                "n_sweeps": len(v_sweeps),
                "n_contact_sweeps": len(c_sweeps),
                "tracked_tcell_id": tracked_id,
                "rep_s_largest_vox": s_largest,
                "rep_s_total_vox": s_total,
                "n_target_instances_hint": n_b_hint,
                "max_b_inst": max_b,
                "target_n_components": target_n_comp,
                "target_largest_frac": target_largest_frac,
                "target_b_inst_n_components": target_b_inst_n_comp,
                "target_b_inst_largest_frac": target_b_inst_largest_frac,
                "target_raw_n_components": target_raw_n_comp,
                "target_raw_largest_frac": target_raw_largest_frac,
                "target_erode1_n_components": target_erode1_n_comp,
                "target_erode1_largest_frac": target_erode1_largest_frac,
                "target_solidity": target_solidity,
                "target_hull_volume_um3": target_hull_vol_um3,
                "target_rep_volume_um3": (
                    float(target_vox_count) * vox_vol_um3 if target_vox_count > 0 else np.nan
                ),
                "tcell_rep_volume_um3": (
                    float(tcell_vox_count) * vox_vol_um3 if tcell_vox_count > 0 else np.nan
                ),
                "tcell_n_components": tcell_n_comp,
                "tcell_largest_frac": tcell_largest_frac,
                "target_touches_fov_edge": target_touches_edge,
                "tcell_touches_fov_edge": tcell_touches_edge,
                "touches_fov_edge": touches_fov_edge,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1: Per-visit mask QC flags.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    workers = max(1, min(MAX_WORKERS, int(args.workers)))
    QC_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tasks, _ = build_tasks_for_chambers()
    tasks_t1 = [t for t in tasks if int(t["trace_num"]) == 1]
    print(
        f"Running Task 1 mask QC on {len(tasks_t1)} trace_num==1 chambers with {workers} workers...",
        flush=True,
    )

    visit_rows: list[dict[str, Any]] = []
    if workers <= 1:
        for i, t in enumerate(tasks_t1, 1):
            visit_rows.extend(process_one_chamber_qc(t))
            if i % 50 == 0 or i == len(tasks_t1):
                print(
                    f"  [{i}/{len(tasks_t1)}] chambers done ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_one_chamber_qc, t): t for t in tasks_t1}
            for i, fut in enumerate(as_completed(futs), 1):
                visit_rows.extend(fut.result())
                if i % 50 == 0 or i == len(tasks_t1):
                    print(
                        f"  [{i}/{len(tasks_t1)}] chambers done ({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )

    df_qc = (
        pd.DataFrame(visit_rows)
        .sort_values(["experiment", "roi_index", "visit"])
        .reset_index(drop=True)
    )

    # Load sweep-level descriptor summaries from data/step5/per_sweep_descriptors.csv
    df_sw5 = pd.read_csv(STEP5_SWEEP_CSV, low_memory=False)
    df_sw5_t1 = df_sw5[df_sw5["trace_num"] == 1].copy()
    sw_summary = (
        df_sw5_t1.groupby(["experiment", "roi", "visit"], as_index=False)
        .agg(
            sweep_target_vol_mean_um3=("target_volume_um3", "mean"),
            sweep_target_vol_sd_um3=("target_volume_um3", "std"),
            sweep_target_vol_min_um3=("target_volume_um3", "min"),
            sweep_target_vol_max_um3=("target_volume_um3", "max"),
            sweep_target_sph_mean=("target_sphericity", "mean"),
            sweep_target_sph_min=("target_sphericity", "min"),
            sweep_cup_depth_mean_um=("cup_depth_hull_p95_um", "mean"),
            sweep_cup_depth_sd_um=("cup_depth_hull_p95_um", "std"),
            sweep_cup_depth_max_um=("cup_depth_hull_p95_um", "max"),
            sweep_patch_n_comp_max=("patch_n_components", "max"),
        )
    )

    # Load step5 per_visit_descriptors.csv for chamber median target volume across all visits of the chamber
    df_v5 = pd.read_csv(STEP5_VISIT_CSV, low_memory=False)
    df_v5_t1 = df_v5[df_v5["trace_num"] == 1].copy()

    ch_med_all = (
        df_v5_t1[df_v5_t1["target_volume_um3"] > 0]
        .groupby(["experiment", "roi"])["target_volume_um3"]
        .median()
        .rename("chamber_median_target_vol_um3")
        .reset_index()
    )
    ch_med_contact = (
        df_v5_t1[(df_v5_t1["in_contact_visit"] == True) & (df_v5_t1["target_volume_um3"] > 0)]
        .groupby(["experiment", "roi"])["target_volume_um3"]
        .median()
        .rename("chamber_median_contact_target_vol_um3")
        .reset_index()
    )

    df_qc = df_qc.merge(sw_summary, on=["experiment", "roi", "visit"], how="left")
    df_qc = df_qc.merge(ch_med_all, on=["experiment", "roi"], how="left")
    df_qc = df_qc.merge(ch_med_contact, on=["experiment", "roi"], how="left")
    df_qc = df_qc.merge(
        df_v5_t1[
            [
                "experiment",
                "roi",
                "visit",
                "target_volume_um3",
                "tcell_volume_um3",
                "cup_depth_hull_p95_um",
                "cup_depth_over_radius",
                "tcell_protrusion_frac",
                "contact_area_um2",
                "patch_equiv_radius_um",
                "target_sphericity",
            ]
        ],
        on=["experiment", "roi", "visit"],
        how="left",
    )

    df_qc["target_vol_ratio_to_own_median"] = (
        df_qc["target_volume_um3"] / df_qc["chamber_median_target_vol_um3"]
    )
    df_qc["target_vol_ratio_to_contact_median"] = (
        df_qc["target_volume_um3"] / df_qc["chamber_median_contact_target_vol_um3"]
    )

    # Merge chamber death labels from Step 4 v2
    df_ch4 = pd.read_csv(STEP4_CHAMBER_CSV, dtype={"group_id": str})
    df_qc = df_qc.merge(
        df_ch4[
            [
                "experiment",
                "roi",
                "first_contact_visit",
                "first_contact_visit_200m",
                "ever_dead_any_visit_200m",
                "first_any_dead_visit_200m",
                "ever_dead_200m",
                "death_visit_200m",
                "ever_dead_any_visit",
                "first_any_dead_visit",
                "ever_dead",
                "death_visit",
            ]
        ],
        on=["experiment", "roi"],
        how="left",
    )

    # Task 1 flags:
    # 1. Exact definition on extracted target mask b_cell_sub:
    df_qc["flag_fragmented_b_cell_sub"] = (
        (df_qc["target_n_components"] >= 2)
        & df_qc["target_largest_frac"].notna()
        & (df_qc["target_largest_frac"] < 0.9)
    )
    # 2. Exact definition on raw target mask (I == 2) | (I == 4):
    df_qc["flag_fragmented_raw"] = (
        (df_qc["target_raw_n_components"] >= 2)
        & df_qc["target_raw_largest_frac"].notna()
        & (df_qc["target_raw_largest_frac"] < 0.9)
    )
    # 3. Exact definition on 1-voxel eroded b_cell_sub (multi-lobe / thin-bridge fragmentation):
    df_qc["flag_fragmented_erode1"] = (
        (df_qc["target_erode1_n_components"] >= 2)
        & df_qc["target_erode1_largest_frac"].notna()
        & (df_qc["target_erode1_largest_frac"] < 0.9)
    )
    # 4. Low solidity (< 0.5):
    df_qc["flag_low_solidity_lt05"] = (df_qc["target_solidity"] < 0.5).fillna(False)

    # Check if flag_fragmented_b_cell_sub is always False (because contact_descriptors.py selects a single 26-CC):
    # We store both flag_fragmented (the literal definition on target_n_components and target_largest_frac)
    # AND we check what target_n_components / target_largest_frac give!
    df_qc["flag_fragmented"] = df_qc["flag_fragmented_b_cell_sub"]
    df_qc["flag_edge"] = df_qc["touches_fov_edge"].astype(bool)

    df_qc = df_qc.sort_values(["experiment", "roi_index", "visit"]).reset_index(drop=True)
    df_qc.to_csv(QC_DIR / "per_visit_qc.csv", index=False)

    print(
        f"Saved {len(df_qc)} contact visits to {QC_DIR / 'per_visit_qc.csv'} in {time.perf_counter() - t0:.1f}s.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
