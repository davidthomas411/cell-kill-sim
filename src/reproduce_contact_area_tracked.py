#!/usr/bin/env python3
"""
Step 3c: Reproduce the Fei lab's published per-cell 3D contact area (CA) from the
deposited 3D U-Net masks (`*_predictions.h5`) with the lab's multi-frame 3D T-cell
tracker (`tracker` in `step2_instance_seg.m` and `if trace_num == 1` in `step1_setup_analysis.m`)
and MATLAB `(X, Y, Z)` dimension-order EDT tie-breaking (`[~, IDX] = bwdist(map_origin > 0)`).

Imports core functions from `src/reproduce_actin_measures.py` and `src/reproduce_contact_area.py`
without modifying either file.

Outputs (never writes into `data/step3/`):
  - `data/step3_tracked/<experiment>/per_chamber.csv`
  - `data/step3_tracked/<experiment>/per_sweep.csv`
  - `data/step3_tracked/summary.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

try:
    from src.paths import DATA_DIR, REPO_ROOT
except ImportError:
    from paths import DATA_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from reproduce_contact_area import (
    CA_VOXEL_FACTOR_UM2,
    MANIFEST_CSV,
    STRUCT26,
    V_TO_GROUP,
    bbox_slices,
    discover_complete_chambers,
)
from reproduce_actin_measures import (
    SWEEPS_PER_VISIT,
    edt_indices_matlab_xyz,
    instance_seg_matlab_xyz,
    summarize_chamber_void_max,
    tracker_tcell,
)

STEP3_TRACKED_DIR: Path = DATA_DIR / "step3_tracked"


def extract_tcell_regionprops3_table(
    t_inst_sub: np.ndarray, z0: int, y0: int, x0: int
) -> tuple[np.ndarray, int]:
    """
    Exact translation of MATLAB `stats = regionprops3(L); data_point = cat(2, stats.Centroid, stats.Volume);`
    in `step2_instance_seg.m` (lines 227-232) on `L = abs(t_instance(:,:,:,i))`.

    In MATLAB:
      - `h5read` loads `/predictions` with shape `(X, Y, Z) = (256, 256, Z)`.
      - `regionprops3(L)` on an integer label array returns a table with `max(L(:))` rows
        indexed `1 .. max(L(:))` (any skipped integer label `k` with 0 voxels has `Volume = 0`
        and `Centroid = [NaN, NaN, NaN]`, which `data_point(j, 4) > min_volume` skips while
        preserving 1-based row index `j = k`).
      - `Centroid` in MATLAB `regionprops3` on `(X, Y, Z)` is 1-based `[col(Y), row(X), slice(Z)]`.
    """
    max_t = int(t_inst_sub.max())
    if max_t <= 0:
        return np.empty((0, 4), dtype=np.float64), 0

    zz, yy, xx = np.nonzero(t_inst_sub > 0)
    lbls = t_inst_sub[zz, yy, xx]
    vols = np.bincount(lbls, minlength=max_t + 1)[1:].astype(np.float64)
    n_present = int(np.sum(vols > 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        cx = np.where(
            vols > 0,
            np.bincount(lbls, weights=(yy + y0 + 1.0), minlength=max_t + 1)[1:] / vols,
            np.nan,
        )
        cy = np.where(
            vols > 0,
            np.bincount(lbls, weights=(xx + x0 + 1.0), minlength=max_t + 1)[1:] / vols,
            np.nan,
        )
        cz = np.where(
            vols > 0,
            np.bincount(lbls, weights=(zz + z0 + 1.0), minlength=max_t + 1)[1:] / vols,
            np.nan,
        )

    return np.column_stack([cx, cy, cz, vols]), n_present


def compute_surface_for_tracked_id(
    I: np.ndarray,
    tracked_id: int,
    t_inst_sub: np.ndarray | None = None,
    sl_t: tuple[slice, slice, slice] | None = None,
) -> dict[str, int]:
    """
    Compute the 3D contact surface for a specific T-cell instance `j == tracked_id`
    following `step2_instance_seg.m` (lines 60-91) and `cl_factor.m` (lines 3-14):
      1. `b_instance = instance_seg(I == 2, I == 4)` with MATLAB `(X, Y, Z)` EDT tie-breaking.
      2. Note on `step2_instance_seg.m` lines 72-91: in MATLAB, the loop `for j = 1:max_index`
         writes `surface(surface_mask) = j; surface(edge_mask) = -j;` into a single shared 3D array
         `surface`. Could a higher-indexed touching T-cell `j2 > tracked_id` overwrite any voxel of
         `surface == tracked_id`?
         Let's run the exact loop over all touching `j = 1 .. max_index` when `max_index > tracked_id`
         and multiple T-cells touch `b_round`, so `surface` is 100% identical to MATLAB even on
         any shared boundary voxel!
      3. `s_real = (surface == tracked_id)`.
      4. `s_largest` = voxel count of the largest 26-connected component of `s_real`
         (and `s_largest_valid` = largest 26-connected component with geodesic distance `max_dist >= 1`
         from `s_edge = (surface == -tracked_id)`, matching `cl_factor.m` line 31).
    """
    if tracked_id <= 0:
        return {
            "n_target_instances": 0,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_smallest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }

    sl_all = bbox_slices(I > 0, 16, I.shape)
    if sl_all is None:
        return {
            "n_target_instances": 0,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_smallest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }

    I_sub = I[sl_all]
    if t_inst_sub is not None and sl_t is not None:
        t_inst = np.zeros(I_sub.shape, dtype=np.int8)
        tz0 = sl_t[0].start - sl_all[0].start
        ty0 = sl_t[1].start - sl_all[1].start
        tx0 = sl_t[2].start - sl_all[2].start
        t_inst[
            tz0 : tz0 + t_inst_sub.shape[0],
            ty0 : ty0 + t_inst_sub.shape[1],
            tx0 : tx0 + t_inst_sub.shape[2],
        ] = t_inst_sub
    else:
        t_inst = instance_seg_matlab_xyz(I_sub == 1, I_sub == 3)

    if not np.any(t_inst == tracked_id):
        b_has = np.any((I_sub == 2) | (I_sub == 4))
        n_b = int(np.sum(np.unique(instance_seg_matlab_xyz(I_sub == 2, I_sub == 4)) > 0)) if b_has else 0
        return {
            "n_target_instances": n_b,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_smallest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }

    b_inst = instance_seg_matlab_xyz(I_sub == 2, I_sub == 4)
    b_ids = [int(x) for x in np.unique(b_inst) if x > 0]
    n_target = len(b_ids)
    if n_target == 0:
        return {
            "n_target_instances": 0,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_smallest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }

    b_mask_full = b_inst != 0
    sl_cells = bbox_slices(b_mask_full | (t_inst != 0), 16, I_sub.shape)
    t_c = t_inst[sl_cells]
    b_c = b_inst[sl_cells]
    b_mask = b_c != 0

    edt_b = distance_transform_edt(~b_mask)
    not_b_dilate = ~(edt_b <= 10.0)
    b_round = edt_b <= 3.0

    max_index = int(t_c.max())
    touching_js: list[int] = []
    for j in range(1, max_index + 1):
        if np.any(b_round & (t_c == j)):
            touching_js.append(j)

    n_touching = len(touching_js)
    if tracked_id not in touching_js:
        return {
            "n_target_instances": n_target,
            "n_touching_tcell_instances": n_touching,
            "s_largest": 0,
            "s_smallest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }

    # Execute the exact loop from step2_instance_seg.m lines 72-91 for j >= tracked_id
    # (since any j < tracked_id only writes +-j, which does not affect `surface == +-tracked_id`,
    # whereas j > tracked_id could overwrite shared boundary voxels if two T-cells touch simultaneously).
    surface = np.zeros(t_c.shape, dtype=np.int8)
    for j in touching_js:
        if j < tracked_id:
            continue
        t_actin = t_c == j
        t_dilate = distance_transform_edt(~t_actin) <= 10.0
        bg_raw = (~t_dilate) & not_b_dilate
        bg_lbl, n_bg = label(bg_raw, structure=STRUCT26)
        if n_bg == 0:
            background = np.zeros_like(bg_raw, dtype=bool)
        else:
            # MATLAB line 82: `background = labelmatrix(CC) == 1` (first component in scan order)
            background = bg_lbl == 1

        map_origin = (
            b_mask.astype(np.int8) * 3
            + t_actin.astype(np.int8) * 2
            + background.astype(np.int8)
        )
        idx_xyz = edt_indices_matlab_xyz(map_origin == 0)
        map_exp = map_origin[idx_xyz]

        surface_mask = (map_exp == 2) & (distance_transform_edt(map_exp != 3) <= 1.0)
        edge_mask = surface_mask & (distance_transform_edt(map_exp != 1) <= 5.0)
        surface[surface_mask] = j
        surface[edge_mask] = -j

    s_real = surface == tracked_id
    s_edge = surface == -tracked_id
    s_total = int(np.sum(s_real))
    s_with_edge = int(np.sum(s_real | s_edge))

    if s_total == 0:
        s_largest = 0
        s_smallest = 0
        n_components = 0
    else:
        s_lbl, n_components = label(s_real, structure=STRUCT26)
        s_counts = np.bincount(s_lbl.ravel())[1:]
        s_largest = int(np.max(s_counts))
        s_smallest = int(np.min(s_counts))

    return {
        "n_target_instances": n_target,
        "n_touching_tcell_instances": n_touching,
        "s_largest": s_largest,
        "s_smallest": s_smallest,
        "s_total": s_total,
        "n_components": int(n_components),
        "s_with_edge": s_with_edge,
    }


def process_one_chamber(task: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single chamber `(experiment, roi)` end-to-end:
      1. Pass 1 (Tracker pass over all timepoints `1..timepoint` in the chamber):
         - For `mod(i, sample_num) == 0` (every 6th sweep, the tubulin sweep), `step2_instance_seg.m`
           lines 31-35 skip `instance_seg`, leaving `labels(:,:,:,1,i) = 0` (`0` detections).
         - For all other sweeps, run `instance_seg_matlab_xyz(I == 1, I == 3)` and record
           `regionprops3(abs(t_instance))` centroids and volumes.
         - Run `tracker_tcell(points_per_timepoint)` (`min_length = 50`, `max_distance = 200`,
           `miss_tolerance = 10`, `assignDetectionsToTracks` with `max_distance / 2 = 100`).
         - Rearrange `t_trace` into `(trace_num, sample_num, group_num)` as in `step2_instance_seg.m`
           lines 106-122 (`group_num = ceil(timepoint / sample_num)`).
      2. Pass 2 (Contact surface pass for Track 1 `t_trace(1, :, :)`):
         - If `trace_num >= 1`, for each visit `v = 1..group_num` and sweep `j = 1..sample_num-1` (1..5):
           let `mask_index = int(t_trace_re[0, j-1, v-1])`.
           If `mask_index > 0`, compute contact surface for T-cell `mask_index` only (using cached
           `t_inst_sub` from Pass 1).
           If `mask_index == 0`, `s_temp(j) = 0` (matching `step3_analysis.m` line 46 `if mask_index > 0`).
         - Combine sweeps 1..5 per visit by averaging over `s_temp > 0` (`step3_analysis.m` lines 58-64).
         - Chamber `CA_um2 = max_{v: s_all(v) > 0} s_all(v) * 0.0385` (`summary.m` line 18).
    """
    t0 = time.perf_counter()
    exp: str = task["experiment"]
    roi: str = task["roi"]
    roi_idx: int = int(task["roi_index"])
    gid: str = task["group_id"]
    cond: str = task["condition"]
    rel_paths: list[str] = task["rel_paths"]
    visits: list[int] = task["visits"]
    sweeps: list[int] = task["sweeps"]

    timepoint = len(rel_paths)
    sample_num = SWEEPS_PER_VISIT  # 6

    points_per_tp: list[np.ndarray] = []
    cached_t_inst: list[tuple[np.ndarray | None, tuple[slice, slice, slice] | None, int]] = []

    for idx0 in range(timepoint):
        # MATLAB 1-based index i = idx0 + 1: `if mod(i, sample_num) == 0` -> skip instance_seg
        if (idx0 + 1) % sample_num == 0:
            points_per_tp.append(np.empty((0, 4), dtype=np.float64))
            cached_t_inst.append((None, None, 0))
            continue

        abs_path = REPO_ROOT / rel_paths[idx0]
        with h5py.File(abs_path, "r") as hf:
            I = hf["predictions"][:]

        sl_t = bbox_slices((I == 1) | (I == 3), 2, I.shape)
        if sl_t is None:
            points_per_tp.append(np.empty((0, 4), dtype=np.float64))
            cached_t_inst.append((None, None, 0))
            continue

        I_t = I[sl_t]
        t_inst_sub = instance_seg_matlab_xyz(I_t == 1, I_t == 3)
        pts, n_present = extract_tcell_regionprops3_table(
            t_inst_sub, sl_t[0].start, sl_t[1].start, sl_t[2].start
        )
        points_per_tp.append(pts)
        cached_t_inst.append((t_inst_sub, sl_t, n_present))

    # Run T-cell tracker across all timepoints
    t_trace = tracker_tcell(points_per_tp)
    trace_num = int(t_trace.shape[0])
    track_lengths = [int(np.sum(t_trace[r] > 0)) for r in range(trace_num)]

    # Rearrange into (trace_num, sample_num, group_num) as in step2_instance_seg.m lines 106-120
    group_num = int(np.ceil(timepoint / float(sample_num)))
    t_trace_re = np.zeros((trace_num, sample_num, group_num), dtype=np.uint8)
    for v0 in range(group_num):
        for j0 in range(sample_num):
            idx0 = j0 + v0 * sample_num
            if idx0 < timepoint and trace_num > 0:
                t_trace_re[:, j0, v0] = t_trace[:, idx0]

    # Track 1 summary statistics
    if trace_num > 0:
        tr1_all = t_trace[0, :]
        tr1_actin = t_trace_re[0, : sample_num - 1, :].T.ravel()  # sweeps 1..5 across all visits
        tr1_detected_frames = int(np.sum(tr1_all > 0))
        tr1_nonzero_ids = tr1_all[tr1_all > 0].astype(int)
        tr1_unique_ids = sorted(int(x) for x in np.unique(tr1_nonzero_ids))
        vals_u, cnts_u = np.unique(tr1_nonzero_ids, return_counts=True)
        tr1_mode_id = int(vals_u[int(np.argmax(cnts_u))]) if vals_u.size > 0 else 0
        tr1_id_switches = int(np.sum(np.diff(tr1_nonzero_ids) != 0)) if tr1_nonzero_ids.size > 1 else 0
        # Compact per-visit string of Track 1 IDs on sweeps 1..5 (e.g. "v01:11111;v02:11101;...")
        visit_id_tokens = []
        for v0 in range(group_num):
            ids_v = "".join(str(int(x)) for x in t_trace_re[0, : sample_num - 1, v0])
            visit_id_tokens.append(f"v{v0+1:02d}:{ids_v}")
        tr1_per_visit_summary = ";".join(visit_id_tokens)
    else:
        tr1_detected_frames = 0
        tr1_unique_ids = []
        tr1_mode_id = 0
        tr1_id_switches = 0
        tr1_per_visit_summary = ""

    # Compute contact surface per sweep (sweeps 1..5 of each visit) for Track 1 (`t_trace(1,:,:)`)
    sweep_rows: list[dict[str, Any]] = []
    visit_s_largest: dict[int, float] = {}
    visit_s_total: dict[int, float] = {}

    for v0 in range(group_num):
        v_idx = v0 + 1
        s_temp_largest: list[int] = []
        s_temp_total: list[int] = []

        for j0 in range(sample_num - 1):  # j = 1..5
            idx0 = j0 + v0 * sample_num
            if idx0 >= timepoint:
                break
            s_idx = j0 + 1
            t_inst_sub, sl_t, n_t_present = cached_t_inst[idx0]
            tracked_id = int(t_trace_re[0, j0, v0]) if trace_num >= 1 else 0

            if tracked_id > 0:
                abs_path = REPO_ROOT / rel_paths[idx0]
                with h5py.File(abs_path, "r") as hf:
                    I = hf["predictions"][:]
                surf_res = compute_surface_for_tracked_id(
                    I=I,
                    tracked_id=tracked_id,
                    t_inst_sub=t_inst_sub,
                    sl_t=sl_t,
                )
            else:
                surf_res = {
                    "n_target_instances": 0,
                    "n_touching_tcell_instances": 0,
                    "s_largest": 0,
                    "s_smallest": 0,
                    "s_total": 0,
                    "n_components": 0,
                    "s_with_edge": 0,
                }

            s_l = int(surf_res["s_largest"])
            s_sm = int(surf_res["s_smallest"])
            s_t = int(surf_res["s_total"])
            s_temp_largest.append(s_l)
            s_temp_total.append(s_t)

            sweep_rows.append(
                {
                    "experiment": exp,
                    "roi": roi,
                    "visit": int(visits[idx0]),
                    "sweep": int(sweeps[idx0]),
                    "timepoint_idx": idx0 + 1,
                    "trace_num": trace_num,
                    "tracked_tcell_id": tracked_id,
                    "n_tcell_instances": int(n_t_present),
                    "n_target_instances": int(surf_res["n_target_instances"]),
                    "n_touching_tcell_instances": int(surf_res["n_touching_tcell_instances"]),
                    "s_largest": s_l,
                    "s_smallest": s_sm,
                    "s_total": s_t,
                    "n_components": int(surf_res["n_components"]),
                    "s_with_edge": int(surf_res["s_with_edge"]),
                }
            )

        arr_l = np.array(s_temp_largest, dtype=np.float64)
        arr_t = np.array(s_temp_total, dtype=np.float64)
        pos_l = arr_l[arr_l > 0]
        pos_t = arr_t[arr_t > 0]
        visit_s_largest[v_idx] = float(pos_l.mean()) if pos_l.size > 0 else 0.0
        visit_s_total[v_idx] = float(pos_t.mean()) if pos_t.size > 0 else 0.0

    # Summarize chamber CA via max over non-zero visits * 0.0385 (summary.m line 18)
    s_visit_arr = np.array([visit_s_largest.get(v_i, 0.0) for v_i in range(1, group_num + 1)], dtype=np.float64)
    s_total_arr = np.array([visit_s_total.get(v_i, 0.0) for v_i in range(1, group_num + 1)], dtype=np.float64)
    max_s_largest, peak_visit_1b = summarize_chamber_void_max(s_visit_arr, void_val=0.0)
    max_s_total, _ = summarize_chamber_void_max(s_total_arr, void_val=0.0)
    ca_um2 = max_s_largest * CA_VOXEL_FACTOR_UM2
    ca_total_um2 = max_s_total * CA_VOXEL_FACTOR_UM2

    n_t_per_sweep = [c[2] for idx0, c in enumerate(cached_t_inst) if (idx0 + 1) % sample_num != 0]
    max_t_inst = max(n_t_per_sweep) if n_t_per_sweep else 0
    multi_t_sweeps = sum(1 for x in n_t_per_sweep if x > 1)
    multi_touch_sweeps = sum(1 for r in sweep_rows if r["n_touching_tcell_instances"] > 1)
    max_b_inst = max((r["n_target_instances"] for r in sweep_rows), default=0)
    n_contact_visits = sum(1 for val in visit_s_largest.values() if val > 0)
    n_contact_sweeps = sum(1 for r in sweep_rows if r["s_largest"] > 0)
    multi_comp_sweeps = sum(1 for r in sweep_rows if r["n_components"] > 1)

    elapsed_s = time.perf_counter() - t0

    chamber_row: dict[str, Any] = {
        "experiment": exp,
        "roi": roi,
        "roi_index": roi_idx,
        "group_id": gid,
        "condition": cond,
        "trace_num": trace_num,
        "included_by_trace_rule": bool(trace_num == 1),
        "track_lengths": json.dumps(track_lengths),
        "track1_detected_frames": tr1_detected_frames,
        "track1_mode_id": tr1_mode_id,
        "track1_unique_ids": json.dumps(tr1_unique_ids),
        "track1_id_switches": tr1_id_switches,
        "track1_ids_per_visit": tr1_per_visit_summary,
        "CA_um2": round(ca_um2, 4),
        "CA_total_um2": round(ca_total_um2, 4),
        "peak_visit": peak_visit_1b,
        "max_visit_s_largest": round(max_s_largest, 3),
        "max_visit_s_total": round(max_s_total, 3),
        "n_timepoints": timepoint,
        "n_visits": group_num,
        "n_contact_visits": n_contact_visits,
        "n_contact_sweeps": n_contact_sweeps,
        "max_tcell_instances": max_t_inst,
        "multi_tcell_sweeps": multi_t_sweeps,
        "multi_touching_tcell_sweeps": multi_touch_sweeps,
        "max_target_instances": max_b_inst,
        "multi_component_sweeps": multi_comp_sweeps,
        "elapsed_s": round(elapsed_s, 3),
    }

    for v_i in range(1, group_num + 1):
        s_val = visit_s_largest.get(v_i, 0.0)
        chamber_row[f"visit_{v_i:02d}_s"] = round(s_val, 3)
        chamber_row[f"visit_{v_i:02d}_CA_um2"] = round(s_val * CA_VOXEL_FACTOR_UM2, 4)

    return {
        "chamber_row": chamber_row,
        "sweep_rows": sweep_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Fei lab tracked 3D contact area (CA) from _predictions.h5 masks."
    )
    parser.add_argument("--experiment", type=str, default=None, help="Filter by experiment folder name.")
    parser.add_argument("--roi", type=str, default=None, help="Filter by ROI folder name (e.g. MIP_ROI_0).")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker processes (default: 6, max: 6).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=STEP3_TRACKED_DIR,
        help="Output base directory (default: data/step3_tracked).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose data/step3_tracked/<experiment>/per_chamber.csv already has all chambers.",
    )
    args = parser.parse_args()

    workers = min(6, max(1, int(args.workers)))
    out_base: Path = args.out_dir
    if out_base.resolve() == (DATA_DIR / "step3").resolve():
        raise ValueError("Refusing to write into data/step3/; use data/step3_tracked/ instead.")
    out_base.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(MANIFEST_CSV)
    complete_chambers = discover_complete_chambers(
        manifest_df, experiment_filter=args.experiment, roi_filter=args.roi
    )
    if not complete_chambers:
        print("No complete chambers found matching filter criteria.")
        return 1

    # Count expected chambers per experiment
    exp_expected_counts: dict[str, int] = {}
    for exp, _roi, _cdf in complete_chambers:
        exp_expected_counts[str(exp)] = exp_expected_counts.get(str(exp), 0) + 1

    skipped_exps: set[str] = set()
    if args.skip_existing and args.roi is None:
        for exp_name, exp_cnt in exp_expected_counts.items():
            existing_csv = out_base / exp_name / "per_chamber.csv"
            if existing_csv.is_file():
                try:
                    df_ex = pd.read_csv(existing_csv)
                    if len(df_ex) == exp_cnt:
                        skipped_exps.add(exp_name)
                except Exception:
                    pass
        if skipped_exps:
            print(f"Skipping {len(skipped_exps)} already-completed experiment(s): {sorted(skipped_exps)}", flush=True)

    tasks: list[dict[str, Any]] = []
    for exp, roi, cdf in complete_chambers:
        if str(exp) in skipped_exps:
            continue
        cdf_sorted = cdf.sort_values(["visit_index", "sweep_index"])
        roi_idx = int(roi.split("_")[-1]) if roi.split("_")[-1].isdigit() else -1
        gid, cond = V_TO_GROUP.get(str(exp), ("", ""))
        tasks.append(
            {
                "experiment": str(exp),
                "roi": str(roi),
                "roi_index": roi_idx,
                "group_id": str(gid),
                "condition": str(cond),
                "rel_paths": [str(p) for p in cdf_sorted["path"].tolist()],
                "visits": [int(v) for v in cdf_sorted["visit_index"].tolist()],
                "sweeps": [int(s) for s in cdf_sorted["sweep_index"].tolist()],
            }
        )

    print(
        f"Running tracked contact-area pipeline on {len(tasks)} chamber(s) "
        f"with {workers} worker(s)...",
        flush=True,
    )
    t_start = time.perf_counter()
    chamber_rows: list[dict[str, Any]] = []
    sweep_rows_by_exp: dict[str, list[dict[str, Any]]] = {}
    done_by_exp: dict[str, list[dict[str, Any]]] = {}

    def _save_experiment_outputs(exp_name: str, rows_exp: list[dict[str, Any]], sweeps_exp: list[dict[str, Any]]) -> None:
        exp_dir = out_base / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        df_exp = pd.DataFrame(rows_exp).sort_values("roi_index").reset_index(drop=True)
        max_v_exp = int(df_exp["n_visits"].max())
        drop_cols = [
            col
            for col in df_exp.columns
            if col.startswith("visit_")
            and (col.endswith("_s") or col.endswith("_CA_um2"))
            and int(col.split("_")[1]) > max_v_exp
        ]
        df_exp_clean = df_exp.drop(columns=drop_cols)
        df_exp_clean.to_csv(exp_dir / "per_chamber.csv", index=False)
        if sweeps_exp:
            df_sw = pd.DataFrame(sweeps_exp).sort_values(["roi", "visit", "sweep"]).reset_index(drop=True)
            df_sw.to_csv(exp_dir / "per_sweep.csv", index=False)

    if len(tasks) == 0:
        print("All matching experiments are already completed.", flush=True)
    elif workers <= 1:
        for idx_done, task in enumerate(tasks, start=1):
            res = process_one_chamber(task)
            cr = res["chamber_row"]
            exp_n = cr["experiment"]
            chamber_rows.append(cr)
            sweep_rows_by_exp.setdefault(exp_n, []).extend(res["sweep_rows"])
            done_by_exp.setdefault(exp_n, []).append(cr)
            if args.roi is not None or len(done_by_exp[exp_n]) == exp_expected_counts.get(exp_n, 0):
                _save_experiment_outputs(exp_n, done_by_exp[exp_n], sweep_rows_by_exp.get(exp_n, []))
            el = time.perf_counter() - t_start
            print(
                f"  [{idx_done:3d}/{len(tasks):3d}] {cr['experiment']} {cr['roi']:10s} "
                f"trace_num={cr['trace_num']} tr1_frames={cr['track1_detected_frames']:3d} "
                f"CA={cr['CA_um2']:7.3f} um^2 ({cr['elapsed_s']:.2f}s, wall={el:.1f}s)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(process_one_chamber, task): task for task in tasks}
            for idx_done, fut in enumerate(as_completed(fut_map), start=1):
                res = fut.result()
                cr = res["chamber_row"]
                exp_n = cr["experiment"]
                chamber_rows.append(cr)
                sweep_rows_by_exp.setdefault(exp_n, []).extend(res["sweep_rows"])
                done_by_exp.setdefault(exp_n, []).append(cr)
                if args.roi is not None or len(done_by_exp[exp_n]) == exp_expected_counts.get(exp_n, 0):
                    _save_experiment_outputs(exp_n, done_by_exp[exp_n], sweep_rows_by_exp.get(exp_n, []))
                    print(
                        f"  --> Saved completed experiment {exp_n} ({len(done_by_exp[exp_n])} chambers, "
                        f"trace_num==1: {sum(1 for r in done_by_exp[exp_n] if r['trace_num'] == 1)})",
                        flush=True,
                    )
                if idx_done % 10 == 0 or idx_done == len(tasks):
                    el = time.perf_counter() - t_start
                    print(
                        f"  [{idx_done:3d}/{len(tasks):3d}] latest={cr['experiment']}/{cr['roi']} "
                        f"trace_num={cr['trace_num']} CA={cr['CA_um2']:.2f} um^2 "
                        f"(elapsed={el:.1f}s, {el / idx_done:.2f}s wall/chamber)",
                        flush=True,
                    )

    wall_elapsed_s = time.perf_counter() - t_start

    # Combine with any skipped experiments on disk so per_chamber_all.csv and summary.json cover all experiments
    all_dfs: list[pd.DataFrame] = []
    if chamber_rows:
        all_dfs.append(pd.DataFrame(chamber_rows))
    for sk_exp in sorted(skipped_exps):
        sk_csv = out_base / sk_exp / "per_chamber.csv"
        if sk_csv.is_file():
            all_dfs.append(pd.read_csv(sk_csv))

    df_all_chambers = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["experiment", "roi_index"])
        .reset_index(drop=True)
    )

    # Save combined table and summary JSON if running across multiple experiments or full run
    if args.experiment is None and args.roi is None:
        df_all_chambers.to_csv(out_base / "per_chamber_all.csv", index=False)

    cpu_sum_s = float(df_all_chambers["elapsed_s"].sum())
    summary_payload = {
        "n_chambers": len(df_all_chambers),
        "workers": workers,
        "wall_elapsed_s": round(wall_elapsed_s, 3),
        "wall_s_per_chamber": round(wall_elapsed_s / max(1, len(df_all_chambers)), 3),
        "cpu_s_per_chamber_mean": round(cpu_sum_s / max(1, len(df_all_chambers)), 3),
        "trace_num_counts": {
            str(k): int(v)
            for k, v in df_all_chambers["trace_num"].value_counts().sort_index().items()
        },
        "included_trace_num_eq_1": int((df_all_chambers["trace_num"] == 1).sum()),
    }
    summary_suffix = (
        f"_{args.experiment}_{args.roi}"
        if (args.experiment and args.roi)
        else (f"_{args.experiment}" if args.experiment else "")
    )
    summary_path = out_base / f"summary{summary_suffix}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        f"Completed {len(df_all_chambers)} chamber(s) in {wall_elapsed_s:.2f}s "
        f"(wall: {wall_elapsed_s / max(1, len(df_all_chambers)):.2f}s/chamber, "
        f"single-worker CPU mean: {cpu_sum_s / max(1, len(df_all_chambers)):.2f}s/chamber).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
