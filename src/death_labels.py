#!/usr/bin/env python3
"""
Step 4: Per-chamber and per-visit target cell death labels and death times
from 2D 405-nm SYTOX max-intensity projections (`*_MIPxy_405.tif`, `*_MIPxz_405.tif`)
and 3D U-Net segmentation masks (`*_predictions.h5`) across all 13 experiments
(502 chambers, 10,510 chamber-visits), preserving the pilot outputs for Groups 1-3.

Includes:
  1. Calibration against the Fei lab's 3D rule (`step2_instance_seg.m` + `dead_count.m`)
     on all 20 local raw 3D volumes (`data/scidb_19235_chambers/*/<stem>.h5`), plus
     exact verification of the deposited 2D MIP projection geometry:
       - `MIPxy_405 == raw[0].max(axis=0).T` (exact uint16 match, max diff = 0)
       - `MIPxz_405 == raw[0].max(axis=1).T` (exact uint16 match, max diff = 0)
  2. 2D projection-based dead-dye rule:
       - Target projection masks matching MIP orientation:
           `b_xy = ((pred == 2) | (pred == 4)).max(axis=0).T`
           `b_xz = ((pred == 2) | (pred == 4)).max(axis=1).T`
       - 2D Gaussian filter (`sigma = 3.0`, `mode = 'nearest'`, `truncate = 2.0`),
         threshold `> 404.0` (compensating for the +4.6 to +5.5 ADU extreme-value shift
         of the 2D max projection over Z=166-190 or Y=256 slices relative to the 3D
         threshold `398.0`), 2D morphological closing (`disk(3)`), and 8-connected area
         opening (`>= 20` pixels) in both `xy` and `xz`.
       - Per-visit positive call: `dead_visit = (ov_xy >= 20) and (ov_xz >= 20)`
         (where `ov_xy` and `ov_xz` are pixel overlaps of the cleaned 2D dead-dye masks
         with `b_xy` and `b_xz`; we also record the 10-pixel dilated overlap `ov10_xy`,
         `ov10_xz` and the single-visit union `ever_dead_any_visit`).
       - Per-chamber sustained death call: `ever_dead` is True when the chamber exhibits
         sustained positive signal (`>= 2` consecutive positive visits), with `death_visit`
         set to the first visit of the sustained positive run and
         `death_time_min = death_visit * 10.0`.
       - Survival & censoring columns: `first_contact_visit`, `contact_to_death_min`,
         `last_observed_visit`, `censored`, `km_contact_time_min`, `km_event_sustained`,
         `any_contact_to_death_min`, `censored_any_visit`, `km_any_contact_time_min`,
         `km_event_any_visit`.
  3. Outputs under `data/step4/`:
       - `calibration_20_raw_volumes.csv`
       - Pilot tables (Groups 1-3, preserved):
           `per_visit_death_labels.csv`, `per_chamber_death_labels.csv`,
           `group_kill_rate_comparison.csv`, `summary.json`
       - All-experiment tables (13 experiments, 502 chambers):
           `per_visit_death_labels_all.csv`, `per_chamber_death_labels_all.csv`,
           `group_kill_rate_comparison_all.csv`, `summary_all.json`
       - Per-experiment directories `data/step4/<experiment>/`:
           `per_visit_death_labels.csv`, `per_chamber_death_labels.csv`
  4. Figures following House Style:
       - `figures/step4_death_labels_pilot.png` (preserved)
       - `figures/step4_death_labels_all.png`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from scipy.stats import beta
import tifffile

try:
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from reproduce_contact_area import (
    STRUCT26,
    bbox_slices,
    bwareaopen_26,
    compute_contact_surface_from_mask_array,
    imfill_holes_26,
    sphere_dilate_edt,
)

# 6-connected structuring element matching MATLAB `strel('sphere', 1)` in 3D
SPHERE1: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
SPHERE1[1, 1, :] = True
SPHERE1[1, :, 1] = True
SPHERE1[:, 1, 1] = True


def edt_indices_matlab_xyz(zero_mask_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nearest-nonzero indices matching MATLAB `[~, IDX] = bwdist(~zero_mask)` by running
    SciPy's separable 3D Maurer EDT in MATLAB's `(X, Y, Z)` dimension order (`transpose(2, 1, 0)`),
    then mapping the returned `(ix, iy, iz)` indices back to `(iz, iy, ix)`.
    """
    _, inds_xyz = distance_transform_edt(zero_mask_zyx.transpose(2, 1, 0), return_indices=True)
    ix = inds_xyz[0].transpose(2, 1, 0)
    iy = inds_xyz[1].transpose(2, 1, 0)
    iz = inds_xyz[2].transpose(2, 1, 0)
    return (iz, iy, ix)


def instance_seg_matlab_xyz(nucleus: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """
    Exact translation of `instance_seg(nucleus, membrane)` from `step2_instance_seg.m`
    using MATLAB `(X, Y, Z)` EDT tie-breaking when `max_index > 1`.
    """
    result = np.zeros(nucleus.shape, dtype=np.int8)
    nuc_clean, nuc_lbl, max_index = bwareaopen_26(nucleus, 3000)

    if max_index == 0:
        if int(np.sum(membrane)) > 3000:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                counts = np.bincount(mem_lbl.ravel())
                counts[0] = 0
                max_idx = int(np.argmax(counts))
                filled = imfill_holes_26(mem_lbl == max_idx)
                result[filled] = 1
        return result
    elif max_index == 1:
        membrane_instance = membrane.astype(np.int8)
    else:
        sl = bbox_slices(nuc_clean | membrane, 2, nucleus.shape)
        membrane_instance = np.zeros(nucleus.shape, dtype=np.int8)
        if sl is not None:
            inds = edt_indices_matlab_xyz(~nuc_clean[sl])
            sub_inst = nuc_lbl[sl][inds].astype(np.int8)
            sub_inst[~membrane[sl]] = 0
            membrane_instance[sl] = sub_inst

    for i in range(1, max_index + 1):
        membrane_mask = membrane_instance == i
        nucleus_mask = nuc_lbl == i
        temp = membrane_mask & sphere_dilate_edt(nucleus_mask, 25.0)
        if int(np.sum(temp)) < 1000:
            continue
        mem_lbl, n_mem = label(membrane_mask, structure=STRUCT26)
        if n_mem > 0:
            counts = np.bincount(mem_lbl.ravel())
            counts[0] = 0
            max_idx = int(np.argmax(counts))
            combined = nucleus_mask | (mem_lbl == max_idx)
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
                max_idx = int(np.argmax(counts))
                filled = imfill_holes_26(mem_lbl == max_idx)
                result[filled] = 1

    return result

MIPS_DIR: Path = DATA_DIR / "scidb_19235_mips"
MIPS_MANIFEST_CSV: Path = MIPS_DIR / "manifest.csv"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
CHAMBERS_DIR: Path = DATA_DIR / "scidb_19235_chambers"
STEP3_DIR: Path = DATA_DIR / "step3"
STEP3_TRACKED_DIR: Path = DATA_DIR / "step3_tracked"
STEP4_DIR: Path = DATA_DIR / "step4"
CLEAN_419_CSV: Path = DATA_DIR / "clean_per_cell_419.csv"
ROW_MAP_CSV: Path = REPO_ROOT / "results" / "chamber_to_row_map.csv"
ROW_MAP_TRACKED_CSV: Path = REPO_ROOT / "results" / "chamber_to_row_map_tracked.csv"
GROUP_INV_CSV: Path = DATA_DIR / "dent_step1_group_inventory.csv"

PILOT_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("1", "v01_6_40_120_0-6_hd1", "Untreated high-killing CAR-T"),
    ("2", "v02_6_40_120_0-6_hd3", "Untreated high-killing CAR-T"),
    ("3", "v03_6_40_120_0-6_hd4", "Untreated high-killing CAR-T"),
)

ALL_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("1", "v01_6_40_120_0-6_hd1", "Untreated high-killing CAR-T"),
    ("2", "v02_6_40_120_0-6_hd3", "Untreated high-killing CAR-T"),
    ("3", "v03_6_40_120_0-6_hd4", "Untreated high-killing CAR-T"),
    ("4", "v04_6_40_120_0-6_hd6", "Untreated high-killing CAR-T"),
    ("5", "v05_6_40_120_0-6_hd7", "Untreated high-killing CAR-T"),
    ("6", "v06_6_40_120_0-6_hd8", "Untreated low-killing CAR-T"),
    ("7", "v07_6_39_120_0-6_hd9", "Untreated low-killing CAR-T"),
    ("8", "v08_6_40_120_0-6_hd2", "Untreated low-killing CAR-T"),
    ("9", "v09_6_40_120_0-6_hd10", "Untreated low-killing CAR-T"),
    ("10", "v10_6_40_120_0-6_hd5", "Untreated low-killing CAR-T"),
    ("8*", "v11_6_36_120_0-6_hd2_ds", "Dasatinib 50 nM CAR-T"),
    ("9*", "v12_6_35_120_0-6_hd10ds", "Dasatinib 50 nM CAR-T"),
    ("10*", "v13_6_40_120_0-6_hd5_ds", "Dasatinib 50 nM CAR-T"),
)

GROUP_ORDER_MAP: dict[str, int] = {gid: i for i, (gid, _, _) in enumerate(ALL_EXPERIMENTS)}

# 3D sphere(3) and 2D disk(3) structuring elements
_zz, _yy3, _xx3 = np.ogrid[-3:4, -3:4, -3:4]
SPHERE3_3D: np.ndarray = (_zz**2 + _yy3**2 + _xx3**2) <= 9

_yy2, _xx2 = np.ogrid[-3:4, -3:4]
DISK3_2D: np.ndarray = (_yy2**2 + _xx2**2) <= 9
STRUCT8_2D: np.ndarray = np.ones((3, 3), dtype=bool)

# Calibrated 2D rule parameters
GAUSS_SIGMA_2D: float = 3.0
GAUSS_TRUNCATE: float = 2.0
THRESH_3D_ADU: float = 398.0
THRESH_2D_ADU: float = 404.0
MIN_COMPONENT_AREA_2D: int = 20
MIN_OVERLAP_PX_2D: int = 20
VISIT_INTERVAL_MIN: float = 10.0


def compute_3d_dead_mask(raw405: np.ndarray, threshold: float = THRESH_3D_ADU) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact translation of `step2_instance_seg.m` lines 43-45:
        D = imgaussfilt3(raw_temp(:,:,:,1), 3);
        d_mask = imclose(D > 398, strel('sphere', 3));
        d_mask = bwareaopen(d_mask, 20, 26);
    """
    D = gaussian_filter(
        raw405.astype(np.float64),
        sigma=GAUSS_SIGMA_2D,
        mode="nearest",
        truncate=GAUSS_TRUNCATE,
    )
    binary = D > threshold
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool), D
    closed = binary_closing(binary, structure=SPHERE3_3D)
    lbl, n_c = label(closed, structure=STRUCT26)
    if n_c == 0:
        return np.zeros_like(binary, dtype=bool), D
    cnts = np.bincount(lbl.ravel())
    cnts[0] = 0
    keep = np.isin(lbl, np.where(cnts >= 20)[0])
    return keep, D


def compute_2d_dead_mask(
    mip: np.ndarray,
    threshold: float = THRESH_2D_ADU,
    min_area: int = MIN_COMPONENT_AREA_2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2D counterpart of the lab's dead-dye mask on a max-intensity projection:
        g = imgaussfilt(mip, 3);
        d_2d = bwareaopen(imclose(g > threshold, strel('disk', 3)), min_area, 8);
    """
    g = gaussian_filter(
        mip.astype(np.float64),
        sigma=GAUSS_SIGMA_2D,
        mode="nearest",
        truncate=GAUSS_TRUNCATE,
    )
    binary = g > threshold
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool), g
    closed = binary_closing(binary, structure=DISK3_2D)
    lbl, n_c = label(closed, structure=STRUCT8_2D)
    if n_c == 0:
        return np.zeros_like(binary, dtype=bool), g
    cnts = np.bincount(lbl.ravel())
    cnts[0] = 0
    keep = np.isin(lbl, np.where(cnts >= min_area)[0])
    return keep, g


def dilate_2d_edt(mask: np.ndarray, radius: float = 10.0) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    return distance_transform_edt(~mask) <= radius


def calibrate_on_local_raw_volumes() -> pd.DataFrame:
    """
    Task 2: Evaluate the Fei lab's 3D dead-dye rule on all 20 local raw volumes under
    `data/scidb_19235_chambers/`, verify deposited MIPxy_405 and MIPxz_405 identity
    against `raw[0]`, and compare the 2D projection rule against the 3D decisions.
    """
    raw_files = sorted(
        [p for p in CHAMBERS_DIR.rglob("*.h5") if not p.name.endswith("_predictions.h5")]
    )
    rows: list[dict[str, Any]] = []

    for p in raw_files:
        stem = p.stem
        parent_name = p.parent.name
        marker = "_MIP_ROI_"
        idx = parent_name.find(marker)
        exp = parent_name[:idx]
        roi = parent_name[idx + 1 :]

        pred_p = p.parent / f"{stem}_predictions.h5"
        if not pred_p.exists():
            pred_p = MASKS_DIR / exp / roi / f"{stem}_predictions.h5"
        xy_tif_p = p.parent / f"{stem}_MIPxy_405.tif"
        xz_tif_p = p.parent / f"{stem}_MIPxz_405.tif"

        with h5py.File(p, "r") as hf:
            raw405 = hf["raw"][0][:]  # (Z, 256, 256) uint16
        with h5py.File(pred_p, "r") as hf:
            pred = hf["predictions"][:]  # (Z, 256, 256) uint8

        # 1. Verify MIP geometry and intensity scale against deposited TIFFs
        mip_xy_from_raw = raw405.max(axis=0).T
        mip_xz_from_raw = raw405.max(axis=1).T
        has_deposited_tif = xy_tif_p.exists() and xz_tif_p.exists()
        if has_deposited_tif:
            xy_tif = tifffile.imread(xy_tif_p)
            xz_tif = tifffile.imread(xz_tif_p)
            max_diff_xy = int(np.max(np.abs(mip_xy_from_raw.astype(int) - xy_tif.astype(int))))
            max_diff_xz = int(np.max(np.abs(mip_xz_from_raw.astype(int) - xz_tif.astype(int))))
        else:
            max_diff_xy = -1
            max_diff_xz = -1

        # 2. Compute Fei lab 3D dead-dye mask and 3D decisions
        d_mask_3d, D_3d = compute_3d_dead_mask(raw405, THRESH_3D_ADU)
        d_vox_3d = int(d_mask_3d.sum())
        d_dil10_3d = sphere_dilate_edt(d_mask_3d, 10.0) if d_vox_3d > 0 else np.zeros_like(d_mask_3d)

        res_3d = compute_contact_surface_from_mask_array(pred, return_volumes=True)
        b_inst = res_3d["b_instance"]
        s_mask = res_3d["surface"] != 0
        b_any_3d = b_inst > 0
        raw_b_3d = (pred == 2) | (pred == 4)

        # Exact dead_count.m loop over b_instance
        dead_count_3d = 0
        max_b_id = int(b_inst.max()) if b_inst.size > 0 else 0
        if np.any(s_mask) and max_b_id > 0:
            for i_id in range(1, max_b_id + 1):
                bi = b_inst == i_id
                touch = s_mask & sphere_dilate_edt(bi, 1.0)
                if not np.any(touch):
                    continue
                if np.any(bi & d_dil10_3d):
                    dead_count_3d += 1

        ov10_binst_3d = int(np.sum(b_any_3d & d_dil10_3d))
        ov10_rawb_3d = int(np.sum(raw_b_3d & d_dil10_3d))
        dec_3d_binst = ov10_binst_3d > 0
        dec_3d_rawb = ov10_rawb_3d > 0

        if d_vox_3d > 0 and np.any(b_any_3d):
            min_dist_binst_3d = float(distance_transform_edt(~d_mask_3d)[b_any_3d].min())
        else:
            min_dist_binst_3d = np.inf
        if d_vox_3d > 0 and np.any(raw_b_3d):
            min_dist_rawb_3d = float(distance_transform_edt(~d_mask_3d)[raw_b_3d].min())
        else:
            min_dist_rawb_3d = np.inf

        # 3. Compute 2D projections and 2D rule metrics
        b_xy = raw_b_3d.max(axis=0).T
        b_xz = raw_b_3d.max(axis=1).T
        binst_xy = b_any_3d.max(axis=0).T
        binst_xz = b_any_3d.max(axis=1).T

        d_xy, g_xy = compute_2d_dead_mask(mip_xy_from_raw, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xz, g_xz = compute_2d_dead_mask(mip_xz_from_raw, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xy_dil10 = dilate_2d_edt(d_xy, 10.0)
        d_xz_dil10 = dilate_2d_edt(d_xz, 10.0)

        ov_xy_rawb = int(np.sum(d_xy & b_xy))
        ov_xz_rawb = int(np.sum(d_xz & b_xz))
        ov10_xy_rawb = int(np.sum(d_xy_dil10 & b_xy))
        ov10_xz_rawb = int(np.sum(d_xz_dil10 & b_xz))

        ov_xy_binst = int(np.sum(d_xy & binst_xy))
        ov_xz_binst = int(np.sum(d_xz & binst_xz))

        dec_2d_direct = (ov_xy_rawb >= MIN_OVERLAP_PX_2D) and (ov_xz_rawb >= MIN_OVERLAP_PX_2D)
        dec_2d_dil10 = (ov10_xy_rawb >= MIN_OVERLAP_PX_2D) and (ov10_xz_rawb >= MIN_OVERLAP_PX_2D)
        dec_2d_binst = (ov_xy_binst >= MIN_OVERLAP_PX_2D) and (ov_xz_binst >= MIN_OVERLAP_PX_2D)

        rows.append(
            {
                "experiment": exp,
                "roi": roi,
                "stem": stem,
                "z_slices": int(raw405.shape[0]),
                "has_deposited_tif": has_deposited_tif,
                "mip_xy_max_abs_diff": max_diff_xy,
                "mip_xz_max_abs_diff": max_diff_xz,
                "raw405_min": int(raw405.min()),
                "raw405_max": int(raw405.max()),
                "D3d_max": round(float(D_3d.max()), 2),
                "g2d_xy_in_target_max": round(float(g_xy[b_xy].max()) if np.any(b_xy) else 0.0, 2),
                "g2d_xz_in_target_max": round(float(g_xz[b_xz].max()) if np.any(b_xz) else 0.0, 2),
                "d3d_voxels": d_vox_3d,
                "b_inst_voxels": int(b_any_3d.sum()),
                "raw_b_voxels": int(raw_b_3d.sum()),
                "s_contact_voxels": int(s_mask.sum()),
                "min_dist_3d_binst_vox": round(min_dist_binst_3d, 2) if np.isfinite(min_dist_binst_3d) else np.nan,
                "min_dist_3d_rawb_vox": round(min_dist_rawb_3d, 2) if np.isfinite(min_dist_rawb_3d) else np.nan,
                "ov10_3d_binst_vox": ov10_binst_3d,
                "ov10_3d_rawb_vox": ov10_rawb_3d,
                "dead_count_3d_strict": dead_count_3d,
                "dec_3d_binst_within10": dec_3d_binst,
                "dec_3d_rawb_within10": dec_3d_rawb,
                "d2d_xy_pixels": int(d_xy.sum()),
                "d2d_xz_pixels": int(d_xz.sum()),
                "ov_2d_xy_rawb_px": ov_xy_rawb,
                "ov_2d_xz_rawb_px": ov_xz_rawb,
                "ov10_2d_xy_rawb_px": ov10_xy_rawb,
                "ov10_2d_xz_rawb_px": ov10_xz_rawb,
                "ov_2d_xy_binst_px": ov_xy_binst,
                "ov_2d_xz_binst_px": ov_xz_binst,
                "dec_2d_direct": dec_2d_direct,
                "dec_2d_dil10": dec_2d_dil10,
                "dec_2d_binst": dec_2d_binst,
            }
        )

    return pd.DataFrame(rows)


def _process_one_chamber(args: tuple[str, str, str, str, list[tuple[int, str, Path, Path, Path, float]]]) -> list[dict[str, Any]]:
    group_id, exp, cond, roi, visit_items = args
    roi_idx = int(roi.split("_")[-1]) if roi.split("_")[-1].isdigit() else -1
    rows: list[dict[str, Any]] = []

    for v_idx, stem, xy_path, xz_path, pred_path, ca_um2 in visit_items:
        mip_xy = tifffile.imread(xy_path)
        mip_xz = tifffile.imread(xz_path)
        with h5py.File(pred_path, "r") as hf:
            pred = hf["predictions"][:]

        raw_b = (pred == 2) | (pred == 4)
        b_xy = raw_b.max(axis=0).T
        b_xz = raw_b.max(axis=1).T

        d_xy, g_xy = compute_2d_dead_mask(mip_xy, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xz, g_xz = compute_2d_dead_mask(mip_xz, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xy_dil10 = dilate_2d_edt(d_xy, 10.0)
        d_xz_dil10 = dilate_2d_edt(d_xz, 10.0)

        target_xy_px = int(b_xy.sum())
        target_xz_px = int(b_xz.sum())
        d_xy_px = int(d_xy.sum())
        d_xz_px = int(d_xz.sum())

        ov_xy = int((d_xy & b_xy).sum())
        ov_xz = int((d_xz & b_xz).sum())
        ov10_xy = int((d_xy_dil10 & b_xy).sum())
        ov10_xz = int((d_xz_dil10 & b_xz).sum())

        frac_xy = (ov_xy / target_xy_px) if target_xy_px > 0 else 0.0
        frac_xz = (ov_xz / target_xz_px) if target_xz_px > 0 else 0.0

        bg_xy = float(np.median(g_xy[~b_xy])) if np.any(~b_xy) else float(np.median(g_xy))
        bg_xz = float(np.median(g_xz[~b_xz])) if np.any(~b_xz) else float(np.median(g_xz))

        if target_xy_px > 0:
            vals_xy = g_xy[b_xy]
            max_g_xy = float(vals_xy.max())
            p95_g_xy = float(np.percentile(vals_xy, 95))
            mean_g_xy = float(vals_xy.mean())
            max_raw_xy = int(mip_xy[b_xy].max())
        else:
            max_g_xy, p95_g_xy, mean_g_xy, max_raw_xy = 0.0, 0.0, 0.0, 0

        if target_xz_px > 0:
            vals_xz = g_xz[b_xz]
            max_g_xz = float(vals_xz.max())
            p95_g_xz = float(np.percentile(vals_xz, 95))
            mean_g_xz = float(vals_xz.mean())
            max_raw_xz = int(mip_xz[b_xz].max())
        else:
            max_g_xz, p95_g_xz, mean_g_xz, max_raw_xz = 0.0, 0.0, 0.0, 0

        dead_visit = (ov_xy >= MIN_OVERLAP_PX_2D) and (ov_xz >= MIN_OVERLAP_PX_2D)
        dead_visit_dil10 = (ov10_xy >= MIN_OVERLAP_PX_2D) and (ov10_xz >= MIN_OVERLAP_PX_2D)

        rows.append(
            {
                "group_id": group_id,
                "experiment": exp,
                "condition": cond,
                "roi": roi,
                "roi_index": roi_idx,
                "visit": v_idx,
                "time_min": round(v_idx * VISIT_INTERVAL_MIN, 1),
                "stem": stem,
                "contact_area_um2": round(float(ca_um2), 4),
                "has_contact": bool(ca_um2 > 0.0),
                "target_xy_px": target_xy_px,
                "target_xz_px": target_xz_px,
                "dead_dye_xy_px": d_xy_px,
                "dead_dye_xz_px": d_xz_px,
                "overlap_xy_px": ov_xy,
                "overlap_xz_px": ov_xz,
                "overlap_dil10_xy_px": ov10_xy,
                "overlap_dil10_xz_px": ov10_xz,
                "overlap_xy_frac": round(frac_xy, 4),
                "overlap_xz_frac": round(frac_xz, 4),
                "max_raw405_in_target_xy": max_raw_xy,
                "max_raw405_in_target_xz": max_raw_xz,
                "max_gauss405_in_target_xy": round(max_g_xy, 2),
                "max_gauss405_in_target_xz": round(max_g_xz, 2),
                "p95_gauss405_in_target_xy": round(p95_g_xy, 2),
                "p95_gauss405_in_target_xz": round(p95_g_xz, 2),
                "mean_gauss405_in_target_xy": round(mean_g_xy, 2),
                "mean_gauss405_in_target_xz": round(mean_g_xz, 2),
                "bg_median_gauss405_xy": round(bg_xy, 2),
                "bg_median_gauss405_xz": round(bg_xz, 2),
                "dead_visit": bool(dead_visit),
                "dead_visit_dil10": bool(dead_visit_dil10),
            }
        )
    return rows


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact 95% Clopper-Pearson binomial confidence interval for k successes in n trials."""
    if n <= 0:
        return (np.nan, np.nan)
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def load_paper_chamber_mapping() -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load paper-row to chamber mapping from `results/chamber_to_row_map.csv` across all 13 groups.
    Records both `in_paper` (any row in the 419-row mapping) and `in_paper_firm_probable`
    (rows with `label` in `{"firm", "probable"}`).
    """
    df_map = pd.read_csv(ROW_MAP_CSV, dtype={"group_id": str})

    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df_map.iterrows():
        key = (str(row["experiment"]), str(row["roi"]))
        lbl = str(row["label"])
        mapping[key] = {
            "in_paper": True,
            "in_paper_firm_probable": lbl in ("firm", "probable"),
            "paper_cell_index": int(row["cell_index"]),
            "paper_CA_um2": float(row["paper_CA_um2"]),
            "alignment_label": lbl,
        }
    return mapping


def discover_chamber_visit_items(
    exp: str,
    roi: str,
    n_visits_s3: int,
    v_ca_map: dict[int, float],
) -> list[tuple[int, str, Path, Path, Path, float]]:
    """
    Discover valid `(visit_index, stem, xy_path, xz_path, pred_path, ca_um2)` items
    directly from `data/scidb_19235_mips/<exp>/<roi>/` and `data/scidb_19235_masks/<exp>/<roi>/`,
    handling 20, 21, or 22 visits and truncated chambers (such as `v07_6_39_120_0-6_hd9/MIP_ROI_36`
    where predictions end at visit 14).
    """
    roi_mip_dir = MIPS_DIR / exp / roi
    roi_mask_dir = MASKS_DIR / exp / roi

    xy_files = sorted(
        roi_mip_dir.glob("*_MIPxy_405.tif"),
        key=lambda p: int(p.name.split("_")[0]) if p.name.split("_")[0].isdigit() else p.name,
    )
    items: list[tuple[int, str, Path, Path, Path, float]] = []
    for idx0, xy_p in enumerate(xy_files):
        v_idx = idx0 + 1
        if v_idx > n_visits_s3:
            break
        stem = xy_p.name[: -len("_MIPxy_405.tif")]
        xz_p = roi_mip_dir / f"{stem}_MIPxz_405.tif"
        pred_p = roi_mask_dir / f"{stem}_predictions.h5"
        if not xz_p.exists() or not pred_p.exists():
            break
        ca_val = v_ca_map.get(v_idx, 0.0)
        items.append((v_idx, stem.zfill(5), xy_p, xz_p, pred_p, ca_val))
    return items


def build_chamber_and_visit_tables(
    experiments: tuple[tuple[str, str, str], ...] = ALL_EXPERIMENTS,
    workers: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-visit and per-chamber death labels across the specified experiments
    (up to all 13 experiments, 502 chambers, 10,510 visits), plus the per-group comparison
    table against the paper (`Source_Data Figure 5g`).
    """
    paper_map = load_paper_chamber_mapping()
    df_paper = pd.read_csv(CLEAN_419_CSV, dtype={"group_id": str})
    df_grp_inv = pd.read_csv(GROUP_INV_CSV, dtype={"group_id": str})

    # Index step3 per-chamber contact area series
    ca_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    tasks: list[tuple[str, str, str, str, list[tuple[int, str, Path, Path, Path, float]]]] = []

    for gid, exp, cond in experiments:
        ch_csv = STEP3_DIR / exp / "per_chamber_contact_area.csv"
        df_ch = pd.read_csv(ch_csv)
        df_ch = df_ch.sort_values("roi_index").reset_index(drop=True)
        for _, row in df_ch.iterrows():
            roi = str(row["roi"])
            n_v = int(row["n_visits"])
            v_ca: dict[int, float] = {}
            for v in range(1, n_v + 1):
                col = f"visit_{v:02d}_CA_um2"
                v_ca[v] = float(row[col]) if col in row and pd.notna(row[col]) else 0.0
            ca_lookup[(exp, roi)] = {
                "CA_um2": float(row["CA_um2"]),
                "n_visits": n_v,
                "n_contact_visits": int(row["n_contact_visits"]),
                "max_tcell_instances": int(row["max_tcell_instances"]),
                "multi_tcell_sweeps": int(row["multi_tcell_sweeps"]),
                "visit_ca": v_ca,
            }
            visit_items = discover_chamber_visit_items(exp, roi, n_v, v_ca)
            tasks.append((gid, exp, cond, roi, visit_items))

    visit_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 2))) as executor:
        for res in executor.map(_process_one_chamber, tasks):
            visit_rows.extend(res)

    df_visits = pd.DataFrame(visit_rows)
    df_visits["_grp_order"] = df_visits["group_id"].map(
        lambda g: GROUP_ORDER_MAP.get(str(g), 999)
    )
    df_visits = (
        df_visits.sort_values(["_grp_order", "roi_index", "visit"])
        .drop(columns=["_grp_order"])
        .reset_index(drop=True)
    )

    # Experiment-level nominal max visits (to identify truncated chambers)
    exp_max_visits: dict[str, int] = (
        df_visits.groupby("experiment")["visit"].max().astype(int).to_dict()
    )

    # Build per-chamber summary table
    chamber_rows: list[dict[str, Any]] = []
    for (gid, exp, cond, roi, roi_idx), grp in df_visits.groupby(
        ["group_id", "experiment", "condition", "roi", "roi_index"], sort=False
    ):
        grp = grp.sort_values("visit").reset_index(drop=True)
        n_visits = len(grp)
        last_observed_visit = int(grp["visit"].iloc[-1])
        last_observed_time_min = round(float(last_observed_visit * VISIT_INTERVAL_MIN), 1)
        is_truncated = bool(last_observed_visit < exp_max_visits.get(str(exp), last_observed_visit))

        pos = grp["dead_visit"].to_numpy(dtype=bool)
        pos_dil10 = grp["dead_visit_dil10"].to_numpy(dtype=bool)
        has_c = grp["has_contact"].to_numpy(dtype=bool)

        # First contact visit and time
        c_indices = np.where(has_c)[0]
        if c_indices.size > 0:
            first_contact_visit: int | None = int(grp.loc[c_indices[0], "visit"])
            first_contact_time_min: float | None = float(first_contact_visit * VISIT_INTERVAL_MIN)
            no_contact = False
        else:
            first_contact_visit = None
            first_contact_time_min = None
            no_contact = True

        # Sustained positive death_visit (>= 2 consecutive positive visits)
        sust_idx: int | None = None
        for i in range(n_visits - 1):
            if pos[i] and pos[i + 1]:
                sust_idx = i
                break

        ever_dead = sust_idx is not None
        if sust_idx is not None:
            death_visit: int | None = int(grp.loc[sust_idx, "visit"])
            death_time_min: float | None = float(death_visit * VISIT_INTERVAL_MIN)
        else:
            death_visit = None
            death_time_min = None

        # Single-visit (any visit) positive and 10-px dilated sustained positive
        any_pos_idx = np.where(pos)[0]
        ever_dead_any_visit = bool(any_pos_idx.size > 0)
        first_any_dead_visit: int | None = (
            int(grp.loc[any_pos_idx[0], "visit"]) if ever_dead_any_visit else None
        )
        first_any_dead_time_min: float | None = (
            float(first_any_dead_visit * VISIT_INTERVAL_MIN)
            if first_any_dead_visit is not None
            else None
        )

        sust_dil10_idx: int | None = None
        for i in range(n_visits - 1):
            if pos_dil10[i] and pos_dil10[i + 1]:
                sust_dil10_idx = i
                break
        ever_dead_dil10 = sust_dil10_idx is not None

        # Dead at start (visit 1) or before first contact
        dead_at_visit_1 = bool(pos[0])
        dead_before_contact = bool(
            (death_visit is not None and first_contact_visit is not None and death_visit < first_contact_visit)
            or dead_at_visit_1
        )

        if death_time_min is not None and first_contact_time_min is not None:
            contact_to_death_min: float | None = round(death_time_min - first_contact_time_min, 1)
        else:
            contact_to_death_min = None

        if first_any_dead_time_min is not None and first_contact_time_min is not None:
            any_contact_to_death_min: float | None = round(
                first_any_dead_time_min - first_contact_time_min, 1
            )
        else:
            any_contact_to_death_min = None

        # Censoring and survival time from first contact (censored at last observed visit)
        censored = not ever_dead
        censored_any_visit = not ever_dead_any_visit
        if first_contact_time_min is not None:
            censor_from_contact = round(last_observed_time_min - first_contact_time_min, 1)
            if ever_dead and contact_to_death_min is not None:
                km_contact_time_min: float | None = max(0.0, float(contact_to_death_min))
                km_event_sustained: int | None = 1
            else:
                km_contact_time_min = max(0.0, float(censor_from_contact))
                km_event_sustained = 0

            if ever_dead_any_visit and any_contact_to_death_min is not None:
                km_any_contact_time_min: float | None = max(0.0, float(any_contact_to_death_min))
                km_event_any_visit: int | None = 1
            else:
                km_any_contact_time_min = max(0.0, float(censor_from_contact))
                km_event_any_visit = 0
        else:
            km_contact_time_min = None
            km_event_sustained = None
            km_any_contact_time_min = None
            km_event_any_visit = None

        ca_info = ca_lookup[(exp, roi)]
        p_info = paper_map.get(
            (exp, roi),
            {
                "in_paper": False,
                "in_paper_firm_probable": False,
                "paper_cell_index": None,
                "paper_CA_um2": np.nan,
                "alignment_label": "excluded",
            },
        )

        min_both_series = np.minimum(
            grp["max_gauss405_in_target_xy"].to_numpy(),
            grp["max_gauss405_in_target_xz"].to_numpy(),
        )

        chamber_rows.append(
            {
                "group_id": gid,
                "experiment": exp,
                "condition": cond,
                "roi": roi,
                "roi_index": int(roi_idx),
                "in_paper": bool(p_info["in_paper"]),
                "in_paper_firm_probable": bool(p_info["in_paper_firm_probable"]),
                "paper_cell_index": p_info["paper_cell_index"],
                "alignment_label": p_info["alignment_label"],
                "paper_CA_um2": p_info["paper_CA_um2"],
                "our_CA_um2": round(ca_info["CA_um2"], 4),
                "n_visits": n_visits,
                "last_observed_visit": last_observed_visit,
                "last_observed_time_min": last_observed_time_min,
                "is_truncated": is_truncated,
                "n_contact_visits": ca_info["n_contact_visits"],
                "max_tcell_instances": ca_info["max_tcell_instances"],
                "multi_tcell_sweeps": ca_info["multi_tcell_sweeps"],
                "no_contact": no_contact,
                "first_contact_visit": first_contact_visit,
                "first_contact_time_min": first_contact_time_min,
                "ever_dead": ever_dead,
                "death_visit": death_visit,
                "death_time_min": death_time_min,
                "contact_to_death_min": contact_to_death_min,
                "censored": censored,
                "km_contact_time_min": km_contact_time_min,
                "km_event_sustained": km_event_sustained,
                "dead_at_visit_1": dead_at_visit_1,
                "dead_before_contact": dead_before_contact,
                "n_dead_visits": int(pos.sum()),
                "ever_dead_any_visit": ever_dead_any_visit,
                "first_any_dead_visit": first_any_dead_visit,
                "first_any_dead_time_min": first_any_dead_time_min,
                "any_contact_to_death_min": any_contact_to_death_min,
                "censored_any_visit": censored_any_visit,
                "km_any_contact_time_min": km_any_contact_time_min,
                "km_event_any_visit": km_event_any_visit,
                "ever_dead_dil10": ever_dead_dil10,
                "peak_overlap_xy_px": int(grp["overlap_xy_px"].max()),
                "peak_overlap_xz_px": int(grp["overlap_xz_px"].max()),
                "peak_gauss405_xy": round(float(grp["max_gauss405_in_target_xy"].max()), 2),
                "peak_gauss405_xz": round(float(grp["max_gauss405_in_target_xz"].max()), 2),
                "peak_min_xy_xz_gauss405": round(float(min_both_series.max()), 2),
            }
        )

    df_chambers = pd.DataFrame(chamber_rows)
    df_chambers["_grp_order"] = df_chambers["group_id"].map(
        lambda g: GROUP_ORDER_MAP.get(str(g), 999)
    )
    df_chambers = (
        df_chambers.sort_values(["_grp_order", "roi_index"])
        .drop(columns=["_grp_order"])
        .reset_index(drop=True)
    )

    # Build per-group comparison table
    group_rows: list[dict[str, Any]] = []
    for gid, exp, cond in experiments:
        sub_all = df_chambers[df_chambers["group_id"] == gid]
        if sub_all.empty:
            continue
        sub_fp = sub_all[sub_all["in_paper_firm_probable"]]
        sub_pap_all419 = sub_all[sub_all["in_paper"]]

        inv_row = df_grp_inv[df_grp_inv["group_id"] == gid].iloc[0]
        pub_rate_419 = float(df_paper[df_paper["group_id"] == gid]["group_kill_rate"].iloc[0])
        pub_n_cells = int(inv_row["n_pairs_in_published_xlsx"])
        pub_n_killed = int(inv_row["n_killed_published"])
        pub_rate_exact = float(inv_row["kill_rate_published"])
        pub_ci_lo, pub_ci_hi = clopper_pearson_ci(pub_n_killed, pub_n_cells)

        n_all = len(sub_all)
        n_fp = len(sub_fp)
        n_pap_all419 = len(sub_pap_all419)

        # Primary sustained rule (>=2 consecutive visits) on firm/probable paper chambers & all chambers
        k_fp_sust = int(sub_fp["ever_dead"].sum())
        k_all_sust = int(sub_all["ever_dead"].sum())
        k_pap419_sust = int(sub_pap_all419["ever_dead"].sum())
        fp_sust_lo, fp_sust_hi = clopper_pearson_ci(k_fp_sust, n_fp)
        all_sust_lo, all_sust_hi = clopper_pearson_ci(k_all_sust, n_all)

        # Any-visit rule (>=1 visit, matching summary.m max over time)
        k_fp_any = int(sub_fp["ever_dead_any_visit"].sum())
        k_all_any = int(sub_all["ever_dead_any_visit"].sum())
        k_pap419_any = int(sub_pap_all419["ever_dead_any_visit"].sum())
        fp_any_lo, fp_any_hi = clopper_pearson_ci(k_fp_any, n_fp)
        all_any_lo, all_any_hi = clopper_pearson_ci(k_all_any, n_all)

        # Dilated 10-px sustained rule
        k_fp_dil10 = int(sub_fp["ever_dead_dil10"].sum())
        k_all_dil10 = int(sub_all["ever_dead_dil10"].sum())
        k_pap419_dil10 = int(sub_pap_all419["ever_dead_dil10"].sum())

        min_v = int(sub_all["n_visits"].min())
        max_v = int(sub_all["n_visits"].max())

        group_rows.append(
            {
                "group_id": gid,
                "experiment": exp,
                "condition": cond,
                "n_visits_per_chamber": max_v,
                "min_visits_per_chamber": min_v,
                "n_truncated_chambers": int(sub_all["is_truncated"].sum()),
                "paper_n_cells": pub_n_cells,
                "paper_n_killed": pub_n_killed,
                "paper_group_kill_rate": pub_rate_419,
                "paper_group_kill_rate_exact": round(pub_rate_exact, 5),
                "paper_kill_rate_ci95_lo": round(pub_ci_lo, 4),
                "paper_kill_rate_ci95_hi": round(pub_ci_hi, 4),
                # Firm + probable paper chambers (requested in Step 4 all-chambers spec)
                "our_paper_chambers_n": n_fp,
                "our_paper_killed_sustained": k_fp_sust,
                "our_paper_kill_rate_sustained": round(k_fp_sust / n_fp, 4) if n_fp > 0 else np.nan,
                "our_paper_sustained_ci95_lo": round(fp_sust_lo, 4),
                "our_paper_sustained_ci95_hi": round(fp_sust_hi, 4),
                "our_paper_killed_any_visit": k_fp_any,
                "our_paper_kill_rate_any_visit": round(k_fp_any / n_fp, 4) if n_fp > 0 else np.nan,
                "our_paper_any_visit_ci95_lo": round(fp_any_lo, 4),
                "our_paper_any_visit_ci95_hi": round(fp_any_hi, 4),
                "our_paper_killed_dil10_sustained": k_fp_dil10,
                "our_paper_kill_rate_dil10_sustained": round(k_fp_dil10 / n_fp, 4) if n_fp > 0 else np.nan,
                # All 419 mapped rows (including unmatched/ambiguous) for reference
                "our_paper_all419_n": n_pap_all419,
                "our_paper_all419_killed_sustained": k_pap419_sust,
                "our_paper_all419_kill_rate_sustained": round(k_pap419_sust / n_pap_all419, 4)
                if n_pap_all419 > 0
                else np.nan,
                "our_paper_all419_killed_any_visit": k_pap419_any,
                "our_paper_all419_kill_rate_any_visit": round(k_pap419_any / n_pap_all419, 4)
                if n_pap_all419 > 0
                else np.nan,
                "our_paper_all419_killed_dil10_sustained": k_pap419_dil10,
                # All chambers in deposit
                "our_all_chambers_n": n_all,
                "our_all_killed_sustained": k_all_sust,
                "our_all_kill_rate_sustained": round(k_all_sust / n_all, 4),
                "our_all_sustained_ci95_lo": round(all_sust_lo, 4),
                "our_all_sustained_ci95_hi": round(all_sust_hi, 4),
                "our_all_killed_any_visit": k_all_any,
                "our_all_kill_rate_any_visit": round(k_all_any / n_all, 4),
                "our_all_any_visit_ci95_lo": round(all_any_lo, 4),
                "our_all_any_visit_ci95_hi": round(all_any_hi, 4),
                "our_all_killed_dil10_sustained": k_all_dil10,
                "our_all_kill_rate_dil10_sustained": round(k_all_dil10 / n_all, 4),
                # Special flags
                "n_dead_at_visit_1_all": int(sub_all["dead_at_visit_1"].sum()),
                "n_dead_at_visit_1_paper": int(sub_fp["dead_at_visit_1"].sum()),
                "n_dead_before_contact_all": int(sub_all["dead_before_contact"].sum()),
                "n_dead_before_contact_paper": int(sub_fp["dead_before_contact"].sum()),
                "n_no_contact_all": int(sub_all["no_contact"].sum()),
                "n_no_contact_paper": int(sub_fp["no_contact"].sum()),
                "n_censored_all": int(sub_all["censored"].sum()),
                "n_censored_paper": int(sub_fp["censored"].sum()),
                "median_death_time_min_paper": float(sub_fp["death_time_min"].dropna().median())
                if sub_fp["death_time_min"].notna().any()
                else np.nan,
                "median_contact_to_death_min_paper": float(
                    sub_fp["contact_to_death_min"].dropna().median()
                )
                if sub_fp["contact_to_death_min"].notna().any()
                else np.nan,
                "median_contact_to_death_min_all": float(
                    sub_all["contact_to_death_min"].dropna().median()
                )
                if sub_all["contact_to_death_min"].notna().any()
                else np.nan,
            }
        )

    df_groups = pd.DataFrame(group_rows)
    return df_visits, df_chambers, df_groups


def make_pilot_figure(
    df_visits: pd.DataFrame,
    df_chambers: pd.DataFrame,
    df_groups: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Task 5: Create `figures/step4_death_labels_pilot.png` following House Style:
      (a) Example chamber (`v02_6_40_120_0-6_hd3 / MIP_ROI_13`): xy dead-dye MIP with
          projected target mask outline (`#C0561F`) and cleaned dead-dye contour (`#1F4E79`)
          at a pre-death visit (Visit 2, 20 min) and a post-death visit (Visit 10, 100 min).
      (b) Per-chamber timeline raster across all 120 chambers (3 panels of 40 rows x 20 visits):
          state colors for No contact (`#F6F6F6`), Contact (`#8FBCE6`), and Dead (`#C0561F`),
          with paper-included chambers vs excluded chambers and `dead_at_visit_1` / `no_contact`
          markers.
      (c) Our kill fraction vs published `group_kill_rate` for groups 1-3 (showing Paper
          published rate, Our rule on paper chambers, Our rule on all 40 chambers, and the
          single-visit max-over-time variant).
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "DejaVu Sans",
            ],
            "axes.edgecolor": "#C9C9C9",
            "axes.linewidth": 0.8,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    fig = plt.figure(figsize=(13.5, 8.2), dpi=200)
    gs_top = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.15, 1.0],
        hspace=0.34,
        wspace=0.25,
        left=0.055,
        right=0.975,
        top=0.93,
        bottom=0.08,
    )

    # -------------------------------------------------------------------------
    # Panel (a): Pre-death vs Post-death xy dead-dye MIP with target mask outline
    # Example chamber: Group 2 (v02_6_40_120_0-6_hd3), MIP_ROI_13 (firm paper match,
    # CA = 200.3 um^2, contact from visit 1, death_visit = 4 [40 min]).
    # -------------------------------------------------------------------------
    gs_a = gs_top[0, 0].subgridspec(1, 2, wspace=0.12)
    exp_a = "v02_6_40_120_0-6_hd3"
    roi_a = "MIP_ROI_13"
    sub_a = df_visits[(df_visits["experiment"] == exp_a) & (df_visits["roi"] == roi_a)].sort_values("visit")

    v_pre, v_post = 2, 10
    row_pre = sub_a[sub_a["visit"] == v_pre].iloc[0]
    row_post = sub_a[sub_a["visit"] == v_post].iloc[0]

    examples = [
        (v_pre, str(row_pre["stem"]).zfill(5), f"Pre-death (visit {v_pre}, {v_pre*10} min)"),
        (v_post, str(row_post["stem"]).zfill(5), f"Post-death (visit {v_post}, {v_post*10} min)"),
    ]

    # Compute common crop box around target in MIP_ROI_13
    crop_masks = []
    loaded_example_data = []
    for v_num, stem, title_str in examples:
        xy_p = MIPS_DIR / exp_a / roi_a / f"{stem}_MIPxy_405.tif"
        pred_p = MASKS_DIR / exp_a / roi_a / f"{stem}_predictions.h5"
        mip_xy = tifffile.imread(xy_p)
        with h5py.File(pred_p, "r") as hf:
            pred = hf["predictions"][:]
        b_xy = ((pred == 2) | (pred == 4)).max(axis=0).T
        t_xy = ((pred == 1) | (pred == 3)).max(axis=0).T
        d_xy, g_xy = compute_2d_dead_mask(mip_xy, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        crop_masks.append(b_xy | t_xy | d_xy)
        loaded_example_data.append((v_num, title_str, mip_xy, g_xy, b_xy, t_xy, d_xy))

    union_mask = crop_masks[0] | crop_masks[1]
    ys, xs = np.where(union_mask)
    pad = 24
    y0, y1 = max(0, int(ys.min()) - pad), min(256, int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(256, int(xs.max()) + pad)
    # Make square crop
    side = max(y1 - y0, x1 - x0, 120)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(256 - side, cy - side // 2))
    y1 = y0 + side
    x0 = max(0, min(256 - side, cx - side // 2))
    x1 = x0 + side

    vmin_adu, vmax_adu = 395.0, 445.0
    for idx_panel, (v_num, title_str, mip_xy, g_xy, b_xy, t_xy, d_xy) in enumerate(loaded_example_data):
        ax = fig.add_subplot(gs_a[0, idx_panel])
        sub_img = mip_xy[y0:y1, x0:x1]
        sub_b = b_xy[y0:y1, x0:x1]
        sub_t = t_xy[y0:y1, x0:x1]
        sub_d = d_xy[y0:y1, x0:x1]
        ov_px = int((b_xy & d_xy).sum())
        max_g = float(g_xy[b_xy].max()) if np.any(b_xy) else 0.0

        ax.imshow(sub_img, cmap="Greys_r", vmin=vmin_adu, vmax=vmax_adu, interpolation="nearest")
        if np.any(sub_b):
            ax.contour(
                sub_b.astype(float),
                levels=[0.5],
                colors=["#C0561F"],
                linewidths=1.8,
            )
        if np.any(sub_t):
            ax.contour(
                sub_t.astype(float),
                levels=[0.5],
                colors=["#2E7D5B"],
                linewidths=1.1,
                linestyles="--",
            )
        if np.any(sub_d):
            ax.contour(
                sub_d.astype(float),
                levels=[0.5],
                colors=["#7FB3FF"],
                linewidths=1.4,
            )

        ax.set_title(title_str, fontsize=10.5, fontweight="bold", color="#111111", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.04,
            0.05,
            f"Target overlap: {ov_px} px\nMax smoothed: {max_g:.1f} ADU",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#FFFFFF",
            bbox={"facecolor": "#111111", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        )
        if idx_panel == 0:
            ax.text(
                -0.08,
                1.12,
                "a  405-nm dead-dye MIPxy & target outline (Group 2, ROI 13)",
                transform=ax.transAxes,
                fontsize=11.5,
                fontweight="bold",
                color="#111111",
            )
            # Scale bar: 10 um = 10 / 0.1962 = 51.0 px
            bar_px = 10.0 / 0.1962
            bx0 = side - bar_px - 10
            by0 = side - 10
            ax.plot([bx0, bx0 + bar_px], [by0, by0], color="#FFFFFF", lw=2.2)
            ax.text(
                bx0 + bar_px / 2,
                by0 - 5,
                "10 µm",
                color="#FFFFFF",
                fontsize=8,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:
            legend_handles = [
                Patch(edgecolor="#C0561F", facecolor="none", linewidth=1.8, label="Target mask (2|4)"),
                Patch(edgecolor="#2E7D5B", facecolor="none", linewidth=1.2, linestyle="--", label="T-cell mask (1|3)"),
                Patch(edgecolor="#7FB3FF", facecolor="none", linewidth=1.5, label="Dead dye (>404 ADU)"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                facecolor="#111111",
                edgecolor="none",
                labelcolor="#FFFFFF",
                fontsize=7.5,
                handlelength=1.4,
            )

    # -------------------------------------------------------------------------
    # Panel (c): Group kill fraction comparison (Groups 1-3)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs_top[0, 1])
    x = np.arange(len(df_groups))
    w = 0.20

    r_pub = df_groups["paper_group_kill_rate"].to_numpy() * 100.0
    r_sust_pap = df_groups["our_paper_kill_rate_sustained"].to_numpy() * 100.0
    r_sust_all = df_groups["our_all_kill_rate_sustained"].to_numpy() * 100.0
    r_any_pap = df_groups["our_paper_kill_rate_any_visit"].to_numpy() * 100.0

    b1 = ax_c.bar(
        x - 1.5 * w,
        r_pub,
        width=w,
        color="#111111",
        edgecolor="#111111",
        label="Paper group_kill_rate (Fig. 5g)",
    )
    b2 = ax_c.bar(
        x - 0.5 * w,
        r_sust_pap,
        width=w,
        color="#1F4E79",
        edgecolor="#111111",
        linewidth=0.6,
        label="Our sustained rule (paper chambers)",
    )
    b3 = ax_c.bar(
        x + 0.5 * w,
        r_sust_all,
        width=w,
        color="#C0561F",
        edgecolor="#111111",
        linewidth=0.6,
        label="Our sustained rule (all 40 chambers)",
    )
    b4 = ax_c.bar(
        x + 1.5 * w,
        r_any_pap,
        width=w,
        color="#2E7D5B",
        edgecolor="#111111",
        linewidth=0.6,
        label="Single-visit max rule (paper chambers)",
    )

    for bars in (b1, b2, b3, b4):
        for rect in bars:
            h = rect.get_height()
            ax_c.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#1F1F1F",
            )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [
            f"Group 1\n(hd1, n={df_groups.loc[0, 'paper_n_cells']}/40)",
            f"Group 2\n(hd3, n={df_groups.loc[1, 'paper_n_cells']}/40)",
            f"Group 3\n(hd4, n={df_groups.loc[2, 'paper_n_cells']}/40)",
        ],
        fontsize=9.5,
    )
    ax_c.set_ylabel("Target kill percentage (%)", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0, 108)
    ax_c.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    ax_c.set_axisbelow(True)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.8,
        ncol=2,
    )
    ax_c.text(
        -0.08,
        1.12,
        "c  Kill fraction vs. published group_kill_rate (Groups 1–3)",
        transform=ax_c.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
    )

    # -------------------------------------------------------------------------
    # Panel (b): Per-chamber timeline raster (rows = chambers, columns = visits)
    # 0 = No contact (#F6F6F6), 1 = Contact (#8FBCE6), 2 = Dead (#C0561F)
    # -------------------------------------------------------------------------
    gs_b = gs_top[1, :].subgridspec(1, 3, wspace=0.18)
    cmap_raster = ListedColormap(["#F6F6F6", "#8FBCE6", "#C0561F"])

    for idx_g, (gid, exp, _cond) in enumerate(PILOT_EXPERIMENTS):
        ax_b = fig.add_subplot(gs_b[0, idx_g])
        sub_ch = df_chambers[df_chambers["group_id"].astype(str) == str(gid)].sort_values("roi_index").reset_index(drop=True)
        sub_v = df_visits[df_visits["group_id"].astype(str) == str(gid)]

        n_ch = len(sub_ch)
        n_vis = 20
        mat = np.zeros((n_ch, n_vis), dtype=int)

        for r_i, ch_row in sub_ch.iterrows():
            roi = str(ch_row["roi"])
            v_rows = sub_v[sub_v["roi"] == roi].sort_values("visit")
            for _, vr in v_rows.iterrows():
                v_i = int(vr["visit"]) - 1
                if 0 <= v_i < n_vis:
                    if bool(vr["dead_visit"]):
                        mat[r_i, v_i] = 2
                    elif bool(vr["has_contact"]):
                        mat[r_i, v_i] = 1
                    else:
                        mat[r_i, v_i] = 0

        ax_b.imshow(
            mat,
            aspect="auto",
            cmap=cmap_raster,
            vmin=0,
            vmax=2,
            interpolation="nearest",
            extent=[0.5, n_vis + 0.5, n_ch - 0.5, -0.5],
        )

        # Overlay marker for sustained death_visit (black dot) and flag excluded/special chambers
        for r_i, ch_row in sub_ch.iterrows():
            dv = ch_row["death_visit"]
            if pd.notna(dv):
                ax_b.plot(int(dv), r_i, marker="o", markersize=3.2, color="#111111")
            if bool(ch_row["dead_at_visit_1"]):
                ax_b.plot(1, r_i, marker="s", markersize=4.2, markerfacecolor="none", markeredgecolor="#B3261E", markeredgewidth=1.2)

        # Horizontal hairlines every 5 rows
        for y_line in range(5, n_ch, 5):
            ax_b.axhline(y_line - 0.5, color="#E3E3E3", linewidth=0.5)

        ytick_pos = list(range(0, n_ch, 5)) + [n_ch - 1]
        ytick_lbls = []
        for y_i in ytick_pos:
            roi_idx = int(sub_ch.loc[y_i, "roi_index"])
            in_p = bool(sub_ch.loc[y_i, "in_paper"])
            no_c = bool(sub_ch.loc[y_i, "no_contact"])
            suffix = "" if in_p else "*"
            ytick_lbls.append(f"ROI {roi_idx}{suffix}")

        ax_b.set_yticks(ytick_pos)
        ax_b.set_yticklabels(ytick_lbls, fontsize=7.5)
        ax_b.set_xticks([1, 5, 10, 15, 20])
        ax_b.set_xticklabels(["1\n(10m)", "5\n(50m)", "10\n(100m)", "15\n(150m)", "20\n(200m)"], fontsize=8.0)
        ax_b.set_xlabel("Visit index (time)", fontsize=9.0)
        if idx_g == 0:
            ax_b.set_ylabel("Chamber (MIP_ROI_n)", fontsize=9.5, fontweight="bold")
            ax_b.text(
                -0.14,
                1.09,
                "b  Per-chamber contact & dead-dye timeline raster (40 chambers × 20 visits per group)",
                transform=ax_b.transAxes,
                fontsize=11.5,
                fontweight="bold",
                color="#111111",
            )

        k_sust = int(sub_ch["ever_dead"].sum())
        k_pap = int(sub_ch[sub_ch["in_paper"]]["ever_dead"].sum())
        n_pap = int(sub_ch["in_paper"].sum())
        ax_b.set_title(
            f"Group {gid} ({exp[:3]}): {k_pap}/{n_pap} paper ({k_sust}/40 all)",
            fontsize=10.0,
            fontweight="bold",
            pad=5,
        )

        if idx_g == 2:
            raster_handles = [
                Patch(facecolor="#F6F6F6", edgecolor="#C9C9C9", label="No contact"),
                Patch(facecolor="#8FBCE6", edgecolor="#C9C9C9", label="T-cell contact"),
                Patch(facecolor="#C0561F", edgecolor="#C9C9C9", label="Dead dye positive"),
            ]
            ax_b.legend(
                handles=raster_handles,
                loc="lower left",
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor="#C9C9C9",
                fontsize=7.5,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_km_curve(
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute standard product-limit Kaplan-Meier survival curve S(t) for right-censored data:
      - `times`: non-negative durations from first contact (min)
      - `events`: 1 if target died, 0 if censored at last observed visit
    Returns `(t_step, s_step, censor_times)` suitable for `ax.step(..., where='post')`.
    """
    valid = np.isfinite(times) & np.isfinite(events)
    t_arr = np.maximum(0.0, times[valid].astype(float))
    e_arr = events[valid].astype(int)
    if t_arr.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([])

    uniq_t = np.sort(np.unique(t_arr))
    t_pts = [0.0]
    s_pts = [1.0]
    s_cur = 1.0

    for t_val in uniq_t:
        n_risk = int(np.sum(t_arr >= t_val))
        d_i = int(np.sum((t_arr == t_val) & (e_arr == 1)))
        if n_risk > 0 and d_i > 0:
            s_cur = s_cur * (1.0 - d_i / float(n_risk))
        t_pts.append(float(t_val))
        s_pts.append(float(s_cur))

    censor_t = np.sort(np.unique(t_arr[e_arr == 0]))
    return np.array(t_pts), np.array(s_pts), censor_t


def make_all_figure(
    df_chambers: pd.DataFrame,
    df_groups: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Create `figures/step4_death_labels_all.png` following House Style:
      (a) Kill fraction vs paper per group across all 13 groups for both definitions
          (sustained >= 2 visits vs any-visit >= 1 visit) on paper chambers
          (firm + probable) and on all chambers, with 95% Clopper-Pearson binomial intervals.
      (b) Kaplan-Meier target survival curves of time from first contact to death per group
          (censored at last observed visit), distinguishing untreated high-killing (Groups 1-5),
          untreated low-killing (Groups 6-10), and dasatinib 50 nM (Groups 8*, 9*, 10*) groups.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "DejaVu Sans",
            ],
            "axes.edgecolor": "#C9C9C9",
            "axes.linewidth": 0.8,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    fig = plt.figure(figsize=(14.0, 9.5), dpi=200)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.12, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.20,
        left=0.055,
        right=0.975,
        top=0.885,
        bottom=0.075,
    )

    # -------------------------------------------------------------------------
    # Panel (a): Kill fraction vs published Figure 5g across all 13 groups
    #            with 95% Clopper-Pearson binomial intervals
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    n_g = len(df_groups)
    x = np.arange(n_g, dtype=float)

    r_pub = df_groups["paper_group_kill_rate_exact"].to_numpy() * 100.0
    pub_lo = df_groups["paper_kill_rate_ci95_lo"].to_numpy() * 100.0
    pub_hi = df_groups["paper_kill_rate_ci95_hi"].to_numpy() * 100.0

    r_sust_fp = df_groups["our_paper_kill_rate_sustained"].to_numpy() * 100.0
    sust_fp_lo = df_groups["our_paper_sustained_ci95_lo"].to_numpy() * 100.0
    sust_fp_hi = df_groups["our_paper_sustained_ci95_hi"].to_numpy() * 100.0

    r_any_fp = df_groups["our_paper_kill_rate_any_visit"].to_numpy() * 100.0
    any_fp_lo = df_groups["our_paper_any_visit_ci95_lo"].to_numpy() * 100.0
    any_fp_hi = df_groups["our_paper_any_visit_ci95_hi"].to_numpy() * 100.0

    r_sust_all = df_groups["our_all_kill_rate_sustained"].to_numpy() * 100.0
    sust_all_lo = df_groups["our_all_sustained_ci95_lo"].to_numpy() * 100.0
    sust_all_hi = df_groups["our_all_sustained_ci95_hi"].to_numpy() * 100.0

    r_any_all = df_groups["our_all_kill_rate_any_visit"].to_numpy() * 100.0
    any_all_lo = df_groups["our_all_any_visit_ci95_lo"].to_numpy() * 100.0
    any_all_hi = df_groups["our_all_any_visit_ci95_hi"].to_numpy() * 100.0

    # Offsets for 5 series per group
    dx = [-0.26, -0.13, 0.0, 0.13, 0.26]

    # 1. Paper Figure 5g
    ax_a.errorbar(
        x + dx[0],
        r_pub,
        yerr=[np.maximum(0.0, r_pub - pub_lo), np.maximum(0.0, pub_hi - r_pub)],
        fmt="s",
        markersize=5.8,
        color="#111111",
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        elinewidth=1.2,
        capsize=2.5,
        label="Paper Fig. 5g (curated chambers)",
        zorder=5,
    )
    # 2. Sustained (>=2 visits), paper firm+probable chambers
    ax_a.errorbar(
        x + dx[1],
        r_sust_fp,
        yerr=[np.maximum(0.0, r_sust_fp - sust_fp_lo), np.maximum(0.0, sust_fp_hi - r_sust_fp)],
        fmt="o",
        markersize=5.6,
        color="#1F4E79",
        markerfacecolor="#1F4E79",
        markeredgecolor="#1F4E79",
        elinewidth=1.2,
        capsize=2.5,
        label="Sustained (>=2 visits), paper firm+probable",
        zorder=4,
    )
    # 3. Sustained (>=2 visits), all chambers
    ax_a.errorbar(
        x + dx[2],
        r_sust_all,
        yerr=[np.maximum(0.0, r_sust_all - sust_all_lo), np.maximum(0.0, sust_all_hi - r_sust_all)],
        fmt="o",
        markersize=5.4,
        color="#1F4E79",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#1F4E79",
        markeredgewidth=1.3,
        elinewidth=1.0,
         linestyle="none",
        capsize=2.2,
        label="Sustained (>=2 visits), all chambers",
        zorder=3,
    )
    # 4. Any-visit (>=1 visit), paper firm+probable chambers
    ax_a.errorbar(
        x + dx[3],
        r_any_fp,
        yerr=[np.maximum(0.0, r_any_fp - any_fp_lo), np.maximum(0.0, any_fp_hi - r_any_fp)],
        fmt="^",
        markersize=5.8,
        color="#C0561F",
        markerfacecolor="#C0561F",
        markeredgecolor="#C0561F",
        elinewidth=1.2,
        capsize=2.5,
        label="Any visit (>=1 visit), paper firm+probable",
        zorder=4,
    )
    # 5. Any-visit (>=1 visit), all chambers
    ax_a.errorbar(
        x + dx[4],
        r_any_all,
        yerr=[np.maximum(0.0, r_any_all - any_all_lo), np.maximum(0.0, any_all_hi - r_any_all)],
        fmt="^",
        markersize=5.6,
        color="#C0561F",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#C0561F",
        markeredgewidth=1.3,
        elinewidth=1.0,
        linestyle="none",
        capsize=2.2,
        label="Any visit (>=1 visit), all chambers",
        zorder=3,
    )

    # Vertical condition dividers
    ax_a.axvline(4.5, color="#C9C9C9", linestyle="--", linewidth=0.9)
    ax_a.axvline(9.5, color="#C9C9C9", linestyle="--", linewidth=0.9)

    ax_a.text(2.0, 95.5, "Untreated high-killing (Groups 1–5)", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#1F1F1F")
    ax_a.text(7.0, 95.5, "Untreated low-killing (Groups 6–10)", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#1F1F1F")
    ax_a.text(11.0, 95.5, "Dasatinib 50 nM (8*–10*)", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#1F1F1F")

    xtick_lbls = []
    for _, r in df_groups.iterrows():
        gid = str(r["group_id"])
        exp_short = str(r["experiment"]).split("_")[-1]
        if exp_short == "ds":
            exp_short = "_".join(str(r["experiment"]).split("_")[-2:])
        n_fp = int(r["our_paper_chambers_n"])
        n_all = int(r["our_all_chambers_n"])
        xtick_lbls.append(f"G{gid}\n({exp_short})\n{n_fp}/{n_all}")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xtick_lbls, fontsize=8.2)
    ax_a.set_xlim(-0.6, n_g - 0.4)
    ax_a.set_ylim(-3, 102)
    ax_a.set_ylabel("Target kill fraction (%)", fontsize=10.0, fontweight="bold")
    ax_a.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=8.2,
        ncol=3,
    )
    ax_a.text(
        -0.04,
        1.19,
        "a  Kill fraction per group vs. Figure 5g (95% Clopper-Pearson intervals; ticks show firm+probable / all chambers)",
        transform=ax_a.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color="#111111",
    )

    # -------------------------------------------------------------------------
    # Panel (b): Kaplan-Meier curves of time from first contact to death
    #            Left: Sustained (>=2 visits) | Right: Any-visit (>=1 visit)
    # -------------------------------------------------------------------------
    km_specs = [
        (
            gs[1, 0],
            "km_contact_time_min",
            "km_event_sustained",
            "b  Time from first contact to death: sustained rule (>=2 visits)",
        ),
        (
            gs[1, 1],
            "km_any_contact_time_min",
            "km_event_any_visit",
            "c  Time from first contact to death: any-visit rule (>=1 visit)",
        ),
    ]

    for sub_spec, time_col, event_col, panel_title in km_specs:
        ax_km = fig.add_subplot(sub_spec)

        # Plot per-group curves
        for _, g_row in df_groups.iterrows():
            gid = str(g_row["group_id"])
            cond = str(g_row["condition"])
            sub_c = df_chambers[
                (df_chambers["group_id"].astype(str) == gid) & (~df_chambers["no_contact"])
            ]
            t_vals = sub_c[time_col].to_numpy(dtype=float)
            e_vals = sub_c[event_col].to_numpy(dtype=float)
            t_step, s_step, c_times = compute_km_curve(t_vals, e_vals)

            if "high-killing" in cond:
                color = "#1F4E79"
                ls = "-"
                lw = 1.15
                alpha = 0.65
            elif "low-killing" in cond:
                color = "#2E7D5B"
                ls = "-"
                lw = 1.15
                alpha = 0.65
            else:
                color = "#C0561F"
                ls = "--"
                lw = 1.35
                alpha = 0.85

            ax_km.step(t_step, s_step * 100.0, where="post", color=color, linestyle=ls, linewidth=lw, alpha=alpha)
            if t_step.size > 0:
                ax_km.plot(t_step[-1], s_step[-1] * 100.0, marker="|", markersize=5.5, color=color, alpha=alpha)

        # Also plot pooled condition KM curves in bold with direct right-margin labels
        cond_Order = [
            ("Untreated high-killing CAR-T", "High-killing\n(G1–G5)", "#1F4E79", "-", 2.5),
            ("Untreated low-killing CAR-T", "Low-killing\n(G6–G10)", "#2E7D5B", "-", 2.5),
            ("Dasatinib 50 nM CAR-T", "Dasatinib\n(G8*–G10*)", "#C0561F", "--", 2.5),
        ]
        for cond_key, end_lbl, color, ls, lw in cond_Order:
            sub_c = df_chambers[
                (df_chambers["condition"] == cond_key) & (~df_chambers["no_contact"])
            ]
            t_vals = sub_c[time_col].to_numpy(dtype=float)
            e_vals = sub_c[event_col].to_numpy(dtype=float)
            t_step, s_step, _ = compute_km_curve(t_vals, e_vals)
            ax_km.step(t_step, s_step * 100.0, where="post", color=color, linestyle=ls, linewidth=lw, zorder=5)
            if t_step.size > 0:
                ax_km.text(
                    t_step[-1] + 3.0,
                    s_step[-1] * 100.0,
                    end_lbl,
                    fontsize=7.6,
                    color=color,
                    va="center",
                    fontweight="bold",
                )

        legend_lines = [
            Line2D([0], [0], color="#1F4E79", lw=2.3, linestyle="-", label="Untreated high-killing (Groups 1–5; thin = group, bold = pooled)"),
            Line2D([0], [0], color="#2E7D5B", lw=2.3, linestyle="-", label="Untreated low-killing (Groups 6–10; thin = group, bold = pooled)"),
            Line2D([0], [0], color="#C0561F", lw=2.3, linestyle="--", label="Dasatinib 50 nM (Groups 8*–10*; thin = group, bold = pooled)"),
        ]
        ax_km.legend(
            handles=legend_lines,
            loc="lower left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#E3E3E3",
            fontsize=7.6,
        )
        ax_km.set_xlim(0, 242)
        ax_km.set_ylim(0, 105)
        ax_km.set_xlabel("Time from first T-cell contact (min)", fontsize=9.5, fontweight="bold")
        ax_km.set_ylabel("Target survival (%)", fontsize=9.5, fontweight="bold")
        ax_km.grid(axis="both", color="#E3E3E3", linewidth=0.6)
        ax_km.set_axisbelow(True)
        ax_km.spines["top"].set_visible(False)
        ax_km.spines["right"].set_visible(False)
        ax_km.text(
            -0.08,
            1.06,
            panel_title,
            transform=ax_km.transAxes,
            fontsize=10.8,
            fontweight="bold",
            color="#111111",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Step 4 v2: Contacted-Target Instance Rule, MIP_ROI_0 Artifact Exclusion,
# and Full-Recording vs Fixed 200-Minute Window (Visits 1-20)
# =============================================================================

ARTIFACT_PEAK_ADU: float = 5000.0
FIXED_WINDOW_MAX_VISIT: int = 20


def detect_mip_artifact_405(
    mip_xy: np.ndarray,
    mip_xz: np.ndarray,
    threshold_adu: float = THRESH_2D_ADU,
    peak_adu: float = ARTIFACT_PEAK_ADU,
) -> dict[str, Any]:
    """
    Documented, minimal detector for the late 405-nm stage-origin deskew artifact
    in `MIP_ROI_0`:
      - In `MIPxy_405`, the artifact is a full-height vertical saturation band anchored
        at the left border (`col = 0 .. W-1`, width W in [31, 121] px) where column 0
        has median intensity > 404 ADU (`573-784` ADU) and horizontal streaks peak at
        `14,296-25,317` raw ADU (whereas across all other 10,471 chamber-visits in the
        dataset, `max(MIPxy_405)` never exceeds 2,997 ADU).
      - In `MIPxz_405`, the same 30-degree oblique plane projects to a full-height left
        band `z = 0 .. Z0-1` (`Z0 approx W / sqrt(3)` in [19, 71] slices) with identical
        row maxima and reversed column profile `MIPxz_405[:, Z0-1-k] == MIPxy_405[:, k]`.
    """
    col0_median = float(np.median(mip_xy[:, 0]))
    max_xy = int(mip_xy.max())
    max_xz = int(mip_xz.max())
    art_cols_xy = int((mip_xy.min(axis=0) > threshold_adu).sum())
    art_cols_xz = int((mip_xz.min(axis=0) > threshold_adu).sum())
    is_artifact = bool((col0_median > threshold_adu) and (max_xy > peak_adu))
    return {
        "artifact_405_band": is_artifact,
        "mip_xy_max_raw405": max_xy,
        "mip_xz_max_raw405": max_xz,
        "mip_xy_col0_median_raw405": round(col0_median, 1),
        "mip_xy_artifact_cols_n": art_cols_xy,
        "mip_xz_artifact_cols_n": art_cols_xz,
    }


def _compute_contacted_b_ids_for_sweep(
    I: np.ndarray,
    b_sub: np.ndarray,
    sl_b: tuple[slice, slice, slice],
    trace_num: int,
    tracked_id: int,
    candidate_bids: list[int],
) -> tuple[list[int], list[int]]:
    """
    Given a 3D prediction volume `I`, cropped target instance array `b_sub` on `sl_b`,
    and `candidate_bids` (`1..max(b_sub)`), compute the exact 3D contact surface mask
    (`step2_instance_seg.m` + `step3_analysis.m`) and return:
      - `contacted_voronoi`: candidate `bid`s whose target instance or Voronoi-expanded
        membrane (`(map_exp == 3) & (b_inst[idx_xyz] == bid)`) touches `s_mask` within 1 voxel
      - `contacted_strict`: candidate `bid`s whose unexpanded `b_instance == bid` touches
        `s_mask` within 1 voxel (`s_mask & imdilate(b_instance == bid, strel('sphere', 1))`)
    """
    b_inst = np.zeros(I.shape, dtype=np.int32)
    b_inst[sl_b] = b_sub
    b_mask = ((b_inst > 0) & (I == 4)).astype(np.uint8)

    t_cell = I == 1
    t_actin = I == 3
    if np.any(t_cell) or np.any(t_actin):
        sl_t = bbox_slices(t_cell | t_actin, 2, I.shape)
        t_sub = instance_seg_matlab_xyz(t_cell[sl_t], t_actin[sl_t])
        t_inst = np.zeros(I.shape, dtype=np.int32)
        t_inst[sl_t] = t_sub
    else:
        return [], []

    t_act = ((t_inst > 0) & (I == 3)).astype(np.uint8)
    bg = (I == 0).astype(np.uint8)
    map_origin = b_mask * 3 + t_act * 2 + bg
    idx_xyz = edt_indices_matlab_xyz(map_origin == 0)
    map_exp = map_origin[idx_xyz]
    surface_mask = (map_exp == 2) & binary_dilation(map_exp == 3, structure=SPHERE1)
    surface = t_inst[idx_xyz] * surface_mask.astype(np.int32)

    if trace_num == 1:
        s_mask = surface == tracked_id
    else:
        s_mask = surface > 0

    if not np.any(s_mask):
        return [], []

    b_nearest = b_inst[idx_xyz]
    contacted_voronoi: list[int] = []
    contacted_strict: list[int] = []
    for bid in candidate_bids:
        bi = b_inst == bid
        if np.any(s_mask & binary_dilation(bi, structure=SPHERE1)):
            contacted_strict.append(bid)
            contacted_voronoi.append(bid)
        else:
            bi_vor = bi | ((map_exp == 3) & (b_nearest == bid))
            if np.any(s_mask & binary_dilation(bi_vor, structure=SPHERE1)):
                contacted_voronoi.append(bid)
    return contacted_voronoi, contacted_strict


def _process_one_chamber_v2(
    args: tuple[
        str,
        str,
        str,
        str,
        int,
        list[tuple[int, str, Path, Path, Path, float, float, list[tuple[int, Path, int, int, int]]]],
    ],
) -> list[dict[str, Any]]:
    """
    Evaluate all visits of one chamber under the Step 4 v2 pipeline:
      1. Detect the `MIP_ROI_0` late 405-nm artifact (`detect_mip_artifact_405`).
      2. Compute the old v1 all-target 2D projection and death label for comparison.
      3. Identify the target instance(s) (`b_instance`) contacted by the tracked T cell
         (Track 1 for `trace_num == 1`; any touching T cell for `trace_num != 1`, flagged
         with `non_single_track_flag = True`) across sweeps 1..5 of the visit, project
         only the contacted target instance mask to `xy` and `xz`, and evaluate the
         2D `xy`+`xz` dead-dye overlap rule (`THRESH_2D_ADU = 404.0`, `MIN_OVERLAP_PX_2D = 20`).
    """
    group_id, exp, cond, roi, trace_num, visit_items = args
    roi_idx = int(roi.split("_")[-1]) if roi.split("_")[-1].isdigit() else -1
    non_single_track_flag = bool(trace_num != 1)
    rows: list[dict[str, Any]] = []

    for (
        v_idx,
        stem,
        xy_path,
        xz_path,
        pred_path_s1,
        ca_tracked_um2,
        ca_all_um2,
        sweep_specs,
    ) in visit_items:
        mip_xy = tifffile.imread(xy_path)
        mip_xz = tifffile.imread(xz_path)
        with h5py.File(pred_path_s1, "r") as hf:
            pred_s1 = hf["predictions"][:]

        # 1. Artifact detection on raw 405-nm MIPs
        art_info = detect_mip_artifact_405(mip_xy, mip_xz, THRESH_2D_ADU, ARTIFACT_PEAK_ADU)
        is_artifact = bool(art_info["artifact_405_band"])

        # 2. Old v1 all-target projection mask (`(pred == 2) | (pred == 4)` on Sweep 1)
        raw_b_s1 = (pred_s1 == 2) | (pred_s1 == 4)
        b_all_xy = raw_b_s1.max(axis=0).T
        b_all_xz = raw_b_s1.max(axis=1).T

        d_xy, g_xy = compute_2d_dead_mask(mip_xy, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xz, g_xz = compute_2d_dead_mask(mip_xz, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xy_dil10 = dilate_2d_edt(d_xy, 10.0)
        d_xz_dil10 = dilate_2d_edt(d_xz, 10.0)

        target_all_xy_px = int(b_all_xy.sum())
        target_all_xz_px = int(b_all_xz.sum())
        d_xy_px = int(d_xy.sum())
        d_xz_px = int(d_xz.sum())

        ov_all_xy = int((d_xy & b_all_xy).sum())
        ov_all_xz = int((d_xz & b_all_xz).sum())
        ov10_all_xy = int((d_xy_dil10 & b_all_xy).sum())
        ov10_all_xz = int((d_xz_dil10 & b_all_xz).sum())

        dead_visit_all_targets_v1 = (ov_all_xy >= MIN_OVERLAP_PX_2D) and (
            ov_all_xz >= MIN_OVERLAP_PX_2D
        )
        dead_visit_all_targets_no_art = dead_visit_all_targets_v1 and (not is_artifact)

        bg_xy = float(np.median(g_xy[~b_all_xy])) if np.any(~b_all_xy) else float(np.median(g_xy))
        bg_xz = float(np.median(g_xz[~b_all_xz])) if np.any(~b_all_xz) else float(np.median(g_xz))

        if target_all_xy_px > 0:
            vals_xy = g_xy[b_all_xy]
            max_g_xy = float(vals_xy.max())
            p95_g_xy = float(np.percentile(vals_xy, 95))
            mean_g_xy = float(vals_xy.mean())
            max_raw_xy = int(mip_xy[b_all_xy].max())
        else:
            max_g_xy, p95_g_xy, mean_g_xy, max_raw_xy = 0.0, 0.0, 0.0, 0

        if target_all_xz_px > 0:
            vals_xz = g_xz[b_all_xz]
            max_g_xz = float(vals_xz.max())
            p95_g_xz = float(np.percentile(vals_xz, 95))
            mean_g_xz = float(vals_xz.mean())
            max_raw_xz = int(mip_xz[b_all_xz].max())
        else:
            max_g_xz, p95_g_xz, mean_g_xz, max_raw_xz = 0.0, 0.0, 0.0, 0

        # 3. Contacted target instance mask across sweeps 1..5 of this visit
        n_contact_sweeps = sum(1 for (_s, _p, s_tot, _tid, _nt) in sweep_specs if s_tot > 0)
        ca_effective = ca_tracked_um2 if trace_num == 1 else ca_all_um2

        best_ov_xy = 0
        best_ov_xz = 0
        best_ov10_xy = 0
        best_ov10_xz = 0
        best_c_xy_px = 0
        best_c_xz_px = 0
        dead_contacted_before_art = False
        dead_contacted_strict_before_art = False
        dead_contacted_dil10_before_art = False

        # Only need 3D target instance segmentation on sweeps with s_total > 0
        # when the 2D dead-dye mask overlaps at least 1 pixel of any target voxel
        if n_contact_sweeps > 0 and (
            ov_all_xy > 0 or ov_all_xz > 0 or ov10_all_xy > 0 or ov10_all_xz > 0
        ):
            for s_idx, s_pred_path, s_tot, tracked_id, _n_targ in sweep_specs:
                if s_tot <= 0 or (not s_pred_path.exists()):
                    continue
                if s_idx == 1:
                    I_s = pred_s1
                else:
                    with h5py.File(s_pred_path, "r") as hf_s:
                        I_s = hf_s["predictions"][:]

                b_cell = I_s == 2
                b_mem = I_s == 4
                if not (np.any(b_cell) or np.any(b_mem)):
                    continue
                sl_b = bbox_slices(b_cell | b_mem, 2, I_s.shape)
                b_sub = instance_seg_matlab_xyz(b_cell[sl_b], b_mem[sl_b])
                max_b = int(b_sub.max())
                if max_b == 0:
                    continue

                z_sl, y_sl, x_sl = sl_b
                if max_b == 1:
                    # Unique target instance in a sweep with s_total > 0 is guaranteed contacted
                    contacted_vor = [1]
                    contacted_str = [1]
                    b_proj_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                    bi_sub = b_sub == 1
                    bi_xy = np.zeros((256, 256), dtype=bool)
                    bi_xz = np.zeros((256, I_s.shape[0]), dtype=bool)
                    bi_xy[x_sl, y_sl] = bi_sub.max(axis=0).T
                    bi_xz[x_sl, z_sl] = bi_sub.max(axis=1).T
                    b_proj_map[1] = (bi_xy, bi_xz)
                else:
                    b_proj_map = {}
                    cand_bids: list[int] = []
                    for bid in range(1, max_b + 1):
                        bi_sub = b_sub == bid
                        bi_xy = np.zeros((256, 256), dtype=bool)
                        bi_xz = np.zeros((256, I_s.shape[0]), dtype=bool)
                        bi_xy[x_sl, y_sl] = bi_sub.max(axis=0).T
                        bi_xz[x_sl, z_sl] = bi_sub.max(axis=1).T
                        b_proj_map[bid] = (bi_xy, bi_xz)
                        if (
                            np.any(d_xy & bi_xy)
                            or np.any(d_xz & bi_xz)
                            or np.any(d_xy_dil10 & bi_xy)
                            or np.any(d_xz_dil10 & bi_xz)
                        ):
                            cand_bids.append(bid)
                    if not cand_bids:
                        continue
                    contacted_vor, contacted_str = _compute_contacted_b_ids_for_sweep(
                        I_s, b_sub, sl_b, trace_num, tracked_id, cand_bids
                    )

                for bid in contacted_vor:
                    bi_xy, bi_xz = b_proj_map[bid]
                    c_xy_px = int(bi_xy.sum())
                    c_xz_px = int(bi_xz.sum())
                    ov_xy_b = int((d_xy & bi_xy).sum())
                    ov_xz_b = int((d_xz & bi_xz).sum())
                    ov10_xy_b = int((d_xy_dil10 & bi_xy).sum())
                    ov10_xz_b = int((d_xz_dil10 & bi_xz).sum())

                    if (ov_xy_b + ov_xz_b) >= (best_ov_xy + best_ov_xz):
                        best_ov_xy = ov_xy_b
                        best_ov_xz = ov_xz_b
                        best_c_xy_px = c_xy_px
                        best_c_xz_px = c_xz_px
                    if (ov10_xy_b + ov10_xz_b) >= (best_ov10_xy + best_ov10_xz):
                        best_ov10_xy = ov10_xy_b
                        best_ov10_xz = ov10_xz_b

                    if (ov_xy_b >= MIN_OVERLAP_PX_2D) and (ov_xz_b >= MIN_OVERLAP_PX_2D):
                        dead_contacted_before_art = True
                        if bid in contacted_str:
                            dead_contacted_strict_before_art = True
                    if (ov10_xy_b >= MIN_OVERLAP_PX_2D) and (ov10_xz_b >= MIN_OVERLAP_PX_2D):
                        dead_contacted_dil10_before_art = True

        dead_visit_v2 = dead_contacted_before_art and (not is_artifact)
        dead_visit_strict_v2 = dead_contacted_strict_before_art and (not is_artifact)
        dead_visit_dil10_v2 = dead_contacted_dil10_before_art and (not is_artifact)

        frac_xy = (best_ov_xy / best_c_xy_px) if best_c_xy_px > 0 else 0.0
        frac_xz = (best_ov_xz / best_c_xz_px) if best_c_xz_px > 0 else 0.0

        rows.append(
            {
                "group_id": group_id,
                "experiment": exp,
                "condition": cond,
                "roi": roi,
                "roi_index": roi_idx,
                "trace_num": trace_num,
                "non_single_track_flag": non_single_track_flag,
                "visit": v_idx,
                "time_min": round(v_idx * VISIT_INTERVAL_MIN, 1),
                "in_200min_window": bool(v_idx <= FIXED_WINDOW_MAX_VISIT),
                "stem": stem,
                "contact_area_um2": round(float(ca_effective), 4),
                "contact_area_tracked_um2": round(float(ca_tracked_um2), 4),
                "contact_area_all_tcells_um2": round(float(ca_all_um2), 4),
                "has_contact": bool(n_contact_sweeps > 0),
                "n_contact_sweeps_in_visit": int(n_contact_sweeps),
                "artifact_405_band": is_artifact,
                "mip_xy_max_raw405": art_info["mip_xy_max_raw405"],
                "mip_xz_max_raw405": art_info["mip_xz_max_raw405"],
                "mip_xy_col0_median_raw405": art_info["mip_xy_col0_median_raw405"],
                "mip_xy_artifact_cols_n": art_info["mip_xy_artifact_cols_n"],
                "mip_xz_artifact_cols_n": art_info["mip_xz_artifact_cols_n"],
                "contacted_target_xy_px": best_c_xy_px,
                "contacted_target_xz_px": best_c_xz_px,
                "target_all_xy_px": target_all_xy_px,
                "target_all_xz_px": target_all_xz_px,
                "dead_dye_xy_px": d_xy_px,
                "dead_dye_xz_px": d_xz_px,
                "overlap_xy_px": best_ov_xy,
                "overlap_xz_px": best_ov_xz,
                "overlap_dil10_xy_px": best_ov10_xy,
                "overlap_dil10_xz_px": best_ov10_xz,
                "overlap_xy_frac": round(frac_xy, 4),
                "overlap_xz_frac": round(frac_xz, 4),
                "overlap_all_targets_xy_px": ov_all_xy,
                "overlap_all_targets_xz_px": ov_all_xz,
                "overlap_all_targets_dil10_xy_px": ov10_all_xy,
                "overlap_all_targets_dil10_xz_px": ov10_all_xz,
                "max_raw405_in_target_xy": max_raw_xy,
                "max_raw405_in_target_xz": max_raw_xz,
                "max_gauss405_in_target_xy": round(max_g_xy, 2),
                "max_gauss405_in_target_xz": round(max_g_xz, 2),
                "p95_gauss405_in_target_xy": round(p95_g_xy, 2),
                "p95_gauss405_in_target_xz": round(p95_g_xz, 2),
                "mean_gauss405_in_target_xy": round(mean_g_xy, 2),
                "mean_gauss405_in_target_xz": round(mean_g_xz, 2),
                "bg_median_gauss405_xy": round(bg_xy, 2),
                "bg_median_gauss405_xz": round(bg_xz, 2),
                "dead_visit": bool(dead_visit_v2),
                "dead_visit_strict_touch": bool(dead_visit_strict_v2),
                "dead_visit_dil10": bool(dead_visit_dil10_v2),
                "dead_visit_contacted_before_art": bool(dead_contacted_before_art),
                "dead_visit_all_targets_no_art": bool(dead_visit_all_targets_no_art),
                "dead_visit_all_targets_v1": bool(dead_visit_all_targets_v1),
            }
        )
    return rows


def load_paper_chamber_mapping_tracked() -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load the Step 3c tracked chamber-to-paper-row mapping (`results/chamber_to_row_map_tracked.csv`).
    """
    df_map = pd.read_csv(ROW_MAP_TRACKED_CSV, dtype={"group_id": str})
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df_map.iterrows():
        key = (str(row["experiment"]), str(row["roi"]))
        mapping[key] = {
            "in_paper_tracked": True,
            "paper_cell_index_tracked": int(row["cell_index"]),
            "paper_CA_um2_tracked": float(row["paper_CA_um2"]),
            "identity_label_tracked": str(row["identity_label"]),
            "value_label_tracked": str(row["value_label"]),
        }
    return mapping


def _compute_survival_window_metrics(
    visits: np.ndarray,
    pos: np.ndarray,
    has_c: np.ndarray,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Compute sustained (>=2 consecutive visits) and any-visit (>=1 visit) death calls,
    death times, and Kaplan-Meier time-from-first-contact survival fields on a given
    visit sequence (`suffix=""` for full recording, `suffix="_200m"` for visits 1-20).
    """
    n_v = len(visits)
    last_obs_v = int(visits[-1]) if n_v > 0 else 0
    last_obs_t = round(float(last_obs_v * VISIT_INTERVAL_MIN), 1)

    c_idx = np.where(has_c)[0]
    if c_idx.size > 0:
        first_c_v: int | None = int(visits[c_idx[0]])
        first_c_t: float | None = float(first_c_v * VISIT_INTERVAL_MIN)
        no_contact = False
    else:
        first_c_v = None
        first_c_t = None
        no_contact = True

    sust_idx: int | None = None
    for i in range(n_v - 1):
        if pos[i] and pos[i + 1]:
            sust_idx = i
            break

    ever_dead = sust_idx is not None
    death_visit: int | None = int(visits[sust_idx]) if sust_idx is not None else None
    death_time_min: float | None = (
        float(death_visit * VISIT_INTERVAL_MIN) if death_visit is not None else None
    )

    any_idx = np.where(pos)[0]
    ever_dead_any = bool(any_idx.size > 0)
    first_any_v: int | None = int(visits[any_idx[0]]) if ever_dead_any else None
    first_any_t: float | None = (
        float(first_any_v * VISIT_INTERVAL_MIN) if first_any_v is not None else None
    )

    if death_time_min is not None and first_c_t is not None:
        c2d_sust: float | None = round(death_time_min - first_c_t, 1)
    else:
        c2d_sust = None

    if first_any_t is not None and first_c_t is not None:
        c2d_any: float | None = round(first_any_t - first_c_t, 1)
    else:
        c2d_any = None

    censored_sust = not ever_dead
    censored_any = not ever_dead_any

    if first_c_t is not None:
        censor_dt = max(0.0, round(last_obs_t - first_c_t, 1))
        if ever_dead and c2d_sust is not None:
            km_t_sust: float | None = max(0.0, float(c2d_sust))
            km_ev_sust: int | None = 1
        else:
            km_t_sust = censor_dt
            km_ev_sust = 0

        if ever_dead_any and c2d_any is not None:
            km_t_any: float | None = max(0.0, float(c2d_any))
            km_ev_any: int | None = 1
        else:
            km_t_any = censor_dt
            km_ev_any = 0
    else:
        km_t_sust = None
        km_ev_sust = None
        km_t_any = None
        km_ev_any = None

    return {
        f"n_visits{suffix}": n_v,
        f"n_positive_visits{suffix}": int(pos.sum()),
        f"n_contact_visits{suffix}": int(has_c.sum()),
        f"first_contact_visit{suffix}": first_c_v,
        f"first_contact_time_min{suffix}": first_c_t,
        f"last_observed_visit{suffix}": last_obs_v,
        f"last_observed_time_min{suffix}": last_obs_t,
        f"no_contact{suffix}": no_contact,
        f"ever_dead{suffix}": ever_dead,
        f"death_visit{suffix}": death_visit,
        f"death_time_min{suffix}": death_time_min,
        f"contact_to_death_min{suffix}": c2d_sust,
        f"censored{suffix}": censored_sust,
        f"km_contact_time_min{suffix}": km_t_sust,
        f"km_event_sustained{suffix}": km_ev_sust,
        f"ever_dead_any_visit{suffix}": ever_dead_any,
        f"first_any_dead_visit{suffix}": first_any_v,
        f"first_any_dead_time_min{suffix}": first_any_t,
        f"any_contact_to_death_min{suffix}": c2d_any,
        f"censored_any_visit{suffix}": censored_any,
        f"km_any_contact_time_min{suffix}": km_t_any,
        f"km_event_any_visit{suffix}": km_ev_any,
    }


def _sustained_and_any_from_bool(pos: np.ndarray) -> tuple[bool, bool]:
    ever_any = bool(np.any(pos))
    ever_sust = bool(np.any(pos[:-1] & pos[1:])) if len(pos) >= 2 else False
    return ever_sust, ever_any


def build_chamber_and_visit_tables_v2(
    experiments: tuple[tuple[str, str, str], ...] = ALL_EXPERIMENTS,
    workers: int = 6,
    reuse_visits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Step 4 v2 per-visit and per-chamber death labels across all 13 experiments
    (502 chambers, 10,510 chamber-visits), returning:
      - `df_visits_v2`: 10,510 rows
      - `df_chambers_v2`: 502 rows (with both Full Recording and Fixed 200-min Window columns,
        plus v1 and before-artifact ablation columns)
      - `df_groups_v2`: 13 rows (per-group comparison vs paper `kill.csv` / `Figure 5g`)
      - `df_conditions_v2`: 5 rows (High-killing, Low-killing, Low-killing complete,
        Dasatinib/Latrunculin, and All 13 Groups pooled comparison)
      - `df_artifact_roi0`: 13 rows (characterisation and before/after impact of the
        `MIP_ROI_0` late 405-nm artifact across all 13 experiments)
    """
    paper_map_v1 = load_paper_chamber_mapping()
    paper_map_tracked = load_paper_chamber_mapping_tracked()
    df_paper = pd.read_csv(CLEAN_419_CSV, dtype={"group_id": str})
    df_grp_inv = pd.read_csv(GROUP_INV_CSV, dtype={"group_id": str})

    ch_meta_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    tasks: list[
        tuple[
            str,
            str,
            str,
            str,
            int,
            list[tuple[int, str, Path, Path, Path, float, float, list[tuple[int, Path, int, int, int]]]],
        ]
    ] = []

    for gid, exp, cond in experiments:
        ch_all_df = pd.read_csv(STEP3_DIR / exp / "per_chamber_contact_area.csv").sort_values(
            "roi_index"
        )
        ch_trk_df = pd.read_csv(STEP3_TRACKED_DIR / exp / "per_chamber.csv").sort_values(
            "roi_index"
        )
        sw_all_df = pd.read_csv(STEP3_DIR / exp / "per_sweep_contact_area.csv")
        sw_trk_df = pd.read_csv(STEP3_TRACKED_DIR / exp / "per_sweep.csv")

        # Index per-sweep metadata by (roi, visit, sweep_in_visit)
        sw_all_map: dict[tuple[str, int, int], tuple[int, int]] = {}
        for r in sw_all_df.itertuples(index=False):
            sw_all_map[(str(r.roi), int(r.visit), int(r.sweep))] = (
                int(r.s_total),
                int(r.n_target_instances),
            )

        sw_trk_map: dict[tuple[str, int, int], tuple[int, int, int]] = {}
        for r in sw_trk_df.itertuples(index=False):
            sw_trk_map[(str(r.roi), int(r.visit), int(r.sweep))] = (
                int(r.s_total),
                int(r.tracked_tcell_id),
                int(r.n_target_instances),
            )

        trk_row_by_roi = {str(r["roi"]): r for _, r in ch_trk_df.iterrows()}

        for _, row_all in ch_all_df.iterrows():
            roi = str(row_all["roi"])
            row_trk = trk_row_by_roi[roi]
            trace_num = int(row_trk["trace_num"])
            n_v = int(row_all["n_visits"])

            v_ca_all: dict[int, float] = {}
            v_ca_trk: dict[int, float] = {}
            for v in range(1, n_v + 1):
                col = f"visit_{v:02d}_CA_um2"
                v_ca_all[v] = (
                    float(row_all[col]) if col in row_all and pd.notna(row_all[col]) else 0.0
                )
                v_ca_trk[v] = (
                    float(row_trk[col]) if col in row_trk and pd.notna(row_trk[col]) else 0.0
                )

            ca_trk_val = float(row_trk["CA_um2"]) if pd.notna(row_trk["CA_um2"]) else 0.0
            ca_all_val = float(row_all["CA_um2"]) if pd.notna(row_all["CA_um2"]) else 0.0
            ch_meta_lookup[(exp, roi)] = {
                "trace_num": trace_num,
                "included_by_trace_rule": bool(trace_num == 1),
                "non_single_track_flag": bool(trace_num != 1),
                "CA_tracked_um2": ca_trk_val,
                "CA_all_tcells_um2": ca_all_val,
                "CA_effective_um2": ca_trk_val if trace_num == 1 else ca_all_val,
                "max_tcell_instances": int(row_all["max_tcell_instances"]),
                "multi_tcell_sweeps": int(row_all["multi_tcell_sweeps"]),
            }

            base_items = discover_chamber_visit_items(exp, roi, n_v, v_ca_all)
            roi_mask_dir = MASKS_DIR / exp / roi
            visit_items_v2: list[
                tuple[int, str, Path, Path, Path, float, float, list[tuple[int, Path, int, int, int]]]
            ] = []
            for v_idx, stem, xy_p, xz_p, pred_p_s1, ca_all_v in base_items:
                ca_trk_v = v_ca_trk.get(v_idx, 0.0)
                sweep_specs: list[tuple[int, Path, int, int, int]] = []
                for s_idx in range(1, 6):
                    s_stem = f"{int(stem) + s_idx - 1:05d}"
                    s_pred_p = roi_mask_dir / f"{s_stem}_predictions.h5"
                    if trace_num == 1:
                        s_tot, tid, n_targ = sw_trk_map.get((roi, v_idx, s_idx), (0, 0, 0))
                    else:
                        s_tot, n_targ = sw_all_map.get((roi, v_idx, s_idx), (0, 0))
                        tid = -1
                    sweep_specs.append((s_idx, s_pred_p, s_tot, tid, n_targ))
                visit_items_v2.append(
                    (v_idx, stem, xy_p, xz_p, pred_p_s1, ca_trk_v, ca_all_v, sweep_specs)
                )
            tasks.append((gid, exp, cond, roi, trace_num, visit_items_v2))

    cached_visits_csv = STEP4_DIR / "per_visit_death_labels_all_v2.csv"
    if reuse_visits and len(experiments) == len(ALL_EXPERIMENTS) and cached_visits_csv.exists():
        df_visits = pd.read_csv(cached_visits_csv, dtype={"group_id": str, "stem": str})
    else:
        visit_rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
            for res in executor.map(_process_one_chamber_v2, tasks):
                visit_rows.extend(res)

        df_visits = pd.DataFrame(visit_rows)
        df_visits["_grp_order"] = df_visits["group_id"].map(
            lambda g: GROUP_ORDER_MAP.get(str(g), 999)
        )
        df_visits = (
            df_visits.sort_values(["_grp_order", "roi_index", "visit"])
            .drop(columns=["_grp_order"])
            .reset_index(drop=True)
        )
        if len(experiments) == len(ALL_EXPERIMENTS):
            STEP4_DIR.mkdir(parents=True, exist_ok=True)
            df_visits.to_csv(cached_visits_csv, index=False)

    exp_max_visits: dict[str, int] = (
        df_visits.groupby("experiment")["visit"].max().astype(int).to_dict()
    )

    # Build per-chamber v2 summary table (502 rows)
    chamber_rows: list[dict[str, Any]] = []
    for (gid, exp, cond, roi, roi_idx), grp in df_visits.groupby(
        ["group_id", "experiment", "condition", "roi", "roi_index"], sort=False
    ):
        grp = grp.sort_values("visit").reset_index(drop=True)
        meta = ch_meta_lookup[(str(exp), str(roi))]
        p_trk = paper_map_tracked.get((str(exp), str(roi)), {})
        p_v1 = paper_map_v1.get((str(exp), str(roi)), {})

        visits_arr = grp["visit"].to_numpy(dtype=int)
        pos_v2 = grp["dead_visit"].to_numpy(dtype=bool)
        pos_strict = grp["dead_visit_strict_touch"].to_numpy(dtype=bool)
        pos_dil10 = grp["dead_visit_dil10"].to_numpy(dtype=bool)
        pos_before_art = grp["dead_visit_contacted_before_art"].to_numpy(dtype=bool)
        pos_all_no_art = grp["dead_visit_all_targets_no_art"].to_numpy(dtype=bool)
        pos_all_v1 = grp["dead_visit_all_targets_v1"].to_numpy(dtype=bool)
        has_c = grp["has_contact"].to_numpy(dtype=bool)
        art_arr = grp["artifact_405_band"].to_numpy(dtype=bool)

        last_obs_v = int(visits_arr[-1])
        is_truncated = bool(last_obs_v < exp_max_visits.get(str(exp), last_obs_v))

        # Full recording survival & death metrics
        m_full = _compute_survival_window_metrics(visits_arr, pos_v2, has_c, suffix="")

        # Fixed 200-minute window (visits 1-20) survival & death metrics
        mask_200 = visits_arr <= FIXED_WINDOW_MAX_VISIT
        m_200 = _compute_survival_window_metrics(
            visits_arr[mask_200], pos_v2[mask_200], has_c[mask_200], suffix="_200m"
        )

        # Comparison / ablation flags (full & 200m)
        ever_sust_strict, ever_any_strict = _sustained_and_any_from_bool(pos_strict)
        ever_sust_strict_200, ever_any_strict_200 = _sustained_and_any_from_bool(
            pos_strict[mask_200]
        )
        ever_sust_dil10, ever_any_dil10 = _sustained_and_any_from_bool(pos_dil10)
        ever_sust_before_art, ever_any_before_art = _sustained_and_any_from_bool(pos_before_art)
        ever_sust_all_no_art, ever_any_all_no_art = _sustained_and_any_from_bool(pos_all_no_art)
        ever_sust_all_v1, ever_any_all_v1 = _sustained_and_any_from_bool(pos_all_v1)
        ever_sust_all_v1_200, ever_any_all_v1_200 = _sustained_and_any_from_bool(
            pos_all_v1[mask_200]
        )

        dead_at_v1 = bool(pos_v2[0])
        fc_v = m_full["first_contact_visit"]
        dv = m_full["death_visit"]
        dead_before_contact = bool(
            (dv is not None and fc_v is not None and dv < fc_v) or dead_at_v1
        )

        row_dict: dict[str, Any] = {
            "group_id": gid,
            "experiment": exp,
            "condition": cond,
            "roi": roi,
            "roi_index": int(roi_idx),
            "trace_num": meta["trace_num"],
            "included_by_trace_rule": meta["included_by_trace_rule"],
            "non_single_track_flag": meta["non_single_track_flag"],
            "in_paper_tracked": bool(p_trk.get("in_paper_tracked", False)),
            "paper_cell_index_tracked": p_trk.get("paper_cell_index_tracked", None),
            "paper_CA_um2_tracked": p_trk.get("paper_CA_um2_tracked", None),
            "identity_label_tracked": p_trk.get("identity_label_tracked", "excluded_by_tracker"),
            "value_label_tracked": p_trk.get("value_label_tracked", "excluded"),
            "in_paper_v1": bool(p_v1.get("in_paper", False)),
            "in_paper_firm_probable_v1": bool(p_v1.get("in_paper_firm_probable", False)),
            "CA_um2": round(meta["CA_effective_um2"], 4),
            "CA_tracked_um2": round(meta["CA_tracked_um2"], 4),
            "CA_all_tcells_um2": round(meta["CA_all_tcells_um2"], 4),
            "is_truncated": is_truncated,
            "had_artifact_405_band": bool(np.any(art_arr)),
            "n_artifact_405_visits": int(art_arr.sum()),
        }
        row_dict.update(m_full)
        row_dict.update(
            {
                "dead_at_visit_1": dead_at_v1,
                "dead_before_contact": dead_before_contact,
                "ever_dead_strict_touch": ever_sust_strict,
                "ever_dead_any_visit_strict_touch": ever_any_strict,
                "ever_dead_strict_touch_200m": ever_sust_strict_200,
                "ever_dead_any_visit_strict_touch_200m": ever_any_strict_200,
                "ever_dead_dil10": ever_sust_dil10,
                "ever_dead_any_visit_dil10": ever_any_dil10,
                "ever_dead_contacted_before_art": ever_sust_before_art,
                "ever_dead_any_visit_contacted_before_art": ever_any_before_art,
                "ever_dead_all_targets_no_art": ever_sust_all_no_art,
                "ever_dead_any_visit_all_targets_no_art": ever_any_all_no_art,
                "ever_dead_all_targets_v1": ever_sust_all_v1,
                "ever_dead_any_visit_all_targets_v1": ever_any_all_v1,
                "ever_dead_all_targets_v1_200m": ever_sust_all_v1_200,
                "ever_dead_any_visit_all_targets_v1_200m": ever_any_all_v1_200,
                "max_overlap_xy_px": int(grp["overlap_xy_px"].max()),
                "max_overlap_xz_px": int(grp["overlap_xz_px"].max()),
                "max_overlap_all_targets_xy_px": int(grp["overlap_all_targets_xy_px"].max()),
                "max_overlap_all_targets_xz_px": int(grp["overlap_all_targets_xz_px"].max()),
                "max_gauss405_in_target_xy": round(float(grp["max_gauss405_in_target_xy"].max()), 2),
                "max_gauss405_in_target_xz": round(float(grp["max_gauss405_in_target_xz"].max()), 2),
                "max_tcell_instances": meta["max_tcell_instances"],
                "multi_tcell_sweeps": meta["multi_tcell_sweeps"],
            }
        )
        row_dict.update(m_200)
        chamber_rows.append(row_dict)

    df_chambers = pd.DataFrame(chamber_rows)
    df_chambers["_grp_order"] = df_chambers["group_id"].map(
        lambda g: GROUP_ORDER_MAP.get(str(g), 999)
    )
    df_chambers = (
        df_chambers.sort_values(["_grp_order", "roi_index"])
        .drop(columns=["_grp_order"])
        .reset_index(drop=True)
    )

    # Build MIP_ROI_0 artifact characterisation table (13 rows)
    art_rows: list[dict[str, Any]] = []
    for gid, exp, cond in experiments:
        sub_v = df_visits[(df_visits["experiment"] == exp) & (df_visits["roi"] == "MIP_ROI_0")].sort_values("visit")
        sub_c = df_chambers[(df_chambers["experiment"] == exp) & (df_chambers["roi"] == "MIP_ROI_0")].iloc[0]
        art_v = sub_v[sub_v["artifact_405_band"]]
        clean_v = sub_v[~sub_v["artifact_405_band"]]
        art_visit_list = ",".join(str(int(v)) for v in art_v["visit"].tolist())
        pos_before_list = ",".join(
            str(int(v)) for v in sub_v[sub_v["dead_visit_all_targets_v1"]]["visit"].tolist()
        )
        pos_noart_all_list = ",".join(
            str(int(v)) for v in sub_v[sub_v["dead_visit_all_targets_no_art"]]["visit"].tolist()
        )
        pos_after_list = ",".join(
            str(int(v)) for v in sub_v[sub_v["dead_visit"]]["visit"].tolist()
        )
        art_rows.append(
            {
                "group_id": gid,
                "experiment": exp,
                "condition": cond,
                "roi": "MIP_ROI_0",
                "trace_num": int(sub_c["trace_num"]),
                "n_visits_total": int(len(sub_v)),
                "artifact_visits": art_visit_list,
                "n_artifact_visits": int(len(art_v)),
                "clean_visits_max_raw405_xy": int(clean_v["mip_xy_max_raw405"].max()) if len(clean_v) else 0,
                "artifact_visits_min_col0_median_adu": round(float(art_v["mip_xy_col0_median_raw405"].min()), 1) if len(art_v) else 0.0,
                "artifact_visits_max_raw405_xy": int(art_v["mip_xy_max_raw405"].max()) if len(art_v) else 0,
                "artifact_visits_max_raw405_xz": int(art_v["mip_xz_max_raw405"].max()) if len(art_v) else 0,
                "artifact_band_width_xy_px": int(art_v["mip_xy_artifact_cols_n"].max()) if len(art_v) else 0,
                "artifact_band_width_xz_px": int(art_v["mip_xz_artifact_cols_n"].max()) if len(art_v) else 0,
                "positive_visits_v1_before": pos_before_list if pos_before_list else "none",
                "positive_visits_all_targets_no_art": pos_noart_all_list if pos_noart_all_list else "none",
                "positive_visits_v2_after": pos_after_list if pos_after_list else "none",
                "ever_dead_v1_sustained_before": bool(sub_c["ever_dead_all_targets_v1"]),
                "ever_dead_v1_any_before": bool(sub_c["ever_dead_any_visit_all_targets_v1"]),
                "ever_dead_all_targets_no_art": bool(sub_c["ever_dead_all_targets_no_art"]),
                "ever_dead_v2_sustained_after": bool(sub_c["ever_dead"]),
                "ever_dead_v2_any_after": bool(sub_c["ever_dead_any_visit"]),
                "death_visit_v2_after": sub_c["death_visit"],
                "artifact_only_false_positive_eliminated": bool(
                    sub_c["ever_dead_all_targets_v1"] and (not sub_c["ever_dead_all_targets_no_art"])
                ),
                "false_positive_eliminated": bool(
                    sub_c["ever_dead_all_targets_v1"] and (not sub_c["ever_dead_any_visit"])
                ),
            }
        )
    df_artifact_roi0 = pd.DataFrame(art_rows)

    # Build per-group comparison table (13 rows)
    group_rows: list[dict[str, Any]] = []
    for gid, exp, cond in experiments:
        sub_c = df_chambers[df_chambers["experiment"] == exp]
        sub_trk1 = sub_c[sub_c["included_by_trace_rule"]]
        sub_non1 = sub_c[~sub_c["included_by_trace_rule"]]
        sub_p = df_paper[df_paper["group_id"] == gid]
        inv_row = df_grp_inv[df_grp_inv["group_id"] == gid].iloc[0]

        n_paper = int(len(sub_p))
        k_paper = int(round(float(sub_p["group_kill_rate"].iloc[0]) * n_paper))
        rate_paper = k_paper / n_paper if n_paper > 0 else np.nan
        p_ci_lo, p_ci_hi = clopper_pearson_ci(k_paper, n_paper)

        n_trk1 = int(len(sub_trk1))
        n_non1 = int(len(sub_non1))
        n_all = int(len(sub_c))
        n_v_exp = int(sub_c["n_visits"].max()) if n_all > 0 else 0

        # Tracked single-T-cell chambers (trace_num == 1) — primary comparison to paper
        k_trk1_sust_full = int(sub_trk1["ever_dead"].sum())
        k_trk1_any_full = int(sub_trk1["ever_dead_any_visit"].sum())
        k_trk1_sust_200 = int(sub_trk1["ever_dead_200m"].sum())
        k_trk1_any_200 = int(sub_trk1["ever_dead_any_visit_200m"].sum())

        # Ablation counts on trace_num == 1
        k_trk1_v1_sust = int(sub_trk1["ever_dead_all_targets_v1"].sum())
        k_trk1_v1_any = int(sub_trk1["ever_dead_any_visit_all_targets_v1"].sum())
        k_trk1_noart_sust = int(sub_trk1["ever_dead_all_targets_no_art"].sum())
        k_trk1_noart_any = int(sub_trk1["ever_dead_any_visit_all_targets_no_art"].sum())
        k_trk1_before_art_sust = int(sub_trk1["ever_dead_contacted_before_art"].sum())
        k_trk1_before_art_any = int(sub_trk1["ever_dead_any_visit_contacted_before_art"].sum())

        # All chambers (502 total) and flagged non-single-track chambers (trace_num != 1)
        k_all_sust_full = int(sub_c["ever_dead"].sum())
        k_all_any_full = int(sub_c["ever_dead_any_visit"].sum())
        k_all_sust_200 = int(sub_c["ever_dead_200m"].sum())
        k_all_any_200 = int(sub_c["ever_dead_any_visit_200m"].sum())
        k_non1_sust_full = int(sub_non1["ever_dead"].sum())
        k_non1_any_full = int(sub_non1["ever_dead_any_visit"].sum())

        ci_sust_full = clopper_pearson_ci(k_trk1_sust_full, n_trk1)
        ci_any_full = clopper_pearson_ci(k_trk1_any_full, n_trk1)
        ci_sust_200 = clopper_pearson_ci(k_trk1_sust_200, n_trk1)
        ci_any_200 = clopper_pearson_ci(k_trk1_any_200, n_trk1)

        group_rows.append(
            {
                "group_id": gid,
                "experiment": exp,
                "condition": cond,
                "n_visits_full": n_v_exp,
                "paper_n_cells": n_paper,
                "paper_n_killed": k_paper,
                "paper_kill_rate_table": round(float(inv_row["kill_rate_published"]), 4),
                "paper_kill_rate_exact": round(rate_paper, 4),
                "paper_ci95_low": round(p_ci_lo, 4),
                "paper_ci95_high": round(p_ci_hi, 4),
                "trace1_chambers_n": n_trk1,
                "trace1_killed_sustained_full": k_trk1_sust_full,
                "trace1_kill_rate_sustained_full": round(k_trk1_sust_full / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_ci95_low_sustained_full": round(ci_sust_full[0], 4),
                "trace1_ci95_high_sustained_full": round(ci_sust_full[1], 4),
                "trace1_killed_any_visit_full": k_trk1_any_full,
                "trace1_kill_rate_any_visit_full": round(k_trk1_any_full / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_ci95_low_any_visit_full": round(ci_any_full[0], 4),
                "trace1_ci95_high_any_visit_full": round(ci_any_full[1], 4),
                "trace1_killed_sustained_200m": k_trk1_sust_200,
                "trace1_kill_rate_sustained_200m": round(k_trk1_sust_200 / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_ci95_low_sustained_200m": round(ci_sust_200[0], 4),
                "trace1_ci95_high_sustained_200m": round(ci_sust_200[1], 4),
                "trace1_killed_any_visit_200m": k_trk1_any_200,
                "trace1_kill_rate_any_visit_200m": round(k_trk1_any_200 / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_ci95_low_any_visit_200m": round(ci_any_200[0], 4),
                "trace1_ci95_high_any_visit_200m": round(ci_any_200[1], 4),
                "trace1_killed_v1_all_targets_sustained": k_trk1_v1_sust,
                "trace1_kill_rate_v1_all_targets_sustained": round(k_trk1_v1_sust / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_killed_v1_all_targets_any_visit": k_trk1_v1_any,
                "trace1_kill_rate_v1_all_targets_any_visit": round(k_trk1_v1_any / n_trk1, 4) if n_trk1 else np.nan,
                "trace1_killed_all_targets_no_art_sustained": k_trk1_noart_sust,
                "trace1_killed_all_targets_no_art_any_visit": k_trk1_noart_any,
                "trace1_killed_contacted_before_art_sustained": k_trk1_before_art_sust,
                "trace1_killed_contacted_before_art_any_visit": k_trk1_before_art_any,
                "flagged_non_trace1_chambers_n": n_non1,
                "flagged_non_trace1_killed_sustained_full": k_non1_sust_full,
                "flagged_non_trace1_killed_any_visit_full": k_non1_any_full,
                "all_chambers_n": n_all,
                "all_killed_sustained_full": k_all_sust_full,
                "all_kill_rate_sustained_full": round(k_all_sust_full / n_all, 4) if n_all else np.nan,
                "all_killed_any_visit_full": k_all_any_full,
                "all_kill_rate_any_visit_full": round(k_all_any_full / n_all, 4) if n_all else np.nan,
                "all_killed_sustained_200m": k_all_sust_200,
                "all_kill_rate_sustained_200m": round(k_all_sust_200 / n_all, 4) if n_all else np.nan,
                "all_killed_any_visit_200m": k_all_any_200,
                "all_kill_rate_any_visit_200m": round(k_all_any_200 / n_all, 4) if n_all else np.nan,
            }
        )
    df_groups = pd.DataFrame(group_rows)

    # Build pooled condition comparison table
    regime_specs = [
        ("Untreated high-killing (Groups 1-5)", ["1", "2", "3", "4", "5"]),
        ("Untreated low-killing (Groups 6-10)", ["6", "7", "8", "9", "10"]),
        ("Untreated low-killing complete (Groups 6,8,9,10)", ["6", "8", "9", "10"]),
        ("Dasatinib / Latrunculin (Groups 8*-10*)", ["8*", "9*", "10*"]),
        ("All 13 groups total", [g for g, _, _ in experiments]),
        ("All 12 complete groups (excl. Group 7)", [g for g, _, _ in experiments if g != "7"]),
    ]
    cond_rows: list[dict[str, Any]] = []
    for label_str, g_list in regime_specs:
        sub_g = df_groups[df_groups["group_id"].isin(g_list)]
        if sub_g.empty:
            continue
        p_n = int(sub_g["paper_n_cells"].sum())
        p_k = int(sub_g["paper_n_killed"].sum())
        t1_n = int(sub_g["trace1_chambers_n"].sum())
        t1_sust_full = int(sub_g["trace1_killed_sustained_full"].sum())
        t1_any_full = int(sub_g["trace1_killed_any_visit_full"].sum())
        t1_sust_200 = int(sub_g["trace1_killed_sustained_200m"].sum())
        t1_any_200 = int(sub_g["trace1_killed_any_visit_200m"].sum())
        t1_v1_sust = int(sub_g["trace1_killed_v1_all_targets_sustained"].sum())
        t1_v1_any = int(sub_g["trace1_killed_v1_all_targets_any_visit"].sum())
        all_n = int(sub_g["all_chambers_n"].sum())
        all_sust_full = int(sub_g["all_killed_sustained_full"].sum())
        all_any_full = int(sub_g["all_killed_any_visit_full"].sum())
        all_sust_200 = int(sub_g["all_killed_sustained_200m"].sum())
        all_any_200 = int(sub_g["all_killed_any_visit_200m"].sum())

        cond_rows.append(
            {
                "regime": label_str,
                "groups": ",".join(g_list),
                "paper_n_cells": p_n,
                "paper_n_killed": p_k,
                "paper_kill_rate": round(p_k / p_n, 4) if p_n else np.nan,
                "trace1_chambers_n": t1_n,
                "trace1_killed_sustained_full": t1_sust_full,
                "trace1_kill_rate_sustained_full": round(t1_sust_full / t1_n, 4) if t1_n else np.nan,
                "trace1_killed_any_visit_full": t1_any_full,
                "trace1_kill_rate_any_visit_full": round(t1_any_full / t1_n, 4) if t1_n else np.nan,
                "trace1_killed_sustained_200m": t1_sust_200,
                "trace1_kill_rate_sustained_200m": round(t1_sust_200 / t1_n, 4) if t1_n else np.nan,
                "trace1_killed_any_visit_200m": t1_any_200,
                "trace1_kill_rate_any_visit_200m": round(t1_any_200 / t1_n, 4) if t1_n else np.nan,
                "trace1_killed_v1_all_targets_sustained": t1_v1_sust,
                "trace1_kill_rate_v1_all_targets_sustained": round(t1_v1_sust / t1_n, 4) if t1_n else np.nan,
                "trace1_killed_v1_all_targets_any_visit": t1_v1_any,
                "trace1_kill_rate_v1_all_targets_any_visit": round(t1_v1_any / t1_n, 4) if t1_n else np.nan,
                "all_chambers_n": all_n,
                "all_killed_sustained_full": all_sust_full,
                "all_kill_rate_sustained_full": round(all_sust_full / all_n, 4) if all_n else np.nan,
                "all_killed_any_visit_full": all_any_full,
                "all_kill_rate_any_visit_full": round(all_any_full / all_n, 4) if all_n else np.nan,
                "all_killed_sustained_200m": all_sust_200,
                "all_kill_rate_sustained_200m": round(all_sust_200 / all_n, 4) if all_n else np.nan,
                "all_killed_any_visit_200m": all_any_200,
                "all_kill_rate_any_visit_200m": round(all_any_200 / all_n, 4) if all_n else np.nan,
            }
        )
    df_conditions = pd.DataFrame(cond_rows)

    return df_visits, df_chambers, df_groups, df_conditions, df_artifact_roi0


def make_v2_figure(
    df_visits: pd.DataFrame,
    df_chambers: pd.DataFrame,
    df_groups: pd.DataFrame,
    df_artifact_roi0: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Generate `figures/step4_death_labels_v2.png` following House Style:
      - Panel (a): `MIP_ROI_0` late 405-nm artifact characterisation (`Visit 18` clean vs
        `Visit 19` corrupted `MIPxy_405` and `MIPxz_405`, plus column intensity profiles
        and before/after positive visits across all 13 experiments).
      - Panel (b): Per-group kill rate agreement with paper (`Figure 5g` / `kill.csv`) across
        all 13 groups comparing Paper (`#111111`), v2 Contacted Any-Visit Full & 200-min
        (`#1F4E79`), v2 Contacted Sustained Full & 200-min (`#2E7D5B`), and v1 All-Target
        Any-Visit (`#C0561F`).
      - Panel (c): Kaplan-Meier target survival curves from first contact — Sustained
        (`>= 2` consecutive visits) for High-Killing (1-5), Low-Killing (6-10), and
        Dasatinib/Latrunculin (8*-10*), showing both Full Recording and Fixed 200-min window.
      - Panel (d): Kaplan-Meier target survival curves from first contact — Any-Visit
        (`>= 1` visit, matching `summary.m` `max(B,[],2)`) for High-Killing (1-5),
        Low-Killing (6-10), and Dasatinib/Latrunculin (8*-10*), with Paper `Figure 5g`
        survival references.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.8,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    fig = plt.figure(figsize=(20.0, 13.6), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.02, 1.18],
        hspace=0.34,
        wspace=0.22,
        left=0.045,
        right=0.975,
        top=0.94,
        bottom=0.065,
    )

    # -------------------------------------------------------------------------
    # Panel (a): MIP_ROI_0 Late 405-nm Artifact Characterisation & Before/After
    # -------------------------------------------------------------------------
    gs_a = gs[0, 0].subgridspec(2, 3, width_ratios=[1.0, 1.0, 1.35], hspace=0.34, wspace=0.28)
    exp_ex = "v01_6_40_120_0-6_hd1"
    roi_ex = "MIP_ROI_0"
    mip_dir_ex = MIPS_DIR / exp_ex / roi_ex
    mask_dir_ex = MASKS_DIR / exp_ex / roi_ex

    # Visit 18 (stem 00086, clean) and Visit 19 (stem 00091, artifact)
    ex_items = discover_chamber_visit_items(exp_ex, roi_ex, 20, {})
    _, _, xy18_p, xz18_p, pred18_p, _ = ex_items[17]
    _, _, xy19_p, xz19_p, pred19_p, _ = ex_items[18]
    xy_18 = tifffile.imread(xy18_p)
    xz_18 = tifffile.imread(xz18_p)
    xy_19 = tifffile.imread(xy19_p)
    xz_19 = tifffile.imread(xz19_p)
    with h5py.File(pred18_p, "r") as hf:
        p18 = hf["predictions"][:]
    with h5py.File(pred19_p, "r") as hf:
        p19 = hf["predictions"][:]

    b18_xy = ((p18 == 2) | (p18 == 4)).max(axis=0).T
    b19_xy = ((p19 == 2) | (p19 == 4)).max(axis=0).T
    b18_xz = ((p18 == 2) | (p18 == 4)).max(axis=1).T
    b19_xz = ((p19 == 2) | (p19 == 4)).max(axis=1).T
    d19_xy, _ = compute_2d_dead_mask(xy_19, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
    d19_xz, _ = compute_2d_dead_mask(xz_19, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)

    ax_a00 = fig.add_subplot(gs_a[0, 0])
    ax_a01 = fig.add_subplot(gs_a[0, 1])
    ax_a10 = fig.add_subplot(gs_a[1, 0])
    ax_a11 = fig.add_subplot(gs_a[1, 1])
    ax_a_prof = fig.add_subplot(gs_a[:, 2])

    ax_a00.imshow(xy_18, cmap="gray", vmin=340, vmax=900, aspect="auto")
    if np.any(b18_xy):
        ax_a00.contour(b18_xy.astype(float), levels=[0.5], colors=["#1F4E79"], linewidths=0.9)
    ax_a00.set_title("Visit 18 MIPxy (180 min)\nmax = 469 ADU (clean)", fontsize=8.5, fontweight="bold")
    ax_a00.set_ylabel("row (y, px)", fontsize=8.0)
    ax_a00.tick_params(labelsize=7.5)

    ax_a01.imshow(xy_19, cmap="gray", vmin=340, vmax=900, aspect="auto")
    if np.any(b19_xy):
        ax_a01.contour(b19_xy.astype(float), levels=[0.5], colors=["#1F4E79"], linewidths=0.9)
    if np.any(d19_xy):
        ax_a01.contour(d19_xy.astype(float), levels=[0.5], colors=["#C0561F"], linewidths=0.9)
    ax_a01.set_title("Visit 19 MIPxy (190 min)\ncol 0-91 band, max = 23,670", fontsize=8.5, fontweight="bold")
    ax_a01.tick_params(labelsize=7.5)

    ax_a10.imshow(xz_18, cmap="gray", vmin=340, vmax=900, aspect="auto")
    if np.any(b18_xz):
        ax_a10.contour(b18_xz.astype(float), levels=[0.5], colors=["#1F4E79"], linewidths=0.9)
    ax_a10.set_title("Visit 18 MIPxz (180 min)\nmax = 469 ADU (clean)", fontsize=8.5, fontweight="bold")
    ax_a10.set_xlabel("slice (z) / col (x)", fontsize=8.0)
    ax_a10.set_ylabel("row (y, px)", fontsize=8.0)
    ax_a10.tick_params(labelsize=7.5)

    ax_a11.imshow(xz_19, cmap="gray", vmin=340, vmax=900, aspect="auto")
    if np.any(b19_xz):
        ax_a11.contour(b19_xz.astype(float), levels=[0.5], colors=["#1F4E79"], linewidths=0.9)
    if np.any(d19_xz):
        ax_a11.contour(d19_xz.astype(float), levels=[0.5], colors=["#C0561F"], linewidths=0.9)
    ax_a11.set_title("Visit 19 MIPxz (190 min)\nz 0-53 band (30° deskew)", fontsize=8.5, fontweight="bold")
    ax_a11.set_xlabel("slice (z, px)", fontsize=8.0)
    ax_a11.tick_params(labelsize=7.5)

    # Column profile plot in ax_a_prof
    cols_x = np.arange(256)
    ax_a_prof.plot(
        cols_x,
        np.median(xy_19, axis=0),
        color="#C0561F",
        linewidth=1.8,
        label="Visit 19 col median (artifact)",
    )
    ax_a_prof.plot(
        cols_x,
        np.min(xy_19, axis=0),
        color="#C0561F",
        linewidth=1.0,
        linestyle="--",
        label="Visit 19 col minimum (>404 in col 0-91)",
    )
    ax_a_prof.plot(
        cols_x,
        np.median(xy_18, axis=0),
        color="#1F4E79",
        linewidth=1.5,
        label="Visit 18 col median (clean baseline)",
    )
    ax_a_prof.axhline(
        THRESH_2D_ADU,
        color="#111111",
        linestyle=":",
        linewidth=1.1,
        label="Threshold 404 ADU",
    )
    ax_a_prof.set_xlim(0, 255)
    ax_a_prof.set_ylim(340, 880)
    ax_a_prof.set_xlabel("MIPxy column x (px)", fontsize=8.5, fontweight="bold")
    ax_a_prof.set_ylabel("Raw 405-nm intensity (ADU)", fontsize=8.5, fontweight="bold")
    ax_a_prof.set_title(
        "MIP_ROI_0 column profile & exclusion impact\n(39/39 visits at visit >= 19 across 13/13 groups)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_a_prof.grid(axis="both", color="#E3E3E3", linewidth=0.6)
    ax_a_prof.spines["top"].set_visible(False)
    ax_a_prof.spines["right"].set_visible(False)
    ax_a_prof.tick_params(labelsize=7.8)
    ax_a_prof.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E3E3E3",
        fontsize=7.1,
    )

    # Summary text box inside ax_a_prof showing before/after across the 13 experiments
    n_roi0_v1_dead = int(df_artifact_roi0["ever_dead_v1_sustained_before"].sum())
    n_roi0_noart_all = int(df_artifact_roi0["ever_dead_all_targets_no_art"].sum())
    n_roi0_v2_dead = int(df_artifact_roi0["ever_dead_v2_sustained_after"].sum())
    n_roi0_art_only_elim = int(df_artifact_roi0["artifact_only_false_positive_eliminated"].sum())
    summary_txt = (
        "Rule: col 0 median > 404 ADU\n"
        "      and max(MIPxy) > 5,000 ADU\n"
        "• Matched: 39/39 MIP_ROI_0 visits\n"
        "  at visit >= 19 (14,296–25,317 ADU)\n"
        "• Matched elsewhere: 0 / 10,471 visits\n"
        f"• MIP_ROI_0 dead before (v1): {n_roi0_v1_dead}/13\n"
        f"• After artifact rule alone:  {n_roi0_noart_all}/13\n"
        f"  ({n_roi0_art_only_elim} artifact-only FPs removed)\n"
        f"• After v2 (+bystander fix):  {n_roi0_v2_dead}/13\n"
        "  (v04,v05 true kills; v02,v03 bystander)"
    )
    ax_a_prof.text(
        0.34,
        0.18,
        summary_txt,
        transform=ax_a_prof.transAxes,
        fontsize=7.3,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "#F6F6F6", "edgecolor": "#C9C9C9", "linewidth": 0.7},
    )

    ax_a00.text(
        -0.22,
        1.18,
        "(a) MIP_ROI_0 late 405-nm artifact (visit >= 19 in all 13 groups)",
        transform=ax_a00.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color="#111111",
    )

    # -------------------------------------------------------------------------
    # Panel (b): Per-Group Kill Rate Agreement vs Paper (`Figure 5g`, 13 Groups)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(df_groups))
    w = 0.21

    r_paper = df_groups["paper_kill_rate_exact"].to_numpy(dtype=float) * 100.0
    r_any_full = df_groups["trace1_kill_rate_any_visit_full"].to_numpy(dtype=float) * 100.0
    r_any_200 = df_groups["trace1_kill_rate_any_visit_200m"].to_numpy(dtype=float) * 100.0
    r_sust_full = df_groups["trace1_kill_rate_sustained_full"].to_numpy(dtype=float) * 100.0
    r_sust_200 = df_groups["trace1_kill_rate_sustained_200m"].to_numpy(dtype=float) * 100.0
    r_v1_any = df_groups["trace1_kill_rate_v1_all_targets_any_visit"].to_numpy(dtype=float) * 100.0

    ax_b.bar(
        x - 1.5 * w,
        r_paper,
        width=w,
        color="#111111",
        edgecolor="#111111",
        linewidth=0.6,
        label="Paper kill.csv (Figure 5g, n = 419)",
    )
    ax_b.bar(
        x - 0.5 * w,
        r_any_full,
        width=w,
        color="#1F4E79",
        edgecolor="#111111",
        linewidth=0.6,
        label="v2 Contacted Any-visit Full (n = 416 trace_num == 1)",
    )
    ax_b.bar(
        x + 0.5 * w,
        r_sust_full,
        width=w,
        color="#2E7D5B",
        edgecolor="#111111",
        linewidth=0.6,
        label="v2 Contacted Sustained >=2 visits Full (n = 416)",
    )
    ax_b.bar(
        x + 1.5 * w,
        r_v1_any,
        width=w,
        color="#C0561F",
        edgecolor="#111111",
        linewidth=0.6,
        alpha=0.85,
        label="v1 All-target Any-visit (before fixes)",
    )

    # Overlay 200-min window markers on the 21-22 visit groups
    ax_b.scatter(
        x - 0.5 * w,
        r_any_200,
        s=26,
        facecolors="#FFFFFF",
        edgecolors="#111111",
        linewidths=1.2,
        zorder=5,
        label="Fixed 200-min window (visits 1–20)",
    )
    ax_b.scatter(
        x + 0.5 * w,
        r_sust_200,
        s=26,
        facecolors="#FFFFFF",
        edgecolors="#111111",
        linewidths=1.2,
        zorder=5,
    )

    # Error bars (95% Clopper-Pearson) on Paper and v2 Any-Visit Full
    p_lo = df_groups["paper_ci95_low"].to_numpy(dtype=float) * 100.0
    p_hi = df_groups["paper_ci95_high"].to_numpy(dtype=float) * 100.0
    a_lo = df_groups["trace1_ci95_low_any_visit_full"].to_numpy(dtype=float) * 100.0
    a_hi = df_groups["trace1_ci95_high_any_visit_full"].to_numpy(dtype=float) * 100.0
    ax_b.errorbar(
        x - 1.5 * w,
        r_paper,
        yerr=[np.maximum(0.0, r_paper - p_lo), np.maximum(0.0, p_hi - r_paper)],
        fmt="none",
        ecolor="#5C5C5C",
        elinewidth=0.8,
        capsize=2.0,
    )
    ax_b.errorbar(
        x - 0.5 * w,
        r_any_full,
        yerr=[np.maximum(0.0, r_any_full - a_lo), np.maximum(0.0, a_hi - r_any_full)],
        fmt="none",
        ecolor="#111111",
        elinewidth=0.8,
        capsize=2.0,
    )

    ax_b.axvline(4.5, color="#C9C9C9", linestyle="--", linewidth=0.9)
    ax_b.axvline(9.5, color="#C9C9C9", linestyle="--", linewidth=0.9)
    ax_b.text(2.0, 88.5, "High-killing (1–5)\nPaper 57.5% | v2 Any 52.1% | Sust 37.7%", ha="center", fontsize=7.6, fontweight="bold", color="#1F4E79")
    ax_b.text(7.0, 88.5, "Low-killing (6–10)\nPaper 27.0% | v2 Any 24.4% (200m: 21.2%)", ha="center", fontsize=7.6, fontweight="bold", color="#2E7D5B")
    ax_b.text(11.0, 88.5, "Dasatinib (8*–10*)\nPaper 4.5% | v2 Any 5.6% (200m: 4.5%)", ha="center", fontsize=7.6, fontweight="bold", color="#C0561F")

    xlabels = [
        f"G{row['group_id']}\n({int(row['paper_n_killed'])}/{int(row['paper_n_cells'])})"
        for _, row in df_groups.iterrows()
    ]
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xlabels, fontsize=7.8)
    ax_b.set_ylim(0, 114)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.set_xlim(-0.65, len(df_groups) - 0.35)
    ax_b.set_ylabel("Target kill fraction (%)", fontsize=9.5, fontweight="bold")
    ax_b.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E3E3E3",
        fontsize=7.1,
        ncol=3,
    )
    ax_b.text(
        -0.06,
        1.06,
        "(b) Per-group kill rate agreement vs paper (trace_num == 1 chambers, Full vs 200-min)",
        transform=ax_b.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color="#111111",
    )

    # -------------------------------------------------------------------------
    # Panels (c) and (d): Kaplan-Meier Target Survival Curves (Sustained vs Any-Visit)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    cond_Order = [
        ("Untreated high-killing CAR-T", "#1F4E79", "-", "High-killing (Groups 1–5)"),
        ("Untreated low-killing CAR-T", "#2E7D5B", "-", "Low-killing (Groups 6–10)"),
        ("Dasatinib 50 nM CAR-T", "#C0561F", "--", "Dasatinib / Latrunculin (Groups 8*–10*)"),
    ]
    paper_surv_by_cond = {
        "Untreated high-killing CAR-T": (1.0 - 96.0 / 167.0) * 100.0,
        "Untreated low-killing CAR-T": (1.0 - 44.0 / 163.0) * 100.0,
        "Dasatinib 50 nM CAR-T": (1.0 - 4.0 / 89.0) * 100.0,
    }

    df_trk1 = df_chambers[df_chambers["included_by_trace_rule"]].copy()

    for ax_km, time_col, event_col, time_200_col, event_200_col, panel_title in [
        (
            ax_c,
            "km_contact_time_min",
            "km_event_sustained",
            "km_contact_time_min_200m",
            "km_event_sustained_200m",
            "(c) Kaplan-Meier survival from first contact — Sustained (>= 2 visits)",
        ),
        (
            ax_d,
            "km_any_contact_time_min",
            "km_event_any_visit",
            "km_any_contact_time_min_200m",
            "km_event_any_visit_200m",
            "(d) Kaplan-Meier survival from first contact — Any-visit (>= 1 visit)",
        ),
    ]:
        # Thin per-group KM curves
        for gid, exp, cond in ALL_EXPERIMENTS:
            sub_g = df_trk1[(df_trk1["experiment"] == exp) & df_trk1[time_col].notna()]
            if sub_g.empty:
                continue
            color = (
                "#1F4E79"
                if "high-killing" in cond
                else ("#2E7D5B" if "low-killing" in cond else "#C0561F")
            )
            ls = "--" if "Dasatinib" in cond else "-"
            t_arr = sub_g[time_col].to_numpy(dtype=float)
            e_arr = sub_g[event_col].to_numpy(dtype=int)
            t_step, s_step, _ = compute_km_curve(t_arr, e_arr)
            ax_km.step(t_step, s_step * 100.0, where="post", color=color, linestyle=ls, linewidth=0.9, alpha=0.35)

        # Bold pooled condition KM curves (Full recording) + 200-min survival callout
        for cond_name, color, ls, short_lbl in cond_Order:
            sub_c = df_trk1[(df_trk1["condition"] == cond_name) & df_trk1[time_col].notna()]
            if sub_c.empty:
                continue
            t_arr = sub_c[time_col].to_numpy(dtype=float)
            e_arr = sub_c[event_col].to_numpy(dtype=int)
            t_step, s_step, c_times = compute_km_curve(t_arr, e_arr)
            ax_km.step(t_step, s_step * 100.0, where="post", color=color, linestyle=ls, linewidth=2.4, zorder=4)
            if c_times.size > 0:
                c_surv = np.interp(c_times, t_step, s_step)
                ax_km.scatter(
                    c_times,
                    c_surv * 100.0,
                    marker="|",
                    s=28,
                    color=color,
                    linewidths=1.0,
                    zorder=5,
                )

            # Compute 200-min window pooled event count as well
            sub_c200 = df_trk1[(df_trk1["condition"] == cond_name) & df_trk1[time_200_col].notna()]
            n_ev_full = int(e_arr.sum())
            n_ev_200 = int(sub_c200[event_200_col].sum())
            n_tot_c = len(t_arr)
            end_lbl = f"{s_step[-1]*100:.1f}% (kills: {n_ev_full}/{n_tot_c} full, {n_ev_200}/{len(sub_c200)} <=200m)"
            y_pos = float(s_step[-1] * 100.0)
            if "Dasatinib" in cond_name:
                y_pos = min(98.5, y_pos + 2.0)
            ax_km.text(
                t_step[-1] + 2.5,
                y_pos,
                end_lbl,
                fontsize=7.4,
                color=color,
                va="center",
                fontweight="bold",
            )

            # Paper reference survival tick at right edge
            p_surv = paper_surv_by_cond[cond_name]
            ax_km.scatter(
                [285.0],
                [p_surv],
                marker="D",
                s=30,
                facecolors="#FFFFFF",
                edgecolors=color,
                linewidths=1.5,
                zorder=6,
            )

        ax_km.axvline(190.0, color="#5C5C5C", linestyle=":", linewidth=0.9)
        ax_km.text(
            188.0,
            8.0,
            "200-min window limit\n(max dt <= 190 min)",
            fontsize=7.1,
            color="#5C5C5C",
            ha="right",
        )

        legend_lines = [
            Line2D([0], [0], color="#1F4E79", lw=2.3, linestyle="-", label="Untreated high-killing (Groups 1–5; thin = group, bold = pooled)"),
            Line2D([0], [0], color="#2E7D5B", lw=2.3, linestyle="-", label="Untreated low-killing (Groups 6–10; thin = group, bold = pooled)"),
            Line2D([0], [0], color="#C0561F", lw=2.3, linestyle="--", label="Dasatinib / Latrunculin (Groups 8*–10*; thin = group, bold = pooled)"),
            Line2D([0], [0], color="#111111", marker="D", markerfacecolor="#FFFFFF", lw=0, label="Paper Figure 5g final survival (100% - kill%)"),
        ]
        ax_km.legend(
            handles=legend_lines,
            loc="lower left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#E3E3E3",
            fontsize=7.3,
        )
        ax_km.set_xlim(0, 295)
        ax_km.set_ylim(0, 105)
        ax_km.set_xlabel("Time from first T-cell contact (min)", fontsize=9.5, fontweight="bold")
        ax_km.set_ylabel("Target survival (%)", fontsize=9.5, fontweight="bold")
        ax_km.grid(axis="both", color="#E3E3E3", linewidth=0.6)
        ax_km.set_axisbelow(True)
        ax_km.spines["top"].set_visible(False)
        ax_km.spines["right"].set_visible(False)
        ax_km.text(
            -0.06,
            1.06,
            panel_title,
            transform=ax_km.transAxes,
            fontsize=10.8,
            fontweight="bold",
            color="#111111",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4 (v2 default, --v1 legacy): compute per-chamber and per-visit target cell death labels across all 13 experiments (502 chambers)."
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Run the legacy v1 all-target pipeline instead of v2.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional single experiment folder name to run (default: all 13 experiments).",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run only the 3 pilot experiments (Groups 1-3, v1 mode).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker processes (default: 6 for v2 leaving CPU headroom on 10-core Mac).",
    )
    parser.add_argument(
        "--reuse-visits",
        action="store_true",
        help="Reuse existing data/step4/per_visit_death_labels_all_v2.csv if present.",
    )
    args = parser.parse_args()

    if args.pilot_only:
        exp_list = PILOT_EXPERIMENTS
    elif args.experiment is not None:
        matched = tuple(item for item in ALL_EXPERIMENTS if item[1] == args.experiment)
        if not matched:
            raise ValueError(f"Unknown experiment: {args.experiment}")
        exp_list = matched
    else:
        exp_list = ALL_EXPERIMENTS

    t0 = time.perf_counter()
    STEP4_DIR.mkdir(parents=True, exist_ok=True)

    if not args.v1:
        workers = max(1, min(args.workers, 8))
        print(
            f"1. Running Step 4 v2 (contacted target instance + MIP_ROI_0 artifact exclusion + "
            f"full vs 200-min window) across {len(exp_list)} experiment(s) (workers={workers})..."
        )
        (
            df_visits_v2,
            df_chambers_v2,
            df_groups_v2,
            df_conditions_v2,
            df_artifact_roi0,
        ) = build_chamber_and_visit_tables_v2(
            experiments=exp_list,
            workers=workers,
            reuse_visits=args.reuse_visits,
        )

        # Save per-experiment v2 tables without overwriting v1 files
        for _gid, exp, _cond in exp_list:
            exp_dir = STEP4_DIR / exp
            exp_dir.mkdir(parents=True, exist_ok=True)
            df_visits_v2[df_visits_v2["experiment"] == exp].to_csv(
                exp_dir / "per_visit_death_labels_v2.csv", index=False
            )
            df_chambers_v2[df_chambers_v2["experiment"] == exp].to_csv(
                exp_dir / "per_chamber_death_labels_v2.csv", index=False
            )

        elapsed = time.perf_counter() - t0

        if len(exp_list) == len(ALL_EXPERIMENTS):
            visits_v2_csv = STEP4_DIR / "per_visit_death_labels_all_v2.csv"
            chambers_v2_csv = STEP4_DIR / "per_chamber_death_labels_all_v2.csv"
            groups_v2_csv = STEP4_DIR / "group_kill_rate_comparison_all_v2.csv"
            cond_v2_csv = STEP4_DIR / "condition_kill_rate_comparison_v2.csv"
            art_v2_csv = STEP4_DIR / "mip_roi0_artifact_characterization_v2.csv"

            df_visits_v2.to_csv(visits_v2_csv, index=False)
            df_chambers_v2.to_csv(chambers_v2_csv, index=False)
            df_groups_v2.to_csv(groups_v2_csv, index=False)
            df_conditions_v2.to_csv(cond_v2_csv, index=False)
            df_artifact_roi0.to_csv(art_v2_csv, index=False)

            print("2. Generating figure figures/step4_death_labels_v2.png...")
            fig_v2_path = FIGURES_DIR / "step4_death_labels_v2.png"
            make_v2_figure(
                df_visits_v2,
                df_chambers_v2,
                df_groups_v2,
                df_artifact_roi0,
                fig_v2_path,
            )

            summary_v2_payload = {
                "version": "v2",
                "elapsed_s": round(elapsed, 2),
                "workers": workers,
                "n_experiments": len(df_groups_v2),
                "n_chambers_total": len(df_chambers_v2),
                "n_visits_total": len(df_visits_v2),
                "n_trace1_chambers": int(df_chambers_v2["included_by_trace_rule"].sum()),
                "n_non_trace1_chambers_flagged": int(df_chambers_v2["non_single_track_flag"].sum()),
                "n_artifact_405_visits_flagged": int(df_visits_v2["artifact_405_band"].sum()),
                "n_artifact_405_chambers_flagged": int(df_chambers_v2["had_artifact_405_band"].sum()),
                "n_mip_roi0_false_positive_chambers_eliminated": int(
                    df_artifact_roi0["false_positive_eliminated"].sum()
                ),
                "rule_parameters": {
                    "gauss_sigma_2d": GAUSS_SIGMA_2D,
                    "gauss_truncate": GAUSS_TRUNCATE,
                    "threshold_3d_adu": THRESH_3D_ADU,
                    "threshold_2d_adu": THRESH_2D_ADU,
                    "min_component_area_2d_px": MIN_COMPONENT_AREA_2D,
                    "min_overlap_px_2d": MIN_OVERLAP_PX_2D,
                    "visit_interval_min": VISIT_INTERVAL_MIN,
                    "sustained_consecutive_visits": 2,
                    "fixed_window_max_visit": FIXED_WINDOW_MAX_VISIT,
                    "fixed_window_duration_min": FIXED_WINDOW_MAX_VISIT * VISIT_INTERVAL_MIN,
                    "artifact_rule": "np.median(mip_xy[:, 0]) > 404.0 and mip_xy.max() > 5000",
                    "target_instance_rule": "instance_seg_matlab_xyz(pred==2, pred==4) contacted by tracked T cell (trace_num==1) or any touching T cell (trace_num!=1, flagged)",
                },
                "condition_comparison": df_conditions_v2.to_dict(orient="records"),
                "group_comparison": df_groups_v2.to_dict(orient="records"),
                "mip_roi0_artifact_summary": df_artifact_roi0.to_dict(orient="records"),
            }
            with open(STEP4_DIR / "summary_all_v2.json", "w", encoding="utf-8") as f:
                json.dump(summary_v2_payload, f, indent=2)

        print("3. Pooled regime comparison (trace_num == 1 vs Paper):")
        print(
            df_conditions_v2[
                [
                    "regime",
                    "paper_n_cells",
                    "paper_n_killed",
                    "paper_kill_rate",
                    "trace1_chambers_n",
                    "trace1_killed_any_visit_full",
                    "trace1_kill_rate_any_visit_full",
                    "trace1_killed_any_visit_200m",
                    "trace1_kill_rate_any_visit_200m",
                    "trace1_killed_sustained_full",
                    "trace1_kill_rate_sustained_full",
                    "trace1_killed_sustained_200m",
                    "trace1_kill_rate_sustained_200m",
                ]
            ].to_string(index=False)
        )
        print(f"Done in {elapsed:.1f}s.")
        return 0

    workers = max(1, min(args.workers, 2))
    print("1. Running 3D vs. 2D calibration on 20 local raw volumes...")
    df_cal = calibrate_on_local_raw_volumes()
    cal_csv = STEP4_DIR / "calibration_20_raw_volumes.csv"
    df_cal.to_csv(cal_csv, index=False)

    n_raw = len(df_cal)
    n_tif = int(df_cal["has_deposited_tif"].sum())
    exact_xy = int((df_cal[df_cal["has_deposited_tif"]]["mip_xy_max_abs_diff"] == 0).sum())
    exact_xz = int((df_cal[df_cal["has_deposited_tif"]]["mip_xz_max_abs_diff"] == 0).sum())
    agree_binst = int((df_cal["dec_2d_direct"] == df_cal["dec_3d_binst_within10"]).sum())
    agree_rawb_direct = int((df_cal["dec_2d_direct"] == df_cal["dec_3d_rawb_within10"]).sum())
    agree_rawb_dil10 = int((df_cal["dec_2d_dil10"] == df_cal["dec_3d_rawb_within10"]).sum())
    print(
        f"   Verified MIP geometry on {n_tif}/{n_raw} raw volumes with deposited TIFFs: "
        f"exact MIPxy={exact_xy}/{n_tif}, exact MIPxz={exact_xz}/{n_tif}"
    )
    print(
        f"   2D vs. 3D agreement on {n_raw} volumes: "
        f"direct vs 3D b_instance={agree_binst}/{n_raw} ({100*agree_binst/n_raw:.1f}%), "
        f"direct vs 3D raw_target={agree_rawb_direct}/{n_raw} ({100*agree_rawb_direct/n_raw:.1f}%), "
        f"dil10 vs 3D raw_target={agree_rawb_dil10}/{n_raw} ({100*agree_rawb_dil10/n_raw:.1f}%)"
    )

    print(
        f"2. Computing per-visit and per-chamber death labels across {len(exp_list)} experiment(s) "
        f"(workers={workers})..."
    )
    df_visits, df_chambers, df_groups = build_chamber_and_visit_tables(
        experiments=exp_list,
        workers=workers,
    )

    # Save per-experiment tables under data/step4/<experiment>/
    for _gid, exp, _cond in exp_list:
        exp_dir = STEP4_DIR / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        df_visits[df_visits["experiment"] == exp].to_csv(
            exp_dir / "per_visit_death_labels.csv", index=False
        )
        df_chambers[df_chambers["experiment"] == exp].to_csv(
            exp_dir / "per_chamber_death_labels.csv", index=False
        )

    elapsed = time.perf_counter() - t0
    rule_params = {
        "gauss_sigma_2d": GAUSS_SIGMA_2D,
        "gauss_truncate": GAUSS_TRUNCATE,
        "threshold_3d_adu": THRESH_3D_ADU,
        "threshold_2d_adu": THRESH_2D_ADU,
        "min_component_area_2d_px": MIN_COMPONENT_AREA_2D,
        "min_overlap_px_2d": MIN_OVERLAP_PX_2D,
        "visit_interval_min": VISIT_INTERVAL_MIN,
        "sustained_consecutive_visits": 2,
    }
    cal_summary = {
        "n_raw_volumes": n_raw,
        "n_with_deposited_tifs": n_tif,
        "exact_mip_xy_matches": exact_xy,
        "exact_mip_xz_matches": exact_xz,
        "agreement_direct_vs_3d_binst": f"{agree_binst}/{n_raw}",
        "agreement_direct_vs_3d_raw_target": f"{agree_rawb_direct}/{n_raw}",
        "agreement_dil10_vs_3d_raw_target": f"{agree_rawb_dil10}/{n_raw}",
    }

    if len(exp_list) == len(ALL_EXPERIMENTS):
        # Save all-experiment tables (keeping pilot files intact)
        visits_all_csv = STEP4_DIR / "per_visit_death_labels_all.csv"
        chambers_all_csv = STEP4_DIR / "per_chamber_death_labels_all.csv"
        groups_all_csv = STEP4_DIR / "group_kill_rate_comparison_all.csv"
        df_visits.to_csv(visits_all_csv, index=False)
        df_chambers.to_csv(chambers_all_csv, index=False)
        df_groups.to_csv(groups_all_csv, index=False)

        print("3. Generating all-experiment figure figures/step4_death_labels_all.png...")
        fig_all_path = FIGURES_DIR / "step4_death_labels_all.png"
        make_all_figure(df_chambers, df_groups, fig_all_path)

        summary_all_payload = {
            "elapsed_s": round(elapsed, 2),
            "n_experiments": len(df_groups),
            "n_chambers_total": len(df_chambers),
            "n_visits_total": len(df_visits),
            "n_paper_firm_probable_total": int(df_chambers["in_paper_firm_probable"].sum()),
            "n_paper_all419_total": int(df_chambers["in_paper"].sum()),
            "n_dead_at_visit_1_all": int(df_chambers["dead_at_visit_1"].sum()),
            "n_dead_at_visit_1_paper_firm_probable": int(
                df_chambers[df_chambers["in_paper_firm_probable"]]["dead_at_visit_1"].sum()
            ),
            "n_no_contact_all": int(df_chambers["no_contact"].sum()),
            "n_no_contact_paper_firm_probable": int(
                df_chambers[df_chambers["in_paper_firm_probable"]]["no_contact"].sum()
            ),
            "n_censored_all": int(df_chambers["censored"].sum()),
            "n_truncated_chambers": int(df_chambers["is_truncated"].sum()),
            "rule_parameters": rule_params,
            "calibration_summary": cal_summary,
            "group_comparison": df_groups.to_dict(orient="records"),
        }
        with open(STEP4_DIR / "summary_all.json", "w", encoding="utf-8") as f:
            json.dump(summary_all_payload, f, indent=2)

    elif args.pilot_only:
        visits_csv = STEP4_DIR / "per_visit_death_labels.csv"
        chambers_csv = STEP4_DIR / "per_chamber_death_labels.csv"
        groups_csv = STEP4_DIR / "group_kill_rate_comparison.csv"
        df_visits.to_csv(visits_csv, index=False)
        df_chambers.to_csv(chambers_csv, index=False)
        df_groups.to_csv(groups_csv, index=False)
        fig_path = FIGURES_DIR / "step4_death_labels_pilot.png"
        make_pilot_figure(df_visits, df_chambers, df_groups, fig_path)

    print("4. Summary of per-group comparison:")
    print(
        df_groups[
            [
                "group_id",
                "paper_n_cells",
                "paper_n_killed",
                "paper_group_kill_rate_exact",
                "our_paper_chambers_n",
                "our_paper_killed_sustained",
                "our_paper_kill_rate_sustained",
                "our_paper_killed_any_visit",
                "our_paper_kill_rate_any_visit",
                "our_all_chambers_n",
                "our_all_killed_sustained",
                "our_all_kill_rate_sustained",
                "our_all_killed_any_visit",
                "our_all_kill_rate_any_visit",
            ]
        ].to_string(index=False)
    )
    print(f"Done in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
