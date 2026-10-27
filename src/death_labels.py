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
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, label
from scipy.stats import beta
import tifffile

try:
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from reproduce_contact_area import (
    STRUCT26,
    compute_contact_surface_from_mask_array,
    sphere_dilate_edt,
)

MIPS_DIR: Path = DATA_DIR / "scidb_19235_mips"
MIPS_MANIFEST_CSV: Path = MIPS_DIR / "manifest.csv"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
CHAMBERS_DIR: Path = DATA_DIR / "scidb_19235_chambers"
STEP3_DIR: Path = DATA_DIR / "step3"
STEP4_DIR: Path = DATA_DIR / "step4"
CLEAN_419_CSV: Path = DATA_DIR / "clean_per_cell_419.csv"
ROW_MAP_CSV: Path = REPO_ROOT / "results" / "chamber_to_row_map.csv"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4: calibrate 2D dead-dye MIP rule and compute per-chamber death labels across all 13 experiments (502 chambers)."
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
        help="Run only the 3 pilot experiments (Groups 1-3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes (default: 2, capped at 2).",
    )
    args = parser.parse_args()
    workers = max(1, min(args.workers, 2))

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
