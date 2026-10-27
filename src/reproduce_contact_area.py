#!/usr/bin/env python3
"""
Step 3 (Part 1): Reproduce the Fei lab's published per-cell 3D contact area (CA)
from the deposited 3D U-Net masks (`*_predictions.h5`) using the exact definition
in `external/feilab_CART_analysis/main/analysis/`:
  - `step2_instance_seg.m` (`instance_seg` and 3D Voronoi/EDT contact surface loop)
  - `interface/cl_factor.m` (26-connected components of `surface == j`)
  - `step3_analysis.m` (mean over sweeps 1..5 where `s > 0` per visit)
  - `summary/summary.m` (max over visits times `0.0385 um^2`)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, spearmanr

try:
    from src.constants import VOXEL_UM
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from constants import VOXEL_UM
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
MANIFEST_CSV: Path = MASKS_DIR / "manifest.csv"
CLEAN_419_CSV: Path = DATA_DIR / "clean_per_cell_419.csv"
STEP3_DIR: Path = DATA_DIR / "step3"

# summary.m line 18: write_summary(..., 'surface_result.csv', ..., 0.0385, true)
# Consistent with VOXEL_UM = 0.1962 um (0.1962^2 = 0.03849444 um^2/voxel)
CA_VOXEL_FACTOR_UM2: float = 0.0385

# 26-connectivity structuring element in 3D
STRUCT26: np.ndarray = np.ones((3, 3, 3), dtype=bool)

# Experiment-to-group mapping from src/run_dent_stiffness_fixes_all_frames.py (lines 141-155)
V_TO_GROUP: dict[str, tuple[str, str]] = {
    "v01_6_40_120_0-6_hd1":    ("1",   "Untreated high-killing"),
    "v02_6_40_120_0-6_hd3":    ("2",   "Untreated high-killing"),
    "v03_6_40_120_0-6_hd4":    ("3",   "Untreated high-killing"),
    "v04_6_40_120_0-6_hd6":    ("4",   "Untreated high-killing"),
    "v05_6_40_120_0-6_hd7":    ("5",   "Untreated high-killing"),
    "v06_6_40_120_0-6_hd8":    ("6",   "Untreated low-killing"),
    "v07_6_39_120_0-6_hd9":    ("7",   "Untreated low-killing"),
    "v08_6_40_120_0-6_hd2":    ("8",   "Untreated low-killing"),
    "v09_6_40_120_0-6_hd10":   ("9",   "Untreated low-killing"),
    "v10_6_40_120_0-6_hd5":    ("10",  "Untreated low-killing"),
    "v11_6_36_120_0-6_hd2_ds": ("8*",  "Dasatinib 50 nM"),
    "v12_6_35_120_0-6_hd10ds": ("9*",  "Dasatinib 50 nM"),
    "v13_6_40_120_0-6_hd5_ds": ("10*", "Dasatinib 50 nM"),
}


def bbox_slices(
    mask: np.ndarray, pad: int, shape: tuple[int, int, int]
) -> tuple[slice, slice, slice] | None:
    """Return 3D bounding-box slices of nonzero elements in `mask`, padded by `pad` voxels."""
    z_any = np.any(mask, axis=(1, 2))
    if not np.any(z_any):
        return None
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))
    z_idx = np.where(z_any)[0]
    y_idx = np.where(y_any)[0]
    x_idx = np.where(x_any)[0]
    z0, z1 = max(0, int(z_idx[0]) - pad), min(shape[0], int(z_idx[-1]) + pad + 1)
    y0, y1 = max(0, int(y_idx[0]) - pad), min(shape[1], int(y_idx[-1]) + pad + 1)
    x0, x1 = max(0, int(x_idx[0]) - pad), min(shape[2], int(x_idx[-1]) + pad + 1)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def imfill_holes_26(mask: np.ndarray) -> np.ndarray:
    """
    3D hole filling matching MATLAB `imfill(mask, 26, 'holes')`.
    Padding by 1 voxel around the bounding box (clamped to volume boundaries) preserves
    all boundary connections identically to full-volume hole filling.
    """
    sl = bbox_slices(mask, 1, mask.shape)
    if sl is None:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[sl] = binary_fill_holes(mask[sl], structure=STRUCT26)
    return out


def sphere_dilate_edt(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Exact Euclidean sphere dilation of `mask` by `radius` voxels using EDT threshold:
    `distance_transform_edt(~mask) <= radius`.
    """
    pad = int(np.ceil(radius)) + 1
    sl = bbox_slices(mask, pad, mask.shape)
    if sl is None:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[sl] = distance_transform_edt(~mask[sl]) <= radius
    return out


def bwareaopen_26(
    mask: np.ndarray, min_size: int = 3000
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove 26-connected components smaller than `min_size` voxels (`bwareaopen(mask, 3000, 26)`),
    and return `(cleaned_mask, relabeled_array, n_components)`.
    """
    lbl, n_comp = label(mask, structure=STRUCT26)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool), lbl, 0
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    keep_ids = np.where(counts >= min_size)[0]
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(lbl), 0
    remap = np.zeros(n_comp + 1, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, start=1):
        remap[old_id] = new_id
    relabeled = remap[lbl]
    return relabeled > 0, relabeled, int(keep_ids.size)


def instance_seg(nucleus: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """
    Exact translation of `instance_seg(nucleus, membrane)` from `step2_instance_seg.m` (lines 150-220).
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
            _, inds = distance_transform_edt(~nuc_clean[sl], return_indices=True)
            sub_inst = nuc_lbl[sl][tuple(inds)].astype(np.int8)
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


def compute_contact_surface_from_mask_array(
    I: np.ndarray, return_volumes: bool = False
) -> dict[str, Any]:
    """
    Run `instance_seg` and the `step2_instance_seg.m` / `cl_factor.m` contact surface
    construction on a 3D uint8 label volume `I` (0=bg, 1=T-cell body, 2=target interior,
    3=T-cell dense actin, 4=target membrane).
    """
    sl_all = bbox_slices(I > 0, 16, I.shape)
    if sl_all is None:
        out = {
            "n_tcell_instances": 0,
            "n_target_instances": 0,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }
        if return_volumes:
            out["t_instance"] = np.zeros_like(I, dtype=np.int8)
            out["b_instance"] = np.zeros_like(I, dtype=np.int8)
            out["surface"] = np.zeros_like(I, dtype=np.int8)
        return out

    I_sub = I[sl_all]
    t_inst = instance_seg(I_sub == 1, I_sub == 3)
    b_inst = instance_seg(I_sub == 2, I_sub == 4)

    t_ids = [int(x) for x in np.unique(t_inst) if x > 0]
    b_ids = [int(x) for x in np.unique(b_inst) if x > 0]
    n_tcell = len(t_ids)
    n_target = len(b_ids)

    if n_tcell == 0 or n_target == 0:
        out = {
            "n_tcell_instances": n_tcell,
            "n_target_instances": n_target,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }
        if return_volumes:
            full_t = np.zeros_like(I, dtype=np.int8)
            full_b = np.zeros_like(I, dtype=np.int8)
            full_t[sl_all] = t_inst
            full_b[sl_all] = b_inst
            out["t_instance"] = full_t
            out["b_instance"] = full_b
            out["surface"] = np.zeros_like(I, dtype=np.int8)
        return out

    b_mask_full = b_inst != 0
    sl_cells = bbox_slices(b_mask_full | (t_inst != 0), 16, I_sub.shape)
    t_c = t_inst[sl_cells]
    b_c = b_inst[sl_cells]
    b_mask = b_c != 0

    edt_b = distance_transform_edt(~b_mask)
    not_b_dilate = ~(edt_b <= 10.0)
    b_round = edt_b <= 3.0

    # Find all T-cell instances touching b_round (target dilated by radius-3 sphere)
    touching_t: list[tuple[int, int]] = []
    for j in t_ids:
        t_j = t_c == j
        if np.any(b_round & t_j):
            touching_t.append((int(np.sum(t_j)), j))

    if not touching_t:
        out = {
            "n_tcell_instances": n_tcell,
            "n_target_instances": n_target,
            "n_touching_tcell_instances": 0,
            "s_largest": 0,
            "s_total": 0,
            "n_components": 0,
            "s_with_edge": 0,
        }
        if return_volumes:
            full_t = np.zeros_like(I, dtype=np.int8)
            full_b = np.zeros_like(I, dtype=np.int8)
            full_t[sl_all] = t_inst
            full_b[sl_all] = b_inst
            out["t_instance"] = full_t
            out["b_instance"] = full_b
            out["surface"] = np.zeros_like(I, dtype=np.int8)
        return out

    # If several T-cell instances are in contact with the target, choose the largest
    touching_t.sort(reverse=True)
    _, j_sel = touching_t[0]
    t_actin = t_c == j_sel

    t_dilate = distance_transform_edt(~t_actin) <= 10.0
    bg_raw = (~t_dilate) & not_b_dilate
    bg_lbl, n_bg = label(bg_raw, structure=STRUCT26)
    if n_bg == 0:
        background = np.zeros_like(bg_raw, dtype=bool)
    else:
        bg_counts = np.bincount(bg_lbl.ravel())
        bg_counts[0] = 0
        background = bg_lbl == int(np.argmax(bg_counts))

    map_origin = (
        b_mask.astype(np.int8) * 3
        + t_actin.astype(np.int8) * 2
        + background.astype(np.int8)
    )
    _, idx = distance_transform_edt(map_origin == 0, return_indices=True)
    map_exp = map_origin[tuple(idx)]

    surface_mask = (map_exp == 2) & (distance_transform_edt(map_exp != 3) <= 1.0)
    edge_mask = surface_mask & (distance_transform_edt(map_exp != 1) <= 5.0)

    # In step2_instance_seg.m lines 89-90:
    #   surface(surface_mask) = j;
    #   surface(edge_mask) = -j;
    # And in cl_factor.m line 3:
    #   s_real = s_mask == mask_index;  (voxels with label +j, i.e. surface_mask & ~edge_mask)
    s_real = surface_mask & (~edge_mask)
    s_total = int(np.sum(s_real))
    s_with_edge = int(np.sum(surface_mask))

    if s_total == 0:
        s_largest = 0
        n_components = 0
    else:
        s_lbl, n_components = label(s_real, structure=STRUCT26)
        s_counts = np.bincount(s_lbl.ravel())
        s_counts[0] = 0
        s_largest = int(np.max(s_counts))

    out = {
        "n_tcell_instances": n_tcell,
        "n_target_instances": n_target,
        "n_touching_tcell_instances": len(touching_t),
        "s_largest": s_largest,
        "s_total": s_total,
        "n_components": int(n_components),
        "s_with_edge": s_with_edge,
    }

    if return_volumes:
        full_t = np.zeros_like(I, dtype=np.int8)
        full_b = np.zeros_like(I, dtype=np.int8)
        full_surf = np.zeros_like(I, dtype=np.int8)
        sub_surf = np.zeros_like(t_inst, dtype=np.int8)
        cell_surf = np.zeros_like(t_c, dtype=np.int8)
        cell_surf[surface_mask] = j_sel
        cell_surf[edge_mask] = -j_sel
        sub_surf[sl_cells] = cell_surf
        full_t[sl_all] = t_inst
        full_b[sl_all] = b_inst
        full_surf[sl_all] = sub_surf
        out["t_instance"] = full_t
        out["b_instance"] = full_b
        out["surface"] = full_surf
        out["selected_tcell_id"] = j_sel

    return out


def process_one_file(task: tuple[str, str, int, int, str]) -> dict[str, Any]:
    """Worker function for multiprocessing over mask files."""
    experiment, roi, visit_index, sweep_index, rel_path = task
    abs_path = REPO_ROOT / rel_path
    t0 = time.perf_counter()
    with h5py.File(abs_path, "r") as hf:
        I = hf["predictions"][:]
    res = compute_contact_surface_from_mask_array(I, return_volumes=False)
    elapsed_s = time.perf_counter() - t0
    return {
        "experiment": experiment,
        "roi": roi,
        "visit": int(visit_index),
        "sweep": int(sweep_index),
        "n_tcell_instances": res["n_tcell_instances"],
        "n_target_instances": res["n_target_instances"],
        "n_touching_tcell_instances": res["n_touching_tcell_instances"],
        "s_largest": res["s_largest"],
        "s_total": res["s_total"],
        "n_components": res["n_components"],
        "s_with_edge": res["s_with_edge"],
        "elapsed_s": elapsed_s,
    }


def discover_complete_chambers(
    manifest_df: pd.DataFrame,
    experiment_filter: str | None = None,
    roi_filter: str | None = None,
) -> list[tuple[str, str, pd.DataFrame]]:
    """
    Return `(experiment, roi, chamber_df)` for chambers where all expected files
    are present on disk and non-empty (or marked `ok == True` in `manifest.csv`).
    """
    df = manifest_df
    if experiment_filter is not None:
        df = df[df["experiment"] == experiment_filter]
    if roi_filter is not None:
        df = df[df["roi"] == roi_filter]

    complete_chambers: list[tuple[str, str, pd.DataFrame]] = []
    grouped = df.groupby(["experiment", "roi"], sort=False)
    for (exp, roi), cdf in grouped:
        all_ok = True
        for _, row in cdf.iterrows():
            if bool(row.get("ok", False)) is True or str(row.get("ok", "")) == "True":
                p = REPO_ROOT / str(row["path"])
                if not p.exists() or p.stat().st_size == 0:
                    all_ok = False
                    break
            else:
                p = REPO_ROOT / str(row["path"])
                if not p.exists() or p.stat().st_size == 0:
                    all_ok = False
                    break
        if all_ok:
            complete_chambers.append((str(exp), str(roi), cdf))

    complete_chambers.sort(
        key=lambda x: (
            x[0],
            int(x[1].split("_")[-1]) if x[1].split("_")[-1].isdigit() else x[1],
        )
    )
    return complete_chambers


def save_sanity_check_slice_figure(
    h5_path: Path,
    out_path: Path,
    experiment: str,
    roi: str,
    visit: int,
    sweep: int,
) -> dict[str, Any]:
    """
    Save a sanity-check figure showing an xy slice through the contact interface
    with the raw mask labels, instance segmentation, and `surface` (+j interior vs -j edge) voxels.
    Follows house style (light paper background, system sans font, clean labels).
    """
    with h5py.File(h5_path, "r") as hf:
        I = hf["predictions"][:]
    res = compute_contact_surface_from_mask_array(I, return_volumes=True)
    surf = res["surface"]
    t_inst = res["t_instance"]
    b_inst = res["b_instance"]
    j_sel = res.get("selected_tcell_id", 1)

    s_real_mask = surf == j_sel
    s_edge_mask = surf == -j_sel
    z_s_counts = np.sum(s_real_mask, axis=(1, 2))
    z_t_counts = np.sum(t_inst == j_sel, axis=(1, 2))
    z_b_counts = np.sum(b_inst != 0, axis=(1, 2))
    both_present = (z_t_counts >= 400) & (z_b_counts >= 400) & (z_s_counts > 0)
    if np.any(both_present):
        score = np.where(both_present, z_s_counts * np.minimum(z_t_counts, z_b_counts), 0)
        z_slice = int(np.argmax(score))
    elif np.any(z_s_counts > 0):
        z_slice = int(np.argmax(z_s_counts))
    else:
        z_slice = I.shape[0] // 2

    # Crop display window around cells in xy
    cell_xy = np.any((t_inst != 0) | (b_inst != 0), axis=0)
    ys, xs = np.where(cell_xy)
    if ys.size > 0:
        y0, y1 = max(0, int(ys.min()) - 18), min(I.shape[1], int(ys.max()) + 19)
        x0, x1 = max(0, int(xs.min()) - 18), min(I.shape[2], int(xs.max()) + 19)
    else:
        y0, y1, x0, x1 = 0, I.shape[1], 0, I.shape[2]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#C9C9C9",
        "axes.labelcolor": "#1F1F1F",
        "xtick.color": "#5C5C5C",
        "ytick.color": "#5C5C5C",
        "text.color": "#111111",
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    extent_um = [
        x0 * VOXEL_UM,
        x1 * VOXEL_UM,
        y1 * VOXEL_UM,
        y0 * VOXEL_UM,
    ]

    # Panel 1: Deposited 4-class prediction mask at z = z_slice
    raw_rgb = np.full((y1 - y0, x1 - x0, 3), 0.965, dtype=float)
    raw_slice = I[z_slice, y0:y1, x0:x1]
    # Plot palette from house style:
    # T-cell body (1): light blue-slate; T-cell actin (3): #1F4E79
    # Target interior (2): light sand; Target membrane (4): #2E7D5B
    raw_rgb[raw_slice == 1] = [0.68, 0.78, 0.88]
    raw_rgb[raw_slice == 3] = [0.122, 0.306, 0.475]  # #1F4E79
    raw_rgb[raw_slice == 2] = [0.76, 0.88, 0.82]
    raw_rgb[raw_slice == 4] = [0.180, 0.490, 0.357]  # #2E7D5B

    axes[0].imshow(raw_rgb, extent=extent_um, interpolation="nearest")
    axes[0].set_title(
        f"Deposited mask (z = {z_slice}, {z_slice * VOXEL_UM:.1f} µm)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    axes[0].set_xlabel("x (µm)", fontsize=10)
    axes[0].set_ylabel("y (µm)", fontsize=10)

    # Panel 2: Filled instances + extracted contact surface (`s_real` and `edge_mask`)
    inst_rgb = np.full((y1 - y0, x1 - x0, 3), 0.965, dtype=float)
    t_sl = t_inst[z_slice, y0:y1, x0:x1] == j_sel
    b_sl = b_inst[z_slice, y0:y1, x0:x1] != 0
    s_real_sl = s_real_mask[z_slice, y0:y1, x0:x1]
    s_edge_sl = s_edge_mask[z_slice, y0:y1, x0:x1]

    inst_rgb[t_sl] = [0.76, 0.84, 0.92]
    inst_rgb[b_sl] = [0.80, 0.90, 0.85]
    inst_rgb[s_edge_sl] = [0.420, 0.298, 0.604]  # #6B4C9A (--plot-4, edge = -j)
    inst_rgb[s_real_sl] = [0.753, 0.337, 0.122]  # #C0561F (--plot-2, surface = +j)

    axes[1].imshow(inst_rgb, extent=extent_um, interpolation="nearest")
    axes[1].set_title(
        f"Contact surface (s = {res['s_largest']} vox, {res['s_largest'] * CA_VOXEL_FACTOR_UM2:.1f} µm²)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    axes[1].set_xlabel("x (µm)", fontsize=10)
    axes[1].set_ylabel("y (µm)", fontsize=10)

    # Direct inline text key on Panel 2
    axes[1].text(
        0.03,
        0.96,
        f"T-cell instance (blue)\nTarget instance (green)\nsurface == +j ({int(s_real_sl.sum())} px in slice, orange)\nedge == -j ({int(s_edge_sl.sum())} px in slice, purple)",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#1F1F1F",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="#FFFFFF", edgecolor="#E3E3E3", alpha=0.92),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "z_slice": z_slice,
        "s_largest": res["s_largest"],
        "s_total": res["s_total"],
        "n_components": res["n_components"],
        "s_with_edge": res["s_with_edge"],
        "slice_s_real_voxels": int(s_real_sl.sum()),
        "slice_s_edge_voxels": int(s_edge_sl.sum()),
        "figure_path": str(out_path.relative_to(REPO_ROOT)),
    }


def compare_with_paper_group1(
    chamber_df: pd.DataFrame,
    clean_419_path: Path = CLEAN_419_CSV,
) -> dict[str, Any]:
    """
    Compare our 40 v01 chamber CA values against the 33 Group-1 rows in `clean_per_cell_419.csv`:
      - Distribution statistics (medians, IQRs, two-sample KS test)
      - Optimal 1-to-1 assignment (`scipy.optimize.linear_sum_assignment` on `|log CA_our - log CA_paper|`)
      - Agreement counts within 2%, 5%, 10% relative error
      - Order preservation: Spearman correlation between matched ROI index and `cell_index`
      - Null/chance baseline: repeating the assignment 2,000 times against:
          (1) log-uniform random values in `[min(paper), max(paper)]`
          (2) bootstrap-resampled + perturbed draws from the empirical distribution
          (3) random permutations for the ROI-order Spearman test
      - Analysis of the 7 unmatched chambers
    """
    p419 = pd.read_csv(clean_419_path)
    g1 = p419[p419["group_id"].astype(str) == "1"].sort_values("cell_index").reset_index(drop=True)
    paper_ca = g1["CA_um2"].to_numpy(dtype=float)
    paper_cell_idx = g1["cell_index"].to_numpy(dtype=int)

    v01_df = chamber_df[chamber_df["experiment"] == "v01_6_40_120_0-6_hd1"].sort_values("roi_index").reset_index(drop=True)
    our_ca_all = v01_df["CA_um2"].to_numpy(dtype=float)
    our_pos_mask = our_ca_all > 0
    our_ca_pos = our_ca_all[our_pos_mask]

    ks_all_stat, ks_all_p = ks_2samp(our_ca_all, paper_ca)
    ks_pos_stat, ks_pos_p = ks_2samp(our_ca_pos, paper_ca) if len(our_ca_pos) > 0 else (np.nan, np.nan)

    # Cost matrix: |log(CA_our) - log(CA_paper)|, setting cost = 1e6 if CA_our <= 0
    n_paper = len(paper_ca)
    n_our = len(our_ca_all)
    eps = 1e-6
    cost = np.zeros((n_paper, n_our), dtype=float)
    for i in range(n_paper):
        for j in range(n_our):
            if our_ca_all[j] <= 0:
                cost[i, j] = 1e6
            else:
                cost[i, j] = abs(np.log(max(eps, our_ca_all[j])) - np.log(max(eps, paper_ca[i])))

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs: list[dict[str, Any]] = []
    rel_errors: list[float] = []
    abs_log_diffs: list[float] = []
    for r_i, c_j in zip(row_ind, col_ind):
        p_val = float(paper_ca[r_i])
        o_val = float(our_ca_all[c_j])
        rel_err = abs(o_val - p_val) / p_val if p_val > 0 else np.nan
        log_diff = abs(np.log(max(eps, o_val)) - np.log(max(eps, p_val)))
        rel_errors.append(rel_err)
        abs_log_diffs.append(log_diff)
        matched_pairs.append({
            "cell_index": int(paper_cell_idx[r_i]),
            "paper_CA_um2": round(p_val, 3),
            "roi": str(v01_df.loc[c_j, "roi"]),
            "roi_index": int(v01_df.loc[c_j, "roi_index"]),
            "our_CA_um2": round(o_val, 3),
            "our_CA_total_um2": round(float(v01_df.loc[c_j, "CA_total_um2"]), 3),
            "abs_diff_um2": round(abs(o_val - p_val), 3),
            "rel_err_pct": round(100.0 * rel_err, 3),
            "abs_log_diff": round(log_diff, 5),
            "max_tcell_instances": int(v01_df.loc[c_j, "max_tcell_instances"]),
            "multi_tcell_sweeps": int(v01_df.loc[c_j, "multi_tcell_sweeps"]),
            "n_contact_visits": int(v01_df.loc[c_j, "n_contact_visits"]),
        })

    rel_errors_arr = np.array(rel_errors, dtype=float)
    within_1pct = int(np.sum(rel_errors_arr <= 0.01))
    within_2pct = int(np.sum(rel_errors_arr <= 0.02))
    within_5pct = int(np.sum(rel_errors_arr <= 0.05))
    within_10pct = int(np.sum(rel_errors_arr <= 0.10))

    matched_cell_indices = np.array([m["cell_index"] for m in matched_pairs], dtype=int)
    matched_roi_indices = np.array([m["roi_index"] for m in matched_pairs], dtype=int)
    rho_order, p_order = spearmanr(matched_cell_indices, matched_roi_indices)

    # Also evaluate strict order-preserving (monotonic ROI order) dynamic programming / hypothesis:
    # What if the 33 Group-1 rows are simply 33 of the 40 chambers in strictly increasing ROI order?
    # Let's find the best order-preserving subset of 33 chambers via dynamic programming on cost!
    dp = np.full((n_paper + 1, n_our + 1), np.inf, dtype=float)
    parent = np.zeros((n_paper + 1, n_our + 1), dtype=int)
    dp[0, :] = 0.0
    for i in range(1, n_paper + 1):
        for j in range(i, n_our + 1):
            # Option 1: skip chamber j-1
            best_val = dp[i, j - 1]
            best_prev = j - 1
            # Option 2: match paper cell i-1 to chamber j-1
            cand = dp[i - 1, j - 1] + cost[i - 1, j - 1]
            if cand < best_val:
                best_val = cand
                best_prev = -(j - 1) - 1
            dp[i, j] = best_val
            parent[i, j] = best_prev

    mono_cols: list[int] = []
    cur_i, cur_j = n_paper, n_our
    while cur_i > 0:
        p_code = parent[cur_i, cur_j]
        if p_code < 0:
            col_idx = -p_code - 1
            mono_cols.append(col_idx)
            cur_i -= 1
            cur_j = col_idx
        else:
            cur_j = p_code
    mono_cols.reverse()
    mono_rel_errs = np.array([
        abs(our_ca_all[c_j] - paper_ca[r_i]) / paper_ca[r_i]
        for r_i, c_j in enumerate(mono_cols)
    ])

    # Unmatched 7 chambers from unconstrained assignment
    matched_col_set = set(int(x) for x in col_ind)
    unmatched_chambers: list[dict[str, Any]] = []
    for c_j in range(n_our):
        if c_j not in matched_col_set:
            row_c = v01_df.loc[c_j]
            unmatched_chambers.append({
                "roi": str(row_c["roi"]),
                "roi_index": int(row_c["roi_index"]),
                "our_CA_um2": round(float(row_c["CA_um2"]), 3),
                "our_CA_total_um2": round(float(row_c["CA_total_um2"]), 3),
                "n_visits": int(row_c["n_visits"]),
                "n_contact_visits": int(row_c["n_contact_visits"]),
                "n_contact_sweeps": int(row_c["n_contact_sweeps"]),
                "max_tcell_instances": int(row_c["max_tcell_instances"]),
                "multi_tcell_sweeps": int(row_c["multi_tcell_sweeps"]),
                "multi_touching_tcell_sweeps": int(row_c["multi_touching_tcell_sweeps"]),
                "max_target_instances": int(row_c["max_target_instances"]),
            })

    # Null / chance models (2,000 Monte Carlo trials with fixed seed)
    rng = np.random.default_rng(42)
    n_trials = 2000

    # Null Model A: 40 candidates drawn log-normally / bootstrap-smoothed from paper_ca distribution
    log_paper = np.log(paper_ca)
    mu_log, sd_log = float(np.mean(log_paper)), float(np.std(log_paper, ddof=1))
    null_lognorm_2pct = np.zeros(n_trials, dtype=int)
    null_lognorm_5pct = np.zeros(n_trials, dtype=int)
    null_lognorm_10pct = np.zeros(n_trials, dtype=int)
    null_lognorm_rho = np.zeros(n_trials, dtype=float)

    # Null Model B: Perturb our 40 CA values by multiplicative Gaussian noise (sigma = 25% SD)
    # or shuffle assignment across independent random draws from empirical KDE
    null_kde_2pct = np.zeros(n_trials, dtype=int)
    null_kde_5pct = np.zeros(n_trials, dtype=int)
    null_kde_10pct = np.zeros(n_trials, dtype=int)

    n_pos_our = int(np.sum(our_ca_all > 0))
    for t in range(n_trials):
        # Model A: 36 positive lognormal draws + 4 zeros (matching the positive count of our 40 chambers)
        sim_pos = np.exp(rng.normal(mu_log, sd_log, size=n_pos_our))
        sim_all = np.concatenate([sim_pos, np.zeros(n_our - n_pos_our)])
        rng.shuffle(sim_all)
        c_mat = np.where(
            sim_all[None, :] > 0,
            np.abs(np.log(np.maximum(eps, sim_all[None, :])) - log_paper[:, None]),
            1e6,
        )
        ri, ci = linear_sum_assignment(c_mat)
        re = np.abs(sim_all[ci] - paper_ca[ri]) / paper_ca[ri]
        null_lognorm_2pct[t] = int(np.sum(re <= 0.02))
        null_lognorm_5pct[t] = int(np.sum(re <= 0.05))
        null_lognorm_10pct[t] = int(np.sum(re <= 0.10))
        null_lognorm_rho[t], _ = spearmanr(ri, ci)

        # Model B: Bootstrap with replacement from paper_ca + 15% log-jitter
        boot_pos = rng.choice(paper_ca, size=n_pos_our, replace=True) * np.exp(rng.normal(0.0, 0.15, size=n_pos_our))
        boot_all = np.concatenate([boot_pos, np.zeros(n_our - n_pos_our)])
        rng.shuffle(boot_all)
        c_mat_b = np.where(
            boot_all[None, :] > 0,
            np.abs(np.log(np.maximum(eps, boot_all[None, :])) - log_paper[:, None]),
            1e6,
        )
        ri_b, ci_b = linear_sum_assignment(c_mat_b)
        re_b = np.abs(boot_all[ci_b] - paper_ca[ri_b]) / paper_ca[ri_b]
        null_kde_2pct[t] = int(np.sum(re_b <= 0.02))
        null_kde_5pct[t] = int(np.sum(re_b <= 0.05))
        null_kde_10pct[t] = int(np.sum(re_b <= 0.10))

    return {
        "n_paper_group1": int(n_paper),
        "n_our_v01_chambers": int(n_our),
        "n_our_positive_ca_chambers": int(len(our_ca_pos)),
        "paper_median_um2": round(float(np.median(paper_ca)), 3),
        "paper_iqr_um2": [
            round(float(np.percentile(paper_ca, 25)), 3),
            round(float(np.percentile(paper_ca, 75)), 3),
        ],
        "paper_mean_um2": round(float(np.mean(paper_ca)), 3),
        "our_all40_median_um2": round(float(np.median(our_ca_all)), 3),
        "our_all40_iqr_um2": [
            round(float(np.percentile(our_ca_all, 25)), 3),
            round(float(np.percentile(our_ca_all, 75)), 3),
        ],
        "our_positive_median_um2": round(float(np.median(our_ca_pos)), 3) if len(our_ca_pos) > 0 else 0.0,
        "our_positive_iqr_um2": [
            round(float(np.percentile(our_ca_pos, 25)), 3),
            round(float(np.percentile(our_ca_pos, 75)), 3),
        ] if len(our_ca_pos) > 0 else [0.0, 0.0],
        "our_matched33_median_um2": round(float(np.median([m["our_CA_um2"] for m in matched_pairs])), 3),
        "ks_test_all40_vs_paper33": {
            "statistic": round(float(ks_all_stat), 4),
            "p_value": round(float(ks_all_p), 4),
        },
        "ks_test_pos_vs_paper33": {
            "statistic": round(float(ks_pos_stat), 4),
            "p_value": round(float(ks_pos_p), 4),
        },
        "optimal_assignment": {
            "within_1pct_count": within_1pct,
            "within_2pct_count": within_2pct,
            "within_5pct_count": within_5pct,
            "within_10pct_count": within_10pct,
            "median_rel_err_pct": round(float(np.median(rel_errors_arr) * 100.0), 3),
            "mean_rel_err_pct": round(float(np.mean(rel_errors_arr) * 100.0), 3),
            "max_rel_err_pct": round(float(np.max(rel_errors_arr) * 100.0), 3),
            "spearman_roi_vs_cell_index_r": round(float(rho_order), 4),
            "spearman_roi_vs_cell_index_p": float(p_order),
        },
        "monotonic_order_assignment": {
            "within_2pct_count": int(np.sum(mono_rel_errs <= 0.02)),
            "within_5pct_count": int(np.sum(mono_rel_errs <= 0.05)),
            "within_10pct_count": int(np.sum(mono_rel_errs <= 0.10)),
            "median_rel_err_pct": round(float(np.median(mono_rel_errs) * 100.0), 3),
            "unmatched_rois": [
                str(v01_df.loc[c_j, "roi"])
                for c_j in range(n_our)
                if c_j not in set(mono_cols)
            ],
        },
        "chance_baseline_2000_trials": {
            "lognormal_null_within_2pct_mean_sd": [
                round(float(np.mean(null_lognorm_2pct)), 2),
                round(float(np.std(null_lognorm_2pct)), 2),
            ],
            "lognormal_null_within_2pct_95pct_quantile": int(np.percentile(null_lognorm_2pct, 95)),
            "lognormal_null_within_5pct_mean_sd": [
                round(float(np.mean(null_lognorm_5pct)), 2),
                round(float(np.std(null_lognorm_5pct)), 2),
            ],
            "lognormal_null_within_5pct_95pct_quantile": int(np.percentile(null_lognorm_5pct, 95)),
            "lognormal_null_within_10pct_mean_sd": [
                round(float(np.mean(null_lognorm_10pct)), 2),
                round(float(np.std(null_lognorm_10pct)), 2),
            ],
            "lognormal_null_within_10pct_95pct_quantile": int(np.percentile(null_lognorm_10pct, 95)),
            "lognormal_null_spearman_r_mean_sd": [
                round(float(np.mean(null_lognorm_rho)), 3),
                round(float(np.std(null_lognorm_rho)), 3),
            ],
            "perturbed_bootstrap_15pct_within_2pct_mean_sd": [
                round(float(np.mean(null_kde_2pct)), 2),
                round(float(np.std(null_kde_2pct)), 2),
            ],
            "perturbed_bootstrap_15pct_within_5pct_mean_sd": [
                round(float(np.mean(null_kde_5pct)), 2),
                round(float(np.std(null_kde_5pct)), 2),
            ],
            "perturbed_bootstrap_15pct_within_10pct_mean_sd": [
                round(float(np.mean(null_kde_10pct)), 2),
                round(float(np.std(null_kde_10pct)), 2),
            ],
        },
        "matched_pairs": matched_pairs,
        "unmatched_chambers": unmatched_chambers,
    }


def save_reproduction_figure(
    comparison: dict[str, Any],
    chamber_df: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "step3_contact_area_reproduction.png",
) -> None:
    """
    Create `figures/step3_contact_area_reproduction.png`:
      - Left: our CA vs paper CA for matched pairs with y=x line.
      - Right: the two distributions (ECDF + box/strip comparison of Paper Group 1 n=33 vs Our v01).
    Follows user house style (system sans, light mode paper #FFFFFF, plot palette #1F4E79 / #C0561F / #2E7D5B).
    """
    matched = comparison["matched_pairs"]
    paper_vals = np.array([m["paper_CA_um2"] for m in matched], dtype=float)
    our_matched_vals = np.array([m["our_CA_um2"] for m in matched], dtype=float)
    rel_errs = np.array([m["rel_err_pct"] for m in matched], dtype=float)

    v01_df = chamber_df[chamber_df["experiment"] == "v01_6_40_120_0-6_hd1"].sort_values("roi_index")
    our_all_vals = v01_df["CA_um2"].to_numpy(dtype=float)
    our_pos_vals = our_all_vals[our_all_vals > 0]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#C9C9C9",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#111111",
        "xtick.color": "#5C5C5C",
        "ytick.color": "#5C5C5C",
        "text.color": "#111111",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Left panel: Our CA vs Paper CA for the 33 matched pairs
    max_val = max(float(np.max(paper_vals)), float(np.max(our_matched_vals))) * 1.06
    ax1.plot([0, max_val], [0, max_val], color="#5C5C5C", linestyle="--", linewidth=1.0, label="y = x")

    close_mask = rel_errs <= 5.0
    ax1.scatter(
        paper_vals[close_mask],
        our_matched_vals[close_mask],
        s=32,
        color="#1F4E79",
        alpha=0.9,
        edgecolors="none",
        label=f"≤ 5% diff (n = {int(np.sum(close_mask))}/33)",
    )
    if np.any(~close_mask):
        ax1.scatter(
            paper_vals[~close_mask],
            our_matched_vals[~close_mask],
            s=36,
            color="#C0561F",
            alpha=0.9,
            edgecolors="none",
            label=f"> 5% diff (n = {int(np.sum(~close_mask))}/33)",
        )

    opt = comparison["optimal_assignment"]
    ax1.text(
        0.04,
        0.95,
        (
            f"n = 33 matched pairs\n"
            f"Within 2%: {opt['within_2pct_count']}/33\n"
            f"Within 5%: {opt['within_5pct_count']}/33\n"
            f"Within 10%: {opt['within_10pct_count']}/33\n"
            f"Median |ΔCA|/CA: {opt['median_rel_err_pct']:.2f}%\n"
            f"ROI vs cell_index ρ = {opt['spearman_roi_vs_cell_index_r']:.3f}"
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#1F1F1F",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#F6F6F6", edgecolor="#E3E3E3"),
    )
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_xlabel("Published contact area CA (µm²)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Reproduced contact area CA (µm²)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Matched pairs (Group 1 / v01)", fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(loc="lower right", frameon=False, fontsize=8.5)

    # Right panel: Empirical cumulative distributions (Paper n=33 vs Reproduced contacting chambers & all 40)
    def ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    xp, yp = ecdf(paper_vals)
    xo_match, yo_match = ecdf(our_matched_vals)
    xo_all, yo_all = ecdf(our_all_vals)

    ax2.step(
        xp,
        yp,
        where="post",
        color="#111111",
        linewidth=1.8,
        label=f"Published Group 1 (n = {len(paper_vals)}, median {np.median(paper_vals):.1f} µm²)",
    )
    ax2.step(
        xo_match,
        yo_match,
        where="post",
        color="#1F4E79",
        linewidth=1.6,
        linestyle="--",
        label=f"Reproduced matched (n = {len(our_matched_vals)}, median {np.median(our_matched_vals):.1f} µm²)",
    )
    ax2.step(
        xo_all,
        yo_all,
        where="post",
        color="#C0561F",
        linewidth=1.4,
        linestyle=":",
        label=f"Reproduced all ROIs (n = {len(our_all_vals)}, median {np.median(our_all_vals):.1f} µm²)",
    )

    ks_all = comparison["ks_test_all40_vs_paper33"]
    ks_match_stat, ks_match_p = ks_2samp(our_matched_vals, paper_vals)
    ax2.text(
        0.36,
        0.27,
        (
            f"KS (all 40 vs paper 33): D = {ks_all['statistic']:.3f}, p = {ks_all['p_value']:.3f}\n"
            f"KS (matched 33 vs paper 33): D = {ks_match_stat:.3f}, p = {ks_match_p:.3f}"
        ),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#5C5C5C",
    )

    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("Contact area CA (µm²)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Cumulative fraction", fontsize=10.5, fontweight="bold")
    ax2.set_title("Distribution comparison", fontsize=11.5, fontweight="bold", loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=8.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Fei lab 3D contact area (CA) from deposited _predictions.h5 masks."
    )
    parser.add_argument("--experiment", type=str, default=None, help="Filter by experiment folder name.")
    parser.add_argument("--roi", type=str, default=None, help="Filter by ROI folder name (e.g. MIP_ROI_0).")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--sanity-figure",
        type=str,
        default=None,
        help="Optional path to save an xy contact-slice sanity figure for the peak sweep.",
    )
    args = parser.parse_args()

    manifest_df = pd.read_csv(MANIFEST_CSV)
    complete_chambers = discover_complete_chambers(
        manifest_df, experiment_filter=args.experiment, roi_filter=args.roi
    )

    if not complete_chambers:
        print("No complete chambers found matching filter criteria.")
        return 1

    tasks: list[tuple[str, str, int, int, str]] = []
    for exp, roi, cdf in complete_chambers:
        # Sweeps 1..5 only (sweep 6 is the tubulin sweep; step2_instance_seg.m skips it)
        sweeps_df = cdf[cdf["sweep_index"] <= 5].sort_values(["visit_index", "sweep_index"])
        for _, row in sweeps_df.iterrows():
            tasks.append((
                exp,
                roi,
                int(row["visit_index"]),
                int(row["sweep_index"]),
                str(row["path"]),
            ))

    print(
        f"Processing {len(tasks)} non-tubulin mask files across {len(complete_chambers)} "
        f"complete chamber(s) with {args.workers} worker(s)..."
    )

    t_start = time.perf_counter()
    sweep_results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            sweep_results.append(process_one_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fut_map = {pool.submit(process_one_file, task): task for task in tasks}
            for idx_done, fut in enumerate(as_completed(fut_map), start=1):
                sweep_results.append(fut.result())
                if idx_done % 250 == 0 or idx_done == len(tasks):
                    el = time.perf_counter() - t_start
                    print(
                        f"  [{idx_done}/{len(tasks)}] elapsed={el:.1f}s "
                        f"({el / idx_done:.3f}s wall/mask)"
                    )

    wall_elapsed_s = time.perf_counter() - t_start
    cpu_sum_s = sum(float(r["elapsed_s"]) for r in sweep_results)

    # Sort sweep results deterministically by experiment, roi_index, visit, sweep
    sweep_results.sort(
        key=lambda r: (
            r["experiment"],
            int(r["roi"].split("_")[-1]) if r["roi"].split("_")[-1].isdigit() else r["roi"],
            r["visit"],
            r["sweep"],
        )
    )

    STEP3_DIR.mkdir(parents=True, exist_ok=True)
    sweep_csv_cols = [
        "experiment",
        "roi",
        "visit",
        "sweep",
        "n_tcell_instances",
        "n_target_instances",
        "s_largest",
        "s_total",
        "n_components",
    ]
    sweep_df = pd.DataFrame(sweep_results)
    per_sweep_csv_path = STEP3_DIR / "per_sweep_contact_area.csv"
    sweep_df[sweep_csv_cols].to_csv(per_sweep_csv_path, index=False)

    # Build per-chamber summary table
    chamber_rows: list[dict[str, Any]] = []
    multi_tcell_chambers: list[dict[str, Any]] = []

    max_visit_global = int(sweep_df["visit"].max()) if not sweep_df.empty else 20
    for (exp, roi), gdf in sweep_df.groupby(["experiment", "roi"], sort=False):
        roi_idx = int(roi.split("_")[-1]) if roi.split("_")[-1].isdigit() else -1
        gid, cond = V_TO_GROUP.get(str(exp), ("", ""))

        visit_s_largest: dict[int, float] = {}
        visit_s_total: dict[int, float] = {}
        for v_idx, vdf in gdf.groupby("visit", sort=True):
            pos_l = vdf[vdf["s_largest"] > 0]["s_largest"]
            pos_t = vdf[vdf["s_total"] > 0]["s_total"]
            visit_s_largest[int(v_idx)] = float(pos_l.mean()) if len(pos_l) > 0 else 0.0
            visit_s_total[int(v_idx)] = float(pos_t.mean()) if len(pos_t) > 0 else 0.0

        max_s_largest = max(visit_s_largest.values()) if visit_s_largest else 0.0
        max_s_total = max(visit_s_total.values()) if visit_s_total else 0.0
        ca_um2 = max_s_largest * CA_VOXEL_FACTOR_UM2
        ca_total_um2 = max_s_total * CA_VOXEL_FACTOR_UM2

        max_t_inst = int(gdf["n_tcell_instances"].max())
        multi_t_sweeps = int(np.sum(gdf["n_tcell_instances"] > 1))
        multi_touch_sweeps = int(np.sum(gdf["n_touching_tcell_instances"] > 1))
        max_b_inst = int(gdf["n_target_instances"].max())
        n_contact_visits = sum(1 for val in visit_s_largest.values() if val > 0)
        n_contact_sweeps = int(np.sum(gdf["s_largest"] > 0))
        multi_comp_sweeps = int(np.sum(gdf["n_components"] > 1))

        if max_t_inst > 1:
            multi_tcell_chambers.append({
                "experiment": str(exp),
                "roi": str(roi),
                "roi_index": roi_idx,
                "max_tcell_instances": max_t_inst,
                "multi_tcell_sweeps": multi_t_sweeps,
                "multi_touching_tcell_sweeps": multi_touch_sweeps,
                "CA_um2": round(ca_um2, 3),
            })

        crow: dict[str, Any] = {
            "experiment": str(exp),
            "roi": str(roi),
            "roi_index": roi_idx,
            "group_id": gid,
            "condition": cond,
            "CA_um2": round(ca_um2, 4),
            "CA_total_um2": round(ca_total_um2, 4),
            "max_visit_s_largest": round(max_s_largest, 3),
            "max_visit_s_total": round(max_s_total, 3),
            "n_visits": len(visit_s_largest),
            "n_contact_visits": n_contact_visits,
            "n_contact_sweeps": n_contact_sweeps,
            "max_tcell_instances": max_t_inst,
            "multi_tcell_sweeps": multi_t_sweeps,
            "multi_touching_tcell_sweeps": multi_touch_sweeps,
            "max_target_instances": max_b_inst,
            "multi_component_sweeps": multi_comp_sweeps,
        }
        for v_i in range(1, max_visit_global + 1):
            s_val = visit_s_largest.get(v_i, 0.0)
            crow[f"visit_{v_i:02d}_s"] = round(s_val, 3)
            crow[f"visit_{v_i:02d}_CA_um2"] = round(s_val * CA_VOXEL_FACTOR_UM2, 4)
        chamber_rows.append(crow)

    chamber_df = pd.DataFrame(chamber_rows).sort_values(["experiment", "roi_index"]).reset_index(drop=True)
    per_chamber_csv_path = STEP3_DIR / "per_chamber_contact_area.csv"
    chamber_df.to_csv(per_chamber_csv_path, index=False)

    sanity_info: dict[str, Any] | None = None
    if args.sanity_figure:
        pos_sweeps = sweep_df[sweep_df["s_largest"] > 0]
        target_row = pos_sweeps.sort_values("s_largest", ascending=False).iloc[0] if not pos_sweeps.empty else sweep_df.iloc[0]
        match_m = manifest_df[
            (manifest_df["experiment"] == target_row["experiment"])
            & (manifest_df["roi"] == target_row["roi"])
            & (manifest_df["visit_index"] == target_row["visit"])
            & (manifest_df["sweep_index"] == target_row["sweep"])
        ].iloc[0]
        sanity_info = save_sanity_check_slice_figure(
            h5_path=REPO_ROOT / str(match_m["path"]),
            out_path=Path(args.sanity_figure) if Path(args.sanity_figure).is_absolute() else (REPO_ROOT / args.sanity_figure),
            experiment=str(target_row["experiment"]),
            roi=str(target_row["roi"]),
            visit=int(target_row["visit"]),
            sweep=int(target_row["sweep"]),
        )

    summary_payload: dict[str, Any] = {
        "timing": {
            "n_chambers": len(complete_chambers),
            "n_masks_processed": len(sweep_results),
            "workers": args.workers,
            "wall_elapsed_s": round(wall_elapsed_s, 3),
            "wall_seconds_per_mask": round(wall_elapsed_s / max(1, len(sweep_results)), 4),
            "cpu_seconds_per_mask_mean": round(cpu_sum_s / max(1, len(sweep_results)), 4),
        },
        "component_choice_diagnostics": {
            "total_sweeps": len(sweep_df),
            "contact_sweeps": int(np.sum(sweep_df["s_largest"] > 0)),
            "multi_component_sweeps": int(np.sum(sweep_df["n_components"] > 1)),
            "max_chamber_CA_diff_total_minus_largest_um2": round(
                float(np.max(chamber_df["CA_total_um2"] - chamber_df["CA_um2"])), 4
            ),
            "chambers_where_CA_total_differs_from_CA_largest_by_gt_1pct": int(
                np.sum(
                    (chamber_df["CA_um2"] > 0)
                    & ((chamber_df["CA_total_um2"] - chamber_df["CA_um2"]) / chamber_df["CA_um2"] > 0.01)
                )
            ),
        },
        "multi_tcell_chambers_count": len(multi_tcell_chambers),
        "multi_tcell_chambers": multi_tcell_chambers,
    }

    if sanity_info is not None:
        summary_payload["sanity_check_slice"] = sanity_info

    v01_count = int(np.sum(chamber_df["experiment"] == "v01_6_40_120_0-6_hd1"))
    if v01_count == 40:
        comparison = compare_with_paper_group1(chamber_df)
        summary_payload["v01_vs_paper_group1"] = comparison
        save_reproduction_figure(
            comparison=comparison,
            chamber_df=chamber_df,
            out_path=FIGURES_DIR / "step3_contact_area_reproduction.png",
        )

    summary_json_path = STEP3_DIR / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(
        f"Done in {wall_elapsed_s:.2f}s "
        f"(wall: {wall_elapsed_s / max(1, len(sweep_results)):.3f}s/mask, "
        f"single-worker CPU mean: {cpu_sum_s / max(1, len(sweep_results)):.3f}s/mask)."
    )
    print(f"Wrote: {per_sweep_csv_path}")
    print(f"Wrote: {per_chamber_csv_path}")
    print(f"Wrote: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
