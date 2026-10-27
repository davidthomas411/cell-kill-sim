#!/usr/bin/env python3
"""
Step 3 (Part 2) & Step 4 (3D vs 2D Death Calibration):
Port the Fei lab's 3D actin and death-dye analysis pipeline from MATLAB to Python,
following the exact specification in `external/feilab_CART_analysis/main/analysis/`:
  - `step2_instance_seg.m` (`spot_detection` on 488-nm channel, `-j` sign on `t_instance`,
    3D dead-dye mask `imgaussfilt3(..., 3) > 398`, `imclose(..., sphere(3))`, `bwareaopen(..., 20, 26)`,
    and `tracker` across sweeps)
  - `interface/cl_factor.m` (ADC = OLS slope of 488-nm actin intensity vs normalized geodesic
    distance `1 - dist_map / max_dist` from the synapse edge `surface == -j`, plus component
    selection by maximum slope `b(1)` among 26-connected components with `max_dist >= 1`)
  - `interface/synapse_tracking.m` (ARFS = actin retrograde flow from 3D spot tracking across
    sweeps 1..5, T-cell centroid drift subtraction when `||td|| < 10`, mean of top-4 track
    displacement vectors projected onto `(t_mean - s_mean) / ||t_mean - s_mean||`)
  - `interface/dead_count.m` (target instance touching `surface == j` within `sphere(1)` and
    within `sphere(10)` of 3D dead-dye mask `d_mask`)
  - `step3_analysis.m` (per-visit combination across sweeps 1..5)
  - `summary/summary.m` (scaling factors `19.6` for ARFS -> nm/s, `1.0` for ADC, `0.0385` for
    CA -> um^2; max over non-void visits)
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
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation, correlate1d, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment

try:
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from reproduce_contact_area import (
    CA_VOXEL_FACTOR_UM2,
    STRUCT26,
    bbox_slices,
    bwareaopen_26,
    compute_contact_surface_from_mask_array,
    imfill_holes_26,
    instance_seg,
    sphere_dilate_edt,
)
from death_labels import (
    MIN_COMPONENT_AREA_2D,
    MIN_OVERLAP_PX_2D,
    THRESH_2D_ADU,
    THRESH_3D_ADU,
    compute_2d_dead_mask,
    compute_3d_dead_mask,
    dilate_2d_edt,
)

RAW_DIR: Path = DATA_DIR / "scidb_19235_raw"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
MIPS_DIR: Path = DATA_DIR / "scidb_19235_mips"
STEP3_DIR: Path = DATA_DIR / "step3"
STEP4_DIR: Path = DATA_DIR / "step4"
STEP3_ACTIN_DIR: Path = DATA_DIR / "step3_actin"

ARFS_FACTOR_NM_S: float = 19.6
ADC_FACTOR: float = 1.0
SWEEPS_PER_VISIT: int = 6
ACTIN_SWEEPS_PER_VISIT: int = 5
SPOT_FILTER_SIZE: int = 11
SPOT_FILTER_SIGMA: float = 3.5
SPOT_THR_488: float = 400.0
SPOT_R_THR_488: float = 0.5

# 6-connected structuring element matching MATLAB `strel('sphere', 1)` in 3D
STRUCT6: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
STRUCT6[1, 1, :] = True
STRUCT6[1, :, 1] = True
STRUCT6[:, 1, 1] = True

# Target chambers firmly matched to the paper
TARGET_CHAMBERS: tuple[dict[str, Any], ...] = (
    {
        "experiment": "v01_6_40_120_0-6_hd1",
        "roi": "MIP_ROI_32",
        "group_id": "1",
        "paper_cell_index": 28,
        "paper_CA_um2": 129.114,
        "paper_ADC": 0.1807,
        "paper_ARFS_nm_s": 53.877,
        "paper_target_fate": "dies (proj death_visit 10)",
    },
    {
        "experiment": "v01_6_40_120_0-6_hd1",
        "roi": "MIP_ROI_2",
        "group_id": "1",
        "paper_cell_index": 3,
        "paper_CA_um2": 93.547,
        "paper_ADC": 1.0965,
        "paper_ARFS_nm_s": 137.110,
        "paper_target_fate": "survives",
    },
)


def edt_indices_matlab_xyz(zero_mask_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nearest-nonzero indices matching MATLAB `[~, IDX] = bwdist(~zero_mask)` by running
    SciPy's separable 3D Maurer EDT in MATLAB's `(X, Y, Z)` dimension order (`transpose(2, 1, 0)`),
    then mapping the returned `(ix, iy, iz)` indices back to `(iz, iy, ix)`.
    This resolves the ~0.03-0.37% equidistant Voronoi boundary tie-breaking difference between
    Python `(Z, Y, X)` and MATLAB `(X, Y, Z)`.
    """
    _, inds_xyz = distance_transform_edt(zero_mask_zyx.transpose(2, 1, 0), return_indices=True)
    ix = inds_xyz[0].transpose(2, 1, 0)
    iy = inds_xyz[1].transpose(2, 1, 0)
    iz = inds_xyz[2].transpose(2, 1, 0)
    return (iz, iy, ix)


def instance_seg_matlab_xyz(nucleus: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """
    Copy of `instance_seg` from `src/reproduce_contact_area.py` using `edt_indices_matlab_xyz`
    when `max_index > 1` (`[~, IDX] = bwdist(nucleus)` in `step2_instance_seg.m` line 176).
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


def compute_contact_surface_matlab_xyz(
    I: np.ndarray, return_volumes: bool = False
) -> dict[str, Any]:
    """
    Copy of `compute_contact_surface_from_mask_array` from `src/reproduce_contact_area.py`
    with MATLAB `(X, Y, Z)` dimension-order tie-breaking in `[~, IDX] = bwdist(map_origin > 0)`
    (`step2_instance_seg.m` line 85) and `background = labelmatrix(CC) == 1` (`line 82`).
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
    t_inst = instance_seg_matlab_xyz(I_sub == 1, I_sub == 3)
    b_inst = instance_seg_matlab_xyz(I_sub == 2, I_sub == 4)

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
    idx_xyz = edt_indices_matlab_xyz(map_origin == 0)
    map_exp = map_origin[idx_xyz]

    surface_mask = (map_exp == 2) & (distance_transform_edt(map_exp != 3) <= 1.0)
    edge_mask = surface_mask & (distance_transform_edt(map_exp != 1) <= 5.0)

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


def build_gaussian_filter_3d(
    size: int = SPOT_FILTER_SIZE, sigma: float = SPOT_FILTER_SIGMA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the 3D Gaussian kernel matching MATLAB `fspecial3('gaussian', size, sigma)`,
    plus its 1D separable factor `g1d`.
    """
    assert size % 2 == 1
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g1d = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g1d /= g1d.sum()
    kernel_3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    kernel_3d /= kernel_3d.sum()
    return kernel_3d, g1d


def spot_detection_vectorized(
    img: np.ndarray,
    thr: float = SPOT_THR_488,
    size: int = SPOT_FILTER_SIZE,
    sigma: float = SPOT_FILTER_SIGMA,
    r_thr: float = SPOT_R_THR_488,
    restrict_mask: np.ndarray | None = None,
    return_corr: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Vectorized exact equivalent of `spot_detection(img, thr, fliter, r_thr)` in `step2_instance_seg.m`
    using separable 1D box sums and separable 1D Gaussian correlations.

    For each `(2r+1)^3` patch `P` around an interior voxel `(z, y, x)` with `img[z, y, x] >= thr`:
        R(P, F) = (sum(P * F) - mean(F) * sum(P)) / (||F - mean(F)||_2 * sqrt(sum(P^2) - sum(P)^2 / N))

    If `restrict_mask` is provided (e.g. `t_instance != 0`, since `step2_instance_seg.m` multiplies
    `t_instance .* mask`), computation is restricted to `bbox_slices(restrict_mask, r + 1, img.shape)`,
    giving 100% identical results inside `restrict_mask` ~45x faster than full-volume filtering.
    """
    r = size // 2
    fliter, g1d = build_gaussian_filter_3d(size=size, sigma=sigma)
    n_vox = float(size ** 3)
    f_mean = 1.0 / n_vox
    f_centered = fliter - f_mean
    f_norm = float(np.sqrt(np.sum(f_centered * f_centered)))
    ones1d = np.ones(size, dtype=np.float64)

    if restrict_mask is not None:
        sl = bbox_slices(restrict_mask > 0, r + 1, img.shape)
        if sl is None:
            empty_m = np.zeros(img.shape, dtype=np.int8)
            if return_corr:
                return empty_m, np.zeros(img.shape, dtype=np.float64)
            return empty_m
        sub_img = img[sl].astype(np.float64)
    else:
        sl = (slice(0, img.shape[0]), slice(0, img.shape[1]), slice(0, img.shape[2]))
        sub_img = img.astype(np.float64)

    def _box_sum_3d(arr: np.ndarray) -> np.ndarray:
        out = correlate1d(arr, ones1d, axis=0, mode="constant", cval=0.0)
        out = correlate1d(out, ones1d, axis=1, mode="constant", cval=0.0)
        return correlate1d(out, ones1d, axis=2, mode="constant", cval=0.0)

    def _gauss_sum_3d(arr: np.ndarray) -> np.ndarray:
        out = correlate1d(arr, g1d, axis=0, mode="constant", cval=0.0)
        out = correlate1d(out, g1d, axis=1, mode="constant", cval=0.0)
        return correlate1d(out, g1d, axis=2, mode="constant", cval=0.0)

    s1 = _box_sum_3d(sub_img)
    s2 = _box_sum_3d(sub_img * sub_img)
    sf = _gauss_sum_3d(sub_img)

    num = sf - f_mean * s1
    var_p = np.maximum(0.0, s2 - (s1 * s1) / n_vox)
    den = f_norm * np.sqrt(var_p)
    r_sub = np.where(den > 1e-12, num / den, 0.0)

    mask_sub = (sub_img >= thr) & (r_sub > r_thr)
    if restrict_mask is not None:
        mask_sub &= (restrict_mask[sl] != 0)

    mask_full = np.zeros(img.shape, dtype=np.int8)
    mask_full[sl] = mask_sub.astype(np.int8)

    # Enforce exact MATLAB interior loop bounds `r+1 : dim-r` (0-based `r : dim-r`)
    mask_full[:r, :, :] = 0
    mask_full[img.shape[0] - r :, :, :] = 0
    mask_full[:, :r, :] = 0
    mask_full[:, img.shape[1] - r :, :] = 0
    mask_full[:, :, :r] = 0
    mask_full[:, :, img.shape[2] - r :] = 0

    if return_corr:
        r_full = np.zeros(img.shape, dtype=np.float64)
        r_full[sl] = r_sub
        return mask_full, r_full
    return mask_full


def spot_detection_literal_loop(
    img: np.ndarray,
    thr: float = SPOT_THR_488,
    size: int = SPOT_FILTER_SIZE,
    sigma: float = SPOT_FILTER_SIGMA,
    r_thr: float = SPOT_R_THR_488,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Literal voxel-by-voxel translation of `spot_detection` in `step2_instance_seg.m` (lines 127-147)
    calling `np.corrcoef(img_patch.ravel(), fliter.ravel())[0, 1]` for equivalence verification.
    """
    r = size // 2
    fliter, _ = build_gaussian_filter_3d(size=size, sigma=sigma)
    fliter_flat = fliter.ravel()
    img_f = img.astype(np.float64)
    h, w, d = img_f.shape
    mask = np.zeros((h, w, d), dtype=np.int8)
    r_map = np.zeros((h, w, d), dtype=np.float64)

    for k in range(r, h - r):
        for j in range(r, w - r):
            for i in range(r, d - r):
                if img_f[k, j, i] < thr:
                    continue
                patch = img_f[k - r : k + r + 1, j - r : j + r + 1, i - r : i + r + 1]
                c = float(np.corrcoef(patch.ravel(), fliter_flat)[0, 1])
                r_map[k, j, i] = c
                if c > r_thr:
                    mask[k, j, i] = 1
    return mask, r_map


def prove_spot_detection_equivalence(raw_h5_path: Path, pred_h5_path: Path) -> dict[str, Any]:
    """
    Prove numerical and binary-mask equivalence between:
      1. Full-volume `spot_detection_vectorized`
      2. `t_instance`-restricted `spot_detection_vectorized`
      3. Literal per-voxel `corrcoef` loop `spot_detection_literal_loop` on a 3D crop around the cell.
    """
    with h5py.File(raw_h5_path, "r") as hf:
        img488 = hf["raw"][1][:]
    with h5py.File(pred_h5_path, "r") as hf:
        pred = hf["predictions"][:]

    res = compute_contact_surface_from_mask_array(pred, return_volumes=True)
    t_inst = res["t_instance"]

    t0 = time.perf_counter()
    mask_full, r_full = spot_detection_vectorized(
        img488, thr=SPOT_THR_488, r_thr=SPOT_R_THR_488, restrict_mask=None, return_corr=True
    )
    elapsed_full_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    mask_restr = spot_detection_vectorized(
        img488, thr=SPOT_THR_488, r_thr=SPOT_R_THR_488, restrict_mask=(t_inst != 0), return_corr=False
    )
    elapsed_restr_s = time.perf_counter() - t1

    # Verify restricted vs full inside `t_instance != 0`
    t_signed_full = np.where(mask_full > 0, -t_inst, t_inst)
    t_signed_restr = np.where(mask_restr > 0, -t_inst, t_inst)
    full_vs_restr_mismatch = int(np.sum(t_signed_full != t_signed_restr))

    # Literal loop on 3D crop around the T-cell (padded by r + 2)
    r = SPOT_FILTER_SIZE // 2
    sl_crop = bbox_slices(t_inst != 0, r + 2, img488.shape)
    if sl_crop is None:
        sl_crop = (slice(50, 100), slice(100, 160), slice(100, 160))
    crop_img = img488[sl_crop]

    t2 = time.perf_counter()
    mask_loop, r_loop = spot_detection_literal_loop(
        crop_img, thr=SPOT_THR_488, r_thr=SPOT_R_THR_488
    )
    elapsed_loop_crop_s = time.perf_counter() - t2

    # Interior of crop where both crop-loop and full-volume have complete 11x11x11 neighborhoods
    z0, z1 = sl_crop[0].start, sl_crop[0].stop
    y0, y1 = sl_crop[1].start, sl_crop[1].stop
    x0, x1 = sl_crop[2].start, sl_crop[2].stop

    mask_vec_int = mask_full[z0 + r : z1 - r, y0 + r : y1 - r, x0 + r : x1 - r]
    r_vec_int = r_full[z0 + r : z1 - r, y0 + r : y1 - r, x0 + r : x1 - r]
    mask_loop_int = mask_loop[r:-r, r:-r, r:-r]
    r_loop_int = r_loop[r:-r, r:-r, r:-r]
    crop_int = crop_img[r:-r, r:-r, r:-r]

    active = crop_int >= SPOT_THR_488
    max_abs_r_diff = float(np.max(np.abs(r_vec_int[active] - r_loop_int[active]))) if np.any(active) else 0.0
    crop_mask_mismatch = int(np.sum(mask_vec_int != mask_loop_int))

    return {
        "raw_h5_path": str(raw_h5_path.relative_to(REPO_ROOT)),
        "pred_h5_path": str(pred_h5_path.relative_to(REPO_ROOT)),
        "volume_shape": list(img488.shape),
        "crop_shape": list(crop_img.shape),
        "crop_interior_voxels_tested": int(mask_loop_int.size),
        "crop_voxels_above_thr400": int(active.sum()),
        "crop_positive_spot_voxels": int(mask_loop_int.sum()),
        "full_volume_positive_spot_voxels": int(mask_full.sum()),
        "tcell_restricted_positive_spot_voxels": int(mask_restr.sum()),
        "max_abs_corrcoef_diff": max_abs_r_diff,
        "crop_mask_mismatch_voxels": crop_mask_mismatch,
        "full_vs_tcell_restricted_t_instance_mismatch_voxels": full_vs_restr_mismatch,
        "elapsed_literal_loop_crop_s": round(elapsed_loop_crop_s, 4),
        "elapsed_vectorized_full_s": round(elapsed_full_s, 4),
        "elapsed_vectorized_tcell_restricted_s": round(elapsed_restr_s, 4),
    }


def cl_factor(
    img_actin: np.ndarray,
    s_mask: np.ndarray,
    mask_index: int = 1,
    return_mask: bool = False,
) -> dict[str, Any]:
    """
    Exact translation of `interface/cl_factor.m`:
      - `s_real = (s_mask == mask_index)`
      - `s_edge = (s_mask == -mask_index)`
      - 26-connected components of `s_real` (`bwconncomp(s_real, 26)`)
      - For each component `i`:
          Geodesic step distance `dist_map` from `s_edge` into `s_label == i` using
          `strel('sphere', 1)` (6-connected dilation).
          If `max_dist < 1`, skip component.
          Compute OLS linear regression of `y = double(img_actin(comp))` on
          `x = double(1 - dist_map(comp) / max_dist)`.
          Select component `i` with maximum slope `b(1)`.
    Also computes `cl_on_largest` (slope on the largest component by voxel count) and
    `s_largest` / `s_total` for direct comparison with masks-only `reproduce_contact_area.py`.
    """
    sl = bbox_slices(s_mask != 0, 2, s_mask.shape)
    if sl is None:
        out = {
            "cl": 0.0,
            "s_actin": 0,
            "s_largest": 0,
            "s_total": 0,
            "n_components": 0,
            "n_valid_components": 0,
            "selected_comp_is_largest": True,
            "cl_on_largest": 0.0,
        }
        if return_mask:
            out["mask"] = np.zeros_like(s_mask, dtype=bool)
        return out

    s_sub = s_mask[sl]
    a_sub = img_actin[sl]
    s_real = s_sub == mask_index
    s_total = int(np.sum(s_real))
    if s_total == 0:
        out = {
            "cl": 0.0,
            "s_actin": 0,
            "s_largest": 0,
            "s_total": 0,
            "n_components": 0,
            "n_valid_components": 0,
            "selected_comp_is_largest": True,
            "cl_on_largest": 0.0,
        }
        if return_mask:
            out["mask"] = np.zeros_like(s_mask, dtype=bool)
        return out

    s_edge = s_sub == -mask_index
    s_lbl, n_comp = label(s_real, structure=STRUCT26)
    counts = np.bincount(s_lbl.ravel())
    counts[0] = 0
    largest_comp_id = int(np.argmax(counts))
    s_largest = int(counts[largest_comp_id])

    # Because 26-connected components of `s_real` are disjoint and separated by at least
    # 26-connectivity, 6-connected dilation from `s_edge` into `s_real` never crosses
    # between components. We run the exact geodesic BFS loop.
    s_remain = s_real.copy()
    s_seeds = s_edge.copy()
    dist_map = np.zeros(s_real.shape, dtype=np.float64)
    counter = 0
    while np.any(s_seeds):
        temp = s_remain & binary_dilation(s_seeds, structure=STRUCT6)
        s_seeds = temp
        s_remain[temp] = False
        dist_map[temp] = float(counter)
        counter += 1

    cl_best = -np.inf
    s_idx = 0
    s_actin = 0
    n_valid = 0
    cl_on_largest = 0.0

    for c_i in range(1, n_comp + 1):
        comp_mask = s_lbl == c_i
        max_dist = float(dist_map[comp_mask].max())
        if max_dist < 1.0:
            continue
        n_valid += 1
        y = a_sub[comp_mask].astype(np.float64)
        x = 1.0 - dist_map[comp_mask] / max_dist
        xm = x - x.mean()
        denom = float(np.dot(xm, xm))
        if denom <= 1e-15:
            b1 = 0.0
        else:
            ym = y - y.mean()
            b1 = float(np.dot(xm, ym) / denom)
        if c_i == largest_comp_id:
            cl_on_largest = b1
        if b1 > cl_best:
            cl_best = b1
            s_idx = c_i
            s_actin = int(counts[c_i])

    if s_idx == 0:
        cl_best = 0.0
        s_actin = 0

    out = {
        "cl": float(cl_best),
        "s_actin": int(s_actin),
        "s_largest": int(s_largest),
        "s_total": int(s_total),
        "n_components": int(n_comp),
        "n_valid_components": int(n_valid),
        "selected_comp_is_largest": bool(s_idx == largest_comp_id or s_idx == 0),
        "cl_on_largest": float(cl_on_largest),
    }

    if return_mask:
        full_mask = np.zeros_like(s_mask, dtype=bool)
        if s_idx > 0:
            full_mask[sl] = s_lbl == s_idx
        out["mask"] = full_mask

    return out


def assign_detections_to_tracks(
    cost: np.ndarray, cost_of_non_assignment: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact equivalent of MATLAB's `assignDetectionsToTracks(cost, costOfNonAssignment)`:
    minimizes sum_{(r, c) in A} cost[r, c] + (M - |A|) * costOfNonAssignment + (N - |A|) * costOfNonAssignment.
    """
    m_tracks, n_dets = cost.shape
    if m_tracks == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(n_dets, dtype=int),
        )
    if n_dets == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(m_tracks, dtype=int),
            np.empty(0, dtype=int),
        )

    big = 1e9
    aug = np.full((m_tracks + n_dets, m_tracks + n_dets), big, dtype=np.float64)
    # In MATLAB assignDetectionsToTracks, a pair (r, c) can only be assigned if cost < 2 * costOfNonAssignment
    aug[:m_tracks, :n_dets] = np.where(cost < 2.0 * cost_of_non_assignment, cost, big)
    for r_i in range(m_tracks):
        aug[r_i, n_dets + r_i] = cost_of_non_assignment
    for c_j in range(n_dets):
        aug[m_tracks + c_j, c_j] = cost_of_non_assignment
    aug[m_tracks:, n_dets:] = 0.0

    row_ind, col_ind = linear_sum_assignment(aug)
    assignments: list[tuple[int, int]] = []
    unassigned_tracks: list[int] = []
    assigned_dets: set[int] = set()

    for r_i, c_j in zip(row_ind, col_ind):
        if r_i < m_tracks:
            if c_j < n_dets and cost[r_i, c_j] < 2.0 * cost_of_non_assignment:
                assignments.append((int(r_i), int(c_j)))
                assigned_dets.add(int(c_j))
            else:
                unassigned_tracks.append(int(r_i))

    unassigned_dets = [int(c_j) for c_j in range(n_dets) if c_j not in assigned_dets]
    return (
        np.array(assignments, dtype=int).reshape(-1, 2),
        np.array(unassigned_tracks, dtype=int),
        np.array(unassigned_dets, dtype=int),
    )


def regionprops3_centroids_volumes(
    mask_3d: np.ndarray, matlab_column_major_order: bool = True
) -> np.ndarray:
    """
    Fast, exact equivalent of MATLAB:
        stats = regionprops3(L > 0);
        data_point = cat(2, stats.Centroid, stats.Volume);
    where `L` has shape `(X, Y, Z)` in MATLAB (`(256, 256, Z)` from `h5read`) and `(Z, Y, X)` in Python.
    When `matlab_column_major_order=True`, components are labeled in MATLAB column-major order
    (`L_matlab = mask_3d.transpose(2, 1, 0)`), and Centroid is `[c_y_matlab, c_x_matlab, c_z_matlab]`
    matching `regionprops3`.
    Returns `(N_components, 4)` array `[cx, cy, cz, volume]`.
    """
    sl = bbox_slices(mask_3d > 0, 0, mask_3d.shape)
    if sl is None:
        return np.empty((0, 4), dtype=np.float64)

    sub = mask_3d[sl] > 0
    z0, y0, x0 = sl[0].start, sl[1].start, sl[2].start

    if matlab_column_major_order:
        # In MATLAB, h5read loads (Z, Y, X)_C as (X, Y, Z)_F.
        # `bwconncomp(L > 0, 26)` scans in Fortran order where X varies fastest, then Y, then Z,
        # which is equivalent to C-order labeling on `(Z, Y, X)`!
        # Let's verify: in Fortran order on (X, Y, Z), index 1 (X) varies fastest, then 2 (Y), then 3 (Z).
        # In C order on (Z, Y, X), index 2 (X) varies fastest, then 1 (Y), then 0 (Z)!
        # So C-order memory layout of (Z, Y, X) IS IDENTICAL voxel-for-voxel to Fortran-order memory
        # layout of (X, Y, Z)!
        lbl, n_c = label(sub, structure=STRUCT26)
    else:
        lbl, n_c = label(sub, structure=STRUCT26)

    if n_c == 0:
        return np.empty((0, 4), dtype=np.float64)

    zz, yy, xx = np.nonzero(lbl)
    comp_ids = lbl[zz, yy, xx]
    vols = np.bincount(comp_ids, minlength=n_c + 1)[1:].astype(np.float64)
    # In MATLAB regionprops3 on array of size (X, Y, Z), Centroid is [col(Y), row(X), slice(Z)] (1-based)
    cx = np.bincount(comp_ids, weights=(yy + y0 + 1.0), minlength=n_c + 1)[1:] / vols
    cy = np.bincount(comp_ids, weights=(xx + x0 + 1.0), minlength=n_c + 1)[1:] / vols
    cz = np.bincount(comp_ids, weights=(zz + z0 + 1.0), minlength=n_c + 1)[1:] / vols
    return np.column_stack([cx, cy, cz, vols])


def tracker_tcell(
    points_per_timepoint: list[np.ndarray],
    min_volume: float = 0.0,
    max_volume: float = 500000.0,
    max_distance: float = 200.0,
    miss_tolerance: int = 10,
    min_length: int = 50,
) -> np.ndarray:
    """
    Exact translation of `tracker(label)` in `step2_instance_seg.m` (lines 223-328),
    which tracks `abs(t_instance)` across all 120 timepoints of a chamber (`timepoint = 120`,
    where every 6th sweep has 0 detections and is bridged via `miss_tolerance = 10`).
    Returns `(n_tracks, timepoint)` uint8 array `t_trace`.
    """
    trace_result, _ = tracker_actin(
        points_per_sweep=points_per_timepoint,
        min_volume=min_volume,
        max_volume=max_volume,
        max_distance=max_distance,
        miss_tolerance=miss_tolerance,
        min_length=min_length,
    )
    return trace_result.astype(np.uint8)


def tracker_actin(
    points_per_sweep: list[np.ndarray],
    min_volume: float = 0.0,
    max_volume: float = 500000.0,
    max_distance: float = 6.0,
    miss_tolerance: int = 0,
    min_length: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact translation of `tracker(actin_mask)` in `interface/synapse_tracking.m` (lines 119-233).
    """
    timepoint = len(points_per_sweep)
    trace_result = np.empty((0, timepoint + 3), dtype=np.float64)
    v_result = np.empty((0, timepoint + 3, 3), dtype=np.float64)

    for i0 in range(timepoint):
        data_point = points_per_sweep[i0]
        if trace_result.shape[0] == 0:
            new_traces = []
            for j0 in range(data_point.shape[0]):
                if data_point[j0, 3] > min_volume:
                    tr = np.zeros(timepoint + 3, dtype=np.float64)
                    tr[i0] = float(j0 + 1)
                    tr[i0 + 1 : i0 + 4] = data_point[j0, :3]
                    new_traces.append(tr)
            if new_traces:
                trace_result = np.vstack(new_traces)
                v_result = np.zeros((len(new_traces), timepoint + 3, 3), dtype=np.float64)
        else:
            trace_id: list[int] = []
            predictions: list[np.ndarray] = []
            volume_id: list[int] = []
            detections: list[np.ndarray] = []

            for j0 in range(trace_result.shape[0]):
                if (
                    trace_result[j0, i0] > 0
                    or trace_result[j0, i0 + 1] > 0
                    or trace_result[j0, i0 + 2] > 0
                ):
                    flag = False
                    # MATLAB: try flag = sum(trace_result(j, i-miss_tolerance-1 : i-1)) == 0; catch; end
                    # In 1-based indexing (`i = i0 + 1`), `i - miss_tolerance - 1 >= 1` iff `i0 - miss_tolerance - 1 >= 0`.
                    start_idx = i0 - miss_tolerance - 1
                    if start_idx >= 0:
                        flag = float(np.sum(trace_result[j0, start_idx:i0])) == 0.0
                    if flag:
                        trace_result[j0, i0 : i0 + 3] = 0.0
                    else:
                        trace_id.append(j0)
                        predictions.append(trace_result[j0, i0 : i0 + 3].copy())

            for j0 in range(data_point.shape[0]):
                if min_volume < data_point[j0, 3] < max_volume:
                    volume_id.append(j0 + 1)
                    detections.append(data_point[j0, :3].copy())

            if len(predictions) == 0:
                assignments = np.empty((0, 2), dtype=int)
                unassigned_tracks = np.empty(0, dtype=int)
                unassigned_dets = np.arange(len(detections), dtype=int)
            elif len(detections) == 0:
                assignments = np.empty((0, 2), dtype=int)
                unassigned_tracks = np.arange(len(predictions), dtype=int)
                unassigned_dets = np.empty(0, dtype=int)
            else:
                pred_arr = np.vstack(predictions)
                det_arr = np.vstack(detections)
                diff = pred_arr[:, None, :] - det_arr[None, :, :]
                cost = np.sqrt(np.sum(diff * diff, axis=2))
                assignments, unassigned_tracks, unassigned_dets = assign_detections_to_tracks(
                    cost, max_distance / 2.0
                )

            for r_i, c_j in assignments:
                tid = trace_id[r_i]
                pend = detections[c_j]
                pstart = trace_result[tid, i0 : i0 + 3].copy()
                v_result[tid, i0, :] = pend - pstart
                trace_result[tid, i0] = float(volume_id[c_j])
                trace_result[tid, i0 + 1 : i0 + 4] = pend

            for r_i in unassigned_tracks:
                tid = trace_id[r_i]
                trace_result[tid, i0 + 1 : i0 + 4] = trace_result[tid, i0 : i0 + 3]
                trace_result[tid, i0] = 0.0

            if len(unassigned_dets) > 0:
                new_trs = []
                for c_j in unassigned_dets:
                    tr = np.zeros(timepoint + 3, dtype=np.float64)
                    tr[i0] = float(volume_id[c_j])
                    tr[i0 + 1 : i0 + 4] = detections[c_j]
                    new_trs.append(tr)
                trace_result = np.vstack([trace_result, np.vstack(new_trs)])
                v_result = np.concatenate(
                    [v_result, np.zeros((len(new_trs), timepoint + 3, 3), dtype=np.float64)],
                    axis=0,
                )

    if trace_result.shape[0] > 0:
        trace_result = trace_result[:, :timepoint]
        v_result = v_result[:, :timepoint, :]
        keep = np.sum(trace_result > 0, axis=1) >= min_length
        trace_result = trace_result[keep]
        v_result = v_result[keep]

    return trace_result, v_result


def synapse_tracking_visit(
    spot_points_per_sweep: list[np.ndarray],
    t_centers: np.ndarray,
    s_centers: np.ndarray,
    any_s_mask: bool,
) -> dict[str, Any]:
    """
    Exact translation of `interface/synapse_tracking.m` (lines 1-92) on the 5 sweeps of a visit.
    """
    if not any_s_mask:
        return {
            "v_raw": 0.0,
            "arfs_nm_s": 0.0,
            "n_tracks_len3": 0,
            "used_fallback_001": False,
            "drift_exceeded_10": False,
            "max_td_norm": 0.0,
        }

    t_steps = len(spot_points_per_sweep)
    t_valid = np.linalg.norm(t_centers, axis=1) > 0
    s_valid = np.linalg.norm(s_centers, axis=1) > 0
    t_num = int(np.sum(t_valid))
    s_num = int(np.sum(s_valid))
    t_mean = np.sum(t_centers[t_valid], axis=0) / max(1, t_num)
    s_mean = np.sum(s_centers[s_valid], axis=0) / max(1, s_num)

    trace_result, v_result = tracker_actin(spot_points_per_sweep)
    trace_num = int(v_result.shape[0])

    drift_exceeded_10 = False
    max_td_norm = 0.0
    for i0 in range(1, t_steps):
        if np.linalg.norm(t_centers[i0 - 1]) > 0 and np.linalg.norm(t_centers[i0]) > 0:
            td = t_centers[i0] - t_centers[i0 - 1]
            td_norm = float(np.linalg.norm(td))
            if td_norm > max_td_norm:
                max_td_norm = td_norm
            if td_norm < 10.0:
                for j0 in range(trace_num):
                    if np.linalg.norm(v_result[j0, i0, :]) > 0:
                        v_result[j0, i0, :] = v_result[j0, i0, :] - td
            else:
                v_result[:, :, :] = 0.01
                drift_exceeded_10 = True
                break

    v_sum = np.sum(v_result, axis=1)  # shape (trace_num, 3)
    vnorm = np.linalg.norm(v_sum, axis=1) if trace_num > 0 else np.empty(0, dtype=np.float64)
    idx_desc = np.argsort(-vnorm, kind="stable")

    num_top = 4
    if trace_num >= num_top:
        v_max = np.mean(v_sum[idx_desc[:num_top]], axis=0)
        used_fallback = False
    else:
        v_max = np.ones(3, dtype=np.float64) * 0.01
        used_fallback = True

    if t_num > 0 and s_num > 0:
        temp = t_mean - s_mean
        norm_temp = float(np.linalg.norm(temp))
        if norm_temp > 1e-12:
            unit_vec = temp / norm_temp
            v_val = float(np.dot(unit_vec, v_max))
        else:
            v_val = 0.0
    else:
        v_val = 0.0

    return {
        "v_raw": v_val,
        "arfs_nm_s": v_val * ARFS_FACTOR_NM_S,
        "n_tracks_len3": trace_num,
        "used_fallback_001": used_fallback,
        "drift_exceeded_10": drift_exceeded_10,
        "max_td_norm": max_td_norm,
    }


def dead_count_3d(
    s_real_mask: np.ndarray, b_instance: np.ndarray, d_mask: np.ndarray
) -> dict[str, Any]:
    """
    Exact translation of `interface/dead_count.m`:
      - If `~any(s_real_mask(:))` -> `count = 0`
      - For `i = 1:max(b_instance(:))`:
          `b_i = (b_instance == i)`
          `touch = s_real_mask & imdilate(b_i, strel('sphere', 1))`
          If `~any(touch(:))`, continue
          `dead = b_i & imdilate(d_mask, strel('sphere', 10))`
          If `any(dead(:))`, `count += 1`
    Also returns `dead_notouch` (whether any target instance `b_instance > 0` overlaps
    `imdilate(d_mask, sphere(10))` regardless of contact surface on that sweep) and
    `ov10_binst_vox` (number of target instance voxels within 10 voxels of `d_mask`).
    """
    d_vox = int(np.sum(d_mask))
    b_any = b_instance > 0
    if d_vox == 0 or not np.any(b_any):
        return {
            "dead_count_strict": 0,
            "dead_any_target_within10": False,
            "d3d_voxels": d_vox,
            "ov10_binst_vox": 0,
            "min_dist_binst_to_d3d": np.inf,
        }

    sl = bbox_slices(b_any | d_mask | s_real_mask, 12, b_instance.shape)
    b_sub = b_instance[sl]
    d_sub = d_mask[sl]
    s_sub = s_real_mask[sl]

    edt_d = distance_transform_edt(~d_sub)
    d_dil10 = edt_d <= 10.0
    b_any_sub = b_sub > 0
    ov10_vox = int(np.sum(b_any_sub & d_dil10))
    min_dist = float(edt_d[b_any_sub].min()) if np.any(b_any_sub) else np.inf

    count = 0
    if np.any(s_sub):
        max_b = int(b_sub.max())
        for b_id in range(1, max_b + 1):
            b_i = b_sub == b_id
            if not np.any(b_i):
                continue
            touch = s_sub & binary_dilation(b_i, structure=STRUCT6)
            if not np.any(touch):
                continue
            if np.any(b_i & d_dil10):
                count += 1

    return {
        "dead_count_strict": int(count),
        "dead_any_target_within10": bool(ov10_vox > 0),
        "d3d_voxels": int(d_vox),
        "ov10_binst_vox": int(ov10_vox),
        "min_dist_binst_to_d3d": round(min_dist, 3) if np.isfinite(min_dist) else np.nan,
    }


def process_one_visit(task: dict[str, Any]) -> dict[str, Any]:
    """
    Worker function to process all 5 actin sweeps (sweeps 1..5) of a single visit `(experiment, roi, visit)`.
    """
    t0 = time.perf_counter()
    exp = task["experiment"]
    roi = task["roi"]
    visit = int(task["visit"])
    sweep_items = task["sweep_items"]  # list of (sweep_idx, stem, raw_path, pred_path)

    sweep_rows: list[dict[str, Any]] = []
    spot_points_per_sweep: list[np.ndarray] = []
    tcell_points_per_sweep: list[np.ndarray] = []
    t_centers = np.zeros((ACTIN_SWEEPS_PER_VISIT, 3), dtype=np.float64)
    s_centers = np.zeros((ACTIN_SWEEPS_PER_VISIT, 3), dtype=np.float64)
    any_s_tracked = False

    s_actin_list: list[int] = []
    s_largest_list: list[int] = []
    s_total_list: list[int] = []
    cl_list: list[float] = []
    cl_largest_list: list[float] = []
    d_strict_list: list[int] = []
    d_within10_list: list[bool] = []
    d3d_vox_list: list[int] = []
    ov10_vox_list: list[int] = []

    mip_xy_from_raw_s1: np.ndarray | None = None
    mip_xz_from_raw_s1: np.ndarray | None = None
    pred_s1: np.ndarray | None = None

    for j0, (sweep_idx, stem, raw_path, pred_path) in enumerate(sweep_items):
        with h5py.File(pred_path, "r") as hf:
            pred = hf["predictions"][:]
        with h5py.File(raw_path, "r") as hf:
            raw_ds = hf["raw"]
            raw405 = raw_ds[0][:]
            img488 = raw_ds[1][:]

        if j0 == 0:
            mip_xy_from_raw_s1 = raw405.max(axis=0).T
            mip_xz_from_raw_s1 = raw405.max(axis=1).T
            pred_s1 = pred

        # 1. Instance segmentation & contact surface (MATLAB xyz EDT tie-breaking + SciPy zyx baseline)
        surf_res = compute_contact_surface_matlab_xyz(pred, return_volumes=True)
        surf_res_zyx = compute_contact_surface_from_mask_array(pred, return_volumes=True)
        t_inst = surf_res["t_instance"]
        b_inst = surf_res["b_instance"]
        s_mask = surf_res["surface"]
        mask_index = int(surf_res.get("selected_tcell_id", 1)) if surf_res["n_tcell_instances"] > 0 else 0

        # Record all T-cell instance centroids/volumes for chamber-wide tracker_tcell
        n_t_inst = int(surf_res["n_tcell_instances"])
        if n_t_inst > 0:
            t_pts_all = []
            for tid in range(1, n_t_inst + 1):
                pts_tid = regionprops3_centroids_volumes(t_inst == tid)
                if pts_tid.shape[0] > 0:
                    # Total volume and centroid of instance `tid`
                    zz, yy, xx = np.nonzero(t_inst == tid)
                    vol_tid = float(zz.size)
                    t_pts_all.append([float(yy.mean()) + 1.0, float(xx.mean()) + 1.0, float(zz.mean()) + 1.0, vol_tid])
            tcell_points_per_sweep.append(np.array(t_pts_all, dtype=np.float64) if t_pts_all else np.empty((0, 4), dtype=np.float64))
        else:
            tcell_points_per_sweep.append(np.empty((0, 4), dtype=np.float64))

        if mask_index > 0:
            # 2. cl_factor (ADC + component selection) on both xyz (MATLAB) and zyx (SciPy) surfaces
            cl_res = cl_factor(img488, s_mask, mask_index=mask_index, return_mask=True)
            cl_res_zyx = cl_factor(img488, surf_res_zyx["surface"], mask_index=mask_index, return_mask=False)
            s_tracked_j = cl_res["mask"]
            if np.any(s_tracked_j):
                any_s_tracked = True
                sc_pts = regionprops3_centroids_volumes(s_tracked_j)
                if sc_pts.shape[0] > 0:
                    s_centers[j0] = sc_pts[int(np.argmax(sc_pts[:, 3])), :3]

            # 3. T-cell mask & centroid
            t_mask_j = t_inst == mask_index
            tc_pts = regionprops3_centroids_volumes(t_mask_j)
            if tc_pts.shape[0] > 0:
                t_centers[j0] = tc_pts[int(np.argmax(tc_pts[:, 3])), :3]

            # 4. Vectorized spot_detection on 488-nm channel inside t_mask_j
            spot_mask_j = spot_detection_vectorized(
                img488,
                thr=SPOT_THR_488,
                size=SPOT_FILTER_SIZE,
                sigma=SPOT_FILTER_SIGMA,
                r_thr=SPOT_R_THR_488,
                restrict_mask=t_mask_j,
                return_corr=False,
            )
            spot_pts = regionprops3_centroids_volumes(spot_mask_j > 0)
            spot_points_per_sweep.append(spot_pts)

            # 5. 3D dead-dye mask & dead_count
            d_mask_3d, _ = compute_3d_dead_mask(raw405, threshold=THRESH_3D_ADU)
            dead_res = dead_count_3d(s_mask == mask_index, b_inst, d_mask_3d)
        else:
            cl_res = {
                "cl": 0.0,
                "s_actin": 0,
                "s_largest": 0,
                "s_total": 0,
                "n_components": 0,
                "n_valid_components": 0,
                "selected_comp_is_largest": True,
                "cl_on_largest": 0.0,
            }
            cl_res_zyx = cl_res
            spot_mask_j = np.zeros_like(pred, dtype=np.int8)
            spot_pts = np.empty((0, 4), dtype=np.float64)
            spot_points_per_sweep.append(spot_pts)
            d_mask_3d, _ = compute_3d_dead_mask(raw405, threshold=THRESH_3D_ADU)
            dead_res = dead_count_3d(np.zeros_like(pred, dtype=bool), b_inst, d_mask_3d)

        s_actin_list.append(cl_res["s_actin"])
        s_largest_list.append(surf_res_zyx["s_largest"])
        s_total_list.append(cl_res["s_total"])
        cl_list.append(cl_res["cl"])
        cl_largest_list.append(cl_res_zyx["cl"])
        d_strict_list.append(dead_res["dead_count_strict"])
        d_within10_list.append(dead_res["dead_any_target_within10"])
        d3d_vox_list.append(dead_res["d3d_voxels"])
        ov10_vox_list.append(dead_res["ov10_binst_vox"])

        sweep_rows.append(
            {
                "experiment": exp,
                "roi": roi,
                "visit": visit,
                "sweep": int(sweep_idx),
                "stem": stem,
                "n_tcell_instances": int(surf_res["n_tcell_instances"]),
                "n_target_instances": int(surf_res["n_target_instances"]),
                "selected_tcell_id": int(mask_index),
                "n_surface_components": int(cl_res["n_components"]),
                "n_valid_surface_components": int(cl_res["n_valid_components"]),
                "selected_comp_is_largest": bool(cl_res["selected_comp_is_largest"]),
                "s_actin_vox": int(cl_res["s_actin"]),
                "s_xyz_largest_vox": int(cl_res["s_largest"]),
                "s_zyx_actin_vox": int(cl_res_zyx["s_actin"]),
                "s_largest_vox": int(surf_res_zyx["s_largest"]),
                "s_total_vox": int(cl_res["s_total"]),
                "cl_slope": round(float(cl_res["cl"]), 6),
                "cl_zyx_slope": round(float(cl_res_zyx["cl"]), 6),
                "cl_on_largest": round(float(cl_res["cl_on_largest"]), 6),
                "n_spot_voxels": int(np.sum(spot_mask_j > 0)),
                "n_spot_components": int(spot_pts.shape[0]),
                "d3d_voxels": int(dead_res["d3d_voxels"]),
                "ov10_3d_binst_vox": int(dead_res["ov10_binst_vox"]),
                "min_dist_binst_to_d3d": dead_res["min_dist_binst_to_d3d"],
                "dead_count_3d_strict": int(dead_res["dead_count_strict"]),
                "dead_3d_any_target_within10": bool(dead_res["dead_any_target_within10"]),
            }
        )

    # Combine sweeps 1..5 per step3_analysis.m lines 57-66:
    s_actin_arr = np.array(s_actin_list, dtype=np.float64)
    s_largest_arr = np.array(s_largest_list, dtype=np.float64)
    s_total_arr = np.array(s_total_list, dtype=np.float64)
    cl_arr = np.array(cl_list, dtype=np.float64)
    cl_largest_arr = np.array(cl_largest_list, dtype=np.float64)

    pos_actin = s_actin_arr > 0
    if np.any(pos_actin):
        s_visit_actin = float(np.mean(s_actin_arr[pos_actin]))
        cl_visit = float(np.mean(cl_arr[pos_actin]))
    else:
        s_visit_actin = 0.0
        cl_visit = 0.0

    pos_largest = s_largest_arr > 0
    if np.any(pos_largest):
        s_visit_largest = float(np.mean(s_largest_arr[pos_largest]))
        cl_visit_on_largest = float(np.mean(cl_largest_arr[pos_largest]))
    else:
        s_visit_largest = 0.0
        cl_visit_on_largest = 0.0

    pos_total = s_total_arr > 0
    s_visit_total = float(np.mean(s_total_arr[pos_total])) if np.any(pos_total) else 0.0

    # ARFS via synapse_tracking
    arfs_res = synapse_tracking_visit(
        spot_points_per_sweep=spot_points_per_sweep,
        t_centers=t_centers,
        s_centers=s_centers,
        any_s_mask=any_s_tracked,
    )

    # 2D projection rule on Sweep 1 (imported from src/death_labels.py)
    assert mip_xy_from_raw_s1 is not None and mip_xz_from_raw_s1 is not None and pred_s1 is not None
    raw_b_s1 = (pred_s1 == 2) | (pred_s1 == 4)
    b_xy = raw_b_s1.max(axis=0).T
    b_xz = raw_b_s1.max(axis=1).T
    d_xy, g_xy = compute_2d_dead_mask(mip_xy_from_raw_s1, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
    d_xz, g_xz = compute_2d_dead_mask(mip_xz_from_raw_s1, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
    ov_xy = int(np.sum(d_xy & b_xy))
    ov_xz = int(np.sum(d_xz & b_xz))
    ov10_xy = int(np.sum(dilate_2d_edt(d_xy, 10.0) & b_xy))
    ov10_xz = int(np.sum(dilate_2d_edt(d_xz, 10.0) & b_xz))
    dead_2d_visit = bool((ov_xy >= MIN_OVERLAP_PX_2D) and (ov_xz >= MIN_OVERLAP_PX_2D))

    dead_3d_strict_visit = int(np.max(d_strict_list)) > 0
    dead_3d_within10_visit = bool(np.any(d_within10_list))
    dead_3d_s1_strict = int(d_strict_list[0]) > 0
    dead_3d_s1_within10 = bool(d_within10_list[0])

    visit_row = {
        "experiment": exp,
        "roi": roi,
        "visit": visit,
        "time_min": round(visit * 10.0, 1),
        "stem_s1": sweep_items[0][1],
        "n_contact_sweeps_actin": int(np.sum(pos_actin)),
        "n_contact_sweeps_largest": int(np.sum(pos_largest)),
        "s_visit_actin_vox": round(s_visit_actin, 4),
        "s_visit_largest_vox": round(s_visit_largest, 4),
        "s_visit_total_vox": round(s_visit_total, 4),
        "CA_actin_um2": round(s_visit_actin * CA_VOXEL_FACTOR_UM2, 4),
        "CA_masks_largest_um2": round(s_visit_largest * CA_VOXEL_FACTOR_UM2, 4),
        "CA_masks_total_um2": round(s_visit_total * CA_VOXEL_FACTOR_UM2, 4),
        "CA_actin_minus_largest_um2": round((s_visit_actin - s_visit_largest) * CA_VOXEL_FACTOR_UM2, 4),
        "ADC": round(cl_visit * ADC_FACTOR, 6),
        "ADC_on_largest": round(cl_visit_on_largest * ADC_FACTOR, 6),
        "v_raw_vox": round(arfs_res["v_raw"], 6),
        "ARFS_nm_s": round(arfs_res["arfs_nm_s"], 4),
        "ARFS_abs_nm_s": round(abs(arfs_res["arfs_nm_s"]), 4),
        "n_tracks_len3": int(arfs_res["n_tracks_len3"]),
        "used_fallback_001": bool(arfs_res["used_fallback_001"]),
        "drift_exceeded_10": bool(arfs_res["drift_exceeded_10"]),
        "max_td_norm": round(arfs_res["max_td_norm"], 4),
        "d_all_3d_strict": int(np.max(d_strict_list)),
        "dead_3d_strict_visit": bool(dead_3d_strict_visit),
        "dead_3d_within10_visit": bool(dead_3d_within10_visit),
        "dead_3d_s1_strict": bool(dead_3d_s1_strict),
        "dead_3d_s1_within10": bool(dead_3d_s1_within10),
        "max_d3d_voxels": int(np.max(d3d_vox_list)),
        "s1_d3d_voxels": int(d3d_vox_list[0]),
        "max_ov10_3d_binst_vox": int(np.max(ov10_vox_list)),
        "s1_ov10_3d_binst_vox": int(ov10_vox_list[0]),
        "ov_2d_xy_px": ov_xy,
        "ov_2d_xz_px": ov_xz,
        "ov10_2d_xy_px": ov10_xy,
        "ov10_2d_xz_px": ov10_xz,
        "dead_2d_proj_visit": bool(dead_2d_visit),
        "agree_3d_strict_vs_2d_proj": bool(dead_3d_strict_visit == dead_2d_visit),
        "agree_3d_s1_strict_vs_2d_proj": bool(dead_3d_s1_strict == dead_2d_visit),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

    return {
        "visit_row": visit_row,
        "sweep_rows": sweep_rows,
        "tcell_points_per_sweep": tcell_points_per_sweep,
    }


def summarize_chamber_void_max(values: np.ndarray, void_val: float = 0.0) -> tuple[float, int]:
    """
    Exact translation of `write_csv(..., void)` + `write_summary(..., factor, ismax=true)` in `summary.m`:
    Visits equal to `void` (0.0) are written as empty CSV cells (`NaN` in `readmatrix`), replaced by
    `-inf`, and `max` is taken over the remaining visits (returning `0.0` if all visits are `-inf`).
    Returns `(max_value, argmax_visit_1based)`.
    """
    valid_mask = np.abs(values - void_val) > 1e-12
    if not np.any(valid_mask):
        return 0.0, 0
    masked = np.where(valid_mask, values, -np.inf)
    best_idx0 = int(np.argmax(masked))
    return float(masked[best_idx0]), best_idx0 + 1


def first_sustained_visit(flags: np.ndarray) -> int | None:
    """First 1-based visit index where `flags[i]` and `flags[i+1]` are both True (`>=2` consecutive)."""
    for i in range(len(flags) - 1):
        if flags[i] and flags[i + 1]:
            return i + 1
    return None


def first_any_visit(flags: np.ndarray) -> int | None:
    """First 1-based visit index where `flags[i]` is True."""
    idx = np.where(flags)[0]
    return int(idx[0] + 1) if idx.size > 0 else None


def make_reproduction_figure(
    df_visits: pd.DataFrame,
    df_chambers: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Task 5: Create `figures/step3_actin_reproduction.png` following House Style:
      - Per-visit ARFS (nm/s), ADC (slope), and CA (µm²: actin-selected vs masks-only largest)
        for both chambers (`MIP_ROI_32` and `MIP_ROI_2`) with the paper's chamber value as a
        horizontal reference line.
      - A panel with 3D vs 2D projection death calls per visit for both chambers.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.edgecolor": "#C9C9C9",
            "axes.linewidth": 0.8,
            "xtick.color": "#5C5C5C",
            "ytick.color": "#5C5C5C",
            "axes.labelcolor": "#1F1F1F",
            "text.color": "#111111",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    # 4 rows (CA, ADC, ARFS, 3D vs 2D Death Calls) x 2 columns (MIP_ROI_32, MIP_ROI_2)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(11.2, 9.6),
        dpi=200,
         sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72], "hspace": 0.28, "wspace": 0.18},
    )

    c_blue = "#1F4E79"   # --plot-1
    c_orange = "#C0561F" # --plot-2
    c_green = "#2E7D5B"  # --plot-3
    c_purple = "#6B4C9A" # --plot-4
    c_ink = "#111111"
    c_muted = "#5C5C5C"
    c_rule = "#E3E3E3"

    chambers_order = ["MIP_ROI_32", "MIP_ROI_2"]

    for col_idx, roi in enumerate(chambers_order):
        sub = df_visits[df_visits["roi"] == roi].sort_values("visit").reset_index(drop=True)
        ch = df_chambers[df_chambers["roi"] == roi].iloc[0]
        visits = sub["visit"].to_numpy(dtype=int)

        cell_idx = int(ch["paper_cell_index"])
        col_title = f"{roi} (Group 1, Cell {cell_idx})"

        # Row 0: Contact Area (CA, um^2)
        ax_ca = axes[0, col_idx]
        ax_ca.set_title(col_title, fontsize=11, fontweight="bold", loc="left", pad=6)
        ax_ca.grid(True, axis="y", color=c_rule, linewidth=0.6)
        ax_ca.plot(
            visits,
            sub["CA_masks_largest_um2"],
            color=c_muted,
            linestyle="--",
            linewidth=1.4,
            marker="s",
            markersize=3.6,
            label=f"Masks-only CA (max = {ch['our_CA_masks_largest_um2']:.3f} µm²)",
        )
        ax_ca.plot(
            visits,
            sub["CA_actin_um2"],
            color=c_blue,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            label=f"Actin-selected CA (max = {ch['our_CA_actin_um2']:.3f} µm²)",
        )
        ax_ca.axhline(
            ch["paper_CA_um2"],
            color=c_orange,
            linestyle="-",
            linewidth=1.4,
            label=f"Paper CA = {ch['paper_CA_um2']:.3f} µm²",
        )
        peak_v_ca = int(ch["peak_visit_CA_actin"])
        if peak_v_ca > 0:
            peak_val_ca = float(sub.loc[sub["visit"] == peak_v_ca, "CA_actin_um2"].iloc[0])
            ax_ca.scatter([peak_v_ca], [peak_val_ca], color=c_orange, s=46, zorder=5)
        ca_min = float(min(sub["CA_actin_um2"].min(), sub["CA_masks_largest_um2"].min()))
        ca_max = float(max(sub["CA_actin_um2"].max(), sub["CA_masks_largest_um2"].max(), ch["paper_CA_um2"]))
        ca_span = max(10.0, ca_max - ca_min)
        if col_idx == 0:
            ax_ca.set_ylim(max(0.0, ca_min - 0.42 * ca_span), ca_max + 0.14 * ca_span)
            ax_ca.set_ylabel("CA (µm²)", fontsize=10, fontweight="bold")
            ax_ca.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor=c_rule, fontsize=7.8)
        else:
            ax_ca.set_ylim(max(0.0, ca_min - 0.08 * ca_span), ca_max + 0.36 * ca_span)
            ax_ca.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=c_rule, fontsize=7.8)

        # Row 1: Actin Depletion Coefficient (ADC)
        ax_adc = axes[1, col_idx]
        ax_adc.grid(True, axis="y", color=c_rule, linewidth=0.6)
        ax_adc.axhline(0.0, color=c_rule, linestyle=":", linewidth=0.9)
        ax_adc.plot(
            visits,
            sub["ADC"],
            color=c_green,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            label=f"Visit ADC (max = {ch['our_ADC']:.4f})",
        )
        ax_adc.axhline(
            ch["paper_ADC"],
            color=c_orange,
            linestyle="-",
            linewidth=1.4,
            label=f"Paper ADC = {ch['paper_ADC']:.4f}",
        )
        peak_v_adc = int(ch["peak_visit_ADC"])
        if peak_v_adc > 0:
            peak_val_adc = float(sub.loc[sub["visit"] == peak_v_adc, "ADC"].iloc[0])
            ax_adc.scatter([peak_v_adc], [peak_val_adc], color=c_orange, s=46, zorder=5)
        adc_min = float(sub["ADC"].min())
        adc_max = float(max(sub["ADC"].max(), ch["paper_ADC"]))
        adc_span = max(1.0, adc_max - adc_min)
        ax_adc.set_ylim(adc_min - 0.10 * adc_span, adc_max + 0.32 * adc_span)
        if col_idx == 0:
            ax_adc.set_ylabel("ADC (slope)", fontsize=10, fontweight="bold")
        ax_adc.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor=c_rule, fontsize=8)

        # Row 2: Actin Retrograde Flow Speed (ARFS, nm/s)
        ax_arfs = axes[2, col_idx]
        ax_arfs.grid(True, axis="y", color=c_rule, linewidth=0.6)
        ax_arfs.axhline(0.0, color=c_rule, linestyle=":", linewidth=0.9)
        ax_arfs.plot(
            visits,
            sub["ARFS_nm_s"],
            color=c_purple,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            label=f"Visit ARFS (max = {ch['our_ARFS_nm_s']:.3f} nm/s)",
        )
        ax_arfs.axhline(
            ch["paper_ARFS_nm_s"],
            color=c_orange,
            linestyle="-",
            linewidth=1.4,
            label=f"Paper ARFS = {ch['paper_ARFS_nm_s']:.3f} nm/s",
        )
        peak_v_arfs = int(ch["peak_visit_ARFS"])
        if peak_v_arfs > 0:
            peak_val_arfs = float(sub.loc[sub["visit"] == peak_v_arfs, "ARFS_nm_s"].iloc[0])
            ax_arfs.scatter([peak_v_arfs], [peak_val_arfs], color=c_orange, s=46, zorder=5)
        arfs_min = float(sub["ARFS_nm_s"].min())
        arfs_max = float(max(sub["ARFS_nm_s"].max(), ch["paper_ARFS_nm_s"]))
        arfs_span = max(10.0, arfs_max - arfs_min)
        ax_arfs.set_ylim(arfs_min - 0.10 * arfs_span, arfs_max + 0.32 * arfs_span)
        if col_idx == 0:
            ax_arfs.set_ylabel("ARFS (nm/s)", fontsize=10, fontweight="bold")
        ax_arfs.legend(
            loc="upper right" if col_idx == 0 else "lower left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=c_rule,
            fontsize=8,
        )

        # Row 3: 3D vs 2D Projection Death Calls per visit
        ax_d = axes[3, col_idx]
        ax_d.grid(True, axis="x", color=c_rule, linewidth=0.5)
        d3_strict = sub["dead_3d_strict_visit"].to_numpy(dtype=int)
        d3_s1 = sub["dead_3d_s1_strict"].to_numpy(dtype=int)
        d2_proj = sub["dead_2d_proj_visit"].to_numpy(dtype=int)

        # Plot horizontal binary tracks at y = 2 (3D sweeps 1..5), y = 1 (3D sweep 1), y = 0 (2D proj)
        for y_pos, arr_b, lbl_txt, col_c in [
            (2.0, d3_strict, "3D rule (sweeps 1–5)", c_blue),
            (1.0, d3_s1, "3D rule (sweep 1)", c_green),
            (0.0, d2_proj, "2D projection rule", c_orange),
        ]:
            alive_v = visits[arr_b == 0]
            dead_v = visits[arr_b == 1]
            ax_d.scatter(
                alive_v,
                np.full_like(alive_v, y_pos, dtype=float),
                facecolors="#FFFFFF",
                edgecolors="#A8A8A8",
                s=34,
                linewidths=1.0,
                zorder=3,
            )
            if dead_v.size > 0:
                ax_d.scatter(
                    dead_v,
                    np.full_like(dead_v, y_pos, dtype=float),
                    facecolors=col_c,
                    edgecolors=c_ink,
                    s=46,
                    linewidths=0.9,
                    zorder=4,
                )

        if pd.notna(ch["death_visit_2d_sustained"]):
            dv2 = int(ch["death_visit_2d_sustained"])
            ax_d.axvline(dv2, color=c_orange, linestyle="--", linewidth=1.1, zorder=2)

        ax_d.set_ylim(-0.6, 2.7)
        ax_d.set_yticks([0.0, 1.0, 2.0])
        if col_idx == 0:
            ax_d.set_yticklabels(["2D MIP", "3D (sw 1)", "3D (sw 1–5)"], fontsize=8.5, fontweight="bold")
        else:
            ax_d.set_yticklabels([])
        ax_d.set_xticks(range(1, 21, 2))
        ax_d.set_xlim(0.4, 20.6)
        ax_d.set_xlabel("Visit index (10 min / visit)", fontsize=10, fontweight="bold")

        agree_pct = float(ch["death_visit_agreement_3d_vs_2d_pct"])
        ax_d.text(
            0.02,
            0.88,
            f"3D vs 2D agreement: {int(ch['death_visit_agreement_3d_vs_2d_count'])}/20 ({agree_pct:.0f}%)",
            transform=ax_d.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=c_ink,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Fei lab 3D actin measures (ARFS, ADC, actin-selected CA) and 3D death calibration."
    )
    parser.add_argument("--workers", type=int, default=4, help="Max worker processes (default: 4, max: 4).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=STEP3_ACTIN_DIR,
        help="Output directory for CSV/JSON tables (default: data/step3_actin).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=FIGURES_DIR / "step3_actin_reproduction.png",
        help="Output figure path (default: figures/step3_actin_reproduction.png).",
    )
    args = parser.parse_args()

    workers = min(4, max(1, int(args.workers)))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # 1. Build task list for all 20 visits of both chambers
    tasks: list[dict[str, Any]] = []
    first_raw_path: Path | None = None
    first_pred_path: Path | None = None

    for ch_meta in TARGET_CHAMBERS:
        exp = ch_meta["experiment"]
        roi = ch_meta["roi"]
        raw_ch_dir = RAW_DIR / exp / roi
        mask_ch_dir = MASKS_DIR / exp / roi

        pred_files = sorted(
            mask_ch_dir.glob("*_predictions.h5"),
            key=lambda p: int(p.name.split("_")[0]),
        )
        if len(pred_files) != 120:
            raise RuntimeError(f"Expected 120 mask files for {exp}/{roi}, found {len(pred_files)}")

        for v_idx in range(1, 21):
            base = (v_idx - 1) * SWEEPS_PER_VISIT
            sweep_items = []
            for s_offset in range(ACTIN_SWEEPS_PER_VISIT):
                pf = pred_files[base + s_offset]
                stem = pf.name[: -len("_predictions.h5")]
                rf = raw_ch_dir / f"{stem}.h5"
                if not rf.exists():
                    raise FileNotFoundError(f"Missing raw file: {rf}")
                if first_raw_path is None:
                    first_raw_path = rf
                    first_pred_path = pf
                sweep_items.append((s_offset + 1, stem, rf, pf))
            tasks.append(
                {
                    "experiment": exp,
                    "roi": roi,
                    "visit": v_idx,
                    "sweep_items": sweep_items,
                }
            )

    # 2. Prove spot_detection equivalence on a 3D crop
    assert first_raw_path is not None and first_pred_path is not None
    print(f"Running spot_detection equivalence proof on {first_raw_path.name}...", flush=True)
    eq_report = prove_spot_detection_equivalence(first_raw_path, first_pred_path)
    eq_json_path = out_dir / "spot_detection_equivalence.json"
    eq_json_path.write_text(json.dumps(eq_report, indent=2), encoding="utf-8")
    print(
        f"  max |R_vec - R_loop| = {eq_report['max_abs_corrcoef_diff']:.3e}, "
        f"crop mismatch = {eq_report['crop_mask_mismatch_voxels']} vox, "
        f"restricted vs full mismatch = {eq_report['full_vs_tcell_restricted_t_instance_mismatch_voxels']} vox",
        flush=True,
    )

    # 3. Process all 40 visits across both chambers using ProcessPoolExecutor (max 4 workers)
    print(f"Processing {len(tasks)} visits across {len(TARGET_CHAMBERS)} chambers with {workers} workers...", flush=True)
    visit_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    tcell_pts_by_chamber_visit: dict[tuple[str, str, int], list[np.ndarray]] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx_done, res in enumerate(executor.map(process_one_visit, tasks), start=1):
            vr = res["visit_row"]
            visit_rows.append(vr)
            sweep_rows.extend(res["sweep_rows"])
            tcell_pts_by_chamber_visit[(vr["experiment"], vr["roi"], int(vr["visit"]))] = res["tcell_points_per_sweep"]
            if idx_done % 5 == 0 or idx_done == len(tasks):
                print(
                    f"  [{idx_done:2d}/{len(tasks):2d}] {vr['roi']} v{vr['visit']:02d}: "
                    f"CA_actin={vr['CA_actin_um2']:.2f} (masks={vr['CA_masks_largest_um2']:.2f}), "
                    f"ADC={vr['ADC']:.4f}, ARFS={vr['ARFS_nm_s']:.2f}, "
                    f"dead3D={vr['dead_3d_strict_visit']} dead2D={vr['dead_2d_proj_visit']} "
                    f"({vr['elapsed_s']:.2f}s)",
                    flush=True,
                )

    df_sweeps = pd.DataFrame(sweep_rows).sort_values(["roi", "visit", "sweep"]).reset_index(drop=True)
    df_visits = pd.DataFrame(visit_rows).sort_values(["roi", "visit"]).reset_index(drop=True)

    # 4. Build chamber-level summary and compare with paper & step4 death_labels
    chamber_rows: list[dict[str, Any]] = []
    for ch_meta in TARGET_CHAMBERS:
        exp = ch_meta["experiment"]
        roi = ch_meta["roi"]
        sub_v = df_visits[(df_visits["experiment"] == exp) & (df_visits["roi"] == roi)].sort_values("visit")
        sub_s = df_sweeps[(df_sweeps["experiment"] == exp) & (df_sweeps["roi"] == roi)]

        ca_actin_max, v_ca_actin = summarize_chamber_void_max(sub_v["CA_actin_um2"].to_numpy(), 0.0)
        ca_masks_max, v_ca_masks = summarize_chamber_void_max(sub_v["CA_masks_largest_um2"].to_numpy(), 0.0)
        ca_total_max, v_ca_total = summarize_chamber_void_max(sub_v["CA_masks_total_um2"].to_numpy(), 0.0)
        adc_max, v_adc = summarize_chamber_void_max(sub_v["ADC"].to_numpy(), 0.0)
        adc_largest_max, v_adc_largest = summarize_chamber_void_max(sub_v["ADC_on_largest"].to_numpy(), 0.0)
        arfs_max, v_arfs = summarize_chamber_void_max(sub_v["ARFS_nm_s"].to_numpy(), 0.0)
        arfs_abs_max, v_arfs_abs = summarize_chamber_void_max(sub_v["ARFS_abs_nm_s"].to_numpy(), 0.0)

        d3_strict_arr = sub_v["dead_3d_strict_visit"].to_numpy(dtype=bool)
        d3_s1_arr = sub_v["dead_3d_s1_strict"].to_numpy(dtype=bool)
        d2_proj_arr = sub_v["dead_2d_proj_visit"].to_numpy(dtype=bool)

        agree_strict_2d = int(np.sum(d3_strict_arr == d2_proj_arr))
        agree_s1_2d = int(np.sum(d3_s1_arr == d2_proj_arr))

        # Run step2_instance_seg.m tracker(t_instance) across all 120 timepoints
        pts_120: list[np.ndarray] = []
        for v_i in range(1, 21):
            pts_5 = tcell_pts_by_chamber_visit[(exp, roi, v_i)]
            pts_120.extend(pts_5)
            # 6th sweep of each visit is tubulin (mod(i, sample_num) == 0 -> t_instance is all 0)
            pts_120.append(np.empty((0, 4), dtype=np.float64))
        t_trace = tracker_tcell(pts_120)
        n_tcell_tracks = int(t_trace.shape[0])
        n_tracked_sweeps = int(np.sum(t_trace[0] > 0)) if n_tcell_tracks > 0 else 0

        chamber_rows.append(
            {
                "experiment": exp,
                "roi": roi,
                "group_id": ch_meta["group_id"],
                "paper_cell_index": ch_meta["paper_cell_index"],
                "paper_CA_um2": ch_meta["paper_CA_um2"],
                "our_CA_actin_um2": round(ca_actin_max, 4),
                "our_CA_masks_largest_um2": round(ca_masks_max, 4),
                "our_CA_masks_total_um2": round(ca_total_max, 4),
                "CA_actin_minus_paper_um2": round(ca_actin_max - ch_meta["paper_CA_um2"], 4),
                "CA_actin_rel_err_pct": round(
                    100.0 * (ca_actin_max - ch_meta["paper_CA_um2"]) / ch_meta["paper_CA_um2"], 4
                ),
                "CA_actin_minus_masks_largest_um2": round(ca_actin_max - ca_masks_max, 4),
                "peak_visit_CA_actin": v_ca_actin,
                "peak_visit_CA_masks": v_ca_masks,
                "n_tcell_tracks_step2": n_tcell_tracks,
                "n_tracked_actin_sweeps_step2": n_tracked_sweeps,
                "n_sweeps_comp_differs_from_largest": int(np.sum(~sub_s["selected_comp_is_largest"])),
                "paper_ADC": ch_meta["paper_ADC"],
                "our_ADC": round(adc_max, 4),
                "our_ADC_on_largest": round(adc_largest_max, 4),
                "ADC_minus_paper": round(adc_max - ch_meta["paper_ADC"], 4),
                "ADC_rel_err_pct": round(
                    100.0 * (adc_max - ch_meta["paper_ADC"]) / abs(ch_meta["paper_ADC"]), 4
                ),
                "peak_visit_ADC": v_adc,
                "paper_ARFS_nm_s": ch_meta["paper_ARFS_nm_s"],
                "our_ARFS_nm_s": round(arfs_max, 4),
                "our_ARFS_abs_nm_s": round(arfs_abs_max, 4),
                "ARFS_minus_paper_nm_s": round(arfs_max - ch_meta["paper_ARFS_nm_s"], 4),
                "ARFS_rel_err_pct": round(
                    100.0 * (arfs_max - ch_meta["paper_ARFS_nm_s"]) / ch_meta["paper_ARFS_nm_s"], 4
                ),
                "peak_visit_ARFS": v_arfs,
                "peak_visit_ARFS_abs": v_arfs_abs,
                "ever_dead_3d_strict_any_visit": bool(np.any(d3_strict_arr)),
                "first_dead_visit_3d_strict_any": first_any_visit(d3_strict_arr),
                "death_visit_3d_strict_sustained": first_sustained_visit(d3_strict_arr),
                "ever_dead_3d_s1_any_visit": bool(np.any(d3_s1_arr)),
                "first_dead_visit_3d_s1_any": first_any_visit(d3_s1_arr),
                "death_visit_3d_s1_sustained": first_sustained_visit(d3_s1_arr),
                "ever_dead_2d_any_visit": bool(np.any(d2_proj_arr)),
                "first_dead_visit_2d_any": first_any_visit(d2_proj_arr),
                "death_visit_2d_sustained": first_sustained_visit(d2_proj_arr),
                "n_dead_visits_3d_strict": int(np.sum(d3_strict_arr)),
                "n_dead_visits_3d_s1": int(np.sum(d3_s1_arr)),
                "n_dead_visits_2d_proj": int(np.sum(d2_proj_arr)),
                "death_visit_agreement_3d_vs_2d_count": agree_strict_2d,
                "death_visit_agreement_3d_vs_2d_pct": round(100.0 * agree_strict_2d / len(sub_v), 2),
                "death_visit_agreement_3d_s1_vs_2d_count": agree_s1_2d,
                "death_visit_agreement_3d_s1_vs_2d_pct": round(100.0 * agree_s1_2d / len(sub_v), 2),
            }
        )

    df_chambers = pd.DataFrame(chamber_rows)

    # Save tables
    sweeps_csv = out_dir / "per_sweep_actin_measures.csv"
    visits_csv = out_dir / "per_visit_actin_measures.csv"
    chambers_csv = out_dir / "per_chamber_actin_summary.csv"
    death_comp_csv = out_dir / "death_rule_3d_vs_2d_comparison.csv"

    df_sweeps.to_csv(sweeps_csv, index=False)
    df_visits.to_csv(visits_csv, index=False)
    df_chambers.to_csv(chambers_csv, index=False)

    death_cols = [
        "experiment",
        "roi",
        "visit",
        "time_min",
        "stem_s1",
        "CA_actin_um2",
        "d_all_3d_strict",
        "dead_3d_strict_visit",
        "dead_3d_within10_visit",
        "dead_3d_s1_strict",
        "dead_3d_s1_within10",
        "max_d3d_voxels",
        "s1_d3d_voxels",
        "max_ov10_3d_binst_vox",
        "s1_ov10_3d_binst_vox",
        "ov_2d_xy_px",
        "ov_2d_xz_px",
        "ov10_2d_xy_px",
        "ov10_2d_xz_px",
        "dead_2d_proj_visit",
        "agree_3d_strict_vs_2d_proj",
        "agree_3d_s1_strict_vs_2d_proj",
    ]
    df_visits[death_cols].to_csv(death_comp_csv, index=False)

    # 5. Generate figure
    make_reproduction_figure(df_visits, df_chambers, args.figure)

    summary_payload = {
        "elapsed_total_s": round(time.perf_counter() - t_start, 2),
        "workers": workers,
        "spot_detection_equivalence": eq_report,
        "chambers": chamber_rows,
        "outputs": {
            "per_sweep_csv": str(sweeps_csv.relative_to(REPO_ROOT)),
            "per_visit_csv": str(visits_csv.relative_to(REPO_ROOT)),
            "per_chamber_csv": str(chambers_csv.relative_to(REPO_ROOT)),
            "death_comparison_csv": str(death_comp_csv.relative_to(REPO_ROOT)),
            "figure_png": str(args.figure.relative_to(REPO_ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("=" * 94)
    print("STEP 3 (PART 2) ACTIN & STEP 4 3D DEATH CALIBRATION SUMMARY")
    print("=" * 94)
    for row in chamber_rows:
        print(
            f"{row['roi']} (Cell {row['paper_cell_index']}):\n"
            f"  CA (µm²):   paper={row['paper_CA_um2']:.3f} | our_actin={row['our_CA_actin_um2']:.3f} "
            f"(v{row['peak_visit_CA_actin']:02d}) | our_masks={row['our_CA_masks_largest_um2']:.3f} "
            f"| diff={row['CA_actin_minus_paper_um2']:+.3f} ({row['CA_actin_rel_err_pct']:+.2f}%)\n"
            f"  ADC:        paper={row['paper_ADC']:.4f} | our_ADC={row['our_ADC']:.4f} "
            f"(v{row['peak_visit_ADC']:02d}) | diff={row['ADC_minus_paper']:+.4f} ({row['ADC_rel_err_pct']:+.2f}%)\n"
            f"  ARFS(nm/s): paper={row['paper_ARFS_nm_s']:.3f} | our_ARFS={row['our_ARFS_nm_s']:.3f} "
            f"(v{row['peak_visit_ARFS']:02d}) | diff={row['ARFS_minus_paper_nm_s']:+.3f} ({row['ARFS_rel_err_pct']:+.2f}%)\n"
            f"  Death rule: 3D_strict_sustained={row['death_visit_3d_strict_sustained']} "
            f"(any={row['first_dead_visit_3d_strict_any']}, n={row['n_dead_visits_3d_strict']}) | "
            f"2D_sustained={row['death_visit_2d_sustained']} (any={row['first_dead_visit_2d_any']}, n={row['n_dead_visits_2d_proj']}) | "
            f"visit agreement={row['death_visit_agreement_3d_vs_2d_count']}/20 ({row['death_visit_agreement_3d_vs_2d_pct']:.1f}%)"
        )
    print("=" * 94, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
