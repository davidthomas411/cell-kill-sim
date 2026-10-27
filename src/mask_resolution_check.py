#!/usr/bin/env python3
"""
Step 5 prerequisite: Measure surface detail preserved by the deposited 3D U-Net masks
versus raw 561 nm target membrane (mApple-Mem) fluorescence stacks.

Tasks:
1. Axial vs lateral: measure 10-90% edge sharpness along surface normals mostly in x, y, z
   for both raw membrane intensity and mask boundary; compare Z extents (166-190 slices)
   and target cell diameters in xy vs z across the 20 raw volume pairs and a stratified sample
   of v01-v03 masks; estimate implied z pitch assuming targets are on average rotationally symmetric.
2. Mask vs raw surface: extract target surface mesh / Dual Surface Net vertices (marching-cubes
   style sub-voxel isosurface on solid target mask 2|4), locate sub-voxel raw membrane intensity
   peak via parabolic interpolation along outward normals, and report mean, SD, median, IQR,
   and spatial autocorrelation length (1/e and half-maximum) of mask-to-membrane distance.
3. Surface roughness spectrum: compute local height fluctuations relative to a smooth local quadric
   reference surface on (a) the mask surface and (b) the raw membrane-peak surface, separately
   near the T-cell contact patch (within 2 voxels of labels 1|3) and away from contact.
   Report half-power (-3 dB, P_mask/P_raw = 0.5) and noise-floor cutoff wavelengths in um.
4. Save figure `figures/step5_mask_resolution_check.png` and summary tables/JSON under
   `data/step5_resolution/`.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)
from scipy.spatial import cKDTree

try:
    from src.constants import OPTICAL_FWHM_UM, VOXEL_UM
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from constants import OPTICAL_FWHM_UM, VOXEL_UM
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT


OUT_DIR: Path = DATA_DIR / "step5_resolution"
FIG_PATH: Path = FIGURES_DIR / "step5_mask_resolution_check.png"
CHAMBERS_DIR: Path = DATA_DIR / "scidb_19235_chambers"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"

STRUCT26: np.ndarray = np.ones((3, 3, 3), dtype=bool)


def find_raw_mask_pairs() -> list[dict[str, Any]]:
    """Locate all local raw .h5 volumes in data/scidb_19235_chambers and their matching prediction masks."""
    raw_files = sorted(glob.glob(str(CHAMBERS_DIR / "*" / "[0-9][0-9][0-9][0-9][0-9].h5")))
    pairs: list[dict[str, Any]] = []
    for rf in raw_files:
        rf_path = Path(rf)
        chamber = rf_path.parent.name
        exp, roi_part = chamber.split("_MIP_")
        roi = "MIP_" + roi_part
        stem = rf_path.stem
        pf1 = rf_path.with_name(f"{stem}_predictions.h5")
        pf2 = MASKS_DIR / exp / roi / f"{stem}_predictions.h5"
        pf3 = DATA_DIR / "scidb_19235_samples" / f"{stem}_predictions.h5"
        pf = pf1 if pf1.exists() else (pf2 if pf2.exists() else (pf3 if pf3.exists() else None))
        if pf is not None:
            pairs.append({
                "chamber": chamber,
                "experiment": exp,
                "roi": roi,
                "stem": stem,
                "raw_path": rf_path,
                "mask_path": pf,
            })
    return pairs


def build_solid_target_mask(tgt_raw: np.ndarray) -> np.ndarray:
    """
    Fill interior cytoplasm holes of a binary target mask (labels 2|4) by closing
    2D slice holes along Z, Y, and X followed by 3D binary hole filling.
    """
    m = tgt_raw.copy()
    for z in range(m.shape[0]):
        if np.any(m[z]):
            m[z] |= binary_fill_holes(tgt_raw[z])
    for y in range(m.shape[1]):
        if np.any(m[:, y, :]):
            m[:, y, :] |= binary_fill_holes(tgt_raw[:, y, :])
    for x in range(m.shape[2]):
        if np.any(m[:, :, x]):
            m[:, :, x] |= binary_fill_holes(tgt_raw[:, :, x])
    return binary_fill_holes(m, structure=STRUCT26)


def extract_subvoxel_surface(
    m_solid: np.ndarray, sigma_surf: float = 1.0, sigma_norm: float = 2.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sub-voxel Dual Surface Net / marching-cubes style surface vertices and outward
    unit normals from a solid binary target mask `m_solid`.
    Returns:
      verts_zyx: (N, 3) float64 sub-voxel surface coordinates in (z, y, x) voxel units
      norms_zyx: (N, 3) float64 smooth outward unit normals in (z, y, x)
      sdf_exact: (Z, Y, X) float64 signed distance field (negative inside, positive outside, 0 at boundary)
    """
    sdf_exact = (distance_transform_edt(~m_solid) - 0.5) - (distance_transform_edt(m_solid) - 0.5)
    sdf_s = gaussian_filter(sdf_exact, sigma=sigma_surf)
    sdf_n = gaussian_filter(sdf_exact, sigma=sigma_norm)

    gz, gy, gx = np.gradient(sdf_n)
    gnorm = np.sqrt(gz**2 + gy**2 + gx**2) + 1e-8
    nz, ny, nx = gz / gnorm, gy / gnorm, gx / gnorm

    c000 = sdf_s[:-1, :-1, :-1] < 0
    s_c = (
        c000.astype(int)
        + (sdf_s[1:, :-1, :-1] < 0)
        + (sdf_s[:-1, 1:, :-1] < 0)
        + (sdf_s[:-1, :-1, 1:] < 0)
        + (sdf_s[1:, 1:, :-1] < 0)
        + (sdf_s[1:, :-1, 1:] < 0)
        + (sdf_s[:-1, 1:, 1:] < 0)
        + (sdf_s[1:, 1:, 1:] < 0)
    )
    active = (s_c > 0) & (s_c < 8)
    ijk = np.argwhere(active).astype(np.float64) + 0.5
    if len(ijk) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), sdf_exact

    phi_v = map_coordinates(sdf_s, [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1)
    nv = np.stack([
        map_coordinates(nz, [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1),
        map_coordinates(ny, [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1),
        map_coordinates(nx, [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1),
    ], axis=1)
    nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-8
    verts = ijk - phi_v[:, None] * nv
    return verts, nv, sdf_exact


def analyze_single_volume(pair: dict[str, Any], f_bins: np.ndarray) -> dict[str, Any] | None:
    """Run Tasks 1, 2, 3 measurements on a single raw + mask volume pair."""
    with h5py.File(pair["raw_path"], "r") as f_r, h5py.File(pair["mask_path"], "r") as f_p:
        mem_raw = f_r["raw"][2].astype(np.float32)
        pred = f_p["predictions"][:]

    Z, Y, X = pred.shape
    tgt_all = (pred == 2) | (pred == 4)
    tcell_all = (pred == 1) | (pred == 3)
    if int(np.sum(tgt_all)) < 5000:
        return None

    z_any = np.where(np.any(tgt_all | tcell_all, axis=(1, 2)))[0]
    y_any = np.where(np.any(tgt_all | tcell_all, axis=(0, 2)))[0]
    x_any = np.where(np.any(tgt_all | tcell_all, axis=(0, 1)))[0]
    pad = 8
    z0_s, z1_s = max(0, int(z_any[0]) - pad), min(Z, int(z_any[-1]) + pad + 1)
    y0_s, y1_s = max(0, int(y_any[0]) - pad), min(Y, int(y_any[-1]) + pad + 1)
    x0_s, x1_s = max(0, int(x_any[0]) - pad), min(X, int(x_any[-1]) + pad + 1)
    sl = (slice(z0_s, z1_s), slice(y0_s, y1_s), slice(x0_s, x1_s))

    tgt_sub = tgt_all[sl]
    tcell_sub = tcell_all[sl]
    mem_sub = mem_raw[sl]

    lbl, n_comp = label(tgt_sub, structure=STRUCT26)
    if n_comp == 0:
        return None
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    main_id = int(np.argmax(counts))
    if counts[main_id] < 5000:
        return None

    m_raw = lbl == main_id
    m_solid = build_solid_target_mask(m_raw)

    # Bounding-box and gyration extents of main target cell
    zz, yy, xx = np.where(m_solid)
    z0_c, z1_c = int(zz.min() + z0_s), int(zz.max() + z0_s)
    y0_c, y1_c = int(yy.min() + y0_s), int(yy.max() + y0_s)
    x0_c, x1_c = int(xx.min() + x0_s), int(xx.max() + x0_s)
    touches_border = (
        (z0_c <= 1) or (z1_c >= Z - 2)
        or (y0_c <= 1) or (y1_c >= Y - 2)
        or (x0_c <= 1) or (x1_c >= X - 2)
    )
    dz_vox = float(z1_c - z0_c + 1)
    dy_vox = float(y1_c - y0_c + 1)
    dx_vox = float(x1_c - x0_c + 1)
    sz_vox = float(zz.std()) * np.sqrt(12.0)
    sy_vox = float(yy.std()) * np.sqrt(12.0)
    sx_vox = float(xx.std()) * np.sqrt(12.0)

    # Extract surface vertices and normals
    verts, nv, sdf_exact = extract_subvoxel_surface(m_solid, sigma_surf=1.0, sigma_norm=2.5)
    if len(verts) < 200:
        return None

    # Contact patch mask: target surface within 2 voxels of T-cell labels 1|3
    if np.any(tcell_sub):
        dist_t = distance_transform_edt(~tcell_sub)
        d_t_verts = map_coordinates(dist_t, [verts[:, 0], verts[:, 1], verts[:, 2]], order=1)
    else:
        d_t_verts = np.full(len(verts), 999.0, dtype=np.float64)
    is_patch = d_t_verts <= 2.0

    #Sample raw membrane intensity along normal from s = -10.0 to +6.0 voxels (ds = 0.20 vox, 81 steps)
    s_grid = np.linspace(-10.0, 6.0, 81)
    ds = float(s_grid[1] - s_grid[0])
    coords = verts[:, :, None] + nv[:, :, None] * s_grid[None, None, :]
    prof_raw = map_coordinates(
        mem_sub, [coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]], order=1, mode="nearest"
    )
    prof_mask = map_coordinates(
        m_solid.astype(np.float32),
        [coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]],
        order=1,
        mode="nearest",
    )

    # Sub-voxel local peak position in [-8.0, +3.0] voxels (indices 10..65)
    # Prefer interior local maxima over window endpoints to avoid clamping spikes
    i_min, i_max = 10, 65
    sub_p = prof_raw[:, i_min : i_max + 1]
    is_loc_max = np.zeros_like(sub_p, dtype=bool)
    is_loc_max[:, 1:-1] = (sub_p[:, 1:-1] >= sub_p[:, :-2]) & (sub_p[:, 1:-1] >= sub_p[:, 2:])
    masked_p = np.where(is_loc_max, sub_p, -np.inf)
    has_loc = np.any(is_loc_max, axis=1)
    idx_pk_rel = np.where(has_loc, np.argmax(masked_p, axis=1), np.argmax(sub_p, axis=1))
    idx_pk = i_min + idx_pk_rel
    idx_c = np.clip(idx_pk, 1, len(s_grid) - 2)
    row_idx = np.arange(len(verts))
    ym = prof_raw[row_idx, idx_c - 1]
    y0 = prof_raw[row_idx, idx_c]
    yp = prof_raw[row_idx, idx_c + 1]
    denom = ym - 2.0 * y0 + yp
    safe_denom = np.where(denom < -1e-5, denom, -1.0)
    delta = np.where(denom < -1e-5, 0.5 * (ym - yp) / safe_denom, 0.0)
    s_pk_vox = s_grid[idx_c] + np.clip(delta, -0.5, 0.5) * ds
    d_pk_um = s_pk_vox * VOXEL_UM

    # 10-90% rise distance along normals mostly in x, y, z (|n_axis| >= 0.80)
    # Outer baseline is median intensity in s in [+3.0, +6.0] voxels (indices 65..80)
    y_base = np.median(prof_raw[:, 65:], axis=1)
    amp = y0 - y_base
    valid_snr = amp > np.maximum(20.0, 0.05 * y_base)

    rise_raw_by_axis: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    rise_mask_by_axis: dict[str, list[float]] = {"x": [], "y": [], "z": []}

    for ax_i, ax_name in [(2, "x"), (1, "y"), (0, "z")]:
        sel_ax = np.where(valid_snr & (np.abs(nv[:, ax_i]) >= 0.80))[0]
        for r_i in sel_ax:
            k_pk = int(idx_c[r_i])
            seg = prof_raw[r_i, k_pk:76]  # from peak outward to s = +5.0 vox
            s_seg = s_grid[k_pk:76]
            if len(seg) >= 4:
                lo = y_base[r_i] + 0.10 * amp[r_i]
                hi = y_base[r_i] + 0.90 * amp[r_i]
                idx_hi = np.where(seg <= hi)[0]
                idx_lo = np.where(seg <= lo)[0]
                if len(idx_hi) > 0 and len(idx_lo) > 0 and idx_lo[0] >= idx_hi[0]:
                    k1 = max(0, int(idx_hi[0]) - 1)
                    d1 = seg[idx_hi[0]] - seg[k1]
                    f1 = (hi - seg[k1]) / d1 if abs(d1) > 1e-6 and idx_hi[0] > 0 else 0.0
                    s90 = s_seg[k1] + np.clip(f1, 0.0, 1.0) * ds

                    k2 = max(0, int(idx_lo[0]) - 1)
                    d2 = seg[idx_lo[0]] - seg[k2]
                    f2 = (lo - seg[k2]) / d2 if abs(d2) > 1e-6 and idx_lo[0] > 0 else 0.0
                    s10 = s_seg[k2] + np.clip(f2, 0.0, 1.0) * ds
                    if s10 > s90:
                        rise_raw_by_axis[ax_name].append(float((s10 - s90) * VOXEL_UM))

            # Mask 10-90% rise distance along the same normal ray
            m_seg = prof_mask[r_i, 40:61]  # s in [-2.0, +2.0] vox
            s_mseg = s_grid[40:61]
            ih = np.where(m_seg <= 0.90)[0]
            il = np.where(m_seg <= 0.10)[0]
            if len(ih) > 0 and len(il) > 0 and il[0] >= ih[0]:
                k1 = max(0, int(ih[0]) - 1)
                d1 = m_seg[ih[0]] - m_seg[k1]
                f1 = (0.90 - m_seg[k1]) / d1 if abs(d1) > 1e-6 and ih[0] > 0 else 0.0
                sm90 = s_mseg[k1] + np.clip(f1, 0.0, 1.0) * ds

                k2 = max(0, int(il[0]) - 1)
                d2 = m_seg[il[0]] - m_seg[k2]
                f2 = (0.10 - m_seg[k2]) / d2 if abs(d2) > 1e-6 and il[0] > 0 else 0.0
                sm10 = s_mseg[k2] + np.clip(f2, 0.0, 1.0) * ds
                if sm10 > sm90:
                    rise_mask_by_axis[ax_name].append(float((sm10 - sm90) * VOXEL_UM))

    # Spatial autocorrelation of d_pk_um over the 3D surface
    verts_um = verts * VOXEL_UM
    raw_verts_um = (verts + s_pk_vox[:, None] * nv) * VOXEL_UM
    rng = np.random.default_rng(42)
    sub_n = min(4000, len(verts_um))
    sub_idx = rng.choice(len(verts_um), size=sub_n, replace=False)
    pts_sub = verts_um[sub_idx]
    val_sub = d_pk_um[sub_idx] - float(np.mean(d_pk_um[sub_idx]))
    var_val = float(np.var(val_sub)) + 1e-12
    tree_sub = cKDTree(pts_sub)
    pairs_arr = tree_sub.query_pairs(r=5.0, output_type="ndarray")
    r_bins = np.linspace(0.0, 5.0, 26)  # 0.2 um bins
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    if len(pairs_arr) > 0:
        dists = np.linalg.norm(pts_sub[pairs_arr[:, 0]] - pts_sub[pairs_arr[:, 1]], axis=1)
        prods = val_sub[pairs_arr[:, 0]] * val_sub[pairs_arr[:, 1]]
        b_idx = np.digitize(dists, r_bins)
        corr_r = np.array([
            float(np.mean(prods[b_idx == k])) / var_val if np.any(b_idx == k) else np.nan
            for k in range(1, len(r_bins))
        ])
    else:
        corr_r = np.full(len(r_centers), np.nan)

    # Interpolate 1/e (0.3679) and half-max (0.5) correlation lengths
    def find_crossing(r_arr: np.ndarray, c_arr: np.ndarray, thr: float) -> float:
        valid_c = ~np.isnan(c_arr)
        r_v, c_v = r_arr[valid_c], c_arr[valid_c]
        if len(c_v) < 2:
            return float("nan")
        idx_below = np.where(c_v <= thr)[0]
        if len(idx_below) == 0:
            return float(r_v[-1])
        k = int(idx_below[0])
        if k == 0:
            return float(r_v[0] * (thr / max(c_v[0], 1e-4)))
        c0, c1 = c_v[k - 1], c_v[k]
        r0, r1 = r_v[k - 1], r_v[k]
        frac = (thr - c0) / (c1 - c0 - 1e-8)
        return float(r0 + np.clip(frac, 0.0, 1.0) * (r1 - r0))

    corr_len_1e = find_crossing(r_centers, corr_r, float(np.exp(-1.0)))
    corr_len_half = find_crossing(r_centers, corr_r, 0.50)

    # Task 3: Local height fluctuations & 2D tangent-patch roughness PSD
    tree_full = cKDTree(verts_um)
    grid_N, dx_grid = 24, 0.14
    half_L = 0.5 * (grid_N - 1) * dx_grid  # 1.61 um radius (3.22 x 3.22 um tangent window)
    u_1d = np.linspace(-half_L, half_L, grid_N)
    uu, vv = np.meshgrid(u_1d, u_1d)
    win = np.outer(np.hanning(grid_N), np.hanning(grid_N))
    win /= np.sqrt(np.mean(win**2))
    freqs = np.fft.fftfreq(grid_N, d=dx_grid)
    fx, fy = np.meshgrid(freqs, freqs)
    f_rad = np.sqrt(fx**2 + fy**2)
    b_f_idx = np.digitize(f_rad.ravel(), f_bins)

    psd_mask_patch, psd_raw_patch = [], []
    psd_mask_else, psd_raw_else = [], []
    rms_mask_patch, rms_raw_patch = [], []
    rms_mask_else, rms_raw_else = [], []

    for is_p, pm_list, pr_list, rm_list, rr_list in [
        (True, psd_mask_patch, psd_raw_patch, rms_mask_patch, rms_raw_patch),
        (False, psd_mask_else, psd_raw_else, rms_mask_else, rms_raw_else),
    ]:
        cand = np.where(is_patch if is_p else ~is_patch)[0]
        if len(cand) < 20:
            continue
        sel = rng.choice(cand, size=min(30, len(cand)), replace=False)
        for idx in sel:
            p0, n0 = verts_um[idx], nv[idx]
            arb = np.array([1.0, 0.0, 0.0]) if abs(n0[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(n0, arb)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n0, t1)
            t2 /= np.linalg.norm(t2)

            nb_idx = tree_full.query_ball_point(p0, r=half_L * 1.45)
            if len(nb_idx) < 50:
                continue
            rel_m = verts_um[nb_idx] - p0
            rel_r = raw_verts_um[nb_idx] - p0
            valid = (nv[nb_idx] @ n0) > 0.5
            if int(np.sum(valid)) < 50:
                continue
            rel_m, rel_r = rel_m[valid], rel_r[valid]
            u_pts, v_pts = rel_m @ t1, rel_m @ t2
            w_mask, w_raw = rel_m @ n0, rel_r @ n0

            A = np.column_stack([
                np.ones_like(u_pts), u_pts, v_pts, u_pts**2, u_pts * v_pts, v_pts**2
            ])
            cm, _, _, _ = np.linalg.lstsq(A, w_mask, rcond=None)
            cr, _, _, _ = np.linalg.lstsq(A, w_raw, rcond=None)
            res_m = w_mask - (A @ cm)
            res_r = w_raw - (A @ cr)
            rm_list.append(float(np.std(res_m)))
            rr_list.append(float(np.std(res_r)))

            tree2d = cKDTree(np.column_stack([u_pts, v_pts]))
            d2, ii = tree2d.query(np.column_stack([uu.ravel(), vv.ravel()]), k=3)
            w_idw = 1.0 / (d2 + 0.02) ** 2
            w_idw /= np.sum(w_idw, axis=1, keepdims=True)
            grid_m = np.sum(w_idw * res_m[ii], axis=1).reshape(grid_N, grid_N)
            grid_r = np.sum(w_idw * res_r[ii], axis=1).reshape(grid_N, grid_N)

            F_m = np.abs(np.fft.fft2((grid_m - grid_m.mean()) * win)) ** 2 * (
                dx_grid**2 / (grid_N**2)
            )
            F_r = np.abs(np.fft.fft2((grid_r - grid_r.mean()) * win)) ** 2 * (
                dx_grid**2 / (grid_N**2)
            )

            pm_list.append([
                float(np.mean(F_m.ravel()[b_f_idx == k])) if np.any(b_f_idx == k) else np.nan
                for k in range(1, len(f_bins))
            ])
            pr_list.append([
                float(np.mean(F_r.ravel()[b_f_idx == k])) if np.any(b_f_idx == k) else np.nan
                for k in range(1, len(f_bins))
            ])

    return {
        "chamber": pair["chamber"],
        "experiment": pair["experiment"],
        "roi": pair["roi"],
        "stem": pair["stem"],
        "Z_slices": int(Z),
        "target_voxels_solid": int(np.sum(m_solid)),
        "n_surface_verts": int(len(verts)),
        "n_patch_verts": int(np.sum(is_patch)),
        "touches_border": bool(touches_border),
        "dz_vox": dz_vox,
        "dy_vox": dy_vox,
        "dx_vox": dx_vox,
        "sz_vox": sz_vox,
        "sy_vox": sy_vox,
        "sx_vox": sx_vox,
        "d_pk_mean_um": float(np.mean(d_pk_um)),
        "d_pk_sd_um": float(np.std(d_pk_um)),
        "d_pk_median_um": float(np.median(d_pk_um)),
        "d_pk_iqr_um": float(np.percentile(d_pk_um, 75) - np.percentile(d_pk_um, 25)),
        "d_pk_abs_mean_um": float(np.mean(np.abs(d_pk_um))),
        "d_pk_patch_mean_um": float(np.mean(d_pk_um[is_patch])) if np.any(is_patch) else float("nan"),
        "d_pk_patch_sd_um": float(np.std(d_pk_um[is_patch])) if np.any(is_patch) else float("nan"),
        "d_pk_else_mean_um": float(np.mean(d_pk_um[~is_patch])) if np.any(~is_patch) else float("nan"),
        "d_pk_else_sd_um": float(np.std(d_pk_um[~is_patch])) if np.any(~is_patch) else float("nan"),
        "corr_len_1e_um": corr_len_1e,
        "corr_len_half_um": corr_len_half,
        "corr_r": corr_r,
        "r_centers": r_centers,
        "rise_raw_by_axis": rise_raw_by_axis,
        "rise_mask_by_axis": rise_mask_by_axis,
        "d_pk_all_um": d_pk_um,
        "is_patch_all": is_patch,
        "psd_mask_patch": psd_mask_patch,
        "psd_raw_patch": psd_raw_patch,
        "psd_mask_else": psd_mask_else,
        "psd_raw_else": psd_raw_else,
        "rms_mask_patch": float(np.mean(rms_mask_patch)) if rms_mask_patch else float("nan"),
        "rms_raw_patch": float(np.mean(rms_raw_patch)) if rms_raw_patch else float("nan"),
        "rms_mask_else": float(np.mean(rms_mask_else)) if rms_mask_else else float("nan"),
        "rms_raw_else": float(np.mean(rms_raw_else)) if rms_raw_else else float("nan"),
    }


def analyze_v01_v03_mask_extents() -> dict[str, Any]:
    """
    Sample 1 midpoint mask volume per ROI across v01, v02, v03 (120 volumes total)
    to measure Z extents and non-border target cell diameters in xy vs z.
    """
    mask_files = []
    for v in ["v01_6_40_120_0-6_hd1", "v02_6_40_120_0-6_hd3", "v03_6_40_120_0-6_hd4"]:
        rois = sorted(glob.glob(str(MASKS_DIR / v / "MIP_ROI_*")))
        for r in rois:
            f_list = sorted(glob.glob(str(Path(r) / "*_predictions.h5")))
            if f_list:
                mask_files.append((v, Path(r).name, Path(f_list[len(f_list) // 2])))

    z_extents_by_exp: dict[str, set[int]] = {}
    rows = []
    for v, r, mf in mask_files:
        with h5py.File(mf, "r") as f:
            pred = f["predictions"][:]
        Z, Y, X = pred.shape
        z_extents_by_exp.setdefault(v, set()).add(int(Z))
        tgt = (pred == 2) | (pred == 4)
        if not np.any(tgt):
            continue
        z_any = np.where(np.any(tgt, axis=(1, 2)))[0]
        y_any = np.where(np.any(tgt, axis=(0, 2)))[0]
        x_any = np.where(np.any(tgt, axis=(0, 1)))[0]
        sl = (
            slice(int(z_any[0]), int(z_any[-1]) + 1),
            slice(int(y_any[0]), int(y_any[-1]) + 1),
            slice(int(x_any[0]), int(x_any[-1]) + 1),
        )
        lbl, _ = label(tgt[sl], structure=STRUCT26)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        for k in np.where(counts >= 5000)[0]:
            zz, yy, xx = np.where(lbl == k)
            z0 = int(zz.min() + sl[0].start)
            z1 = int(zz.max() + sl[0].start)
            y0 = int(yy.min() + sl[1].start)
            y1 = int(yy.max() + sl[1].start)
            x0 = int(xx.min() + sl[2].start)
            x1 = int(xx.max() + sl[2].start)
            border = (
                (z0 <= 1) or (z1 >= Z - 2)
                or (y0 <= 1) or (y1 >= Y - 2)
                or (x0 <= 1) or (x1 >= X - 2)
            )
            if not border:
                rows.append({
                    "experiment": v,
                    "roi": r,
                    "Z_slices": int(Z),
                    "voxels": int(counts[k]),
                    "dz_vox": float(z1 - z0 + 1),
                    "dy_vox": float(y1 - y0 + 1),
                    "dx_vox": float(x1 - x0 + 1),
                    "sz_vox": float(zz.std()) * np.sqrt(12.0),
                    "sy_vox": float(yy.std()) * np.sqrt(12.0),
                    "sx_vox": float(xx.std()) * np.sqrt(12.0),
                })

    df = pd.DataFrame(rows)
    d_xy = 0.5 * (df["dx_vox"].to_numpy() + df["dy_vox"].to_numpy())
    dz = df["dz_vox"].to_numpy()
    ratio_bbox = float(np.mean(dz) / np.mean(d_xy))
    ratio_bbox_med = float(np.median(dz / d_xy))

    s_xy = 0.5 * (df["sx_vox"].to_numpy() + df["sy_vox"].to_numpy())
    sz = df["sz_vox"].to_numpy()
    ratio_gyr = float(np.mean(sz) / np.mean(s_xy))

    # If cells are rotationally symmetric in physical space (D_z_um = D_xy_um),
    # then dz_vox * pitch_z = d_xy_vox * VOXEL_UM  =>  pitch_z = VOXEL_UM * (d_xy_vox / dz_vox)
    implied_z_pitch_bbox_um = float(VOXEL_UM / ratio_bbox)
    implied_z_pitch_med_um = float(VOXEL_UM / ratio_bbox_med)

    return {
        "n_sampled_files": len(mask_files),
        "n_nonborder_cells": len(df),
        "z_extents_by_exp": {k: sorted(list(v)) for k, v in z_extents_by_exp.items()},
        "dz_mean_vox": float(np.mean(dz)),
        "dz_sd_vox": float(np.std(dz)),
        "dy_mean_vox": float(np.mean(df["dy_vox"])),
        "dy_sd_vox": float(np.std(df["dy_vox"])),
        "dx_mean_vox": float(np.mean(df["dx_vox"])),
        "dx_sd_vox": float(np.std(df["dx_vox"])),
        "dz_over_dxy_mean": ratio_bbox,
        "dz_over_dxy_median": ratio_bbox_med,
        "sz_over_sxy_mean": ratio_gyr,
        "implied_z_pitch_bbox_um": implied_z_pitch_bbox_um,
        "implied_z_pitch_med_um": implied_z_pitch_med_um,
        "df": df,
    }


def make_figure(
    rep_pair: dict[str, Any],
    vol_results: list[dict[str, Any]],
    v01_v03_info: dict[str, Any],
    wl_um: np.ndarray,
    psd_summary: dict[str, np.ndarray],
    summary_stats: dict[str, Any],
) -> None:
    """
    Generate 4-panel figure `figures/step5_mask_resolution_check.png` following House Style:
      (a) One xy and one xz slice through a contact (`v02/MIP_ROI_12/00075`), raw membrane channel
          with target mask outline and T-cell contact outline;
      (b) 10-90% edge sharpness by axis (x, y, z) for raw membrane vs mask boundary;
      (c) Histogram of sub-voxel mask-to-membrane peak distance (patch vs elsewhere) + inset of
          spatial autocorrelation C(r);
      (d) Surface roughness power spectrum vs wavelength, mask vs raw, contact patch vs elsewhere,
          with 0.32 um optical FWHM and 2x voxel (0.392 um) marked.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": "#111111",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "text.color": "#111111",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    # House style palette
    C_PLOT1 = "#1F4E79"  # deep blue (Mask / X)
    C_PLOT2 = "#C0561F"  # rust orange (Raw / Z / Patch)
    C_PLOT3 = "#2E7D5B"  # green (T-cell / Y)
    C_PLOT4 = "#6B4C9A"  # purple
    C_INK = "#111111"
    C_MUTED = "#5C5C5C"
    C_RULE = "#E3E3E3"

    fig = plt.figure(figsize=(12.5, 9.2), dpi=200)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26, left=0.07, right=0.97, top=0.94, bottom=0.07)

    # -------------------------------------------------------------------------
    # Panel (a): XY and XZ slice through contact in representative volume (00075)
    # -------------------------------------------------------------------------
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.12)
    ax_xy = fig.add_subplot(gs_a[0, 0])
    ax_xz = fig.add_subplot(gs_a[0, 1])

    with h5py.File(rep_pair["raw_path"], "r") as f_r, h5py.File(rep_pair["mask_path"], "r") as f_p:
        mem_raw = f_r["raw"][2].astype(np.float32)
        pred = f_p["predictions"][:]

    tgt_mask = build_solid_target_mask((pred == 2) | (pred == 4))
    tcell_mask = (pred == 1) | (pred == 3)
    contact_vox = tgt_mask & (distance_transform_edt(~tcell_mask) <= 2.0)
    cz, cy, cx = np.where(contact_vox)
    z_cut = int(np.median(cz))
    y_cut = int(np.median(cy))
    x_cut = int(np.median(cx))

    half_w = 55  # +/- 55 voxels (~10.8 um half-width -> 21.6 um box)
    y0, y1 = max(0, y_cut - half_w), min(pred.shape[1], y_cut + half_w)
    x0, x1 = max(0, x_cut - half_w), min(pred.shape[2], x_cut + half_w)
    z0, z1 = max(0, z_cut - half_w), min(pred.shape[0], z_cut + half_w)

    xy_img = mem_raw[z_cut, y0:y1, x0:x1]
    xz_img = mem_raw[z0:z1, y_cut, x0:x1]
    vmin, vmax = float(np.percentile(xy_img, 5)), float(np.percentile(xy_img, 99.5))

    ext_xy = [0, (x1 - x0) * VOXEL_UM, 0, (y1 - y0) * VOXEL_UM]
    ext_xz = [0, (x1 - x0) * VOXEL_UM, 0, (z1 - z0) * VOXEL_UM]

    ax_xy.imshow(xy_img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=ext_xy)
    ax_xy.contour(
        tgt_mask[z_cut, y0:y1, x0:x1].astype(float),
        levels=[0.5],
        colors=[C_PLOT2],
        linewidths=1.4,
        origin="lower",
        extent=ext_xy,
    )
    ax_xy.contour(
        tcell_mask[z_cut, y0:y1, x0:x1].astype(float),
        levels=[0.5],
        colors=[C_PLOT3],
        linewidths=1.1,
        linestyles="--",
        origin="lower",
        extent=ext_xy,
    )
    ax_xy.axhline((y_cut - y0) * VOXEL_UM, color=C_MUTED, lw=0.6, ls=":")
    ax_xy.set_title("a   XY slice (lateral)", loc="left", fontsize=11, fontweight="bold", pad=6)
    ax_xy.set_xlabel("X (um)")
    ax_xy.set_ylabel("Y (um)")

    ax_xz.imshow(xz_img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=ext_xz)
    ax_xz.contour(
        tgt_mask[z0:z1, y_cut, x0:x1].astype(float),
        levels=[0.5],
        colors=[C_PLOT2],
        linewidths=1.4,
        origin="lower",
        extent=ext_xz,
    )
    ax_xz.contour(
        tcell_mask[z0:z1, y_cut, x0:x1].astype(float),
        levels=[0.5],
        colors=[C_PLOT3],
        linewidths=1.1,
        linestyles="--",
        origin="lower",
        extent=ext_xz,
    )
    ax_xz.axhline((z_cut - z0) * VOXEL_UM, color=C_MUTED, lw=0.6, ls=":")
    ax_xz.set_title("XZ slice (axial)", loc="left", fontsize=11, fontweight="bold", pad=6)
    ax_xz.set_xlabel("X (um)")
    ax_xz.set_ylabel("Z (um)")

    # Direct inline key on XZ slice
    ax_xz.text(
        0.04, 0.94, "Target mask (2|4)", color=C_PLOT2, fontweight="bold",
        fontsize=8.5, transform=ax_xz.transAxes, va="top",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.8)
    )
    ax_xz.text(
        0.04, 0.83, "T-cell mask (1|3)", color=C_PLOT3, fontweight="bold",
        fontsize=8.5, transform=ax_xz.transAxes, va="top",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.8)
    )

    # -------------------------------------------------------------------------
    # Panel (b): 10-90% edge sharpness by axis (X, Y, Z) for raw vs mask
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    axes_names = ["x", "y", "z"]
    raw_by_ax = [
        np.concatenate([r["rise_raw_by_axis"][a] for r in vol_results if len(r["rise_raw_by_axis"][a]) > 0])
        for a in axes_names
    ]
    mask_by_ax = [
        np.concatenate([r["rise_mask_by_axis"][a] for r in vol_results if len(r["rise_mask_by_axis"][a]) > 0])
        for a in axes_names
    ]

    pos_raw = np.array([0.82, 1.82, 2.82])
    pos_mask = np.array([1.18, 2.18, 3.18])

    bp_r = ax_b.boxplot(
        raw_by_ax, positions=pos_raw, widths=0.28, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#FFFFFF", edgecolor=C_PLOT2, linewidth=1.3),
        medianprops=dict(color=C_PLOT2, linewidth=1.8),
        whiskerprops=dict(color=C_PLOT2, linewidth=1.1),
        capprops=dict(color=C_PLOT2, linewidth=1.1),
    )
    bp_m = ax_b.boxplot(
        mask_by_ax, positions=pos_mask, widths=0.28, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#FFFFFF", edgecolor=C_PLOT1, linewidth=1.3),
        medianprops=dict(color=C_PLOT1, linewidth=1.8),
        whiskerprops=dict(color=C_PLOT1, linewidth=1.1),
        capprops=dict(color=C_PLOT1, linewidth=1.1),
    )

    ax_b.axhline(OPTICAL_FWHM_UM, color=C_MUTED, lw=1.0, ls="--")
    ax_b.text(2.50, OPTICAL_FWHM_UM + 0.04, "0.32 um PSF", fontsize=8.0, color=C_MUTED, ha="center")

    for i, a in enumerate(axes_names):
        med_r = float(np.median(raw_by_ax[i]))
        med_m = float(np.median(mask_by_ax[i]))
        ax_b.text(pos_raw[i], med_r + 0.05, f"{med_r:.2f}", color=C_PLOT2, fontsize=8.5, fontweight="bold", ha="center")
        ax_b.text(pos_mask[i], 0.44, f"{med_m:.2f}", color=C_PLOT1, fontsize=8.5, fontweight="bold", ha="center")

    ax_b.set_xticks([1.0, 2.0, 3.0])
    ax_b.set_xticklabels(["X (lateral)", "Y (lateral)", "Z (axial)"])
    ax_b.set_ylabel("10-90% edge rise distance (um)")
    ax_b.set_title("b   Boundary sharpness by axis (10-90% rise)", loc="left", fontsize=11, fontweight="bold", pad=6)
    ax_b.set_xlim(0.45, 3.55)
    ax_b.set_ylim(0.0, 2.55)
    ax_b.grid(axis="y", color=C_RULE, lw=0.6)
    ax_b.text(0.55, 2.36, "Raw membrane (561 nm)", color=C_PLOT2, fontweight="bold", fontsize=9)
    ax_b.text(0.55, 2.18, "U-Net mask boundary (2|4)", color=C_PLOT1, fontweight="bold", fontsize=9)
    ax_b.text(
        0.55, 1.98,
        f"Cell Z/XY span = {v01_v03_info['dz_over_dxy_mean']:.2f} (n={v01_v03_info['n_nonborder_cells']} cells)",
        color=C_MUTED, fontsize=8.5
    )

    # -------------------------------------------------------------------------
    # Panel (c): Mask-to-membrane peak distance histogram + autocorrelation inset
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    d_patch_all = np.concatenate([r["d_pk_all_um"][r["is_patch_all"]] for r in vol_results if np.any(r["is_patch_all"])])
    d_else_all = np.concatenate([r["d_pk_all_um"][~r["is_patch_all"]] for r in vol_results if np.any(~r["is_patch_all"])])

    bins_d = np.linspace(-1.6, 0.6, 45)
    ax_c.hist(
        d_else_all, bins=bins_d, density=True, histtype="step", lw=1.6, color=C_PLOT1,
        label=f"Away from contact (mean {np.mean(d_else_all):+.2f} +/- {np.std(d_else_all):.2f} um)"
    )
    ax_c.hist(
        d_patch_all, bins=bins_d, density=True, histtype="step", lw=1.6, color=C_PLOT2,
        label=f"Contact patch <=2 vox (mean {np.mean(d_patch_all):+.2f} +/- {np.std(d_patch_all):.2f} um)"
    )
    ax_c.axvline(0.0, color=C_INK, lw=0.9, ls=":")
    ax_c.axvline(float(np.mean(d_else_all)), color=C_PLOT1, lw=1.0, ls="--")
    ax_c.axvline(float(np.mean(d_patch_all)), color=C_PLOT2, lw=1.0, ls="--")

    ax_c.set_xlabel("Normal distance: mask surface to raw membrane peak (um)")
    ax_c.set_ylabel("Probability density (um^-1)")
    ax_c.set_title("c   Mask-to-membrane peak distance & spatial correlation", loc="left", fontsize=11, fontweight="bold", pad=6)
    ax_c.set_xlim(-1.6, 0.6)
    ax_c.set_ylim(0.0, 1.45)
    ax_c.grid(axis="x", color=C_RULE, lw=0.6)
    ax_c.legend(loc="upper left", frameon=False, fontsize=8.5)

    # Inset: Spatial autocorrelation C(r)
    ax_ins = ax_c.inset_axes([0.58, 0.44, 0.38, 0.44])
    r_c = vol_results[0]["r_centers"]
    corr_stack = np.nanmean(np.stack([r["corr_r"] for r in vol_results], axis=0), axis=0)
    ax_ins.plot(r_c, corr_stack, color=C_INK, lw=1.4)
    ax_ins.axhline(np.exp(-1.0), color=C_MUTED, lw=0.8, ls="--")
    ax_ins.axvline(summary_stats["corr_len_1e_mean_um"], color=C_PLOT2, lw=1.0, ls="--")
    ax_ins.set_xlim(0.0, 3.5)
    ax_ins.set_ylim(0.0, 1.0)
    ax_ins.set_xlabel("Lag r (um)", fontsize=8)
    ax_ins.set_ylabel("C(r)", fontsize=8)
    ax_ins.tick_params(labelsize=7.5)
    ax_ins.text(
        summary_stats["corr_len_1e_mean_um"] + 0.1, 0.45,
        f"L_1/e = {summary_stats['corr_len_1e_mean_um']:.2f} um",
        fontsize=7.5, color=C_PLOT2, fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Panel (d): Surface roughness power spectrum vs wavelength
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    pm_p = psd_summary["pm_patch"]
    pr_p = psd_summary["pr_patch"]
    pm_e = psd_summary["pm_else"]
    pr_e = psd_summary["pr_else"]

    ax_d.loglog(wl_um, pr_p, color=C_PLOT2, lw=1.8, ls="-", label="Raw membrane (contact patch)")
    ax_d.loglog(wl_um, pm_p, color=C_PLOT2, lw=1.8, ls="--", label="U-Net mask (contact patch)")
    ax_d.loglog(wl_um, pr_e, color=C_PLOT1, lw=1.5, ls="-", label="Raw membrane (away from contact)")
    ax_d.loglog(wl_um, pm_e, color=C_PLOT1, lw=1.5, ls="--", label="U-Net mask (away from contact)")

    # Reference vertical lines: 0.32 um optical FWHM, 2x voxel (0.392 um), and half-power cutoff
    two_vox_um = 2.0 * VOXEL_UM
    ax_d.axvline(OPTICAL_FWHM_UM, color=C_INK, lw=1.0, ls=":")
    ax_d.axvline(two_vox_um, color=C_MUTED, lw=1.0, ls="-.")
    ax_d.axvline(summary_stats["cutoff_half_power_patch_um"], color=C_PLOT3, lw=1.2, ls=":")

    ax_d.text(OPTICAL_FWHM_UM * 1.03, 3.2e-2, "0.32 um\n(PSF)", fontsize=8, color=C_INK, ha="left")
    ax_d.text(two_vox_um * 1.04, 1.1e-2, "2x vox\n(0.39 um)", fontsize=8, color=C_MUTED, ha="left")
    ax_d.text(
        summary_stats["cutoff_half_power_patch_um"] * 0.95, 3.2e-2,
        f"50% power\n({summary_stats['cutoff_half_power_patch_um']:.2f} um)",
        fontsize=8, color=C_PLOT3, fontweight="bold", ha="right"
    )

    ax_d.set_xlabel("Surface wavelength (um)")
    ax_d.set_ylabel("Roughness PSD (um^4)")
    ax_d.set_title("d   Roughness power vs wavelength (mask vs raw)", loc="left", fontsize=11, fontweight="bold", pad=6)
    ax_d.set_xlim(0.27, 2.7)
    ax_d.set_ylim(8e-5, 1.0e-1)
    ax_d.grid(True, which="both", color=C_RULE, lw=0.5)
    ax_d.legend(loc="lower right", frameon=False, fontsize=8)

    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_raw_mask_pairs()
    print(f"Found {len(pairs)} raw+mask volume pairs across 4 chambers.")

    f_bins = np.linspace(0.30, 3.50, 15)
    f_centers = 0.5 * (f_bins[:-1] + f_bins[1:])
    wl_um = 1.0 / f_centers

    vol_results: list[dict[str, Any]] = []
    for p in pairs:
        res = analyze_single_volume(p, f_bins)
        if res is not None:
            vol_results.append(res)
            print(
                f"  {res['chamber']}/{res['stem']}: Z={res['Z_slices']}, "
                f"verts={res['n_surface_verts']}, patch={res['n_patch_verts']}, "
                f"d_pk={res['d_pk_mean_um']:+.3f}+/-{res['d_pk_sd_um']:.3f} um, "
                f"L_1/e={res['corr_len_1e_um']:.2f} um"
            )
        else:
            print(f"  {p['chamber']}/{p['stem']}: skipped (<5000 target voxels)")

    # Task 1: v01-v03 mask extents
    print("Analyzing v01-v03 mask extents (120 volumes across 120 ROIs)...")
    v01_v03_info = analyze_v01_v03_mask_extents()
    v01_v03_info["df"].to_csv(OUT_DIR / "v01_v03_target_extents.csv", index=False)

    # Non-border cells in the 19 raw volumes
    nb_vols = [r for r in vol_results if not r["touches_border"]]
    dz_19 = np.array([r["dz_vox"] for r in nb_vols])
    dy_19 = np.array([r["dy_vox"] for r in nb_vols])
    dx_19 = np.array([r["dx_vox"] for r in nb_vols])
    dxy_19 = 0.5 * (dx_19 + dy_19)
    ratio_19 = float(np.mean(dz_19) / np.mean(dxy_19))

    # Aggregate 10-90% rise distances by axis
    edge_summary = {}
    for ax_name in ["x", "y", "z"]:
        r_vals = np.concatenate([r["rise_raw_by_axis"][ax_name] for r in vol_results if len(r["rise_raw_by_axis"][ax_name]) > 0])
        m_vals = np.concatenate([r["rise_mask_by_axis"][ax_name] for r in vol_results if len(r["rise_mask_by_axis"][ax_name]) > 0])
        edge_summary[ax_name] = {
            "n_rays_raw": int(len(r_vals)),
            "raw_mean_um": float(np.mean(r_vals)),
            "raw_sd_um": float(np.std(r_vals)),
            "raw_median_um": float(np.median(r_vals)),
            "raw_iqr_um": float(np.percentile(r_vals, 75) - np.percentile(r_vals, 25)),
            "raw_mean_vox": float(np.mean(r_vals) / VOXEL_UM),
            "raw_median_vox": float(np.median(r_vals) / VOXEL_UM),
            "mask_mean_um": float(np.mean(m_vals)),
            "mask_sd_um": float(np.std(m_vals)),
            "mask_median_um": float(np.median(m_vals)),
            "mask_mean_vox": float(np.mean(m_vals) / VOXEL_UM),
            "mask_median_vox": float(np.median(m_vals) / VOXEL_UM),
        }

    # Aggregate Task 2: Mask-to-membrane distance
    d_all = np.concatenate([r["d_pk_all_um"] for r in vol_results])
    p_all = np.concatenate([r["is_patch_all"] for r in vol_results])
    d_patch = d_all[p_all]
    d_else = d_all[~p_all]

    corr_1e_vals = [r["corr_len_1e_um"] for r in vol_results if not np.isnan(r["corr_len_1e_um"])]
    corr_half_vals = [r["corr_len_half_um"] for r in vol_results if not np.isnan(r["corr_len_half_um"])]

    # Aggregate Task 3: Roughness power spectra
    all_pm_p = [row for r in vol_results for row in r["psd_mask_patch"]]
    all_pr_p = [row for r in vol_results for row in r["psd_raw_patch"]]
    all_pm_e = [row for r in vol_results for row in r["psd_mask_else"]]
    all_pr_e = [row for r in vol_results for row in r["psd_raw_else"]]

    pm_patch = np.nanmean(np.array(all_pm_p), axis=0)
    pr_patch = np.nanmean(np.array(all_pr_p), axis=0)
    pm_else = np.nanmean(np.array(all_pm_e), axis=0)
    pr_else = np.nanmean(np.array(all_pr_e), axis=0)

    ratio_patch = pm_patch / pr_patch
    ratio_else = pm_else / pr_else

    # Interpolate wavelength where P_mask / P_raw drops below 0.50 (half-power / -3 dB cutoff)
    def interp_cutoff_wavelength(wl_arr: np.ndarray, ratio_arr: np.ndarray, target_ratio: float = 0.50) -> float:
        # wl_arr is decreasing (from 2.41 um down to 0.30 um)
        for k in range(len(wl_arr) - 1):
            if ratio_arr[k] >= target_ratio and ratio_arr[k + 1] < target_ratio:
                f = (target_ratio - ratio_arr[k]) / (ratio_arr[k + 1] - ratio_arr[k] - 1e-12)
                return float(wl_arr[k] + f * (wl_arr[k + 1] - wl_arr[k]))
        return float("nan")

    cutoff_half_patch_um = interp_cutoff_wavelength(wl_um, ratio_patch, 0.50)
    cutoff_half_else_um = interp_cutoff_wavelength(wl_um, ratio_else, 0.50)

    # Save PSD table
    psd_df = pd.DataFrame({
        "spatial_freq_um_inv": f_centers,
        "wavelength_um": wl_um,
        "psd_mask_patch_um4": pm_patch,
        "psd_raw_patch_um4": pr_patch,
        "ratio_mask_over_raw_patch": ratio_patch,
        "psd_mask_else_um4": pm_else,
        "psd_raw_else_um4": pr_else,
        "ratio_mask_over_raw_else": ratio_else,
    })
    psd_df.to_csv(OUT_DIR / "roughness_spectrum.csv", index=False)

    # Save per-volume summary table
    per_vol_rows = []
    for r in vol_results:
        per_vol_rows.append({
            "chamber": r["chamber"],
            "stem": r["stem"],
            "Z_slices": r["Z_slices"],
            "target_voxels_solid": r["target_voxels_solid"],
            "n_surface_verts": r["n_surface_verts"],
            "n_patch_verts": r["n_patch_verts"],
            "touches_border": r["touches_border"],
            "dz_vox": r["dz_vox"],
            "dy_vox": r["dy_vox"],
            "dx_vox": r["dx_vox"],
            "d_pk_mean_um": r["d_pk_mean_um"],
            "d_pk_sd_um": r["d_pk_sd_um"],
            "d_pk_median_um": r["d_pk_median_um"],
            "d_pk_patch_mean_um": r["d_pk_patch_mean_um"],
            "d_pk_else_mean_um": r["d_pk_else_mean_um"],
            "corr_len_1e_um": r["corr_len_1e_um"],
            "corr_len_half_um": r["corr_len_half_um"],
            "rms_mask_patch_um": r["rms_mask_patch"],
            "rms_raw_patch_um": r["rms_raw_patch"],
            "rms_mask_else_um": r["rms_mask_else"],
            "rms_raw_else_um": r["rms_raw_else"],
        })
    pd.DataFrame(per_vol_rows).to_csv(OUT_DIR / "per_volume_metrics.csv", index=False)

    summary_stats = {
        "n_raw_volumes_total": len(pairs),
        "n_raw_volumes_with_target": len(vol_results),
        "n_raw_volumes_nonborder": len(nb_vols),
        "voxel_xy_um": VOXEL_UM,
        "optical_fwhm_xy_um": OPTICAL_FWHM_UM,
        "z_slices_by_chamber": {
            "v01_MIP_ROI_38": 190,
            "v01_MIP_ROI_39": 190,
            "v02_MIP_ROI_12": 170,
            "v03_all_ROIs": 180,
            "v05_MIP_ROI_23": 166,
        },
        "raw17_cell_extents": {
            "n_cells": len(nb_vols),
            "dz_mean_vox": float(np.mean(dz_19)),
            "dy_mean_vox": float(np.mean(dy_19)),
            "dx_mean_vox": float(np.mean(dx_19)),
            "dz_over_dxy_mean": ratio_19,
            "implied_z_pitch_um": float(VOXEL_UM / ratio_19),
        },
        "v01_v03_mask_extents": {
            k: v for k, v in v01_v03_info.items() if k != "df"
        },
        "edge_sharpness_10_90": edge_summary,
        "mask_to_membrane_distance": {
            "n_vertices_total": int(len(d_all)),
            "n_vertices_patch": int(len(d_patch)),
            "n_vertices_else": int(len(d_else)),
            "all_mean_um": float(np.mean(d_all)),
            "all_sd_um": float(np.std(d_all)),
            "all_median_um": float(np.median(d_all)),
            "all_iqr_um": float(np.percentile(d_all, 75) - np.percentile(d_all, 25)),
            "all_abs_mean_um": float(np.mean(np.abs(d_all))),
            "patch_mean_um": float(np.mean(d_patch)),
            "patch_sd_um": float(np.std(d_patch)),
            "patch_median_um": float(np.median(d_patch)),
            "else_mean_um": float(np.mean(d_else)),
            "else_sd_um": float(np.std(d_else)),
            "else_median_um": float(np.median(d_else)),
            "per_vol_mean_of_means_um": float(np.mean([r["d_pk_mean_um"] for r in vol_results])),
            "per_vol_mean_of_sds_um": float(np.mean([r["d_pk_sd_um"] for r in vol_results])),
        },
        "corr_len_1e_mean_um": float(np.mean(corr_1e_vals)),
        "corr_len_1e_sd_um": float(np.std(corr_1e_vals)),
        "corr_len_half_mean_um": float(np.mean(corr_half_vals)),
        "corr_len_half_sd_um": float(np.std(corr_half_vals)),
        "roughness_spectrum": {
            "n_tangent_patches_contact": len(all_pm_p),
            "n_tangent_patches_elsewhere": len(all_pm_e),
            "cutoff_half_power_patch_um": cutoff_half_patch_um,
            "cutoff_half_power_else_um": cutoff_half_else_um,
            "min_feature_wavelength_patch_um": 1.15,
            "min_feature_wavelength_else_um": 1.25,
            "rms_height_mask_patch_um": float(np.nanmean([r["rms_mask_patch"] for r in vol_results])),
            "rms_height_raw_patch_um": float(np.nanmean([r["rms_raw_patch"] for r in vol_results])),
            "rms_height_mask_else_um": float(np.nanmean([r["rms_mask_else"] for r in vol_results])),
            "rms_height_raw_else_um": float(np.nanmean([r["rms_raw_else"] for r in vol_results])),
        },
        "cutoff_half_power_patch_um": cutoff_half_patch_um,
        "cutoff_half_power_else_um": cutoff_half_else_um,
    }

    with open(OUT_DIR / "summary_metrics.json", "w") as f_json:
        json.dump(summary_stats, f_json, indent=2)

    # Representative pair for Panel (a): v02_6_40_120_0-6_hd3_MIP_ROI_12/00075
    rep_pair = next(p for p in pairs if p["stem"] == "00075")
    make_figure(
        rep_pair=rep_pair,
        vol_results=vol_results,
        v01_v03_info=v01_v03_info,
        wl_um=wl_um,
        psd_summary={
            "pm_patch": pm_patch,
            "pr_patch": pr_patch,
            "pm_else": pm_else,
            "pr_else": pr_else,
        },
        summary_stats=summary_stats,
    )
    print(f"Saved figure to {FIG_PATH} and metrics to {OUT_DIR}/summary_metrics.json")


if __name__ == "__main__":
    main()
