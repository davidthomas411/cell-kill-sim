#!/usr/bin/env python3
"""
Does the dent predict the kill? — End-to-End Reproducible Pipeline (Steps 1–6)

Implements the 6-step protocol specified in the task file:
  Step 1. Inventory all 502 chambers across the 13 experiments in ScienceDB 10.57760/sciencedb.19235
          (`data/scidb_19235_all_urls.txt`) and match to `Source_Data_Wang2024_NatCommun.xlsx`.
  Step 2. Define 3D target-membrane indentation ("dent", um) via `measure_dent_3d`, seeding target
          instances on target interior (`label == 2`, preventing touching-doublet merges as in
          `step2_instance_seg.m`), excluding internal false-positive specks, fitting an undeformed
          ellipsoid (with spherical fallback when unstable) to the non-contact target surface, and
          measuring the inward depression of the contact surface. Logs every rejected frame and why.
  Step 3. Validate `measure_dent_3d` on 35 synthetic 3D cell pairs with known dents at 0.320 um/voxel
          (passing through the exact same `measure_dent_3d` function) and hand-check 20 real frames
          spanning small and large dents with cross-sectional visual overlays.
  Step 4. Per-pair summary across all chambers: maximum dent (`dent_max_um`), dent at peak contact
          area (`dent_at_peak_ca_um`), time from first contact to maximum dent (`t_to_max_dent_min`),
          3D contact area (`ca_3d_max_um2`), 405-nm SYTOX Blue dye signal (`dye_405_delta_bg`), and
          alignment with the 419 published pairs (`Source_Data_Wang2024_NatCommun.xlsx`).
  Step 5. Out-of-sample comparison: fit on High-Killing Donor Groups 1–5, freeze parameters, and
          score on Held-Out Low-Killing Donor Groups 6–10 and Dasatinib Groups 8*, 9*, 10* with
          500-iteration bootstrap 95% confidence intervals over pairs:
            - Contact area alone (`CA / (CA + K)` and `1 - exp(-a * CA)`)
            - Dent + Contact area (`1 - exp(-(a * CA + b * dent))`)
            - Dent alone (`1 - exp(-b * dent)`)
  Step 6. Generate House-Style figure (`figures/dent_vs_kill_by_donor_group.png` and
          `figures/dent_step3_20_real_frames_handcheck.png`) and summary tables.

Constants Inventory (Rule 5 — every constant labeled as measured, cited, or chosen):
  - DX_UM = 0.320 um/voxel          [cited: Wang et al., Nat Commun 2024 / Sci Data 2026 HBOPM voxel pitch]
  - CA_VOXEL_FACTOR_UM2 = 0.0385    [cited: github.com/feilab-hust/CAR-T-analysis/main/analysis/summary/summary.m line 16]
  - VISIT_INTERVAL_MIN = 10.0       [cited: Wang et al., Nat Commun 2024 Fig. 5a/S9 10-min visit cadence, 6 sweeps/visit]
  - SWEEPS_PER_VISIT = 6            [cited: configure.m / step2_instance_seg.m `sample_num = 6`]
  - DYE_THRESHOLD_RAW = 398.0       [cited: step2_instance_seg.m line 35 `D > 398` after Gaussian smoothing]
  - MIN_TARGET_INT_VOXELS = 250     [chosen: minimum 3D target interior volume = 8.2 um^3 to reject unclosed/debris masks]
  - MIN_IMMUNE_VOXELS = 150         [chosen: minimum 3D CAR-T volume = 4.9 um^3 to confirm immune cell presence]
  - TARGET_REQ_BOUNDS_UM = (2.8,8.5)[chosen: single-cell Nalm6 equivalent radius bounds to reject merged doublets/fragments]
  - MIN_NONCONTACT_SURF_VOX = 80    [chosen: minimum non-contact surface voxels (~8.2 um^2) for stable shape fit]
  - MIN_HALFSPACE_COVERAGE = 5      [chosen: non-contact surface must span >=5 of 6 coordinate half-spaces around centroid]
  - RADIUS_CONSISTENCY = (0.75,1.25)[chosen: fitted R0 / R_eq bounds to reject open membrane arcs]
  - MIN_CONTACT_SURF_VOX = 5        [chosen: minimum contact surface voxels (~0.5 um^2) to define non-zero contact]
  - MAX_ELLIPSOID_ASPECT = 1.50     [chosen: stability bound above which ellipsoid fit falls back to spherical fit]
  - DENT_PERCENTILE = 97.0          [chosen: high-percentile inward distance across contact patch to read maximum dent robustly]
"""

import io
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy import stats
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
)
from scipy.optimize import least_squares

# ==============================================================================
# CONSTANTS (Rule 5: every constant labeled measured, cited, or chosen)
# ==============================================================================
DX_UM = 0.320                    # [cited] Wang et al. 2024 isotropic voxel size (um)
CA_VOXEL_FACTOR_UM2 = 0.0385     # [cited] feilab_CART_analysis summary.m surface area factor (um^2/voxel)
VISIT_INTERVAL_MIN = 10.0        # [cited] Wang et al. 2024 time-lapse visit interval (min)
SWEEPS_PER_VISIT = 6             # [cited] step2_instance_seg.m sample_num = 6 sweeps per visit
DYE_THRESHOLD_RAW = 398.0        # [cited] step2_instance_seg.m 405nm dead-cell dye threshold
MIN_TARGET_INT_VOXELS = 250      # [chosen] 8.2 um^3 minimum closed target interior volume
MIN_IMMUNE_VOXELS = 150          # [chosen] 4.9 um^3 minimum CAR-T volume
TARGET_REQ_MIN_UM = 2.8          # [chosen] minimum single-cell equivalent radius (um)
TARGET_REQ_MAX_UM = 10.8         # [chosen] maximum single-cell equivalent radius (um) including membrane halo
MIN_NONCONTACT_SURF_VOX = 80     # [chosen] minimum non-contact surface voxels for fit stability
MIN_HALFSPACE_COVERAGE = 5       # [chosen] minimum coordinate half-spaces (out of 6) covered by non-contact surface
MIN_CONTACT_SURF_VOX = 5         # [chosen] minimum contact voxels for non-zero contact patch
MAX_ELLIPSOID_ASPECT = 1.50      # [chosen] maximum semi-axis ratio before sphere fallback
DENT_PERCENTILE = 97.0           # [chosen] percentile of inward surface deviation for robust max dent

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CACHE_DIR = DATA_DIR / "scidb_dent_cache"
RAW_CACHE_DIR = CACHE_DIR / "files"
CHAMBER_DIR = DATA_DIR / "scidb_19235_chambers"
URLS_FILE = DATA_DIR / "scidb_19235_all_urls.txt"
XLSX_FILE = ROOT / "external" / "feilab_lightsheet_data" / "Source_Data_Wang2024_NatCommun.xlsx"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

V_TO_GROUP = {
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

GROUP_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "8*", "9*", "10*"]
TRAIN_GROUPS = ["1", "2", "3", "4", "5"]
LOW_GROUPS = ["6", "7", "8", "9", "10"]
DAS_GROUPS = ["8*", "9*", "10*"]


# ==============================================================================
# STEP 2: UNIFIED 3D DENT MEASUREMENT FUNCTION (Used on both Synthetic & Real)
# ==============================================================================
def measure_dent_3d(seg: np.ndarray, dx_um: float = DX_UM, return_debug: bool = False) -> dict:
    """
    Measure the 3D indentation depth ("dent", in um) of the target cell membrane at the
    immune-cell contact patch, passing both synthetic and real 3D label volumes through
    the exact same code (Rule 4).

    Labels in `seg` (uint8):
      1 = CAR-T body, 2 = Target interior, 3 = CAR-T dense F-actin, 4 = Target membrane.
    """
    b_any = (seg == 2) | (seg == 4)
    t_any = (seg == 1) | (seg == 3)

    n_b_total = int(b_any.sum())
    n_t_total = int(t_any.sum())
    if n_b_total < MIN_TARGET_INT_VOXELS and n_t_total < MIN_IMMUNE_VOXELS:
        return {"accepted": False, "reject_reason": "both_cells_missing", "dent_um": np.nan, "ca_um2": np.nan}
    if n_b_total < MIN_TARGET_INT_VOXELS:
        return {"accepted": False, "reject_reason": "target_cell_missing_or_too_small", "dent_um": np.nan, "ca_um2": np.nan}
    if n_t_total < MIN_IMMUNE_VOXELS:
        return {"accepted": False, "reject_reason": "immune_cell_missing_or_too_small", "dent_um": np.nan, "ca_um2": np.nan}

    # 2x downsampled Voronoi instance segmentation of target (matching step2_instance_seg.m lines 150-185)
    seg_ds = seg[::2, ::2, ::2]
    nuc_ds = seg_ds == 2
    mem_ds = seg_ds == 4
    t_ds = (seg_ds == 1) | (seg_ds == 3)

    lbl_nuc, n_nuc = label(nuc_ds)
    valid_k = []
    if n_nuc > 0:
        sizes_n = np.bincount(lbl_nuc.ravel())
        for k in range(1, n_nuc + 1):
            if sizes_n[k] >= 35:  # >= 280 full-res interior voxels
                valid_k.append(k)

    if len(valid_k) == 0:
        return {"accepted": False, "reject_reason": "target_interior_missing_or_unclosed", "dent_um": np.nan, "ca_um2": np.nan}
    elif len(valid_k) == 1:
        k = valid_k[0]
        n_mask_ds = lbl_nuc == k
        m_mask_ds = mem_ds & (maximum_filter(n_mask_ds, size=21) > 0)
        b_inst_ds = binary_fill_holes(n_mask_ds | m_mask_ds)
    else:
        # Voronoi partition membrane between multiple target nuclei so touching doublets never merge
        seed_map = np.zeros_like(lbl_nuc)
        for k in valid_k:
            seed_map[lbl_nuc == k] = k
        dist_ds, inds_ds = distance_transform_edt(seed_map == 0, return_indices=True)
        nearest_k = seed_map[tuple(inds_ds)]
        t_near_ds = maximum_filter(t_ds, size=5) > 0
        best_k, best_touch, best_sz = valid_k[0], -1, 0
        for k in valid_k:
            inst_k = (seed_map == k) | (mem_ds & (nearest_k == k) & (dist_ds <= 12))
            sz_k = int(inst_k.sum())
            touch_k = int((inst_k & t_near_ds).sum())
            if touch_k > best_touch or (touch_k == best_touch and sz_k > best_sz):
                best_touch, best_sz, best_k = touch_k, sz_k, k
        b_inst_ds = binary_fill_holes((seed_map == best_k) | (mem_ds & (nearest_k == best_k) & (dist_ds <= 12)))

    zyx_ds = np.argwhere(b_inst_ds) * 2
    zmin, ymin, xmin = np.maximum(0, zyx_ds.min(axis=0) - 4)
    zmax, ymax, xmax = np.minimum(seg.shape, zyx_ds.max(axis=0) + 6)
    sub_seg = seg[zmin:zmax, ymin:ymax, xmin:xmax]

    # Full-resolution Voronoi instance mask inside tight bounding box
    lbl_sub_nuc, n_sub_nuc = label(sub_seg == 2)
    if n_sub_nuc == 0:
        return {"accepted": False, "reject_reason": "target_interior_bbox_empty", "dent_um": np.nan, "ca_um2": np.nan}
    sizes_sub_nuc = np.bincount(lbl_sub_nuc.ravel())
    sizes_sub_nuc[0] = 0
    valid_sub_k = [k for k in range(1, n_sub_nuc + 1) if sizes_sub_nuc[k] >= MIN_TARGET_INT_VOXELS]
    if not valid_sub_k:
        return {"accepted": False, "reject_reason": "target_interior_below_min_volume", "dent_um": np.nan, "ca_um2": np.nan}

    if len(valid_sub_k) == 1:
        k_main = valid_sub_k[0]
        b_int = binary_fill_holes(lbl_sub_nuc == k_main)
        mem_attached = (sub_seg == 4)
    else:
        seed_sub = np.zeros_like(lbl_sub_nuc)
        for k in valid_sub_k:
            seed_sub[lbl_sub_nuc == k] = k
        dist_sub, inds_sub = distance_transform_edt(seed_sub == 0, return_indices=True)
        nearest_sub = seed_sub[tuple(inds_sub)]
        # Pick component matching centroid of b_inst_ds
        c_ds_local = zyx_ds.mean(axis=0) - np.array([zmin, ymin, xmin])
        best_sub_k = min(
            valid_sub_k,
            key=lambda k: np.linalg.norm(np.argwhere(seed_sub == k).mean(axis=0) - c_ds_local),
        )
        b_int = binary_fill_holes(seed_sub == best_sub_k)
        mem_attached = (sub_seg == 4) & (nearest_sub == best_sub_k) & (dist_sub <= 24)

    # Keep largest connected component of (b_int | mem_attached), close small membrane gaps, and fill internal holes
    lbl_b_full, _ = label(b_int | mem_attached)
    sizes_bf = np.bincount(lbl_b_full.ravel())
    sizes_bf[0] = 0
    b_raw = lbl_b_full == int(np.argmax(sizes_bf))
    b_filled = binary_fill_holes(binary_closing(b_raw, iterations=2))
    mem_attached = mem_attached & b_raw

    vol_um3 = float(b_filled.sum() * (dx_um ** 3))
    r_eq_um = float((3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    if r_eq_um < TARGET_REQ_MIN_UM or r_eq_um > TARGET_REQ_MAX_UM:
        return {"accepted": False, "reject_reason": "target_radius_out_of_single_cell_bounds", "dent_um": np.nan, "ca_um2": np.nan}

    zyx_all = np.argwhere(b_filled) * dx_um
    c_init = zyx_all.mean(axis=0)

    # Isolate genuine EXTERNAL immune cell connected components:
    # Exclude voxels deep inside b_filled, and require each immune component's center of mass
    # to lie outside the target cell (||c_T - c_init|| >= 0.85 * r_eq_um) so U-Net internal
    # misclassification specks inside the target nucleus/cytoplasm are never treated as T cells.
    b_core = binary_erosion(b_filled, iterations=2)
    t_ext_raw = ((sub_seg == 1) | (sub_seg == 3)) & (~b_core)
    lbl_t, n_t = label(t_ext_raw)
    sub_t = np.zeros_like(b_filled, dtype=bool)
    t_centroids = []
    if n_t > 0:
        sizes_t = np.bincount(lbl_t.ravel())
        sizes_t[0] = 0
        for kt in range(1, n_t + 1):
            if sizes_t[kt] >= MIN_IMMUNE_VOXELS:
                pts_kt = np.argwhere(lbl_t == kt) * dx_um
                c_kt = pts_kt.mean(axis=0)
                if np.linalg.norm(c_kt - c_init) >= 0.85 * r_eq_um:
                    sub_t |= (lbl_t == kt)
                    t_centroids.append(c_kt)

    surf_1vox = b_filled & (~binary_erosion(b_filled, iterations=1))
    t_near = maximum_filter(sub_t, size=3) > 0
    t_far = maximum_filter(sub_t, size=7) > 0
    contact_mem_vox = int((mem_attached & t_near).sum())
    ca_um2 = float(contact_mem_vox * CA_VOXEL_FACTOR_UM2)

    contact_surf = surf_1vox & t_near
    non_contact_surf = surf_1vox & (~t_far)

    n_cs = int(contact_surf.sum())
    n_ncs = int(non_contact_surf.sum())
    if n_ncs < MIN_NONCONTACT_SURF_VOX:
        return {"accepted": False, "reject_reason": "insufficient_noncontact_surface", "dent_um": np.nan, "ca_um2": ca_um2}

    zyx_nc = np.argwhere(non_contact_surf) * dx_um

    rel_nc = zyx_nc - c_init
    halfspaces = (
        int((rel_nc[:, 0] > 0).any()) + int((rel_nc[:, 0] < 0).any())
        + int((rel_nc[:, 1] > 0).any()) + int((rel_nc[:, 1] < 0).any())
        + int((rel_nc[:, 2] > 0).any()) + int((rel_nc[:, 2] < 0).any())
    )
    if halfspaces < MIN_HALFSPACE_COVERAGE:
        return {"accepted": False, "reject_reason": "noncontact_surface_incomplete_arc", "dent_um": np.nan, "ca_um2": ca_um2}

    pts_nc = zyx_nc[np.linspace(0, len(zyx_nc) - 1, min(400, len(zyx_nc))).astype(int)]
    r_init = float(np.clip(np.median(np.linalg.norm(pts_nc - c_init, axis=1)), 2.5, 11.5))

    def sph_res(p):
        c, R = p[:3], p[3]
        d = np.linalg.norm(pts_nc - c, axis=1) - R
        reg = 0.15 * (c - c_init)
        return np.concatenate([d, reg])

    sol_sph = least_squares(
        sph_res,
        x0=np.append(c_init, r_init),
        bounds=(np.append(c_init - 2.5, 2.5), np.append(c_init + 2.5, 11.5)),
        loss="soft_l1",
        max_nfev=30,
    )
    c_sph = sol_sph.x[:3]
    R_sph = float(sol_sph.x[3])

    if R_sph / r_eq_um > 1.30 or R_sph / r_eq_um < 0.72:
        return {"accepted": False, "reject_reason": "fitted_radius_inconsistent_with_volume", "dent_um": np.nan, "ca_um2": ca_um2}

    if n_cs < MIN_CONTACT_SURF_VOX:
        res = {
            "accepted": True,
            "reject_reason": "none",
            "dent_um": 0.0,
            "dent_median_um": 0.0,
            "ca_um2": 0.0,
            "R0_um": round(R_sph, 4),
            "aspect_ratio": 1.0,
            "fit_type": "no_contact",
            "n_contact_vox": n_cs,
            "n_noncontact_vox": n_ncs,
        }
        if return_debug:
            res["sub_seg"] = sub_seg
            res["b_filled"] = b_filled
            res["sub_t"] = sub_t
            res["c_fit_zyx"] = c_sph / dx_um
            res["a_fit_zyx"] = np.array([R_sph, R_sph, R_sph]) / dx_um
        return res

    def ell_res(p):
        c, a = p[:3], p[3:6]
        r_norm = np.linalg.norm((pts_nc - c) / a, axis=1)
        d = (r_norm - 1.0) * np.mean(a)
        return np.concatenate([d, 0.25 * (c - c_sph), 0.60 * (a - R_sph)])

    sol_ell = least_squares(
        ell_res,
        x0=np.concatenate([c_sph, [R_sph, R_sph, R_sph]]),
        bounds=(np.concatenate([c_sph - 1.5, [0.75 * R_sph] * 3]), np.concatenate([c_sph + 1.5, [1.30 * R_sph] * 3])),
        loss="soft_l1",
        max_nfev=30,
    )
    c_ell, a_ell = sol_ell.x[:3], sol_ell.x[3:6]
    aspect_ratio = float(max(a_ell) / max(1e-3, min(a_ell)))

    zyx_c = np.argwhere(contact_surf) * dx_um
    if sol_ell.success and aspect_ratio <= MAX_ELLIPSOID_ASPECT:
        fit_type = "ellipsoid"
        v_c = zyx_c - c_ell
        r_c = np.linalg.norm(v_c, axis=1)
        denom = np.linalg.norm(v_c / a_ell, axis=1)
        r_fit_c = np.where(denom > 1e-6, r_c / denom, R_sph)
        inward = np.where(r_c >= 0.45 * R_sph, r_fit_c - r_c, 0.0)
        c_used, a_used = c_ell, a_ell
    else:
        fit_type = "sphere_fallback"
        r_c = np.linalg.norm(zyx_c - c_sph, axis=1)
        inward = np.where(r_c >= 0.45 * R_sph, R_sph - r_c, 0.0)
        c_used, a_used = c_sph, np.array([R_sph, R_sph, R_sph])

    dent_max = float(max(0.0, np.percentile(inward, DENT_PERCENTILE)))
    dent_med = float(max(0.0, np.median(inward)))

    res = {
        "accepted": True,
        "reject_reason": "none",
        "dent_um": round(dent_max, 4),
        "dent_median_um": round(dent_med, 4),
        "ca_um2": round(ca_um2, 4),
        "R0_um": round(R_sph, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "fit_type": fit_type,
        "n_contact_vox": n_cs,
        "n_noncontact_vox": n_ncs,
    }
    if return_debug:
        res["sub_seg"] = sub_seg
        res["b_filled"] = b_filled
        res["sub_t"] = sub_t
        res["c_fit_zyx"] = c_used / dx_um
        res["a_fit_zyx"] = a_used / dx_um
    return res


def make_synthetic_pair(
    R_target_um: float = 6.5,
    aspect: tuple = (1.0, 1.0, 1.0),
    dent_true_um: float = 1.5,
    R_immune_um: float = 5.5,
    axis_idx: int = 2,
    shape: tuple = (120, 120, 120),
) -> np.ndarray:
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = [shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0]
    center[axis_idx] -= 10.0
    cz, cy, cx = center

    dz = (zz - cz) * DX_UM / aspect[0]
    dy = (yy - cy) * DX_UM / aspect[1]
    dx = (xx - cx) * DX_UM / aspect[2]
    r_ell = np.sqrt(dz**2 + dy**2 + dx**2)
    target_undeformed = r_ell <= R_target_um

    offset_um = (R_target_um * aspect[axis_idx] - dent_true_um) + R_immune_um
    imm_c = [cz * DX_UM, cy * DX_UM, cx * DX_UM]
    imm_c[axis_idx] += offset_um
    r_imm = np.sqrt(
        (zz * DX_UM - imm_c[0])**2
        + (yy * DX_UM - imm_c[1])**2
        + (xx * DX_UM - imm_c[2])**2
    )
    immune_body = r_imm <= R_immune_um

    target_body = target_undeformed & (~immune_body)
    target_eroded = binary_erosion(target_body, iterations=2)
    target_mem = target_body & (~target_eroded)
    target_int = target_eroded

    immune_eroded = binary_erosion(immune_body, iterations=2)
    immune_actin = immune_body & (~immune_eroded)

    seg = np.zeros(shape, dtype=np.uint8)
    seg[immune_body] = 1
    seg[immune_actin] = 3
    seg[target_int] = 2
    seg[target_mem] = 4
    return seg


# ==============================================================================
# STEP 1: COMPLETE DEPOSIT INVENTORY
# ==============================================================================
def run_step1_inventory():
    print("=== STEP 1: Building Complete ScienceDB 19235 Deposit Inventory ===")
    chambers = defaultdict(
        lambda: {
            "pred_h5": {},
            "raw_h5": {},
            "mipxy_405": {},
            "mipxz_405": {},
        }
    )

    with open(URLS_FILE, "r") as f:
        for line in f:
            if "path=" not in line:
                continue
            url = line.strip()
            p = url.split("path=")[1].split("&")[0]
            parts = p.strip("/").split("/")
            if len(parts) < 4:
                continue
            v_dir, roi_dir = parts[1], parts[2]
            if not roi_dir.startswith("MIP_ROI_"):
                continue
            sub, fname = parts[3], parts[-1]
            c = chambers[(v_dir, roi_dir)]
            if sub == "ROIauto":
                if fname.endswith("_predictions.h5"):
                    stem = fname.replace("_predictions.h5", "")
                    c["pred_h5"][stem] = url
                elif fname.endswith(".h5"):
                    stem = fname.replace(".h5", "")
                    c["raw_h5"][stem] = url
            elif sub == "MIPxy_405" and fname.endswith(".tif"):
                stem = fname.replace(".tif", "")
                c["mipxy_405"][stem] = url
            elif sub == "MIPxz_405" and fname.endswith(".tif"):
                stem = fname.replace(".tif", "")
                c["mipxz_405"][stem] = url

    wb = openpyxl.load_workbook(XLSX_FILE, data_only=True)
    ws_ca = wb["Figure 5c"]
    ws_adc = wb["Figure 5d"]
    ws_arfs = wb["Figure 5e"]
    ws_mpa = wb["Figure 5f"]
    ws_kill = wb["Figure 5g"]

    grp_meta = {}
    xlsx_cells_by_group = defaultdict(list)
    for col in range(2, 15):
        gid = str(ws_kill.cell(4, col).value)
        n_cells = int(ws_kill.cell(6, col).value)
        n_killed = int(ws_kill.cell(7, col).value)
        kill_rate = float(ws_kill.cell(8, col).value)
        grp_meta[gid] = {
            "n_cells_xlsx": n_cells,
            "n_killed_xlsx": n_killed,
            "kill_rate_xlsx": kill_rate,
        }
        for r in range(6, 46):
            ca_v = ws_ca.cell(r, col).value
            if isinstance(ca_v, (int, float)):
                adc_v = ws_adc.cell(r, col).value
                arfs_v = ws_arfs.cell(r, col).value
                mpa_v = ws_mpa.cell(r, col).value
                xlsx_cells_by_group[gid].append({
                    "group_id": gid,
                    "cell_index": int(ws_ca.cell(r, 1).value),
                    "CA_xlsx_um2": float(ca_v),
                    "ADC_xlsx": float(adc_v) if isinstance(adc_v, (int, float)) else 0.0,
                    "ARFS_xlsx_nm_s": float(arfs_v) if isinstance(arfs_v, (int, float)) else 0.0,
                    "MPA_xlsx_deg": float(mpa_v) if isinstance(mpa_v, (int, float)) else np.nan,
                    "group_kill_rate": kill_rate,
                })

    chamber_rows = []
    for (v_dir, roi_dir), c in sorted(
        chambers.items(), key=lambda x: (x[0][0], int(x[0][1].split("_")[-1]))
    ):
        gid, cond = V_TO_GROUP[v_dir]
        roi_idx = int(roi_dir.split("_")[-1])
        pred_stems = set(c["pred_h5"].keys())
        raw_stems = set(c["raw_h5"].keys())
        c405_stems = set(c["mipxy_405"].keys())
        chamber_rows.append({
            "experiment_dir": v_dir,
            "group_id": gid,
            "condition": cond,
            "roi_dir": roi_dir,
            "roi_index": roi_idx,
            "n_outlined_frames": len(pred_stems),
            "n_raw_frames": len(raw_stems),
            "n_raw_only_frames": len(raw_stems - pred_stems),
            "n_outlined_with_405_dye": len(pred_stems & c405_stems),
            "has_3d_outlines": len(pred_stems) > 0,
            "has_blue_dye": len(c405_stems) > 0,
        })

    df_ch = pd.DataFrame(chamber_rows)
    df_ch.to_csv(DATA_DIR / "dent_step1_chamber_inventory.csv", index=False)

    grp_rows = []
    for v_dir in sorted(V_TO_GROUP.keys()):
        gid, cond = V_TO_GROUP[v_dir]
        sub = df_ch[df_ch["experiment_dir"] == v_dir]
        gm = grp_meta[gid]
        grp_rows.append({
            "experiment_dir": v_dir,
            "group_id": gid,
            "condition": cond,
            "n_chambers_in_deposit": len(sub),
            "n_chambers_with_3d_outlines": int(sub["has_3d_outlines"].sum()),
            "n_chambers_raw_only": int((~sub["has_3d_outlines"]).sum()),
            "n_chambers_with_405_dye": int(sub["has_blue_dye"].sum()),
            "total_outlined_frames": int(sub["n_outlined_frames"].sum()),
            "total_raw_only_frames": int(sub["n_raw_only_frames"].sum()),
            "n_pairs_in_published_xlsx": gm["n_cells_xlsx"],
            "n_chambers_excluded_multi_or_zero_tcell": len(sub) - gm["n_cells_xlsx"],
            "n_killed_published": gm["n_killed_xlsx"],
            "kill_rate_published": gm["kill_rate_xlsx"],
        })
    df_grp_inv = pd.DataFrame(grp_rows)
    df_grp_inv.to_csv(DATA_DIR / "dent_step1_group_inventory.csv", index=False)
    print(f"   Saved dent_step1_chamber_inventory.csv ({len(df_ch)} chambers) and dent_step1_group_inventory.csv (13 groups).")
    return chambers, grp_meta, xlsx_cells_by_group, df_ch, df_grp_inv


# ==============================================================================
# STEP 3: SYNTHETIC VALIDATION & 20-FRAME REAL VISUAL HAND-CHECK
# ==============================================================================
def run_step3_validation():
    print("=== STEP 3: Validating Dent Measurement on Synthetic Pairs & 20 Real Frames ===")
    synth_configs = [
        {"label": "Sphere R=6.5, +X indent", "R_t": 6.5, "aspect": (1.00, 1.00, 1.00), "R_i": 5.5, "axis": 2},
        {"label": "Prolate (1.12,0.92,1.0), +X indent", "R_t": 6.5, "aspect": (1.12, 0.92, 1.00), "R_i": 5.5, "axis": 2},
        {"label": "Oblate (0.90,1.08,1.05), +Y indent", "R_t": 6.2, "aspect": (0.90, 1.08, 1.05), "R_i": 5.2, "axis": 1},
        {"label": "Small Target R=5.2, +Z indent", "R_t": 5.2, "aspect": (1.04, 0.96, 1.00), "R_i": 4.8, "axis": 0},
        {"label": "Large Target R=7.5, +X indent", "R_t": 7.5, "aspect": (1.06, 1.02, 0.94), "R_i": 6.0, "axis": 2},
    ]
    true_dents = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    synth_rows = []
    for cfg in synth_configs:
        for d_true in true_dents:
            seg = make_synthetic_pair(
                R_target_um=cfg["R_t"],
                aspect=cfg["aspect"],
                dent_true_um=d_true,
                R_immune_um=cfg["R_i"],
                axis_idx=cfg["axis"],
            )
            out = measure_dent_3d(seg, dx_um=DX_UM)
            err = float(out["dent_um"] - d_true)
            synth_rows.append({
                "geometry": cfg["label"],
                "R_target_true_um": cfg["R_t"],
                "dent_true_um": d_true,
                "dent_measured_um": out["dent_um"],
                "error_um": round(err, 4),
                "abs_error_um": round(abs(err), 4),
                "abs_error_voxels": round(abs(err) / DX_UM, 3),
                "ca_measured_um2": out["ca_um2"],
                "R0_fitted_um": out["R0_um"],
                "fit_type": out["fit_type"],
            })

    df_synth = pd.DataFrame(synth_rows)
    df_synth.to_csv(DATA_DIR / "dent_step3_synthetic_validation.csv", index=False)
    mae_um = float(df_synth["abs_error_um"].mean())
    max_err_um = float(df_synth["abs_error_um"].max())
    print(f"   Synthetic validation (35 pairs): MAE = {mae_um:.3f} um ({mae_um/DX_UM:.2f} vox), Max error = {max_err_um:.3f} um.")

    cached_h5s = sorted(CHAMBER_DIR.glob("*/*_predictions.h5"))
    real_candidates = []
    for hp in cached_h5s:
        if int(hp.name.split("_")[0]) % 6 == 0:
            continue
        with h5py.File(hp, "r") as hf:
            seg = hf["predictions"][:]
        res = measure_dent_3d(seg, dx_um=DX_UM, return_debug=True)
        if res["accepted"] and res["ca_um2"] > 0.0:
            res["h5_path"] = str(hp)
            res["chamber"] = hp.parent.name
            res["frame_id"] = hp.name.replace("_predictions.h5", "")
            real_candidates.append(res)

    real_candidates.sort(key=lambda r: r["dent_um"])
    idx_20 = np.linspace(0, len(real_candidates) - 1, min(20, len(real_candidates))).astype(int)
    selected_20 = [real_candidates[i] for i in idx_20]

    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    handcheck_rows = []
    for k, (ax, item) in enumerate(zip(axes.flat, selected_20)):
        sub_seg = item["sub_seg"]
        b_filled = item["b_filled"]
        sub_t = item["sub_t"]
        cz, cy, cx = item["c_fit_zyx"]
        az, ay, ax_r = item["a_fit_zyx"]
        z_slice = int(np.clip(round(cz), 0, sub_seg.shape[0] - 1))

        canvas = np.ones((sub_seg.shape[1], sub_seg.shape[2], 3), dtype=np.float32) * 0.97
        t_sl = sub_t[z_slice]
        b_sl = b_filled[z_slice]
        m_sl = (sub_seg[z_slice] == 4) & b_sl

        canvas[b_sl] = [0.85, 0.91, 0.96]
        canvas[t_sl] = [0.96, 0.84, 0.76]
        canvas[m_sl] = [0.12, 0.31, 0.47]

        ax.imshow(canvas, origin="lower")
        dz_norm = (z_slice - cz) / max(1e-3, az)
        if abs(dz_norm) < 0.98:
            scale_xy = np.sqrt(1.0 - dz_norm**2)
            theta = np.linspace(0, 2 * np.pi, 120)
            ex = cx + ax_r * scale_xy * np.cos(theta)
            ey = cy + ay * scale_xy * np.sin(theta)
            ax.plot(ex, ey, color="#111111", linestyle="--", linewidth=1.2)

        ax.set_title(
            f"#{k+1} dent={item['dent_um']:.2f} µm | CA={item['ca_um2']:.0f} µm²",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
            pad=3,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        handcheck_rows.append({
            "check_index": k + 1,
            "chamber": item["chamber"],
            "frame_id": item["frame_id"],
            "dent_um": item["dent_um"],
            "dent_median_um": item["dent_median_um"],
            "ca_um2": item["ca_um2"],
            "R0_um": item["R0_um"],
            "aspect_ratio": item["aspect_ratio"],
            "fit_type": item["fit_type"],
        })

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dent_step3_20_real_frames_handcheck.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(handcheck_rows).to_csv(DATA_DIR / "dent_step3_20_real_frames_handcheck.csv", index=False)
    print("   Saved dent_step3_20_real_frames_handcheck.png and CSV.")
    return df_synth, pd.DataFrame(handcheck_rows)


# ==============================================================================
# STEP 2 & STEP 4: MULTI-FRAME DENT EXTRACTION ACROSS ALL 502 CHAMBERS
# ==============================================================================
def run_step2_and_step4_chamber_series(chambers: dict, xlsx_cells_by_group: dict):
    print("=== STEP 2 & 4: Measuring 3D Dents & 405nm Dye Across All 502 Chambers ===")
    cache_json_path = CACHE_DIR / "all_502_chambers_frame_measurements_v4.json"
    if cache_json_path.exists():
        with open(cache_json_path, "r") as f:
            cached_results = json.load(f)
    else:
        cached_results = {}

    tasks = []
    for (v_dir, roi_dir), c in sorted(
        chambers.items(), key=lambda x: (x[0][0], int(x[0][1].split("_")[-1]))
    ):
        key = f"{v_dir}__{roi_dir}"
        if key in cached_results:
            prev_frames = cached_results[key].get("frames", [])
            has_http_err = any(str(fr.get("reject_reason", "")).startswith("io_or_h5_error") for fr in prev_frames)
            if len(prev_frames) >= 3 and not has_http_err:
                continue
        stems_all = sorted(c["pred_h5"].keys(), key=lambda s: int(s))
        valid_stems = [s for idx, s in enumerate(stems_all, start=1) if (idx % SWEEPS_PER_VISIT) != 0]
        if not valid_stems:
            valid_stems = stems_all
        n_v = len(valid_stems)
        pick_indices = sorted(set([
            max(0, int(0.20 * (n_v - 1))),
            max(0, int(0.55 * (n_v - 1))),
            max(0, int(0.85 * (n_v - 1))),
        ]))
        picked = [(idx_pos, valid_stems[idx_pos], c["pred_h5"][valid_stems[idx_pos]]) for idx_pos in pick_indices]
        last_405_stem = sorted(c["mipxy_405"].keys(), key=lambda s: int(s))[-1] if c["mipxy_405"] else None
        last_405_url = c["mipxy_405"].get(last_405_stem) if last_405_stem else None
        tasks.append((key, v_dir, roi_dir, len(stems_all), picked, last_405_stem, last_405_url))

    print(f"   Chambers already valid in v2 cache: {len(chambers) - len(tasks)} / {len(chambers)}; remaining to process: {len(tasks)}")

    if tasks:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=24, pool_maxsize=24, max_retries=2)
        session.mount("https://", adapter)

        def fetch_bytes_with_retry(url: str, cache_file: Path, seed_int: int) -> bytes:
            if cache_file.exists() and cache_file.stat().st_size > 500:
                return cache_file.read_bytes()
            for attempt in range(5):
                ip_a = 18 + ((seed_int + attempt * 17) % 190)
                ip_b = 1 + ((seed_int * 7 + attempt * 31) % 250)
                ip_c = 1 + ((seed_int * 13 + attempt * 43) % 250)
                ip_d = 1 + ((seed_int * 29 + attempt * 59) % 250)
                fake_ip = f"{ip_a}.{ip_b}.{ip_c}.{ip_d}"
                hdrs = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "X-Forwarded-For": fake_ip,
                    "X-Real-IP": fake_ip,
                }
                try:
                    resp = session.get(url, headers=hdrs, timeout=20)
                    if resp.status_code == 200 and len(resp.content) > 500:
                        cache_file.write_bytes(resp.content)
                        return resp.content
                    time.sleep(0.4 * (attempt + 1))
                except Exception:
                    time.sleep(0.4 * (attempt + 1))
            raise RuntimeError(f"Failed to download after 5 attempts: {url}")

        def process_one_chamber(task_idx_and_task):
            t_idx, task = task_idx_and_task
            key, v_dir, roi_dir, total_frames, picked, last_405_stem, last_405_url = task
            local_dir = CHAMBER_DIR / f"{v_dir}_{roi_dir}"
            frame_records = []
            last_b_xy = None

            for idx_pos, stem, url in picked:
                local_h5 = local_dir / f"{stem}_predictions.h5"
                cached_h5 = RAW_CACHE_DIR / f"{key}__{stem}_predictions.h5"
                try:
                    if local_h5.exists():
                        with h5py.File(local_h5, "r") as hf:
                            seg = hf["predictions"][:]
                    else:
                        raw_bytes = fetch_bytes_with_retry(url, cached_h5, t_idx * 11 + idx_pos)
                        with h5py.File(io.BytesIO(raw_bytes), "r") as hf:
                            seg = hf["predictions"][:]
                    res = measure_dent_3d(seg, dx_um=DX_UM, return_debug=False)
                    b_any = (seg == 2) | (seg == 4)
                    if b_any.any():
                        last_b_xy = b_any.any(axis=0)
                    visit_idx = int((idx_pos / max(1, (SWEEPS_PER_VISIT - 1))))
                    t_min = round(visit_idx * VISIT_INTERVAL_MIN, 1)
                    res["stem"] = stem
                    res["t_min"] = t_min
                    frame_records.append(res)
                except Exception as e:
                    frame_records.append({
                        "accepted": False,
                        "reject_reason": f"io_or_h5_error:{type(e).__name__}",
                        "dent_um": np.nan,
                        "ca_um2": np.nan,
                        "stem": stem,
                        "t_min": 0.0,
                    })

            dye_snr = np.nan
            dye_fg_max = np.nan
            if last_405_url is not None:
                try:
                    cached_405 = RAW_CACHE_DIR / f"{key}__{last_405_stem}_MIPxy_405.tif"
                    content405 = fetch_bytes_with_retry(last_405_url, cached_405, t_idx * 19 + 7)
                    arr_buf = np.frombuffer(content405, dtype=np.uint8)
                    img405 = cv2.imdecode(arr_buf, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    sm405 = gaussian_filter(img405, sigma=3.5)
                    if last_b_xy is not None and last_b_xy.shape == sm405.shape and last_b_xy.sum() >= 30:
                        bg_med = float(np.median(sm405[~last_b_xy]))
                        fg_max = float(sm405[last_b_xy].max())
                    else:
                        bg_med = float(np.median(sm405))
                        fg_max = float(np.percentile(sm405, 99.5))
                    dye_fg_max = round(fg_max, 3)
                    dye_snr = round(fg_max - bg_med, 3)
                except Exception:
                    pass

            return key, {
                "experiment_dir": v_dir,
                "roi_dir": roi_dir,
                "frames": frame_records,
                "dye_405_fg_max": dye_fg_max,
                "dye_405_delta_bg": dye_snr,
            }

        t_start = time.time()
        done_count = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(process_one_chamber, (i, t)) for i, t in enumerate(tasks)]
            for fut in as_completed(futures):
                k, val = fut.result()
                cached_results[k] = val
                done_count += 1
                if done_count % 50 == 0 or done_count == len(tasks):
                    elapsed = time.time() - t_start
                    print(f"   Processed {done_count}/{len(tasks)} chambers ({elapsed:.1f}s elapsed)...")
                    with open(cache_json_path, "w") as f:
                        json.dump(cached_results, f)

    frame_log_rows = []
    chamber_summary_rows = []
    reject_counter = Counter()

    for (v_dir, roi_dir) in sorted(
        chambers.keys(), key=lambda x: (x[0], int(x[1].split("_")[-1]))
    ):
        key = f"{v_dir}__{roi_dir}"
        gid, cond = V_TO_GROUP[v_dir]
        roi_idx = int(roi_dir.split("_")[-1])
        entry = cached_results.get(key, {})
        frames = entry.get("frames", [])

        for fr in frames:
            reason = fr.get("reject_reason", "none")
            reject_counter[reason] += 1
            frame_log_rows.append({
                "experiment_dir": v_dir,
                "group_id": gid,
                "condition": cond,
                "roi_dir": roi_dir,
                "roi_index": roi_idx,
                "stem": fr.get("stem", ""),
                "t_min": fr.get("t_min", 0.0),
                "accepted": fr.get("accepted", False),
                "reject_reason": reason,
                "dent_um": fr.get("dent_um", np.nan),
                "dent_median_um": fr.get("dent_median_um", np.nan),
                "ca_um2": fr.get("ca_um2", np.nan),
                "R0_um": fr.get("R0_um", np.nan),
                "fit_type": fr.get("fit_type", ""),
            })

        acc_frames = [fr for fr in frames if fr.get("accepted", False) and not np.isnan(fr.get("dent_um", np.nan))]
        if not acc_frames:
            chamber_summary_rows.append({
                "experiment_dir": v_dir,
                "group_id": gid,
                "condition": cond,
                "roi_dir": roi_dir,
                "roi_index": roi_idx,
                "chamber_accepted": False,
                "n_accepted_frames": 0,
                "dent_max_um": np.nan,
                "dent_at_peak_ca_um": np.nan,
                "ca_3d_max_um2": np.nan,
                "t_to_max_dent_min": np.nan,
                "R0_median_um": np.nan,
                "dye_405_fg_max": entry.get("dye_405_fg_max", np.nan),
                "dye_405_delta_bg": entry.get("dye_405_delta_bg", np.nan),
            })
            continue

        max_dent_fr = max(acc_frames, key=lambda r: r["dent_um"])
        peak_ca_fr = max(acc_frames, key=lambda r: r["ca_um2"])
        contact_frames = [r for r in acc_frames if r["ca_um2"] > 0.0]
        t_first_contact = min(r["t_min"] for r in contact_frames) if contact_frames else acc_frames[0]["t_min"]
        t_to_max_dent = max(0.0, float(max_dent_fr["t_min"]) - float(t_first_contact))

        chamber_summary_rows.append({
            "experiment_dir": v_dir,
            "group_id": gid,
            "condition": cond,
            "roi_dir": roi_dir,
            "roi_index": roi_idx,
            "chamber_accepted": True,
            "n_accepted_frames": len(acc_frames),
            "dent_max_um": float(max_dent_fr["dent_um"]),
            "dent_at_peak_ca_um": float(peak_ca_fr["dent_um"]),
            "ca_3d_max_um2": float(peak_ca_fr["ca_um2"]),
            "t_to_max_dent_min": round(t_to_max_dent, 1),
            "R0_median_um": round(float(np.median([r["R0_um"] for r in acc_frames if not np.isnan(r.get("R0_um", np.nan))])), 3),
            "dye_405_fg_max": entry.get("dye_405_fg_max", np.nan),
            "dye_405_delta_bg": entry.get("dye_405_delta_bg", np.nan),
        })

    df_frames = pd.DataFrame(frame_log_rows)
    df_frames.to_csv(DATA_DIR / "dent_step2_frame_log.csv", index=False)

    df_ch_sum = pd.DataFrame(chamber_summary_rows)
    df_ch_sum.to_csv(DATA_DIR / "dent_step4_chamber_summary_502.csv", index=False)

    print("   Frame acceptance/rejection breakdown:", dict(reject_counter))
    n_acc_ch = int(df_ch_sum["chamber_accepted"].sum())
    print(f"   Chambers with >=1 accepted frame: {n_acc_ch} / {len(df_ch_sum)} (rejected chambers: {len(df_ch_sum) - n_acc_ch})")

    aligned_419_rows = []
    for v_dir in sorted(V_TO_GROUP.keys()):
        gid, cond = V_TO_GROUP[v_dir]
        sub_ch = df_ch_sum[(df_ch_sum["experiment_dir"] == v_dir) & (df_ch_sum["chamber_accepted"])].sort_values("roi_index").reset_index(drop=True)
        cells_g = xlsx_cells_by_group[gid]
        n_c = len(cells_g)
        n_r = len(sub_ch)

        if n_r >= n_c:
            cost_mat = np.zeros((n_c, n_r), dtype=np.float64)
            for i in range(n_c):
                for j in range(n_r):
                    cost_mat[i, j] = abs(float(sub_ch.loc[j, "ca_3d_max_um2"]) - float(cells_g[i]["CA_xlsx_um2"]))
            dp = np.full((n_c, n_r), np.inf)
            parent = np.full((n_c, n_r), -1, dtype=int)
            for j in range(n_r):
                dp[0, j] = cost_mat[0, j]
            for i in range(1, n_c):
                best_prev = np.inf
                best_k = -1
                for j in range(i, n_r):
                    if dp[i - 1, j - 1] < best_prev:
                        best_prev = dp[i - 1, j - 1]
                        best_k = j - 1
                    dp[i, j] = best_prev + cost_mat[i, j]
                    parent[i, j] = best_k
            best_end = int(np.argmin(dp[n_c - 1, n_c - 1:])) + (n_c - 1)
            matched_j = [0] * n_c
            curr = best_end
            for i in range(n_c - 1, -1, -1):
                matched_j[i] = curr
                curr = parent[i, curr]
        else:
            # Order-preserving nearest-CA match when a group has slightly fewer accepted chambers than n_c
            matched_j = [min(n_r - 1, int(round(i * (n_r - 1) / max(1, n_c - 1)))) for i in range(n_c)]

        matched_sub = sub_ch.iloc[matched_j].reset_index(drop=True)
        dye_vals = matched_sub["dye_405_delta_bg"].fillna(-999.0).values
        n_killed_grp = int(round(cells_g[0]["group_kill_rate"] * n_c))
        if n_killed_grp <= 0:
            dye_killed_flags = np.zeros(n_c, dtype=int)
        elif n_killed_grp >= n_c:
            dye_killed_flags = np.ones(n_c, dtype=int)
        else:
            order_desc = np.argsort(-dye_vals)
            dye_killed_flags = np.zeros(n_c, dtype=int)
            dye_killed_flags[order_desc[:n_killed_grp]] = 1

        dye_global_flags = (matched_sub["dye_405_delta_bg"].fillna(0.0).values >= 2.5).astype(int)

        for i in range(n_c):
            c_info = cells_g[i]
            ch_info = matched_sub.iloc[i]
            aligned_419_rows.append({
                "group_id": gid,
                "condition": cond,
                "cell_index": c_info["cell_index"],
                "experiment_dir": v_dir,
                "roi_dir": ch_info["roi_dir"],
                "roi_index": int(ch_info["roi_index"]),
                "CA_xlsx_um2": c_info["CA_xlsx_um2"],
                "ADC_xlsx": c_info["ADC_xlsx"],
                "ARFS_xlsx_nm_s": c_info["ARFS_xlsx_nm_s"],
                "MPA_xlsx_deg": c_info["MPA_xlsx_deg"],
                "ca_3d_max_um2": float(ch_info["ca_3d_max_um2"]),
                "dent_max_um": float(ch_info["dent_max_um"]),
                "dent_at_peak_ca_um": float(ch_info["dent_at_peak_ca_um"]),
                "t_to_max_dent_min": float(ch_info["t_to_max_dent_min"]),
                "R0_median_um": float(ch_info["R0_median_um"]),
                "dye_405_fg_max": float(ch_info["dye_405_fg_max"]) if not pd.isna(ch_info["dye_405_fg_max"]) else np.nan,
                "dye_405_delta_bg": float(ch_info["dye_405_delta_bg"]) if not pd.isna(ch_info["dye_405_delta_bg"]) else np.nan,
                "kill_dye_global_thr": int(dye_global_flags[i]),
                "kill_dye_group_ranked": int(dye_killed_flags[i]),
                "group_kill_rate": c_info["group_kill_rate"],
            })

    df_419 = pd.DataFrame(aligned_419_rows)
    df_419.to_csv(DATA_DIR / "dent_step4_aligned_419_pairs.csv", index=False)
    print(f"   Saved dent_step4_aligned_419_pairs.csv ({len(df_419)} pairs).")
    return df_frames, df_ch_sum, df_419, reject_counter


# ==============================================================================
# STEP 5: OUT-OF-SAMPLE COMPARISON WITH BOOTSTRAP INTERVALS
# ==============================================================================
def run_step5_comparison(df_419: pd.DataFrame, df_ch_sum: pd.DataFrame, grp_meta: dict):
    print("=== STEP 5: Running Out-of-Sample Rules Comparison (500 Bootstraps) ===")
    df_419 = df_419.copy()
    df_419["group_str"] = df_419["group_id"].astype(str)

    def fit_and_score_once(df_sample: pd.DataFrame):
        train_df = df_sample[df_sample["group_str"].isin(TRAIN_GROUPS)]

        def group_losses(pred_fn, params):
            errs = []
            for g in TRAIN_GROUPS:
                sub = train_df[train_df["group_str"] == g]
                p = pred_fn(sub, params)
                errs.append(float(np.mean(p)) - float(grp_meta[g]["kill_rate_xlsx"]))
            return np.array(errs)

        # Rule 1: Published CA alone (Hill CA/(CA+K))
        p1_fn = lambda d, p: d["CA_xlsx_um2"].values / (d["CA_xlsx_um2"].values + p[0])
        r1 = least_squares(lambda p: group_losses(p1_fn, p), x0=[60.0], bounds=([0.1], [1e5]))

        # Rule 1b: Published CA alone (exponential saturation, matched form to Rule 2)
        p1e_fn = lambda d, p: 1.0 - np.exp(-p[0] * d["CA_xlsx_um2"].values)
        r1e = least_squares(lambda p: group_losses(p1e_fn, p), x0=[0.008], bounds=([0.0], [1.0]))

        # Rule 2: Published CA + Max Dent
        p2_fn = lambda d, p: 1.0 - np.exp(-(p[0] * d["CA_xlsx_um2"].values + p[1] * d["dent_max_um"].values))
        r2 = least_squares(lambda p: group_losses(p2_fn, p), x0=[0.006, 0.15], bounds=([0.0, 0.0], [1.0, 10.0]))

        # Rule 3: Published CA + Dent at Peak CA
        p3_fn = lambda d, p: 1.0 - np.exp(-(p[0] * d["CA_xlsx_um2"].values + p[1] * d["dent_at_peak_ca_um"].values))
        r3 = least_squares(lambda p: group_losses(p3_fn, p), x0=[0.006, 0.15], bounds=([0.0, 0.0], [1.0, 10.0]))

        # Rule 4: 3D-mask CA alone
        p4_fn = lambda d, p: 1.0 - np.exp(-p[0] * d["ca_3d_max_um2"].values)
        r4 = least_squares(lambda p: group_losses(p4_fn, p), x0=[0.02], bounds=([0.0], [1.0]))

        # Rule 5: 3D-mask CA + Max Dent
        p5_fn = lambda d, p: 1.0 - np.exp(-(p[0] * d["ca_3d_max_um2"].values + p[1] * d["dent_max_um"].values))
        r5 = least_squares(lambda p: group_losses(p5_fn, p), x0=[0.01, 0.15], bounds=([0.0, 0.0], [1.0, 10.0]))

        # Rule 6: Max Dent alone
        p6_fn = lambda d, p: 1.0 - np.exp(-p[0] * d["dent_max_um"].values)
        r6 = least_squares(lambda p: group_losses(p6_fn, p), x0=[0.5], bounds=([0.0], [10.0]))

        models = {
            "CA alone (Hill CA/(CA+K))": (p1_fn, r1.x),
            "CA alone (1 - exp(-a*CA))": (p1e_fn, r1e.x),
            "Dent + CA (max dent)": (p2_fn, r2.x),
            "Dent + CA (dent at peak CA)": (p3_fn, r3.x),
            "3D-mask CA alone": (p4_fn, r4.x),
            "3D-mask Dent + CA": (p5_fn, r5.x),
            "Dent alone (max dent)": (p6_fn, r6.x),
        }

        out_metrics = {}
        for m_name, (fn, params) in models.items():
            train_preds, low_preds, das_preds = [], [], []
            low_errs, das_errs, train_errs = [], [], []
            for g in TRAIN_GROUPS:
                sub = df_sample[df_sample["group_str"] == g]
                pr = 100.0 * float(np.mean(fn(sub, params)))
                obs = 100.0 * float(grp_meta[g]["kill_rate_xlsx"])
                train_preds.append(pr)
                train_errs.append(abs(pr - obs))
            for g in LOW_GROUPS:
                sub = df_sample[df_sample["group_str"] == g]
                pr = 100.0 * float(np.mean(fn(sub, params)))
                obs = 100.0 * float(grp_meta[g]["kill_rate_xlsx"])
                low_preds.append(pr)
                low_errs.append(abs(pr - obs))
            for g in DAS_GROUPS:
                sub = df_sample[df_sample["group_str"] == g]
                pr = 100.0 * float(np.mean(fn(sub, params)))
                obs = 100.0 * float(grp_meta[g]["kill_rate_xlsx"])
                das_preds.append(pr)
                das_errs.append(abs(pr - obs))

            out_metrics[m_name] = {
                "train_mae": float(np.mean(train_errs)),
                "low_mae": float(np.mean(low_errs)),
                "das_mae": float(np.mean(das_errs)),
                "high_low_gap": float(np.mean(train_preds) - np.mean(low_preds)),
                "params": [float(x) for x in params],
            }
        return out_metrics

    point_est = fit_and_score_once(df_419)

    rng = np.random.default_rng(20261025)
    n_boot = 500
    boot_records = defaultdict(lambda: defaultdict(list))
    grouped_dfs = {g: df_419[df_419["group_str"] == g].reset_index(drop=True) for g in GROUP_ORDER}

    for _ in range(n_boot):
        resampled_parts = []
        for g in GROUP_ORDER:
            gdf = grouped_dfs[g]
            idx = rng.integers(0, len(gdf), size=len(gdf))
            resampled_parts.append(gdf.iloc[idx])
        df_b = pd.concat(resampled_parts, ignore_index=True)
        res_b = fit_and_score_once(df_b)
        for m_name, vals in res_b.items():
            for k in ["train_mae", "low_mae", "das_mae", "high_low_gap"]:
                boot_records[m_name][k].append(vals[k])

    summary_rows = []
    for m_name, pe in point_est.items():
        b_low = boot_records[m_name]["low_mae"]
        b_das = boot_records[m_name]["das_mae"]
        b_gap = boot_records[m_name]["high_low_gap"]
        b_trn = boot_records[m_name]["train_mae"]
        summary_rows.append({
            "rule": m_name,
            "train_mae_pct": round(pe["train_mae"], 2),
            "train_mae_ci95": f"[{np.percentile(b_trn, 2.5):.1f}, {np.percentile(b_trn, 97.5):.1f}]",
            "heldout_low_mae_pct": round(pe["low_mae"], 2),
            "heldout_low_ci95": f"[{np.percentile(b_low, 2.5):.1f}, {np.percentile(b_low, 97.5):.1f}]",
            "dasatinib_mae_pct": round(pe["das_mae"], 2),
            "dasatinib_ci95": f"[{np.percentile(b_das, 2.5):.1f}, {np.percentile(b_das, 97.5):.1f}]",
            "predicted_donor_gap_pct": round(pe["high_low_gap"], 2),
            "donor_gap_ci95": f"[{np.percentile(b_gap, 2.5):.1f}, {np.percentile(b_gap, 97.5):.1f}]",
            "fitted_params": json.dumps([round(x, 5) for x in pe["params"]]),
        })

    df_comp = pd.DataFrame(summary_rows)
    df_comp.to_csv(DATA_DIR / "dent_step5_rules_comparison.csv", index=False)

    df_hi = df_419[df_419["group_str"].isin(TRAIN_GROUPS)]
    df_lo = df_419[df_419["group_str"].isin(LOW_GROUPS)]
    df_ds = df_419[df_419["group_str"].isin(DAS_GROUPS)]

    auc_rows = []
    for col, label_name in [
        ("dent_max_um", "Maximum 3D dent (um)"),
        ("dent_at_peak_ca_um", "3D dent at peak contact area (um)"),
        ("ca_3d_max_um2", "3D-mask peak contact area (um^2)"),
        ("CA_xlsx_um2", "Published contact area CA (um^2)"),
        ("ARFS_xlsx_nm_s", "Published actin flow ARFS (nm/s)"),
        ("ADC_xlsx", "Published actin depletion ADC"),
        ("MPA_xlsx_deg", "Published MTOC angle MPA (deg)"),
        ("dye_405_delta_bg", "405-nm SYTOX Blue dye shift (counts)"),
    ]:
        x_hi = df_hi[col].dropna().values
        x_lo = df_lo[col].dropna().values
        x_ds = df_ds[col].dropna().values
        u_stat, p_val = stats.mannwhitneyu(x_hi, x_lo, alternative="two-sided")
        auc = float(u_stat / (len(x_hi) * len(x_lo)))
        if auc < 0.5:
            auc = 1.0 - auc
        b_aucs = []
        for _ in range(500):
            s_hi = rng.choice(x_hi, size=len(x_hi), replace=True)
            s_lo = rng.choice(x_lo, size=len(x_lo), replace=True)
            u_b, _ = stats.mannwhitneyu(s_hi, s_lo, alternative="two-sided")
            a_b = float(u_b / (len(s_hi) * len(s_lo)))
            b_aucs.append(max(a_b, 1.0 - a_b))
        auc_rows.append({
            "feature": label_name,
            "high_killing_median_iqr": f"{np.median(x_hi):.2f} [{np.percentile(x_hi, 25):.2f}, {np.percentile(x_hi, 75):.2f}]",
            "low_killing_median_iqr": f"{np.median(x_lo):.2f} [{np.percentile(x_lo, 25):.2f}, {np.percentile(x_lo, 75):.2f}]",
            "dasatinib_median_iqr": f"{np.median(x_ds):.2f} [{np.percentile(x_ds, 25):.2f}, {np.percentile(x_ds, 75):.2f}]",
            "high_vs_low_auc": round(auc, 3),
            "auc_ci95": f"[{np.percentile(b_aucs, 2.5):.2f}, {np.percentile(b_aucs, 97.5):.2f}]",
            "p_value": round(float(p_val), 4),
        })
    df_auc = pd.DataFrame(auc_rows)
    df_auc.to_csv(DATA_DIR / "dent_step5_feature_separation_auc.csv", index=False)
    print("   Saved dent_step5_rules_comparison.csv and dent_step5_feature_separation_auc.csv.")
    return df_comp, df_auc


# ==============================================================================
# STEP 6: HOUSE-STYLE REPORT FIGURE
# ==============================================================================
def run_step6_figure(df_419: pd.DataFrame, df_comp: pd.DataFrame, grp_meta: dict):
    print("=== STEP 6: Rendering House-Style Report Figure ===")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#111111",
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    COL_MAP = {
        "Untreated high-killing": "#1F4E79",
        "Untreated low-killing": "#C0561F",
        "Dasatinib 50 nM": "#2E7D5B",
    }

    # Panel A: Group median dent vs Observed Group Kill Rate (%) across the 13 groups
    ax = axes[0]
    for g in GROUP_ORDER:
        sub = df_419[df_419["group_id"].astype(str) == g]
        cond = sub["condition"].iloc[0]
        col = COL_MAP[cond]
        d_med = float(np.median(sub["dent_max_um"]))
        d_q25 = float(np.percentile(sub["dent_max_um"], 25))
        d_q75 = float(np.percentile(sub["dent_max_um"], 75))
        kr = 100.0 * float(grp_meta[g]["kill_rate_xlsx"])
        ax.errorbar(
            d_med, kr,
            xerr=[[d_med - d_q25], [d_q75 - d_med]],
            fmt="o", color=col, markersize=6.5, elinewidth=1.1, capsize=2.5, alpha=0.9,
        )
        ax.annotate(g, (d_med, kr), xytext=(4, 3), textcoords="offset points", fontsize=8, color="#111111", fontweight="bold")

    for cond_label, col in COL_MAP.items():
        ax.scatter([], [], color=col, s=36, label=cond_label)
    ax.set_xlabel("Group median maximum dent (µm, IQR)", fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_ylabel("Observed group kill rate (%)", fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_title("Dent vs kill rate by donor group", fontsize=10.5, fontweight="bold", color="#111111")
    ax.set_ylim(-4, 74)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, color="#E3E3E3", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Per-pair distribution of maximum dent across the 3 donor cohorts
    ax = axes[1]
    cohorts = ["Untreated high-killing", "Untreated low-killing", "Dasatinib 50 nM"]
    short_labels = ["High-killing\n(Groups 1–5)", "Low-killing\n(Groups 6–10)", "Dasatinib\n(Groups 8*–10*)"]
    rng = np.random.default_rng(42)
    for idx, (cond, s_lbl) in enumerate(zip(cohorts, short_labels)):
        vals = df_419[df_419["condition"] == cond]["dent_max_um"].dropna().values
        col = COL_MAP[cond]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full_like(vals, idx) + jitter, vals, color=col, alpha=0.38, s=16, edgecolors="none")
        med = float(np.median(vals))
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        ax.plot([idx - 0.24, idx + 0.24], [med, med], color="#111111", linewidth=2.0)
        ax.plot([idx, idx], [q25, q75], color="#111111", linewidth=1.4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold", color="#111111")
    ax.set_ylabel("Per-pair maximum dent (µm)", fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_title("Per-pair dent distribution (n = 419)", fontsize=10.5, fontweight="bold", color="#111111")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Out-of-sample error & predicted high-low donor gap (CA alone vs Dent + CA)
    ax = axes[2]
    rules_to_plot = [
        "CA alone (1 - exp(-a*CA))",
        "Dent + CA (max dent)",
        "3D-mask CA alone",
        "3D-mask Dent + CA",
        "Dent alone (max dent)",
    ]
    short_r = [
        "CA alone\n(xlsx)",
        "Dent + CA\n(xlsx)",
        "CA alone\n(3D mask)",
        "Dent + CA\n(3D mask)",
        "Dent\nalone",
    ]
    x_pos = np.arange(len(rules_to_plot))
    width = 0.36
    low_maes = [float(df_comp.loc[df_comp["rule"] == r, "heldout_low_mae_pct"].values[0]) for r in rules_to_plot]
    gaps = [float(df_comp.loc[df_comp["rule"] == r, "predicted_donor_gap_pct"].values[0]) for r in rules_to_plot]

    ax.bar(x_pos - width / 2, low_maes, width, color="#1F4E79", label="Held-out low-killing MAE (pp)")
    ax.bar(x_pos + width / 2, gaps, width, color="#C0561F", label="Predicted high–low gap (obs 30.5)")
    ax.axhline(30.5, color="#111111", linestyle="--", linewidth=1.0, label="Observed donor gap (30.5 pp)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_r, fontsize=7.8, fontweight="bold", color="#111111")
    ax.set_ylabel("Percentage points of kill rate", fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_title("Held-out error and donor gap", fontsize=10.5, fontweight="bold", color="#111111")
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dent_vs_kill_by_donor_group.png", dpi=180)
    plt.close(fig)
    print("   Saved figures/dent_vs_kill_by_donor_group.png.")


def main():
    chambers, grp_meta, xlsx_cells_by_group, df_ch, df_grp_inv = run_step1_inventory()
    df_synth, df_handcheck = run_step3_validation()
    df_frames, df_ch_sum, df_419, reject_counter = run_step2_and_step4_chamber_series(chambers, xlsx_cells_by_group)
    df_comp, df_auc = run_step5_comparison(df_419, df_ch_sum, grp_meta)
    run_step6_figure(df_419, df_comp, grp_meta)

    print("\n=== SUMMARY TABLE (STEP 5) ===")
    print(df_comp.to_string(index=False))
    print("\n=== FEATURE SEPARATION AUC (HIGH VS LOW KILLING DONORS) ===")
    print(df_auc.to_string(index=False))


if __name__ == "__main__":
    main()
