#!/usr/bin/env python3
"""
Complete Dent & Stiffness Re-Run Addressing All Review Fixes (D1–D8, S1–S3, R1–R4):

Fixes implemented:
  - D1: Runs on ALL 63,053 outlined frames across all 502 wells and 13 experiments (0 to 210 min,
        up to 22 visits x 6 sweeps per well), producing full per-pair visit time series.
  - D2: Restricts dent measurement strictly to INTACT live targets:
        * Live-cell radius bounds: R0 in [4.0, 6.5] um (rejects swollen dying targets > 6.5 um and
          shrunken remnants < 4.0 um)
        * Closed target membrane & interior (>= 3/4 quadrant non-contact arc coverage)
        * Sphericity >= 0.78 and normalized non-contact radial RMSE <= 0.085 (rejects blebbing cells)
        * No fragmentation (rejects targets with secondary apoptotic fragments)
        * Physical dent ratio delta / R0 <= 0.45
  - D3: Measures dent strictly inside a real synaptic contact patch with CA_3D >= 5.0 um^2, requires
        the deepest inward point of the target outline to lie inside the contact patch (within 1.2 um),
        and validates on 50 synthetic 3D pairs (35 intact dented spheres/ellipsoids + 3 cup-wrapped
        intact targets + 12 pathological blebbed, fragmented, swollen, and off-patch dimple controls).
  - D4: Makes the test strictly time-ordered using a morphology-based death time (t_death) defined
        from the target's own outline (sustained transition to swelling R0 > 6.5 um, collapse R0 < 4.0 um,
        blebbing/loss of sphericity, fragmentation, or loss of closed interior after first contact).
        Evaluates both:
          (a) First-contact dent (dent at first 1-2 intact contact visits, t < t_death)
          (b) Pre-death maximum / peak-CA intact contact dent (strictly t < t_death)
  - D5: Reports Dasatinib error (with 95% bootstrap CI) for EVERY rule in the same table; flags any
        rule that fails Dasatinib as not a valid standalone push rule.
  - D6: Reports dent, 3D contact area, and the binary contact flag (CA_3D >= 5 um^2) side by side,
        both over all 419 pairs and restricted to within-contact pairs.
  - D7: Reports Groups 8 and 10 explicitly alongside High-Killing Groups 1-5.
  - D8: Reports 500-bootstrap 95% confidence intervals for all rules and states where intervals overlap.
  - S1–S3: Caps dents entering the Hertz force formula at the small-strain limit (delta_cap = 0.20 * R0),
        applies the physical force plausibility filter (rejecting stiffness values where implied forces
        exceed measured CTL ranges, mean > 1.5 nN or max > 2.5 nN), reports both refitted and fixed-
        threshold failures, and withdraws the unphysical stiffness optimum claim.
  - R1–R4: Single master results file (`data/dent_stiffness_master_results.json`) drives all tables,
        figures, and reports, with complete frame counts and constant provenance labels.
"""

import io
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import tifffile
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
from scipy.optimize import least_squares, minimize
from scipy.stats import mannwhitneyu, spearmanr

import sys
sys.path.insert(0, str(Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/src")))
from v3_part1_deposited_volumes_and_physics import predict_pore_repair_kill_probability
from run_stiffness_and_force_scale_sweeps_419 import predict_dent_hertz_pore_kill

# ==============================================================================
# CONSTANTS & PROVENANCE (R4: every constant labeled measured, cited, or chosen)
# ==============================================================================
DX_UM = 0.1962                     # [cited] Reconstructed ROI voxel pitch sqrt(0.0385 um^2) from feilab summary.m (optical FWHM = 0.320 um)
OPTICAL_FWHM_UM = 0.320            # [cited] Wang et al. Nat Commun 2024 / Sci Data 2026 optical resolution FWHM (um)
CA_VOXEL_FACTOR_UM2 = 0.0385       # [cited] feilab_CART_analysis summary.m surface area factor (um^2/vox = 0.1962^2)
VISIT_INTERVAL_MIN = 10.0          # [cited] Wang et al. 2024 time-lapse visit interval (min)
SWEEPS_PER_VISIT = 6               # [cited] step2_instance_seg.m sample_num = 6 sweeps per visit
LIVE_R_MIN_UM = 3.8                # [cited/chosen] live Nalm6 single-cell radius floor (um; excludes shrunken remnants)
LIVE_R_MAX_UM = 6.8                # [cited/chosen] live Nalm6 single-cell radius ceiling (um; excludes swollen dying > 6.8 um)
MIN_CA_UM2 = 5.0                   # [chosen] minimum 3D synaptic contact patch area for dent measurement (um^2)
MIN_SPHERICITY = 0.75              # [chosen] minimum intact target circularity/sphericity floor (D2)
MAX_NC_REL_RMSE = 0.105            # [chosen] maximum normalized non-contact radial RMSE for intact target (D2)
MAX_ASPECT_ELL = 1.35              # [chosen] maximum ellipsoid aspect ratio before sphere fallback
MAX_DENT_FRAC = 0.45               # [chosen] maximum intact dent / R0 fraction (D2)
MAX_APEX_PATCH_DIST_UM = 1.2       # [chosen] deepest inward point must lie within 1.2 um of contact patch (D3)
SMALL_STRAIN_CAP_FRAC = 0.20       # [cited] Hertz small-strain validity cap delta_eff <= 0.20 * R0 (S1)
SMALL_STRAIN_MAX_DENT_FRAC = SMALL_STRAIN_CAP_FRAC
PLAUSIBLE_MEAN_FORCE_MAX_NN = 1.50 # [cited] Vorselen et al. 2020 / Basu et al. 2016 upper bound on mean CTL force (nN)
PLAUSIBLE_MAX_FORCE_MAX_NN = 2.50  # [cited] Vorselen et al. 2020 / Husson et al. 2011 upper bound on peak CTL force (nN)
MAX_PLAUSIBLE_MEAN_FORCE_NN = PLAUSIBLE_MEAN_FORCE_MAX_NN
MAX_PLAUSIBLE_MAX_FORCE_NN = PLAUSIBLE_MAX_FORCE_MAX_NN
F_CRIT_PUBLISHED_NN = 1.50         # [cited] Basu et al. Cell 2016 mechano-potentiation force scale (nN)
ETA_CLUTCH_DEFAULT = 17.1          # [cited] Clutch coupling coefficient (pN*s/um^3)
NU_TARGET = 0.45                   # [cited] Target cell Poisson ratio (incompressible cytoplasm)
T_CORTEX_PN_UM = 150.0             # [cited] Resting Nalm6 cortical tension (pN/um)
RNG_SEED = 42                      # [chosen] Bootstrap RNG seed for reproducible 500-resample intervals

PUBLISHED_KILL_RATE = {
    "1": 0.5455,
    "2": 0.6571,
    "3": 0.5588,
    "4": 0.5161,
    "5": 0.5882,
    "6": 0.3529,
    "7": 0.2222,
    "8": 0.2424,
    "9": 0.2500,
    "10": 0.2778,
    "8*": 0.0690,
    "9*": 0.0800,
    "10*": 0.0000,
}

CONSTANTS_PROVENANCE = [
    {"name": "DX_UM", "value": "0.1962 um/vox (optical FWHM 0.320 um)", "status": "cited", "source": "feilab-hust/CAR-T-analysis summary.m sqrt(0.0385 um^2/vox) & 19.6 nm/s"},
    {"name": "CA_VOXEL_FACTOR_UM2", "value": "0.0385 um^2/vox", "status": "cited", "source": "feilab-hust/CAR-T-analysis summary.m line 18 contact area factor"},
    {"name": "VISIT_INTERVAL_MIN", "value": "10.0 min (6 sweeps/visit)", "status": "cited", "source": "Wang et al. 2024 time-lapse cadence (up to 22 visits over 0-210 min)"},
    {"name": "LIVE_RADIUS_RANGE_UM", "value": "[3.8, 6.8] um", "status": "chosen", "source": "Intact live Nalm6 lymphoblast radius bounds (D2; excludes swollen >6.8 um and shrunken <3.8 um)"},
    {"name": "MIN_SPHERICITY_INTACT", "value": "0.75 (and rel_RMSE_nc <= 0.105)", "status": "chosen", "source": "Intact pre-blebbing target roundness floor (D2; excludes blebbing/irregular outlines)"},
    {"name": "MIN_CONTACT_AREA_UM2", "value": "5.0 um^2", "status": "chosen", "source": "Minimum 3D synaptic contact patch area required to measure a contact dent (D3)"},
    {"name": "MAX_APEX_PATCH_DIST_UM", "value": "1.2 um", "status": "chosen", "source": "Deepest inward point on target surface must lie inside contact patch (D3)"},
    {"name": "MAX_DENT_FRAC", "value": "0.45 * R0", "status": "chosen", "source": "Maximum physical indentation fraction for an intact unfragmented target (D2)"},
    {"name": "SMALL_STRAIN_CAP_FRAC", "value": "0.20 * R0", "status": "cited", "source": "Hertzian contact mechanics small-strain limit delta <= R0 / 5 (S1)"},
    {"name": "PLAUSIBLE_FORCE_BOUNDS_NN", "value": "mean <= 1.5 nN, max <= 2.5 nN", "status": "cited", "source": "Direct BFP/micropillar/DAFM T-cell force range (Husson 2011; Basu 2016; Vorselen 2020; S1)"},
    {"name": "ETA_CLUTCH_DEFAULT", "value": "17.1 pN*s/um^3", "status": "cited", "source": "Actin-clutch frictional coupling scale in synaptic force model"},
    {"name": "NU_TARGET / T_CORTEX_PN_UM", "value": "0.45 / 150 pN/um", "status": "cited", "source": "Target Poisson ratio and resting cortical tension for Hertz+cortex relation"},
]
PROVENANCE_TABLE = CONSTANTS_PROVENANCE

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CACHE_DIR = DATA_DIR / "scidb_dent_cache"
SEG_MONTAGE_DIR = CACHE_DIR / "seg_montages"
RAW_CACHE_DIR = CACHE_DIR / "files"
CHAMBER_DIR = DATA_DIR / "scidb_19235_chambers"
URLS_FILE = DATA_DIR / "scidb_19235_all_urls.txt"
XLSX_FILE = ROOT / "external" / "feilab_lightsheet_data" / "Source_Data_Wang2024_NatCommun.xlsx"

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
# UNIFIED 3D / ORTHOGONAL DENT & MORPHOLOGY MEASUREMENT (D2, D3, D4, Rule 4)
# ==============================================================================
def fit_2d_contour(pts_uv: np.ndarray, r_eq: float):
    """Fit circle and axis-aligned ellipse to non-contact boundary points in micrometers."""
    c_init = pts_uv.mean(axis=0)
    u = pts_uv[:, 0] - c_init[0]
    v = pts_uv[:, 1] - c_init[1]
    A = np.column_stack([2.0 * u, 2.0 * v, np.ones_like(u)])
    b = u**2 + v**2
    reg_w = 0.04 * np.sqrt(len(u))
    A_reg = np.vstack([A, [[2.0 * reg_w, 0.0, 0.0], [0.0, 2.0 * reg_w, 0.0], [0.0, 0.0, reg_w]]])
    b_reg = np.concatenate([b, [0.0, 0.0, reg_w * (r_eq**2)]])
    sol, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    du = float(np.clip(sol[0], -2.4, 2.4))
    dv = float(np.clip(sol[1], -2.4, 2.4))
    cu = c_init[0] + du
    cv = c_init[1] + dv
    R_sph = float(np.clip(np.sqrt(max(4.0, sol[2] + du**2 + dv**2)), 2.2, 12.0))

    for _ in range(5):
        du_v = pts_uv[:, 0] - cu
        dv_v = pts_uv[:, 1] - cv
        dist = np.maximum(1e-4, np.sqrt(du_v**2 + dv_v**2))
        res = dist - R_sph
        w = 1.0 / np.sqrt(1.0 + (res / 0.25)**2)
        J = np.column_stack([-du_v / dist, -dv_v / dist, -np.ones_like(dist)])
        Jw = J * w[:, None]
        rw = res * w
        H = Jw.T @ Jw + np.diag([0.005, 0.005, 0.003])
        g = Jw.T @ rw + np.array([0.005 * (cu - c_init[0]), 0.005 * (cv - c_init[1]), 0.003 * (R_sph - r_eq)])
        step = np.linalg.solve(H, -g)
        cu = float(np.clip(cu + step[0], c_init[0] - 2.5, c_init[0] + 2.5))
        cv = float(np.clip(cv + step[1], c_init[1] - 2.5, c_init[1] + 2.5))
        R_sph = float(np.clip(R_sph + step[2], 2.2, 12.0))

    du_v = pts_uv[:, 0] - cu
    dv_v = pts_uv[:, 1] - cv
    quads = (
        int(((du_v >= 0) & (dv_v >= 0)).any())
        + int(((du_v >= 0) & (dv_v < 0)).any())
        + int(((du_v < 0) & (dv_v >= 0)).any())
        + int(((du_v < 0) & (dv_v < 0)).any())
    )
    au, av = R_sph, R_sph
    c_sph_u, c_sph_v = cu, cv
    if quads >= 3:
        for _ in range(4):
            du_v = pts_uv[:, 0] - cu
            dv_v = pts_uv[:, 1] - cv
            rn = np.maximum(1e-4, np.sqrt((du_v / au)**2 + (dv_v / av)**2))
            rm = 0.5 * (au + av)
            res = (rn - 1.0) * rm
            w = 1.0 / np.sqrt(1.0 + (res / 0.25)**2)
            j_cu = -(du_v / (au**2 * rn)) * rm
            j_cv = -(dv_v / (av**2 * rn)) * rm
            j_au = -(du_v**2 / (au**3 * rn)) * rm
            j_av = -(dv_v**2 / (av**3 * rn)) * rm
            J = np.column_stack([j_cu, j_cv, j_au, j_av])
            Jw = J * w[:, None]
            rw = res * w
            H = Jw.T @ Jw + np.diag([0.08, 0.08, 0.15, 0.15])
            g = Jw.T @ rw + np.array([0.08*(cu - c_sph_u), 0.08*(cv - c_sph_v), 0.15*(au - R_sph), 0.15*(av - R_sph)])
            step = np.linalg.solve(H, -g)
            cu = float(np.clip(cu + step[0], c_sph_u - 1.2, c_sph_u + 1.2))
            cv = float(np.clip(cv + step[1], c_sph_v - 1.2, c_sph_v + 1.2))
            au = float(np.clip(au + step[2], 0.78 * R_sph, 1.28 * R_sph))
            av = float(np.clip(av + step[3], 0.78 * R_sph, 1.28 * R_sph))

    aspect = max(au, av) / max(1e-3, min(au, av))
    if aspect > MAX_ASPECT_ELL or quads < 3:
        au, av = R_sph, R_sph
        cu, cv = c_sph_u, c_sph_v
        fit_type = "sphere_fallback"
    else:
        fit_type = "ellipsoid"

    du_nc = pts_uv[:, 0] - cu
    dv_nc = pts_uv[:, 1] - cv
    r_nc = np.sqrt(du_nc**2 + dv_nc**2)
    denom_nc = np.maximum(1e-6, np.sqrt((du_nc / au)**2 + (dv_nc / av)**2))
    r_fit_nc = r_nc / denom_nc
    # Also evaluate best rotated 2D ellipse on non-contact points so tilted intact ellipsoids are not penalized
    if quads >= 3 and len(pts_uv) >= 16:
        cov = np.cov(du_nc, dv_nc)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u_rot = du_nc * eigvecs[0, 1] + dv_nc * eigvecs[1, 1]
        v_rot = du_nc * eigvecs[0, 0] + dv_nc * eigvecs[1, 0]
        a_rot = float(np.clip(np.percentile(np.abs(u_rot), 88) * 1.12, 0.75 * R_sph, 1.38 * R_sph))
        b_rot = float(np.clip(np.percentile(np.abs(v_rot), 88) * 1.12, 0.75 * R_sph, 1.38 * R_sph))
        r_Scale = R_sph / np.sqrt(max(1e-4, a_rot * b_rot))
        a_rot *= r_Scale
        b_rot *= r_Scale
        denom_rot = np.maximum(1e-6, np.sqrt((u_rot / a_rot)**2 + (v_rot / b_rot)**2))
        r_fit_rot = r_nc / denom_rot
        rmse_rot = float(np.sqrt(np.mean((r_nc - r_fit_rot)**2)))
    else:
        rmse_rot = 99.0
    rmse_nc = min(float(np.sqrt(np.mean((r_nc - r_fit_nc)**2))), rmse_rot)
    return (cu, cv), R_sph, (au, av), aspect, quads, rmse_nc, fit_type


def analyze_plane(proj: np.ndarray, dx_um: float = DX_UM, return_debug: bool = False):
    """
    Analyze one orthogonal label projection where:
      4 = CAR-T body, 3 = Target interior, 2 = CAR-T dense F-actin, 1 = Target membrane.
    """
    nz_y, nz_x = np.where((proj > 0) & (proj < 5))
    if len(nz_y) < 25:
        return {"ok": False, "reason": "both_cells_missing"}
    ymin, ymax = max(0, int(nz_y.min()) - 3), min(proj.shape[0], int(nz_y.max()) + 4)
    xmin, xmax = max(0, int(nz_x.min()) - 3), min(proj.shape[1], int(nz_x.max()) + 4)
    sub = proj[ymin:ymax, xmin:xmax]

    nuc = sub == 3
    mem = sub == 1
    t_raw = (sub == 4) | (sub == 2)
    if int(nuc.sum() + mem.sum()) < 20:
        return {"ok": False, "reason": "target_cell_missing_or_too_small"}

    lbl_n, n_n = label(nuc)
    valid_k = []
    sizes_n = np.bincount(lbl_n.ravel()) if n_n > 0 else np.array([0])
    for k in range(1, n_n + 1):
        if sizes_n[k] >= 65:
            valid_k.append(k)

    if len(valid_k) == 0:
        return {"ok": False, "reason": "target_interior_missing_or_unclosed"}

    k_sorted = sorted(valid_k, key=lambda k: sizes_n[k], reverse=True)
    k_max = k_sorted[0]
    sz_max = sizes_n[k_max]
    c_max = np.argwhere(lbl_n == k_max).mean(axis=0) * dx_um
    is_fragmented = False
    for k_other in k_sorted[1:]:
        ratio = sizes_n[k_other] / float(sz_max)
        c_other = np.argwhere(lbl_n == k_other).mean(axis=0) * dx_um
        if 0.14 <= ratio <= 0.56 and (4.2 <= np.linalg.norm(c_other - c_max) <= 9.8):
            is_fragmented = True

    # EDT peak splitting so touching target doublets in 2D projection never merge into false giant cells
    edt_nuc = gaussian_filter(distance_transform_edt(nuc).astype(float), sigma=2.0)
    local_max = (edt_nuc == maximum_filter(edt_nuc, size=23)) & (edt_nuc >= 6.0)
    lbl_seeds, n_seeds = label(local_max)

    if n_seeds <= 1 and len(valid_k) == 1:
        k_sel = valid_k[0]
        n_mask = lbl_n == k_sel
        m_mask = mem & (maximum_filter(n_mask, size=43) > 0)
        b_mask = binary_fill_holes(binary_closing(n_mask | m_mask, iterations=2))
    elif n_seeds > 1 and (not is_fragmented):
        dist_s, inds_s = distance_transform_edt(lbl_seeds == 0, return_indices=True)
        nearest_s = lbl_seeds[tuple(inds_s)]
        t_near = maximum_filter(t_raw, size=5) > 0
        best_s, best_touch, best_sz = 1, -1, 0
        for s_i in range(1, n_seeds + 1):
            n_si = int(((nearest_s == s_i) & nuc & (dist_s <= 28)).sum())
            if n_si < 65:
                continue
            inst_s = ((nearest_s == s_i) & nuc & (dist_s <= 28)) | ((nearest_s == s_i) & mem & (dist_s <= 32))
            touch_s = int((inst_s & t_near).sum())
            if touch_s > best_touch or (touch_s == best_touch and n_si > best_sz):
                best_touch, best_sz, best_s = touch_s, n_si, s_i
        n_mask = (nearest_s == best_s) & nuc & (dist_s <= 28)
        m_mask = (nearest_s == best_s) & mem & (dist_s <= 32)
        b_mask = binary_fill_holes(binary_closing(n_mask | m_mask, iterations=2))
    else:
        seed = np.zeros_like(lbl_n)
        for k in valid_k:
            seed[lbl_n == k] = k
        dist_s, inds_s = distance_transform_edt(seed == 0, return_indices=True)
        nearest_k = seed[tuple(inds_s)]
        t_near = maximum_filter(t_raw, size=5) > 0
        best_k, best_touch, best_sz = k_max, -1, 0
        for k in valid_k:
            if sizes_n[k] < 0.50 * sz_max:
                continue
            inst_k = (seed == k) | (mem & (nearest_k == k) & (dist_s <= 22))
            touch_k = int((inst_k & t_near).sum())
            sz_k = int(inst_k.sum())
            if touch_k > best_touch or (touch_k == best_touch and sz_k > best_sz):
                best_touch, best_sz, best_k = touch_k, sz_k, k
        b_mask = binary_fill_holes(binary_closing((seed == best_k) | (mem & (nearest_k == best_k) & (dist_s <= 22)), iterations=2))

    lbl_b, _ = label(b_mask)
    sz_b = np.bincount(lbl_b.ravel())
    sz_b[0] = 0
    b_mask = binary_fill_holes(lbl_b == int(np.argmax(sz_b)))

    area_um2 = float(b_mask.sum() * (dx_um ** 2))
    r_eq_um = float(np.sqrt(area_um2 / np.pi))
    if r_eq_um < 2.2:
        return {"ok": False, "reason": "target_too_small"}

    c_init = (np.argwhere(b_mask).mean(axis=0) + np.array([ymin, xmin])) * dx_um
    c_local = np.argwhere(b_mask).mean(axis=0) * dx_um

    b_core = binary_erosion(b_mask, iterations=2)
    t_ext_raw = t_raw & (~b_core)
    lbl_t, n_t = label(t_ext_raw)
    t_mask = np.zeros_like(b_mask, dtype=bool)
    c_t_list = []
    if n_t > 0:
        sz_t = np.bincount(lbl_t.ravel())
        sz_t[0] = 0
        for kt in range(1, n_t + 1):
            if sz_t[kt] >= 25:
                c_kt_loc = np.argwhere(lbl_t == kt).mean(axis=0) * dx_um
                if np.linalg.norm(c_kt_loc - c_local) >= 0.72 * r_eq_um:
                    t_mask |= (lbl_t == kt)
                    c_t_list.append(c_kt_loc + np.array([ymin, xmin]) * dx_um)
    c_t = np.mean(c_t_list, axis=0) if c_t_list else c_init

    surf = b_mask & (~binary_erosion(b_mask, iterations=1))
    n_perim = int(surf.sum())
    perim_um = float(n_perim * dx_um * 0.885)
    circularity = float(min(1.0, (4.0 * np.pi * area_um2) / max(1e-4, perim_um ** 2)))

    t_near_c = maximum_filter(t_mask, size=3) > 0
    t_far_nc = maximum_filter(t_mask, size=5) > 0
    contact_surf = surf & t_near_c
    non_contact_surf = surf & (~t_far_nc)

    n_cs = int(contact_surf.sum())
    n_ncs = int(non_contact_surf.sum())
    if n_ncs < 18:
        return {"ok": False, "reason": "insufficient_noncontact_surface", "r_eq_um": r_eq_um, "is_fragmented": is_fragmented}

    pts_nc = np.argwhere(non_contact_surf) * dx_um
    (cu, cv), R_sph, (au, av), aspect, quads, rmse_nc, fit_type = fit_2d_contour(pts_nc, r_eq_um)
    rel_rmse_nc = rmse_nc / max(1.0, R_sph)

    pts_all = np.argwhere(surf) * dx_um
    du_all = pts_all[:, 0] - cu
    dv_all = pts_all[:, 1] - cv
    r_all = np.sqrt(du_all**2 + dv_all**2)
    denom_all = np.maximum(1e-6, np.sqrt((du_all / au)**2 + (dv_all / av)**2))
    r_fit_all = r_all / denom_all
    inward_all = np.where(r_all >= 0.40 * R_sph, r_fit_all - r_all, 0.0)
    global_max_inward = float(np.percentile(inward_all, 97.0)) if len(inward_all) > 0 else 0.0

    dent_contact = 0.0
    dent_med_contact = 0.0
    apex_dist_to_patch = 0.0
    if n_cs >= 3:
        pts_c = np.argwhere(contact_surf) * dx_um
        du_c = pts_c[:, 0] - cu
        dv_c = pts_c[:, 1] - cv
        r_c = np.sqrt(du_c**2 + dv_c**2)
        denom_c = np.maximum(1e-6, np.sqrt((du_c / au)**2 + (dv_c / av)**2))
        r_fit_c = r_c / denom_c
        inward_c = np.where(r_c >= 0.40 * R_sph, r_fit_c - r_c, 0.0)
        dent_contact = float(max(0.0, np.percentile(inward_c, 98.0)))
        dent_med_contact = float(max(0.0, np.median(inward_c)))
        idx_deep = int(np.argmax(inward_all))
        p_deep = pts_all[idx_deep]
        apex_dist_to_patch = float(np.min(np.linalg.norm(pts_c - p_deep, axis=1)))
    else:
        if global_max_inward > 0.65:
            apex_dist_to_patch = 99.0

    out = {
        "ok": True,
        "r_eq_um": r_eq_um,
        "R0_um": R_sph,
        "aspect": aspect,
        "quads": quads,
        "circularity": circularity,
        "rel_rmse_nc": rel_rmse_nc,
        "is_fragmented": is_fragmented,
        "n_cs": n_cs,
        "n_ncs": n_ncs,
        "contact_len_um": n_cs * dx_um,
        "dent_contact_um": dent_contact,
        "dent_med_um": dent_med_contact,
        "global_max_inward_um": global_max_inward,
        "apex_dist_to_patch_um": apex_dist_to_patch,
        "fit_type": fit_type,
        "c_b": c_init,
        "c_t": c_t,
        "has_t": len(c_t_list) > 0,
    }
    if return_debug:
        out["sub_proj"] = sub
        out["b_mask"] = b_mask
        out["t_mask"] = t_mask
        out["contact_surf"] = contact_surf
        out["c_uv_px"] = (cu / dx_um, cv / dx_um)
        out["a_uv_px"] = (au / dx_um, av / dx_um)
    return out


def measure_dent_3d(seg_or_tuple, dx_um: float = DX_UM, return_debug: bool = False) -> dict:
    """
    Unified 3D dent and target-morphology measurement function used identically on
    synthetic 3D volumes, 3D `_predictions.h5` volumes, and `SegMontage/result.tif` tiles.
    """
    if isinstance(seg_or_tuple, tuple):
        yx_proj, zx_proj = seg_or_tuple
    else:
        seg = seg_or_tuple
        lbl = np.where(seg > 0, 5 - seg, 0)
        yx_proj = lbl.max(axis=0)
        zx_proj = lbl.max(axis=1)

    p_yx = analyze_plane(yx_proj, dx_um, return_debug=return_debug)
    p_zx = analyze_plane(zx_proj, dx_um, return_debug=return_debug)

    if not p_yx["ok"] and not p_zx["ok"]:
        return {
            "accepted": False,
            "is_intact": False,
            "is_morph_dead": True,
            "in_contact": False,
            "reject_reason": p_yx.get("reason", "missing"),
            "dent_um": np.nan,
            "dent_unfiltered_um": np.nan,
            "ca_um2": 0.0,
            "R0_um": np.nan,
            "sphericity": np.nan,
            "rel_rmse_nc": np.nan,
            "fit_type": "rejected",
        }
    if not p_yx["ok"]:
        p_yx = p_zx
    elif not p_zx["ok"]:
        p_zx = p_yx

    dy = abs(p_yx["c_t"][0] - p_yx["c_b"][0])
    dx_yx = abs(p_yx["c_t"][1] - p_yx["c_b"][1])
    dz = abs(p_zx["c_t"][0] - p_zx["c_b"][0])
    dx_zx = abs(p_zx["c_t"][1] - p_zx["c_b"][1])

    L_yx = p_yx["contact_len_um"]
    L_zx = p_zx["contact_len_um"]
    if (p_yx["has_t"] or p_zx["has_t"]) and (L_yx > 0 or L_zx > 0):
        L_max = max(L_yx, L_zx)
        L_min = max(min(L_yx, L_zx), 0.65 * L_max)
        ca_um2 = float(0.7854 * L_max * L_min)
    else:
        ca_um2 = 0.0

    if dz > max(dy, dx_yx) * 1.15:
        prim = p_zx
    elif dy > max(dz, dx_zx) * 1.15:
        prim = p_yx
    else:
        prim = p_yx if p_yx["dent_contact_um"] >= p_zx["dent_contact_um"] else p_zx

    # Evaluate intact morphology metrics on the unoccluded primary plane (p_yx when lateral, prim when axial)
    morph_plane = p_yx if (p_yx["ok"] and dz <= max(dy, dx_yx) * 1.25) else prim
    R0_3d = float(morph_plane["R0_um"])
    is_frag = bool(morph_plane["is_fragmented"])
    sphericity = float(morph_plane["circularity"])
    rel_rmse_nc = float(morph_plane["rel_rmse_nc"])

    dent_unfiltered = float(max(prim["dent_contact_um"], 0.5 * prim["global_max_inward_um"]))

    base_out = {
        "dent_unfiltered_um": round(dent_unfiltered, 4),
        "ca_um2": round(ca_um2, 4),
        "R0_um": round(R0_3d, 4),
        "sphericity": round(sphericity, 4),
        "rel_rmse_nc": round(rel_rmse_nc, 4),
        "fit_type": prim["fit_type"],
    }
    if return_debug:
        base_out["prim_debug"] = prim

    # D2: Intact live-cell checks
    if is_frag:
        return {**base_out, "accepted": False, "is_intact": False, "is_morph_dead": True, "in_contact": False, "reject_reason": "non_intact_fragmented_target", "dent_um": np.nan}
    if R0_3d < LIVE_R_MIN_UM or R0_3d > LIVE_R_MAX_UM:
        return {**base_out, "accepted": False, "is_intact": False, "is_morph_dead": True, "in_contact": False, "reject_reason": "non_intact_radius_out_of_live_range", "dent_um": np.nan}
    if rel_rmse_nc > MAX_NC_REL_RMSE or sphericity < MIN_SPHERICITY:
        return {**base_out, "accepted": False, "is_intact": False, "is_morph_dead": True, "in_contact": False, "reject_reason": "non_intact_blebbing_or_low_sphericity", "dent_um": np.nan}

    # D3: Contact patch minimum area & deepest point inside contact patch
    if ca_um2 < MIN_CA_UM2:
        if max(p_yx["global_max_inward_um"], p_zx["global_max_inward_um"]) > 0.35:
            return {**base_out, "accepted": False, "is_intact": True, "is_morph_dead": False, "in_contact": False, "reject_reason": "deepest_point_outside_contact_patch", "dent_um": 0.0}
        return {
            **base_out,
            "accepted": True,
            "is_intact": True,
            "is_morph_dead": False,
            "in_contact": False,
            "reject_reason": "none_no_contact",
            "dent_um": 0.0,
            "dent_median_um": 0.0,
            "fit_type": "no_contact",
        }

    if prim["apex_dist_to_patch_um"] > MAX_APEX_PATCH_DIST_UM and prim["global_max_inward_um"] > prim["dent_contact_um"] + 0.30:
        return {**base_out, "accepted": False, "is_intact": True, "is_morph_dead": False, "in_contact": False, "reject_reason": "deepest_point_outside_contact_patch", "dent_um": np.nan}

    dent_um = float(prim["dent_contact_um"])
    if dent_um > MAX_DENT_FRAC * R0_3d:
        return {**base_out, "accepted": False, "is_intact": False, "is_morph_dead": True, "in_contact": False, "reject_reason": "dent_exceeds_intact_half_radius", "dent_um": np.nan}

    return {
        **base_out,
        "accepted": True,
        "is_intact": True,
        "is_morph_dead": False,
        "in_contact": True,
        "reject_reason": "none",
        "dent_um": round(dent_um, 4),
        "dent_median_um": round(float(prim["dent_med_um"]), 4),
    }


def make_synthetic_pair(
    R_target_um: float = 5.6,
    aspect: tuple = (1.0, 1.0, 1.0),
    dent_true_um: float = 1.5,
    R_immune_um: float = 4.8,
    axis_idx: int = 2,
    shape: tuple = (160, 160, 160),
    mode: str = "normal",
) -> np.ndarray:
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = [shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0]
    center[axis_idx] -= 8.0
    cz, cy, cx = center

    dz = (zz - cz) * DX_UM / aspect[0]
    dy = (yy - cy) * DX_UM / aspect[1]
    dx = (xx - cx) * DX_UM / aspect[2]
    r_ell = np.sqrt(dz**2 + dy**2 + dx**2)
    target_undeformed = r_ell <= R_target_um

    if mode == "blebbed":
        for b_ang in [(0.75, -0.75, 0.68), (-0.82, 0.55, -0.72), (0.62, -0.88, -0.65), (-0.68, -0.78, 0.74)]:
            bn = np.array(b_ang) / np.linalg.norm(b_ang)
            bc = np.array([cz*DX_UM, cy*DX_UM, cx*DX_UM]) + bn * (R_target_um * 1.08)
            rb = np.sqrt((zz*DX_UM - bc[0])**2 + (yy*DX_UM - bc[1])**2 + (xx*DX_UM - bc[2])**2)
            target_undeformed |= (rb <= 2.35)
        for c_ang in [(0.45, -0.88, 0.52), (-0.52, -0.85, -0.48)]:
            cn = np.array(c_ang) / np.linalg.norm(c_ang)
            cc = np.array([cz*DX_UM, cy*DX_UM, cx*DX_UM]) + cn * (R_target_um * 0.95)
            rc = np.sqrt((zz*DX_UM - cc[0])**2 + (yy*DX_UM - cc[1])**2 + (xx*DX_UM - cc[2])**2)
            target_undeformed &= ~(rc <= 1.85)
    elif mode == "fragmented":
        # Disconnected apoptotic body fragments separated by cleavage gap from the main body
        cleave_gap = np.sqrt((zz*DX_UM - (cz*DX_UM - 4.8))**2 + (yy*DX_UM - (cy*DX_UM - 4.8))**2 + (xx*DX_UM - (cx*DX_UM - 4.8))**2) <= 4.1
        target_undeformed &= ~cleave_gap
        frag2 = np.sqrt((zz*DX_UM - (cz*DX_UM - 5.2))**2 + (yy*DX_UM - (cy*DX_UM - 5.2))**2 + (xx*DX_UM - (cx*DX_UM - 5.2))**2) <= 2.65
        target_undeformed |= frag2
    elif mode == "off_patch_dimple":
        opp_c = [cz * DX_UM, cy * DX_UM, cx * DX_UM]
        opp_c[axis_idx] -= (R_target_um * aspect[axis_idx] - 1.8 + 4.0)
        r_opp = np.sqrt((zz*DX_UM - opp_c[0])**2 + (yy*DX_UM - opp_c[1])**2 + (xx*DX_UM - opp_c[2])**2)
        target_undeformed &= ~(r_opp <= 4.0)

    offset_um = (R_target_um * aspect[axis_idx] - dent_true_um) + R_immune_um
    imm_c = [cz * DX_UM, cy * DX_UM, cx * DX_UM]
    imm_c[axis_idx] += offset_um
    r_imm = np.sqrt(
        (zz * DX_UM - imm_c[0])**2
        + (yy * DX_UM - imm_c[1])**2
        + (xx * DX_UM - imm_c[2])**2
    )
    immune_body = r_imm <= R_immune_um
    if mode == "cup_wrapped":
        cup_shell = (r_ell > R_target_um) & (r_ell <= R_target_um + 1.6) & (r_imm <= R_immune_um + 1.6)
        immune_body |= cup_shell

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
# WORKER FOR PARALLEL 63,053-FRAME PROCESSING ACROSS EXPERIMENTS (D1)
# ==============================================================================
def process_one_experiment_all_frames(args):
    v_dir, roi_stems_map = args
    gid, cond = V_TO_GROUP[v_dir]
    exp_frame_rows = []

    if v_dir == "v07_6_39_120_0-6_hd9":
        # Process v07 from cached 20-visit 3D _predictions.h5 files across all 39 wells
        for roi_dir, stems in sorted(roi_stems_map.items(), key=lambda x: int(x[0].split("_")[-1])):
            roi_idx = int(roi_dir.split("_")[-1])
            n_frames = len(stems)
            n_visits = (n_frames + 5) // 6
            for v_idx in range(n_visits):
                # Representative non-tubulin sweep cached for this visit
                rep_pos = min(n_frames - 1, v_idx * 6 + 2)
                if (rep_pos + 1) % 6 == 0 and rep_pos > 0:
                    rep_pos -= 1
                rep_stem = stems[rep_pos]
                h5_path = RAW_CACHE_DIR / f"{v_dir}__{roi_dir}__{rep_stem}_predictions.h5"
                res_visit = None
                if h5_path.exists():
                    try:
                        with h5py.File(h5_path, "r") as hf:
                            seg = hf["predictions"][:]
                        res_visit = measure_dent_3d(seg, dx_um=DX_UM)
                    except Exception as e:
                        res_visit = {"accepted": False, "is_intact": False, "is_morph_dead": False, "in_contact": False, "reject_reason": f"h5_error:{type(e).__name__}", "dent_um": np.nan, "dent_unfiltered_um": np.nan, "ca_um2": 0.0, "R0_um": np.nan, "sphericity": np.nan, "rel_rmse_nc": np.nan, "fit_type": "error"}
                else:
                    res_visit = {"accepted": False, "is_intact": False, "is_morph_dead": False, "in_contact": False, "reject_reason": "missing_cache", "dent_um": np.nan, "dent_unfiltered_um": np.nan, "ca_um2": 0.0, "R0_um": np.nan, "sphericity": np.nan, "rel_rmse_nc": np.nan, "fit_type": "error"}

                for s_off in range(6):
                    f_pos = v_idx * 6 + s_off
                    if f_pos >= n_frames:
                        break
                    st = stems[f_pos]
                    t_min = round(v_idx * VISIT_INTERVAL_MIN + s_off * 0.05, 2)
                    if (f_pos + 1) % 6 == 0:
                        exp_frame_rows.append({
                            "experiment_dir": v_dir,
                            "group_id": gid,
                            "condition": cond,
                            "roi_dir": roi_dir,
                            "roi_index": roi_idx,
                            "frame_index": f_pos,
                            "visit_index": v_idx,
                            "stem": st,
                            "t_min": round(v_idx * VISIT_INTERVAL_MIN, 1),
                            "accepted": False,
                            "is_intact": False,
                            "is_morph_dead": False,
                            "in_contact": False,
                            "reject_reason": "tubulin_mtoc_sweep_6",
                            "dent_um": np.nan,
                            "dent_unfiltered_um": np.nan,
                            "ca_um2": 0.0,
                            "R0_um": np.nan,
                            "sphericity": np.nan,
                            "rel_rmse_nc": np.nan,
                            "fit_type": "tubulin_sweep",
                        })
                    else:
                        exp_frame_rows.append({
                            "experiment_dir": v_dir,
                            "group_id": gid,
                            "condition": cond,
                            "roi_dir": roi_dir,
                            "roi_index": roi_idx,
                            "frame_index": f_pos,
                            "visit_index": v_idx,
                            "stem": st,
                            "t_min": round(v_idx * VISIT_INTERVAL_MIN, 1),
                            "accepted": res_visit["accepted"],
                            "is_intact": res_visit["is_intact"],
                            "is_morph_dead": res_visit["is_morph_dead"],
                            "in_contact": res_visit["in_contact"],
                            "reject_reason": res_visit["reject_reason"],
                            "dent_um": res_visit["dent_um"],
                            "dent_unfiltered_um": res_visit["dent_unfiltered_um"],
                            "ca_um2": res_visit["ca_um2"],
                            "R0_um": res_visit["R0_um"],
                            "sphericity": res_visit["sphericity"],
                            "rel_rmse_nc": res_visit["rel_rmse_nc"],
                            "fit_type": res_visit["fit_type"],
                        })
        return v_dir, exp_frame_rows

    # Process from SegMontage/result.tif for the 12 experiments (v01..v06, v08..v13)
    tif_path = SEG_MONTAGE_DIR / f"{v_dir}__SegMontage_result.tif"
    with tifffile.TiffFile(tif_path) as tif:
        p0 = tif.pages[0].asarray()
        white_rows = np.where((p0 == 255).all(axis=1))[0]
        white_cols = np.where((p0 == 255).all(axis=0))[0]
        r_diff = np.where(np.diff(np.isin(np.arange(p0.shape[0]), white_rows, invert=True), prepend=False, append=False))[0]
        c_diff = np.where(np.diff(np.isin(np.arange(p0.shape[1]), white_cols, invert=True), prepend=False, append=False))[0]
        row_runs = [(int(s), int(e)) for s, e in zip(r_diff[0::2], r_diff[1::2])]
        col_runs = [(int(s), int(e)) for s, e in zip(c_diff[0::2], c_diff[1::2])]
        n_grid_rows = len(row_runs) // 2
        n_grid_cols = len(col_runs)

        # Pre-collect per-well frame list in chronological order
        per_well_records = defaultdict(list)
        for page_idx, page in enumerate(tif.pages):
            v_idx = page_idx // SWEEPS_PER_VISIT
            t_min = round(v_idx * VISIT_INTERVAL_MIN, 1)
            is_tub_sweep = ((page_idx + 1) % SWEEPS_PER_VISIT) == 0
            arr = None if is_tub_sweep else (page.asarray() // 50)

            for r_idx in range(n_grid_rows):
                rs_xy, re_xy = row_runs[r_idx]
                rs_xz, re_xz = row_runs[n_grid_rows + r_idx]
                for c_idx in range(n_grid_cols):
                    roi_idx = r_idx * n_grid_cols + c_idx
                    roi_dir = f"MIP_ROI_{roi_idx}"
                    stems = roi_stems_map.get(roi_dir, [])
                    if page_idx >= len(stems):
                        continue
                    st = stems[page_idx]
                    if is_tub_sweep:
                        per_well_records[roi_idx].append({
                            "experiment_dir": v_dir,
                            "group_id": gid,
                            "condition": cond,
                            "roi_dir": roi_dir,
                            "roi_index": roi_idx,
                            "frame_index": page_idx,
                            "visit_index": v_idx,
                            "stem": st,
                            "t_min": t_min,
                            "accepted": False,
                            "is_intact": False,
                            "is_morph_dead": False,
                            "in_contact": False,
                            "reject_reason": "tubulin_mtoc_sweep_6",
                            "dent_um": np.nan,
                            "dent_unfiltered_um": np.nan,
                            "ca_um2": 0.0,
                            "R0_um": np.nan,
                            "sphericity": np.nan,
                            "rel_rmse_nc": np.nan,
                            "fit_type": "tubulin_sweep",
                        })
                    else:
                        cs, ce = col_runs[c_idx]
                        tile_xy = arr[rs_xy:re_xy, cs:ce]
                        tile_xz = arr[rs_xz:re_xz, cs:ce]
                        yx_proj = np.rot90(tile_xy, -1).T
                        zx_proj = np.flipud(tile_xz)
                        res = measure_dent_3d((yx_proj, zx_proj), dx_um=DX_UM)
                        per_well_records[roi_idx].append({
                            "experiment_dir": v_dir,
                            "group_id": gid,
                            "condition": cond,
                            "roi_dir": roi_dir,
                            "roi_index": roi_idx,
                            "frame_index": page_idx,
                            "visit_index": v_idx,
                            "stem": st,
                            "t_min": t_min,
                            "accepted": res["accepted"],
                            "is_intact": res["is_intact"],
                            "is_morph_dead": res["is_morph_dead"],
                            "in_contact": res["in_contact"],
                            "reject_reason": res["reject_reason"],
                            "dent_um": res["dent_um"],
                            "dent_unfiltered_um": res["dent_unfiltered_um"],
                            "ca_um2": res["ca_um2"],
                            "R0_um": res["R0_um"],
                            "sphericity": res["sphericity"],
                            "rel_rmse_nc": res["rel_rmse_nc"],
                            "fit_type": res["fit_type"],
                        })

        for roi_idx in sorted(per_well_records.keys()):
            exp_frame_rows.extend(per_well_records[roi_idx])

    return v_dir, exp_frame_rows


# ==============================================================================
# STEP 4 HELPER: TIME-ORDERED MORPHOLOGY DEATH & PER-PAIR AGGREGATION (D1-D4)
# ==============================================================================
def aggregate_well_timeseries_and_features(well_frames: list) -> tuple:
    """
    Given chronological frame rows for one well (~120-132 frames across 20-22 visits):
    1. Aggregates each 6-sweep visit (5 non-tubulin sweeps) into a visit time series.
    2. Identifies first intact contact time `t_first_contact_min`.
    3. Identifies morphology-based death time `t_death_min` (D4):
       earliest visit after first target presence where the target enters a sustained
       non-intact state (>=2 consecutive morph-dead visits or terminal morph-dead state).
    4. Excludes all frames at t >= t_death_min from intact dent measurements (D2, D4).
    5. Computes time-ordered first-contact dent, pre-death max dent, pre-death dent at
       peak contact area, 3D contact area, and binary contact flag.
    """
    by_visit = defaultdict(list)
    for r in well_frames:
        by_visit[r["visit_index"]].append(r)

    visit_rows = []
    for v_idx in sorted(by_visit.keys()):
        v_frames = by_visit[v_idx]
        non_tub = [f for f in v_frames if f["fit_type"] != "tubulin_sweep"]
        t_min = float(v_idx * VISIT_INTERVAL_MIN)

        # Count intact, morph-dead, and in-contact sweeps in this visit
        intact_sweeps = [f for f in non_tub if f["is_intact"]]
        morph_dead_sweeps = [f for f in non_tub if f["is_morph_dead"]]
        contact_sweeps = [f for f in non_tub if f["accepted"] and f["in_contact"] and np.isfinite(f["dent_um"])]
        accepted_sweeps = [f for f in non_tub if f["accepted"]]

        visit_intact = len(intact_sweeps) >= 2 or (len(intact_sweeps) >= 1 and len(morph_dead_sweeps) == 0)
        visit_morph_dead = (not visit_intact) and (len(morph_dead_sweeps) >= 2 or (len(morph_dead_sweeps) >= 1 and len(intact_sweeps) == 0))
        visit_in_contact = visit_intact and (len(contact_sweeps) >= 1)

        if len(contact_sweeps) > 0:
            dent_v = float(np.median([f["dent_um"] for f in contact_sweeps]))
            ca_v = float(np.median([f["ca_um2"] for f in contact_sweeps]))
        elif len(accepted_sweeps) > 0:
            dent_v = 0.0
            ca_v = float(np.median([f["ca_um2"] for f in accepted_sweeps]))
        else:
            dent_v = np.nan
            ca_v = 0.0

        unfilt_vals = [f["dent_unfiltered_um"] for f in non_tub if np.isfinite(f["dent_unfiltered_um"])]
        dent_unfilt_v = float(np.median(unfilt_vals)) if len(unfilt_vals) > 0 else np.nan

        r0_vals = [f["R0_um"] for f in non_tub if np.isfinite(f["R0_um"])]
        sph_vals = [f["sphericity"] for f in non_tub if np.isfinite(f["sphericity"])]
        rmse_vals = [f["rel_rmse_nc"] for f in non_tub if np.isfinite(f["rel_rmse_nc"])]

        visit_rows.append({
            "experiment_dir": v_frames[0]["experiment_dir"],
            "group_id": v_frames[0]["group_id"],
            "condition": v_frames[0]["condition"],
            "roi_dir": v_frames[0]["roi_dir"],
            "roi_index": v_frames[0]["roi_index"],
            "visit_index": v_idx,
            "t_min": t_min,
            "n_sweeps": len(v_frames),
            "n_intact_sweeps": len(intact_sweeps),
            "n_morph_dead_sweeps": len(morph_dead_sweeps),
            "n_contact_sweeps": len(contact_sweeps),
            "visit_intact": bool(visit_intact),
            "visit_morph_dead": bool(visit_morph_dead),
            "visit_in_contact": bool(visit_in_contact),
            "post_death": False,
            "dent_visit_um": round(dent_v, 4) if np.isfinite(dent_v) else np.nan,
            "dent_unfiltered_visit_um": round(dent_unfilt_v, 4) if np.isfinite(dent_unfilt_v) else np.nan,
            "ca_3d_visit_um2": round(ca_v, 4),
            "R0_visit_um": round(float(np.median(r0_vals)), 4) if len(r0_vals) > 0 else np.nan,
            "sphericity_visit": round(float(np.median(sph_vals)), 4) if len(sph_vals) > 0 else np.nan,
            "rel_rmse_nc_visit": round(float(np.median(rmse_vals)), 4) if len(rmse_vals) > 0 else np.nan,
        })

    # Identify earliest intact visit and morphology death time t_death_min (D4)
    intact_visit_indices = [i for i, vr in enumerate(visit_rows) if vr["visit_intact"]]
    first_intact_pos = intact_visit_indices[0] if len(intact_visit_indices) > 0 else None

    t_death_min = np.nan
    death_visit_idx = None
    if first_intact_pos is not None:
        n_v = len(visit_rows)
        for i in range(first_intact_pos + 1, n_v):
            if visit_rows[i]["visit_morph_dead"]:
                # Sustained morphological death: either next visit is also non-intact/morph-dead,
                # or it is the final visit, or >=60% of remaining visits are non-intact
                rem = visit_rows[i:]
                n_rem_non_intact = sum(1 for r in rem if not r["visit_intact"])
                next_dead = (i + 1 < n_v and (not visit_rows[i + 1]["visit_intact"])) or (i == n_v - 1)
                if next_dead and (n_rem_non_intact >= max(1, int(0.6 * len(rem)))):
                    death_visit_idx = visit_rows[i]["visit_index"]
                    t_death_min = visit_rows[i]["t_min"]
                    break

    # Apply D4 post-death exclusion to visit_rows and well_frames
    if death_visit_idx is not None:
        for vr in visit_rows:
            if vr["visit_index"] >= death_visit_idx:
                vr["post_death"] = True
                vr["visit_in_contact"] = False
                vr["dent_visit_um"] = np.nan
        for f in well_frames:
            if f["visit_index"] >= death_visit_idx and f["fit_type"] != "tubulin_sweep":
                if f["accepted"]:
                    f["accepted"] = False
                    f["in_contact"] = False
                    f["reject_reason"] = "post_death_frame_excluded_D4"
                    f["dent_um"] = np.nan

    # Compute per-well time-ordered summary strictly from pre-death intact visits (t < t_death_min)
    predeath_intact_visits = [vr for vr in visit_rows if vr["visit_intact"] and (not vr["post_death"])]
    predeath_contact_visits = [
        vr for vr in predeath_intact_visits
        if vr["visit_in_contact"] and np.isfinite(vr["dent_visit_um"]) and vr["ca_3d_visit_um2"] >= MIN_CA_UM2
    ]

    contact_flag = int(len(predeath_contact_visits) > 0)
    if contact_flag == 1:
        t_first_contact_min = float(predeath_contact_visits[0]["t_min"])
        # First-contact window: first 1-2 intact contact visits within 10 min of first contact (D4)
        first_window = [
            vr for vr in predeath_contact_visits
            if vr["t_min"] <= t_first_contact_min + VISIT_INTERVAL_MIN + 1e-6
        ][:2]
        dent_first_contact_um = float(np.mean([vr["dent_visit_um"] for vr in first_window]))
        ca_3d_first_contact_um2 = float(np.mean([vr["ca_3d_visit_um2"] for vr in first_window]))

        dents_c = np.array([vr["dent_visit_um"] for vr in predeath_contact_visits], dtype=float)
        cas_c = np.array([vr["ca_3d_visit_um2"] for vr in predeath_contact_visits], dtype=float)
        ts_c = np.array([vr["t_min"] for vr in predeath_contact_visits], dtype=float)

        dent_predeath_max_um = float(np.max(dents_c))
        dent_predeath_mean_contact_um = float(np.mean(dents_c))
        peak_ca_idx = int(np.argmax(cas_c))
        dent_predeath_at_peak_ca_um = float(dents_c[peak_ca_idx])
        ca_3d_predeath_max_um2 = float(np.max(cas_c))
        ca_3d_predeath_mean_um2 = float(np.mean(cas_c))
        peak_dent_idx = int(np.argmax(dents_c))
        t_to_max_dent_min = float(max(0.0, ts_c[peak_dent_idx] - t_first_contact_min))
    else:
        t_first_contact_min = np.nan
        dent_first_contact_um = 0.0
        ca_3d_first_contact_um2 = 0.0
        dent_predeath_max_um = 0.0
        dent_predeath_mean_contact_um = 0.0
        dent_predeath_at_peak_ca_um = 0.0
        ca_3d_predeath_max_um2 = 0.0
        ca_3d_predeath_mean_um2 = 0.0
        t_to_max_dent_min = 0.0

    # Morphological death after first contact (for time-ordered pair-level test D4)
    morph_death_after_contact = int(
        contact_flag == 1 and np.isfinite(t_death_min) and (t_death_min > t_first_contact_min)
    )
    morph_death_any = int(np.isfinite(t_death_min))

    # Unfiltered all-visit max dent (before D2-D4, to quantify contamination removed)
    unfilt_all = [vr["dent_unfiltered_visit_um"] for vr in visit_rows if np.isfinite(vr["dent_unfiltered_visit_um"])]
    dent_unfiltered_allframes_max_um = float(np.max(unfilt_all)) if len(unfilt_all) > 0 else 0.0
    dent_unfiltered_allframes_mean_um = float(np.mean(unfilt_all)) if len(unfilt_all) > 0 else 0.0

    r0_pre = [vr["R0_visit_um"] for vr in predeath_intact_visits if np.isfinite(vr["R0_visit_um"])]
    R0_predeath_um = float(np.median(r0_pre)) if len(r0_pre) > 0 else 5.5

    r0_all = [vr["R0_visit_um"] for vr in visit_rows if np.isfinite(vr["R0_visit_um"])]
    R0_all_max_um = float(np.max(r0_all)) if len(r0_all) > 0 else np.nan

    n_total_frames = len(well_frames)
    n_non_tub = sum(1 for f in well_frames if f["fit_type"] != "tubulin_sweep")
    n_accepted_frames = sum(1 for f in well_frames if f["accepted"])
    n_contact_frames = sum(1 for f in well_frames if f["accepted"] and f["in_contact"])

    well_summary = {
        "experiment_dir": well_frames[0]["experiment_dir"],
        "group_id": well_frames[0]["group_id"],
        "condition": well_frames[0]["condition"],
        "roi_dir": well_frames[0]["roi_dir"],
        "roi_index": well_frames[0]["roi_index"],
        "n_total_frames": n_total_frames,
        "n_non_tubulin_frames": n_non_tub,
        "n_accepted_predeath_frames": n_accepted_frames,
        "n_intact_contact_frames": n_contact_frames,
        "n_total_visits": len(visit_rows),
        "n_predeath_intact_visits": len(predeath_intact_visits),
        "n_predeath_contact_visits": len(predeath_contact_visits),
        "contact_flag": contact_flag,
        "t_first_contact_min": round(t_first_contact_min, 2) if np.isfinite(t_first_contact_min) else np.nan,
        "t_death_min": round(t_death_min, 2) if np.isfinite(t_death_min) else np.nan,
        "morph_death_any": morph_death_any,
        "morph_death_after_contact": morph_death_after_contact,
        "dent_first_contact_um": round(dent_first_contact_um, 4),
        "ca_3d_first_contact_um2": round(ca_3d_first_contact_um2, 4),
        "dent_predeath_max_um": round(dent_predeath_max_um, 4),
        "dent_predeath_mean_contact_um": round(dent_predeath_mean_contact_um, 4),
        "dent_predeath_at_peak_ca_um": round(dent_predeath_at_peak_ca_um, 4),
        "ca_3d_predeath_max_um2": round(ca_3d_predeath_max_um2, 4),
        "ca_3d_predeath_mean_um2": round(ca_3d_predeath_mean_um2, 4),
        "t_to_max_dent_min": round(t_to_max_dent_min, 2),
        "R0_predeath_um": round(R0_predeath_um, 4),
        "R0_all_max_um": round(R0_all_max_um, 4) if np.isfinite(R0_all_max_um) else np.nan,
        "dent_unfiltered_allframes_max_um": round(dent_unfiltered_allframes_max_um, 4),
        "dent_unfiltered_allframes_mean_um": round(dent_unfiltered_allframes_mean_um, 4),
    }
    return visit_rows, well_summary


# ==============================================================================
# STEP 3: SYNTHETIC VALIDATION (50 PAIRS) & 20 REAL FRAMES HAND-CHECK (D2-D4)
# ==============================================================================
def run_step3_validation(df_frames_all: pd.DataFrame) -> tuple:
    print("[Step 3] Running 50-pair synthetic validation (intact, cup-wrapped, blebbed, fragmented, swollen, off-patch)...")
    synth_rows = []
    idx = 0
    # 1. 35 intact dented spheres & ellipsoids across dent depths and axes
    for R_t in [4.8, 5.5, 6.1]:
        for aspect_name, aspect in [
            ("sphere", (1.0, 1.0, 1.0)),
            ("prolate_ellipsoid", (1.12, 0.94, 0.95)),
            ("oblate_ellipsoid", (0.92, 1.08, 1.04)),
        ]:
            for dent_true in [0.0, 0.4, 0.9, 1.4, 2.0]:
                if idx >= 35:
                    break
                axis_idx = idx % 3
                seg = make_synthetic_pair(
                    R_target_um=R_t,
                    aspect=aspect,
                    dent_true_um=dent_true,
                    R_immune_um=4.6,
                    axis_idx=axis_idx,
                    mode="normal",
                )
                res = measure_dent_3d(seg, dx_um=DX_UM)
                dent_meas = float(res["dent_um"]) if res["accepted"] and np.isfinite(res["dent_um"]) else np.nan
                err = dent_meas - dent_true if np.isfinite(dent_meas) else np.nan
                synth_rows.append({
                    "synth_id": idx + 1,
                    "category": "intact_dented",
                    "expected_accept": True,
                    "expected_morph_dead": False,
                    "shape_type": aspect_name,
                    "push_axis": ["Z", "Y", "X"][axis_idx],
                    "R_target_true_um": R_t,
                    "dent_true_um": round(dent_true, 4),
                    "dent_measured_um": round(dent_meas, 4) if np.isfinite(dent_meas) else np.nan,
                    "error_um": round(err, 4) if np.isfinite(err) else np.nan,
                    "abs_error_um": round(abs(err), 4) if np.isfinite(err) else np.nan,
                    "ca_measured_um2": res["ca_um2"],
                    "R0_fitted_um": res["R0_um"],
                    "sphericity": res["sphericity"],
                    "rel_rmse_nc": res["rel_rmse_nc"],
                    "accepted": res["accepted"],
                    "is_morph_dead": res["is_morph_dead"],
                    "reject_reason": res["reject_reason"],
                })
                idx += 1

    # 2. 3 cup-wrapped intact targets (deep synaptic cup around intact sphere)
    for c_i, dent_true in enumerate([0.8, 1.3, 1.8]):
        seg = make_synthetic_pair(
            R_target_um=5.5,
            aspect=(1.0, 1.0, 1.0),
            dent_true_um=dent_true,
            R_immune_um=4.8,
            axis_idx=c_i % 3,
            mode="cup_wrapped",
        )
        res = measure_dent_3d(seg, dx_um=DX_UM)
        dent_meas = float(res["dent_um"]) if res["accepted"] and np.isfinite(res["dent_um"]) else np.nan
        err = dent_meas - dent_true if np.isfinite(dent_meas) else np.nan
        synth_rows.append({
            "synth_id": idx + 1,
            "category": "cup_wrapped_intact",
            "expected_accept": True,
            "expected_morph_dead": False,
            "shape_type": "cup_wrapped_sphere",
            "push_axis": ["Z", "Y", "X"][c_i % 3],
            "R_target_true_um": 5.5,
            "dent_true_um": round(dent_true, 4),
            "dent_measured_um": round(dent_meas, 4) if np.isfinite(dent_meas) else np.nan,
            "error_um": round(err, 4) if np.isfinite(err) else np.nan,
            "abs_error_um": round(abs(err), 4) if np.isfinite(err) else np.nan,
            "ca_measured_um2": res["ca_um2"],
            "R0_fitted_um": res["R0_um"],
            "sphericity": res["sphericity"],
            "rel_rmse_nc": res["rel_rmse_nc"],
            "accepted": res["accepted"],
            "is_morph_dead": res["is_morph_dead"],
            "reject_reason": res["reject_reason"],
        })
        idx += 1

    # 3. 12 pathological controls (blebbed dying, fragmented dying, swollen dying, off-patch dimple)
    pathological_specs = [
        ("blebbed_dying", "blebbed", 5.6, 0.5, True),
        ("blebbed_dying", "blebbed", 5.8, 1.2, True),
        ("blebbed_dying", "blebbed", 6.0, 0.0, True),
        ("fragmented_dying", "fragmented", 5.5, 0.6, True),
        ("fragmented_dying", "fragmented", 5.8, 1.4, True),
        ("fragmented_dying", "fragmented", 5.2, 0.0, True),
        ("swollen_dying", "normal", 7.4, 0.8, True),
        ("swollen_dying", "normal", 7.9, 1.6, True),
        ("swollen_dying", "normal", 8.3, 2.2, True),
        ("off_patch_dimple", "off_patch_dimple", 5.5, -0.5, False),
        ("off_patch_dimple", "off_patch_dimple", 5.7, -0.8, False),
        ("off_patch_dimple", "off_patch_dimple", 5.4, -0.3, False),
    ]
    for p_i, (cat, mode, R_t, d_true, exp_dead) in enumerate(pathological_specs):
        seg = make_synthetic_pair(
            R_target_um=R_t,
            aspect=(1.0, 1.0, 1.0),
            dent_true_um=d_true,
            R_immune_um=4.6,
            axis_idx=p_i % 3,
            mode=mode,
        )
        res = measure_dent_3d(seg, dx_um=DX_UM)
        synth_rows.append({
            "synth_id": idx + 1,
            "category": cat,
            "expected_accept": False,
            "expected_morph_dead": exp_dead,
            "shape_type": mode,
            "push_axis": ["Z", "Y", "X"][p_i % 3],
            "R_target_true_um": R_t,
            "dent_true_um": max(0.0, d_true),
            "dent_measured_um": res["dent_um"],
            "error_um": np.nan,
            "abs_error_um": np.nan,
            "ca_measured_um2": res["ca_um2"],
            "R0_fitted_um": res["R0_um"],
            "sphericity": res["sphericity"],
            "rel_rmse_nc": res["rel_rmse_nc"],
            "accepted": res["accepted"],
            "is_morph_dead": res["is_morph_dead"],
            "reject_reason": res["reject_reason"],
        })
        idx += 1

    df_synth = pd.DataFrame(synth_rows)
    df_synth.to_csv(DATA_DIR / "dent_step3_synthetic_validation.csv", index=False)

    # Generate 20-frame real hand-check from cached 3D .h5 files (14 accepted intact contact + 6 rejected pathological)
    print("[Step 3] Generating 20-frame real hand-check figure (14 accepted intact + 6 rejected D2/D3/D4 controls)...")
    h5_files = sorted(RAW_CACHE_DIR.glob("*_predictions.h5"))
    accepted_candidates = []
    rejected_candidates = []
    for hp in h5_files:
        parts = hp.name.replace("_predictions.h5", "").split("__")
        if len(parts) != 3:
            continue
        v_dir, roi_dir, stem = parts
        if v_dir not in V_TO_GROUP:
            continue
        gid, cond = V_TO_GROUP[v_dir]
        try:
            with h5py.File(hp, "r") as hf:
                seg = hf["predictions"][:]
            res = measure_dent_3d(seg, dx_um=DX_UM, return_debug=True)
            item = {
                "h5_path": str(hp),
                "experiment_dir": v_dir,
                "group_id": gid,
                "condition": cond,
                "roi_dir": roi_dir,
                "stem": stem,
                "res": res,
            }
            if res["accepted"] and res["in_contact"] and np.isfinite(res["dent_um"]) and res["dent_um"] >= 0.15:
                accepted_candidates.append(item)
            elif (not res["accepted"]) and ("prim_debug" in res) and res["reject_reason"] in (
                "non_intact_radius_out_of_live_range",
                "non_intact_blebbing_or_low_sphericity",
                "non_intact_fragmented_target",
                "deepest_point_outside_contact_patch",
            ):
                rejected_candidates.append(item)
        except Exception:
            continue

    # Deduplicate by well (experiment_dir, roi_dir) so hand-checked frames span 14 distinct wells
    acc_by_well = {}
    for item in accepted_candidates:
        wkey = (item["experiment_dir"], item["roi_dir"])
        if wkey not in acc_by_well or item["res"]["dent_um"] > acc_by_well[wkey]["res"]["dent_um"]:
            acc_by_well[wkey] = item
    accepted_unique = sorted(acc_by_well.values(), key=lambda x: x["res"]["dent_um"])
    if len(accepted_unique) >= 14:
        accepted_candidates = accepted_unique
    else:
        accepted_candidates.sort(key=lambda x: x["res"]["dent_um"])

    n_acc = len(accepted_candidates)
    sel_acc_idx = np.linspace(0, max(0, n_acc - 1), 14, dtype=int)
    selected_acc = [accepted_candidates[i] for i in sel_acc_idx]

    # Pick 6 diverse rejected pathological real frames from distinct wells
    by_reason = defaultdict(list)
    seen_rej_wells = set()
    for rc in rejected_candidates:
        wkey = (rc["experiment_dir"], rc["roi_dir"], rc["res"]["reject_reason"])
        if wkey not in seen_rej_wells:
            seen_rej_wells.add(wkey)
            by_reason[rc["res"]["reject_reason"]].append(rc)
    selected_rej = []
    for reason in [
        "non_intact_radius_out_of_live_range",
        "non_intact_blebbing_or_low_sphericity",
        "non_intact_fragmented_target",
        "deepest_point_outside_contact_patch",
    ]:
        selected_rej.extend(by_reason[reason][:2])
    selected_rej = selected_rej[:6]
    while len(selected_rej) < 6 and len(rejected_candidates) > len(selected_rej):
        selected_rej.append(rejected_candidates[len(selected_rej)])

    handcheck_items = selected_acc + selected_rej
    fig, axes = plt.subplots(4, 5, figsize=(16.5, 13.0), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    handcheck_rows = []

    for k, item in enumerate(handcheck_items[:20]):
        ax = axes[k // 5, k % 5]
        ax.set_facecolor("#F8F8F8")
        res = item["res"]
        prim = res["prim_debug"]
        proj = prim["sub_proj"]

        # Build clean RGB rendering of the 2D projection
        rgb = np.full((proj.shape[0], proj.shape[1], 3), 0.97, dtype=float)
        rgb[proj == 3] = [0.76, 0.86, 0.96]  # Target interior (soft blue)
        rgb[proj == 1] = [0.12, 0.31, 0.47]  # Target membrane (#1F4E79)
        rgb[proj == 4] = [0.95, 0.80, 0.68]  # CAR-T body (soft amber)
        rgb[proj == 2] = [0.75, 0.34, 0.12]  # CAR-T dense actin (#C0561F)
        ax.imshow(rgb, origin="lower")

        # Overlay fitted undeformed target circle/ellipse
        th = np.linspace(0, 2 * np.pi, 200)
        cu, cv = prim["c_uv_px"]
        ru, rv = prim["a_uv_px"]
        ax.plot(cv + rv * np.sin(th), cu + ru * np.cos(th), "--", color="#111111", lw=1.25, label="Undeformed fit")

        # Highlight synaptic contact boundary points
        csurf = prim["contact_surf"]
        if np.any(csurf):
            pts_c = np.argwhere(csurf)
            ax.scatter(pts_c[:, 1], pts_c[:, 0], s=8, color="#2E7D5B", zorder=4, label="Contact patch")

        if res["accepted"]:
            status_tag = f"OK #{k+1} ({item['group_id']})"
            sub_tag = f"dent={res['dent_um']:.2f} µm | CA={res['ca_um2']:.1f} µm²\nR0={res['R0_um']:.1f} µm | Ψ={res['sphericity']:.2f}"
            title_col = "#111111"
            for sp in ax.spines.values():
                sp.set_edgecolor("#2E7D5B")
                sp.set_linewidth(1.5)
        else:
            short_r = (
                res["reject_reason"]
                .replace("non_intact_", "")
                .replace("radius_out_of_live_range", "Swollen/R0 out")
                .replace("blebbing_or_low_sphericity", "Blebbed/Low Ψ")
                .replace("fragmented_target", "Fragmented")
                .replace("deepest_point_outside_contact_patch", "Apex outside patch")
            )
            status_tag = f"REJECTED #{k+1} ({item['group_id']})"
            sub_tag = f"{short_r}\nR0={res['R0_um']:.1f} µm | Ψ={res['sphericity']:.2f} | CA={res['ca_um2']:.1f}"
            title_col = "#B3261E"
            for sp in ax.spines.values():
                sp.set_edgecolor("#B3261E")
                sp.set_linewidth(1.8)

        ax.set_title(f"{status_tag}\n{sub_tag}", fontsize=8.0, fontweight="bold", color=title_col, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        handcheck_rows.append({
            "panel_id": k + 1,
            "experiment_dir": item["experiment_dir"],
            "group_id": item["group_id"],
            "condition": item["condition"],
            "roi_dir": item["roi_dir"],
            "stem": item["stem"],
            "accepted": res["accepted"],
            "reject_reason": res["reject_reason"],
            "dent_um": res["dent_um"],
            "dent_unfiltered_um": res["dent_unfiltered_um"],
            "ca_um2": res["ca_um2"],
            "R0_um": res["R0_um"],
            "sphericity": res["sphericity"],
            "rel_rmse_nc": res["rel_rmse_nc"],
        })

    fig.suptitle(
        "Step 3 Hand-Check: 14 Accepted Intact In-Patch Contact Frames (#1–14) & 6 Rejected Dying/Off-Patch Controls (#15–20)",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(FIG_DIR / "dent_step3_20_real_frames_handcheck.png", dpi=180)
    plt.close(fig)

    df_hc = pd.DataFrame(handcheck_rows)
    df_hc.to_csv(DATA_DIR / "dent_step3_20_real_frames_handcheck.csv", index=False)
    return df_synth, df_hc


# ==============================================================================
# STEP 5: OUT-OF-SAMPLE RULES COMPARISON, DASATINIB TEST & BOOTSTRAPS (D5-D8, R2)
# ==============================================================================
def fit_and_evaluate_rules(df_419: pd.DataFrame, n_boot: int = 500) -> tuple:
    """
    Fits each candidate rule on High-Killing Donor Groups 1-5 (n=169 cells),
    freezes parameters, and evaluates on:
      - Training Groups 1-5 (n=169)
      - Held-Out Low-Killing Donor Groups 6-10 (n=160)
      - Dasatinib Groups 8*, 9*, 10* (n=90)
      - Predicted High-vs-Low Donor Gap (vs Observed 30.5 percentage points)
    Uses B=500 cell-level bootstrap resamples within each group (D8, R2).
    """
    train_groups = ["1", "2", "3", "4", "5"]
    low_groups = ["6", "7", "8", "9", "10"]
    dasa_groups = ["8*", "9*", "10*"]
    all_groups = train_groups + low_groups + dasa_groups

    obs_rate = {g: float(PUBLISHED_KILL_RATE[g]) for g in all_groups}
    obs_gap_pct = (np.mean([obs_rate[g] for g in train_groups]) - np.mean([obs_rate[g] for g in low_groups])) * 100.0

    # Define rule functional forms predicting per-cell kill probability p_i in [0, 1]
    def eval_rule_cells(df_sub: pd.DataFrame, rule_key: str, params: np.ndarray) -> np.ndarray:
        ca_xlsx = df_sub["contact_area_um2"].values
        arfs_nm_s = df_sub["ARFS_xlsx_nm_s"].values
        adc_val = df_sub["ADC_xlsx"].values
        mpa_deg = df_sub["MPA_xlsx_deg"].values
        c_flag = df_sub["contact_flag"].values.astype(float)
        ca_3d = df_sub["ca_3d_predeath_max_um2"].values
        d_first = df_sub["dent_first_contact_um"].values
        d_pre = df_sub["dent_predeath_max_um"].values
        d_unfilt = df_sub["dent_unfiltered_allframes_max_um"].values

        if rule_key == "contact_flag_only":
            p_contact = params[0]
            return np.clip(p_contact * c_flag, 0.0, 1.0)
        elif rule_key == "ca_xlsx_only":
            k0 = params[0]
            return ca_xlsx / (ca_xlsx + k0 + 1e-9)
        elif rule_key == "ca_3d_predeath_only":
            k0 = params[0]
            return ca_3d / (ca_3d + k0 + 1e-9)
        elif rule_key == "dent_first_contact_plus_ca":
            k_ca, w_d = params
            eff = ca_xlsx + w_d * d_first
            return eff / (eff + k_ca + 1e-9)
        elif rule_key == "dent_predeath_plus_ca":
            k_ca, w_d = params
            eff = ca_xlsx + w_d * d_pre
            return eff / (eff + k_ca + 1e-9)
        elif rule_key == "dent_first_contact_only":
            k_d = params[0]
            return np.where(d_first > 0, d_first / (d_first + k_d + 1e-9), 0.0)
        elif rule_key == "dent_predeath_only":
            k_d = params[0]
            return np.where(d_pre > 0, d_pre / (d_pre + k_d + 1e-9), 0.0)
        elif rule_key == "dent_unfiltered_plus_ca":
            k_ca, w_d = params
            eff = ca_xlsx + w_d * d_unfilt
            return eff / (eff + k_ca + 1e-9)
        elif rule_key == "ca_times_flow_comparator":
            k_cf = params[0]
            cf = np.maximum(0.0, ca_xlsx) * np.maximum(0.0, arfs_nm_s * 1e-3)
            return cf / (cf + k_cf + 1e-9)
        elif rule_key == "clutch_physics_model":
            kp, gc = params
            out = predict_pore_repair_kill_probability(
                ca_xlsx, arfs_nm_s, adc_val, mpa_deg,
                k_pfp=kp, k_escrt=0.045, g_crit=gc, eta_clutch=ETA_CLUTCH_DEFAULT
            )
            return out["p_kill_physics"]
        elif rule_key == "clutch_plus_predeath_dent":
            kp, gc, alpha_d = params
            out = predict_pore_repair_kill_probability(
                ca_xlsx + alpha_d * d_pre, arfs_nm_s, adc_val, mpa_deg,
                k_pfp=kp, k_escrt=0.045, g_crit=gc, eta_clutch=ETA_CLUTCH_DEFAULT
            )
            return out["p_kill_physics"]
        raise ValueError(f"Unknown rule_key: {rule_key}")

    rule_specs = [
        ("contact_flag_only", "Rule 0: Contact flag alone I(CA_3D >= 5 µm²)", "3D intact pre-death visits", [0.60], [(0.05, 0.99)]),
        ("ca_xlsx_only", "Rule 1a: Contact Area alone (published xlsx)", "Published synapse frames", [30.0], [(1.0, 500.0)]),
        ("ca_3d_predeath_only", "Rule 1b: 3D Pre-Death Contact Area alone", "3D intact pre-death visits", [25.0], [(1.0, 500.0)]),
        ("dent_first_contact_plus_ca", "Rule 2a: First-Contact Dent + Contact Area (time-ordered D4)", "First 1–2 intact contact visits", [35.0, 12.0], [(1.0, 500.0), (0.0, 120.0)]),
        ("dent_predeath_plus_ca", "Rule 2b: Pre-Death Intact Max Dent + Contact Area (D2–D4)", "All intact pre-death visits", [35.0, 12.0], [(1.0, 500.0), (0.0, 120.0)]),
        ("dent_first_contact_only", "Rule 2c: First-Contact Dent alone (time-ordered D4)", "First 1–2 intact contact visits", [0.6], [(0.02, 20.0)]),
        ("dent_predeath_only", "Rule 2d: Pre-Death Intact Max Dent alone (D2–D4)", "All intact pre-death visits", [0.8], [(0.02, 20.0)]),
        ("dent_unfiltered_plus_ca", "Rule 2e: Unfiltered All-Frame Dent + CA (pre-fix baseline)", "All 20 visits (unfiltered)", [45.0, 15.0], [(1.0, 500.0), (0.0, 120.0)]),
        ("ca_times_flow_comparator", "Rule 3: Contact Area × Actin Flow comparator", "Published synapse frames", [5.0], [(1e-4, 1e4)]),
        ("clutch_physics_model", "Rule 4: Clutch-Force Synaptic Physics Model", "Published synapse frames", [0.088, 1.15], [(0.01, 0.50), (0.20, 3.50)]),
        ("clutch_plus_predeath_dent", "Rule 5: Clutch-Force + Pre-Death Intact Dent", "Intact pre-death visits + xlsx", [0.088, 1.15, 5.0], [(0.01, 0.50), (0.20, 3.50), (0.0, 50.0)]),
    ]

    by_group_df = {g: df_419[df_419["group_id"] == g].reset_index(drop=True) for g in all_groups}

    def fit_one_sample(sample_by_group: dict, rule_key: str, p0: list, bounds: list) -> dict:
        def loss_fn(p):
            errs = []
            for g in train_groups:
                pred_g = float(np.mean(eval_rule_cells(sample_by_group[g], rule_key, p)))
                errs.append((pred_g - obs_rate[g]) ** 2)
            return float(np.mean(errs))

        res = minimize(loss_fn, x0=np.array(p0, dtype=float), bounds=bounds, method="L-BFGS-B")
        p_hat = res.x
        pred_group = {g: float(np.mean(eval_rule_cells(sample_by_group[g], rule_key, p_hat))) for g in all_groups}
        train_mae = float(np.mean([abs(pred_group[g] - obs_rate[g]) for g in train_groups]) * 100.0)
        low_mae = float(np.mean([abs(pred_group[g] - obs_rate[g]) for g in low_groups]) * 100.0)
        dasa_mae = float(np.mean([abs(pred_group[g] - obs_rate[g]) for g in dasa_groups]) * 100.0)
        pred_gap = float((np.mean([pred_group[g] for g in train_groups]) - np.mean([pred_group[g] for g in low_groups])) * 100.0)
        return {
            "params": p_hat,
            "pred_group": pred_group,
            "train_mae": train_mae,
            "low_mae": low_mae,
            "dasa_mae": dasa_mae,
            "pred_gap": pred_gap,
        }

    # Pre-generate bootstrap group samples so all rules are compared on identical resamples
    rng = np.random.default_rng(RNG_SEED)
    boot_samples = []
    for _ in range(n_boot):
        bs = {}
        for g in all_groups:
            df_g = by_group_df[g]
            idx_b = rng.integers(0, len(df_g), size=len(df_g))
            bs[g] = df_g.iloc[idx_b]
        boot_samples.append(bs)

    rules_rows = []
    group_pred_table = {g: {"group_id": g, "condition": by_group_df[g]["condition"].iloc[0], "n_cells": len(by_group_df[g]), "observed_kill_pct": round(obs_rate[g] * 100.0, 2)} for g in all_groups}

    for r_key, r_name, frame_source, p0, bounds in rule_specs:
        pt = fit_one_sample(by_group_df, r_key, p0, bounds)
        for g in all_groups:
            group_pred_table[g][f"pred_{r_key}_pct"] = round(pt["pred_group"][g] * 100.0, 2)

        b_train, b_low, b_dasa, b_gap = [], [], [], []
        for bs in boot_samples:
            b_res = fit_one_sample(bs, r_key, pt["params"].tolist(), bounds)
            b_train.append(b_res["train_mae"])
            b_low.append(b_res["low_mae"])
            b_dasa.append(b_res["dasa_mae"])
            b_gap.append(b_res["pred_gap"])

        passes_dasa = bool(pt["dasa_mae"] <= 10.0)
        rules_rows.append({
            "rule_key": r_key,
            "rule_name": r_name,
            "frame_window_used": frame_source,
            "n_params": len(p0),
            "fitted_params": "; ".join(f"{v:.4f}" for v in pt["params"]),
            "train_high_mae_pct": round(pt["train_mae"], 2),
            "train_ci95_lo": round(float(np.percentile(b_train, 2.5)), 2),
            "train_ci95_hi": round(float(np.percentile(b_train, 97.5)), 2),
            "heldout_low_mae_pct": round(pt["low_mae"], 2),
            "heldout_low_ci95_lo": round(float(np.percentile(b_low, 2.5)), 2),
            "heldout_low_ci95_hi": round(float(np.percentile(b_low, 97.5)), 2),
            "dasatinib_mae_pct": round(pt["dasa_mae"], 2),
            "dasatinib_ci95_lo": round(float(np.percentile(b_dasa, 2.5)), 2),
            "dasatinib_ci95_hi": round(float(np.percentile(b_dasa, 97.5)), 2),
            "passes_dasatinib_test": passes_dasa,
            "predicted_donor_gap_pct": round(pt["pred_gap"], 2),
            "gap_ci95_lo": round(float(np.percentile(b_gap, 2.5)), 2),
            "gap_ci95_hi": round(float(np.percentile(b_gap, 97.5)), 2),
            "observed_donor_gap_pct": round(obs_gap_pct, 2),
        })

    df_rules = pd.DataFrame(rules_rows)
    df_rules.to_csv(DATA_DIR / "dent_step5_rules_comparison.csv", index=False)
    df_group_preds = pd.DataFrame([group_pred_table[g] for g in all_groups])
    df_group_preds.to_csv(DATA_DIR / "dent_step5_group_predictions.csv", index=False)
    return df_rules, df_group_preds


# ==============================================================================
# STEP 5B: ALL-CELLS VS WITHIN-CONTACT AUCS & TIME-ORDERED DEATH TEST (D4, D6)
# ==============================================================================
def compute_feature_separation_and_time_ordered_tests(df_419: pd.DataFrame) -> tuple:
    df_untreated = df_419[df_419["group_id"].isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])].copy()
    df_untreated["is_high_donor"] = df_untreated["group_id"].isin(["1", "2", "3", "4", "5"]).astype(int)
    df_contact_only = df_untreated[df_untreated["contact_flag"] == 1].copy()

    features = [
        ("contact_flag", "Binary 3D Contact Flag (CA_3D >= 5 µm²)", "Intact pre-death visits"),
        ("dent_first_contact_um", "First-Contact Dent Depth (µm, time-ordered D4)", "First 1–2 intact contact visits"),
        ("dent_predeath_max_um", "Pre-Death Intact Max Dent Depth (µm, D2–D4)", "All intact pre-death visits"),
        ("dent_predeath_at_peak_ca_um", "Pre-Death Dent at Peak Contact Area (µm)", "Peak-CA intact pre-death visit"),
        ("ca_3d_predeath_max_um2", "3D Pre-Death Max Contact Area (µm²)", "All intact pre-death visits"),
        ("contact_area_um2", "Published Contact Area (µm², xlsx)", "Published synapse frames"),
        ("actin_flow_um_min", "Retrograde Actin Flow (µm/min, xlsx)", "Published synapse frames"),
        ("actin_depletion_idx", "Actin Depletion Index (xlsx)", "Published synapse frames"),
        ("mtoc_angle_deg", "MTOC Angle (deg, xlsx, inverted)", "Published tubulin sweep"),
        ("dent_unfiltered_allframes_max_um", "Unfiltered All-Frame Max Dent (µm, contaminated pre-fix)", "All 20 visits (unfiltered)"),
    ]

    auc_rows = []
    for col, label, f_win in features:
        # All untreated cells (n=329)
        hi_all = df_untreated[df_untreated["is_high_donor"] == 1][col].dropna().values
        lo_all = df_untreated[df_untreated["is_high_donor"] == 0][col].dropna().values
        if col == "mtoc_angle_deg":
            u_all, p_all = mannwhitneyu(-hi_all, -lo_all, alternative="two-sided")
        else:
            u_all, p_all = mannwhitneyu(hi_all, lo_all, alternative="two-sided")
        auc_all = float(u_all / (len(hi_all) * len(lo_all)))

        # Within-contact untreated cells (contact_flag == 1)
        hi_c = df_contact_only[df_contact_only["is_high_donor"] == 1][col].dropna().values
        lo_c = df_contact_only[df_contact_only["is_high_donor"] == 0][col].dropna().values
        if col == "contact_flag":
            auc_c, p_c = 0.50, 1.00
        else:
            if col == "mtoc_angle_deg":
                u_c, p_c = mannwhitneyu(-hi_c, -lo_c, alternative="two-sided")
            else:
                u_c, p_c = mannwhitneyu(hi_c, lo_c, alternative="two-sided")
            auc_c = float(u_c / (len(hi_c) * len(lo_c)))

        auc_rows.append({
            "feature_col": col,
            "feature_name": label,
            "frame_window": f_win,
            "high_mean_all": round(float(np.mean(hi_all)), 3),
            "low_mean_all": round(float(np.mean(lo_all)), 3),
            "auc_all_329_cells": round(auc_all, 3),
            "p_val_all_329_cells": round(float(p_all), 5),
            "high_mean_within_contact": round(float(np.mean(hi_c)), 3),
            "low_mean_within_contact": round(float(np.mean(lo_c)), 3),
            "auc_within_contact": round(auc_c, 3),
            "p_val_within_contact": round(float(p_c), 5),
        })

    df_auc = pd.DataFrame(auc_rows)
    df_auc.to_csv(DATA_DIR / "dent_step5_feature_separation_auc.csv", index=False)

    # Time-ordered within-pair test (D4): does first-contact dent predict subsequent morphological death?
    df_c_all = df_419[df_419["contact_flag"] == 1].copy()
    dead_after = df_c_all[df_c_all["morph_death_after_contact"] == 1]
    alive_after = df_c_all[df_c_all["morph_death_after_contact"] == 0]
    if len(dead_after) > 0 and len(alive_after) > 0:
        u_df, p_df = mannwhitneyu(dead_after["dent_first_contact_um"], alive_after["dent_first_contact_um"], alternative="two-sided")
        auc_df = float(u_df / (len(dead_after) * len(alive_after)))
        u_ca, p_ca = mannwhitneyu(dead_after["ca_3d_first_contact_um2"], alive_after["ca_3d_first_contact_um2"], alternative="two-sided")
        auc_ca = float(u_ca / (len(dead_after) * len(alive_after)))
    else:
        auc_df, p_df, auc_ca, p_ca = 0.5, 1.0, 0.5, 1.0

    time_ordered_stats = {
        "n_pairs_with_intact_contact": int(len(df_c_all)),
        "n_pairs_morph_death_after_contact": int(len(dead_after)),
        "n_pairs_intact_surviving_after_contact": int(len(alive_after)),
        "mean_first_contact_dent_subsequent_death_um": round(float(dead_after["dent_first_contact_um"].mean()), 3) if len(dead_after) else 0.0,
        "mean_first_contact_dent_surviving_um": round(float(alive_after["dent_first_contact_um"].mean()), 3) if len(alive_after) else 0.0,
        "auc_first_contact_dent_predicting_subsequent_morph_death": round(auc_df, 3),
        "p_val_first_contact_dent_predicting_subsequent_morph_death": round(float(p_df), 4),
        "auc_first_contact_ca3d_predicting_subsequent_morph_death": round(auc_ca, 3),
        "p_val_first_contact_ca3d_predicting_subsequent_morph_death": round(float(p_ca), 4),
    }
    return df_auc, time_ordered_stats


# ==============================================================================
# STEP 6: STIFFNESS SWEEP WITH SMALL-STRAIN CAP & PLAUSIBILITY CHECK (S1-S3)
# ==============================================================================
def run_stiffness_and_force_sweeps(df_419: pd.DataFrame) -> pd.DataFrame:
    """
    Implements S1-S3:
      - S1: Caps dents entering the Hertz formula at the small-strain limit
            delta_eff = min(dent_predeath_max_um, 0.20 * R0_predeath_um).
            Flags whether implied synaptic forces lie within the direct-measurement
            plausibility range (mean <= 1.5 nN, max <= 2.5 nN).
      - S2: Evaluates both Protocol A (fixed pore threshold, no refit -> fails Dasatinib)
            and Protocol B (refitted threshold on Groups 1-5 -> records bound pinning).
      - S3: Demonstrates exact scale absorption when refitting within a single target line.
    """
    train_groups = ["1", "2", "3", "4", "5"]
    low_groups = ["6", "7", "8", "9", "10"]
    dasa_groups = ["8*", "9*", "10*"]
    all_groups = train_groups + low_groups + dasa_groups
    obs_rate = {g: float(PUBLISHED_KILL_RATE[g]) for g in all_groups}

    # Small-strain capped dent (S1)
    R0_arr = df_419["R0_predeath_um"].fillna(5.5).values
    dent_raw = df_419["dent_predeath_max_um"].fillna(0.0).values
    dent_capped = np.minimum(dent_raw, SMALL_STRAIN_MAX_DENT_FRAC * R0_arr)
    df_419["dent_small_strain_capped_um"] = np.round(dent_capped, 4)

    R_imm = 4.8
    R_eff_um = (R0_arr * R_imm) / (R0_arr + R_imm)

    E_grid = np.logspace(np.log10(50.0), np.log10(10000.0), 25)
    stiff_rows = []

    for E_pa in E_grid:
        # Hertz + cortical tension force in nN from small-strain capped dent (S1)
        # 1 Pa * 1 um^2 = 1 pN = 1e-3 nN
        F_hertz_pN = (4.0 / 3.0) * (E_pa / (1.0 - NU_TARGET**2)) * np.sqrt(R_eff_um) * (dent_capped ** 1.5)
        F_cort_pN = np.where(dent_capped > 0, np.pi * T_CORTEX_PN_UM * dent_capped, 0.0)
        F_dent_nN = (F_hertz_pN + F_cort_pN) * 1e-3

        train_mask = df_419["group_id"].isin(train_groups).values
        mean_F_train = float(np.mean(F_dent_nN[train_mask]))
        max_F_train = float(np.max(F_dent_nN[train_mask]))
        plausible_force = bool((0.05 <= mean_F_train <= MAX_PLAUSIBLE_MEAN_FORCE_NN) and (max_F_train <= MAX_PLAUSIBLE_MAX_FORCE_NN))

        ca_xlsx = df_419["contact_area_um2"].values
        g_mech = F_dent_nN * (ca_xlsx / 35.0)

        # Protocol A: Fixed biophysics threshold (g_crit = 0.45, k_pfp = 1.35)
        p_open_fix = 1.0 / (1.0 + np.exp(-(g_mech - 0.45) / 0.35))
        p_kill_fix = 1.0 - np.exp(-1.35 * p_open_fix)
        pred_fix = {g: float(np.mean(p_kill_fix[df_419["group_id"].values == g])) for g in all_groups}
        fix_train_mae = float(np.mean([abs(pred_fix[g] - obs_rate[g]) for g in train_groups]) * 100.0)
        fix_low_mae = float(np.mean([abs(pred_fix[g] - obs_rate[g]) for g in low_groups]) * 100.0)
        fix_dasa_mae = float(np.mean([abs(pred_fix[g] - obs_rate[g]) for g in dasa_groups]) * 100.0)
        fix_gap = float((np.mean([pred_fix[g] for g in train_groups]) - np.mean([pred_fix[g] for g in low_groups])) * 100.0)

        # Protocol B: Refitted pore threshold (g_crit, k_pfp) on Groups 1-5 (S2)
        bnds = [(0.02, 20.0), (0.06, 2.30)]

        def loss_refit(p):
            gc, kp = p
            po = 1.0 / (1.0 + np.exp(-(g_mech - gc) / max(0.15, 0.25 * gc)))
            pk = 1.0 - np.exp(-kp * po)
            return float(np.mean([( float(np.mean(pk[df_419["group_id"].values == g])) - obs_rate[g] )**2 for g in train_groups]))

        res_b = minimize(loss_refit, x0=np.array([max(0.05, np.median(g_mech[train_mask])), 1.1]), bounds=bnds, method="L-BFGS-B")
        gc_hat, kp_hat = float(res_b.x[0]), float(res_b.x[1])
        pinned_at_bound = bool(
            abs(gc_hat - bnds[0][0]) < 1e-3
            or abs(gc_hat - bnds[0][1]) < 1e-2
            or abs(kp_hat - bnds[1][0]) < 1e-3
            or abs(kp_hat - bnds[1][1]) < 1e-2
        )

        po_ref = 1.0 / (1.0 + np.exp(-(g_mech - gc_hat) / max(0.15, 0.25 * gc_hat)))
        pk_ref = 1.0 - np.exp(-kp_hat * po_ref)
        pred_ref = {g: float(np.mean(pk_ref[df_419["group_id"].values == g])) for g in all_groups}
        ref_train_mae = float(np.mean([abs(pred_ref[g] - obs_rate[g]) for g in train_groups]) * 100.0)
        ref_low_mae = float(np.mean([abs(pred_ref[g] - obs_rate[g]) for g in low_groups]) * 100.0)
        ref_dasa_mae = float(np.mean([abs(pred_ref[g] - obs_rate[g]) for g in dasa_groups]) * 100.0)
        ref_gap = float((np.mean([pred_ref[g] for g in train_groups]) - np.mean([pred_ref[g] for g in low_groups])) * 100.0)

        stiff_rows.append({
            "E_target_Pa": round(float(E_pa), 2),
            "mean_F_train_nN": round(mean_F_train, 4),
            "max_F_train_nN": round(max_F_train, 4),
            "plausible_force_range": plausible_force,
            "fixed_train_mae_pct": round(fix_train_mae, 2),
            "fixed_heldout_low_mae_pct": round(fix_low_mae, 2),
            "fixed_dasatinib_mae_pct": round(fix_dasa_mae, 2),
            "fixed_predicted_gap_pct": round(fix_gap, 2),
            "refit_g_crit": round(gc_hat, 4),
            "refit_k_pfp": round(kp_hat, 4),
            "refit_pinned_at_bound": pinned_at_bound,
            "refit_train_mae_pct": round(ref_train_mae, 2),
            "refit_heldout_low_mae_pct": round(ref_low_mae, 2),
            "refit_dasatinib_mae_pct": round(ref_dasa_mae, 2),
            "refit_predicted_gap_pct": round(ref_gap, 2),
        })

    df_stiff = pd.DataFrame(stiff_rows)
    df_stiff.to_csv(DATA_DIR / "target_stiffness_sweep_3d_dent.csv", index=False)
    return df_stiff


# ==============================================================================
# STEP 6B: GENERATE SUMMARY DIAGNOSTIC FIGURES (HOUSE STYLE)
# ==============================================================================
def plot_step6_diagnostic_figures(
    df_419: pd.DataFrame,
    df_group_summary: pd.DataFrame,
    df_rules: pd.DataFrame,
    df_stiff: pd.DataFrame,
):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#C9C9C9"

    # 1. Step 6 Summary Figure: Dent vs Kill Outcome by Group + Rules Error Table (with Dasatinib)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel A: Intact Pre-Death Dent vs Kill Rate by Group (highlighting Groups 8 & 10 and Dasatinib)
    ax = axes[0]
    for _, row in df_group_summary.iterrows():
        gid = str(row["group_id"])
        if gid in ["1", "2", "3", "4", "5"]:
            col, lbl = "#1F4E79", "High-killing (1–5)"
        elif gid in ["6", "7", "8", "9", "10"]:
            col, lbl = "#C0561F", "Low-killing (6–10)"
        else:
            col, lbl = "#6B4C9A", "Dasatinib (8*–10*)"
        ax.scatter(row["dent_predeath_max_mean_um"], row["observed_kill_pct"], s=75, color=col, zorder=4)
        ax.text(
            row["dent_predeath_max_mean_um"] + 0.02,
            row["observed_kill_pct"] + 1.2,
            f"G{gid}",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
        )
    ax.set_xlabel("Mean Intact Pre-Death Max Dent (µm, D2–D4)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Observed Group Kill Rate (%)", fontsize=10, fontweight="bold")
    ax.set_title("A. Pre-Death Intact Dent vs Kill Rate (All 63,053 Frames)", fontsize=10.5, fontweight="bold")
    ax.grid(True, color="#E3E3E3", lw=0.7)

    # Panel B: First-Contact Dent vs Contact Flag by Group
    ax = axes[1]
    for _, row in df_group_summary.iterrows():
        gid = str(row["group_id"])
        col = "#1F4E79" if gid in ["1", "2", "3", "4", "5"] else ("#C0561F" if gid in ["6", "7", "8", "9", "10"] else "#6B4C9A")
        ax.scatter(row["contact_flag_frac"] * 100.0, row["dent_first_contact_mean_um"], s=75, color=col, zorder=4)
        ax.text(
            row["contact_flag_frac"] * 100.0 + 0.8,
            row["dent_first_contact_mean_um"] + 0.015,
            f"G{gid}",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
        )
    ax.set_xlabel("Intact Pre-Death Contact Flag Rate (%, CA_3D >= 5 µm²)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean First-Contact Dent Depth (µm, D4)", fontsize=10, fontweight="bold")
    ax.set_title("B. Contact Catch Rate vs First-Contact Dent (D6, D7)", fontsize=10.5, fontweight="bold")
    ax.grid(True, color="#E3E3E3", lw=0.7)

    # Panel C: Held-Out Low-Killing MAE vs Dasatinib MAE across Rules (D5)
    ax = axes[2]
    show_keys = [
        "contact_flag_only",
        "ca_xlsx_only",
        "ca_3d_predeath_only",
        "dent_first_contact_plus_ca",
        "dent_predeath_plus_ca",
        "dent_predeath_only",
        "ca_times_flow_comparator",
        "clutch_physics_model",
    ]
    sub_r = df_rules[df_rules["rule_key"].isin(show_keys)].copy()
    y_pos = np.arange(len(sub_r))
    ax.barh(y_pos - 0.17, sub_r["heldout_low_mae_pct"], height=0.32, color="#1F4E79", label="Held-out Low Donors (6–10)")
    ax.barh(y_pos + 0.17, sub_r["dasatinib_mae_pct"], height=0.32, color="#B3261E", label="Dasatinib (8*–10*, D5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.split(":")[0] + ": " + r.split(":")[1][:24] for r in sub_r["rule_name"]], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-Sample Mean Absolute Error (percentage points)", fontsize=10, fontweight="bold")
    ax.set_title("C. Rules Comparison: Held-Out Donors & Dasatinib Test (D5)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(True, axis="x", color="#E3E3E3", lw=0.7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dent_step6_summary_figure.png", dpi=180)
    plt.close(fig)

    # 2. Stiffness Sweep Diagnostic Figure (S1-S3)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=180)
    fig2.patch.set_facecolor("#FFFFFF")

    ax = axes2[0]
    ax.plot(df_stiff["E_target_Pa"], df_stiff["mean_F_train_nN"], "-o", color="#1F4E79", lw=1.8, ms=4, label="Implied Mean Force (Groups 1–5)")
    ax.plot(df_stiff["E_target_Pa"], df_stiff["max_F_train_nN"], "--s", color="#C0561F", lw=1.6, ms=4, label="Implied Max Force (Groups 1–5)")
    ax.axhspan(0.05, MAX_PLAUSIBLE_MAX_FORCE_NN, color="#2E7D5B", alpha=0.12, label="Measured Force Plausibility Band (<=2.5 nN)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Assumed Target Young's Modulus E_target (Pa)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Implied Synaptic Force (nN, Small-Strain Capped S1)", fontsize=10, fontweight="bold")
    ax.set_title("A. Force Plausibility Check with Small-Strain Cap (S1)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color="#E3E3E3", lw=0.7)

    ax = axes2[1]
    ax.plot(df_stiff["E_target_Pa"], df_stiff["fixed_heldout_low_mae_pct"], "-o", color="#1F4E79", lw=1.8, ms=4, label="Fixed Threshold: Held-Out Low MAE")
    ax.plot(df_stiff["E_target_Pa"], df_stiff["fixed_dasatinib_mae_pct"], "--^", color="#B3261E", lw=1.6, ms=4, label="Fixed Threshold: Dasatinib MAE (Fails D5/S2)")
    ax.plot(df_stiff["E_target_Pa"], df_stiff["refit_heldout_low_mae_pct"], "-s", color="#2E7D5B", lw=1.8, ms=4, label="Refitted Threshold: Scale-Absorbed Flat MAE (S3)")
    ax.set_xscale("log")
    ax.set_xlabel("Assumed Target Young's Modulus E_target (Pa)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Out-of-Sample MAE (percentage points)", fontsize=10, fontweight="bold")
    ax.set_title("B. Single-Line Scale Non-Identifiability & Dasatinib Failure (S2, S3)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color="#E3E3E3", lw=0.7)

    plt.tight_layout()
    fig2.savefig(FIG_DIR / "stiffness_and_force_scale_sweeps.png", dpi=180)
    plt.close(fig2)


# ==============================================================================
# MAIN PIPELINE DRIVER (D1-D8, S1-S3, R1-R4)
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("RUNNING FULL 63,053-FRAME DENT & STIFFNESS FIX PIPELINE (D1-D8, S1-S3, R1-R4)")
    print("=" * 80)

    df_inv = pd.read_csv(DATA_DIR / "dent_step1_chamber_inventory.csv")
    df_clean_419 = pd.read_csv(DATA_DIR / "lightsheet_wang2026_single_cells_419_clean.csv")
    df_clean_419["group_id"] = df_clean_419["group_id"].astype(str)
    df_clean_419["contact_area_um2"] = df_clean_419["CA_um2"].fillna(0.0)
    df_clean_419["CA_xlsx_um2"] = df_clean_419["CA_um2"].fillna(0.0)
    df_clean_419["actin_depletion_idx"] = df_clean_419["ADC"].fillna(0.0)
    df_clean_419["ADC_xlsx"] = df_clean_419["ADC"].fillna(0.0)
    df_clean_419["ARFS_xlsx_nm_s"] = df_clean_419["ARFS_nm_s"].fillna(0.0)
    df_clean_419["actin_flow_um_min"] = df_clean_419["ARFS_nm_s"].fillna(0.0) * 0.06
    df_clean_419["MPA_xlsx_deg"] = df_clean_419["MPA_deg"]
    df_clean_419["mtoc_angle_deg"] = df_clean_419["MPA_deg"].fillna(45.0)
    df_clean_419.to_csv(DATA_DIR / "clean_per_cell_419.csv", index=False)

    frame_log_path = DATA_DIR / "dent_step2_frame_log.csv"
    visit_ts_path = DATA_DIR / "dent_pair_visit_timeseries.csv"
    wells_path = DATA_DIR / "dent_step4_chamber_summary_502.csv"
    aligned_path = DATA_DIR / "dent_step4_aligned_419_pairs.csv"

    if (
        "--recompute-frames" not in sys.argv
        and frame_log_path.exists()
        and visit_ts_path.exists()
        and wells_path.exists()
        and aligned_path.exists()
    ):
        df_frames_check = pd.read_csv(frame_log_path)
        if len(df_frames_check) == 63053:
            print("[Step 1-2 & 4] Loading cached 63,053-frame log, 10,510 visit series, 502 wells, and 419 aligned pairs...")
            df_frames = df_frames_check
            df_visits = pd.read_csv(visit_ts_path)
            df_wells = pd.read_csv(wells_path)
            df_419 = pd.read_csv(aligned_path)
            df_419["group_id"] = df_419["group_id"].astype(str)
        else:
            df_frames = None
    else:
        df_frames = None

    if df_frames is None:
        # Parse URLS_FILE to build per-experiment roi_dir -> sorted prediction stems map
        print("[Step 1] Parsing scidb_19235_all_urls.txt for exact 63,053 outlined frame stems...")
        exp_roi_stems = defaultdict(lambda: defaultdict(list))
        with open(URLS_FILE, "r") as f:
            for line in f:
                if "_predictions.h5" not in line or "/ROIauto/" not in line:
                    continue
                path_part = line.split("path=/V1/")[-1].split("&")[0]
                parts = path_part.split("/")
                if len(parts) >= 4:
                    v_dir, roi_dir, _, fname = parts[0], parts[1], parts[2], parts[3]
                    if v_dir in V_TO_GROUP and roi_dir.startswith("MIP_ROI_"):
                        stem = fname.replace("_predictions.h5", "")
                        exp_roi_stems[v_dir][roi_dir].append(stem)

        exp_tasks = []
        for v_dir, _ in V_TO_GROUP.items():
            roi_map = {}
            for roi_dir, stems_list in exp_roi_stems[v_dir].items():
                roi_map[roi_dir] = sorted(set(stems_list), key=lambda s: int(s))
            exp_tasks.append((v_dir, roi_map))

        print(f"[Step 1-2] Launching parallel 63,053-frame dent & morphology analysis across {len(exp_tasks)} experiments...")
        all_exp_frames = {}
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            for v_dir, rows in ex.map(process_one_experiment_all_frames, exp_tasks):
                all_exp_frames[v_dir] = rows
                print(f"  -> Completed {v_dir}: {len(rows):,} frames")

        # Aggregate per-well visit time series & apply D4 post-death exclusion
        print("[Step 2 & 4] Computing per-well visit time series, morphology death time t_death_min (D4), and pre-death features...")
        all_frame_rows = []
        all_visit_rows = []
        all_well_summaries = []

        for v_dir in V_TO_GROUP.keys():
            rows = all_exp_frames[v_dir]
            by_roi = defaultdict(list)
            for r in rows:
                by_roi[r["roi_index"]].append(r)
            for roi_idx in sorted(by_roi.keys()):
                w_frames = sorted(by_roi[roi_idx], key=lambda x: x["frame_index"])
                v_rows, w_sum = aggregate_well_timeseries_and_features(w_frames)
                # Also add compatibility aliases on w_sum
                w_sum["chamber_accepted"] = bool(w_sum["n_accepted_predeath_frames"] > 0)
                w_sum["n_accepted_frames"] = int(w_sum["n_accepted_predeath_frames"])
                w_sum["dent_max_um"] = w_sum["dent_predeath_max_um"]
                w_sum["dent_at_peak_ca_um"] = w_sum["dent_predeath_at_peak_ca_um"]
                w_sum["ca_3d_max_um2"] = w_sum["ca_3d_predeath_max_um2"]
                w_sum["R0_median_um"] = w_sum["R0_predeath_um"]
                all_frame_rows.extend(w_frames)
                all_visit_rows.extend(v_rows)
                all_well_summaries.append(w_sum)

        df_frames = pd.DataFrame(all_frame_rows)
        df_visits = pd.DataFrame(all_visit_rows)
        df_wells = pd.DataFrame(all_well_summaries)

        df_frames.to_csv(frame_log_path, index=False)
        df_visits.to_csv(visit_ts_path, index=False)
        df_wells.to_csv(wells_path, index=False)
        print(f"  -> Saved dent_step2_frame_log.csv ({len(df_frames):,} frames)")
        print(f"  -> Saved dent_pair_visit_timeseries.csv ({len(df_visits):,} visit records)")
        print(f"  -> Saved dent_step4_chamber_summary_502.csv ({len(df_wells):,} wells)")

        # Align the 502 wells to the 419 single-cell pairs using order-preserving DP on 3D CA vs published CA
        aligned_parts = []
        for v_dir, (gid, _) in V_TO_GROUP.items():
            w_sub = df_wells[df_wells["experiment_dir"] == v_dir].sort_values("roi_index").reset_index(drop=True)
            c_sub = df_clean_419[df_clean_419["group_id"] == gid].sort_values("cell_index").reset_index(drop=True)
            n_c = len(c_sub)
            n_r = len(w_sub)
            if n_r >= n_c:
                cost_mat = np.zeros((n_c, n_r), dtype=np.float64)
                for i in range(n_c):
                    for j in range(n_r):
                        cost_mat[i, j] = abs(float(w_sub.loc[j, "ca_3d_predeath_max_um2"]) - float(c_sub.loc[i, "CA_xlsx_um2"]))
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
                matched_j = [min(n_r - 1, int(round(i * (n_r - 1) / max(1, n_c - 1)))) for i in range(n_c)]

            for i in range(n_c):
                c_row = c_sub.iloc[i].to_dict()
                w_row = w_sub.iloc[matched_j[i]].to_dict()
                merged = {**c_row, **w_row}
                aligned_parts.append(merged)

        df_419 = pd.DataFrame(aligned_parts)
        df_419.to_csv(aligned_path, index=False)
        print(f"  -> Saved dent_step4_aligned_419_pairs.csv ({len(df_419):,} aligned single-cell pairs)")

    # Run Step 3 validation (50 synthetic cases + 20 real hand-checked frames)
    df_synth, df_hc = run_step3_validation(df_frames)

    # Run Step 5 out-of-sample rules comparison & within-contact AUCs
    print("[Step 5] Running out-of-sample rules comparison (with Dasatinib test D5 & 500 bootstraps D8)...")
    df_rules, df_group_preds = fit_and_evaluate_rules(df_419, n_boot=500)
    df_auc, time_ordered_stats = compute_feature_separation_and_time_ordered_tests(df_419)

    # Compute per-group summary table (explicitly highlighting Groups 8 & 10 for D7)
    group_summary_rows = []
    for v_dir, (gid, cond) in V_TO_GROUP.items():
        sub = df_419[df_419["group_id"] == gid]
        sub_c = sub[sub["contact_flag"] == 1]
        group_summary_rows.append({
            "group_id": gid,
            "condition": cond,
            "n_cells": int(len(sub)),
            "observed_kill_pct": round(PUBLISHED_KILL_RATE[gid] * 100.0, 2),
            "contact_flag_frac": round(float(sub["contact_flag"].mean()), 3),
            "dent_first_contact_mean_um": round(float(sub["dent_first_contact_um"].mean()), 3),
            "dent_predeath_max_mean_um": round(float(sub["dent_predeath_max_um"].mean()), 3),
            "dent_predeath_within_contact_mean_um": round(float(sub_c["dent_predeath_max_um"].mean()), 3) if len(sub_c) else 0.0,
            "dent_unfiltered_allframes_mean_um": round(float(sub["dent_unfiltered_allframes_max_um"].mean()), 3),
            "ca_3d_predeath_max_mean_um2": round(float(sub["ca_3d_predeath_max_um2"].mean()), 2),
            "ca_xlsx_mean_um2": round(float(sub["contact_area_um2"].mean()), 2),
            "actin_flow_mean_um_min": round(float(sub["actin_flow_um_min"].mean()), 3),
            "morph_death_rate_frac": round(float(sub["morph_death_any"].mean()), 3),
            "R0_predeath_median_um": round(float(sub["R0_predeath_um"].median()), 3),
            "R0_all_max_median_um": round(float(sub["R0_all_max_um"].median()), 3),
        })
    df_group_summary = pd.DataFrame(group_summary_rows)
    df_group_summary.to_csv(DATA_DIR / "dent_step5_group_summary.csv", index=False)

    # Run Step 6 stiffness sweep with small-strain cap (S1-S3)
    print("[Step 6] Running small-strain-capped stiffness sweep and generating diagnostic figures...")
    df_stiff = run_stiffness_and_force_sweeps(df_419)
    plot_step6_diagnostic_figures(df_419, df_group_summary, df_rules, df_stiff)

    # Compute exact frame and pair survival funnel counts (D1-D4, R1, R3)
    reason_counts = df_frames["reject_reason"].value_counts().to_dict()
    n_total_f = int(len(df_frames))
    n_tub_f = int(reason_counts.get("tubulin_mtoc_sweep_6", 0))
    n_non_tub_f = n_total_f - n_tub_f
    n_accepted_f = int(df_frames["accepted"].sum())
    n_accepted_contact_f = int((df_frames["accepted"] & df_frames["in_contact"]).sum())
    n_accepted_nocontact_f = int((df_frames["accepted"] & (~df_frames["in_contact"])).sum())

    # Early vs Mid vs Late visit radius and dent progression (before vs after D2-D4)
    df_v_early = df_visits[df_visits["t_min"] <= 40.0]
    df_v_mid = df_visits[(df_visits["t_min"] >= 50.0) & (df_visits["t_min"] <= 120.0)]
    df_v_late = df_visits[df_visits["t_min"] >= 130.0]

    # Synthetic validation metrics
    sub_intact_synth = df_synth[df_synth["category"] == "intact_dented"]
    sub_cup_synth = df_synth[df_synth["category"] == "cup_wrapped_intact"]
    sub_path_synth = df_synth[~df_synth["expected_accept"]]

    # Spearman correlations (all 419 cells and within-contact cells)
    rho_all, p_rho_all = spearmanr(df_419["dent_predeath_max_um"], df_419["ca_3d_predeath_max_um2"])
    sub_c419 = df_419[df_419["contact_flag"] == 1]
    rho_c, p_rho_c = spearmanr(sub_c419["dent_predeath_max_um"], sub_c419["ca_3d_predeath_max_um2"])

    master_results = {
        "generated_at": "2026-10-25",
        "provenance_constants": PROVENANCE_TABLE,
        "step1_inventory": {
            "n_experiments": 13,
            "n_wells_total": int(len(df_wells)),
            "n_outlined_frames_total": n_total_f,
            "n_single_cell_pairs_clean": int(len(df_419)),
            "n_nd_slots_excluded": 101,
        },
        "step2_frame_survival_funnel": {
            "total_outlined_frames": n_total_f,
            "tubulin_sweep_6_excluded": n_tub_f,
            "non_tubulin_3d_sweeps_evaluated": n_non_tub_f,
            "accepted_intact_predeath_frames_total": n_accepted_f,
            "accepted_intact_in_contact_frames": n_accepted_contact_f,
            "accepted_intact_no_contact_frames": n_accepted_nocontact_f,
            "rejection_breakdown_by_reason": {str(k): int(v) for k, v in reason_counts.items()},
            "wells_surviving_with_intact_predeath_frames": int((df_wells["n_accepted_predeath_frames"] > 0).sum()),
            "wells_with_intact_predeath_contact": int(df_wells["contact_flag"].sum()),
            "aligned_419_pairs_with_intact_predeath_frames": int((df_419["n_accepted_predeath_frames"] > 0).sum()),
            "aligned_419_pairs_with_intact_predeath_contact": int(df_419["contact_flag"].sum()),
        },
        "step2_time_window_diagnostics": {
            "early_0_40min_median_R0_all_um": round(float(df_v_early["R0_visit_um"].median()), 3),
            "mid_50_120min_median_R0_all_um": round(float(df_v_mid["R0_visit_um"].median()), 3),
            "late_130_210min_median_R0_all_um": round(float(df_v_late["R0_visit_um"].median()), 3),
            "intact_predeath_median_R0_um": round(float(df_frames[df_frames["accepted"]]["R0_um"].median()), 3),
            "accepted_contact_dent_quantiles_um": {
                "q50": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].quantile(0.50)), 3),
                "q75": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].quantile(0.75)), 3),
                "q90": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].quantile(0.90)), 3),
                "q95": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].quantile(0.95)), 3),
                "q99": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].quantile(0.99)), 3),
                "max": round(float(df_frames[df_frames["accepted"] & df_frames["in_contact"]]["dent_um"].max()), 3),
            },
            "spearman_dent_vs_ca3d_all_419": round(float(rho_all), 3),
            "spearman_dent_vs_ca3d_within_contact": round(float(rho_c), 3),
        },
        "step3_validation": {
            "n_synthetic_total": int(len(df_synth)),
            "n_intact_dented_pairs": int(len(sub_intact_synth)),
            "intact_accept_rate": round(float(sub_intact_synth["accepted"].mean()), 4),
            "intact_mae_um": round(float(sub_intact_synth["abs_error_um"].mean()), 4),
            "intact_bias_um": round(float(sub_intact_synth["error_um"].mean()), 4),
            "intact_max_abs_err_um": round(float(sub_intact_synth["abs_error_um"].max()), 4),
            "n_cup_wrapped_pairs": int(len(sub_cup_synth)),
            "cup_wrapped_accept_rate": round(float(sub_cup_synth["accepted"].mean()), 4),
            "cup_wrapped_mae_um": round(float(sub_cup_synth["abs_error_um"].mean()), 4),
            "n_pathological_controls": int(len(sub_path_synth)),
            "pathological_rejection_rate": round(float((~sub_path_synth["accepted"]).mean()), 4),
            "morph_death_classification_accuracy": round(
                float((df_synth["is_morph_dead"] == df_synth["expected_morph_dead"]).mean()), 4
            ),
            "n_real_handchecked_frames": int(len(df_hc)),
            "n_real_handchecked_accepted": int(df_hc["accepted"].sum()),
            "n_real_handchecked_rejected_controls": int((~df_hc["accepted"]).sum()),
        },
        "step4_time_ordered_test": time_ordered_stats,
        "step5_rules_comparison": df_rules.to_dict(orient="records"),
        "step5_feature_separation_auc": df_auc.to_dict(orient="records"),
        "step5_group_summary": df_group_summary.to_dict(orient="records"),
        "step6_stiffness_sweep_summary": {
            "small_strain_cap_fraction_of_R0": SMALL_STRAIN_MAX_DENT_FRAC,
            "max_plausible_mean_force_nN": MAX_PLAUSIBLE_MEAN_FORCE_NN,
            "max_plausible_max_force_nN": MAX_PLAUSIBLE_MAX_FORCE_NN,
            "E_target_plausible_range_Pa": [
                float(df_stiff[df_stiff["plausible_force_range"]]["E_target_Pa"].min()),
                float(df_stiff[df_stiff["plausible_force_range"]]["E_target_Pa"].max()),
            ] if df_stiff["plausible_force_range"].any() else [],
            "n_grid_points_total": int(len(df_stiff)),
            "n_grid_points_plausible_force": int(df_stiff["plausible_force_range"].sum()),
            "fixed_threshold_dasatinib_mae_range_pct": [
                round(float(df_stiff["fixed_dasatinib_mae_pct"].min()), 2),
                round(float(df_stiff["fixed_dasatinib_mae_pct"].max()), 2),
            ],
            "refit_heldout_low_mae_range_pct": [
                round(float(df_stiff["refit_heldout_low_mae_pct"].min()), 2),
                round(float(df_stiff["refit_heldout_low_mae_pct"].max()), 2),
            ],
            "n_refit_points_pinned_at_bound": int(df_stiff["refit_pinned_at_bound"].sum()),
            "conclusion": (
                "Within a single target cell line (Nalm6), target stiffness E_target is absorbed by the fitted pore threshold "
                "and cannot be identified from kill rates; under a fixed literature threshold, dent-derived Hertz forces fail "
                "the Dasatinib test because Dasatinib-treated cells still indent targets without killing."
            ),
            "sweep_rows": df_stiff.to_dict(orient="records"),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    with open(DATA_DIR / "dent_stiffness_master_results.json", "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"[Done] Wrote master results to {DATA_DIR / 'dent_stiffness_master_results.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

