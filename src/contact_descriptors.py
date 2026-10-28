#!/usr/bin/env python3
"""
Step 5a: Compute cheap, mask-based 3D shape descriptors of the T-cell/target contact
across ALL 502 chambers (52,545 actin sweeps across 10,510 visits in 13 experiments).

Constraint from Step 5 Prerequisite (`.agy_step5res_report.md`):
  - The 3D U-Net masks only retain surface features with spatial wavelength >= ~1.4 um
    (synaptic half-power cutoff lambda_50% = 1.38 um; away from contact 1.45 um; smallest
    trustworthy bump diameter >= ~0.7-1.0 um laterally and >= ~0.9-1.2 um axially).
  - All descriptors computed here operate strictly at or above this >= 1.4 um macro-scale
    (whole-cell geometry, synaptic patch axes, macro-cup depth/volume/asymmetry, and
    low-order Zernike-like modes up to order N <= 4 constrained by 2*r_patch / lambda >= 1.4 um).
  - High-frequency residual variance below the 1.4 um wavelength cutoff is reported ONLY
    as a discretization/smoothing noise check (`rough_below_cutoff_noise_um2`,
    `rough_below_cutoff_frac`) and never interpreted as nanoscale microvillar topography.

Axial pitch sensitivity (`z = 0.1962 um` vs `z = 0.2400 um`):
  - Target cells span ~19% fewer Z slices than XY pixels (`Delta Z / Delta XY = 0.807`),
    corresponding to either mild physical oblateness on the chamber floor or an effective
    axial step of ~0.24 um (`0.1962 / 0.807 = 0.243 um`).
  - All geometric descriptors are computed under both `z = 0.1962 um` (nominal, using
    `VOXEL_UM` from `src/constants.py`) and `z = 0.2400 um` (`_z024` columns), and
    descriptors whose median value across contact sweeps changes by > 10% are flagged.

Reuses:
  - `src/constants.py` (`VOXEL_UM`, `OPTICAL_FWHM_UM`)
  - `src/reproduce_contact_area_tracked.py` (tracked T cell `tracked_tcell_id`,
    `trace_num`, `s_largest`, `s_total`, `n_components`, and `instance_seg_matlab_xyz`)

Outputs:
  - `data/step5/per_sweep_descriptors.csv`
  - `data/step5/per_visit_descriptors.csv`
  - `data/step5/per_chamber_descriptors.csv`
  - `data/step5/descriptor_dictionary.csv`
  - `figures/step5_descriptors_overview.png`
  - `.agy_step5a_report.md`
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, label
from scipy.spatial import ConvexHull, cKDTree

try:
    from src.constants import OPTICAL_FWHM_UM, VOXEL_UM
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from constants import OPTICAL_FWHM_UM, VOXEL_UM
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from reproduce_contact_area import (
    CA_VOXEL_FACTOR_UM2,
    MANIFEST_CSV,
    STRUCT26,
    V_TO_GROUP,
    bbox_slices,
    bwareaopen_26,
    discover_complete_chambers,
    imfill_holes_26,
)
from reproduce_actin_measures import (
    SWEEPS_PER_VISIT,
    edt_indices_matlab_xyz,
    instance_seg_matlab_xyz,
)
import reproduce_contact_area_tracked as tracked_mod

STEP3_DIR: Path = DATA_DIR / "step3"
STEP3_TRACKED_DIR: Path = DATA_DIR / "step3_tracked"
STEP5_DIR: Path = DATA_DIR / "step5"
FIG_OVERVIEW_PATH: Path = FIGURES_DIR / "step5_descriptors_overview.png"
REPORT_PATH: Path = REPO_ROOT / ".agy_step5a_report.md"

Z_NOMINAL_UM: float = VOXEL_UM  # 0.1962 um
Z_ALT_UM: float = 0.2400        # 0.2400 um implied axial pitch
RESOLUTION_WAVELENGTH_CUTOFF_UM: float = 1.40
RESOLUTION_BUMP_CUTOFF_UM: float = 0.70

# 6-connected structuring element for exposed boundary faces
STRUCT6: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
STRUCT6[1, 1, :] = True
STRUCT6[1, :, 1] = True
STRUCT6[:, 1, 1] = True


# ==============================================================================
# DESCRIPTOR METADATA & DEFINITIONS (SCALE >= 1.4 um CONSTRAINT)
# ==============================================================================
DESCRIPTOR_SPECS: list[dict[str, str]] = [
    # 1. Contact patch area & shape
    {
        "name": "contact_area_um2",
        "category": "contact_patch",
        "units": "um^2",
        "scale": "macro_patch (>=1.4 um wavelength)",
        "is_noise_check": "False",
        "definition": "Contact surface area of largest 26-connected patch component between tracked T cell and target (s_largest * 0.0385 um^2 at nominal z, or surface-normal scaled at z=0.240 um).",
    },
    {
        "name": "contact_area_total_um2",
        "category": "contact_patch",
        "units": "um^2",
        "scale": "macro_patch (>=1.4 um wavelength)",
        "is_noise_check": "False",
        "definition": "Total contact surface area summed across all 26-connected contact patches (s_total * 0.0385 um^2).",
    },
    {
        "name": "patch_equiv_radius_um",
        "category": "contact_patch",
        "units": "um",
        "scale": "macro_patch (>=1.4 um wavelength)",
        "is_noise_check": "False",
        "definition": "Equivalent circular radius of the primary contact patch: sqrt(contact_area_um2 / pi).",
    },
    {
        "name": "patch_elongation",
        "category": "contact_patch",
        "units": "ratio (>=1)",
        "scale": "macro_patch (>=1.4 um wavelength)",
        "is_noise_check": "False",
        "definition": "Ratio of major to minor principal axes (sqrt(lambda_1 / lambda_2)) of the 3D contact patch projected onto the synaptic tangent plane.",
    },
    {
        "name": "patch_n_components",
        "category": "contact_patch",
        "units": "count",
        "scale": "macro_patch (>=1.4 um wavelength)",
        "is_noise_check": "False",
        "definition": "Number of 26-connected contact surface components between the tracked T cell and contacted target.",
    },
    # 2. Target indentation / synaptic cup geometry
    {
        "name": "cup_depth_hull_p95_um",
        "category": "target_indentation",
        "units": "um",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "95th percentile inward depth of the target contact patch relative to the target's 3D convex hull facets facing the T cell.",
    },
    {
        "name": "cup_depth_hull_max_um",
        "category": "target_indentation",
        "units": "um",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Maximum inward depth of the target contact patch relative to the target's 3D convex hull facets facing the T cell.",
    },
    {
        "name": "cup_depth_sphere_p95_um",
        "category": "target_indentation",
        "units": "um",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "95th percentile inward depth of the target contact patch relative to a regularized 3D sphere fit to the target's non-contact surface.",
    },
    {
        "name": "cup_volume_hull_um3",
        "category": "target_indentation",
        "units": "um^3",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Volume of the synaptic indentation cup integrated between the target contact patch and the target's 3D convex hull reference surface.",
    },
    {
        "name": "cup_volume_sphere_um3",
        "category": "target_indentation",
        "units": "um^3",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Volume of the synaptic indentation cup integrated between the target contact patch and the non-contact reference sphere.",
    },
    {
        "name": "cup_aspect_radius_over_depth",
        "category": "target_indentation",
        "units": "ratio",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Ratio of contact patch opening radius (patch_equiv_radius_um) to cup depth (cup_depth_hull_p95_um).",
    },
    {
        "name": "cup_depth_over_radius",
        "category": "target_indentation",
        "units": "ratio",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Dimensionless cup indentation aspect ratio: cup_depth_hull_p95_um / patch_equiv_radius_um.",
    },
    {
        "name": "cup_asymmetry_norm",
        "category": "target_indentation",
        "units": "dimensionless",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Lateral offset in the tangent plane between the deepest cup point (centroid of top 5% deepest contact points) and the contact patch centroid, normalized by patch_equiv_radius_um.",
    },
    # 3. Low-order target surface shape & roughness power split (cutoff lambda = 1.4 um)
    {
        "name": "zernike_defocus_um",
        "category": "low_order_surface_modes",
        "units": "um",
        "scale": "low_order_mode (wavelength >=1.4 um, N<=2)",
        "is_noise_check": "False",
        "definition": "Radial bowl/cup curvature coefficient (Z_2^0 = 2*rho^2 - 1) of target indentation height on the normalized patch disk (negative = concave inward cup at center).",
    },
    {
        "name": "zernike_tilt_amp_um",
        "category": "low_order_surface_modes",
        "units": "um",
        "scale": "low_order_mode (wavelength >=1.4 um, N=1)",
        "is_noise_check": "False",
        "definition": "First-order dipole tilt amplitude sqrt((Z_1^-1)^2 + (Z_1^1)^2) of target indentation across the contact patch.",
    },
    {
        "name": "zernike_astig_amp_um",
        "category": "low_order_surface_modes",
        "units": "um",
        "scale": "low_order_mode (wavelength >=1.4 um, N=2)",
        "is_noise_check": "False",
        "definition": "Second-order saddle/astigmatism amplitude sqrt((Z_2^-2)^2 + (Z_2^2)^2) of target indentation across the contact patch.",
    },
    {
        "name": "zernike_coma_amp_um",
        "category": "low_order_surface_modes",
        "units": "um",
        "scale": "low_order_mode (wavelength >=1.4 um, N=3 when 2*r_patch>=2.1 um)",
        "is_noise_check": "False",
        "definition": "Third-order asymmetric coma amplitude sqrt((Z_3^-1)^2 + (Z_3^1)^2) when allowed by the 1.4 um cutoff on patch diameter (0 when patch diameter < 2.1 um).",
    },
    {
        "name": "zernike_spherical_um",
        "category": "low_order_surface_modes",
        "units": "um",
        "scale": "low_order_mode (wavelength >=1.4 um, N=4 when 2*r_patch>=2.8 um)",
        "is_noise_check": "False",
        "definition": "Fourth-order radial profile coefficient (Z_4^0 = 6*rho^4 - 6*rho^2 + 1) when allowed by the 1.4 um cutoff on patch diameter.",
    },
    {
        "name": "rough_above_cutoff_um2",
        "category": "low_order_surface_modes",
        "units": "um^2",
        "scale": "macro_topography (wavelength >=1.4 um)",
        "is_noise_check": "False",
        "definition": "Resolved variance (roughness power) of target contact surface height captured by low-order modes at wavelengths >= 1.4 um.",
    },
    {
        "name": "rough_below_cutoff_noise_um2",
        "category": "noise_check_only",
        "units": "um^2",
        "scale": "sub_cutoff_noise_check (wavelength <1.4 um; NOT biological topography)",
        "is_noise_check": "True",
        "definition": "NOISE CHECK ONLY: Residual surface height variance at wavelengths < 1.4 um (dominated by voxel staircase and U-Net smoothing; >60% raw power attenuated).",
    },
    {
        "name": "rough_below_cutoff_frac",
        "category": "noise_check_only",
        "units": "fraction (0-1)",
        "scale": "sub_cutoff_noise_check (wavelength <1.4 um; NOT biological topography)",
        "is_noise_check": "True",
        "definition": "NOISE CHECK ONLY: Fraction of total contact patch height variance lying below the 1.4 um wavelength resolution cutoff.",
    },
    # 4. T-cell shape & protrusion
    {
        "name": "tcell_volume_um3",
        "category": "tcell_shape",
        "units": "um^3",
        "scale": "whole_cell (~8-14 um)",
        "is_noise_check": "False",
        "definition": "3D volume of the tracked T-cell instance (t_vox * dx * dy * dz).",
    },
    {
        "name": "tcell_sphericity",
        "category": "tcell_shape",
        "units": "dimensionless (0-1)",
        "scale": "whole_cell (~8-14 um)",
        "is_noise_check": "False",
        "definition": "Isoperimetric 3D sphericity pi^(1/3) * (6 * V_t)^(2/3) / A_t of the tracked T cell using Calka-corrected exposed face area.",
    },
    {
        "name": "tcell_protrusion_vol_um3",
        "category": "tcell_shape",
        "units": "um^3",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Volume of tracked T-cell voxels lying strictly inside the contacted target cell's 3D convex hull.",
    },
    {
        "name": "tcell_protrusion_frac",
        "category": "tcell_shape",
        "units": "fraction (0-1)",
        "scale": "macro_cup (>=1.5-3.0 um)",
        "is_noise_check": "False",
        "definition": "Fraction of tracked T-cell volume protruding inside the contacted target cell's 3D convex hull (tcell_protrusion_vol_um3 / tcell_volume_um3).",
    },
    {
        "name": "tcell_elongation",
        "category": "tcell_shape",
        "units": "ratio (>=1)",
        "scale": "whole_cell (~8-14 um)",
        "is_noise_check": "False",
        "definition": "3D principal-axis elongation ratio sqrt(lambda_1 / lambda_3) of the tracked T-cell inertia covariance tensor.",
    },
    {
        "name": "tcell_align_cos_normal",
        "category": "tcell_shape",
        "units": "abs_cos (0-1)",
        "scale": "whole_cell (~8-14 um)",
        "is_noise_check": "False",
        "definition": "Absolute cosine |v_1 . n_syn| between the tracked T-cell major principal axis v_1 and the unit contact normal n_syn (1 = polarized perpendicular to synapse, 0 = flattened parallel).",
    },
    # 5. Target shape & volume dynamics
    {
        "name": "target_volume_um3",
        "category": "target_shape",
        "units": "um^3",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "3D volume of the contacted target cell (connected target body containing the contacted instance, avoiding transient Voronoi split jumps).",
    },
    {
        "name": "target_instance_voronoi_vol_um3",
        "category": "target_shape",
        "units": "um^3",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "3D volume of the single dominant Voronoi-partitioned target instance ID from instance_seg_matlab_xyz.",
    },
    {
        "name": "target_sphericity",
        "category": "target_shape",
        "units": "dimensionless (0-1)",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "Isoperimetric 3D sphericity pi^(1/3) * (6 * V_b)^(2/3) / A_b of the contacted target cell.",
    },
    {
        "name": "target_elongation",
        "category": "target_shape",
        "units": "ratio (>=1)",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "3D principal-axis elongation ratio sqrt(lambda_1 / lambda_3) of the contacted target cell inertia covariance tensor.",
    },
    {
        "name": "target_vol_ratio_to_v1",
        "category": "target_dynamics",
        "units": "ratio",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "Ratio of current target_volume_um3 to the chamber's baseline target volume (median over first 3 target-present visits).",
    },
    {
        "name": "target_vol_change_from_v1_um3",
        "category": "target_dynamics",
        "units": "um^3",
        "scale": "whole_cell (~10-16 um)",
        "is_noise_check": "False",
        "definition": "Signed change in target_volume_um3 relative to baseline target volume (target_volume_um3 - V_baseline).",
    },
]

DESCRIPTOR_NAMES: list[str] = [d["name"] for d in DESCRIPTOR_SPECS]


def fast_instance_seg_with_hint(
    nucleus: np.ndarray, membrane: np.ndarray, expected_n_inst: int = -1
) -> np.ndarray:
    """
    Exact equivalent of `instance_seg_matlab_xyz(nucleus, membrane)` that uses a fast path
    when `expected_n_inst == 1` (skipping the 25-voxel Euclidean dilation check since
    `per_sweep.csv` already confirmed the single surviving instance passed `temp >= 1000`).
    Falls back to `instance_seg_matlab_xyz` whenever `max_index != 1` or `expected_n_inst != 1`.
    """
    if expected_n_inst == 1:
        nuc_clean, _, max_index = bwareaopen_26(nucleus, 3000)
        if max_index == 1:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                counts = np.bincount(mem_lbl.ravel())
                counts[0] = 0
                combined = nuc_clean | (mem_lbl == int(np.argmax(counts)))
            else:
                combined = nuc_clean
            filled = imfill_holes_26(combined)
            return filled.astype(np.int8)
    return instance_seg_matlab_xyz(nucleus, membrane)


def fill_solid_cell_mask(mask_3d: np.ndarray) -> np.ndarray:
    """
    Close 2D slice gaps along Z, Y, and X followed by 3D hole filling so that interior
    cytoplasm voids left open by incomplete cortical shell segmentation do not create
    spurious internal boundary points.
    """
    sl = bbox_slices(mask_3d, 1, mask_3d.shape)
    if sl is None:
        return mask_3d
    sub = mask_3d[sl]
    m = sub.copy()
    for z in range(sub.shape[0]):
        if np.any(sub[z]):
            m[z] |= binary_fill_holes(sub[z])
    for y in range(sub.shape[1]):
        if np.any(sub[:, y, :]):
            m[:, y, :] |= binary_fill_holes(sub[:, y, :])
    for x in range(sub.shape[2]):
        if np.any(sub[:, :, x]):
            m[:, :, x] |= binary_fill_holes(sub[:, :, x])
    out = np.zeros_like(mask_3d, dtype=bool)
    out[sl] = binary_fill_holes(m, structure=STRUCT26)
    return out


def compute_exposed_surface_area_um2(
    mask_3d: np.ndarray, dz_um: float, dxy_um: float = VOXEL_UM
) -> float:
    """
    Compute 3D surface area of a binary mask with physical voxel pitches `(dz_um, dxy_um, dxy_um)`.
    Counts exposed 6-connected faces along Z, Y, and X, weighted by their respective face areas
    (`dxy^2` for Z faces, `dz * dxy` for Y and X faces) and multiplied by the theoretical 3D
    grid face-to-surface ratio `2/3 = 0.6667` (since a continuous sphere of radius R has
    exposed voxel face area `6 * pi * R^2` vs continuous surface area `4 * pi * R^2`).
    """
    if not np.any(mask_3d):
        return 0.0
    fz = int(np.sum(mask_3d[0, :, :])) + int(np.sum(mask_3d[-1, :, :]))
    if mask_3d.shape[0] > 1:
        fz += int(np.sum(mask_3d[1:, :, :] != mask_3d[:-1, :, :]))
    fy = int(np.sum(mask_3d[:, 0, :])) + int(np.sum(mask_3d[:, -1, :]))
    if mask_3d.shape[1] > 1:
        fy += int(np.sum(mask_3d[:, 1:, :] != mask_3d[:, :-1, :]))
    fx = int(np.sum(mask_3d[:, :, 0])) + int(np.sum(mask_3d[:, :, -1]))
    if mask_3d.shape[2] > 1:
        fx += int(np.sum(mask_3d[:, :, 1:] != mask_3d[:, :, :-1]))

    raw_area = float(fz) * (dxy_um * dxy_um) + float(fy + fx) * (dz_um * dxy_um)
    return (2.0 / 3.0) * raw_area


def compute_3d_pca(pts_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return sorted eigenvalues `eigvals` (`lambda_1 >= lambda_2 >= lambda_3 >= 0`)
    and corresponding unit eigenvectors `eigvecs[:, i]` (columns) of a 3D point cloud `pts_um`.
    """
    if len(pts_um) < 4:
        return np.array([1e-6, 1e-6, 1e-6]), np.eye(3)
    centered = pts_um - pts_um.mean(axis=0)
    cov = (centered.T @ centered) / float(len(pts_um))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(1e-8, vals[order])
    vecs = vecs[:, order]
    return vals, vecs


def fit_regularized_3d_sphere(
    pts_nc_um: np.ndarray, c_init_um: np.ndarray, r_eq_um: float
) -> tuple[np.ndarray, float]:
    """
    Fit a 3D reference sphere `(c_sph, R_sph)` to the target's non-contact boundary points
    `pts_nc_um`, regularized toward the whole-target centroid `c_init_um` and volume-equivalent
    radius `r_eq_um` so partial angular coverage never causes a runaway sphere radius.
    """
    if len(pts_nc_um) < 20:
        return c_init_um.copy(), float(r_eq_um)

    u = pts_nc_um - c_init_um
    reg_w = 0.08 * np.sqrt(float(len(u)))
    A_reg = np.vstack([
        np.column_stack([2.0 * u, np.ones(len(u), dtype=np.float64)]),
        np.diag([2.0 * reg_w, 2.0 * reg_w, 2.0 * reg_w, reg_w]),
    ])
    b_reg = np.concatenate([
        np.sum(u * u, axis=1),
        np.array([0.0, 0.0, 0.0, reg_w * (r_eq_um * r_eq_um)], dtype=np.float64),
    ])
    sol, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    du_s = np.clip(sol[:3], -2.0, 2.0)
    c_sph = c_init_um + du_s
    R_sph = float(
        np.clip(
            np.sqrt(max(1.0, sol[3] + float(np.sum(du_s * du_s)))),
            0.78 * r_eq_um,
            1.30 * r_eq_um,
        )
    )

    n_pts = float(len(u))
    reg_diag = np.diag([0.05, 0.05, 0.05, 0.06]) * n_pts
    for _ in range(3):
        du = pts_nc_um - c_sph
        dist = np.maximum(1e-4, np.linalg.norm(du, axis=1))
        res = dist - R_sph
        w = 1.0 / np.sqrt(1.0 + (res / 0.30) ** 2)
        J = np.column_stack([-du / dist[:, None], -np.ones_like(dist)])
        Jw = J * w[:, None]
        H = (Jw.T @ Jw) + reg_diag
        g = (Jw.T @ (res * w)) + np.array([
            0.05 * n_pts * (c_sph[0] - c_init_um[0]),
            0.05 * n_pts * (c_sph[1] - c_init_um[1]),
            0.05 * n_pts * (c_sph[2] - c_init_um[2]),
            0.06 * n_pts * (R_sph - r_eq_um),
        ])
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        c_sph = np.clip(c_sph + step[:3], c_init_um - 2.0, c_init_um + 2.0)
        R_sph = float(np.clip(R_sph + step[3], 0.78 * r_eq_um, 1.30 * r_eq_um))

    return c_sph, R_sph


def compute_descriptors_for_z_pitch(
    dz_um: float,
    s_largest_vox: int,
    s_total_vox: int,
    n_components: int,
    t_mask_sub: np.ndarray | None,
    t_surf_vox: np.ndarray | None,
    t_all_vox: np.ndarray | None,
    b_cell_sub: np.ndarray | None,
    b_vor_vox_count: int,
    b_surf_vox: np.ndarray | None,
    b_all_vox: np.ndarray | None,
    is_contact_on_b_surf: np.ndarray | None,
    is_noncontact_on_b_surf: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute all geometric descriptors at a given axial voxel pitch `dz_um` (`0.1962` or `0.2400`).
    All coordinates in `t_surf_vox`, `t_all_vox`, `b_surf_vox`, `b_all_vox` are `(N, 3)` arrays
    in global `(z, y, x)` voxel units.
    """
    dxy_um = VOXEL_UM
    scale_zyx = np.array([dz_um, dxy_um, dxy_um], dtype=np.float64)
    vox_vol_um3 = dz_um * dxy_um * dxy_um

    out: dict[str, float] = {k: np.nan for k in DESCRIPTOR_NAMES}
    out["patch_n_components"] = float(n_components)

    # 1. T-cell whole-cell descriptors (computed whenever T cell is present)
    t_center_um: np.ndarray | None = None
    t_major_vec: np.ndarray | None = None
    if t_mask_sub is not None and t_all_vox is not None and len(t_all_vox) > 0:
        t_vol_um3 = float(len(t_all_vox)) * vox_vol_um3
        t_area_um2 = compute_exposed_surface_area_um2(t_mask_sub, dz_um=dz_um, dxy_um=dxy_um)
        t_sph = (
            float(min(1.0, (np.pi ** (1.0 / 3.0)) * ((6.0 * t_vol_um3) ** (2.0 / 3.0)) / max(1e-4, t_area_um2)))
            if t_area_um2 > 0
            else np.nan
        )
        pts_t_surf_um = t_surf_vox * scale_zyx if (t_surf_vox is not None and len(t_surf_vox) >= 4) else (t_all_vox * scale_zyx)
        t_evals, t_evecs = compute_3d_pca(pts_t_surf_um)
        t_elong = float(np.sqrt(t_evals[0] / max(1e-6, t_evals[2])))
        t_center_um = (t_all_vox.mean(axis=0)) * scale_zyx
        t_major_vec = t_evecs[:, 0]

        out["tcell_volume_um3"] = t_vol_um3
        out["tcell_sphericity"] = t_sph
        out["tcell_elongation"] = t_elong

    # 2. Target whole-cell descriptors (computed whenever target cell is present)
    b_center_um: np.ndarray | None = None
    hull_b: ConvexHull | None = None
    pts_b_surf_um: np.ndarray | None = None
    b_vol_um3 = 0.0
    if b_cell_sub is not None and b_all_vox is not None and len(b_all_vox) > 0:
        b_vol_um3 = float(len(b_all_vox)) * vox_vol_um3
        b_vor_vol_um3 = float(b_vor_vox_count) * vox_vol_um3
        b_area_um2 = compute_exposed_surface_area_um2(b_cell_sub, dz_um=dz_um, dxy_um=dxy_um)
        b_sph = (
            float(min(1.0, (np.pi ** (1.0 / 3.0)) * ((6.0 * b_vol_um3) ** (2.0 / 3.0)) / max(1e-4, b_area_um2)))
            if b_area_um2 > 0
            else np.nan
        )
        pts_b_surf_um = b_surf_vox * scale_zyx if (b_surf_vox is not None and len(b_surf_vox) >= 6) else (b_all_vox * scale_zyx)
        b_evals, _ = compute_3d_pca(pts_b_surf_um)
        b_elong = float(np.sqrt(b_evals[0] / max(1e-6, b_evals[2])))
        b_center_um = (b_all_vox.mean(axis=0)) * scale_zyx

        out["target_volume_um3"] = b_vol_um3
        out["target_instance_voronoi_vol_um3"] = b_vor_vol_um3
        out["target_sphericity"] = b_sph
        out["target_elongation"] = b_elong

        if len(pts_b_surf_um) >= 12:
            try:
                hull_b = ConvexHull(pts_b_surf_um)
            except Exception:
                hull_b = None

    # 3. If no contact on this sweep, contact/cup descriptors are 0 / NaN
    if s_largest_vox <= 0 or pts_b_surf_um is None or is_contact_on_b_surf is None:
        out["contact_area_um2"] = 0.0
        out["contact_area_total_um2"] = 0.0
        out["patch_equiv_radius_um"] = 0.0
        if out["tcell_volume_um3"] > 0 and b_vol_um3 > 0:
            out["tcell_protrusion_vol_um3"] = 0.0
            out["tcell_protrusion_frac"] = 0.0
        return out

    pts_c_um = pts_b_surf_um[is_contact_on_b_surf]
    pts_nc_um = (
        pts_b_surf_um[is_noncontact_on_b_surf]
        if (is_noncontact_on_b_surf is not None and int(np.sum(is_noncontact_on_b_surf)) >= 15)
        else pts_b_surf_um
    )

    if len(pts_c_um) < 4:
        ca_factor = CA_VOXEL_FACTOR_UM2 * (dz_um / Z_NOMINAL_UM) ** 0.5
        ca_um2 = float(s_largest_vox) * ca_factor
        out["contact_area_um2"] = ca_um2
        out["contact_area_total_um2"] = float(s_total_vox) * ca_factor
        out["patch_equiv_radius_um"] = float(np.sqrt(ca_um2 / np.pi))
        return out

    # Contact normal n_syn: unit vector from target centroid toward contact patch centroid (or T-cell centroid)
    p_patch_um = pts_c_um.mean(axis=0)
    assert b_center_um is not None
    n_syn = p_patch_um - b_center_um
    if t_center_um is not None and np.dot(t_center_um - b_center_um, n_syn) < 0:
        n_syn = t_center_um - b_center_um
    n_norm = float(np.linalg.norm(n_syn))
    if n_norm < 1e-6:
        n_syn = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n_syn = n_syn / n_norm

    # Surface area scaling when dz_um != Z_NOMINAL_UM:
    # A surface element with unit normal n = (nz, ny, nx) in isotropic space has area stretched by
    # sqrt(nz^2 + (dz/dxy)^2 * (ny^2 + nx^2)) when Z is stretched by (dz/dxy).
    if abs(dz_um - Z_NOMINAL_UM) < 1e-6:
        area_stretch = 1.0
    else:
        r_z = dz_um / Z_NOMINAL_UM
        nz_sq = float(n_syn[0] ** 2)
        area_stretch = float(np.sqrt(nz_sq + (r_z * r_z) * (1.0 - nz_sq)))

    ca_um2 = float(s_largest_vox) * CA_VOXEL_FACTOR_UM2 * area_stretch
    ca_total_um2 = float(s_total_vox) * CA_VOXEL_FACTOR_UM2 * area_stretch
    r_eq_patch_um = float(np.sqrt(max(1e-6, ca_um2 / np.pi)))

    out["contact_area_um2"] = ca_um2
    out["contact_area_total_um2"] = ca_total_um2
    out["patch_equiv_radius_um"] = r_eq_patch_um

    # Orthonormal tangent basis (e1, e2) perpendicular to n_syn
    arb = np.array([1.0, 0.0, 0.0]) if abs(n_syn[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_syn, arb)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_syn, e1)
    e2 /= np.linalg.norm(e2)

    rel_c = pts_c_um - p_patch_um
    u_c = rel_c @ e1
    v_c = rel_c @ e2

    # Patch principal axes in the tangent plane
    cov_2d = np.cov(u_c, v_c)
    if cov_2d.ndim == 2:
        evals_2d = np.linalg.eigvalsh(cov_2d)
        l_max = max(1e-4, float(evals_2d[1]))
        l_min = max(1e-4, float(evals_2d[0]))
        out["patch_elongation"] = float(np.sqrt(l_max / l_min))
    else:
        out["patch_elongation"] = 1.0

    # T-cell major axis alignment to contact normal
    if t_major_vec is not None:
        out["tcell_align_cos_normal"] = float(np.clip( abs(np.dot(t_major_vec, n_syn)), 0.0, 1.0 ))

    # T-cell protrusion into target's 3D convex hull
    if hull_b is not None and t_all_vox is not None and len(t_all_vox) > 0:
        pts_t_all_um = t_all_vox * scale_zyx
        b_min = pts_b_surf_um.min(axis=0)
        b_max = pts_b_surf_um.max(axis=0)
        in_bbox = np.all((pts_t_all_um >= b_min) & (pts_t_all_um <= b_max), axis=1)
        if np.any(in_bbox):
            pts_cand = pts_t_all_um[in_bbox]
            eqs = hull_b.equations
            inside_hull = np.all((pts_cand @ eqs[:, :3].T) + eqs[:, 3] <= 1e-5, axis=1)
            n_prot_vox = int(np.sum(inside_hull))
        else:
            n_prot_vox = 0
        prot_vol = float(n_prot_vox) * vox_vol_um3
        out["tcell_protrusion_vol_um3"] = prot_vol
        out["tcell_protrusion_frac"] = prot_vol / max(1e-6, out["tcell_volume_um3"])

    # 4. Target indentation relative to (a) 3D convex hull and (b) regularized non-contact sphere
    d_elem_area = ca_um2 / float(len(pts_c_um))

    # (a) Convex hull indentation depth & cup volume
    if hull_b is not None:
        eqs = hull_b.equations
        facing = (eqs[:, :3] @ n_syn) > 0.05
        if not np.any(facing):
            facing = np.ones(len(eqs), dtype=bool)
        dists_hull = -((pts_c_um @ eqs[facing, :3].T) + eqs[facing, 3])
        d_hull_c = np.maximum(0.0, np.min(dists_hull, axis=1))
    else:
        d_hull_c = np.zeros(len(pts_c_um), dtype=np.float64)

    cup_depth_hull_p95 = float(np.percentile(d_hull_c, 95.0))
    cup_depth_hull_max = float(np.max(d_hull_c))
    cup_vol_hull = float(np.sum(d_hull_c) * d_elem_area)

    # (b) Regularized non-contact reference sphere indentation depth & cup volume
    r_eq_target = float((3.0 * b_vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    c_sph, R_sph = fit_regularized_3d_sphere(pts_nc_um, b_center_um, r_eq_target)
    r_c_from_sph = np.linalg.norm(pts_c_um - c_sph, axis=1)
    w_sph_signed = R_sph - r_c_from_sph  # positive = inward indentation relative to sphere
    d_sph_pos = np.maximum(0.0, w_sph_signed)
    cup_depth_sph_p95 = float(np.percentile(d_sph_pos, 95.0))
    cup_vol_sph = float(np.sum(d_sph_pos) * d_elem_area)

    out["cup_depth_hull_p95_um"] = cup_depth_hull_p95
    out["cup_depth_hull_max_um"] = cup_depth_hull_max
    out["cup_depth_sphere_p95_um"] = cup_depth_sph_p95
    out["cup_volume_hull_um3"] = cup_vol_hull
    out["cup_volume_sphere_um3"] = cup_vol_sph
    out["cup_aspect_radius_over_depth"] = (
        float(r_eq_patch_um / max(0.05, cup_depth_hull_p95))
        if cup_depth_hull_p95 > 0.02
        else np.nan
    )
    out["cup_depth_over_radius"] = float(cup_depth_hull_p95 / max(0.20, r_eq_patch_um))

    # Cup asymmetry: lateral offset of top 5% deepest points from patch centroid, normalized by patch radius
    if cup_depth_hull_max > 0.02:
        deep_thr = np.percentile(d_hull_c, 95.0)
        deep_mask = d_hull_c >= deep_thr
        u_deep = float(np.mean(u_c[deep_mask]))
        v_deep = float(np.mean(v_c[deep_mask]))
        out["cup_asymmetry_norm"] = float(np.hypot(u_deep, v_deep) / max(0.25, r_eq_patch_um))
    else:
        out["cup_asymmetry_norm"] = 0.0

    # 5. Low-order Zernike-like basis expansion of target surface around the contact patch
    #    Subject to the 1.4 um wavelength cutoff (half-wavelength bump >= 0.70 um):
    #    Maximum radial/angular order allowed on patch diameter D = 2 * r_disk is N_max = clip(floor(D / 0.70), 2, 4).
    r_pts = np.hypot(u_c, v_c)
    r_disk = max(0.75, float(np.percentile(r_pts, 95.0)), r_eq_patch_um)
    un = np.clip(u_c / r_disk, -1.15, 1.15)
    vn = np.clip(v_c / r_disk, -1.15, 1.15)
    rho2 = un * un + vn * vn

    # Height profile w(u, v): inward indentation relative to the local tangent plane
    # (negative w = inward indentation toward target center, so concave cup has negative defocus Z_2^0 < 0,
    #  or using w_indent = d_hull_c where positive = inward depth; let's use w_surf = (pts_c_um - p_patch_um) @ n_syn,
    #  where negative at center and positive at rim means a concave cup wrapping the T cell -> Z_2^0 > 0 or < 0;
    #  specifically, relative to the reference sphere, inward depth is w_sph_signed = R_sph - ||x - c_sph||,
    #  where center of cup has larger inward depth than rim, so Z_2^0 = 2*rho^2 - 1 has NEGATIVE coefficient!)
    w_profile = w_sph_signed

    B_zern = np.column_stack([
        np.ones_like(un),                                  # 0: Z_0^0 piston
        un, vn,                                            # 1,2: Z_1^-1, Z_1^1 tilt (N=1)
        2.0 * rho2 - 1.0, un * un - vn * vn, 2.0 * un * vn,  # 3,4,5: defocus, astigmatism (N=2)
        (3.0 * rho2 - 2.0) * un, (3.0 * rho2 - 2.0) * vn,  # 6,7: coma (N=3)
        un * (un * un - 3.0 * vn * vn), vn * (3.0 * un * un - vn * vn),  # 8,9: trefoil (N=3)
        6.0 * (rho2 * rho2) - 6.0 * rho2 + 1.0,            # 10: spherical (N=4)
        (4.0 * rho2 - 3.0) * (un * un - vn * vn),          # 11: 2nd astig (N=4)
        (4.0 * rho2 - 3.0) * (2.0 * un * vn),              # 12: 2nd astig (N=4)
        (un * un - vn * vn) ** 2 - 4.0 * (un * un) * (vn * vn),  # 13: quadrafoil (N=4)
        4.0 * un * vn * (un * un - vn * vn),               # 14: quadrafoil (N=4)
    ])

    # Allowed order from 1.4 um wavelength constraint:
    # A radial mode of order N has N nodal half-wavelengths across diameter 2*r_disk,
    # so half-wavelength >= 0.70 um (full wavelength >= 1.40 um) requires N <= floor(2*r_disk / 0.70).
    n_allowed = int(np.clip(np.floor(2.0 * r_disk / RESOLUTION_BUMP_CUTOFF_UM), 2, 4))
    col_limit = {2: 6, 3: 10, 4: 15}[n_allowed]

    # Fit N=2 quadric Zernike coefficients stably for all patches, and up to col_limit for roughness split
    ridge = 1e-3 * float(len(un))
    B2 = B_zern[:, :6]
    c2 = np.linalg.lstsq(
        (B2.T @ B2) + np.diag([0.0, ridge, ridge, ridge, ridge, ridge]),
        B2.T @ w_profile,
        rcond=None,
    )[0]

    Bk = B_zern[:, :col_limit]
    reg_k = np.full(col_limit, ridge, dtype=np.float64)
    reg_k[0] = 0.0
    ck = np.linalg.lstsq((Bk.T @ Bk) + np.diag(reg_k), Bk.T @ w_profile, rcond=None)[0]
    w_fit_k = Bk @ ck

    w_zero_mean = w_profile - float(np.mean(w_profile))
    tot_power = float(np.mean(w_zero_mean * w_zero_mean))
    above_power = float(np.mean((w_fit_k - float(np.mean(w_fit_k))) ** 2))
    below_noise_power = float(np.mean((w_profile - w_fit_k) ** 2))

    out["zernike_defocus_um"] = float(c2[3])
    out["zernike_tilt_amp_um"] = float(np.hypot(c2[1], c2[2]))
    out["zernike_astig_amp_um"] = float(np.hypot(c2[4], c2[5]))
    out["zernike_coma_amp_um"] = float(np.hypot(ck[6], ck[7])) if col_limit >= 10 else 0.0
    out["zernike_spherical_um"] = float(ck[10]) if col_limit >= 15 else 0.0
    out["rough_above_cutoff_um2"] = above_power
    out["rough_below_cutoff_noise_um2"] = below_noise_power
    out["rough_below_cutoff_frac"] = (
        float(np.clip(below_noise_power / max(1e-8, above_power + below_noise_power), 0.0, 1.0))
        if tot_power > 1e-8
        else np.nan
    )

    return out


def process_one_chamber_descriptors(task: dict[str, Any]) -> dict[str, Any]:
    """
    Compute per-sweep descriptors across all actin sweeps (1..5 of each visit) for one chamber
    under both `z = 0.1962 um` and `z = 0.2400 um`.
    """
    t0 = time.perf_counter()
    exp: str = task["experiment"]
    roi: str = task["roi"]
    roi_idx: int = int(task["roi_index"])
    gid: str = task["group_id"]
    cond: str = task["condition"]
    trace_num: int = int(task["trace_num"])
    peak_visit_ca: int = int(task["peak_visit"])
    sweep_meta_list: list[dict[str, Any]] = task["sweep_meta_list"]

    sweep_rows: list[dict[str, Any]] = []

    for sw_meta in sweep_meta_list:
        visit = int(sw_meta["visit"])
        sweep = int(sw_meta["sweep"])
        tp_idx = int(sw_meta["timepoint_idx"])
        rel_path = str(sw_meta["rel_path"])
        tracked_id = int(sw_meta["tracked_tcell_id"])
        n_t_hint = int(sw_meta["n_tcell_instances"])
        n_b_hint = int(sw_meta["n_target_instances"])
        s_largest = int(sw_meta["s_largest"])
        s_total = int(sw_meta["s_total"])
        n_comp = int(sw_meta["n_components"])

        abs_path = REPO_ROOT / rel_path
        with h5py.File(abs_path, "r") as hf:
            I = hf["predictions"][:]

        # 1. Segment T-cell instance if tracked_id > 0 (or if trace_num != 1 and T-cell voxels exist)
        t_mask_sub: np.ndarray | None = None
        t_surf_vox: np.ndarray | None = None
        t_all_vox: np.ndarray | None = None
        tree_t_surf: cKDTree | None = None

        sl_t = bbox_slices((I == 1) | (I == 3), 2, I.shape)
        if sl_t is not None and (tracked_id > 0 or trace_num != 1):
            t_inst_sub = fast_instance_seg_with_hint(
                I[sl_t] == 1, I[sl_t] == 3, expected_n_inst=n_t_hint if trace_num == 1 else -1
            )
            if tracked_id > 0 and np.any(t_inst_sub == tracked_id):
                t_sel = t_inst_sub == tracked_id
            elif trace_num != 1 and np.any(t_inst_sub > 0):
                # For non-single-track chambers when Track 1 is absent, pick largest T cell for T-cell shape
                cnts_t = np.bincount(t_inst_sub.ravel())
                cnts_t[0] = 0
                t_sel = t_inst_sub == int(np.argmax(cnts_t))
            else:
                t_sel = None

            if t_sel is not None and np.any(t_sel):
                t_sel = fill_solid_cell_mask(t_sel)
                t_mask_sub = t_sel
                off_t = np.array([sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64)
                t_all_vox = np.argwhere(t_sel).astype(np.float64) + off_t
                t_surf_mask = t_sel & (~binary_erosion(t_sel, structure=STRUCT26))
                t_surf_vox = np.argwhere(t_surf_mask).astype(np.float64) + off_t
                if len(t_surf_vox) > 0:
                    tree_t_surf = cKDTree(t_surf_vox)

        # 2. Segment Target instance(s)
        b_cell_sub: np.ndarray | None = None
        b_vor_vox_count: int = 0
        b_surf_vox: np.ndarray | None = None
        b_all_vox: np.ndarray | None = None
        is_contact_on_b_surf: np.ndarray | None = None
        is_noncontact_on_b_surf: np.ndarray | None = None

        sl_b = bbox_slices((I == 2) | (I == 4), 2, I.shape)
        if sl_b is not None:
            b_inst_sub = fast_instance_seg_with_hint(
                I[sl_b] == 2, I[sl_b] == 4, expected_n_inst=n_b_hint
            )
            max_b = int(b_inst_sub.max())
            if max_b > 0:
                off_b = np.array([sl_b[0].start, sl_b[1].start, sl_b[2].start], dtype=np.float64)
                cnts_b = np.bincount(b_inst_sub.ravel())
                cnts_b[0] = 0

                if max_b == 1:
                    best_bid = 1
                elif tree_t_surf is not None:
                    # Pick the target instance with the most boundary voxels within 3.5 voxels of T cell
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

                b_vor_vox_count = int(cnts_b[best_bid])

                # Connected target cell body containing best_bid (avoids artificial Voronoi split jumps
                # when a single target cell has a fragmented interior mask I==2)
                if max_b == 1:
                    b_cell_sub = fill_solid_cell_mask(b_inst_sub == 1)
                else:
                    cc_lbl, _ = label(b_inst_sub > 0, structure=STRUCT26)
                    dom_cc = int(np.median(cc_lbl[b_inst_sub == best_bid]))
                    b_cell_sub = fill_solid_cell_mask(cc_lbl == dom_cc)

                b_all_vox = np.argwhere(b_cell_sub).astype(np.float64) + off_b
                b_surf_mask = b_cell_sub & (~binary_erosion(b_cell_sub, structure=STRUCT26))
                b_surf_vox = np.argwhere(b_surf_mask).astype(np.float64) + off_b

                if s_largest > 0 and tree_t_surf is not None and len(b_surf_vox) > 0:
                    d_bt, _ = tree_t_surf.query(b_surf_vox, k=1)
                    # Select the top K closest target boundary voxels to the T cell (matching s_largest_vox)
                    # or all within 2.5 voxels, whichever better matches the contact patch extent
                    within_25 = d_bt <= 2.6
                    if int(np.sum(within_25)) >= 12:
                        is_contact_on_b_surf = within_25
                    else:
                        k_sel = min(len(d_bt), max(12, s_largest))
                        thr_d = min(4.5, float(np.partition(d_bt, k_sel - 1)[k_sel - 1]))
                        is_contact_on_b_surf = d_bt <= thr_d
                    is_noncontact_on_b_surf = d_bt > 3.5

        # 3. Compute descriptors under both z = 0.1962 um and z = 0.2400 um
        desc_nom = compute_descriptors_for_z_pitch(
            dz_um=Z_NOMINAL_UM,
            s_largest_vox=s_largest,
            s_total_vox=s_total,
            n_components=n_comp,
            t_mask_sub=t_mask_sub,
            t_surf_vox=t_surf_vox,
            t_all_vox=t_all_vox,
            b_cell_sub=b_cell_sub,
            b_vor_vox_count=b_vor_vox_count,
            b_surf_vox=b_surf_vox,
            b_all_vox=b_all_vox,
            is_contact_on_b_surf=is_contact_on_b_surf,
            is_noncontact_on_b_surf=is_noncontact_on_b_surf,
        )
        desc_alt = compute_descriptors_for_z_pitch(
            dz_um=Z_ALT_UM,
            s_largest_vox=s_largest,
            s_total_vox=s_total,
            n_components=n_comp,
            t_mask_sub=t_mask_sub,
            t_surf_vox=t_surf_vox,
            t_all_vox=t_all_vox,
            b_cell_sub=b_cell_sub,
            b_vor_vox_count=b_vor_vox_count,
            b_surf_vox=b_surf_vox,
            b_all_vox=b_all_vox,
            is_contact_on_b_surf=is_contact_on_b_surf,
            is_noncontact_on_b_surf=is_noncontact_on_b_surf,
        )

        row_dict: dict[str, Any] = {
            "experiment": exp,
            "roi": roi,
            "roi_index": roi_idx,
            "group_id": gid,
            "visit": visit,
            "sweep": sweep,
            "timepoint_idx": tp_idx,
            "trace_num": trace_num,
            "included_by_trace_rule": bool(trace_num == 1),
            "tracked_tcell_id": tracked_id,
            "in_contact": bool(s_largest > 0),
        }
        for dname in DESCRIPTOR_NAMES:
            row_dict[dname] = desc_nom[dname]
        for dname in DESCRIPTOR_NAMES:
            row_dict[f"{dname}_z024"] = desc_alt[dname]

        sweep_rows.append(row_dict)

    # 4. Fill in target_vol_ratio_to_v1 and target_vol_change_from_v1_um3 per chamber
    df_ch_sw = pd.DataFrame(sweep_rows)
    for sfx in ["", "_z024"]:
        vcol = f"target_volume_um3{sfx}"
        rcol = f"target_vol_ratio_to_v1{sfx}"
        dcol = f"target_vol_change_from_v1_um3{sfx}"
        valid_v = df_ch_sw[df_ch_sw[vcol] > 0]
        if len(valid_v) > 0:
            first_visits = sorted(valid_v["visit"].unique())[:3]
            v_base = float(valid_v[valid_v["visit"].isin(first_visits)][vcol].median())
            if v_base > 0:
                df_ch_sw[rcol] = df_ch_sw[vcol] / v_base
                df_ch_sw[dcol] = df_ch_sw[vcol] - v_base

    elapsed_s = time.perf_counter() - t0
    return {
        "experiment": exp,
        "roi": roi,
        "roi_index": roi_idx,
        "group_id": gid,
        "trace_num": trace_num,
        "peak_visit": peak_visit_ca,
        "elapsed_s": elapsed_s,
        "sweep_df": df_ch_sw,
    }


# ==============================================================================
# VISIT & CHAMBER AGGREGATION, ICC, Z-SENSITIVITY, AND FIGURE GENERATION
# ==============================================================================
def aggregate_visits_and_chambers(
    df_sweeps: pd.DataFrame, df_step3_chambers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-sweep descriptors to:
      1. `per_visit_descriptors.csv`: per visit (mean over sweeps 1-5 and sweep-to-sweep SD
         as a within-visit noise estimate; for contact/cup descriptors, averaged over
         contact sweeps `in_contact == True` matching `step3_analysis.m`).
      2. `per_chamber_descriptors.csv`: per chamber (value at peak-CA visit, mean over all
         contact visits, and value at first-contact visit, plus target volume change rate
         over time `target_vol_slope_um3_per_min` and `target_vol_end_over_start_ratio`).
    """
    all_desc_cols: list[str] = []
    for dname in DESCRIPTOR_NAMES:
        all_desc_cols.append(dname)
    for dname in DESCRIPTOR_NAMES:
        all_desc_cols.append(f"{dname}_z024")

    # Identify which descriptors require `in_contact == True` when averaging over sweeps 1-5
    contact_only_base = {
        d["name"]
        for d in DESCRIPTOR_SPECS
        if d["category"] in ("contact_patch", "target_indentation", "low_order_surface_modes", "noise_check_only")
        or d["name"] in ("tcell_protrusion_vol_um3", "tcell_protrusion_frac", "tcell_align_cos_normal")
    }
    contact_only_cols = contact_only_base | {f"{c}_z024" for c in contact_only_base}
    whole_cell_cols = [c for c in all_desc_cols if c not in contact_only_cols]

    # Group by (experiment, roi, visit)
    grp_keys = ["experiment", "roi", "roi_index", "group_id", "trace_num", "included_by_trace_rule", "visit"]

    # Meta aggregations per visit
    meta_v = (
        df_sweeps.groupby(grp_keys, as_index=False)
        .agg(
            n_sweeps=("sweep", "count"),
            n_contact_sweeps=("in_contact", "sum"),
        )
        .sort_values(["experiment", "roi_index", "visit"])
        .reset_index(drop=True)
    )
    meta_v["in_contact_visit"] = meta_v["n_contact_sweeps"] > 0

    # Whole-cell descriptors: mean and SD across all sweeps 1..5 where present
    wc_mean = df_sweeps.groupby(grp_keys, as_index=False)[whole_cell_cols].mean()
    wc_sd = df_sweeps.groupby(grp_keys, as_index=False)[whole_cell_cols].std(ddof=1)

    # Contact descriptors: mean and SD across contact sweeps (`in_contact == True`) matching step3_analysis.m
    df_contact_sw = df_sweeps[df_sweeps["in_contact"]].copy()
    if len(df_contact_sw) > 0:
        co_cols_list = sorted(contact_only_cols)
        co_mean = df_contact_sw.groupby(grp_keys, as_index=False)[co_cols_list].mean()
        co_sd = df_contact_sw.groupby(grp_keys, as_index=False)[co_cols_list].std(ddof=1)
    else:
        co_cols_list = sorted(contact_only_cols)
        co_mean = meta_v[grp_keys].copy()
        co_sd = meta_v[grp_keys].copy()
        for c in co_cols_list:
            co_mean[c] = np.nan
            co_sd[c] = np.nan

    df_visit = meta_v.merge(wc_mean, on=grp_keys, how="left").merge(co_mean, on=grp_keys, how="left")

    # Fill 0 for contact_area_um2, contact_area_total_um2, patch_equiv_radius_um, patch_n_components on non-contact visits
    for zero_c in ["contact_area_um2", "contact_area_total_um2", "patch_equiv_radius_um", "patch_n_components"]:
        for sfx in ["", "_z024"]:
            cname = f"{zero_c}{sfx}"
            if cname in df_visit.columns:
                df_visit[cname] = np.where(df_visit["in_contact_visit"], df_visit[cname], 0.0)

    # Rename SD columns with `_sd` suffix and merge
    wc_sd_ren = wc_sd.rename(columns={c: f"{c}_sd" for c in whole_cell_cols})
    co_sd_ren = co_sd.rename(columns={c: f"{c}_sd" for c in co_cols_list})
    df_visit = df_visit.merge(wc_sd_ren, on=grp_keys, how="left").merge(co_sd_ren, on=grp_keys, how="left")

    # Reorder df_visit columns cleanly
    ordered_v_cols = list(grp_keys) + ["n_sweeps", "n_contact_sweeps", "in_contact_visit"]
    for dname in DESCRIPTOR_NAMES:
        ordered_v_cols.extend([dname, f"{dname}_sd"])
    for dname in DESCRIPTOR_NAMES:
        ordered_v_cols.extend([f"{dname}_z024", f"{dname}_z024_sd"])
    df_visit = df_visit[ordered_v_cols].sort_values(["experiment", "roi_index", "visit"]).reset_index(drop=True)

    # 2. Aggregate per chamber:
    #    - Value at peak-CA visit (`_peak_ca`)
    #    - Mean over contact visits (`_mean_contact`, plus mean over all visits for whole-cell descriptors when no contact visit exists)
    #    - Value at first-contact visit (`_first_contact`)
    #    - Mean within-visit sweep SD (`_within_visit_sd`)
    #    - Target volume slope over time (`target_vol_slope_um3_per_min`, `target_vol_end_over_start_ratio`)
    peak_map: dict[tuple[str, str], int] = {}
    if df_step3_chambers is not None and len(df_step3_chambers) > 0:
        for _, r in df_step3_chambers.iterrows():
            peak_map[(str(r["experiment"]), str(r["roi"]))] = int(r["peak_visit"])

    chamber_rows: list[dict[str, Any]] = []
    for (exp, roi), vdf in df_visit.groupby(["experiment", "roi"], sort=False):
        vdf_s = vdf.sort_values("visit")
        first_row = vdf_s.iloc[0]
        roi_idx = int(first_row["roi_index"])
        gid = str(first_row["group_id"])
        tr_num = int(first_row["trace_num"])
        incl = bool(first_row["included_by_trace_rule"])

        contact_vdf = vdf_s[vdf_s["in_contact_visit"]]
        n_visits = len(vdf_s)
        n_contact_visits = len(contact_vdf)
        n_contact_sweeps = int(vdf_s["n_contact_sweeps"].sum())

        pk_v = peak_map.get((str(exp), str(roi)), 0)
        if pk_v <= 0 and n_contact_visits > 0:
            pk_v = int(contact_vdf.loc[contact_vdf["contact_area_um2"].idxmax(), "visit"])
        first_c_v = int(contact_vdf["visit"].iloc[0]) if n_contact_visits > 0 else 0

        pk_row = vdf_s[vdf_s["visit"] == pk_v].iloc[0] if (pk_v > 0 and np.any(vdf_s["visit"] == pk_v)) else None
        fc_row = vdf_s[vdf_s["visit"] == first_c_v].iloc[0] if (first_c_v > 0 and np.any(vdf_s["visit"] == first_c_v)) else None

        ch_dict: dict[str, Any] = {
            "experiment": exp,
            "roi": roi,
            "roi_index": roi_idx,
            "group_id": gid,
            "trace_num": tr_num,
            "included_by_trace_rule": incl,
            "n_visits": n_visits,
            "n_contact_visits": n_contact_visits,
            "n_contact_sweeps": n_contact_sweeps,
            "peak_ca_visit": pk_v,
            "first_contact_visit": first_c_v,
        }

        # Compute target volume change over time (OLS slope in um^3 / min across visits, 10 min/visit)
        for sfx in ["", "_z024"]:
            v_col = f"target_volume_um3{sfx}"
            valid_tv = vdf_s[["visit", v_col]].dropna()
            valid_tv = valid_tv[valid_tv[v_col] > 0]
            if len(valid_tv) >= 2:
                t_min = (valid_tv["visit"].to_numpy(dtype=np.float64) - 1.0) * 10.0
                y_vol = valid_tv[v_col].to_numpy(dtype=np.float64)
                tm = t_min - t_min.mean()
                denom = float(np.dot(tm, tm))
                slope = float(np.dot(tm, y_vol - y_vol.mean()) / denom) if denom > 1e-9 else 0.0
                n_edge = min(3, len(y_vol))
                v_start = float(np.median(y_vol[:n_edge]))
                v_end = float(np.median(y_vol[-n_edge:]))
                end_over_start = v_end / max(1e-6, v_start)
            else:
                slope = np.nan
                end_over_start = np.nan
            ch_dict[f"target_vol_slope_um3_per_min{sfx}"] = slope
            ch_dict[f"target_vol_end_over_start_ratio{sfx}"] = end_over_start

        # Aggregate each descriptor at peak_ca, mean_contact, first_contact, and within_visit_sd
        for dcol in all_desc_cols:
            if n_contact_visits > 0:
                val_pk = float(pk_row[dcol]) if pk_row is not None else np.nan
                val_mc = float(contact_vdf[dcol].mean())
                val_fc = float(fc_row[dcol]) if fc_row is not None else np.nan
                val_sd = float(contact_vdf[f"{dcol}_sd"].mean())
            else:
                if dcol in ["contact_area_um2", "contact_area_total_um2", "patch_equiv_radius_um", "patch_n_components",
                            "contact_area_um2_z024", "contact_area_total_um2_z024", "patch_equiv_radius_um_z024", "patch_n_components_z024"]:
                    val_pk = 0.0
                    val_mc = 0.0
                    val_fc = 0.0
                elif dcol in whole_cell_cols:
                    val_pk = np.nan
                    val_mc = float(vdf_s[dcol].mean())
                    val_fc = np.nan
                else:
                    val_pk = np.nan
                    val_mc = np.nan
                    val_fc = np.nan
                val_sd = float(vdf_s[f"{dcol}_sd"].mean()) if dcol in whole_cell_cols else np.nan

            ch_dict[f"{dcol}_peak_ca"] = val_pk
            ch_dict[f"{dcol}_mean_contact"] = val_mc
            ch_dict[f"{dcol}_first_contact"] = val_fc
            ch_dict[f"{dcol}_within_visit_sd"] = val_sd

        chamber_rows.append(ch_dict)

    df_chamber = pd.DataFrame(chamber_rows).sort_values(["experiment", "roi_index"]).reset_index(drop=True)
    return df_visit, df_chamber


def compute_icc_and_z_sensitivity_dictionary(
    df_sweeps: pd.DataFrame, df_visits: pd.DataFrame, df_chambers: pd.DataFrame
) -> pd.DataFrame:
    """
    Build `descriptor_dictionary.csv` with columns:
      - `name`, `definition`, `units`, `scale`, `category`, `is_noise_check`
      - `n_valid_sweeps`, `n_valid_visits`, `n_valid_chambers`
      - `median_z0196`, `iqr_z0196`, `mean_z0196`, `sd_z0196`
      - `within_visit_sd_mean`, `within_visit_sd_median`, `between_chamber_sd`, `between_visit_sd`
      - `ICC` (one-way random-effects intraclass correlation comparing between-chamber spread
        `sigma_between_chamber^2` to within-visit sweep-to-sweep noise variance `sigma_within_visit^2`:
        `ICC = sigma_between_chamber^2 / (sigma_between_chamber^2 + sigma_within_visit^2)`),
      - `ICC_visit_anova` (exact one-way random-effects ANOVA ICC(1) across visits with >= 2 valid sweeps:
        `(MS_B - MS_W) / (MS_B + (k_0 - 1) * MS_W)`),
      - `icc_noise_flag` (`True` if `ICC < 0.25` or `is_noise_check == True`),
      - `median_z0240`, `z_rel_diff_median_pct`, `z_abs_rel_diff_median_pct`, `z_sensitive_gt_10pct`
    """
    dict_rows: list[dict[str, Any]] = []

    # Restrict contact descriptors to contact sweeps/visits so zero non-contact visits do not artificially inflate between-chamber spread
    contact_only_base = {
        d["name"]
        for d in DESCRIPTOR_SPECS
        if d["category"] in ("contact_patch", "target_indentation", "low_order_surface_modes", "noise_check_only")
        or d["name"] in ("tcell_protrusion_vol_um3", "tcell_protrusion_frac", "tcell_align_cos_normal")
    }

    for spec in DESCRIPTOR_SPECS:
        dname = spec["name"]
        dname_z = f"{dname}_z024"
        is_co = dname in contact_only_base

        sw_sub = df_sweeps[df_sweeps["in_contact"]] if is_co else df_sweeps
        v_sub = df_visits[df_visits["in_contact_visit"]] if is_co else df_visits
        ch_sub = df_chambers[df_chambers["n_contact_visits"] > 0] if is_co else df_chambers

        s_nom = sw_sub[dname].dropna().to_numpy(dtype=np.float64)
        s_alt = sw_sub[dname_z].dropna().to_numpy(dtype=np.float64)

        v_means = v_sub[dname].dropna().to_numpy(dtype=np.float64)
        v_sds = v_sub[f"{dname}_sd"].dropna().to_numpy(dtype=np.float64)
        ch_means = ch_sub[f"{dname}_mean_contact"].dropna().to_numpy(dtype=np.float64)
        ch_pks = ch_sub[f"{dname}_peak_ca"].dropna().to_numpy(dtype=np.float64)

        if len(s_nom) > 0:
            med_nom = float(np.median(s_nom))
            iqr_nom = float(np.percentile(s_nom, 75.0) - np.percentile(s_nom, 25.0))
            mean_nom = float(np.mean(s_nom))
            sd_nom = float(np.std(s_nom, ddof=1)) if len(s_nom) > 1 else 0.0
        else:
            med_nom = iqr_nom = mean_nom = sd_nom = np.nan

        if len(s_alt) > 0:
            med_alt = float(np.median(s_alt))
        else:
            med_alt = np.nan

        # Within-visit sweep-to-sweep noise variance vs between-chamber spread
        # 1. Pooled within-visit variance MS_W from sweep-to-sweep SDs within each visit
        if len(v_sds) > 0:
            within_sd_mean = float(np.mean(v_sds))
            within_sd_med = float(np.median(v_sds))
            sigma2_within = float(np.mean(v_sds * v_sds))
        else:
            within_sd_mean = within_sd_med = sigma2_within = np.nan

        # 2. Between-chamber variance across chamber-level contact means (`_mean_contact`) and peak-CA values
        if len(ch_means) > 1:
            between_ch_sd = float(np.std(ch_means, ddof=1))
            sigma2_between_ch = float(np.var(ch_means, ddof=1))
        else:
            between_ch_sd = sigma2_between_ch = np.nan

        if len(v_means) > 1:
            between_v_sd = float(np.std(v_means, ddof=1))
        else:
            between_v_sd = np.nan

        # Intraclass correlation: between-chamber variance vs within-visit sweep variance
        # Note: for `target_vol_ratio_to_v1` at the chamber level, if evaluated at visit 1 it is ~1.0,
        # whereas across contact visits (`ch_means`) or across all visits (`v_means`) it captures swelling/lysis.
        if np.isfinite(sigma2_between_ch) and np.isfinite(sigma2_within) and (sigma2_between_ch + sigma2_within) > 1e-12:
            icc_chamber = float(sigma2_between_ch / (sigma2_between_ch + sigma2_within))
        elif np.isfinite(sigma2_between_ch) and sigma2_within == 0.0 and sigma2_between_ch > 0:
            icc_chamber = 1.0
        else:
            icc_chamber = 0.0

        # Also compute exact one-way ANOVA ICC(1) across visits (visit as group, sweeps 1..5 as replicates)
        sw_valid = sw_sub[["experiment", "roi", "visit", dname]].dropna()
        if len(sw_valid) > 10:
            v_grp = sw_valid.groupby(["experiment", "roi", "visit"])[dname]
            v_cnt = v_grp.count()
            v_cnt_ge2 = v_cnt[v_cnt >= 2]
            if len(v_cnt_ge2) >= 5:
                sw_ge2 = sw_valid.set_index(["experiment", "roi", "visit"]).loc[v_cnt_ge2.index].reset_index()
                g_ge2 = sw_ge2.groupby(["experiment", "roi", "visit"])[dname]
                n_j = g_ge2.count().to_numpy(dtype=np.float64)
                m_j = g_ge2.mean().to_numpy(dtype=np.float64)
                v_j = g_ge2.var(ddof=1).to_numpy(dtype=np.float64)
                N_tot = float(np.sum(n_j))
                K_grp = float(len(n_j))
                grand_m = float(np.sum(n_j * m_j) / N_tot)
                ss_b = float(np.sum(n_j * ((m_j - grand_m) ** 2)))
                ss_w = float(np.sum((n_j - 1.0) * v_j))
                ms_b = ss_b / max(1.0, K_grp - 1.0)
                ms_w = ss_w / max(1.0, N_tot - K_grp)
                k0 = (N_tot - float(np.sum(n_j * n_j)) / N_tot) / max(1.0, K_grp - 1.0)
                sigma2_b_anova = max(0.0, (ms_b - ms_w) / max(1.0, k0))
                icc_visit_anova = (
                    float(sigma2_b_anova / (sigma2_b_anova + ms_w))
                    if (sigma2_b_anova + ms_w) > 1e-12
                    else 0.0
                )
            else:
                icc_visit_anova = np.nan
        else:
            icc_visit_anova = np.nan

        # Z-pitch sensitivity: compare z = 0.2400 um vs z = 0.1962 um across valid sweeps
        both_valid = sw_sub[[dname, dname_z]].dropna()
        if len(both_valid) > 0:
            v196 = both_valid[dname].to_numpy(dtype=np.float64)
            v240 = both_valid[dname_z].to_numpy(dtype=np.float64)
            # Signed relative change of the median (or median pointwise relative change where |v196| > 1e-6)
            if abs(med_nom) > 1e-6:
                rel_diff_med_pct = float((med_alt - med_nom) / abs(med_nom) * 100.0)
            else:
                rel_diff_med_pct = 0.0
            nz_mask = np.abs(v196) > 1e-5
            if np.any(nz_mask):
                pt_rel = np.abs(v240[nz_mask] - v196[nz_mask]) / np.abs(v196[nz_mask]) * 100.0
                abs_rel_med_pct = float(np.median(pt_rel))
            else:
                abs_rel_med_pct = abs(rel_diff_med_pct)
        else:
            rel_diff_med_pct = np.nan
            abs_rel_med_pct = np.nan

        z_sens_flag = bool(
            (np.isfinite(abs_rel_med_pct) and abs_rel_med_pct > 10.0)
            or (np.isfinite(rel_diff_med_pct) and abs(rel_diff_med_pct) > 10.0)
        )
        is_noise_chk = spec["is_noise_check"] == "True"
        icc_flag = bool(icc_chamber < 0.50 or is_noise_chk)

        dict_rows.append(
            {
                "name": dname,
                "category": spec["category"],
                "definition": spec["definition"],
                "units": spec["units"],
                "scale": spec["scale"],
                "is_noise_check": is_noise_chk,
                "n_valid_sweeps": int(len(s_nom)),
                "n_valid_visits": int(len(v_means)),
                "n_valid_chambers": int(len(ch_means)),
                "median_z0196": round(med_nom, 5) if np.isfinite(med_nom) else np.nan,
                "iqr_z0196": round(iqr_nom, 5) if np.isfinite(iqr_nom) else np.nan,
                "mean_z0196": round(mean_nom, 5) if np.isfinite(mean_nom) else np.nan,
                "sd_z0196": round(sd_nom, 5) if np.isfinite(sd_nom) else np.nan,
                "within_visit_sd_mean": round(within_sd_mean, 5) if np.isfinite(within_sd_mean) else np.nan,
                "within_visit_sd_median": round(within_sd_med, 5) if np.isfinite(within_sd_med) else np.nan,
                "between_chamber_sd": round(between_ch_sd, 5) if np.isfinite(between_ch_sd) else np.nan,
                "between_visit_sd": round(between_v_sd, 5) if np.isfinite(between_v_sd) else np.nan,
                "ICC": round(icc_chamber, 4) if np.isfinite(icc_chamber) else np.nan,
                "ICC_visit_anova": round(icc_visit_anova, 4) if np.isfinite(icc_visit_anova) else np.nan,
                "icc_low_or_noise_flag": icc_flag,
                "median_z0240": round(med_alt, 5) if np.isfinite(med_alt) else np.nan,
                "z_rel_diff_median_pct": round(rel_diff_med_pct, 2) if np.isfinite(rel_diff_med_pct) else np.nan,
                "z_abs_rel_diff_median_pct": round(abs_rel_med_pct, 2) if np.isfinite(abs_rel_med_pct) else np.nan,
                "z_sensitive_gt_10pct": z_sens_flag,
            }
        )

    return pd.DataFrame(dict_rows)


def render_overview_figure(
    df_chambers: pd.DataFrame,
    df_dict: pd.DataFrame,
    example_pred_path: Path,
    example_label: str,
    fig_path: Path,
) -> None:
    """
    Generate `figures/step5_descriptors_overview.png` following House Style:
      - Light background (`#FFFFFF`), ink text (`#111111`), muted captions (`#5C5C5C`),
        hairline rules (`#E3E3E3`), data palette (`#1F4E79`, `#C0561F`, `#2E7D5B`, `#6B4C9A`, `#B3261E`).
      - Panel A (top-left): Distributions of 6 core chamber-level descriptors (at peak CA) across
        the 502 chambers (Contact area, Cup depth hull p95, Cup volume hull, Cup asymmetry,
        T-cell protrusion volume, Target sphericity).
      - Panel B (bottom-left): Intraclass correlation (ICC: between-chamber variance vs within-visit
        sweep variance) bar chart across all 30 descriptors, highlighting the noise floor and
        z-pitch sensitivity (>10% change between z=0.1962 and z=0.2400 um).
      - Panel C (right, top & bottom): Example synaptic contact cross-section (`YX` and `ZX` slices
        through the synaptic cup centroid) showing the segmented T cell, target cell, contact patch,
        fitted non-contact reference sphere contour, 3D convex hull reference chord, and cup depth arrow,
        plus the 2D tangent-plane cup depth map and radial indentation profile.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#C9C9C9",
        "axes.linewidth": 0.8,
        "xtick.color": "#1F1F1F",
        "ytick.color": "#1F1F1F",
        "text.color": "#111111",
    })

    C_PLOT1 = "#1F4E79"
    C_PLOT2 = "#C0561F"
    C_PLOT3 = "#2E7D5B"
    C_PLOT4 = "#6B4C9A"
    C_DANGER = "#B3261E"
    C_MUTED = "#5C5C5C"
    C_RULE = "#E3E3E3"
    C_INK = "#111111"

    fig = plt.figure(figsize=(17.5, 11.8), dpi=180, facecolor="#FFFFFF")
    gs_main = GridSpec(2, 2, figure=fig, width_ratios=[1.10, 1.05], height_ratios=[0.95, 1.25],
                       hspace=0.30, wspace=0.26, left=0.21, right=0.97, top=0.92, bottom=0.06)

    fig.suptitle(
        "Step 5a: Mask-Based Synaptic Shape Descriptors (Scale >= 1.4 um Cutoff)",
        fontsize=15, fontweight="bold", color=C_INK, x=0.04, ha="left", y=0.97
    )

    # ------------------------------------------------------------------
    # PANEL A: 2x3 grid of chamber-level distributions (at peak-CA visit)
    # ------------------------------------------------------------------
    gs_dist = gs_main[0, 0].subgridspec(2, 3, hspace=0.48, wspace=0.32)
    ch_contact = df_chambers[df_chambers["n_contact_visits"] > 0]

    dist_specs = [
        ("contact_area_um2_peak_ca", "Contact area (um^2)", C_PLOT1, (0, 300)),
        ("cup_depth_hull_p95_um_peak_ca", "Cup depth, hull p95 (um)", C_PLOT2, (0, 3.5)),
        ("cup_volume_hull_um3_peak_ca", "Cup volume, hull (um^3)", C_PLOT3, (0, 180)),
        ("cup_asymmetry_norm_peak_ca", "Cup asymmetry (offset / r_eq)", C_PLOT4, (0, 1.0)),
        ("tcell_protrusion_vol_um3_peak_ca", "T-cell protrusion in hull (um^3)", C_PLOT1, (0, 120)),
        ("target_sphericity_peak_ca", "Target 3D sphericity", C_PLOT3, (0.4, 1.0)),
    ]

    for idx_d, (col_n, lbl_n, col_hex, xlim_d) in enumerate(dist_specs):
        r_i, c_i = divmod(idx_d, 3)
        ax = fig.add_subplot(gs_dist[r_i, c_i])
        vals = ch_contact[col_n].dropna().to_numpy(dtype=np.float64)
        vals_clip = np.clip(vals, xlim_d[0], xlim_d[1])
        ax.hist(vals_clip, bins=24, color=col_hex, alpha=0.82, edgecolor="#FFFFFF", linewidth=0.5)
        if len(vals) > 0:
            med_v = float(np.median(vals))
            iqr_v = float(np.percentile(vals, 75.0) - np.percentile(vals, 25.0))
            ax.axvline(med_v, color=C_INK, linestyle="--", linewidth=1.2)
            ax.set_title(f"{lbl_n}\nmed={med_v:.2f} [IQR {iqr_v:.2f}]", fontsize=8.5, fontweight="bold", pad=4)
        else:
            ax.set_title(lbl_n, fontsize=8.5, fontweight="bold")
        ax.set_xlim(xlim_d)
        ax.grid(axis="y", color=C_RULE, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5)
        if c_i == 0:
            ax.set_ylabel("Chambers (n)", fontsize=8)

    # ------------------------------------------------------------------
    # PANEL B: Horizontal bar chart of ICC & Z-pitch sensitivity
    # ------------------------------------------------------------------
    ax_icc = fig.add_subplot(gs_main[1, 0])
    df_d_sorted = df_dict.sort_values("ICC", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_d_sorted))
    icc_vals = df_d_sorted["ICC"].fillna(0.0).to_numpy(dtype=np.float64)
    z_sens = df_d_sorted["z_sensitive_gt_10pct"].to_numpy(dtype=bool)
    is_nc = df_d_sorted["is_noise_check"].to_numpy(dtype=bool)

    bar_colors = []
    for icc_v, zs, nc in zip(icc_vals, z_sens, is_nc):
        if nc or icc_v < 0.50:
            bar_colors.append(C_DANGER)
        elif zs:
            bar_colors.append(C_PLOT2)
        else:
            bar_colors.append(C_PLOT1)

    ax_icc.barh(y_pos, icc_vals, height=0.72, color=bar_colors, alpha=0.88)
    ax_icc.axvline(0.50, color=C_DANGER, linestyle=":", linewidth=1.0)
    ax_icc.axvline(0.75, color=C_MUTED, linestyle="--", linewidth=0.9)

    labels_y = []
    for _, r in df_d_sorted.iterrows():
        tag = ""
        if r["is_noise_check"]:
            tag = " [noise check]"
        elif r["z_sensitive_gt_10pct"]:
            tag = f" [z Δ={r['z_rel_diff_median_pct']:+.0f}%]"
        labels_y.append(f"{r['name']}{tag}")

    ax_icc.set_yticks(y_pos)
    ax_icc.set_yticklabels(labels_y, fontsize=7.2)
    ax_icc.set_xlim(0.0, 1.05)
    ax_icc.set_xlabel("Intraclass Correlation (ICC: between-chamber variance / total variance)", fontsize=9, fontweight="bold")
    ax_icc.set_title(
        "Within-Visit Stability (ICC) & Z Sensitivity (Blue: robust, Orange: |Δz|>10%, Red: low ICC / noise)",
        fontsize=8.6, fontweight="bold", loc="left", pad=6
    )
    ax_icc.grid(axis="x", color=C_RULE, linewidth=0.6)
    ax_icc.spines["top"].set_visible(False)
    ax_icc.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # PANEL C: Example chamber synaptic cup & reference surface geometry
    # ------------------------------------------------------------------
    gs_ex = gs_main[:, 1].subgridspec(2, 2, hspace=0.32, wspace=0.38)

    with h5py.File(example_pred_path, "r") as hf:
        I_ex = hf["predictions"][:]

    sl_t = bbox_slices((I_ex == 1) | (I_ex == 3), 2, I_ex.shape)
    sl_b = bbox_slices((I_ex == 2) | (I_ex == 4), 2, I_ex.shape)
    t_sub = instance_seg_matlab_xyz(I_ex[sl_t] == 1, I_ex[sl_t] == 3)
    b_sub = instance_seg_matlab_xyz(I_ex[sl_b] == 2, I_ex[sl_b] == 4)

    t_full = np.zeros(I_ex.shape, dtype=bool)
    b_full = np.zeros(I_ex.shape, dtype=bool)
    cnts_t = np.bincount(t_sub.ravel()); cnts_t[0] = 0
    cnts_b = np.bincount(b_sub.ravel()); cnts_b[0] = 0
    t_full[sl_t] = fill_solid_cell_mask(t_sub == int(np.argmax(cnts_t)))
    cc_lbl, _ = label(b_sub > 0, structure=STRUCT26)
    cnts_cc = np.bincount(cc_lbl.ravel()); cnts_cc[0] = 0
    b_full[sl_b] = fill_solid_cell_mask(cc_lbl == int(np.argmax(cnts_cc)))

    sl_crop = bbox_slices(t_full | b_full, 8, I_ex.shape)
    t_crop = t_full[sl_crop]
    b_crop = b_full[sl_crop]

    b_surf = b_crop & (~binary_erosion(b_crop, structure=STRUCT26))
    t_surf = t_crop & (~binary_erosion(t_crop, structure=STRUCT26))
    pts_b_um = np.argwhere(b_surf).astype(np.float64) * VOXEL_UM
    pts_t_um = np.argwhere(t_surf).astype(np.float64) * VOXEL_UM

    tree_t = cKDTree(pts_t_um)
    d_bt, _ = tree_t.query(pts_b_um, k=1)
    is_c = d_bt <= (2.6 * VOXEL_UM)
    is_nc = d_bt > (3.5 * VOXEL_UM)
    pts_c_um = pts_b_um[is_c]
    pts_nc_um = pts_b_um[is_nc]

    b_vol_um3 = float(np.sum(b_crop)) * (VOXEL_UM ** 3)
    r_eq_b = float((3.0 * b_vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    b_cent_um = np.argwhere(b_crop).mean(axis=0) * VOXEL_UM
    c_sph_um, R_sph_um = fit_regularized_3d_sphere(pts_nc_um, b_cent_um, r_eq_b)
    hull_ex = ConvexHull(pts_b_um)

    p_patch_um = pts_c_um.mean(axis=0)
    z_cut = int(np.clip(round(p_patch_um[0] / VOXEL_UM), 0, b_crop.shape[0] - 1))
    y_cut = int(np.clip(round(p_patch_um[1] / VOXEL_UM), 0, b_crop.shape[1] - 1))

    # Subplot C1: YX cross-section at z = z_cut
    ax_yx = fig.add_subplot(gs_ex[0, 0])
    H_y, W_x = b_crop.shape[1], b_crop.shape[2]
    ext_yx = [0, W_x * VOXEL_UM, 0, H_y * VOXEL_UM]
    canvas_yx = np.ones((H_y, W_x, 3), dtype=np.float64)
    canvas_yx[b_crop[z_cut]] = [0.85, 0.91, 0.96]  # muted target fill
    canvas_yx[t_crop[z_cut]] = [0.97, 0.88, 0.82]  # muted T-cell fill
    ax_yx.imshow(canvas_yx, origin="lower", extent=ext_yx, aspect="equal")
    ax_yx.contour(b_crop[z_cut].astype(float), levels=[0.5], colors=[C_PLOT1], linewidths=1.5, origin="lower", extent=ext_yx)
    ax_yx.contour(t_crop[z_cut].astype(float), levels=[0.5], colors=[C_PLOT2], linewidths=1.5, origin="lower", extent=ext_yx)

    # Draw non-contact reference sphere slice circle at z = z_cut
    dz_slice = (z_cut * VOXEL_UM) - c_sph_um[0]
    if abs(dz_slice) < R_sph_um:
        r_slice = np.sqrt(R_sph_um ** 2 - dz_slice ** 2)
        theta = np.linspace(0, 2 * np.pi, 180)
        ax_yx.plot(c_sph_um[2] + r_slice * np.cos(theta), c_sph_um[1] + r_slice * np.sin(theta),
                   color=C_INK, linestyle="--", linewidth=1.3, label="Ref sphere (non-contact fit)")

    # Overlay contact patch points near z_cut
    near_z = np.abs(pts_c_um[:, 0] - z_cut * VOXEL_UM) <= (1.5 * VOXEL_UM)
    if np.any(near_z):
        ax_yx.scatter(pts_c_um[near_z, 2], pts_c_um[near_z, 1], s=10, color=C_DANGER, zorder=5, label="Contact patch")

    # 2D convex hull of YX slice target points to illustrate hull chord across cup
    pts_yx_slice = np.argwhere(b_crop[z_cut]).astype(np.float64) * VOXEL_UM
    if len(pts_yx_slice) >= 10:
        hull_2d_yx = ConvexHull(pts_yx_slice[:, [1, 0]])  # (x, y)
        hv = np.append(hull_2d_yx.vertices, hull_2d_yx.vertices[0])
        ax_yx.plot(pts_yx_slice[hv, 1], pts_yx_slice[hv, 0], color=C_PLOT3, linestyle=":", linewidth=1.4, label="Convex hull reference")

    ax_yx.set_title(f"Example Cup Cross-Section (XY slice)\n{example_label}", fontsize=9, fontweight="bold")
    ax_yx.set_xlabel("X (um)", fontsize=8.5)
    ax_yx.set_ylabel("Y (um)", fontsize=8.5)
    ax_yx.legend(loc="upper right", fontsize=6.8, frameon=True, facecolor="#FFFFFF", edgecolor=C_RULE)

    # Subplot C2: ZX cross-section at y = y_cut
    ax_zx = fig.add_subplot(gs_ex[0, 1])
    D_z = b_crop.shape[0]
    ext_zx = [0, W_x * VOXEL_UM, 0, D_z * VOXEL_UM]
    canvas_zx = np.ones((D_z, W_x, 3), dtype=np.float64)
    canvas_zx[b_crop[:, y_cut, :]] = [0.85, 0.91, 0.96]
    canvas_zx[t_crop[:, y_cut, :]] = [0.97, 0.88, 0.82]
    ax_zx.imshow(canvas_zx, origin="lower", extent=ext_zx, aspect="equal")
    ax_zx.contour(b_crop[:, y_cut, :].astype(float), levels=[0.5], colors=[C_PLOT1], linewidths=1.5, origin="lower", extent=ext_zx)
    ax_zx.contour(t_crop[:, y_cut, :].astype(float), levels=[0.5], colors=[C_PLOT2], linewidths=1.5, origin="lower", extent=ext_zx)

    dy_slice = (y_cut * VOXEL_UM) - c_sph_um[1]
    if abs(dy_slice) < R_sph_um:
        r_slice_zx = np.sqrt(R_sph_um ** 2 - dy_slice ** 2)
        theta = np.linspace(0, 2 * np.pi, 180)
        ax_zx.plot(c_sph_um[2] + r_slice_zx * np.cos(theta), c_sph_um[0] + r_slice_zx * np.sin(theta),
                   color=C_INK, linestyle="--", linewidth=1.3, label="Ref sphere")

    near_y = np.abs(pts_c_um[:, 1] - y_cut * VOXEL_UM) <= (1.5 * VOXEL_UM)
    if np.any(near_y):
        ax_zx.scatter(pts_c_um[near_y, 2], pts_c_um[near_y, 0], s=10, color=C_DANGER, zorder=5, label="Contact patch")

    pts_zx_slice = np.argwhere(b_crop[:, y_cut, :]).astype(np.float64) * VOXEL_UM
    if len(pts_zx_slice) >= 10:
        hull_2d_zx = ConvexHull(pts_zx_slice[:, [1, 0]])
        hv = np.append(hull_2d_zx.vertices, hull_2d_zx.vertices[0])
        ax_zx.plot(pts_zx_slice[hv, 1], pts_zx_slice[hv, 0], color=C_PLOT3, linestyle=":", linewidth=1.4, label="Convex hull")

    ax_zx.set_title("Axial Cross-Section (XZ slice)\nTarget (blue), T cell (orange), Patch (red)", fontsize=9, fontweight="bold")
    ax_zx.set_xlabel("X (um)", fontsize=8.5)
    ax_zx.set_ylabel("Z (um, nominal 0.1962 um/vox)", fontsize=8.5)

    # Subplot C3: 2D Tangent-plane indentation depth map w(u, v) on the contact patch
    ax_cup = fig.add_subplot(gs_ex[1, 0])
    n_syn = p_patch_um - b_cent_um
    n_syn /= np.linalg.norm(n_syn)
    arb = np.array([1.0, 0.0, 0.0]) if abs(n_syn[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_syn, arb); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_syn, e1); e2 /= np.linalg.norm(e2)
    rel_c = pts_c_um - p_patch_um
    u_c = rel_c @ e1
    v_c = rel_c @ e2

    eqs = hull_ex.equations
    facing = (eqs[:, :3] @ n_syn) > 0.05
    d_hull_c = np.maximum(0.0, np.min(-((pts_c_um @ eqs[facing, :3].T) + eqs[facing, 3]), axis=1))
    sc = ax_cup.scatter(u_c, v_c, c=d_hull_c, s=14, cmap="Blues", vmin=0.0, vmax=max(1.5, float(np.percentile(d_hull_c, 98))))
    cb = fig.colorbar(sc, ax=ax_cup, fraction=0.046, pad=0.04)
    cb.set_label("Hull cup depth (um)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    deep_mask = d_hull_c >= np.percentile(d_hull_c, 95.0)
    u_deep, v_deep = float(np.mean(u_c[deep_mask])), float(np.mean(v_c[deep_mask]))
    ax_cup.scatter([0.0], [0.0], marker="+", s=70, color=C_INK, linewidths=1.5, label="Patch centroid")
    ax_cup.scatter([u_deep], [v_deep], marker="x", s=60, color=C_DANGER, linewidths=1.8, label="Deepest apex (p95)")
    r_eq_ex = float(np.sqrt(len(pts_c_um) * CA_VOXEL_FACTOR_UM2 / np.pi))
    theta = np.linspace(0, 2 * np.pi, 120)
    ax_cup.plot(r_eq_ex * np.cos(theta), r_eq_ex * np.sin(theta), color=C_MUTED, linestyle="--", linewidth=1.0, label=f"r_eq = {r_eq_ex:.2f} um")
    ax_cup.set_aspect("equal")
    ax_cup.set_xlabel("Tangent axis u (um)", fontsize=8.5)
    ax_cup.set_ylabel("Tangent axis v (um)", fontsize=8.5)
    ax_cup.set_title(
        f"Synaptic Cup Depth Map (p95 = {np.percentile(d_hull_c, 95):.2f} um,\nasymmetry = {np.hypot(u_deep, v_deep)/r_eq_ex:.2f})",
        fontsize=9, fontweight="bold"
    )
    ax_cup.legend(loc="upper right", fontsize=6.8, frameon=True, facecolor="#FFFFFF", edgecolor=C_RULE)

    # Subplot C4: Radial indentation profile (Convex Hull vs Sphere Fit vs Zernike Low-Order Fit)
    ax_prof = fig.add_subplot(gs_ex[1, 1])
    r_rad = np.hypot(u_c, v_c)
    d_sph_c = np.maximum(0.0, R_sph_um - np.linalg.norm(pts_c_um - c_sph_um, axis=1))
    rbins = np.linspace(0.0, float(np.percentile(r_rad, 98)), 10)
    rcent = 0.5 * (rbins[:-1] + rbins[1:])
    b_idx = np.digitize(r_rad, rbins)
    m_hull = [float(np.mean(d_hull_c[b_idx == k])) if np.any(b_idx == k) else np.nan for k in range(1, len(rbins))]
    m_sph = [float(np.mean(d_sph_c[b_idx == k])) if np.any(b_idx == k) else np.nan for k in range(1, len(rbins))]

    ax_prof.scatter(r_rad, d_hull_c, s=5, color=C_PLOT1, alpha=0.18)
    ax_prof.plot(rcent, m_hull, color=C_PLOT1, linewidth=2.0, marker="o", markersize=4, label="Hull depth profile (mean)")
    ax_prof.plot(rcent, m_sph, color=C_PLOT2, linewidth=1.8, linestyle="--", marker="s", markersize=3.5, label="Sphere-fit depth profile")
    ax_prof.axvline(RESOLUTION_BUMP_CUTOFF_UM, color=C_DANGER, linestyle=":", linewidth=1.1, label="Min feature radius (0.7 um)")
    ax_prof.set_xlabel("Radial distance from patch centroid (um)", fontsize=8.5)
    ax_prof.set_ylabel("Inward cup depth (um)", fontsize=8.5)
    ax_prof.set_title("Radial Cup Depth Profile & Resolution Cutoff\n(Concave center -> rim profile)", fontsize=9, fontweight="bold")
    ax_prof.grid(color=C_RULE, linewidth=0.6)
    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)
    ax_prof.legend(loc="upper right", fontsize=7.0, frameon=True, facecolor="#FFFFFF", edgecolor=C_RULE)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def build_tasks_for_chambers(
    experiment_filter: str | None = None,
    roi_filter: list[str] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Build chamber task dicts by joining `data/scidb_19235_masks/manifest.csv` with
    `data/step3_tracked/<experiment>/per_sweep.csv` and `per_chamber_all.csv`.
    For chambers with `trace_num != 1` where `tracked_tcell_id == 0`, falls back to
    `data/step3/<experiment>/per_sweep_contact_area.csv` so all 502 chambers have
    sweep-level contact surface counts when any T cell touches a target.
    """
    manifest_df = pd.read_csv(MANIFEST_CSV)
    if experiment_filter:
        manifest_df = manifest_df[manifest_df["experiment"] == experiment_filter]
    if roi_filter:
        manifest_df = manifest_df[manifest_df["roi"].isin(roi_filter)]

    df_step3_ch = pd.read_csv(STEP3_TRACKED_DIR / "per_chamber_all.csv")
    if experiment_filter:
        df_step3_ch = df_step3_ch[df_step3_ch["experiment"] == experiment_filter]
    if roi_filter:
        df_step3_ch = df_step3_ch[df_step3_ch["roi"].isin(roi_filter)]

    # Map (experiment, roi, visit, sweep) -> rel_path
    actin_manifest = manifest_df[manifest_df["sweep_index"] != SWEEPS_PER_VISIT].copy()

    tasks: list[dict[str, Any]] = []
    for exp_name, ch_exp_df in df_step3_ch.groupby("experiment", sort=False):
        sw_tr_path = STEP3_TRACKED_DIR / str(exp_name) / "per_sweep.csv"
        sw_un_path = STEP3_DIR / str(exp_name) / "per_sweep_contact_area.csv"
        df_sw_tr = pd.read_csv(sw_tr_path)
        df_sw_un = pd.read_csv(sw_un_path) if sw_un_path.exists() else None

        if df_sw_un is not None:
            df_sw_merged = df_sw_tr.merge(
                df_sw_un[["roi", "visit", "sweep", "n_tcell_instances", "n_target_instances", "s_largest", "s_total", "n_components"]],
                on=["roi", "visit", "sweep"],
                how="left",
                suffixes=("", "_untracked"),
            )
        else:
            df_sw_merged = df_sw_tr.copy()

        m_exp = actin_manifest[actin_manifest["experiment"] == exp_name]
        path_lookup = {
            (str(r["roi"]), int(r["visit_index"]), int(r["sweep_index"])): str(r["path"])
            for _, r in m_exp.iterrows()
        }

        for _, ch_row in ch_exp_df.sort_values("roi_index").iterrows():
            roi = str(ch_row["roi"])
            roi_idx = int(ch_row["roi_index"])
            gid = str(ch_row["group_id"])
            cond = str(ch_row["condition"])
            tr_num = int(ch_row["trace_num"])
            pk_visit = int(ch_row["peak_visit"])

            sw_roi = df_sw_merged[df_sw_merged["roi"] == roi].sort_values(["visit", "sweep"])
            sweep_meta_list: list[dict[str, Any]] = []
            for _, srow in sw_roi.iterrows():
                v_i = int(srow["visit"])
                s_i = int(srow["sweep"])
                rel_p = path_lookup.get((roi, v_i, s_i))
                if rel_p is None:
                    continue
                tr_id = int(srow["tracked_tcell_id"])
                if tr_num == 1 or tr_id > 0 or df_sw_un is None:
                    n_t = int(srow["n_tcell_instances"])
                    n_b = int(srow["n_target_instances"])
                    s_l = int(srow["s_largest"])
                    s_tot = int(srow["s_total"])
                    n_c = int(srow["n_components"])
                else:
                    # Non-single-track chamber (`trace_num == 0` or `>= 2` when Track 1 is absent):
                    # use untracked touching T-cell surface if present
                    n_t = int(srow.get("n_tcell_instances_untracked", 0))
                    n_b = int(srow.get("n_target_instances_untracked", 0))
                    s_l = int(srow.get("s_largest_untracked", 0))
                    s_tot = int(srow.get("s_total_untracked", 0))
                    n_c = int(srow.get("n_components_untracked", 0))

                sweep_meta_list.append(
                    {
                        "visit": v_i,
                        "sweep": s_i,
                        "timepoint_idx": int(srow["timepoint_idx"]),
                        "rel_path": rel_p,
                        "tracked_tcell_id": tr_id,
                        "n_tcell_instances": n_t,
                        "n_target_instances": n_b,
                        "s_largest": s_l,
                        "s_total": s_tot,
                        "n_components": n_c,
                    }
                )

            tasks.append(
                {
                    "experiment": str(exp_name),
                    "roi": roi,
                    "roi_index": roi_idx,
                    "group_id": gid,
                    "condition": cond,
                    "trace_num": tr_num,
                    "peak_visit": pk_visit,
                    "sweep_meta_list": sweep_meta_list,
                }
            )

    return tasks, df_step3_ch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5a: Compute mask-based shape descriptors of T-cell/target contact across all 502 chambers."
    )
    parser.add_argument("--experiment", type=str, default=None, help="Optional experiment filter for testing.")
    parser.add_argument("--rois", type=str, nargs="*", default=None, help="Optional ROI filter list for testing.")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes (default: 6, leaving 4 cores headroom).")
    parser.add_argument("--out-dir", type=Path, default=STEP5_DIR, help="Output directory (default: data/step5).")
    parser.add_argument("--skip-figure", action="store_true", help="Skip generating overview figure (for quick tests).")
    args = parser.parse_args()

    workers = min(6, max(1, int(args.workers)))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    tasks, df_step3_ch = build_tasks_for_chambers(
        experiment_filter=args.experiment, roi_filter=args.rois
    )
    if not tasks:
        print("No matching chambers found.")
        return 1

    print(
        f"Running Step 5a contact shape descriptors on {len(tasks)} chamber(s) "
        f"with {workers} worker(s) (resolution cutoff >= {RESOLUTION_WAVELENGTH_CUTOFF_UM:.2f} um)...",
        flush=True,
    )

    sweep_dfs: list[pd.DataFrame] = []
    cpu_times: list[float] = []

    if workers <= 1 or len(tasks) == 1:
        for idx_done, task in enumerate(tasks, start=1):
            res = process_one_chamber_descriptors(task)
            sweep_dfs.append(res["sweep_df"])
            cpu_times.append(float(res["elapsed_s"]))
            el = time.perf_counter() - t_start
            print(
                f"  [{idx_done:3d}/{len(tasks):3d}] {res['experiment']}/{res['roi']} "
                f"({res['elapsed_s']:.2f}s CPU, wall={el:.1f}s)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(process_one_chamber_descriptors, t): t for t in tasks}
            for idx_done, fut in enumerate(as_completed(fut_map), start=1):
                res = fut.result()
                sweep_dfs.append(res["sweep_df"])
                cpu_times.append(float(res["elapsed_s"]))
                if idx_done % 20 == 0 or idx_done == len(tasks):
                    el = time.perf_counter() - t_start
                    print(
                        f"  [{idx_done:3d}/{len(tasks):3d}] latest={res['experiment']}/{res['roi']} "
                        f"({res['elapsed_s']:.2f}s CPU, wall={el:.1f}s, {el / idx_done:.2f}s wall/chamber)",
                        flush=True,
                    )

    df_sweeps = (
        pd.concat(sweep_dfs, ignore_index=True)
        .sort_values(["experiment", "roi_index", "visit", "sweep"])
        .reset_index(drop=True)
    )

    print("Aggregating per-visit and per-chamber descriptor tables...", flush=True)
    df_visits, df_chambers = aggregate_visits_and_chambers(df_sweeps, df_step3_ch)
    df_dict = compute_icc_and_z_sensitivity_dictionary(df_sweeps, df_visits, df_chambers)

    sw_csv = out_dir / "per_sweep_descriptors.csv"
    v_csv = out_dir / "per_visit_descriptors.csv"
    ch_csv = out_dir / "per_chamber_descriptors.csv"
    dict_csv = out_dir / "descriptor_dictionary.csv"

    df_sweeps.to_csv(sw_csv, index=False)
    df_visits.to_csv(v_csv, index=False)
    df_chambers.to_csv(ch_csv, index=False)
    df_dict.to_csv(dict_csv, index=False)

    wall_elapsed_s = time.perf_counter() - t_start
    summary_meta = {
        "n_chambers": int(len(df_chambers)),
        "n_visits": int(len(df_visits)),
        "n_sweeps": int(len(df_sweeps)),
        "n_contact_sweeps": int(df_sweeps["in_contact"].sum()),
        "n_contact_visits": int(df_visits["in_contact_visit"].sum()),
        "n_chambers_with_contact": int((df_chambers["n_contact_visits"] > 0).sum()),
        "n_chambers_trace_num_eq_1": int((df_chambers["trace_num"] == 1).sum()),
        "workers": workers,
        "wall_elapsed_s": round(wall_elapsed_s, 3),
        "wall_s_per_chamber": round(wall_elapsed_s / max(1, len(df_chambers)), 3),
        "cpu_s_per_chamber_mean": round(float(np.mean(cpu_times)), 3),
        "cpu_s_total": round(float(np.sum(cpu_times)), 3),
        "z_nominal_um": Z_NOMINAL_UM,
        "z_alt_um": Z_ALT_UM,
        "resolution_wavelength_cutoff_um": RESOLUTION_WAVELENGTH_CUTOFF_UM,
        "resolution_bump_cutoff_um": RESOLUTION_BUMP_CUTOFF_UM,
    }
    (out_dir / "summary_step5a.json").write_text(json.dumps(summary_meta, indent=2), encoding="utf-8")

    if not args.skip_figure:
        ex_path = REPO_ROOT / "data/scidb_19235_masks/v01_6_40_120_0-6_hd1/MIP_ROI_32/01153_predictions.h5"
        if not ex_path.exists():
            ex_path = REPO_ROOT / tasks[0]["sweep_meta_list"][0]["rel_path"]
            ex_lbl = f"{tasks[0]['experiment']} {tasks[0]['roi']}"
        else:
            ex_lbl = "v01_6_40_120_0-6_hd1 / MIP_ROI_32 (Visit 5, Sweep 1)"
        print(f"Rendering overview figure to {FIG_OVERVIEW_PATH}...", flush=True)
        render_overview_figure(df_chambers, df_dict, ex_path, ex_lbl, FIG_OVERVIEW_PATH)

    print(
        f"Done in {wall_elapsed_s:.2f}s wall ({wall_elapsed_s / max(1, len(df_chambers)):.2f}s/chamber; "
        f"CPU mean {np.mean(cpu_times):.2f}s/chamber). Saved {len(df_sweeps)} sweeps, "
        f"{len(df_visits)} visits, {len(df_chambers)} chambers.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
