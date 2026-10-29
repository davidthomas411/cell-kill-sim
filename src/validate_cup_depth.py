#!/usr/bin/env python3
"""
Validate the 3D convex-hull cup-depth measurement (`cup_depth_hull_p95_um` in
`src/contact_descriptors.py` lines 709-715) on synthetic 3D voxel masks with known
analytical ground-truth answers at the pipeline's voxel pitch (`0.1962 um`), and
re-measure the 6 `qc_clean` and 16 `cup_measurement` panels from `figures/review/index.csv`
comparing the pipeline's facet-filtered distance (`n . n_syn > 0.05`) against the
unfiltered all-facets convex-hull distance.

Outputs:
  - `data/step5/cup_depth_synthetic_validation.csv`
  - `data/step5/cup_depth_real_panels_comparison.csv`
  - `figures/review/cup_validation/<case>.png`
  - `figures/cup_depth_validation.png`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.spatial import ConvexHull, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.paths import DATA_DIR, FIGURES_DIR
from src.reproduce_contact_area import MANIFEST_CSV, STRUCT26, bbox_slices
from src.contact_descriptors import (
    compute_descriptors_for_z_pitch,
    fast_instance_seg_with_hint,
    fill_solid_cell_mask,
)

# House-style tokens
COL_PAPER = "#FFFFFF"
COL_PAPER2 = "#F6F6F6"
COL_INK = "#111111"
COL_TEXT = "#1F1F1F"
COL_MUTED = "#5C5C5C"
COL_RULE = "#E3E3E3"
COL_LINE = "#C9C9C9"
COL_DANGER = "#B3261E"
COL_PLOT1 = "#1F4E79"  # Target mask / recorded series
COL_PLOT2 = "#C0561F"  # T-cell mask / sphere series
COL_PLOT3 = "#2E7D5B"  # Convex hull / all-facets series
COL_PLOT4 = "#6B4C9A"  # Contact patch voxels


def setup_house_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "-apple-system",
                "BlinkMacSystemFont",
                "Segoe UI",
                "Roboto",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.edgecolor": COL_LINE,
            "axes.linewidth": 0.8,
            "xtick.color": COL_INK,
            "ytick.color": COL_INK,
            "text.color": COL_INK,
            "figure.facecolor": COL_PAPER,
            "axes.facecolor": COL_PAPER,
            "savefig.facecolor": COL_PAPER,
        }
    )


def intersect_mesh_with_plane(
    V_um: np.ndarray, F: np.ndarray, axis: int, val_um: float
) -> np.ndarray:
    """Exact intersection of 3D mesh `(V_um, F)` with coordinate plane `V_um[:, axis] == val_um`."""
    if V_um is None or F is None or len(F) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    v0 = V_um[F[:, 0]]
    v1 = V_um[F[:, 1]]
    v2 = V_um[F[:, 2]]
    d0 = v0[:, axis] - val_um
    d1 = v1[:, axis] - val_um
    d2 = v2[:, axis] - val_um

    edges = [((v0, d0), (v1, d1)), ((v1, d1), (v2, d2)), ((v2, d2), (v0, d0))]
    cross_list = []
    pts_list = []
    for (va, da), (vb, db) in edges:
        cross = ((da <= 0.0) & (db > 0.0)) | ((da >= 0.0) & (db < 0.0))
        denom = da - db
        t = np.divide(da, denom, out=np.full_like(da, 0.5, dtype=np.float64), where=np.abs(denom) > 1e-12)
        p = va + t[:, None] * (vb - va)
        cross_list.append(cross)
        pts_list.append(p)

    c01, c12, c20 = cross_list
    p01, p12, p20 = pts_list
    active = np.where((c01.astype(int) + c12.astype(int) + c20.astype(int)) >= 2)[0]
    if len(active) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    segs = []
    for idx in active:
        pts_i = []
        if c01[idx]:
            pts_i.append(p01[idx])
        if c12[idx]:
            pts_i.append(p12[idx])
        if c20[idx]:
            pts_i.append(p20[idx])
        if len(pts_i) >= 2:
            segs.append((pts_i[0], pts_i[1]))
    if not segs:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.asarray(segs, dtype=np.float64)


def make_grid(n_vox: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(Z, Y, X, R_perp)` in micrometers centered at grid midpoint."""
    zz, yy, xx = np.mgrid[:n_vox, :n_vox, :n_vox]
    c0 = (n_vox // 2) * VOXEL_UM
    Z = zz * VOXEL_UM - c0
    Y = yy * VOXEL_UM - c0
    X = xx * VOXEL_UM - c0
    R_perp = np.hypot(Y, Z)
    return Z, Y, X, R_perp


def build_flat_contact_masks(
    R_um: float, a_um: float, n_vox: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Case (a): Target sphere of radius `R_um` with a planar cut removing the cap so the flat
    circular contact face has radius `a_eff = min(a_um, R_um)`, and a T cell pressed flat
    against the cut face separated by 1 voxel (`VOXEL_UM`).
    """
    Z, Y, X, _ = make_grid(n_vox)
    a_eff = min(float(a_um), float(R_um))
    x_cut = float(np.sqrt(max(0.0, R_um * R_um - a_eff * a_eff)))
    b_mask = (X * X + Y * Y + Z * Z <= R_um * R_um) & (X <= x_cut)

    # T cell pressed flat against x = x_cut + VOXEL_UM with contact radius a_eff
    R_t = float(np.hypot(a_eff, 2.5))
    x_tc = x_cut + VOXEL_UM - 2.5
    t_mask = ((X - x_tc) ** 2 + Y * Y + Z * Z <= R_t * R_t) & (X >= x_cut + VOXEL_UM)
    t_mask &= ~b_mask
    return b_mask, t_mask


def build_spherical_dent_masks(
    R_um: float, a_um: float, d_um: float, n_vox: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Case (b): Target sphere of radius `R_um` with a concave spherical dent of depth `d_um`
    (measured inward from the convex-hull rim plane `x = sqrt(R^2 - a^2)`) over patch radius
    `a_eff = min(a_um, R_um)`, and a T cell filling the dent separated by 1 voxel (`VOXEL_UM`).
    """
    Z, Y, X, R_perp = make_grid(n_vox)
    a_eff = min(float(a_um), float(R_um))
    x_rim = float(np.sqrt(max(0.0, R_um * R_um - a_eff * a_eff)))
    R_dent = (a_eff * a_eff + d_um * d_um) / (2.0 * d_um)
    x_c_dent = x_rim - d_um + R_dent

    in_target = (X * X + Y * Y + Z * Z) <= (R_um * R_um)
    in_dent = ((X - x_c_dent) ** 2 + Y * Y + Z * Z) < (R_dent * R_dent)
    in_cap_above_rim = (X > x_rim) & (R_perp <= a_eff)
    b_mask = in_target & (~in_dent) & (~in_cap_above_rim)

    R_t_dent = max(0.4, R_dent - VOXEL_UM)
    t_dent = (
        (((X - x_c_dent) ** 2 + Y * Y + Z * Z) <= (R_t_dent * R_t_dent))
        & (R_perp <= a_eff)
        & (X <= x_rim + 0.4)
    )
    R_t_body = float(np.hypot(a_eff, 2.2))
    x_t_body = x_rim + VOXEL_UM - 2.2
    t_body = (((X - x_t_body) ** 2 + Y * Y + Z * Z) <= (R_t_body * R_t_body)) & (
        X >= x_rim + VOXEL_UM
    )
    t_mask = (t_dent | t_body) & (~b_mask)
    return b_mask, t_mask


def build_wide_bowl_masks(
    R_um: float = 6.0, a_um: float = 6.0, sag_um: float = 0.3, n_vox: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Case (d): Wide gently curved contact (target flattened into a large-radius bowl with
    sag `0.3 um` over patch radius `a = 6 um`). For `R_um < a_um`, the target is an oblate
    spheroid of lateral radius `a_um` and axial radius `R_um`.
    """
    Z, Y, X, R_perp = make_grid(n_vox)
    r_lat = max(float(R_um), float(a_um))
    x_rim = float(R_um * np.sqrt(max(0.0, 1.0 - (a_um / r_lat) ** 2)))
    R_bowl = (a_um * a_um + sag_um * sag_um) / (2.0 * sag_um)
    x_c_bowl = x_rim - sag_um + R_bowl

    in_target = ((X / R_um) ** 2 + (Y / r_lat) ** 2 + (Z / r_lat) ** 2) <= 1.0
    in_bowl = ((X - x_c_bowl) ** 2 + Y * Y + Z * Z) < (R_bowl * R_bowl)
    in_cap = (X > x_rim) & (R_perp <= a_um)
    b_mask = in_target & (~in_bowl) & (~in_cap)

    R_t_bowl = max(0.5, R_bowl - VOXEL_UM)
    t_dent = (
        (((X - x_c_bowl) ** 2 + Y * Y + Z * Z) <= (R_t_bowl * R_t_bowl))
        & (R_perp <= a_um)
        & (X <= x_rim + 0.4)
    )
    R_t_body = float(np.hypot(a_um, 2.5))
    x_t_body = x_rim + VOXEL_UM - 2.5
    t_body = (((X - x_t_body) ** 2 + Y * Y + Z * Z) <= (R_t_body * R_t_body)) & (
        X >= x_rim + VOXEL_UM
    )
    t_mask = (t_dent | t_body) & (~b_mask)
    return b_mask, t_mask


def build_two_spheres_masks(
    R_um: float = 5.0, n_vox: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Case (e): Two separate un-indented target spheres of radius `R_um` merged into a single
    target mask with the T cell wedged between them.
    """
    Z, Y, X, _ = make_grid(n_vox)
    y_sep = R_um - 0.15  # Slight overlap/touch or close pair merged as one target mask
    b1 = (X + 1.8) ** 2 + (Y - y_sep) ** 2 + Z * Z <= (R_um * R_um)
    b2 = (X + 1.8) ** 2 + (Y + y_sep) ** 2 + Z * Z <= (R_um * R_um)
    b_mask = b1 | b2

    r_t = 0.82 * R_um
    t_raw = (X - 1.2) ** 2 + Y * Y + Z * Z <= (r_t * r_t)
    t_mask = t_raw & (~binary_dilation(b_mask, structure=STRUCT26))
    return b_mask, t_mask


def build_hollow_shell_masks(
    R_um: float = 5.0,
    a_um: float = 3.0,
    d_um: float = 1.0,
    thickness_um: float = 0.8,
    apply_fill_solid: bool = False,
    n_vox: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Case (f): Hollow shell target of outer radius `R_um`, wall thickness `0.8 um`, and
    spherical dent of depth `1.0 um` over patch radius `a_um = 3.0 um`.
    """
    Z, Y, X, R_perp = make_grid(n_vox)
    a_eff = min(float(a_um), float(R_um))
    x_rim = float(np.sqrt(max(0.0, R_um * R_um - a_eff * a_eff)))
    R_dent = (a_eff * a_eff + d_um * d_um) / (2.0 * d_um)
    x_c_dent = x_rim - d_um + R_dent

    outer = (
        ((X * X + Y * Y + Z * Z) <= (R_um * R_um))
        & (~(((X - x_c_dent) ** 2 + Y * Y + Z * Z) < (R_dent * R_dent)))
        & (~((X > x_rim) & (R_perp <= a_eff)))
    )
    r_in = max(0.5, R_um - thickness_um)
    r_dent_in = R_dent + thickness_um
    inner_cavity = ((X * X + Y * Y + Z * Z) <= (r_in * r_in)) & (
        ~(((X - x_c_dent) ** 2 + Y * Y + Z * Z) < (r_dent_in * r_dent_in))
    )
    b_mask = outer & (~inner_cavity)
    if apply_fill_solid:
        b_mask = fill_solid_cell_mask(b_mask)

    R_t_dent = max(0.4, R_dent - VOXEL_UM)
    t_dent = (
        (((X - x_c_dent) ** 2 + Y * Y + Z * Z) <= (R_t_dent * R_t_dent))
        & (R_perp <= a_eff)
        & (X <= x_rim + 0.4)
    )
    R_t_body = float(np.hypot(a_eff, 2.2))
    x_t_body = x_rim + VOXEL_UM - 2.2
    t_body = (((X - x_t_body) ** 2 + Y * Y + Z * Z) <= (R_t_body * R_t_body)) & (
        X >= x_rim + VOXEL_UM
    )
    t_mask = (t_dent | t_body) & (~outer)
    return b_mask, t_mask


def evaluate_masks_pipeline_and_all_facets(
    b_mask: np.ndarray,
    t_mask: np.ndarray,
    tilt_deg: float = 0.0,
) -> dict[str, Any]:
    """
    Run the exact pipeline contact-patch construction (`src/contact_descriptors.py` lines 935-955)
    and `compute_descriptors_for_z_pitch` (lines 527-765) at `dz_um = VOXEL_UM = 0.1962 um`,
    plus compute the ALL-facets convex-hull distance (`min` over all hull facets without the
    `facing = (eqs[:, :3] @ n_syn) > 0.05` filter).

    When `tilt_deg != 0`, `b_all_vox` centroid is shifted laterally so that `compute_descriptors_for_z_pitch`
    internally computes `n_syn` tilted by `tilt_deg` degrees from the true synapse axis `(0, 0, 1)`
    while keeping `b_surf_vox` and `len(b_all_vox)` unchanged.
    """
    t_all_vox = np.argwhere(t_mask).astype(np.float64)
    t_surf_mask = t_mask & (~binary_erosion(t_mask, structure=STRUCT26))
    t_surf_vox = np.argwhere(t_surf_mask).astype(np.float64)

    b_all_vox_orig = np.argwhere(b_mask).astype(np.float64)
    b_surf_mask = b_mask & (~binary_erosion(b_mask, structure=STRUCT26))
    b_surf_vox = np.argwhere(b_surf_mask).astype(np.float64)

    tree_t = cKDTree(t_surf_vox)
    d_bt, _ = tree_t.query(b_surf_vox, k=1)
    within_25 = d_bt <= 2.6
    if int(np.sum(within_25)) >= 12:
        is_c = within_25
    else:
        k_sel = min(len(d_bt), max(12, int(np.sum(within_25))))
        thr_d = min(4.5, float(np.partition(d_bt, k_sel - 1)[k_sel - 1]))
        is_c = d_bt <= thr_d
    is_nc = d_bt > 3.5
    s_largest = int(np.sum(is_c))

    # First run unshifted to get exact sphere depth when tilt_deg != 0
    desc_unshifted = compute_descriptors_for_z_pitch(
        dz_um=VOXEL_UM,
        s_largest_vox=s_largest,
        s_total_vox=s_largest,
        n_components=1,
        t_mask_sub=t_mask,
        t_surf_vox=t_surf_vox,
        t_all_vox=t_all_vox,
        b_cell_sub=b_mask,
        b_vor_vox_count=len(b_all_vox_orig),
        b_surf_vox=b_surf_vox,
        b_all_vox=b_all_vox_orig,
        is_contact_on_b_surf=is_c,
        is_noncontact_on_b_surf=is_nc,
    )

    b_all_vox_for_desc = b_all_vox_orig.copy()
    p_patch_vox = b_surf_vox[is_c].mean(axis=0)
    b_center_vox = b_all_vox_orig.mean(axis=0)
    if abs(tilt_deg) > 1e-6:
        # Tilt n_syn in the Y-X plane by tilt_deg relative to the true +X axis (0, 0, 1)
        th = np.deg2rad(float(tilt_deg))
        dist_vox = max(10.0, float(np.linalg.norm(p_patch_vox - b_center_vox)))
        desired_n_syn = np.array([0.0, np.sin(th), np.cos(th)], dtype=np.float64)
        desired_b_center_vox = p_patch_vox - dist_vox * desired_n_syn
        shift_vox = desired_b_center_vox - b_center_vox
        b_all_vox_for_desc = b_all_vox_orig + shift_vox[None, :]

    desc = compute_descriptors_for_z_pitch(
        dz_um=VOXEL_UM,
        s_largest_vox=s_largest,
        s_total_vox=s_largest,
        n_components=1,
        t_mask_sub=t_mask,
        t_surf_vox=t_surf_vox,
        t_all_vox=t_all_vox,
        b_cell_sub=b_mask,
        b_vor_vox_count=len(b_all_vox_orig),
        b_surf_vox=b_surf_vox,
        b_all_vox=b_all_vox_for_desc,
        is_contact_on_b_surf=is_c,
        is_noncontact_on_b_surf=is_nc,
    )

    pts_b_um = b_surf_vox * VOXEL_UM
    pts_c_um = pts_b_um[is_c]
    hull = ConvexHull(pts_b_um)
    eqs = hull.equations

    b_center_um = b_all_vox_for_desc.mean(axis=0) * VOXEL_UM
    t_center_um = t_all_vox.mean(axis=0) * VOXEL_UM
    n_syn = pts_c_um.mean(axis=0) - b_center_um
    if np.dot(t_center_um - b_center_um, n_syn) < 0:
        n_syn = t_center_um - b_center_um
    n_syn = n_syn / max(1e-9, float(np.linalg.norm(n_syn)))

    # Recorded (facing > 0.05)
    facing = (eqs[:, :3] @ n_syn) > 0.05
    if not np.any(facing):
        facing = np.ones(len(eqs), dtype=bool)
    eq_f = eqs[facing]
    dists_f = -((pts_c_um @ eq_f[:, :3].T) + eq_f[:, 3])
    min_idx_f = np.argmin(dists_f, axis=1)
    d_hull_rec = np.maximum(0.0, np.min(dists_f, axis=1))
    p95_rec = float(np.percentile(d_hull_rec, 95.0))
    idx_p95_rec = int(np.argmin(np.abs(d_hull_rec - p95_rec)))
    pt_c_rec_um = pts_c_um[idx_p95_rec]
    pt_h_rec_um = pt_c_rec_um + p95_rec * eq_f[min_idx_f[idx_p95_rec], :3]

    # All facets (min over ALL hull facets)
    dists_all = -((pts_c_um @ eqs[:, :3].T) + eqs[:, 3])
    min_idx_all = np.argmin(dists_all, axis=1)
    d_hull_all = np.maximum(0.0, np.min(dists_all, axis=1))
    p95_all = float(np.percentile(d_hull_all, 95.0))
    idx_p95_all = int(np.argmin(np.abs(d_hull_all - p95_all)))
    pt_c_all_um = pts_c_um[idx_p95_all]
    pt_h_all_um = pt_c_all_um + p95_all * eqs[min_idx_all[idx_p95_all], :3]

    return {
        "cup_depth_hull_p95_um": float(desc["cup_depth_hull_p95_um"]),
        "cup_depth_hull_max_um": float(desc["cup_depth_hull_max_um"]),
        "cup_depth_all_facets_p95_um": p95_all,
        "cup_depth_all_facets_max_um": float(np.max(d_hull_all)),
        "cup_depth_sphere_p95_um": float(desc_unshifted["cup_depth_sphere_p95_um"]),
        "patch_equiv_radius_um": float(desc["patch_equiv_radius_um"]),
        "contact_area_um2": float(desc["contact_area_um2"]),
        "n_facets_total": int(len(eqs)),
        "n_facets_facing": int(np.sum(facing)),
        "n_syn": n_syn,
        "hull": hull,
        "pts_b_um": pts_b_um,
        "pts_c_um": pts_c_um,
        "is_c": is_c,
        "b_surf_vox": b_surf_vox,
        "pt_c_rec_vox": pt_c_rec_um / VOXEL_UM,
        "pt_h_rec_vox": pt_h_rec_um / VOXEL_UM,
        "pt_c_all_vox": pt_c_all_um / VOXEL_UM,
        "pt_h_all_vox": pt_h_all_um / VOXEL_UM,
        "p_center_vox": p_patch_vox,
    }


def render_synthetic_case_panel(
    case_id: str,
    title_str: str,
    b_mask: np.ndarray,
    t_mask: np.ndarray,
    eval_res: dict[str, Any],
    expected_um: float | None,
    out_path: Path,
) -> None:
    """
    Render XY and XZ slice panels of a synthetic case with target mask, T-cell mask,
    contact patch voxels, 3D convex hull cross-section, and both depths marked.
    """
    setup_house_style()
    Z_dim, Y_dim, X_dim = b_mask.shape
    p_c = eval_res["p_center_vox"]
    zc = int(np.clip(np.round(p_c[0]), 0, Z_dim - 1))
    yc = int(np.clip(np.round(p_c[1]), 0, Y_dim - 1))
    xc = int(np.clip(np.round(p_c[2]), 0, X_dim - 1))

    contact_full = np.zeros_like(b_mask, dtype=bool)
    c_vox = eval_res["b_surf_vox"][eval_res["is_c"]].astype(int)
    contact_full[c_vox[:, 0], c_vox[:, 1], c_vox[:, 2]] = True

    union_mask = b_mask | t_mask
    coords = np.argwhere(union_mask)
    pad = 12
    z0, y0, x0 = np.maximum(0, coords.min(axis=0) - pad)
    z1, y1, x1 = np.minimum([Z_dim, Y_dim, X_dim], coords.max(axis=0) + pad)
    side = int(max(z1 - z0, y1 - y0, x1 - x0, 72))
    side = min(side, Z_dim)

    def _win(c_idx: int, lim: int) -> tuple[int, int]:
        lo = int(np.clip(c_idx - side // 2, 0, lim - side))
        return lo, lo + side

    z0, z1 = _win( (z0 + z1) // 2, Z_dim )
    y0, y1 = _win( (y0 + y1) // 2, Y_dim )
    x0, x1 = _win( (x0 + x1) // 2, X_dim )

    fig = plt.figure(figsize=(10.0, 4.3), dpi=110)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.05, 1.05, 0.92], wspace=0.14, left=0.03, right=0.98, top=0.86, bottom=0.06
    )
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    fig.suptitle(
        f"Synthetic validation | {case_id} | {title_str}",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )

    for ax, plane, sl_idx in [(ax_xy, "xy", zc), (ax_xz, "xz", yc)]:
        if plane == "xy":
            b_sl = b_mask[zc, y0:y1, x0:x1]
            t_sl = t_mask[zc, y0:y1, x0:x1]
            c_sl = contact_full[max(0, zc - 1) : min(Z_dim, zc + 2), y0:y1, x0:x1].any(axis=0)
            axis_i = 0
            val_um = zc * VOXEL_UM
        else:
            b_sl = b_mask[z0:z1, yc, x0:x1]
            t_sl = t_mask[z0:z1, yc, x0:x1]
            c_sl = contact_full[z0:z1, max(0, yc - 1) : min(Y_dim, yc + 2), x0:x1].any(axis=1)
            axis_i = 1
            val_um = yc * VOXEL_UM

        h_s, w_s = b_sl.shape
        rgb = np.ones((h_s, w_s, 3), dtype=np.float32)
        rgb[b_sl] = [0.85, 0.90, 0.96]
        rgb[t_sl] = [0.96, 0.89, 0.84]
        ax.imshow(rgb, interpolation="nearest", origin="upper")

        if np.any(b_sl):
            ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.8, zorder=4)
        if np.any(t_sl):
            ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_PLOT2], linewidths=1.6, zorder=4)

        # 3D convex hull slice intersection
        hull = eval_res["hull"]
        segs_3d = intersect_mesh_with_plane(eval_res["pts_b_um"], hull.simplices, axis=axis_i, val_um=val_um)
        if len(segs_3d) > 0:
            segs_vox = segs_3d / VOXEL_UM
            xs = segs_vox[:, :, 2] - x0
            ys = (segs_vox[:, :, 1] - y0) if plane == "xy" else (segs_vox[:, :, 0] - z0)
            lc = LineCollection(
                np.stack([xs, ys], axis=-1),
                colors=COL_PLOT3,
                linewidths=1.5,
                linestyles="--",
                zorder=5,
            )
            ax.add_collection(lc)

        # Contact patch voxels
        cy_pts, cx_pts = np.where(c_sl)
        if len(cx_pts) > 0:
            ax.scatter(cx_pts, cy_pts, s=8.0, c=COL_PLOT4, marker="s", alpha=0.85, linewidths=0, zorder=6)

        # Mark Recorded depth vector (#B3261E) and All-facets depth vector (#2E7D5B)
        rec_val = eval_res["cup_depth_hull_p95_um"]
        all_val = eval_res["cup_depth_all_facets_p95_um"]
        for d_val, pc_vox, ph_vox, col, dy_off in [
            (rec_val, eval_res["pt_c_rec_vox"], eval_res["pt_h_rec_vox"], COL_DANGER, -1.2),
            (all_val, eval_res["pt_c_all_vox"], eval_res["pt_h_all_vox"], COL_PLOT3, +1.2),
        ]:
            if d_val > 0.02:
                xc_p = pc_vox[2] - x0
                xh_p = ph_vox[2] - x0
                yc_p = (pc_vox[1] - y0 if plane == "xy" else pc_vox[0] - z0) + dy_off
                yh_p = (ph_vox[1] - y0 if plane == "xy" else ph_vox[0] - z0) + dy_off
                ax.annotate(
                    "",
                    xy=(xc_p, yc_p),
                    xytext=(xh_p, yh_p),
                    arrowprops={"arrowstyle": "<->", "color": col, "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
                    zorder=12,
                )

        # 5 um scale bar
        bar_px = 5.0 / VOXEL_UM
        xr = w_s - 6
        xl = max(4, xr - bar_px)
        yb = h_s - 6
        ax.plot([xl, xr], [yb, yb], color=COL_INK, lw=2.2, zorder=15)
        ax.text(0.5 * (xl + xr), yb - 3, "5 um", fontsize=8.0, fontweight="bold", ha="center", va="bottom", color=COL_INK)
        ax.set_title(
            f"{plane.upper()} slice ({'z' if plane == 'xy' else 'y'} = {sl_idx} vox)",
            fontsize=9.5,
            fontweight="bold",
            color=COL_INK,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    exp_txt = f"{expected_um:.3f} um" if expected_um is not None and np.isfinite(expected_um) else "N/A (artifact test)"
    rec_v = eval_res["cup_depth_hull_p95_um"]
    all_v = eval_res["cup_depth_all_facets_p95_um"]
    sph_v = eval_res["cup_depth_sphere_p95_um"]
    lines = [
        f"Expected depth:       {exp_txt}",
        f"Recorded (facing):    {rec_v:.3f} um",
        f"All-facets hull:      {all_v:.3f} um",
        f"Diff (rec - all):     {rec_v - all_v:+.3f} um",
        f"Sphere depth p95:     {sph_v:.3f} um",
        f"Patch equiv radius:   {eval_res['patch_equiv_radius_um']:.2f} um",
        f"Contact area:         {eval_res['contact_area_um2']:.1f} um^2",
        f"Hull facets (facing): {eval_res['n_facets_facing']} / {eval_res['n_facets_total']}",
        f"Voxel size:           {VOXEL_UM:.4f} um",
    ]
    ax_txt.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=8.6,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.42,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.5"},
    )
    legend_lines = [
        "Legend:",
        "  Blue solid: Target mask",
        "  Orange solid: T-cell mask",
        "  Purple squares: Contact patch",
        "  Green dashed: 3D convex hull",
        "  Red arrow: Recorded depth",
        "  Green arrow: All-facets depth",
    ]
    ax_txt.text(
        0.02,
        0.34,
        "\n".join(legend_lines),
        transform=ax_txt.transAxes,
        fontsize=8.0,
        va="top",
        ha="left",
        color=COL_MUTED,
        linespacing=1.32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_synthetic_suite(render_panels: bool = True) -> pd.DataFrame:
    """Build and evaluate all synthetic cases (a)-(f), optionally rendering slice PNGs."""
    out_dir = FIGURES_DIR / "review" / "cup_validation"
    rows: list[dict[str, Any]] = []

    # (a) Flat contact: R = 4, 5, 6 um; patch radius a = 2, 4, 6 um (for a <= R)
    for R in (4.0, 5.0, 6.0):
        for a in (2.0, 4.0, 6.0):
            if a > R:
                continue
            b_m, t_m = build_flat_contact_masks(R, a)
            res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
            cid = f"a_flat_R{int(R)}_a{int(a)}"
            title = f"(a) Flat contact: R={R:g} um, a={a:g} um"
            if render_panels:
                render_synthetic_case_panel(cid, title, b_m, t_m, res, 0.0, out_dir / f"{cid}.png")
                if R == 5.0 and a == 4.0:
                    render_synthetic_case_panel("case_a_flat", title, b_m, t_m, res, 0.0, out_dir / "case_a_flat.png")
            rows.append({
                "group": "a_flat",
                "case_id": cid,
                "R_um": R,
                "a_um": a,
                "d_um": 0.0,
                "tilt_deg": 0.0,
                "expected_um": 0.0,
                "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
                "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
                "sphere_p95_um": res["cup_depth_sphere_p95_um"],
                "recorded_hull_max_um": res["cup_depth_hull_max_um"],
                "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
                "patch_equiv_radius_um": res["patch_equiv_radius_um"],
            })

    # (b) Spherical dent: R = 4, 5, 6 um; a = 3, 5 um (a <= R); d = 0.5, 1, 2, 3 um
    for R in (4.0, 5.0, 6.0):
        for a in (3.0, 5.0):
            if a > R:
                continue
            for d in (0.5, 1.0, 2.0, 3.0):
                b_m, t_m = build_spherical_dent_masks(R, a, d)
                res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
                d_str = f"{d:g}".replace(".", "p")
                cid = f"b_dent_R{int(R)}_a{int(a)}_d{d_str}"
                title = f"(b) Spherical dent: R={R:g} um, a={a:g} um, d={d:g} um"
                if render_panels and (R == 5.0 and a == 3.0 or (R == 6.0 and a == 5.0 and d in (1.0, 2.0))):
                    render_synthetic_case_panel(cid, title, b_m, t_m, res, d, out_dir / f"{cid}.png")
                    if R == 5.0 and a == 3.0 and d == 2.0:
                        render_synthetic_case_panel("case_b_dent", title, b_m, t_m, res, d, out_dir / "case_b_dent.png")
                rows.append({
                    "group": "b_dent",
                    "case_id": cid,
                    "R_um": R,
                    "a_um": a,
                    "d_um": d,
                    "tilt_deg": 0.0,
                    "expected_um": d,
                    "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
                    "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
                    "sphere_p95_um": res["cup_depth_sphere_p95_um"],
                    "recorded_hull_max_um": res["cup_depth_hull_max_um"],
                    "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
                    "patch_equiv_radius_um": res["patch_equiv_radius_um"],
                })

    # (c) Same as (b) with n_syn tilted by 10, 20, 30 degrees from the true axis
    for R, a in [(5.0, 3.0), (6.0, 5.0)]:
        for d in (0.5, 1.0, 2.0, 3.0):
            b_m, t_m = build_spherical_dent_masks(R, a, d)
            for tilt in (10.0, 20.0, 30.0):
                res = evaluate_masks_pipeline_and_all_facets(b_m, t_m, tilt_deg=tilt)
                d_str = f"{d:g}".replace(".", "p")
                cid = f"c_tilt_R{int(R)}_a{int(a)}_d{d_str}_deg{int(tilt)}"
                title = f"(c) Tilted n_syn ({int(tilt)} deg): R={R:g}, a={a:g}, d={d:g} um"
                if render_panels and R == 5.0 and a == 3.0 and d == 2.0:
                    render_synthetic_case_panel(cid, title, b_m, t_m, res, d, out_dir / f"{cid}.png")
                    if tilt == 30.0:
                        render_synthetic_case_panel("case_c_tilted", title, b_m, t_m, res, d, out_dir / "case_c_tilted.png")
                rows.append({
                    "group": "c_tilted",
                    "case_id": cid,
                    "R_um": R,
                    "a_um": a,
                    "d_um": d,
                    "tilt_deg": tilt,
                    "expected_um": d,
                    "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
                    "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
                    "sphere_p95_um": res["cup_depth_sphere_p95_um"],
                    "recorded_hull_max_um": res["cup_depth_hull_max_um"],
                    "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
                    "patch_equiv_radius_um": res["patch_equiv_radius_um"],
                })

    # (d) Wide gently curved contact (sag 0.3 um over a = 6 um); expected ~0.3 um
    for R in (4.0, 5.0, 6.0):
        b_m, t_m = build_wide_bowl_masks(R_um=R, a_um=6.0, sag_um=0.3)
        res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
        cid = f"d_wide_bowl_R{int(R)}_a6_sag0p3"
        title = f"(d) Wide bowl: R={R:g} um, a=6 um, sag=0.3 um"
        if render_panels:
            render_synthetic_case_panel(cid, title, b_m, t_m, res, 0.3, out_dir / f"{cid}.png")
            if R == 6.0:
                render_synthetic_case_panel("case_d_wide_bowl", title, b_m, t_m, res, 0.3, out_dir / "case_d_wide_bowl.png")
        rows.append({
            "group": "d_wide_bowl",
            "case_id": cid,
            "R_um": R,
            "a_um": 6.0,
            "d_um": 0.3,
            "tilt_deg": 0.0,
            "expected_um": 0.3,
            "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
            "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
            "sphere_p95_um": res["cup_depth_sphere_p95_um"],
            "recorded_hull_max_um": res["cup_depth_hull_max_um"],
            "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
            "patch_equiv_radius_um": res["patch_equiv_radius_um"],
        })

    # (e) Two separate spheres (merged-target case) with T cell between them; expected 0.0 (un-dented spheres)
    for R in (4.0, 5.0, 6.0):
        b_m, t_m = build_two_spheres_masks(R_um=R)
        res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
        cid = f"e_two_spheres_R{int(R)}"
        title = f"(e) Two merged spheres: R={R:g} um, T cell between"
        if render_panels:
            render_synthetic_case_panel(cid, title, b_m, t_m, res, 0.0, out_dir / f"{cid}.png")
            if R == 5.0:
                render_synthetic_case_panel("case_e_two_spheres", title, b_m, t_m, res, 0.0, out_dir / "case_e_two_spheres.png")
        rows.append({
            "group": "e_two_spheres",
            "case_id": cid,
            "R_um": R,
            "a_um": np.nan,
            "d_um": 0.0,
            "tilt_deg": 0.0,
            "expected_um": 0.0,
            "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
            "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
            "sphere_p95_um": res["cup_depth_sphere_p95_um"],
            "recorded_hull_max_um": res["cup_depth_hull_max_um"],
            "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
            "patch_equiv_radius_um": res["patch_equiv_radius_um"],
        })

    # (f) Hollow shell target (thickness 0.8 um) with a dent of 1 um (both raw shell and after fill_solid_cell_mask)
    for R in (4.0, 5.0, 6.0):
        for apply_fill, suffix in [(False, "raw_shell"), (True, "filled_mask")]:
            b_m, t_m = build_hollow_shell_masks(
                R_um=R, a_um=3.0, d_um=1.0, thickness_um=0.8, apply_fill_solid=apply_fill
            )
            res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
            cid = f"f_hollow_R{int(R)}_d1p0_{suffix}"
            title = f"(f) Hollow shell (0.8 um, {suffix}): R={R:g}, a=3, d=1 um"
            if render_panels and R == 5.0:
                render_synthetic_case_panel(cid, title, b_m, t_m, res, 1.0, out_dir / f"{cid}.png")
                if not apply_fill:
                    render_synthetic_case_panel("case_f_hollow_shell", title, b_m, t_m, res, 1.0, out_dir / "case_f_hollow_shell.png")
            rows.append({
                "group": f"f_hollow_{suffix}",
                "case_id": cid,
                "R_um": R,
                "a_um": 3.0,
                "d_um": 1.0,
                "tilt_deg": 0.0,
                "expected_um": 1.0,
                "recorded_hull_p95_um": res["cup_depth_hull_p95_um"],
                "all_facets_hull_p95_um": res["cup_depth_all_facets_p95_um"],
                "sphere_p95_um": res["cup_depth_sphere_p95_um"],
                "recorded_hull_max_um": res["cup_depth_hull_max_um"],
                "all_facets_hull_max_um": res["cup_depth_all_facets_max_um"],
                "patch_equiv_radius_um": res["patch_equiv_radius_um"],
            })

    return pd.DataFrame(rows)


def remeasure_real_review_panels() -> pd.DataFrame:
    """
    Re-measure the 6 `qc_clean` panels and 16 `cup_measurement` panels in `figures/review/index.csv`
    at the exact same `(experiment, roi, visit, sweep)` with both Recorded (`facing > 0.05`) and
    ALL-facets convex-hull distance (plus the 5-sweep visit average for comparison with `index.csv`).
    """
    df_idx = pd.read_csv(FIGURES_DIR / "review" / "index.csv")
    sub = df_idx[df_idx["case"].isin(["qc_clean", "cup_measurement"])].copy()
    man_df = pd.read_csv(MANIFEST_CSV)
    man_lookup = {
        (str(r.experiment), str(r.roi), int(r.visit_index), int(r.sweep_index)): str(r.path)
        for r in man_df.itertuples(index=False)
    }

    def _eval_real_sweep(exp: str, roi: str, visit: int, sweep: int, row_sw: pd.Series) -> dict[str, float]:
        rel_path = man_lookup[(exp, roi, visit, sweep)]
        with h5py.File(REPO_ROOT / rel_path, "r") as hf:
            I = hf["predictions"][:]
        tracked_id = int(row_sw["tracked_tcell_id"])
        n_t_hint = int(row_sw["n_tcell_instances"])
        n_b_hint = int(row_sw["n_target_instances"])
        s_largest = int(row_sw["s_largest"])
        s_total = int(row_sw["s_total"])
        n_comp = int(row_sw["n_components"])
        if s_largest <= 0 or tracked_id <= 0:
            return {"rec": 0.0, "all": 0.0, "sph": 0.0}

        sl_t = bbox_slices((I == 1) | (I == 3), 2, I.shape)
        t_sel = fast_instance_seg_with_hint(I[sl_t] == 1, I[sl_t] == 3, expected_n_inst=n_t_hint) == tracked_id
        t_sel = fill_solid_cell_mask(t_sel)
        off_t = np.array([sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64)
        t_all_vox = np.argwhere(t_sel).astype(np.float64) + off_t
        t_surf_vox = np.argwhere(t_sel & (~binary_erosion(t_sel, structure=STRUCT26))).astype(np.float64) + off_t
        tree_t = cKDTree(t_surf_vox)

        sl_b = bbox_slices((I == 2) | (I == 4), 2, I.shape)
        b_inst = fast_instance_seg_with_hint(I[sl_b] == 2, I[sl_b] == 4, expected_n_inst=n_b_hint)
        off_b = np.array([sl_b[0].start, sl_b[1].start, sl_b[2].start], dtype=np.float64)
        cnts_b = np.bincount(b_inst.ravel())
        cnts_b[0] = 0
        max_b = int(b_inst.max())
        if max_b == 1:
            best_bid = 1
            b_cell_sub = fill_solid_cell_mask(b_inst == 1)
        else:
            best_bid = int(np.argmax(cnts_b))
            best_touch = -1
            for bid in range(1, max_b + 1):
                if cnts_b[bid] == 0:
                    continue
                bi = b_inst == bid
                bi_s = bi & (~binary_erosion(bi, structure=STRUCT26))
                d_bi, _ = tree_t.query(np.argwhere(bi_s).astype(np.float64) + off_b, k=1, distance_upper_bound=3.5)
                n_touch = int(np.sum(np.isfinite(d_bi)))
                if n_touch > best_touch or (n_touch == best_touch and cnts_b[bid] > cnts_b[best_bid]):
                    best_touch = n_touch
                    best_bid = bid
            cc_lbl, _ = label(b_inst > 0, structure=STRUCT26)
            dom_cc = int(np.median(cc_lbl[b_inst == best_bid]))
            b_cell_sub = fill_solid_cell_mask(cc_lbl == dom_cc)

        b_all_vox = np.argwhere(b_cell_sub).astype(np.float64) + off_b
        b_surf_vox = np.argwhere(b_cell_sub & (~binary_erosion(b_cell_sub, structure=STRUCT26))).astype(np.float64) + off_b
        d_bt, _ = tree_t.query(b_surf_vox, k=1)
        within_25 = d_bt <= 2.6
        if int(np.sum(within_25)) >= 12:
            is_c = within_25
        else:
            k_sel = min(len(d_bt), max(12, s_largest))
            thr_d = min(4.5, float(np.partition(d_bt, k_sel - 1)[k_sel - 1]))
            is_c = d_bt <= thr_d
        is_nc = d_bt > 3.5

        desc = compute_descriptors_for_z_pitch(
            VOXEL_UM,
            s_largest,
            s_total,
            n_comp,
            t_sel,
            t_surf_vox,
            t_all_vox,
            b_cell_sub,
            int(cnts_b[best_bid]),
            b_surf_vox,
            b_all_vox,
            is_c,
            is_nc,
        )
        pts_b_um = b_surf_vox * VOXEL_UM
        pts_c_um = pts_b_um[is_c]
        hull = ConvexHull(pts_b_um)
        eqs = hull.equations
        d_all = np.maximum(0.0, np.min(-((pts_c_um @ eqs[:, :3].T) + eqs[:, 3]), axis=1))
        return {
            "rec": float(desc["cup_depth_hull_p95_um"]),
            "all": float(np.percentile(d_all, 95.0)),
            "sph": float(desc["cup_depth_sphere_p95_um"]),
        }

    rows = []
    for _, r in sub.iterrows():
        exp = str(r["experiment"])
        roi = str(r["roi"])
        visit = int(r["visit"])
        sweep = int(r["sweep"])
        sw_df = pd.read_csv(DATA_DIR / "step3_tracked" / exp / "per_sweep.csv")
        vis_rows = sw_df[(sw_df["roi"] == roi) & (sw_df["visit"] == visit) & (sw_df["sweep"].isin([1, 2, 3, 4, 5]))]

        rec_sw, all_sw, sph_sw = 0.0, 0.0, 0.0
        recs_vis, alls_vis = [], []
        for _, rsw in vis_rows.iterrows():
            sw_i = int(rsw["sweep"])
            ev = _eval_real_sweep(exp, roi, visit, sw_i, rsw)
            if int(rsw["s_largest"]) > 0:
                recs_vis.append(ev["rec"])
                alls_vis.append(ev["all"])
            if sw_i == sweep:
                rec_sw, all_sw, sph_sw = ev["rec"], ev["all"], ev["sph"]

        rows.append({
            "case": str(r["case"]),
            "id": str(r["id"]),
            "experiment": exp,
            "roi": roi,
            "visit": visit,
            "sweep": sweep,
            "index_csv_visit_mean_um": float(r["cup_depth_hull_p95_um"]),
            "visit_mean_all_facets_um": float(np.mean(alls_vis)) if alls_vis else 0.0,
            "sweep_recorded_um": rec_sw,
            "sweep_all_facets_um": all_sw,
            "sweep_diff_rec_minus_all_um": rec_sw - all_sw,
            "sweep_sphere_p95_um": sph_sw,
        })
    return pd.DataFrame(rows)


def render_summary_figure(df_syn: pd.DataFrame, df_real: pd.DataFrame, out_path: Path) -> None:
    """
    Render `figures/cup_depth_validation.png`:
      - Panel A: Synthetic cases (a, b, c, d) — Recorded (`facing > 0.05`) vs Expected and
        All-facets vs Expected (with `y = x` identity line and Sphere fit comparison).
      - Panel B: Synthetic error (`measured - expected`) across groups (a) flat, (b) spherical dent,
        (c) tilted `n_syn`, (d) wide bowl, plus artifact checks (e) two merged spheres and (f) hollow shell.
      - Panel C: Real review panels (`6 qc_clean` + `16 cup_measurement`) — Recorded vs All-facets
        side by side on the displayed sweep (and highlighting `qc_clean_02` visit-mean vs sweep-2).
    """
    setup_house_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=120)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.14, wspace=0.25)

    fig.suptitle(
        "Cup-Depth Validation: Synthetic Ground Truth & Real Panel Re-measurement",
        fontsize=11.5,
        fontweight="bold",
        color=COL_INK,
        x=0.05,
        ha="left",
        y=0.96,
    )

    # Panel A: Recorded vs Expected and All-facets vs Expected on ground-truth shapes (a, b, c, d)
    ax0 = axes[0]
    sub_gt = df_syn[df_syn["group"].isin(["a_flat", "b_dent", "c_tilted", "d_wide_bowl"])].copy()
    ax0.plot([0, 3.2], [0, 3.2], color=COL_INK, lw=1.0, ls="--", label="Exact (y = x)", zorder=2)
    ax0.scatter(
        sub_gt["expected_um"] - 0.03,
        sub_gt["recorded_hull_p95_um"],
        s=28,
        c=COL_PLOT1,
        marker="o",
        alpha=0.85,
        label="Recorded (facing > 0.05)",
        zorder=4,
    )
    ax0.scatter(
        sub_gt["expected_um"] + 0.03,
        sub_gt["all_facets_hull_p95_um"],
        s=28,
        c=COL_PLOT3,
        marker="^",
        alpha=0.85,
        label="All facets (min over hull)",
        zorder=5,
    )
    ax0.scatter(
        sub_gt["expected_um"],
        sub_gt["sphere_p95_um"],
        s=20,
        c=COL_PLOT2,
        marker="x",
        alpha=0.65,
        label="Sphere ref (biased by cap sag)",
        zorder=3,
    )
    ax0.set_xlim(-0.18, 3.3)
    ax0.set_ylim(-0.18, 4.4)
    ax0.set_xlabel("Expected true depth from hull (um)", fontsize=9.0, fontweight="bold")
    ax0.set_ylabel("Measured cup depth p95 (um)", fontsize=9.0, fontweight="bold")
    ax0.set_title("A. Synthetic Ground Truth (Cases a-d)", fontsize=9.5, fontweight="bold", loc="left")
    ax0.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax0.grid(True, color=COL_RULE, lw=0.6)

    # Panel B: Summary by synthetic case category (a-f)
    ax1 = axes[1]
    cat_order = [
        ("a_flat", "a. Flat\n(exp 0)"),
        ("b_dent", "b. Dent\n(exp d)"),
        ("c_tilted", "c. Tilt\n10-30°"),
        ("d_wide_bowl", "d. Bowl\n(0.3 um)"),
        ("e_two_spheres", "e. Merged\n2 spheres"),
        ("f_hollow_raw_shell", "f1. Hollow\n(raw)"),
        ("f_hollow_filled_mask", "f2. Hollow\n(fill_solid)"),
    ]
    x_pos = np.arange(len(cat_order))
    rec_err_means = []
    all_err_means = []
    for g_key, _ in cat_order:
        sg = df_syn[df_syn["group"] == g_key]
        rec_err_means.append(float(np.mean(sg["recorded_hull_p95_um"] - sg["expected_um"])))
        all_err_means.append(float(np.mean(sg["all_facets_hull_p95_um"] - sg["expected_um"])))

    w_b = 0.34
    ax1.axhline(0.0, color=COL_INK, lw=0.9, ls="--", zorder=2)
    ax1.bar(x_pos - 0.5 * w_b, rec_err_means, width=w_b, color=COL_PLOT1, label="Recorded - Expected", zorder=4)
    ax1.bar(x_pos + 0.5 * w_b, all_err_means, width=w_b, color=COL_PLOT3, label="All-facets - Expected", zorder=4)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([lbl for _, lbl in cat_order], fontsize=7.6)
    ax1.set_ylabel("Mean signed error: Measured - Expected (um)", fontsize=9.0, fontweight="bold")
    ax1.set_title("B. Bias by Synthetic Geometry (Cases a-f)", fontsize=9.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax1.grid(True, axis="y", color=COL_RULE, lw=0.6)

    # Panel C: Real review panels (6 qc_clean + 16 cup_measurement) Recorded vs All-facets
    ax2 = axes[2]
    lim_hi = max(11.0, float(df_real["sweep_recorded_um"].max()) + 0.5)
    ax2.plot([0, lim_hi], [0, lim_hi], color=COL_INK, lw=1.0, ls="--", label="y = x (0 diff)", zorder=2)
    sub_cup = df_real[df_real["case"] == "cup_measurement"]
    sub_qc = df_real[df_real["case"] == "qc_clean"]
    ax2.scatter(
        sub_cup["sweep_all_facets_um"],
        sub_cup["sweep_recorded_um"],
        s=34,
        c=COL_PLOT1,
        marker="o",
        label="cup_measurement (n=16)",
        zorder=4,
    )
    ax2.scatter(
        sub_qc["sweep_all_facets_um"],
        sub_qc["sweep_recorded_um"],
        s=42,
        c=COL_PLOT4,
        marker="s",
        label="qc_clean (n=6)",
        zorder=5,
    )
    # Annotate cup_01 and qc_clean_02
    r_c01 = df_real[df_real["id"] == "cup_01"].iloc[0]
    ax2.annotate(
        f"cup_01 (+{r_c01['sweep_diff_rec_minus_all_um']:.2f} um)",
        xy=(r_c01["sweep_all_facets_um"], r_c01["sweep_recorded_um"]),
        xytext=(r_c01["sweep_all_facets_um"] - 4.3, r_c01["sweep_recorded_um"] - 0.8),
        fontsize=7.6,
        color=COL_INK,
        arrowprops={"arrowstyle": "->", "color": COL_MUTED, "lw": 0.8},
    )
    r_qc02 = df_real[df_real["id"] == "qc_clean_02"].iloc[0]
    ax2.annotate(
        f"qc_clean_02\nsweep 2: {r_qc02['sweep_recorded_um']:.2f} vs {r_qc02['sweep_all_facets_um']:.2f} um\n(visit mean: {r_qc02['index_csv_visit_mean_um']:.2f} um)",
        xy=(r_qc02["sweep_all_facets_um"], r_qc02["sweep_recorded_um"]),
        xytext=(r_qc02["sweep_all_facets_um"] + 1.2, r_qc02["sweep_recorded_um"] + 1.8),
        fontsize=7.3,
        color=COL_INK,
        arrowprops={"arrowstyle": "->", "color": COL_MUTED, "lw": 0.8},
    )
    ax2.set_xlim(-0.3, lim_hi)
    ax2.set_ylim(-0.3, lim_hi)
    ax2.set_xlabel("All-facets hull depth p95 (um)", fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("Recorded (facing > 0.05) hull depth p95 (um)", fontsize=9.0, fontweight="bold")
    ax2.set_title("C. Real Review Panels (n=22, same sweep)", fontsize=9.5, fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax2.grid(True, color=COL_RULE, lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    df_syn = run_synthetic_suite(render_panels=True)
    syn_csv = DATA_DIR / "step5" / "cup_depth_synthetic_validation.csv"
    syn_csv.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(syn_csv, index=False)

    df_real = remeasure_real_review_panels()
    real_csv = DATA_DIR / "step5" / "cup_depth_real_panels_comparison.csv"
    df_real.to_csv(real_csv, index=False)

    render_summary_figure(df_syn, df_real, FIGURES_DIR / "cup_depth_validation.png")
    print(f"Saved {len(df_syn)} synthetic cases to {syn_csv}")
    print(f"Saved {len(df_real)} real panel comparisons to {real_csv}")
    print(f"Rendered summary figure to {FIGURES_DIR / 'cup_depth_validation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
