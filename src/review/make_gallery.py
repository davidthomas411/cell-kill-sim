#!/usr/bin/env python3
"""
Visual review gallery of REAL segmentation masks, raw fluorescence volumes (where available
locally under data/scidb_19235_raw/ and data/scidb_19235_chambers/), 2D MIPs, contact/cup
measurements, two-shell mechanics fits, initial overlap resolution, mesh size sensitivity,
death-dye calls, dasatinib vs untreated comparisons, and deep vs shallow cups.

Outputs:
  - figures/review/<case>/<id>.png (128 panels total across 8 cases, each <= 400 KB)
  - figures/review/index.csv
  - figures/review/README.md
  - .agy_gallery_report.md
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _env_k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_k, "1")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion, label
from scipy.spatial import ConvexHull, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.paths import DATA_DIR, FIGURES_DIR
from src.reproduce_contact_area import CA_VOXEL_FACTOR_UM2, MANIFEST_CSV, STRUCT26, bbox_slices
from src.contact_descriptors import (
    compute_descriptors_for_z_pitch,
    fast_instance_seg_with_hint,
    fill_solid_cell_mask,
    fit_regularized_3d_sphere,
)
from src.death_labels import (
    MIN_COMPONENT_AREA_2D,
    MIN_OVERLAP_PX_2D,
    THRESH_2D_ADU,
    compute_2d_dead_mask,
    dilate_2d_edt,
)
from src.mech.run_step6b import (
    MESH_EDGE_UM,
    STEP3_TRACKED_DIR,
    STEP4_CHAMBER_CSV,
    STEP5_VISIT_CSV,
    _align_reference_target_to_visit,
    _extract_tcell_and_target_masks_from_h5,
    _tag_synapse_faces_for_patch_area,
)
from src.mech.run_gateB_check import (
    PI_NODES_GATEB,
    _invert_depth_curve_no_extrap,
    _prepare_visit_meshes_at_edge,
)
from src.mech.shell_model import (
    ShellMesh,
    ShellParameters,
    TwoShellSystem,
    compute_mesh_descriptors,
    extract_cell_mesh_marching_cubes,
    make_cup_filled_reference_mesh,
    resolve_initial_contact_overlap,
    solve_two_shell_equilibrium,
)

SEED = 20261028
MAX_WORKERS = 3
REVIEW_DIR = FIGURES_DIR / "review"
RAW_DIR = DATA_DIR / "scidb_19235_raw"
CHAMBERS_DIR = DATA_DIR / "scidb_19235_chambers"
MASKS_DIR = DATA_DIR / "scidb_19235_masks"
MIPS_DIR = DATA_DIR / "scidb_19235_mips"
GATEB_FITS_CSV = DATA_DIR / "step6b" / "gateB_visit_fits.csv"
STEP4_VISIT_CSV = DATA_DIR / "step4" / "per_visit_death_labels_all_v2.csv"

# House-style palette tokens
COL_PAPER = "#FFFFFF"
COL_PAPER2 = "#F6F6F6"
COL_INK = "#111111"
COL_TEXT = "#1F1F1F"
COL_MUTED = "#5C5C5C"
COL_RULE = "#E3E3E3"
COL_LINE = "#C9C9C9"
COL_DANGER = "#B3261E"
COL_PLOT1 = "#1F4E79"  # Target / primary data (deep blue)
COL_PLOT2 = "#C0561F"  # T cell / secondary data (terracotta orange)
COL_PLOT3 = "#2E7D5B"  # Reference / hull / 0.5 um mesh (forest green)
COL_PLOT4 = "#6B4C9A"  # Contact patch / solved model highlight (purple)


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


def find_local_raw_h5(exp: str, roi: str, stem: str) -> Path | None:
    """Return path to local raw 3D .h5 volume if downloaded under scidb_19235_raw or scidb_19235_chambers."""
    p1 = RAW_DIR / exp / roi / f"{stem}.h5"
    if p1.is_file():
        return p1
    p2 = CHAMBERS_DIR / f"{exp}_{roi}" / f"{stem}.h5"
    if p2.is_file():
        return p2
    return None


def intersect_mesh_with_plane(V_um: np.ndarray, F: np.ndarray, axis: int, val_um: float) -> np.ndarray:
    """
    Exact intersection of a 3D triangulated mesh `(V_um, F)` with coordinate plane `V_um[:, axis] == val_um`.
    Returns `(M, 2, 3)` array of 3D line segment endpoints in micrometers.
    """
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
        t = np.where(np.abs(denom) > 1e-12, da / denom, 0.5)
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


def plot_mesh_slice_segments(
    ax: plt.Axes,
    segs_3d_um: np.ndarray,
    plane: str,
    x0_vox: int,
    y0_vox: int,
    z0_vox: int,
    color: str,
    lw: float = 1.6,
    ls: str = "-",
    alpha: float = 0.95,
    zorder: int = 5,
) -> None:
    """
    Plot 3D mesh-plane intersection segments onto a 2D slice axis in cropped voxel coordinates.
    For `plane == 'xy'`: horizontal axis is `x - x0_vox`, vertical axis is `y - y0_vox`.
    For `plane == 'xz'`: horizontal axis is `x - x0_vox`, vertical axis is `z - z0_vox`.
    """
    if segs_3d_um is None or len(segs_3d_um) == 0:
        return
    segs_vox = segs_3d_um / VOXEL_UM  # (M, 2, 3) in (z, y, x) voxel units
    if plane == "xy":
        xs = segs_vox[:, :, 2] - x0_vox
        ys = segs_vox[:, :, 1] - y0_vox
    else:
        xs = segs_vox[:, :, 2] - x0_vox
        ys = segs_vox[:, :, 0] - z0_vox
    segs_2d = np.stack([xs, ys], axis=-1)
    lc = LineCollection(segs_2d, colors=color, linewidths=lw, linestyles=ls, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)


def load_slice_Geometry_and_raw(
    exp: str,
    roi: str,
    visit: int,
    sweep_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Load the real 3D segmentation mask `_predictions.h5` (and raw 3D `.h5` if available locally),
    segment the tracked T cell and contacted target cell using `src/contact_descriptors.py`,
    find the contact patch center `(zc, yc, xc)`, and extract full-resolution XY and XZ slices
    through the contact patch center.
    """
    rel_path = str(sweep_info["rel_path"])
    abs_mask_path = REPO_ROOT / rel_path
    stem = abs_mask_path.name.replace("_predictions.h5", "")
    tracked_id = int(sweep_info["tracked_tcell_id"])
    n_t_hint = int(sweep_info["n_tcell_instances"])
    n_b_hint = int(sweep_info["n_target_instances"])
    s_largest = int(sweep_info["s_largest"])

    with h5py.File(abs_mask_path, "r") as hf:
        I = hf["predictions"][:]  # (Z, Y, X)

    Z_dim, Y_dim, X_dim = I.shape

    # Extract solid masks for tracked T cell and contacted target cell
    t_sel, off_t, b_sel, off_b = _extract_tcell_and_target_masks_from_h5(
        abs_mask_path, tracked_id, n_t_hint, n_b_hint
    )

    t_full = np.zeros_like(I, dtype=bool)
    b_full = np.zeros_like(I, dtype=bool)
    t_surf_vox = np.empty((0, 3), dtype=np.float64)
    t_all_vox = np.empty((0, 3), dtype=np.float64)
    b_surf_vox = np.empty((0, 3), dtype=np.float64)
    b_all_vox = np.empty((0, 3), dtype=np.float64)
    is_contact_on_b_surf = None
    is_noncontact_on_b_surf = None

    if t_sel is not None and off_t is not None and np.any(t_sel):
        zt0, yt0, xt0 = int(off_t[0]), int(off_t[1]), int(off_t[2])
        t_full[zt0 : zt0 + t_sel.shape[0], yt0 : yt0 + t_sel.shape[1], xt0 : xt0 + t_sel.shape[2]] = t_sel
        t_all_vox = np.argwhere(t_sel).astype(np.float64) + off_t
        t_surf_sub = t_sel & (~binary_erosion(t_sel, structure=STRUCT26))
        t_surf_vox = np.argwhere(t_surf_sub).astype(np.float64) + off_t

    if b_sel is not None and off_b is not None and np.any(b_sel):
        zb0, yb0, xb0 = int(off_b[0]), int(off_b[1]), int(off_b[2])
        b_full[zb0 : zb0 + b_sel.shape[0], yb0 : yb0 + b_sel.shape[1], xb0 : xb0 + b_sel.shape[2]] = b_sel
        b_all_vox = np.argwhere(b_sel).astype(np.float64) + off_b
        b_surf_sub = b_sel & (~binary_erosion(b_sel, structure=STRUCT26))
        b_surf_vox = np.argwhere(b_surf_sub).astype(np.float64) + off_b

    if len(t_surf_vox) > 0 and len(b_surf_vox) > 0:
        tree_t = cKDTree(t_surf_vox)
        d_bt, _ = tree_t.query(b_surf_vox, k=1)
        within_25 = d_bt <= 2.6
        if int(np.sum(within_25)) >= 12:
            is_contact_on_b_surf = within_25
        else:
            k_sel = min(len(d_bt), max(12, max(1, s_largest)))
            thr_d = min(4.5, float(np.partition(d_bt, k_sel - 1)[k_sel - 1]))
            is_contact_on_b_surf = d_bt <= thr_d
        is_noncontact_on_b_surf = d_bt > 3.5

    # Contact patch 3D mask and center
    contact_full = np.zeros_like(I, dtype=bool)
    if is_contact_on_b_surf is not None and np.any(is_contact_on_b_surf):
        c_vox = b_surf_vox[is_contact_on_b_surf].astype(int)
        contact_full[c_vox[:, 0], c_vox[:, 1], c_vox[:, 2]] = True
        p_center_vox = b_surf_vox[is_contact_on_b_surf].mean(axis=0)
    elif len(b_all_vox) > 0 and len(t_all_vox) > 0:
        p_center_vox = 0.5 * (b_all_vox.mean(axis=0) + t_all_vox.mean(axis=0))
    elif len(b_all_vox) > 0:
        p_center_vox = b_all_vox.mean(axis=0)
    else:
        p_center_vox = np.array([Z_dim / 2.0, Y_dim / 2.0, X_dim / 2.0])

    zc = int(np.clip(np.round(p_center_vox[0]), 0, Z_dim - 1))
    yc = int(np.clip(np.round(p_center_vox[1]), 0, Y_dim - 1))
    xc = int(np.clip(np.round(p_center_vox[2]), 0, X_dim - 1))

    # Compute 3D convex hull of target surface points in um
    scale_zyx = np.array([VOXEL_UM, VOXEL_UM, VOXEL_UM], dtype=np.float64)
    pts_b_surf_um = b_surf_vox * scale_zyx if len(b_surf_vox) >= 12 else None
    hull_b = None
    d_hull_c = None
    deepest_pt_vox = None
    hull_proj_pt_vox = None
    n_syn = np.array([0.0, 0.0, 1.0])

    if pts_b_surf_um is not None:
        try:
            hull_b = ConvexHull(pts_b_surf_um)
        except Exception:
            hull_b = None

    if hull_b is not None and is_contact_on_b_surf is not None and np.any(is_contact_on_b_surf):
        pts_c_um = pts_b_surf_um[is_contact_on_b_surf]
        b_center_um = b_all_vox.mean(axis=0) * scale_zyx
        t_center_um = t_all_vox.mean(axis=0) * scale_zyx if len(t_all_vox) > 0 else pts_c_um.mean(axis=0)
        n_syn = pts_c_um.mean(axis=0) - b_center_um
        if np.dot(t_center_um - b_center_um, n_syn) < 0:
            n_syn = t_center_um - b_center_um
        n_norm = float(np.linalg.norm(n_syn))
        n_syn = (n_syn / n_norm) if n_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        eqs = hull_b.equations
        facing = (eqs[:, :3] @ n_syn) > 0.05
        if not np.any(facing):
            facing = np.ones(len(eqs), dtype=bool)
        eq_f = eqs[facing]
        dists_hull = -((pts_c_um @ eq_f[:, :3].T) + eq_f[:, 3])
        min_facet_idx = np.argmin(dists_hull, axis=1)
        d_hull_c = np.maximum(0.0, np.min(dists_hull, axis=1))

        # Find contact point at p95 depth to mark the measured cup depth vector
        p95_val = float(np.percentile(d_hull_c, 95.0))
        idx_p95 = int(np.argmin(np.abs(d_hull_c - p95_val)))
        pt_c_p95_um = pts_c_um[idx_p95]
        facet_normal = eq_f[min_facet_idx[idx_p95], :3]
        pt_hull_p95_um = pt_c_p95_um + p95_val * facet_normal
        deepest_pt_vox = pt_c_p95_um / VOXEL_UM
        hull_proj_pt_vox = pt_hull_p95_um / VOXEL_UM

    # Bounding box for display crop around T cell + Target cell
    union_vox = np.vstack([b_all_vox, t_all_vox]) if (len(b_all_vox) > 0 and len(t_all_vox) > 0) else (
        b_all_vox if len(b_all_vox) > 0 else np.array([[zc, yc, xc]], dtype=np.float64)
    )
    pad_vox = 18
    z_min, y_min, x_min = np.maximum(0, np.floor(union_vox.min(axis=0)).astype(int) - pad_vox)
    z_max, y_max, x_max = np.minimum(
        [Z_dim, Y_dim, X_dim], np.ceil(union_vox.max(axis=0)).astype(int) + pad_vox
    )
    # Enforce minimum window size of 96 voxels (~18.8 um)
    for lo_v, hi_v, lim in [(z_min, z_max, Z_dim), (y_min, y_max, Y_dim), (x_min, x_max, X_dim)]:
        pass
    side_xy = max(y_max - y_min, x_max - x_min, 96)
    side_xy = min(side_xy, 256)
    cy_box = (y_min + y_max) // 2
    cx_box = (x_min + x_max) // 2
    y0 = int(np.clip(cy_box - side_xy // 2, 0, Y_dim - side_xy))
    y1 = y0 + side_xy
    x0 = int(np.clip(cx_box - side_xy // 2, 0, X_dim - side_xy))
    x1 = x0 + side_xy

    side_z = max(z_max - z_min, min(Z_dim, side_xy))
    cz_box = (z_min + z_max) // 2
    z0 = int(np.clip(cz_box - side_z // 2, 0, Z_dim - side_z))
    z1 = z0 + side_z

    # Check if raw .h5 exists locally
    raw_h5_path = find_local_raw_h5(exp, roi, stem)
    raw_xy_actin = None
    raw_xy_memb = None
    raw_xz_actin = None
    raw_xz_memb = None
    if raw_h5_path is not None:
        try:
            with h5py.File(raw_h5_path, "r") as hf_r:
                raw_ds = hf_r["raw"]
                # Channel 1 = 488 nm T-cell actin; Channel 2 = 561 nm membrane
                raw_xy_actin = raw_ds[1, zc, y0:y1, x0:x1].astype(np.float32)
                raw_xy_memb = raw_ds[2, zc, y0:y1, x0:x1].astype(np.float32)
                raw_xz_actin = raw_ds[1, z0:z1, yc, x0:x1].astype(np.float32)
                raw_xz_memb = raw_ds[2, z0:z1, yc, x0:x1].astype(np.float32)
        except Exception:
            raw_h5_path = None

    return {
        "stem": stem,
        "abs_mask_path": abs_mask_path,
        "raw_h5_path": raw_h5_path,
        "I_xy": I[zc, y0:y1, x0:x1],
        "I_xz": I[z0:z1, yc, x0:x1],
        "t_xy": t_full[zc, y0:y1, x0:x1],
        "t_xz": t_full[z0:z1, yc, x0:x1],
        "b_xy": b_full[zc, y0:y1, x0:x1],
        "b_xz": b_full[z0:z1, yc, x0:x1],
        "c_xy": contact_full[zc, y0:y1, x0:x1],
        "c_xz": contact_full[z0:z1, yc, x0:x1],
        "c_slab_xy": contact_full[max(0, zc - 1) : min(Z_dim, zc + 2), y0:y1, x0:x1].any(axis=0),
        "c_slab_xz": contact_full[z0:z1, max(0, yc - 1) : min(Y_dim, yc + 2), x0:x1].any(axis=1),
        "zc": zc,
        "yc": yc,
        "xc": xc,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "z0": z0,
        "z1": z1,
        "hull_b": hull_b,
        "pts_b_surf_um": pts_b_surf_um,
        "deepest_pt_vox": deepest_pt_vox,
        "hull_proj_pt_vox": hull_proj_pt_vox,
        "n_syn": n_syn,
        "p_center_vox": p_center_vox,
        "raw_xy_actin": raw_xy_actin,
        "raw_xy_memb": raw_xy_memb,
        "raw_xz_actin": raw_xz_actin,
        "raw_xz_memb": raw_xz_memb,
        "t_sel": t_sel,
        "off_t": off_t,
        "b_sel": b_sel,
        "off_b": off_b,
    }


def make_background_image(
    I_slice: np.ndarray,
    b_slice: np.ndarray,
    t_slice: np.ndarray,
    raw_actin: np.ndarray | None = None,
    raw_memb: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Build clean grayscale background image for a 2D slice (either XY or XZ):
      - If local raw .h5 exists: normalized composite of raw 561-nm membrane + 488-nm actin channels
        on a light background (`Greys` inverted or light-mode clean fluorescence intensity).
      - Otherwise: real 3D U-Net semantic segmentation slice (`I_slice` with interior/boundary
        values `0, 1, 2, 3, 4` and tracked instance fills) rendered in crisp light-mode grayscale.
    Returns `(rgb_img, is_raw)`.
    """
    h, w = I_slice.shape
    if raw_actin is not None and raw_memb is not None:
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo = float(np.percentile(arr, 5.0))
            hi = float(np.percentile(arr, 99.5))
            return np.clip((arr - lo) / max(3.0, hi - lo), 0.0, 1.0)

        na = _norm(raw_actin)
        nm = _norm(raw_memb)
        # Light-mode inverted fluorescence: white background (#FFFFFF), dark signal where membrane/actin are bright
        inten = np.clip(0.60 * nm + 0.55 * na, 0.0, 1.0)
        gray = 1.0 - 0.72 * inten
        rgb = np.stack([gray, gray, gray], axis=-1)
        return rgb, True

    # Real 3D U-Net segmentation mask slice rendered in muted neutral tones on #FFFFFF paper
    rgb = np.ones((h, w, 3), dtype=np.float32)
    # Other cells in chamber (untracked T cells or secondary targets) in very light gray
    other_mask = (I_slice > 0) & (~b_slice) & (~t_slice)
    rgb[other_mask] = [0.92, 0.92, 0.92]
    # Tracked target cell interior fill (soft blue-gray tint)
    rgb[b_slice] = [0.85, 0.90, 0.96]
    # Tracked T-cell interior fill (soft warm-gray tint)
    rgb[t_slice] = [0.96, 0.89, 0.84]
    # Highlight raw U-Net boundary class predictions (I==3 T-cell boundary, I==4 target boundary)
    u_b_bound = (I_slice == 4) & b_slice
    u_t_bound = (I_slice == 3) & t_slice
    rgb[u_b_bound] = [0.74, 0.82, 0.91]
    rgb[u_t_bound] = [0.91, 0.79, 0.71]
    return rgb, False


def add_scale_bar(ax: plt.Axes, h: int, w: int, bar_um: float = 5.0) -> None:
    """Draw a literal scale bar in micrometers at bottom-right of slice axis."""
    bar_px = bar_um / VOXEL_UM
    x_right = w - 8
    x_left = max(4, x_right - bar_px)
    y_bar = h - 8
    ax.plot([x_left, x_right], [y_bar, y_bar], color=COL_INK, linewidth=2.4, zorder=15)
    ax.text(
        0.5 * (x_left + x_right),
        y_bar - 3.5,
        f"{bar_um:g} um",
        color=COL_INK,
        fontsize=8.0,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": COL_PAPER, "alpha": 0.82, "edgecolor": "none", "pad": 1.0},
        zorder=15,
    )


def draw_slice_crosshairs(ax: plt.Axes, x_pos: float, y_pos: float) -> None:
    """Draw subtle dashed crosshairs through the contact patch center."""
    ax.axvline(x_pos, color=COL_LINE, lw=0.7, ls=":", zorder=2)
    ax.axhline(y_pos, color=COL_LINE, lw=0.7, ls=":", zorder=2)


# ==============================================================================
# RENDERERS FOR CASES 1 TO 8
# ==============================================================================

def render_cup_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one panel for Cases 1 (`cup_measurement`), 7 (`dasatinib_vs_untreated`), or 8 (`deep_vs_shallow`).
    Shows:
      - XY slice (`z = zc`) and XZ slice (`y = yc`) through the contact patch center
      - Target mask outline (`#1F4E79`), T-cell mask outline (`#C0561F`), contact patch voxels (`#6B4C9A`)
      - 3D target convex hull cross-section (`#2E7D5B` dashed) and 2D slice convex hull (`#2E7D5B` dotted)
      - Measured cup depth (`cup_depth_hull_p95_um`) marked as a distance arrow from hull to cup
      - Patch diameter (`2 * patch_equiv_radius_um`) marked across the contact patch
      - Key numbers text box & voxel size statement (`dx = dy = dz = 0.1962 um`, `z = 0.196 um`)
    """
    setup_house_style()
    case_name = str(item["case"])
    ex_id = str(item["id"])
    exp = str(item["experiment"])
    roi = str(item["roi"])
    visit = int(item["visit"])
    cond = str(item["condition"])
    sw_info = item["sweep_info"]
    sweep = int(sw_info["sweep"])

    geom = load_slice_Geometry_and_raw(exp, roi, visit, sw_info)
    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc, xc = geom["zc"], geom["yc"], geom["xc"]

    rgb_xy, is_raw_xy = make_background_image(
        geom["I_xy"], geom["b_xy"], geom["t_xy"], geom["raw_xy_actin"], geom["raw_xy_memb"]
    )
    rgb_xz, _ = make_background_image(
        geom["I_xz"], geom["b_xz"], geom["t_xz"], geom["raw_xz_actin"], geom["raw_xz_memb"]
    )

    fig = plt.figure(figsize=(10.2, 4.4), dpi=105)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.92], wspace=0.14, left=0.03, right=0.98, top=0.84, bottom=0.06)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    bg_label = "Raw 561+488 nm + masks" if is_raw_xy else "3D U-Net segmentation mask"
    fig.suptitle(
        f"{case_name} | {ex_id} | {exp} | {roi} | visit {visit} | sweep {sweep}",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.03,
        0.90,
        f"Condition: {cond} | Source: {bg_label} | Voxel pitch: dx = dy = dz = 0.1962 um (z = 0.196 um)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    for ax, rgb, b_sl, t_sl, c_sl, plane, sl_idx, c_coord in [
        (ax_xy, rgb_xy, geom["b_xy"], geom["t_xy"], geom["c_slab_xy"], "xy", zc, (xc - x0, yc - y0)),
        (ax_xz, rgb_xz, geom["b_xz"], geom["t_xz"], geom["c_slab_xz"], "xz", yc, (xc - x0, zc - z0)),
    ]:
        h_s, w_s = rgb.shape[:2]
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

        # Target mask outline
        if np.any(b_sl):
            ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.8, zorder=4)
            # 2D slice convex hull of target in this slice
            pts_2d = np.argwhere(b_sl)  # (row, col) = (y, x)
            if len(pts_2d) >= 6:
                try:
                    hull2d = ConvexHull(pts_2d[:, ::-1])  # (x, y)
                    hv = np.append(hull2d.vertices, hull2d.vertices[0])
                    ax.plot(
                        pts_2d[hv, 1],
                        pts_2d[hv, 0],
                        color=COL_PLOT3,
                        lw=1.3,
                        ls="--",
                        zorder=5,
                    )
                except Exception:
                    pass

        # 3D target convex hull cross-section in this slice plane
        if geom["hull_b"] is not None and geom["pts_b_surf_um"] is not None:
            val_um = (zc * VOXEL_UM) if plane == "xy" else (yc * VOXEL_UM)
            axis_i = 0 if plane == "xy" else 1
            segs_hull = intersect_mesh_with_plane(
                geom["pts_b_surf_um"], geom["hull_b"].simplices, axis=axis_i, val_um=val_um
            )
            plot_mesh_slice_segments(ax, segs_hull, plane, x0, y0, z0, color=COL_PLOT3, lw=1.0, ls=":", zorder=5)

        # T-cell mask outline
        if np.any(t_sl):
            ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_PLOT2], linewidths=1.6, zorder=4)

        # Contact patch voxels
        cy_pts, cx_pts = np.where(c_sl)
        if len(cx_pts) > 0:
            ax.scatter(cx_pts, cy_pts, s=7.0, c=COL_PLOT4, marker="s", alpha=0.85, linewidths=0, zorder=6)

        # Mark measured cup depth (cup_depth_hull_p95_um) as a vector from hull to contact surface
        d_p95 = float(item["cup_depth_hull_p95_um"])
        if geom["deepest_pt_vox"] is not None and geom["hull_proj_pt_vox"] is not None and d_p95 > 0.02:
            p_c = geom["deepest_pt_vox"]
            p_h = geom["hull_proj_pt_vox"]
            if plane == "xy":
                x_c, y_c = p_c[2] - x0, p_c[1] - y0
                x_h, y_h = p_h[2] - x0, p_h[1] - y0
            else:
                x_c, y_c = p_c[2] - x0, p_c[0] - z0
                x_h, y_h = p_h[2] - x0, p_h[0] - z0
            # Ensure visible projected segment along synapse normal if out-of-plane component dominates
            dx_seg, dy_seg = x_h - x_c, y_h - y_c
            if np.hypot(dx_seg, dy_seg) < (0.5 * d_p95 / VOXEL_UM):
                n_2d = np.array([geom["n_syn"][2], geom["n_syn"][1 if plane == "xy" else 0]], dtype=float)
                n_2d_norm = float(np.linalg.norm(n_2d))
                if n_2d_norm > 1e-4:
                    n_2d /= n_2d_norm
                    x_h = x_c + (d_p95 / VOXEL_UM) * n_2d[0]
                    y_h = y_c + (d_p95 / VOXEL_UM) * n_2d[1]
            ax.annotate(
                "",
                xy=(x_c, y_c),
                xytext=(x_h, y_h),
                arrowprops={"arrowstyle": "<->", "color": COL_DANGER, "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
                zorder=10,
            )

        add_scale_bar(ax, h_s, w_s, bar_um=5.0)
        slice_lbl = f"XY slice (z = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)" if plane == "xy" else f"XZ slice (y = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)"
        ax.set_title(slice_lbl, fontsize=9.2, fontweight="bold", color=COL_INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend and key numbers in right column
    legend_handles = [
        Line2D([0], [0], color=COL_PLOT1, lw=1.8, label="Target mask outline"),
        Line2D([0], [0], color=COL_PLOT2, lw=1.6, label="T-cell mask outline"),
        Line2D([0], [0], color=COL_PLOT4, marker="s", ls="none", markersize=5, label="Contact patch voxels"),
        Line2D([0], [0], color=COL_PLOT3, lw=1.3, ls="--", label="Target convex hull"),
        Line2D([0], [0], color=COL_DANGER, lw=1.6, label="Cup depth (p95)"),
    ]
    ax_txt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.2, bbox_to_anchor=(0.0, 1.0))

    d_val = float(item["cup_depth_hull_p95_um"])
    r_val = float(item["patch_equiv_radius_um"])
    dr_val = float(item["cup_depth_over_radius"])
    ca_val = float(item["contact_area_um2"])
    v_b = float(item["target_volume_um3"])
    v_t = float(item["tcell_volume_um3"])
    killed_flag = item.get("killed", None)
    donor_str = str(item.get("donor", exp.split("_")[-1]))
    sel_tag = str(item.get("selection_tag", "stratified_random"))

    lines_txt = [
        f"Selection: {sel_tag}",
        f"Donor / Group: {donor_str}",
        f"Cup depth (p95): {d_val:.2f} um",
        f"Patch radius (R_eq): {r_val:.2f} um",
        f"Depth / radius: {dr_val:.3f}",
        f"Contact area: {ca_val:.1f} um^2",
        f"Target volume: {v_b:.0f} um^3",
        f"T-cell volume: {v_t:.0f} um^3",
    ]
    if killed_flag is not None and pd.notna(killed_flag):
        fate_str = "Killed (<=200 min)" if bool(killed_flag) else "Survived (<=200 min)"
        lines_txt.append(f"Target outcome: {fate_str}")

    ax_txt.text(
        0.02,
        0.54,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=8.6,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.42,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.5"},
    )

    out_path = REVIEW_DIR / case_name / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    src_path_str = str(geom["abs_mask_path"].relative_to(REPO_ROOT))
    if geom["raw_h5_path"] is not None:
        src_path_str += f";{geom['raw_h5_path'].relative_to(REPO_ROOT)}"

    return {
        "case": case_name,
        "id": ex_id,
        "experiment": exp,
        "roi": roi,
        "visit": visit,
        "sweep": sweep,
        "condition": cond,
        "selection_tag": sel_tag,
        "source_data_path": src_path_str,
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "has_local_raw_h5": bool(geom["raw_h5_path"] is not None),
        "voxel_size_um": VOXEL_UM,
        "slice_zc_vox": zc,
        "slice_yc_vox": yc,
        "cup_depth_hull_p95_um": round(d_val, 4),
        "patch_equiv_radius_um": round(r_val, 4),
        "cup_depth_over_radius": round(dr_val, 4),
        "contact_area_um2": round(ca_val, 3),
        "target_volume_um3": round(v_b, 2),
        "tcell_volume_um3": round(v_t, 2),
        "killed_200m": bool(killed_flag) if (killed_flag is not None and pd.notna(killed_flag)) else "",
    }


def prepare_and_solve_gateB_visit(
    item: dict[str, Any],
    pi_solve: float,
    target_edge_um: float = 1.0,
) -> dict[str, Any]:
    """
    Prepare the exact Gate-B unloaded T-cell and target reference meshes for a visit
    using `_prepare_visit_meshes_at_edge` from `src/mech/run_gateB_check.py`,
    and solve the two-shell equilibrium at `pi_solve` (warm-started along `PI_NODES_GATEB`
    up to `pi_solve` so the equilibrium matches Gate B's warm-started continuation path).
    Also returns the pre-push-apart meshes `(V_t_before, V_g_before)` for Case 4 (`initial_overlap`).
    """
    exp = str(item["experiment"])
    roi = str(item["roi"])
    visit = int(item["visit"])
    vrow = item["vrow"]
    sw_info = item["sweep_info"]
    pre_mask_info = item.get("pre_mask_info", None)
    first_t_mask_info = item.get("first_t_mask_info", None)

    V_ref_raw = F_ref_raw = None
    has_pre = False
    if pre_mask_info is not None:
        V_ref_raw, F_ref_raw = extract_cell_mesh_marching_cubes(
            pre_mask_info[0], pre_mask_info[1], target_edge_um=target_edge_um
        )
        has_pre = True

    prep = _prepare_visit_meshes_at_edge(
        vrow=vrow,
        sw_info=sw_info,
        V_ref_base_raw=V_ref_raw,
        F_ref_base_raw=F_ref_raw,
        has_pre_contact_ref=has_pre,
        first_t_mask_info=first_t_mask_info,
        target_edge_um=target_edge_um,
    )
    t_mesh: ShellMesh = prep["t_mesh"]
    g_mesh: ShellMesh = prep["g_mesh"]
    push_apart_um = float(prep["push_apart_um"])
    min_sep_before = float(prep["min_sep_before_push_um"])
    min_sep_after = float(prep["min_sep_after_push_um"])

    # Reconstruct pre-push-apart positions V_t_before, V_g_before from syn_dir and push_apart_um
    c_t0 = np.mean(t_mesh.V0, axis=0)
    c_g0 = np.mean(g_mesh.V0, axis=0)
    syn_w = np.clip(t_mesh.synapse_faces.astype(float), 0.0, 1.0)
    if np.any(syn_w > 0.05):
        f_cents_t = np.mean(t_mesh.V0[t_mesh.F], axis=1)
        w_f = t_mesh.A_f0 * syn_w
        p_syn = np.sum(f_cents_t * w_f[:, None], axis=0) / max(1e-8, float(np.sum(w_f)))
        u_syn = c_g0 - p_syn
    else:
        u_syn = c_g0 - c_t0
    syn_dir = u_syn / max(1e-8, float(np.linalg.norm(u_syn)))
    V_t_before = t_mesh.V0 + 0.5 * push_apart_um * syn_dir[None, :]
    V_g_before = g_mesh.V0 - 0.5 * push_apart_um * syn_dir[None, :]

    if pi_solve < 0.0:
        # Only need initial overlap meshes (no solve needed)
        return {
            "t_mesh": t_mesh,
            "g_mesh": g_mesh,
            "V_t_before": V_t_before,
            "V_g_before": V_g_before,
            "syn_dir": syn_dir,
            "push_apart_um": push_apart_um,
            "min_sep_before_push_um": min_sep_before,
            "min_sep_after_push_um": min_sep_after,
        }

    # Warm-start continuation along PI_NODES_GATEB up to pi_solve
    nodes_below = [float(p) for p in PI_NODES_GATEB if p < pi_solve - 1e-6]
    solve_seq = nodes_below + [float(pi_solve)]
    x_warm = None
    base_hull_d = 0.0
    sol_final = None
    desc_final = None
    d_delta_final = 0.0
    d_raw_final = 0.0

    for idx_s, p_val in enumerate(solve_seq):
        params_p = ShellParameters.from_dimensionless(
            pi_push=float(p_val),
            chi_t=1.0,
            alpha_syn=2.0,
            r_target_um=g_mesh.R_eq0,
            subtract_prestress=True,
        )
        sol_p = solve_two_shell_equilibrium(
            t_mesh, g_mesh, params_p, maxiter=450, gtol=5e-2, x0=x_warm
        )
        x_warm = sol_p["x_eq"]
        if idx_s == 0 or idx_s == len(solve_seq) - 1:
            desc_p = compute_mesh_descriptors(
                sol_p["V_tcell"], t_mesh.F, sol_p["V_target"], g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
            )
            raw_h = float(desc_p["cup_depth_hull_p95_um"])
            if idx_s == 0:
                base_hull_d = raw_h
            if idx_s == len(solve_seq) - 1:
                sol_final = sol_p
                desc_final = desc_p
                d_raw_final = raw_h
                d_delta_final = max(0.0, raw_h - base_hull_d) if p_val > 0.0 else 0.0

    assert sol_final is not None and desc_final is not None
    bd = sol_final["breakdown"]
    return {
        "t_mesh": t_mesh,
        "g_mesh": g_mesh,
        "V_t_before": V_t_before,
        "V_g_before": V_g_before,
        "syn_dir": syn_dir,
        "push_apart_um": push_apart_um,
        "min_sep_before_push_um": min_sep_before,
        "min_sep_after_push_um": min_sep_after,
        "V_t_solved": sol_final["V_tcell"],
        "V_g_solved": sol_final["V_target"],
        "d_delta_solved_um": d_delta_final,
        "d_raw_solved_um": d_raw_final,
        "base_hull_d_um": base_hull_d,
        "min_sep_solved_um": float(bd["min_contact_sep_um"]),
        "max_vol_rel_err": float(max(bd["vol_tcell_rel_err"], bd["vol_target_rel_err"])),
        "n_inverted": int(bd["n_inverted_total"]),
        "nit": int(sol_final["nit"]),
        "converged": bool(sol_final["success"]),
    }


def render_fit_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one panel for Case 2 (`fit_reachable`, solved at `pi_fit_delta`) or Case 3 (`fit_unreachable`, solved at `Pi = 12`).
    Overlays on the real XY and XZ mask/raw slices:
      - Real target mask outline (`#1F4E79` solid) and T-cell mask outline (`#C0561F` solid)
      - Reference (unloaded) target mesh cross-section (`#2E7D5B` dotted)
      - SOLVED target mesh cross-section (`#6B4C9A` dashed bold) and SOLVED T-cell mesh cross-section (`#C0561F` dashed)
    Plus an inset plot of that visit's `depth(Pi)` curve across `Pi in [0..12]` with the observed depth horizontal line.
    """
    setup_house_style()
    case_name = str(item["case"])
    ex_id = str(item["id"])
    exp = str(item["experiment"])
    roi = str(item["roi"])
    visit = int(item["visit"])
    cond = str(item["condition"])
    sw_info = item["sweep_info"]
    sweep = int(sw_info["sweep"])
    gb_row = item["gb_row"]
    sel_tag = str(item.get("selection_tag", "stratified_random"))

    is_unreach = bool(gb_row["unreachable_delta"])
    pi_solve = 12.0 if (case_name == "fit_unreachable" or is_unreach) else float(gb_row["pi_fit_delta"])
    if not np.isfinite(pi_solve):
        pi_solve = 12.0

    geom = load_slice_Geometry_and_raw(exp, roi, visit, sw_info)
    sol_info = prepare_and_solve_gateB_visit(item, pi_solve=pi_solve, target_edge_um=1.0)

    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc, xc = geom["zc"], geom["yc"], geom["xc"]

    rgb_xy, is_raw_xy = make_background_image(
        geom["I_xy"], geom["b_xy"], geom["t_xy"], geom["raw_xy_actin"], geom["raw_xy_memb"]
    )
    rgb_xz, _ = make_background_image(
        geom["I_xz"], geom["b_xz"], geom["t_xz"], geom["raw_xz_actin"], geom["raw_xz_memb"]
    )

    fig = plt.figure(figsize=(10.8, 4.6), dpi=105)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.05], wspace=0.18, left=0.03, right=0.98, top=0.84, bottom=0.08)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    gs_right = gs[0, 2].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.28)
    ax_curve = fig.add_subplot(gs_right[0, 0])
    ax_txt = fig.add_subplot(gs_right[1, 0])
    ax_txt.axis("off")

    bg_label = "Raw 561+488 nm + masks" if is_raw_xy else "3D U-Net segmentation mask"
    fig.suptitle(
        f"{case_name} | {ex_id} | {exp} | {roi} | visit {visit} | sweep {sweep}",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.03,
        0.90,
        f"Condition: {cond} | Source: {bg_label} | Voxel pitch: dx = dy = dz = 0.1962 um (z = 0.196 um)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    t_mesh: ShellMesh = sol_info["t_mesh"]
    g_mesh: ShellMesh = sol_info["g_mesh"]
    V_t_sol = sol_info["V_t_solved"]
    V_g_sol = sol_info["V_g_solved"]

    for ax, rgb, b_sl, t_sl, plane, sl_idx, c_coord in [
        (ax_xy, rgb_xy, geom["b_xy"], geom["t_xy"], "xy", zc, (xc - x0, yc - y0)),
        (ax_xz, rgb_xz, geom["b_xz"], geom["t_xz"], "xz", yc, (xc - x0, zc - z0)),
    ]:
        h_s, w_s = rgb.shape[:2]
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

        # Real segmentation mask contours
        if np.any(b_sl):
            ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.5, zorder=4)
        if np.any(t_sl):
            ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_PLOT2], linewidths=1.4, zorder=4)

        val_um = (zc * VOXEL_UM) if plane == "xy" else (yc * VOXEL_UM)
        axis_i = 0 if plane == "xy" else 1

        # Unloaded reference target mesh cross-section
        segs_g_ref = intersect_mesh_with_plane(g_mesh.V0, g_mesh.F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g_ref, plane, x0, y0, z0, color=COL_PLOT3, lw=1.3, ls=":", zorder=5)

        # Solved target mesh cross-section
        segs_g_sol = intersect_mesh_with_plane(V_g_sol, g_mesh.F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g_sol, plane, x0, y0, z0, color=COL_PLOT4, lw=2.0, ls="--", zorder=7)

        # Solved T-cell mesh cross-section
        segs_t_sol = intersect_mesh_with_plane(V_t_sol, t_mesh.F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_t_sol, plane, x0, y0, z0, color=COL_INK, lw=1.5, ls="--", zorder=6)

        add_scale_bar(ax, h_s, w_s, bar_um=5.0)
        slice_lbl = f"XY slice (z = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)" if plane == "xy" else f"XZ slice (y = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)"
        ax.set_title(slice_lbl, fontsize=9.2, fontweight="bold", color=COL_INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Depth vs Pi curve inset
    d_delta_curve = [float(gb_row[f"d_delta_pi_{str(p).replace('.', 'p')}"]) for p in PI_NODES_GATEB]
    d_raw_curve = [float(gb_row[f"d_raw_pi_{str(p).replace('.', 'p')}"]) for p in PI_NODES_GATEB]
    d_obs = float(gb_row["cup_depth_hull_p95_um"])

    ax_curve.plot(PI_NODES_GATEB, d_delta_curve, "-o", color=COL_PLOT1, lw=1.5, markersize=3.8, label="Solved net d(Pi)-d(0)")
    ax_curve.plot(PI_NODES_GATEB, d_raw_curve, "--s", color=COL_PLOT3, lw=1.1, markersize=3.0, label="Solved raw hull d(Pi)")
    ax_curve.axhline(d_obs, color=COL_DANGER, lw=1.4, ls="-", label=f"Observed d = {d_obs:.2f} um")
    ax_curve.axvline(pi_solve, color=COL_PLOT4, lw=1.2, ls=":", label=f"Solved Pi = {pi_solve:.2f}")
    ax_curve.plot([pi_solve], [sol_info["d_delta_solved_um"]], "D", color=COL_PLOT4, markersize=5.5, zorder=6)

    ax_curve.set_xlabel("Dimensionless push Pi", fontsize=8.2, labelpad=2)
    ax_curve.set_ylabel("Cup depth (um)", fontsize=8.2, labelpad=2)
    ax_curve.set_title("Gate-B depth-vs-Pi curve", fontsize=8.8, fontweight="bold", loc="left", pad=3)
    ax_curve.tick_params(labelsize=7.5)
    ax_curve.grid(True, color=COL_RULE, lw=0.6)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.legend(loc="best", frameon=False, fontsize=6.8)

    # Legend for slices + text box
    ref_type = str(gb_row["ref_type"])
    d_max_del = float(gb_row["max_depth_delta_um"])
    lines_txt = [
        f"Selection: {sel_tag} ({ref_type} ref)",
        f"Observed depth: {d_obs:.2f} um | R_patch: {float(gb_row['patch_equiv_radius_um']):.2f} um",
        f"Solved at Pi = {pi_solve:.2f}: net d = {sol_info['d_delta_solved_um']:.2f} um (raw {sol_info['d_raw_solved_um']:.2f} um)",
        f"Max curve net depth: {d_max_del:.2f} um (at Pi = {float(gb_row['pi_at_max_delta']):.1f})",
        f"Min contact sep: {sol_info['min_sep_solved_um']:.3f} um | Vol err: {sol_info['max_vol_rel_err']*100:.2f}%",
        "Slice key: Target mask (blue), T-cell mask (orange),",
        "  Ref target (green dot), Solved target (purple dash),",
        "  Solved T-cell (black dash)",
    ]
    ax_txt.text(
        0.0,
        0.98,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=7.9,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.32,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.4"},
    )

    out_path = REVIEW_DIR / case_name / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    src_path_str = str(geom["abs_mask_path"].relative_to(REPO_ROOT))
    if geom["raw_h5_path"] is not None:
        src_path_str += f";{geom['raw_h5_path'].relative_to(REPO_ROOT)}"

    return {
        "case": case_name,
        "id": ex_id,
        "experiment": exp,
        "roi": roi,
        "visit": visit,
        "sweep": sweep,
        "condition": cond,
        "selection_tag": sel_tag,
        "source_data_path": src_path_str,
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "has_local_raw_h5": bool(geom["raw_h5_path"] is not None),
        "voxel_size_um": VOXEL_UM,
        "slice_zc_vox": zc,
        "slice_yc_vox": yc,
        "ref_type": ref_type,
        "cup_depth_hull_p95_um": round(d_obs, 4),
        "patch_equiv_radius_um": round(float(gb_row["patch_equiv_radius_um"]), 4),
        "pi_solved": round(pi_solve, 4),
        "solved_depth_delta_um": round(sol_info["d_delta_solved_um"], 4),
        "solved_depth_raw_hull_um": round(sol_info["d_raw_solved_um"], 4),
        "max_curve_depth_delta_um": round(d_max_del, 4),
        "min_contact_sep_um": round(sol_info["min_sep_solved_um"], 4),
        "max_vol_rel_err": round(sol_info["max_vol_rel_err"], 6),
        "n_inverted": sol_info["n_inverted"],
        "converged": sol_info["converged"],
    }


def render_initial_overlap_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one panel for Case 4 (`initial_overlap`).
    Shows reference target outline vs T-cell mesh outline BEFORE push-apart and AFTER push-apart
    (`resolve_initial_contact_overlap`), overlaid on the real XY and XZ mask slices,
    and states the push-apart distance and min contact separation before/after.
    """
    setup_house_style()
    case_name = "initial_overlap"
    ex_id = str(item["id"])
    exp = str(item["experiment"])
    roi = str(item["roi"])
    visit = int(item["visit"])
    cond = str(item["condition"])
    sw_info = item["sweep_info"]
    sweep = int(sw_info["sweep"])
    gb_row = item["gb_row"]
    sel_tag = str(item.get("selection_tag", "stratified_random"))

    geom = load_slice_Geometry_and_raw(exp, roi, visit, sw_info)
    ov_info = prepare_and_solve_gateB_visit(item, pi_solve=-1.0, target_edge_um=1.0)

    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc, xc = geom["zc"], geom["yc"], geom["xc"]

    rgb_xy, is_raw_xy = make_background_image(
        geom["I_xy"], geom["b_xy"], geom["t_xy"], geom["raw_xy_actin"], geom["raw_xy_memb"]
    )
    rgb_xz, _ = make_background_image(
        geom["I_xz"], geom["b_xz"], geom["t_xz"], geom["raw_xz_actin"], geom["raw_xz_memb"]
    )

    fig = plt.figure(figsize=(10.5, 4.4), dpi=105)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.98], wspace=0.15, left=0.03, right=0.98, top=0.84, bottom=0.06)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    bg_label = "Raw 561+488 nm + masks" if is_raw_xy else "3D U-Net segmentation mask"
    fig.suptitle(
        f"{case_name} | {ex_id} | {exp} | {roi} | visit {visit} | sweep {sweep}",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.03,
        0.90,
        f"Condition: {cond} | Source: {bg_label} | Voxel pitch: dx = dy = dz = 0.1962 um (z = 0.196 um)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    t_mesh: ShellMesh = ov_info["t_mesh"]
    g_mesh: ShellMesh = ov_info["g_mesh"]
    V_t_bef = ov_info["V_t_before"]
    V_g_bef = ov_info["V_g_before"]
    push_um = float(ov_info["push_apart_um"])
    sep_bef = float(ov_info["min_sep_before_push_um"])
    sep_aft = float(ov_info["min_sep_after_push_um"])

    for ax, rgb, b_sl, t_sl, plane, sl_idx, c_coord in [
        (ax_xy, rgb_xy, geom["b_xy"], geom["t_xy"], "xy", zc, (xc - x0, yc - y0)),
        (ax_xz, rgb_xz, geom["b_xz"], geom["t_xz"], "xz", yc, (xc - x0, zc - z0)),
    ]:
        h_s, w_s = rgb.shape[:2]
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

        # Real mask contours (subtle)
        if np.any(b_sl):
            ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_LINE], linewidths=1.0, zorder=3)
        if np.any(t_sl):
            ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_LINE], linewidths=1.0, zorder=3)

        val_um = (zc * VOXEL_UM) if plane == "xy" else (yc * VOXEL_UM)
        axis_i = 0 if plane == "xy" else 1

        # BEFORE push-apart: reference target (#B3261E dashed) and T-cell (#C0561F dotted)
        segs_g_bef = intersect_mesh_with_plane(V_g_bef, g_mesh.F, axis=axis_i, val_um=val_um)
        segs_t_bef = intersect_mesh_with_plane(V_t_bef, t_mesh.F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g_bef, plane, x0, y0, z0, color=COL_DANGER, lw=1.6, ls="--", zorder=5)
        plot_mesh_slice_segments(ax, segs_t_bef, plane, x0, y0, z0, color=COL_PLOT2, lw=1.5, ls=":", zorder=5)

        # AFTER push-apart: reference target (#1F4E79 solid) and T-cell (#2E7D5B solid)
        segs_g_aft = intersect_mesh_with_plane(g_mesh.V0, g_mesh.F, axis=axis_i, val_um=val_um)
        segs_t_aft = intersect_mesh_with_plane(t_mesh.V0, t_mesh.F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g_aft, plane, x0, y0, z0, color=COL_PLOT1, lw=2.0, ls="-", zorder=6)
        plot_mesh_slice_segments(ax, segs_t_aft, plane, x0, y0, z0, color=COL_PLOT3, lw=1.8, ls="-", zorder=6)

        add_scale_bar(ax, h_s, w_s, bar_um=5.0)
        slice_lbl = f"XY slice (z = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)" if plane == "xy" else f"XZ slice (y = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)"
        ax.set_title(slice_lbl, fontsize=9.2, fontweight="bold", color=COL_INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Line2D([0], [0], color=COL_DANGER, lw=1.6, ls="--", label="Ref target BEFORE push-apart"),
        Line2D([0], [0], color=COL_PLOT2, lw=1.5, ls=":", label="T-cell mesh BEFORE push-apart"),
        Line2D([0], [0], color=COL_PLOT1, lw=2.0, ls="-", label="Ref target AFTER push-apart"),
        Line2D([0], [0], color=COL_PLOT3, lw=1.8, ls="-", label="T-cell mesh AFTER push-apart"),
        Line2D([0], [0], color=COL_LINE, lw=1.0, ls="-", label="Deformed mask outline"),
    ]
    ax_txt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.0, bbox_to_anchor=(0.0, 1.0))

    lines_txt = [
        f"Selection: {sel_tag}",
        f"Reference type: {gb_row['ref_type']}",
        f"Min sep BEFORE push-apart: {sep_bef:+.3f} um",
        f"Push-apart distance: {push_um:.3f} um",
        f"Min sep AFTER push-apart: {sep_aft:+.3f} um",
        f"Observed cup depth: {float(gb_row['cup_depth_hull_p95_um']):.2f} um",
        f"Patch radius: {float(gb_row['patch_equiv_radius_um']):.2f} um",
        f"Target R_eq0: {float(gb_row['r_target_um']):.2f} um",
    ]
    ax_txt.text(
        0.02,
        0.52,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.40,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.5"},
    )

    out_path = REVIEW_DIR / case_name / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    src_path_str = str(geom["abs_mask_path"].relative_to(REPO_ROOT))
    if geom["raw_h5_path"] is not None:
        src_path_str += f";{geom['raw_h5_path'].relative_to(REPO_ROOT)}"

    return {
        "case": case_name,
        "id": ex_id,
        "experiment": exp,
        "roi": roi,
        "visit": visit,
        "sweep": sweep,
        "condition": cond,
        "selection_tag": sel_tag,
        "source_data_path": src_path_str,
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "has_local_raw_h5": bool(geom["raw_h5_path"] is not None),
        "voxel_size_um": VOXEL_UM,
        "slice_zc_vox": zc,
        "slice_yc_vox": yc,
        "ref_type": str(gb_row["ref_type"]),
        "min_sep_before_push_um": round(sep_bef, 4),
        "push_apart_um": round(push_um, 4),
        "min_sep_after_push_um": round(sep_aft, 4),
        "cup_depth_hull_p95_um": round(float(gb_row["cup_depth_hull_p95_um"]), 4),
        "patch_equiv_radius_um": round(float(gb_row["patch_equiv_radius_um"]), 4),
    }


def render_mesh_size_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one panel for Case 5 (`mesh_size` [6]):
    Solves the SAME visit at `target_edge_um = 1.0 um` and `target_edge_um = 0.5 um` at the visit's
    fitted Pi (`pi_fit_delta`), and overlays both solved target & T-cell outlines on the real XY and XZ mask slices.
    """
    setup_house_style()
    case_name = "mesh_size"
    ex_id = str(item["id"])
    exp = str(item["experiment"])
    roi = str(item["roi"])
    visit = int(item["visit"])
    cond = str(item["condition"])
    sw_info = item["sweep_info"]
    sweep = int(sw_info["sweep"])
    gb_row = item["gb_row"]
    sel_tag = str(item.get("selection_tag", "stratified_random"))

    pi_solve = float(gb_row["pi_fit_delta"]) if np.isfinite(float(gb_row["pi_fit_delta"])) else 6.0
    pi_solve = float(np.clip(pi_solve, 1.2, 8.0))

    geom = load_slice_Geometry_and_raw(exp, roi, visit, sw_info)
    sol_10 = prepare_and_solve_gateB_visit(item, pi_solve=pi_solve, target_edge_um=1.0)
    sol_05 = prepare_and_solve_gateB_visit(item, pi_solve=pi_solve, target_edge_um=0.5)

    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc, xc = geom["zc"], geom["yc"], geom["xc"]

    rgb_xy, is_raw_xy = make_background_image(
        geom["I_xy"], geom["b_xy"], geom["t_xy"], geom["raw_xy_actin"], geom["raw_xy_memb"]
    )
    rgb_xz, _ = make_background_image(
        geom["I_xz"], geom["b_xz"], geom["t_xz"], geom["raw_xz_actin"], geom["raw_xz_memb"]
    )

    fig = plt.figure(figsize=(10.6, 4.4), dpi=105)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.15, left=0.03, right=0.98, top=0.84, bottom=0.06)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    bg_label = "Raw 561+488 nm + masks" if is_raw_xy else "3D U-Net segmentation mask"
    fig.suptitle(
        f"{case_name} | {ex_id} | {exp} | {roi} | visit {visit} | sweep {sweep}",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.03,
        0.90,
        f"Condition: {cond} | Source: {bg_label} | Voxel pitch: dx = dy = dz = 0.1962 um (z = 0.196 um)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    for ax, rgb, b_sl, t_sl, plane, sl_idx, c_coord in [
        (ax_xy, rgb_xy, geom["b_xy"], geom["t_xy"], "xy", zc, (xc - x0, yc - y0)),
        (ax_xz, rgb_xz, geom["b_xz"], geom["t_xz"], "xz", yc, (xc - x0, zc - z0)),
    ]:
        h_s, w_s = rgb.shape[:2]
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

        if np.any(b_sl):
            ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_LINE], linewidths=1.1, zorder=3)
        if np.any(t_sl):
            ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_LINE], linewidths=1.1, zorder=3)

        val_um = (zc * VOXEL_UM) if plane == "xy" else (yc * VOXEL_UM)
        axis_i = 0 if plane == "xy" else 1

        # 1.0 um mesh outlines
        segs_g10 = intersect_mesh_with_plane(sol_10["V_g_solved"], sol_10["g_mesh"].F, axis=axis_i, val_um=val_um)
        segs_t10 = intersect_mesh_with_plane(sol_10["V_t_solved"], sol_10["t_mesh"].F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g10, plane, x0, y0, z0, color=COL_PLOT1, lw=2.0, ls="-", zorder=6)
        plot_mesh_slice_segments(ax, segs_t10, plane, x0, y0, z0, color=COL_PLOT2, lw=1.6, ls="-", zorder=5)

        # 0.5 um mesh outlines
        segs_g05 = intersect_mesh_with_plane(sol_05["V_g_solved"], sol_05["g_mesh"].F, axis=axis_i, val_um=val_um)
        segs_t05 = intersect_mesh_with_plane(sol_05["V_t_solved"], sol_05["t_mesh"].F, axis=axis_i, val_um=val_um)
        plot_mesh_slice_segments(ax, segs_g05, plane, x0, y0, z0, color=COL_PLOT3, lw=1.8, ls="--", zorder=7)
        plot_mesh_slice_segments(ax, segs_t05, plane, x0, y0, z0, color=COL_PLOT4, lw=1.5, ls=":", zorder=6)

        add_scale_bar(ax, h_s, w_s, bar_um=5.0)
        slice_lbl = f"XY slice (z = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)" if plane == "xy" else f"XZ slice (y = {sl_idx}, {sl_idx * VOXEL_UM:.1f} um)"
        ax.set_title(slice_lbl, fontsize=9.2, fontweight="bold", color=COL_INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Line2D([0], [0], color=COL_PLOT1, lw=2.0, ls="-", label="Solved target (h = 1.0 um)"),
        Line2D([0], [0], color=COL_PLOT3, lw=1.8, ls="--", label="Solved target (h = 0.5 um)"),
        Line2D([0], [0], color=COL_PLOT2, lw=1.6, ls="-", label="Solved T-cell (h = 1.0 um)"),
        Line2D([0], [0], color=COL_PLOT4, lw=1.5, ls=":", label="Solved T-cell (h = 0.5 um)"),
        Line2D([0], [0], color=COL_LINE, lw=1.1, ls="-", label="Real mask outline"),
    ]
    ax_txt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.0, bbox_to_anchor=(0.0, 1.0))

    d_obs = float(gb_row["cup_depth_hull_p95_um"])
    d10_raw = float(sol_10["d_raw_solved_um"])
    d05_raw = float(sol_05["d_raw_solved_um"])
    d10_del = float(sol_10["d_delta_solved_um"])
    d05_del = float(sol_05["d_delta_solved_um"])
    rel_diff_raw = abs(d10_raw - d05_raw) / max(0.10, d05_raw)

    lines_txt = [
        f"Selection: {sel_tag}",
        f"Solved at Pi = {pi_solve:.2f} | Obs depth: {d_obs:.2f} um",
        f"1.0 um target mesh: {len(sol_10['g_mesh'].V0)} verts, {len(sol_10['g_mesh'].F)} faces",
        f"0.5 um target mesh: {len(sol_05['g_mesh'].V0)} verts, {len(sol_05['g_mesh'].F)} faces",
        f"Solved raw depth (1.0 vs 0.5): {d10_raw:.2f} vs {d05_raw:.2f} um ({rel_diff_raw*100:.1f}%)",
        f"Solved net depth (1.0 vs 0.5): {d10_del:.2f} vs {d05_del:.2f} um",
        f"Min sep (1.0 / 0.5): {sol_10['min_sep_solved_um']:+.3f} / {sol_05['min_sep_solved_um']:+.3f} um",
        f"Vol err (1.0 / 0.5): {sol_10['max_vol_rel_err']*100:.2f}% / {sol_05['max_vol_rel_err']*100:.2f}%",
    ]
    ax_txt.text(
        0.02,
        0.52,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=8.1,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.36,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.45"},
    )

    out_path = REVIEW_DIR / case_name / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    src_path_str = str(geom["abs_mask_path"].relative_to(REPO_ROOT))
    if geom["raw_h5_path"] is not None:
        src_path_str += f";{geom['raw_h5_path'].relative_to(REPO_ROOT)}"

    return {
        "case": case_name,
        "id": ex_id,
        "experiment": exp,
        "roi": roi,
        "visit": visit,
        "sweep": sweep,
        "condition": cond,
        "selection_tag": sel_tag,
        "source_data_path": src_path_str,
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "has_local_raw_h5": bool(geom["raw_h5_path"] is not None),
        "voxel_size_um": VOXEL_UM,
        "slice_zc_vox": zc,
        "slice_yc_vox": yc,
        "pi_solved": round(pi_solve, 4),
        "cup_depth_hull_p95_um": round(d_obs, 4),
        "d_raw_1p0_um": round(d10_raw, 4),
        "d_raw_0p5_um": round(d05_raw, 4),
        "d_delta_1p0_um": round(d10_del, 4),
        "d_delta_0p5_um": round(d05_del, 4),
        "rel_diff_raw_1p0_vs_0p5": round(rel_diff_raw, 4),
        "n_verts_target_1p0": int(len(sol_10["g_mesh"].V0)),
        "n_verts_target_0p5": int(len(sol_05["g_mesh"].V0)),
    }


def render_death_labels_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one panel for Case 6 (`death_labels` [12 killed + 12 surviving + 6 dasatinib = 30]).
    Shows the real 405-nm death-dye channel (`*_MIPxy_405.tif` and `*_MIPxz_405.tif`, plus 3D raw
    slice if raw `.h5` exists locally) around the target at the 3 visits before / at / after the called
    death visit (or before / peak-signal / after for surviving/dasatinib chambers), with the target
    mask outline (`#1F4E79`), T-cell mask outline (`#C0561F`), cleaned dead-dye mask outline
    (`> 404.0 ADU` after `sigma=3.0` Gaussian + `disk(3)` closing + area `>=20 px` from `compute_2d_dead_mask`),
    and the called per-visit and per-chamber labels.
    """
    setup_house_style()
    case_name = "death_labels"
    ex_id = str(item["id"])
    exp = str(item["experiment"])
    roi = str(item["roi"])
    cond = str(item["condition"])
    subcat = str(item["subcategory"])
    sel_tag = str(item.get("selection_tag", "stratified_random"))
    three_visits: list[dict[str, Any]] = item["three_visits"]
    ch_row = item["ch_row"]

    v_center = int(three_visits[1]["visit"])
    stem_center = str(three_visits[1]["stem"]).zfill(5)

    # Load all 3 visits' MIPxy_405, MIPxz_405, and 3D predictions.h5
    loaded = []
    union_xy = np.zeros((256, 256), dtype=bool)
    max_z = 190
    union_xz_x = np.zeros(256, dtype=bool)
    union_xz_z = np.zeros(256, dtype=bool)

    src_paths = []
    has_any_raw = False

    for v_info in three_visits:
        v_num = int(v_info["visit"])
        stem = str(v_info["stem"]).zfill(5)
        xy_p = MIPS_DIR / exp / roi / f"{stem}_MIPxy_405.tif"
        xz_p = MIPS_DIR / exp / roi / f"{stem}_MIPxz_405.tif"
        pred_p = MASKS_DIR / exp / roi / f"{stem}_predictions.h5"
        raw_p = find_local_raw_h5(exp, roi, stem)
        if raw_p is not None:
            has_any_raw = True

        src_paths.append(str(xy_p.relative_to(REPO_ROOT)))
        src_paths.append(str(xz_p.relative_to(REPO_ROOT)))
        src_paths.append(str(pred_p.relative_to(REPO_ROOT)))
        if raw_p is not None:
            src_paths.append(str(raw_p.relative_to(REPO_ROOT)))

        mip_xy = tifffile.imread(xy_p)
        mip_xz = tifffile.imread(xz_p)
        with h5py.File(pred_p, "r") as hf:
            pred = hf["predictions"][:]  # (Z, 256, 256)

        max_z = pred.shape[0]
        raw_b = (pred == 2) | (pred == 4)
        raw_t = (pred == 1) | (pred == 3)
        # Exact projection orientation from src/death_labels.py lines 28-29 & 445-446:
        b_xy = raw_b.max(axis=0).T
        b_xz = raw_b.max(axis=1).T
        t_xy = raw_t.max(axis=0).T
        t_xz = raw_t.max(axis=1).T

        d_xy, g_xy = compute_2d_dead_mask(mip_xy, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)
        d_xz, g_xz = compute_2d_dead_mask(mip_xz, THRESH_2D_ADU, MIN_COMPONENT_AREA_2D)

        ov_xy = int((d_xy & b_xy).sum())
        ov_xz = int((d_xz & b_xz).sum())
        max_g_xy = float(g_xy[b_xy].max()) if np.any(b_xy) else float(g_xy.max())
        max_g_xz = float(g_xz[b_xz].max()) if np.any(b_xz) else float(g_xz.max())
        dead_v_call = bool((ov_xy >= MIN_OVERLAP_PX_2D) and (ov_xz >= MIN_OVERLAP_PX_2D))

        union_xy |= b_xy | t_xy
        if np.any(b_xz | t_xz):
            # In b_xz (shape 256 x Z), row is X (0..255), col is Z (0..Z-1)
            ux, uz = np.where(b_xz | t_xz)
            union_xz_x[ux] = True
            union_xz_z[uz] = True

        loaded.append(
            {
                "visit": v_num,
                "role_label": str(v_info["role_label"]),
                "stem": stem,
                "mip_xy": mip_xy,
                "mip_xz": mip_xz,
                "b_xy": b_xy,
                "b_xz": b_xz,
                "t_xy": t_xy,
                "t_xz": t_xz,
                "d_xy": d_xy,
                "d_xz": d_xz,
                "ov_xy": ov_xy,
                "ov_xz": ov_xz,
                "max_g_xy": max_g_xy,
                "max_g_xz": max_g_xz,
                "dead_visit": dead_v_call,
            }
        )

    # Determine common crop window around the target across the 3 visits
    if np.any(union_xy):
        ys, xs = np.where(union_xy)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        span = max(int(ys.max() - ys.min()), int(xs.max() - xs.min()), 76)
    else:
        cy, cx, span = 128, 128, 96
    side = min(256, max(96, span + 36))
    y0 = int(np.clip(cy - side // 2, 0, 256 - side))
    y1 = y0 + side
    x0 = int(np.clip(cx - side // 2, 0, 256 - side))
    x1 = x0 + side

    # Note: In MIPxy (`raw.max(axis=0).T`), row is X (original dim 2) and col is Y (original dim 1).
    # In MIPxz (`raw.max(axis=1).T`), row is X (original dim 2, same as row y0:y1 of MIPxy!) and col is Z (0..Z-1).
    # Wait: let's verify! In `raw[0]` of shape `(Z, Y, X)`:
    # `raw[0].max(axis=0)` has shape `(Y, X)`. Its transpose `.T` has shape `(X, Y)`!
    # `raw[0].max(axis=1)` has shape `(Z, X)`. Its transpose `.T` has shape `(X, Z)`!
    # So in BOTH `MIPxy` and `MIPxz`, axis 0 (rows) is `X` (`y0:y1`)!
    # And in `MIPxz`, axis 1 (cols) is `Z` (`0..max_z`). Let's transpose `MIPxz` crop `(y0:y1, z0:z1)` to `(z0:z1, y0:y1)`
    # so that `X` is horizontal and `Z` is vertical, matching standard XZ slices!
    if np.any(union_xz_z):
        zs = np.where(union_xz_z)[0]
        cz = int(np.mean(zs))
    else:
        cz = max_z // 2
    side_z = min(max_z, side)
    z0 = int(np.clip(cz - side_z // 2, 0, max_z - side_z))
    z1 = z0 + side_z

    fig = plt.figure(figsize=(11.2, 5.2), dpi=105)
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1.0, 1.0, 1.0, 1.08], height_ratios=[1.0, 1.0],
        wspace=0.12, hspace=0.22, left=0.03, right=0.98, top=0.84, bottom=0.05
    )

    ever_dead_200 = bool(ch_row["ever_dead_200m"])
    ever_any_200 = bool(ch_row["ever_dead_any_visit_200m"])
    dv_sust = ch_row["death_visit_200m"]
    dv_any = ch_row["first_any_dead_visit_200m"]
    called_ch_str = (
        f"KILLED (sustained death_visit={int(dv_sust)})"
        if (ever_dead_200 and pd.notna(dv_sust))
        else (
            f"Single-visit pos (visit={int(dv_any)}, sustained=False)"
            if (ever_any_200 and pd.notna(dv_any))
            else "SURVIVING (ever_dead=False)"
        )
    )

    fig.suptitle(
        f"{case_name} ({subcat}) | {ex_id} | {exp} | {roi} | center visit {v_center} | sweep 1",
        fontsize=10.5,
        fontweight="bold",
        color=COL_INK,
        x=0.03,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.03,
        0.90,
        f"Condition: {cond} | Chamber call: {called_ch_str} | Voxel pitch: dx = dy = dz = 0.1962 um (z = 0.196 um)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    vmin_adu, vmax_adu = 395.0, 440.0

    for col_i, ld in enumerate(loaded):
        ax_top = fig.add_subplot(gs[0, col_i])
        ax_bot = fig.add_subplot(gs[1, col_i])

        # XY crop (rows y0:y1, cols x0:x1)
        img_xy = ld["mip_xy"][y0:y1, x0:x1].astype(np.float32)
        b_xy_c = ld["b_xy"][y0:y1, x0:x1]
        t_xy_c = ld["t_xy"][y0:y1, x0:x1]
        d_xy_c = ld["d_xy"][y0:y1, x0:x1]

        # XZ crop: rows of mip_xz are y0:y1 (same axis as mip_xy rows!), cols are z0:z1; transpose so Z is vertical
        img_xz = ld["mip_xz"][y0:y1, z0:z1].T.astype(np.float32)
        b_xz_c = ld["b_xz"][y0:y1, z0:z1].T
        t_xz_c = ld["t_xz"][y0:y1, z0:z1].T
        d_xz_c = ld["d_xz"][y0:y1, z0:z1].T

        for ax, img, b_m, t_m, d_m, pl_lbl, ov_px, mg_val in [
            (ax_top, img_xy, b_xy_c, t_xy_c, d_xy_c, "XY 405-nm MIP", ld["ov_xy"], ld["max_g_xy"]),
            (ax_bot, img_xz, b_xz_c, t_xz_c, d_xz_c, "XZ 405-nm MIP", ld["ov_xz"], ld["max_g_xz"]),
        ]:
            # Light-mode inverted display of 405-nm SYTOX intensity (395 ADU -> white, >=440 ADU -> dark charcoal)
            norm_img = np.clip((img - vmin_adu) / (vmax_adu - vmin_adu), 0.0, 1.0)
            ax.imshow(1.0 - 0.80 * norm_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", origin="upper")
            if np.any(b_m):
                ax.contour(b_m.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.6, zorder=4)
            if np.any(t_m):
                ax.contour(t_m.astype(float), levels=[0.5], colors=[COL_PLOT2], linewidths=1.2, linestyles="--", zorder=4)
            if np.any(d_m):
                ax.contour(d_m.astype(float), levels=[0.5], colors=[COL_DANGER], linewidths=1.6, zorder=5)

            h_s, w_s = img.shape
            if col_i == 0:
                add_scale_bar(ax, h_s, w_s, bar_um=5.0)
            ax.set_xticks([])
            ax.set_yticks([])

            call_tag = "DEAD" if ld["dead_visit"] else "alive"
            tag_col = COL_DANGER if ld["dead_visit"] else COL_INK
            if pl_lbl.startswith("XY"):
                ax.set_title(
                    f"{ld['role_label']}: v{ld['visit']} ({ld['visit']*10} min) [{call_tag}]",
                    fontsize=8.6,
                    fontweight="bold",
                    color=tag_col,
                    pad=4,
                )
            ax.text(
                0.03,
                0.04,
                f"{pl_lbl}: ov={ov_px} px, maxG={mg_val:.1f}",
                transform=ax.transAxes,
                fontsize=7.3,
                color=COL_INK,
                bbox={"facecolor": COL_PAPER, "alpha": 0.84, "edgecolor": COL_RULE, "pad": 1.5},
                zorder=10,
            )

    ax_txt = fig.add_subplot(gs[:, 3])
    ax_txt.axis("off")

    legend_handles = [
        Line2D([0], [0], color=COL_PLOT1, lw=1.6, ls="-", label="Target mask (pred 2|4)"),
        Line2D([0], [0], color=COL_PLOT2, lw=1.2, ls="--", label="T-cell mask (pred 1|3)"),
        Line2D([0], [0], color=COL_DANGER, lw=1.6, ls="-", label="Dead-dye mask (>404 ADU)"),
    ]
    ax_txt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.0, bbox_to_anchor=(0.0, 1.0))

    v_mid = loaded[1]
    lines_txt = [
        f"Subcategory: {subcat}",
        f"Selection: {sel_tag}",
        f"Chamber sustained call: {'KILLED' if ever_dead_200 else 'SURVIVED'}",
        f"Sustained death_visit: {int(dv_sust) if pd.notna(dv_sust) else 'None'}",
        f"Any-visit pos: {ever_any_200} (first={int(dv_any) if pd.notna(dv_any) else 'None'})",
        "--- Rule (src/death_labels.py) ---",
        "Gaussian sigma=3.0, thr > 404.0 ADU,",
        "disk(3) close, area >= 20 px;",
        "dead_visit = (ov_xy>=20) & (ov_xz>=20)",
        "--- Shown visits (ov_xy / ov_xz px) ---",
        f"1. v{loaded[0]['visit']}: {loaded[0]['ov_xy']}/{loaded[0]['ov_xz']} px, G={loaded[0]['max_g_xy']:.1f}/{loaded[0]['max_g_xz']:.1f}",
        f"2. v{loaded[1]['visit']}: {loaded[1]['ov_xy']}/{loaded[1]['ov_xz']} px, G={loaded[1]['max_g_xy']:.1f}/{loaded[1]['max_g_xz']:.1f}",
        f"3. v{loaded[2]['visit']}: {loaded[2]['ov_xy']}/{loaded[2]['ov_xz']} px, G={loaded[2]['max_g_xy']:.1f}/{loaded[2]['max_g_xz']:.1f}",
    ]
    ax_txt.text(
        0.01,
        0.74,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=7.9,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.35,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.45"},
    )

    out_path = REVIEW_DIR / case_name / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    return {
        "case": case_name,
        "id": ex_id,
        "experiment": exp,
        "roi": roi,
        "visit": v_center,
        "sweep": 1,
        "condition": cond,
        "selection_tag": f"{subcat}_{sel_tag}",
        "source_data_path": ";".join(src_paths[:3]),
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "has_local_raw_h5": has_any_raw,
        "voxel_size_um": VOXEL_UM,
        "death_subcategory": subcat,
        "ever_dead_200m": ever_dead_200,
        "death_visit_200m": int(dv_sust) if pd.notna(dv_sust) else "",
        "visits_shown": f"{loaded[0]['visit']};{loaded[1]['visit']};{loaded[2]['visit']}",
        "center_visit_dead_call": bool(v_mid["dead_visit"]),
        "center_overlap_xy_px": int(v_mid["ov_xy"]),
        "center_overlap_xz_px": int(v_mid["ov_xz"]),
        "center_max_gauss405_xy": round(float(v_mid["max_g_xy"]), 2),
        "center_max_gauss405_xz": round(float(v_mid["max_g_xz"]), 2),
    }


# ==============================================================================
# SELECTION & DRIVER PIPELINE (SEED = 20261028, STRATIFIED + 3 EXTREMES PER CASE)
# ==============================================================================

def build_rep_sweep_lookup(
    experiments: list[str],
    man_lookup: dict[tuple[str, str, int, int], str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """
    Build lookup `(experiment, roi, visit) -> sweep_info` using the median-contact-area
    representative sweep per visit (matching `src/mech/run_step6b.py` and `src/mech/run_gateB_check.py`).
    """
    rep_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    for exp_name in sorted(set(experiments)):
        sw_csv = STEP3_TRACKED_DIR / exp_name / "per_sweep.csv"
        if not sw_csv.is_file():
            continue
        df_sw = pd.read_csv(sw_csv)
        for (e_k, r_k), g_sw in df_sw.groupby(["experiment", "roi"], sort=False):
            for v_num, gv in g_sw.groupby("visit", sort=False):
                gv_c = gv[gv["s_largest"] > 0]
                if len(gv_c) > 0:
                    med_s = float(gv_c["s_largest"].median())
                    idx_best = (gv_c["s_largest"] - med_s).abs().idxmin()
                    best_r = gv_c.loc[idx_best]
                else:
                    best_r = gv.iloc[len(gv) // 2]
                sw_num = int(best_r["sweep"])
                rel_p = man_lookup.get((str(e_k), str(r_k), int(v_num), sw_num), "")
                if rel_p:
                    rep_map[(str(e_k), str(r_k), int(v_num))] = {
                        "sweep": sw_num,
                        "rel_path": rel_p,
                        "tracked_tcell_id": int(best_r["tracked_tcell_id"]),
                        "n_tcell_instances": int(best_r["n_tcell_instances"]),
                        "n_target_instances": int(best_r["n_target_instances"]),
                        "s_largest": int(best_r["s_largest"]),
                    }
    return rep_map


def select_stratified_with_extremes(
    df_pool: pd.DataFrame,
    n_total: int,
    metric_col: str,
    strata_col: str,
    rng: np.random.Generator,
    extreme_modes: tuple[str, str, str] = ("max", "min", "second_max"),
    prefer_raw_Count: int = 2,
) -> list[tuple[pd.Series, str]]:
    """
    Select `n_total` distinct rows from `df_pool`:
      1. First include the 3 most extreme rows on `metric_col` (`extreme_modes`: e.g. highest, lowest, 2nd highest)
         so outliers can be visually inspected.
      2. Ensure local raw `.h5` examples (`has_local_raw == True`) are represented if available in `df_pool`.
      3. Fill the remaining slots by stratified random sampling across `strata_col` using `rng`.
    """
    sort_cols = [metric_col, "experiment", "roi"] + (["visit"] if "visit" in df_pool.columns else [])
    sort_asc = [False, True, True] + ([True] if "visit" in df_pool.columns else [])
    df_clean = df_pool.dropna(subset=[metric_col]).sort_values(
        sort_cols, ascending=sort_asc
    ).reset_index(drop=True)
    if len(df_clean) <= n_total:
        return [(df_clean.iloc[i], "all_available") for i in range(len(df_clean))]

    chosen_indices: list[int] = []
    chosen_tags: list[str] = []

    # 1. Pick 3 extreme outliers
    for mode in extreme_modes:
        if mode == "max":
            cand_order = list(range(len(df_clean)))
            tag = f"extreme_max_{metric_col}"
        elif mode == "second_max":
            cand_order = list(range(len(df_clean)))
            tag = f"extreme_2nd_max_{metric_col}"
        elif mode == "min":
            cand_order = list(range(len(df_clean) - 1, -1, -1))
            tag = f"extreme_min_{metric_col}"
        else:
            cand_order = list(range(len(df_clean)))
            tag = f"extreme_{mode}_{metric_col}"
        for idx_c in cand_order:
            if idx_c not in chosen_indices:
                chosen_indices.append(idx_c)
                chosen_tags.append(tag)
                break

    # 2. Include up to prefer_raw_Count random examples with local raw .h5 if present
    if "has_local_raw" in df_clean.columns and prefer_raw_Count > 0:
        raw_pool = [i for i in range(len(df_clean)) if bool(df_clean.loc[i, "has_local_raw"]) and i not in chosen_indices]
        if len(raw_pool) > 0:
            n_pick_r = min(prefer_raw_Count, len(raw_pool), max(0, n_total - len(chosen_indices)))
            picks_r = rng.choice(raw_pool, size=n_pick_r, replace=False)
            for idx_r in picks_r:
                chosen_indices.append(int(idx_r))
                chosen_tags.append("stratified_random_with_raw_h5")

    # 3. Fill remaining slots stratified by strata_col
    n_rem = n_total - len(chosen_indices)
    if n_rem > 0:
        rem_df = df_clean.drop(index=chosen_indices)
        strata_vals = sorted(rem_df[strata_col].astype(str).unique())
        n_strata = max(1, len(strata_vals))
        base_k = n_rem // n_strata
        extra_k = n_rem % n_strata

        for s_idx, s_val in enumerate(strata_vals):
            k_s = base_k + (1 if s_idx < extra_k else 0)
            if k_s <= 0:
                continue
            sub_indices = rem_df[rem_df[strata_col].astype(str) == s_val].index.to_numpy()
            if len(sub_indices) == 0:
                continue
            # Prefer distinct chambers when possible
            picks_s = rng.choice(sub_indices, size=min(k_s, len(sub_indices)), replace=False)
            for p_i in picks_s:
                chosen_indices.append(int(p_i))
                chosen_tags.append(f"stratified_random ({s_val})")

    # If any slots still remain (if a stratum had fewer rows than quota), fill randomly from remaining
    while len(chosen_indices) < n_total:
        avail = [i for i in range(len(df_clean)) if i not in chosen_indices]
        if not avail:
            break
        p_i = int(rng.choice(avail))
        chosen_indices.append(p_i)
        chosen_tags.append("stratified_random")

    return [(df_clean.loc[idx], tag) for idx, tag in zip(chosen_indices[:n_total], chosen_tags[:n_total])]


def _worker_dispatch(task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]
    if kind == "cup":
        return render_cup_panel(task["item"])
    if kind == "fit":
        return render_fit_panel(task["item"])
    if kind == "overlap":
        return render_initial_overlap_panel(task["item"])
    if kind == "mesh_size":
        return render_mesh_size_panel(task["item"])
    if kind == "death":
        return render_death_labels_panel(task["item"])
    raise ValueError(f"Unknown task kind: {kind}")


def prepare_chamber_ref_masks_for_gateB(
    ch_keys: set[tuple[str, str]],
    m_t1: pd.DataFrame,
    rep_map: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[tuple[str, str], tuple[Any, Any]]:
    """
    Pre-load `pre_mask_info` and `first_t_mask_info` for each chamber in `ch_keys`
    matching `src/mech/run_gateB_check.py`.
    """
    ref_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    for exp, roi in sorted(ch_keys):
        ch_df = m_t1[(m_t1["experiment"] == exp) & (m_t1["roi"] == roi)].sort_values("visit").reset_index(drop=True)
        valid_c = ch_df[ch_df["is_pre_death_contact"]].sort_values("visit").reset_index(drop=True)
        if len(valid_c) == 0:
            continue
        first_contact_v = int(valid_c["visit"].iloc[0])

        pre_df = ch_df[
            (ch_df["visit"] < first_contact_v)
            & (~ch_df["in_contact_visit"])
            & (ch_df["target_volume_um3"].fillna(0.0) > 20.0)
        ].sort_values("visit")

        pre_mask_info = None
        if len(pre_df) > 0:
            for idx_p in range(len(pre_df) - 1, -1, -1):
                cand_v = int(pre_df.iloc[idx_p]["visit"])
                sw_p = rep_map.get((exp, roi, cand_v))
                if sw_p is not None:
                    h5_p = REPO_ROOT / sw_p["rel_path"]
                    if h5_p.is_file():
                        _, _, b_sel_p, off_b_p = _extract_tcell_and_target_masks_from_h5(
                            h5_p,
                            int(sw_p["tracked_tcell_id"]),
                            int(sw_p["n_tcell_instances"]),
                            int(sw_p["n_target_instances"]),
                        )
                        if b_sel_p is not None and np.any(b_sel_p):
                            pre_mask_info = (b_sel_p, off_b_p)
                            break

        first_t_mask_info = None
        first_sw = rep_map.get((exp, roi, first_contact_v))
        if first_sw is not None:
            h5_0 = REPO_ROOT / first_sw["rel_path"]
            if h5_0.is_file():
                t0, ot0, b0, ob0 = _extract_tcell_and_target_masks_from_h5(
                    h5_0,
                    int(first_sw["tracked_tcell_id"]),
                    int(first_sw["n_tcell_instances"]),
                    int(first_sw["n_target_instances"]),
                )
                if t0 is not None and b0 is not None:
                    first_t_mask_info = (t0, ot0, b0, ob0)

        ref_cache[(exp, roi)] = (pre_mask_info, first_t_mask_info)
    return ref_cache


def write_review_readme() -> None:
    readme_text = """# Visual Review Gallery (`figures/review/`)

All 128 panels in `figures/review/<case>/<id>.png` render **real microscopy / segmentation data** from the deposited 3D U-Net masks (`data/scidb_19235_masks/<experiment>/<roi>/<stem>_predictions.h5`), local 3D raw fluorescence volumes (`data/scidb_19235_raw/` and `data/scidb_19235_chambers/` wherever downloaded locally), and 2D 405-nm SYTOX death-dye maximum-intensity projections (`data/scidb_19235_mips/<experiment>/<roi>/<stem>_MIPxy_405.tif` and `*_MIPxz_405.tif`), overlaid with the actual outputs of our measurement and solver code.

## Voxel Pitch and Coordinate Conventions
- Lateral and nominal axial voxel size: `VOXEL_UM = 0.1962 um` (`z = 0.196 um`, from `src/constants.py`).
- Every 3D contact panel shows an **XY slice (`z = z_c`)** and an **XZ slice (`y = y_c`)** passing directly through the 3D centroid `(z_c, y_c, x_c)` of the segmented contact patch (`is_contact_on_b_surf`), with a `5 um` scale bar (`5.0 / 0.1962 = 25.48 px`).

## Per-Case Overlay Definitions & Code Functions

### 1. `cup_measurement` (16 panels)
- **Source data**: `data/step5/per_visit_descriptors.csv` and `data/scidb_19235_masks/<exp>/<roi>/<stem>_predictions.h5` (plus local raw 561-nm membrane + 488-nm actin `.h5` where present).
- **Overlays & functions**:
  - **Target mask outline (`#1F4E79` solid)** and **T-cell mask outline (`#C0561F` solid)**: 2D `0.5`-level contour of the solid connected cell masks extracted by `_extract_tcell_and_target_masks_from_h5` (`src/mech/run_step6b.py`) using `fast_instance_seg_with_hint` and `fill_solid_cell_mask` (`src/contact_descriptors.py`).
  - **Contact patch voxels (`#6B4C9A` squares)**: Target boundary voxels within the contact patch (`is_contact_on_b_surf` in `process_one_chamber_descriptors`, `src/contact_descriptors.py`).
  - **Target convex hull (`#2E7D5B` dashed / dotted)**: 2D slice convex hull (`scipy.spatial.ConvexHull`) and 3D target convex hull cross-section (`hull_b = ConvexHull(pts_b_surf_um)` in `compute_descriptors_for_z_pitch`, `src/contact_descriptors.py`, intersected with the slice plane via `intersect_mesh_with_plane`).
  - **Measured cup depth (`#B3261E` double-headed arrow)**: 95th-percentile inward distance `cup_depth_hull_p95_um` from the target's T-cell-facing 3D convex hull facets (`facing = (eqs[:, :3] @ n_syn) > 0.05`) to the contact patch surface (`compute_descriptors_for_z_pitch` in `src/contact_descriptors.py`).

### 2. `fit_reachable` (16 panels) & 3. `fit_unreachable` (16 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` (20 in-depth chambers, 207 pre-death contact visits).
- **Overlays & functions**:
  - Each visit is re-solved at its fitted dimensionless push `Pi = pi_fit_delta` (`fit_reachable`) or at the grid ceiling `Pi = 12.0` (`fit_unreachable`) using `_prepare_visit_meshes_at_edge` (`src/mech/run_gateB_check.py`) and `solve_two_shell_equilibrium` (`src/mech/shell_model.py`, `gtol = 5e-2 pN`, `maxiter = 450`, `alpha_syn = 2.0`, `chi_t = 1.0`, `subtract_prestress = True`).
  - **Real target & T-cell mask outlines (`#1F4E79`, `#C0561F` solid)**: Slices of the true 3D segmentation masks.
  - **Reference (unloaded) target outline (`#2E7D5B` dotted)**: Cross-section of `g_mesh.V0` (pre-contact target mesh if available, or cup-filled reference via `make_cup_filled_reference_mesh`, aligned by `_align_reference_target_to_visit` and overlap-resolved by `resolve_initial_contact_overlap`).
  - **Solved target mesh cross-section (`#6B4C9A` dashed)** and **Solved T-cell mesh cross-section (`#111111` dashed)**: Exact plane intersections (`intersect_mesh_with_plane`) of the equilibrium 3D meshes `sol["V_target"]` and `sol["V_tcell"]`.
  - **Depth-vs-Pi curve inset**: Plots the visit's 9-node Gate-B curve `d_delta(Pi) = d(Pi) - d(0)` and `d_raw(Pi)` across `Pi in [0, 0.6, 1.2, 2.4, 4, 6, 8, 10, 12]` with the observed cup depth horizontal line (`cup_depth_hull_p95_um`) and inverted `Pi` (`_invert_depth_curve_no_extrap` in `src/mech/run_gateB_check.py`).

### 4. `initial_overlap` (12 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` and corresponding 3D masks.
- **Overlays & functions**:
  - **Before push-apart (`#B3261E` dashed target, `#C0561F` dotted T cell)**: Reference target mesh `V_g_ref_v` aligned by `_align_reference_target_to_visit` (`src/mech/run_step6b.py`) and segmented T-cell mesh `V_t_v` extracted by `extract_cell_mesh_marching_cubes` (`src/mech/shell_model.py`), prior to overlap resolution (`min_sep_before_push_um` from `TwoShellSystem._contact_energy_and_grad`).
  - **After push-apart (`#1F4E79` solid target, `#2E7D5B` solid T cell)**: Meshes translated apart along `syn_dir` by `push_apart_um` via `resolve_initial_contact_overlap` (`src/mech/shell_model.py`, enforcing `min_sep >= 0.16 um` at `Pi = 0`).

### 5. `mesh_size` (6 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` (including the 3 benchmark visits in `data/step6b/gateB.json` plus 3 extreme/stratified visits).
- **Overlays & functions**:
  - Each visit is extracted and solved at both `target_edge_um = 1.0 um` (~250–350 vertices/shell) and `target_edge_um = 0.5 um` (~1,000–1,400 vertices/shell) via `extract_cell_mesh_marching_cubes`, `_prepare_visit_meshes_at_edge`, and `solve_two_shell_equilibrium` at the same `Pi`.
  - Overlays both `1.0 um` (`#1F4E79` target, `#C0561F` T cell) and `0.5 um` (`#2E7D5B` dashed target, `#6B4C9A` dotted T cell) solved mesh cross-sections on the real XY and XZ mask slices.

### 6. `death_labels` (30 panels: 12 killed + 12 surviving + 6 dasatinib, including 3 borderline cases)
- **Source data**: `data/step4/per_chamber_death_labels_all_v2.csv`, `data/step4/per_visit_death_labels_all_v2.csv`, `data/scidb_19235_mips/<exp>/<roi>/<stem>_MIPxy_405.tif`, `*_MIPxz_405.tif`, and `*_predictions.h5`.
- **Overlays & functions**:
  - Displays the 405-nm SYTOX death-dye MIP (`MIPxy_405` and `MIPxz_405`) at 3 visits (before, at, and after `death_visit_200m` for killed chambers; before, at peak 405-nm overlap/intensity, and after for surviving/dasatinib chambers; including the 3 borderline chambers closest to the 20-px overlap / 404.0-ADU threshold).
  - **Target projection outline (`#1F4E79` solid)**: `b_xy = ((pred == 2) | (pred == 4)).max(axis=0).T` and `b_xz = ((pred == 2) | (pred == 4)).max(axis=1).T` (`_process_one_chamber` in `src/death_labels.py`).
  - **T-cell projection outline (`#C0561F` dashed)**: `t_xy = ((pred == 1) | (pred == 3)).max(axis=0).T` and `t_xz = ((pred == 1) | (pred == 3)).max(axis=1).T`.
  - **Cleaned dead-dye mask outline (`#B3261E` solid)**: `compute_2d_dead_mask(mip, THRESH_2D_ADU=404.0, MIN_COMPONENT_AREA_2D=20)` in `src/death_labels.py` (2D Gaussian filter `sigma = 3.0`, threshold `> 404.0 ADU`, `disk(3)` morphological closing, 8-connected area opening `>= 20 px`). A visit is called `dead_visit = True` iff `overlap_xy_px >= 20` and `overlap_xz_px >= 20`; a chamber is called `ever_dead = True` iff `>= 2` consecutive visits are positive.

### 7. `dasatinib_vs_untreated` (16 panels: 8 dasatinib + 8 untreated from matched donors `hd2`, `hd10`, `hd5`)
- **Source data**: `data/step5/per_visit_descriptors.csv` across paired experiments (`v08_hd2` vs `v11_hd2_ds`, `v09_hd10` vs `v12_hd10ds`, `v10_hd5` vs `v13_hd5_ds`).
- **Overlays & functions**: Same real mask slices and `src/contact_descriptors.py` overlays as Case 1, highlighting `cup_depth_hull_p95_um`, `patch_equiv_radius_um`, and `cup_depth_over_radius` across matched donors.

### 8. `deep_vs_shallow` (16 panels: 8 top-decile + 8 bottom-decile of `cup_depth_over_radius`)
- **Source data**: Untreated pre-death contact visits (`visit <= 20`, `trace_num == 1`) from `data/step5/per_visit_descriptors.csv` merged with `ever_dead_200m` from `data/step4/per_chamber_death_labels_all_v2.csv`.
- **Overlays & functions**: Same real mask slices and `src/contact_descriptors.py` overlays as Case 1 for visits in the top decile (`>= p90 = 0.644`) and bottom decile (`<= p10 = 0.128`) of `cup_depth_over_radius`, stratified by target outcome (`Killed` vs `Survived`).
"""
    (REVIEW_DIR / "README.md").write_text(readme_text, encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading tables (step5, step4, step6b, manifest)...", flush=True)
    df_v_all = pd.read_csv(STEP5_VISIT_CSV, low_memory=False)
    df_d_ch = pd.read_csv(STEP4_CHAMBER_CSV)
    df_d_vis = pd.read_csv(STEP4_VISIT_CSV, low_memory=False)
    df_gb = pd.read_csv(GATEB_FITS_CSV)
    df_man = pd.read_csv(MANIFEST_CSV)

    man_lookup: dict[tuple[str, str, int, int], str] = {}
    for _, mr in df_man.iterrows():
        man_lookup[(str(mr["experiment"]), str(mr["roi"]), int(mr["visit_index"]), int(mr["sweep_index"]))] = str(mr["path"])

    all_exps = sorted(df_v_all["experiment"].astype(str).unique())
    rep_map = build_rep_sweep_lookup(all_exps, man_lookup)

    # Merge step5 per_visit with step4 chamber death metadata
    m = df_v_all.merge(
        df_d_ch[
            [
                "experiment",
                "roi",
                "condition",
                "ever_dead_200m",
                "ever_dead_any_visit_200m",
                "death_visit_200m",
                "first_any_dead_visit_200m",
            ]
        ],
        on=["experiment", "roi"],
        how="left",
    )
    m_t1 = m[(m["trace_num"] == 1) & (m["visit"] <= 20)].copy()
    dead_v = m_t1["first_any_dead_visit_200m"].fillna(999)
    dead_v = np.where(dead_v <= 0, 999, dead_v)
    m_t1["is_pre_death"] = m_t1["visit"] < dead_v
    m_t1["is_pre_death_contact"] = (
        m_t1["in_contact_visit"]
        & m_t1["is_pre_death"]
        & m_t1["cup_depth_hull_p95_um"].notna()
        & (m_t1["patch_equiv_radius_um"] > 0.20)
    )

    # Mark which (experiment, roi, visit) have a local raw .h5 volume
    def _check_raw_row(r: pd.Series) -> bool:
        sw = rep_map.get((str(r["experiment"]), str(r["roi"]), int(r["visit"])))
        if sw is None:
            return False
        stem = Path(sw["rel_path"]).name.replace("_predictions.h5", "")
        return find_local_raw_h5(str(r["experiment"]), str(r["roi"]), stem) is not None

    m_pre = m_t1[m_t1["is_pre_death_contact"]].copy().reset_index(drop=True)
    m_pre["has_local_raw"] = [_check_raw_row(m_pre.iloc[i]) for i in range(len(m_pre))]
    df_gb["has_local_raw"] = [_check_raw_row(df_gb.iloc[i]) for i in range(len(df_gb))]

    # Pre-load reference mask info for the 20 Gate-B chambers
    gb_ch_keys = {(str(r["experiment"]), str(r["roi"])) for _, r in df_gb.iterrows()}
    ref_cache = prepare_chamber_ref_masks_for_gateB(gb_ch_keys, m_t1, rep_map)

    # Lookup map from (experiment, roi, visit) -> m_t1 row
    vrow_lookup: dict[tuple[str, str, int], pd.Series] = {}
    for idx_r in range(len(m_t1)):
        rr = m_t1.iloc[idx_r]
        vrow_lookup[(str(rr["experiment"]), str(rr["roi"]), int(rr["visit"]))] = rr

    tasks: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # CASE 1: cup_measurement [16]
    # -------------------------------------------------------------------------
    c1_picks = select_stratified_with_extremes(
        m_pre,
        n_total=16,
        metric_col="cup_depth_hull_p95_um",
        strata_col="condition",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=3,
    )
    for idx_i, (r_pick, tag) in enumerate(c1_picks, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        tasks.append(
            {
                "kind": "cup",
                "item": {
                    "case": "cup_measurement",
                    "id": f"cup_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "cup_depth_hull_p95_um": float(r_pick["cup_depth_hull_p95_um"]),
                    "patch_equiv_radius_um": float(r_pick["patch_equiv_radius_um"]),
                    "cup_depth_over_radius": float(r_pick["cup_depth_over_radius"]),
                    "contact_area_um2": float(r_pick["contact_area_um2"]),
                    "target_volume_um3": float(r_pick["target_volume_um3"]),
                    "tcell_volume_um3": float(r_pick["tcell_volume_um3"]),
                    "killed": r_pick["ever_dead_200m"],
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 2: fit_reachable [16]
    # -------------------------------------------------------------------------
    gb_reach = df_gb[~df_gb["unreachable_delta"] & df_gb["pi_fit_delta"].notna()].copy().reset_index(drop=True)
    c2_picks = select_stratified_with_extremes(
        gb_reach,
        n_total=16,
        metric_col="pi_fit_delta",
        strata_col="condition",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=3,
    )
    for idx_i, (r_pick, tag) in enumerate(c2_picks, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        pre_m, first_m = ref_cache.get((exp, roi), (None, None))
        tasks.append(
            {
                "kind": "fit",
                "item": {
                    "case": "fit_reachable",
                    "id": f"fit_reach_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "vrow": vrow_lookup[(exp, roi, vis)],
                    "gb_row": r_pick,
                    "pre_mask_info": pre_m,
                    "first_t_mask_info": first_m,
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 3: fit_unreachable [16]
    # -------------------------------------------------------------------------
    gb_unreach = df_gb[df_gb["unreachable_delta"]].copy().reset_index(drop=True)
    gb_unreach["depth_deficit_um"] = gb_unreach["cup_depth_hull_p95_um"] - gb_unreach["max_depth_delta_um"]
    c3_picks = select_stratified_with_extremes(
        gb_unreach,
        n_total=16,
        metric_col="depth_deficit_um",
        strata_col="condition",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=2,
    )
    for idx_i, (r_pick, tag) in enumerate(c3_picks, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        pre_m, first_m = ref_cache.get((exp, roi), (None, None))
        tasks.append(
            {
                "kind": "fit",
                "item": {
                    "case": "fit_unreachable",
                    "id": f"fit_unreach_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "vrow": vrow_lookup[(exp, roi, vis)],
                    "gb_row": r_pick,
                    "pre_mask_info": pre_m,
                    "first_t_mask_info": first_m,
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 4: initial_overlap [12]
    # -------------------------------------------------------------------------
    gb_ov = df_gb[df_gb["push_apart_um"] > 0.01].copy().reset_index(drop=True)
    c4_picks = select_stratified_with_extremes(
        gb_ov,
        n_total=12,
        metric_col="push_apart_um",
        strata_col="condition",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=2,
    )
    for idx_i, (r_pick, tag) in enumerate(c4_picks, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        pre_m, first_m = ref_cache.get((exp, roi), (None, None))
        tasks.append(
            {
                "kind": "overlap",
                "item": {
                    "case": "initial_overlap",
                    "id": f"overlap_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "vrow": vrow_lookup[(exp, roi, vis)],
                    "gb_row": r_pick,
                    "pre_mask_info": pre_m,
                    "first_t_mask_info": first_m,
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 5: mesh_size [6]
    # Include the 3 Gate-B benchmark visits + 3 extreme/stratified visits
    # -------------------------------------------------------------------------
    fixed_mesh_visits = [
        ("v09_6_40_120_0-6_hd10", "MIP_ROI_38", 20, "gateB_benchmark_1"),
        ("v08_6_40_120_0-6_hd2", "MIP_ROI_24", 7, "gateB_benchmark_2"),
        ("v05_6_40_120_0-6_hd7", "MIP_ROI_8", 8, "gateB_benchmark_3"),
    ]
    c5_rows: list[tuple[pd.Series, str]] = []
    used_c5_keys: set[tuple[str, str, int]] = set()
    for e_m, r_m, v_m, t_m in fixed_mesh_visits:
        sub_m = gb_reach[(gb_reach["experiment"] == e_m) & (gb_reach["roi"] == r_m) & (gb_reach["visit"] == v_m)]
        if len(sub_m) > 0:
            c5_rows.append((sub_m.iloc[0], t_m))
            used_c5_keys.add((e_m, r_m, v_m))

    rem_gb_reach = gb_reach[
        ~gb_reach.apply(lambda r: (str(r["experiment"]), str(r["roi"]), int(r["visit"])) in used_c5_keys, axis=1)
    ].copy().reset_index(drop=True)
    extra_c5 = select_stratified_with_extremes(
        rem_gb_reach,
        n_total=6 - len(c5_rows),
        metric_col="cup_depth_hull_p95_um",
        strata_col="condition",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=1,
    )
    c5_rows.extend(extra_c5)

    for idx_i, (r_pick, tag) in enumerate(c5_rows[:6], start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        pre_m, first_m = ref_cache.get((exp, roi), (None, None))
        tasks.append(
            {
                "kind": "mesh_size",
                "item": {
                    "case": "mesh_size",
                    "id": f"mesh_size_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "vrow": vrow_lookup[(exp, roi, vis)],
                    "gb_row": r_pick,
                    "pre_mask_info": pre_m,
                    "first_t_mask_info": first_m,
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 6: death_labels [12 killed + 12 surviving + 6 dasatinib = 30]
    # Including 3 borderline cases (closest to the 20-px overlap / 404.0-ADU threshold)
    # -------------------------------------------------------------------------
    ch_pool = df_d_ch[(df_d_ch["trace_num"] == 1) & (~df_d_ch["no_contact_200m"]) & (df_d_ch["n_visits_200m"] >= 4)].copy()
    vis_200 = df_d_vis[(df_d_vis["trace_num"] == 1) & (df_d_vis["in_200min_window"])].copy()
    vis_200["min_ov_both"] = np.minimum(vis_200["overlap_xy_px"], vis_200["overlap_xz_px"])
    vis_200["min_g_both"] = np.minimum(vis_200["max_gauss405_in_target_xy"], vis_200["max_gauss405_in_target_xz"])

    # Compute per-chamber peak overlap and closeness to threshold (ov=20 px or gauss=404 ADU)
    peak_ov_map: dict[tuple[str, str], float] = {}
    peak_g_map: dict[tuple[str, str], float] = {}
    border_dist_map: dict[tuple[str, str], float] = {}
    peak_vis_map: dict[tuple[str, str], int] = {}

    for (e_k, r_k), gv in vis_200.groupby(["experiment", "roi"], sort=False):
        gv_s = gv.sort_values("visit")
        # Prefer interior visits 2..n-1 for triplet center
        gv_int = gv_s.iloc[1:-1] if len(gv_s) >= 3 else gv_s
        ov_arr = gv_int["min_ov_both"].to_numpy(dtype=float)
        g_arr = gv_int["min_g_both"].to_numpy(dtype=float)
        v_arr = gv_int["visit"].to_numpy(dtype=int)
        # Score combining overlap and smoothed 405 intensity
        score_arr = ov_arr + 0.05 * np.maximum(0.0, g_arr - 395.0)
        best_i = int(np.argmax(score_arr))
        peak_ov_map[(str(e_k), str(r_k))] = float(ov_arr[best_i])
        peak_g_map[(str(e_k), str(r_k))] = float(g_arr[best_i])
        peak_vis_map[(str(e_k), str(r_k))] = int(v_arr[best_i])
        # Distance to threshold (either min_ov == 20 px or min_g == 404 ADU)
        d_thr = float(np.min(np.abs(ov_arr - 20.0) + 0.5 * np.abs(g_arr - 404.0)))
        border_dist_map[(str(e_k), str(r_k))] = d_thr

    ch_pool["peak_min_ov"] = [peak_ov_map.get((str(r["experiment"]), str(r["roi"])), 0.0) for _, r in ch_pool.iterrows()]
    ch_pool["peak_min_g"] = [peak_g_map.get((str(r["experiment"]), str(r["roi"])), 398.0) for _, r in ch_pool.iterrows()]
    ch_pool["border_dist"] = [border_dist_map.get((str(r["experiment"]), str(r["roi"])), 999.0) for _, r in ch_pool.iterrows()]

    # Sub-pools:
    # 1) 12 killed (untreated, ever_dead_200m == True, death_visit_200m >= 2)
    pool_killed = ch_pool[
        (ch_pool["condition"] != "Dasatinib 50 nM CAR-T")
        & (ch_pool["ever_dead_200m"] == True)
        & (ch_pool["death_visit_200m"].fillna(0) >= 2)
    ].copy().reset_index(drop=True)
    # 2) 12 surviving (untreated, ever_dead_200m == False)
    pool_surv = ch_pool[
        (ch_pool["condition"] != "Dasatinib 50 nM CAR-T")
        & (ch_pool["ever_dead_200m"] == False)
    ].copy().reset_index(drop=True)
    # 3) 6 dasatinib (condition == Dasatinib 50 nM CAR-T)
    pool_dasa = ch_pool[ch_pool["condition"] == "Dasatinib 50 nM CAR-T"].copy().reset_index(drop=True)

    # Pick 1 borderline from each sub-pool (or 3 closest to threshold overall across killed/surviving/dasatinib)
    # Plus extreme cases (earliest/latest death or highest/lowest signal) + stratified random
    def _pick_death_subpool(
        df_sub: pd.DataFrame,
        n_target: int,
        subcat_label: str,
        metric_col: str,
    ) -> list[tuple[pd.Series, str, str]]:
        # Sort by border_dist ascending to pick 1 borderline case first
        idx_border = int(df_sub["border_dist"].idxmin())
        row_border = df_sub.loc[idx_border]
        rem = df_sub.drop(index=[idx_border]).reset_index(drop=True)
        rest_picks = select_stratified_with_extremes(
            rem,
            n_total=n_target - 1,
            metric_col=metric_col,
            strata_col="experiment",
            rng=rng,
            extreme_modes=("max", "min", "second_max"),
            prefer_raw_Count=0,
        )
        out_list = [(row_border, subcat_label, "borderline_closest_to_threshold")]
        for r_p, t_p in rest_picks:
            out_list.append((r_p, subcat_label, t_p))
        return out_list

    c6_all = (
        _pick_death_subpool(pool_killed, 12, "killed", "death_visit_200m")
        + _pick_death_subpool(pool_surv, 12, "surviving", "peak_min_g")
        + _pick_death_subpool(pool_dasa, 6, "dasatinib", "peak_min_g")
    )

    for idx_i, (ch_r, subcat, tag) in enumerate(c6_all, start=1):
        exp = str(ch_r["experiment"])
        roi = str(ch_r["roi"])
        gv = vis_200[(vis_200["experiment"] == exp) & (vis_200["roi"] == roi)].sort_values("visit").reset_index(drop=True)
        v_nums = gv["visit"].to_numpy(dtype=int)

        if bool(ch_r["ever_dead_200m"]) and pd.notna(ch_r["death_visit_200m"]):
            v_call = int(ch_r["death_visit_200m"])
            pos_idx = int(np.argmin(np.abs(v_nums - v_call)))
            role_names = ("Before death", "At death call", "After death")
        else:
            v_peak = peak_vis_map.get((exp, roi), int(v_nums[len(v_nums) // 2]))
            pos_idx = int(np.argmin(np.abs(v_nums - v_peak)))
            role_names = ("Before peak", "At peak 405nm", "After peak")

        # Ensure valid triplet indices (idx_prev, idx_mid, idx_next)
        pos_idx = int(np.clip(pos_idx, 1, max(1, len(gv) - 2)))
        idx_triplet = [max(0, pos_idx - 1), pos_idx, min(len(gv) - 1, pos_idx + 1)]
        three_v_list = []
        for r_lbl, i_t in zip(role_names, idx_triplet):
            vr = gv.iloc[i_t]
            three_v_list.append(
                {
                    "visit": int(vr["visit"]),
                    "stem": str(vr["stem"]).zfill(5),
                    "role_label": r_lbl,
                }
            )

        tasks.append(
            {
                "kind": "death",
                "item": {
                    "case": "death_labels",
                    "id": f"death_{subcat}_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "condition": str(ch_r["condition"]),
                    "subcategory": subcat,
                    "selection_tag": tag,
                    "three_visits": three_v_list,
                    "ch_row": ch_r,
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 7: dasatinib_vs_untreated [8 + 8 = 16]
    # Cups from the same donors (hd2, hd10, hd5), with depth, patch radius, and depth/radius shown
    # -------------------------------------------------------------------------
    untr_paired_exps = ["v08_6_40_120_0-6_hd2", "v09_6_40_120_0-6_hd10", "v10_6_40_120_0-6_hd5"]
    dasa_paired_exps = ["v11_6_36_120_0-6_hd2_ds", "v12_6_35_120_0-6_hd10ds", "v13_6_40_120_0-6_hd5_ds"]

    pool_c7_untr = m_pre[m_pre["experiment"].isin(untr_paired_exps)].copy().reset_index(drop=True)
    pool_c7_dasa = m_pre[m_pre["experiment"].isin(dasa_paired_exps)].copy().reset_index(drop=True)

    c7_untr_picks = select_stratified_with_extremes(
        pool_c7_untr,
        n_total=8,
        metric_col="cup_depth_over_radius",
        strata_col="experiment",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=0,
    )
    c7_dasa_picks = select_stratified_with_extremes(
        pool_c7_dasa,
        n_total=8,
        metric_col="cup_depth_over_radius",
        strata_col="experiment",
        rng=rng,
        extreme_modes=("max", "min", "second_max"),
        prefer_raw_Count=0,
    )

    c7_combined = [(r, f"untreated_{t}") for r, t in c7_untr_picks] + [(r, f"dasatinib_{t}") for r, t in c7_dasa_picks]
    for idx_i, (r_pick, tag) in enumerate(c7_combined, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        grp_lbl = "untreated" if exp in untr_paired_exps else "dasatinib"
        tasks.append(
            {
                "kind": "cup",
                "item": {
                    "case": "dasatinib_vs_untreated",
                    "id": f"dasa_vs_untr_{grp_lbl}_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "cup_depth_hull_p95_um": float(r_pick["cup_depth_hull_p95_um"]),
                    "patch_equiv_radius_um": float(r_pick["patch_equiv_radius_um"]),
                    "cup_depth_over_radius": float(r_pick["cup_depth_over_radius"]),
                    "contact_area_um2": float(r_pick["contact_area_um2"]),
                    "target_volume_um3": float(r_pick["target_volume_um3"]),
                    "tcell_volume_um3": float(r_pick["tcell_volume_um3"]),
                    "killed": r_pick["ever_dead_200m"],
                },
            }
        )

    # -------------------------------------------------------------------------
    # CASE 8: deep_vs_shallow [8 + 8 = 16]
    # Top and bottom deciles of cup_depth_over_radius among untreated pre-death visits, with killed/survived shown
    # -------------------------------------------------------------------------
    pool_untr = m_pre[m_pre["condition"] != "Dasatinib 50 nM CAR-T"].copy().reset_index(drop=True)
    q10_dr = float(np.percentile(pool_untr["cup_depth_over_radius"], 10.0))
    q90_dr = float(np.percentile(pool_untr["cup_depth_over_radius"], 90.0))

    pool_deep = pool_untr[pool_untr["cup_depth_over_radius"] >= q90_dr].copy().reset_index(drop=True)
    pool_shallow = pool_untr[pool_untr["cup_depth_over_radius"] <= q10_dr].copy().reset_index(drop=True)
    pool_deep["fate_str"] = np.where(pool_deep["ever_dead_200m"], "killed", "survived")
    pool_shallow["fate_str"] = np.where(pool_shallow["ever_dead_200m"], "killed", "survived")

    c8_deep_picks = select_stratified_with_extremes(
        pool_deep,
        n_total=8,
        metric_col="cup_depth_over_radius",
        strata_col="fate_str",
        rng=rng,
        extreme_modes=("max", "second_max", "min"),
        prefer_raw_Count=2,
    )
    c8_shallow_picks = select_stratified_with_extremes(
        pool_shallow,
        n_total=8,
        metric_col="cup_depth_over_radius",
        strata_col="fate_str",
        rng=rng,
        extreme_modes=("min", "max", "second_max"),
        prefer_raw_Count=2,
    )

    c8_combined = [(r, f"deep_top10pct_{t}") for r, t in c8_deep_picks] + [(r, f"shallow_bot10pct_{t}") for r, t in c8_shallow_picks]
    for idx_i, (r_pick, tag) in enumerate(c8_combined, start=1):
        exp, roi, vis = str(r_pick["experiment"]), str(r_pick["roi"]), int(r_pick["visit"])
        dec_lbl = "deep" if idx_i <= 8 else "shallow"
        tasks.append(
            {
                "kind": "cup",
                "item": {
                    "case": "deep_vs_shallow",
                    "id": f"deep_vs_shallow_{dec_lbl}_{idx_i:02d}",
                    "experiment": exp,
                    "roi": roi,
                    "visit": vis,
                    "condition": str(r_pick["condition"]),
                    "selection_tag": tag,
                    "sweep_info": rep_map[(exp, roi, vis)],
                    "cup_depth_hull_p95_um": float(r_pick["cup_depth_hull_p95_um"]),
                    "patch_equiv_radius_um": float(r_pick["patch_equiv_radius_um"]),
                    "cup_depth_over_radius": float(r_pick["cup_depth_over_radius"]),
                    "contact_area_um2": float(r_pick["contact_area_um2"]),
                    "target_volume_um3": float(r_pick["target_volume_um3"]),
                    "tcell_volume_um3": float(r_pick["tcell_volume_um3"]),
                    "killed": r_pick["ever_dead_200m"],
                },
            }
        )

    print(f"Prepared {len(tasks)} panel tasks across 8 cases. Rendering with <= {MAX_WORKERS} workers...", flush=True)

    index_rows: list[dict[str, Any]] = []
    done_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_to_idx = {pool.submit(_worker_dispatch, t): i for i, t in enumerate(tasks)}
        ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
        for fut in as_completed(fut_to_idx):
            i_task = fut_to_idx[fut]
            res = fut.result()
            ordered_results[i_task] = res
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                print(
                    f"  [{done_count:3d}/{len(tasks)}] rendered {res['case']}/{res['id']}.png "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )

    for r in ordered_results:
        if r is not None:
            index_rows.append(r)

    df_index = pd.DataFrame(index_rows)
    index_csv_path = REVIEW_DIR / "index.csv"
    df_index.to_csv(index_csv_path, index=False)
    write_review_readme()

    # Check PNG sizes
    png_sizes_kb = []
    for r in index_rows:
        p = REPO_ROOT / r["panel_png_path"]
        png_sizes_kb.append(p.stat().st_size / 1024.0)
    max_kb = max(png_sizes_kb) if png_sizes_kb else 0.0
    med_kb = float(np.median(png_sizes_kb)) if png_sizes_kb else 0.0

    elapsed = time.perf_counter() - t0
    print(
        f"Done! {len(index_rows)} panels written to {REVIEW_DIR} in {elapsed:.1f}s "
        f"(median PNG = {med_kb:.1f} KB, max PNG = {max_kb:.1f} KB).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
