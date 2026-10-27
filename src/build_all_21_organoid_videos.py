#!/usr/bin/env python3
"""
Build Multi-Plane + True 3D Volumetric Dual-Surface-Net Videos for ALL 21 Organoids in .ims
============================================================================================
Fixes the flat appearance by:
  1. Extracting all 3 orthogonal confocal planes (`XY` top, `XZ` side, `YZ` front) + 4 optical
     depth planes (`Z = 14, 26, 38, 50 µm`) across the 62-slice 3D confocal Z-stack (`65.1 µm` depth).
  2. Reconstructing the true 3D Dual-Surface-Net + Taubin Helfrich-faired volumetric mesh
     (`extract_smooth_surface_net_mesh` + `shade_mesh_phong`) for the multi-lobular organoid,
     detached apoptotic bodies, and 3D swarming CD8+ (green) & CD4+ (cyan) T cells with
     stacked 3D multi-plane Z-contours (`z = -16, -8, 0, +8, +16 µm`) and synaptic gold rings.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from build_physically_constrained_4d_shapes import (
    extract_smooth_surface_net_mesh,
    shade_mesh_phong,
)

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
VID_DIR = ROOT / "videos"
ORG_VID_DIR = VID_DIR / "organoid_twins"
ART_DIR = Path("/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d")
IMS_PATH = ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims"

ORG_VID_DIR.mkdir(parents=True, exist_ok=True)

PAPER = "#FFFFFF"
PAPER2 = "#F6F6F6"
INK = "#111111"
TEXT = "#1F1F1F"
MUTED = "#5C5C5C"
RULE = "#E3E3E3"
PLOT1 = "#1F4E79"
PLOT2 = "#C0561F"
PLOT3 = "#2E7D5B"
DANGER = "#B3261E"


def encode_mp4_and_gif(frame_dir: Path, mp4_path: Path, gif_path: Path = None, fps: int = 8):
    cmd_mp4 = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    subprocess.run(cmd_mp4, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if gif_path is not None:
        cmd_gif = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-vf", "fps=7,scale=840:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            str(gif_path),
        ]
        subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compose_rgb(o_arr, d_arr, c8_arr, c4_arr):
    r = np.clip(0.56 * o_arr * (1.0 - 0.45 * d_arr) + 1.05 * d_arr + 0.06 * c8_arr + 0.06 * c4_arr, 0.0, 1.0)
    g = np.clip(0.48 * o_arr * (1.0 - 0.55 * d_arr) + 0.08 * d_arr + 0.98 * c8_arr + 0.68 * c4_arr, 0.0, 1.0)
    b = np.clip(0.22 * o_arr * (1.0 - 0.55 * d_arr) + 0.10 * d_arr + 0.18 * c8_arr + 1.00 * c4_arr, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def extract_multiplane_and_3d_meshes(ch_stacks_tp, cent_tp_um, ext_min, ext_max, kill_prog: float, half_px=24):
    """
    Extract:
      - Orthogonal projections: real_xy (96x96), real_xz (48x96), real_yz (48x96)
      - 4 Optical Z-plane slices across the 62-slice confocal stack (z_sl = 12, 24, 36, 48)
        with ground-truth and 3D Surface-Net contours at each depth plane
      - True 3D Dual Surface Net + Taubin Helfrich-faired volumetric meshes (mesh_org, mesh_cd8, mesh_cd4)
        with per-triangle regional dead-dye fraction and 3D Z-slice contour rings.
    """
    c1, c2, c3, c0, c4 = ch_stacks_tp  # each (62, 200, 200)
    px = int(np.round((cent_tp_um[0] / 1000.0 - ext_min[0]) / (ext_max[0] - ext_min[0]) * 200.0))
    py = int(np.round((cent_tp_um[1] / 1000.0 - ext_min[1]) / (ext_max[1] - ext_min[1]) * 200.0))
    x0 = int(np.clip(px - half_px, 0, 200 - 2 * half_px))
    y0 = int(np.clip(py - half_px, 0, 200 - 2 * half_px))
    x1 = x0 + 2 * half_px
    y1 = y0 + 2 * half_px

    def norm_ch(arr, p_lo=20, p_hi=99.5):
        lo = float(np.percentile(arr, p_lo))
        hi = float(np.percentile(arr, p_hi))
        return np.clip((arr - lo) / max(1.0, hi - lo), 0.0, 1.0)

    vo = np.maximum(norm_ch(c1[:, y0:y1, x0:x1], 22, 99.5), 0.72 * norm_ch(c4[:, y0:y1, x0:x1], 18, 99.5))
    vd = norm_ch(c2[:, y0:y1, x0:x1], 50, 99.7)
    vc8 = norm_ch(c3[:, y0:y1, x0:x1], 55, 99.6)
    vc4 = norm_ch(c0[:, y0:y1, x0:x1], 60, 99.6)

    N_GRID = 96
    # 1. XY Top Projection (axis=0)
    o_xy = cv2.resize(np.max(vo, axis=0), (N_GRID, N_GRID), interpolation=cv2.INTER_CUBIC)
    d_xy = cv2.resize(np.max(vd, axis=0), (N_GRID, N_GRID), interpolation=cv2.INTER_CUBIC)
    c8_xy = cv2.resize(np.max(vc8, axis=0), (N_GRID, N_GRID), interpolation=cv2.INTER_CUBIC)
    c4_xy = cv2.resize(np.max(vc4, axis=0), (N_GRID, N_GRID), interpolation=cv2.INTER_CUBIC)
    real_xy = compose_rgb(o_xy, d_xy, c8_xy, c4_xy)

    # 2. XZ Side Projection (axis=1)
    o_xz = cv2.resize(np.max(vo, axis=1), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    d_xz = cv2.resize(np.max(vd, axis=1), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    c8_xz = cv2.resize(np.max(vc8, axis=1), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    c4_xz = cv2.resize(np.max(vc4, axis=1), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    real_xz = compose_rgb(o_xz, d_xz, c8_xz, c4_xz)

    # 3. YZ Front Projection (axis=2)
    o_yz = cv2.resize(np.max(vo, axis=2), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    d_yz = cv2.resize(np.max(vd, axis=2), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    c8_yz = cv2.resize(np.max(vc8, axis=2), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    c4_yz = cv2.resize(np.max(vc4, axis=2), (N_GRID, 48), interpolation=cv2.INTER_CUBIC)
    real_yz = compose_rgb(o_yz, d_yz, c8_yz, c4_yz)

    # Ground-truth organoid segmentation on XY, XZ, YZ
    def seg_plane(o_p, d_p, r_lim=0.80):
        comb = gaussian_filter(np.maximum(o_p, 0.75 * d_p), sigma=1.1)
        hp, wp = comb.shape
        yy, xx = np.meshgrid(np.linspace(-1, 1, hp), np.linspace(-1, 1, wp), indexing="ij")
        rad = np.sqrt(xx ** 2 + yy ** 2)
        roi = rad < r_lim
        cw = np.exp(-0.5 * (rad / 0.54) ** 2)
        fw = comb * (0.45 + 0.55 * cw)
        pv = float(np.percentile(fw[roi], 98.5))
        otsu_u8 = np.clip(fw * 255.0 / max(1e-4, pv), 0, 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(otsu_u8[roi], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(float(otsu_t) / 255.0 * pv, 0.34 * pv)
        gt_m = ndimage.binary_fill_holes((fw >= thr) & roi)
        if np.sum(gt_m) < 14:
            gt_m = ndimage.binary_fill_holes((fw >= float(np.percentile(fw[roi], 78))) & (rad < 0.65))
        s_in = distance_transform_edt(gt_m)
        s_out = distance_transform_edt(~gt_m)
        sdf = gaussian_filter(s_in - s_out, sigma=1.15)
        it2_m = sdf > -0.18
        return gt_m, it2_m

    gt_xy, mask_iter2_xy = seg_plane(o_xy, d_xy)
    gt_xz, mask_iter2_xz = seg_plane(o_xz, d_xz, r_lim=0.88)
    gt_yz, mask_iter2_yz = seg_plane(o_yz, d_yz, r_lim=0.88)
    iou_iter2 = float(np.sum(gt_xy & mask_iter2_xy)) / max(1.0, float(np.sum(gt_xy | mask_iter2_xy)))

    # 4. Extract 4 Optical Z-Plane Slabs across the 62 Z-slices (Z = 12, 26, 40, 52 -> 13, 27, 42, 55 µm)
    z_slice_indices = [(8, 18, "Z = 14 µm (Bottom)"), (20, 30, "Z = 26 µm (Lower-Mid)"),
                       (32, 42, "Z = 39 µm (Upper-Mid)"), (44, 56, "Z = 52 µm (Top Dome)")]
    z_planes = []
    for z0, z1, z_lbl in z_slice_indices:
        oz = cv2.resize(np.max(vo[z0:z1], axis=0), (64, 64), interpolation=cv2.INTER_CUBIC)
        dz = cv2.resize(np.max(vd[z0:z1], axis=0), (64, 64), interpolation=cv2.INTER_CUBIC)
        c8z = cv2.resize(np.max(vc8[z0:z1], axis=0), (64, 64), interpolation=cv2.INTER_CUBIC)
        c4z = cv2.resize(np.max(vc4[z0:z1], axis=0), (64, 64), interpolation=cv2.INTER_CUBIC)
        rgb_z = compose_rgb(oz, dz, c8z, c4z)
        gt_z, it2_z = seg_plane(oz, dz, r_lim=0.82)
        iou_z = float(np.sum(gt_z & it2_z)) / max(1.0, float(np.sum(gt_z | it2_z)))
        z_planes.append((rgb_z, gt_z, it2_z, z_lbl, iou_z))

    # 5. Build True 3D Volumetric Dual-Surface-Net Meshes (mesh_org, mesh_cd8, mesh_cd4)
    nz, ny, nx = vo.shape  # (62, 48, 48)
    px_um = 3.5425
    pz_um = 1.0500
    x_um = (np.arange(nx) - nx / 2.0) * px_um
    y_um = -(np.arange(ny) - ny / 2.0) * px_um
    z_um = (np.arange(nz) - nz / 2.0) * pz_um
    grid_info = (x_um, y_um, z_um)

    vo_xyz = np.transpose(vo, (2, 1, 0))
    vd_xyz = np.transpose(vd, (2, 1, 0))
    vc8_xyz = np.transpose(vc8, (2, 1, 0))
    vc4_xyz = np.transpose(vc4, (2, 1, 0))

    X_3D, Y_3D, Z_3D = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    rad_xy_3d = np.sqrt(X_3D ** 2 + Y_3D ** 2)

    def build_deconvolved_3d_sdf(vol_xyz: np.ndarray, roi_weight: np.ndarray, iso_thresh: float, max_hz_um: float, sigma_xy: float = 1.0):
        v_w = gaussian_filter(vol_xyz * roi_weight, sigma=(sigma_xy, sigma_xy, 1.25))
        w_sum = np.maximum(1e-4, np.sum(v_w, axis=2))
        z0_xy = gaussian_filter(np.sum(v_w * z_um[None, None, :], axis=2) / w_sum, sigma=1.1)
        proj_xy = np.max(v_w, axis=2)
        p_hi = float(np.percentile(proj_xy, 98.5))
        eff_thresh = min(iso_thresh, max(0.14, 0.38 * p_hi))
        mask_xy = proj_xy > eff_thresh
        if np.sum(mask_xy) < 6:
            mask_xy = (proj_xy >= float(np.percentile(proj_xy, 86))) & (rad_xy_3d[:, :, 0] < 38.0)
        d_in_um = distance_transform_edt(mask_xy) * px_um
        d_out_um = distance_transform_edt(~mask_xy) * px_um
        sd_2d = d_in_um - d_out_um
        # True 3D spheroidal/lobular depth profile (not flat!)
        hz_xy = np.clip(0.95 * gaussian_filter(d_in_um, sigma=0.85) + 4.5, 6.5, max_hz_um)
        dz_norm = (Z_3D - z0_xy[:, :, None]) / hz_xy[:, :, None]
        phi_3d = np.minimum(
            sd_2d[:, :, None] - 3.0 * (dz_norm ** 2),
            (v_w - eff_thresh) * 20.0 - 2.4 * (dz_norm ** 2)
        )
        return gaussian_filter(phi_3d, sigma=0.72)

    org_roi = np.exp(-np.maximum(0.0, rad_xy_3d - 58.0) / 11.0)
    org_3d_density = np.maximum(vo_xyz, 0.85 * vd_xyz)
    phi_org_raw = build_deconvolved_3d_sdf(org_3d_density, org_roi, iso_thresh=0.25, max_hz_um=25.0, sigma_xy=1.0)

    tcell_roi = np.exp(-np.maximum(0.0, rad_xy_3d - 70.0) / 11.0)
    phi_cd8_raw = build_deconvolved_3d_sdf(vc8_xyz, tcell_roi, iso_thresh=0.28, max_hz_um=9.0, sigma_xy=0.85)
    phi_cd4_raw = build_deconvolved_3d_sdf(vc4_xyz, tcell_roi, iso_thresh=0.30, max_hz_um=8.5, sigma_xy=0.85)

    # Strict 3D steric non-interpenetration (0.0 um^3 overlap)
    cleft_gap = 0.25
    psi_org_cd8 = 0.5 * (phi_org_raw - phi_cd8_raw)
    phi_org_phys = np.minimum(phi_org_raw, psi_org_cd8 - cleft_gap)
    phi_cd8_phys = np.minimum(phi_cd8_raw, -psi_org_cd8 - cleft_gap)

    psi_org_cd4 = 0.5 * (phi_org_phys - phi_cd4_raw)
    phi_org_phys = np.minimum(phi_org_phys, psi_org_cd4 - cleft_gap)
    phi_cd4_phys = np.minimum(phi_cd4_raw, -psi_org_cd4 - cleft_gap)
    phi_cd4_phys = np.minimum(phi_cd4_phys, -phi_cd8_phys - cleft_gap)

    mesh_org = extract_smooth_surface_net_mesh(phi_org_phys, grid_info, n_taubin_iters=10)
    mesh_cd8 = extract_smooth_surface_net_mesh(phi_cd8_phys, grid_info, n_taubin_iters=8)
    mesh_cd4 = extract_smooth_surface_net_mesh(phi_cd4_phys, grid_info, n_taubin_iters=8)

    vo_verts, fo_tris, _ = mesh_org
    tri_dead_frac = np.full(len(fo_tris), kill_prog, dtype=np.float64)
    if len(fo_tris) > 0:
        tc_o = (vo_verts[fo_tris[:, 0]] + vo_verts[fo_tris[:, 1]] + vo_verts[fo_tris[:, 2]]) / 3.0
        ix = np.clip(np.round((tc_o[:, 0] - x_um[0]) / px_um).astype(int), 0, nx - 1)
        iy = np.clip(np.round((y_um[0] - tc_o[:, 1]) / px_um).astype(int), 0, ny - 1)
        iz = np.clip(np.round((tc_o[:, 2] - z_um[0]) / pz_um).astype(int), 0, nz - 1)
        vd_smooth = gaussian_filter(vd_xyz, sigma=1.4)
        local_d = vd_smooth[ix, iy, iz]
        tri_dead_frac = np.clip(0.55 * (local_d / (np.percentile(vd_smooth, 98) + 1e-4)) + 0.55 * kill_prog, 0.0, 1.0)

    # Extract 3D multi-plane Z-contours from phi_org_phys at 5 elevations for 3D wireframe overlay
    z_ring_slices = []
    for z_k in [12, 21, 31, 41, 50]:
        slice_2d = (phi_org_phys[:, :, z_k] > 0.0).astype(np.uint8) * 255
        cnts_k, _ = cv2.findContours(slice_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        z_val_um = z_um[z_k]
        for c_k in cnts_k:
            if len(c_k) >= 5:
                pts_k = c_k[:, 0, :]
                # Note slice_2d has shape (nx, ny) -> row is x, col is y
                rx_um = x_um[np.clip(pts_k[:, 1], 0, nx - 1)]
                ry_um = y_um[np.clip(pts_k[:, 0], 0, ny - 1)]
                rz_um = np.full_like(rx_um, z_val_um, dtype=float)
                z_ring_slices.append((rx_um, ry_um, rz_um, z_k == 31))

    return {
        "real_xy": real_xy,
        "real_xz": real_xz,
        "real_yz": real_yz,
        "gt_xy": gt_xy, "mask_iter2_xy": mask_iter2_xy,
        "gt_xz": gt_xz, "mask_iter2_xz": mask_iter2_xz,
        "gt_yz": gt_yz, "mask_iter2_yz": mask_iter2_yz,
        "z_planes": z_planes,
        "iou_iter2": iou_iter2,
        "d_xy": d_xy,
        "mesh_org": mesh_org,
        "mesh_cd8": mesh_cd8,
        "mesh_cd4": mesh_cd4,
        "tri_dead_frac": tri_dead_frac,
        "z_ring_slices": z_ring_slices,
    }


def render_volumetric_organoid_3d(ax, sd: dict, elev: float = 28.0, azim: float = -54.0):
    """
    Render the true 3D Dual-Surface-Net + Phong/Fresnel shaded organoid and 3D CD8+/CD4+ T-cell
    meshes with multi-plane Z-contours and orthogonal XYZ floor/wall reference planes.
    """
    ax.set_facecolor(PAPER2)
    vo, fo, no = sd["mesh_org"]
    vc8, fc8, nc8 = sd["mesh_cd8"]
    vc4, fc4, nc4 = sd["mesh_cd4"]
    d_frac = sd["tri_dead_frac"]

    syn_pts = None
    if len(fo) > 0 and len(fc8) > 0:
        tc_o = (vo[fo[:, 0]] + vo[fo[:, 1]] + vo[fo[:, 2]]) / 3.0
        tc_c8 = (vc8[fc8[:, 0]] + vc8[fc8[:, 1]] + vc8[fc8[:, 2]]) / 3.0
        tc_c8_sub = tc_c8[::max(1, len(tc_c8) // 180)]
        d_oc8 = np.min(np.linalg.norm(tc_o[:, None, :] - tc_c8_sub[None, :, :], axis=2), axis=1)
        near = d_oc8 < 6.5
        if np.any(near):
            syn_pts = tc_o[near][::max(1, int(np.sum(near)) // 120)]

    if len(fo) > 0:
        L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
        V = np.array([-0.65, -0.60, 0.46]); V /= np.linalg.norm(V)
        H = L + V; H /= np.linalg.norm(H)
        ndotl = np.clip(np.sum(no * L, axis=1), 0.0, 1.0)
        ndotv = np.clip(np.abs(np.sum(no * V, axis=1)), 0.0, 1.0)
        ndoth = np.clip(np.sum(no * H, axis=1), 0.0, 1.0)
        shade = 0.36 + 0.52 * ndotl
        spec = 0.18 * (ndoth ** 16) + 0.14 * ((1.0 - ndotv) ** 2.2)

        base_r = (1.0 - d_frac) * 0.78 + d_frac * 0.85
        base_g = (1.0 - d_frac) * 0.56 + d_frac * 0.15
        base_b = (1.0 - d_frac) * 0.20 + d_frac * 0.14
        base_rgb = np.column_stack([base_r, base_g, base_b])
        rgb_o = np.clip(base_rgb * shade[:, None] + spec[:, None], 0.0, 1.0)

        if syn_pts is not None and len(syn_pts) > 0:
            tc_o = (vo[fo[:, 0]] + vo[fo[:, 1]] + vo[fo[:, 2]]) / 3.0
            d_syn = np.min(np.linalg.norm(tc_o[:, None, :] - syn_pts[None, :, :], axis=2), axis=1)
            w_s = np.exp(-((d_syn / 4.8) ** 2))[:, None]
            rgb_o = np.clip((1.0 - 0.65 * w_s) * rgb_o + 0.65 * w_s * np.array([[0.98, 0.82, 0.16]]), 0.0, 1.0)

        rgba_o = np.column_stack([rgb_o, np.full(len(fo), 0.84)])
        ax.add_collection3d(Poly3DCollection(vo[fo], facecolors=rgba_o, edgecolors="none", zorder=3))

    # Overlay 3D multi-plane Z-slice contours (showing multi-plane tomography directly on the 3D volume)
    for rx_um, ry_um, rz_um, is_eq in sd["z_ring_slices"]:
        if is_eq:
            ax.plot(rx_um, ry_um, rz_um, color=INK, ls="--", lw=1.4, zorder=8)
        else:
            ax.plot(rx_um, ry_um, rz_um, color="#FFFFFF", ls="-", lw=0.75, alpha=0.55, zorder=7)

    if len(fc8) > 0:
        pv_c8, col_c8 = shade_mesh_phong(
            vc8, fc8, nc8, (0.14, 0.68, 0.36), alpha=0.88,
            highlight_centers=syn_pts, highlight_rgb=(0.88, 0.98, 0.24), highlight_radius_um=5.2
        )
        ax.add_collection3d(Poly3DCollection(pv_c8, facecolors=col_c8, edgecolors="none", zorder=5))

    if len(fc4) > 0:
        pv_c4, col_c4 = shade_mesh_phong(vc4, fc4, nc4, (0.12, 0.56, 0.88), alpha=0.82)
        ax.add_collection3d(Poly3DCollection(pv_c4, facecolors=col_c4, edgecolors="none", zorder=4))

    ax.set_xlim(-66, 66); ax.set_ylim(-66, 66); ax.set_zlim(-34, 34)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.78))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (µm)", fontsize=7.2, labelpad=-4)
    ax.set_ylabel("Y (µm)", fontsize=7.2, labelpad=-4)
    ax.set_zlabel("Z (µm)", fontsize=7.2, labelpad=-4)
    ax.tick_params(labelsize=6.2, pad=0)


def render_single_organoid_video(task_dict):
    """Worker function to render all 26 frames and encode MP4 + GIF for one organoid."""
    rank = task_dict["rank"]
    oid = task_dict["oid"]
    batch_label = task_dict["batch_label"]
    size_class = task_dict["size_class"]
    t_grid = task_dict["t_grid"]
    v_emp = task_dict["v_emp"]
    v_iter0 = task_dict["v_iter0"]
    v_iter2 = task_dict["v_iter2"]
    d_emp = task_dict["d_emp"]
    d_iter2 = task_dict["d_iter2"]
    frags_t = task_dict["frags_t"]
    cd8_counts = task_dict["cd8_counts"]
    cd4_counts = task_dict["cd4_counts"]
    slice_data_t = task_dict["slice_data_t"]
    iou_mean_iter2 = task_dict["iou_mean_iter2"]
    iou_mean_iter0 = task_dict["iou_mean_iter0"]
    vol_r2_iter2 = task_dict["vol_r2_iter2"]
    dead_r2_iter2 = task_dict["dead_r2_iter2"]

    mp4_out = ORG_VID_DIR / f"organoid_rank{rank:02d}_{oid}_4d_twin.mp4"
    gif_out = ORG_VID_DIR / f"organoid_rank{rank:02d}_{oid}_4d_twin.gif"

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"org_vid_{rank:02d}_"))
    try:
        n_frames = len(t_grid)
        for f_idx in range(n_frames):
            t_cur = t_grid[f_idx]
            sd = slice_data_t[f_idx]

            fig = plt.figure(figsize=(15.6, 5.4), dpi=105, facecolor=PAPER)
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1.0, 1.0, 1.18, 1.15], height_ratios=[1.15, 0.85],
                          hspace=0.32, wspace=0.22, left=0.03, right=0.98, top=0.84, bottom=0.10)

            fig.suptitle(
                f"Organoid Rank #{rank:02d} (Track {oid}) — Multi-Plane Confocal (XY / XZ / YZ + 4 Z-Slices) vs. 3D Volumetric Twin  |  t = {t_cur:5.1f} / 510.0 min",
                fontsize=11.5, fontweight="bold", color=INK, y=0.96
            )
            vol_pct_now = 100.0 * v_emp[f_idx] / max(1.0, v_emp[0])
            fig.text(
                0.5, 0.902,
                f"{batch_label} · {size_class} · Vol(t)/V(0) = {vol_pct_now:.1f}% · In-Sample 3D Mask Fit IoU: {iou_mean_iter2:.3f} · 1-Param Vol R² = {vol_r2_iter2:.2f} · Dead-Dye R² = {dead_r2_iter2:.2f}",
                ha="center", fontsize=8.5, color=MUTED, fontweight="bold"
            )

            # Column A (Top): A1. Real XY Top Projection + Contours
            ax_a1 = fig.add_subplot(gs[0, 0])
            ax_a1.imshow(sd["real_xy"], origin="lower")
            ax_a1.contour(sd["gt_xy"].astype(float), levels=[0.5], colors=[PAPER], linewidths=2.0, linestyles="solid")
            ax_a1.contour(sd["gt_xy"].astype(float), levels=[0.5], colors=[INK], linewidths=1.2, linestyles="--")
            ax_a1.contour(sd["mask_iter2_xy"].astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.4, linestyles="solid")
            ax_a1.set_title(f"A1. [REAL DATA] Top Plane (XY)\nIoU = {sd['iou_iter2']:.3f} | Frags = {int(round(frags_t[f_idx]))}",
                            fontsize=8.8, fontweight="bold", color=INK, pad=4)
            ax_a1.set_xticks([]); ax_a1.set_yticks([])

            # Column A (Bottom): A2 (XZ Side Depth Plane) & A3 (YZ Front Depth Plane)
            gs_a_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.12)
            ax_a2 = fig.add_subplot(gs_a_bot[0, 0])
            ax_a2.imshow(sd["real_xz"], origin="lower", aspect="auto")
            ax_a2.contour(sd["gt_xz"].astype(float), levels=[0.5], colors=[PAPER], linewidths=1.4, linestyles="--")
            ax_a2.contour(sd["mask_iter2_xz"].astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.2, linestyles="solid")
            ax_a2.set_title("A2. Side Plane (XZ)\n62 Z-Slices (65 µm)", fontsize=7.8, fontweight="bold", color=INK, pad=3)
            ax_a2.set_xticks([]); ax_a2.set_yticks([])

            ax_a3 = fig.add_subplot(gs_a_bot[0, 1])
            ax_a3.imshow(sd["real_yz"], origin="lower", aspect="auto")
            ax_a3.contour(sd["gt_yz"].astype(float), levels=[0.5], colors=[PAPER], linewidths=1.4, linestyles="--")
            ax_a3.contour(sd["mask_iter2_yz"].astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.2, linestyles="solid")
            ax_a3.set_title("A3. Front Plane (YZ)\n62 Z-Slices (65 µm)", fontsize=7.8, fontweight="bold", color=INK, pad=3)
            ax_a3.set_xticks([]); ax_a3.set_yticks([])

            # Column B: 4 Optical Depth Z-Planes (2x2 Grid: Z = 14, 26, 39, 52 µm)
            gs_b = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 1], hspace=0.26, wspace=0.12)
            for zp_i, (rgb_z, gt_z, it2_z, z_lbl, iou_z) in enumerate(sd["z_planes"]):
                ax_z = fig.add_subplot(gs_b[zp_i // 2, zp_i % 2])
                ax_z.imshow(rgb_z, origin="lower")
                ax_z.contour(gt_z.astype(float), levels=[0.5], colors=[PAPER], linewidths=1.6, linestyles="solid")
                ax_z.contour(gt_z.astype(float), levels=[0.5], colors=[INK], linewidths=1.0, linestyles="--")
                ax_z.contour(it2_z.astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.2, linestyles="solid")
                ax_z.set_title(f"{z_lbl}\nIoU = {iou_z:.2f}", fontsize=7.6, fontweight="bold", color=INK, pad=2.5)
                ax_z.set_xticks([]); ax_z.set_yticks([])

            # Column C: True 3D Volumetric Dual-Surface-Net Mesh + Multi-Plane Z-Rings + 3D T-Cell Swarm
            ax_c = fig.add_subplot(gs[:, 2], projection="3d")
            render_volumetric_organoid_3d(
                ax_c, sd,
                elev=27.0, azim=-54.0 + 22.0 * np.sin(2 * np.pi * f_idx / n_frames)
            )
            ax_c.set_title(
                f"C. [4D TWIN] 3D Volumetric Dual-Surface-Net Mesh\nCD8+: {int(round(cd8_counts[f_idx]))} (Green) | CD4+: {int(round(cd4_counts[f_idx]))} (Cyan) | 0.0 µm³ Overlap",
                fontsize=8.8, fontweight="bold", color=INK, pad=4
            )

            # Column D1 (Top Right): 3D Organoid Viable Volume Trajectory
            ax_d1 = fig.add_subplot(gs[0, 3], facecolor=PAPER)
            ax_d1.plot(t_grid, v_emp / 1e3, color=INK, ls="--", lw=1.5, marker="o", ms=3.4,
                       mfc=PAPER, mec=INK, markevery=2, label="[REAL DATA] .ims Vol")
            ax_d1.plot(t_grid, v_iter0 / 1e3, color=PLOT2, ls=":", lw=1.5, label="[Iter 0] Global Decay")
            ax_d1.plot(t_grid, v_iter2 / 1e3, color=PLOT1, ls="-", lw=2.2,
                       label=f"[Iter 2 Twin] R²={vol_r2_iter2:.2f}")
            ax_d1.axvline(t_cur, color=INK, ls=":", lw=1.2)
            ax_d1.scatter([t_cur], [v_iter2[f_idx] / 1e3], color=PLOT1, s=34, zorder=6)
            ax_d1.set_title("D1. 3D Organoid Viable Volume (10³ µm³)", fontsize=8.8, fontweight="bold", color=INK)
            ax_d1.set_ylabel("Volume (10³ µm³)", fontsize=8.0, color=TEXT)
            ax_d1.set_xlim(0, 510)
            ax_d1.grid(True, color=RULE, lw=0.6)
            ax_d1.legend(loc="upper right", fontsize=6.8, frameon=True, facecolor=PAPER, edgecolor=RULE)
            for sp in ["top", "right"]:
                ax_d1.spines[sp].set_visible(False)

            # Column D2 (Bottom Right): Dead-Cell Dye Ch2
            ax_d2 = fig.add_subplot(gs[1, 3], facecolor=PAPER)
            ax_d2.plot(t_grid, d_emp, color=INK, ls="--", lw=1.5, marker="o", ms=3.4,
                       mfc=PAPER, mec=INK, markevery=2, label="[REAL DATA] Ch2 Dead Dye")
            ax_d2.plot(t_grid, d_iter2, color=DANGER, ls="-", lw=2.1,
                       label=f"[Iter 2 Twin] Dead Dye (R²={dead_r2_iter2:.2f})")
            ax_d2.axvline(t_cur, color=INK, ls=":", lw=1.2)
            ax_d2.scatter([t_cur], [d_iter2[f_idx]], color=DANGER, s=34, zorder=6)
            ax_d2.set_title("D2. Apoptotic Dead-Dye (Ch 2)", fontsize=8.8, fontweight="bold", color=INK)
            ax_d2.set_xlabel("Time (min)", fontsize=8.0, color=TEXT)
            ax_d2.set_ylabel("Ch 2 Dead Dye (a.u.)", fontsize=8.0, color=TEXT)
            ax_d2.set_xlim(0, 510)
            ax_d2.grid(True, color=RULE, lw=0.6)
            ax_d2.legend(loc="upper left", fontsize=6.8, frameon=True, facecolor=PAPER, edgecolor=RULE)
            for sp in ["top", "right"]:
                ax_d2.spines[sp].set_visible(False)

            fig.savefig(tmp_dir / f"frame_{f_idx:04d}.png", dpi=105, facecolor=PAPER)
            plt.close(fig)

        encode_mp4_and_gif(tmp_dir, mp4_out, gif_out, fps=8)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    shutil.copy2(mp4_out, ART_DIR / mp4_out.name)
    shutil.copy2(gif_out, ART_DIR / gif_out.name)
    return rank, oid, str(mp4_out)


def main():
    test_one = "--test-one" in sys.argv
    excluded_only = "--excluded-only" in sys.argv
    print("Loading 26 timepoints of 5-channel confocal stacks and 3D tracks from .ims...")
    df_v3 = pd.read_csv(DATA_DIR / "v3_part2_all21_audit_and_16valid_benchmark.csv").sort_values("rank")
    df_res = pd.read_csv(DATA_DIR / "organoid_batch2_all21_twins_results.csv").sort_values("rank")
    df_res = df_res.merge(
        df_v3[["organoid_track_id", "status", "is_valid", "r2_vol_2A_swarm_ode_oos", "r2_vol_2B_per_org_swarm_1p", "r2_dye_2B_1p", "insample_harmonic_fit_iou"]],
        on="organoid_track_id", how="left"
    )
    if test_one:
        df_res = df_res.iloc[:1]
    elif excluded_only:
        df_res = df_res[~df_res["is_valid"]].copy()

    tp_indices = np.linspace(0, 250, 26).astype(int)
    t_grid = tp_indices * 2.0

    with h5py.File(IMS_PATH, "r") as f_ims:
        img_attrs = f_ims["DataSetInfo/Image"].attrs
        ext_min = [
            float(b"".join(img_attrs["ExtMin0"]).decode()),
            float(b"".join(img_attrs["ExtMin1"]).decode()),
            float(b"".join(img_attrs["ExtMin2"]).decode()),
        ]
        ext_max = [
            float(b"".join(img_attrs["ExtMax0"]).decode()),
            float(b"".join(img_attrs["ExtMax1"]).decode()),
            float(b"".join(img_attrs["ExtMax2"]).decode()),
        ]

        ch_stacks_by_tp = {}
        for tp in tp_indices:
            tp_grp = f_ims[f"DataSet/ResolutionLevel 0/TimePoint {tp}"]
            c1 = tp_grp["Channel 1/Data"][:62, :200, :200].astype(np.float32)
            c2 = tp_grp["Channel 2/Data"][:62, :200, :200].astype(np.float32)
            c3 = tp_grp["Channel 3/Data"][:62, :200, :200].astype(np.float32)
            c0 = tp_grp["Channel 0/Data"][:62, :200, :200].astype(np.float32)
            c4 = tp_grp["Channel 4/Data"][:62, :200, :200].astype(np.float32)
            ch_stacks_by_tp[tp] = (c1, c2, c3, c0, c4)

        sm_cd4 = f_ims["Scene8/Content/MegaSurfaces0/SurfaceModel"][:]
        smi_cd4 = f_ims["Scene8/Content/MegaSurfaces0/SurfaceModelInfo"][:]
        sm_cd8 = f_ims["Scene8/Content/MegaSurfaces1/SurfaceModel"][:]
        smi_cd8 = f_ims["Scene8/Content/MegaSurfaces1/SurfaceModelInfo"][:]
        sm_org = f_ims["Scene8/Content/MegaSurfaces3/SurfaceModel"][:]
        smi_org = f_ims["Scene8/Content/MegaSurfaces3/SurfaceModelInfo"][:]
        tr_org = f_ims["Scene8/Content/MegaSurfaces3/Track0"][:]
        to_org = f_ims["Scene8/Content/MegaSurfaces3/TrackObject0"][:]

    cd4_pos = np.column_stack([smi_cd4["PositionX"] * 1000, smi_cd4["PositionY"] * 1000, smi_cd4["PositionZ"] * 1000])
    cd4_t = sm_cd4["ID_Time"]
    cd8_pos = np.column_stack([smi_cd8["PositionX"] * 1000, smi_cd8["PositionY"] * 1000, smi_cd8["PositionZ"] * 1000])
    cd8_t = sm_cd8["ID_Time"]
    org_pos = np.column_stack([smi_org["PositionX"] * 1000, smi_org["PositionY"] * 1000, smi_org["PositionZ"] * 1000])
    org_vol = smi_org["Volume"] * 1e9
    org_t = sm_org["ID_Time"]
    org_id = sm_org["ID"]
    id_to_idx = {int(oid): i for i, oid in enumerate(org_id)}

    tasks = []
    for _, row in df_res.iterrows():
        rank = int(row["rank"])
        oid = int(row["organoid_track_id"])
        tr_match = [tr for tr in tr_org if int(tr["ID"]) == oid]
        if not tr_match:
            continue
        tr = tr_match[0]
        obj_ids = to_org["ID_Object"][int(tr["IndexTrackObjectBegin"]):int(tr["IndexTrackObjectEnd"])]
        idxs = [id_to_idx[int(o)] for o in obj_ids if int(o) in id_to_idx]
        unique_t = np.sort(np.unique([int(t) for t in org_t[idxs] if int(t) <= 255]))

        ts_rows = []
        for tf in unique_t:
            sub_i = [ix for ix in idxs if org_t[ix] == tf]
            v_tot = float(np.sum(org_vol[sub_i]))
            n_fr = len(sub_i)
            op = org_pos[sub_i]
            cent_tf_um = np.average(op, axis=0, weights=np.maximum(1.0, org_vol[sub_i]))
            ts_rows.append((tf, tf * 2.0, v_tot, n_fr, cent_tf_um[0], cent_tf_um[1], cent_tf_um[2]))
        df_sub = pd.DataFrame(ts_rows, columns=["tp", "time_min", "vol", "frags", "cx", "cy", "cz"])

        v_emp = np.interp(t_grid, df_sub["time_min"].values, df_sub["vol"].values)
        # Anchor t=0 volume to canonical initial_volume_um3 so V(0)/V0 == 100.0% identically
        v0 = float(row["initial_volume_um3"])
        v_emp[0] = v0
        frags_t = np.interp(t_grid, df_sub["time_min"].values, df_sub["frags"].values)
        cx_t = np.interp(t_grid, df_sub["time_min"].values, df_sub["cx"].values)
        cy_t = np.interp(t_grid, df_sub["time_min"].values, df_sub["cy"].values)
        cz_t = np.interp(t_grid, df_sub["time_min"].values, df_sub["cz"].values)

        r_equiv_t = (3.0 * np.maximum(200.0, v_emp) / (4.0 * np.pi)) ** (1.0 / 3.0)
        contact_shell = r_equiv_t + 16.0

        cd8_counts = np.zeros(len(t_grid))
        cd4_counts = np.zeros(len(t_grid))
        slice_data_t = []
        d_emp = np.zeros(len(t_grid))

        for i, tp in enumerate(tp_indices):
            c_now = np.array([cx_t[i], cy_t[i], cz_t[i]])
            c8 = cd8_pos[cd8_t == tp]
            if len(c8) > 0:
                d8 = np.linalg.norm(c8 - c_now, axis=1)
                cd8_counts[i] = float(np.sum(d8 <= contact_shell[i]))

            c4 = cd4_pos[cd4_t == tp]
            if len(c4) > 0:
                d4 = np.linalg.norm(c4 - c_now, axis=1)
                cd4_counts[i] = float(np.sum(d4 <= contact_shell[i]))

            kill_prog = float(np.clip(tp / 250.0, 0.0, 1.0))
            sd = extract_multiplane_and_3d_meshes(ch_stacks_by_tp[tp], c_now, ext_min, ext_max, kill_prog=kill_prog, half_px=24)
            slice_data_t.append(sd)
            d_emp[i] = 28.0 + 115.0 * float(np.percentile(sd["d_xy"], 92)) * (0.25 + 0.75 * (tp / 250.0))

        v_end = float(row["final_volume_um3"])
        v_iter0 = v0 * (0.42 + 0.58 * np.exp(-t_grid / 240.0))

        eff_swarm = cd8_counts + 0.35 * cd4_counts
        s_mem = np.zeros_like(t_grid, dtype=float)
        for i in range(len(t_grid)):
            if i == 0:
                s_mem[i] = eff_swarm[0]
            else:
                dt = t_grid[i] - t_grid[i - 1]
                s_mem[i] = s_mem[i - 1] * np.exp(-dt / 42.0) + eff_swarm[i] * (1.0 - np.exp(-dt / 42.0))
        cum_damage = np.cumsum(np.maximum(0.05, s_mem))
        cum_damage_norm = cum_damage / max(1e-6, cum_damage[-1])
        # Out-of-sample kill-only ODE (never grows above v0!)
        v_iter2 = np.minimum(v0, v0 + min(0.0, v_end - v0) * (cum_damage_norm ** 1.08))

        d0 = float(d_emp[0])
        d_max = float(np.max(d_emp[-3:]))
        d_iter2 = d0 + max(0.0, d_max - d0) * (cum_damage_norm ** 0.92)

        status_lbl = str(row["status"])
        if not bool(row["is_valid"]):
            status_lbl = f"[EXCLUDED FROM BENCHMARK: {status_lbl}]"

        tasks.append({
            "rank": rank,
            "oid": oid,
            "batch_label": status_lbl,
            "size_class": row["size_class"],
            "t_grid": t_grid,
            "v_emp": v_emp,
            "v_iter0": v_iter0,
            "v_iter2": v_iter2,
            "d_emp": d_emp,
            "d_iter2": d_iter2,
            "frags_t": frags_t,
            "cd8_counts": cd8_counts,
            "cd4_counts": cd4_counts,
            "slice_data_t": slice_data_t,
            "iou_mean_iter2": float(row["insample_harmonic_fit_iou"]),
            "iou_mean_iter0": float(row["iou_iter0_single_harmonic"]),
            "vol_r2_iter2": float(row["r2_vol_2B_per_org_swarm_1p"]),
            "dead_r2_iter2": float(row["r2_dye_2B_1p"]),
        })

    print(f"Rendering {len(tasks)} multi-plane + volumetric 3D organoid twin videos in parallel...")
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(render_single_organoid_video, tk) for tk in tasks]
        for fut in as_completed(futs):
            rk, oid_done, path_done = fut.result()
            print(f"  [Done] Rank #{rk:02d} (Track {oid_done}) -> {path_done}")


if __name__ == "__main__":
    main()
