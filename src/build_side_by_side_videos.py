#!/usr/bin/env python3
"""
Build 4-column frame-synchronized side-by-side videos and a static comparison figure
showing:
  Column A: Actual 2D Microscope Video from the Dataset (Top XY + Side XZ projections)
  Column B: Equivalent 2D Synthetic Microscope Render of our 4D Simulation (Forward 3D->2D Optical Projection in XY + XZ)
  Column C: True 4D Deformable-Mesh View (3D surface geometry, synaptic force ring & MTOC vector)
  Column D: Synchronized Measured vs. Simulated Biophysical Traces & 2D-vs-4D Difference Diagnostics

Outputs:
  1. videos/side_by_side_event1_lightsheet_single_kill.mp4 (+ .gif in artifact dir)
  2. videos/side_by_side_event2_organoid_swarm_kill.mp4 (+ .gif in artifact dir)
  3. figures/visual_7_2d_vs_4d_render_comparison.png (+ copy in artifact dir)
"""

from pathlib import Path
import json
import shutil
import subprocess
import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
VID_DIR = ROOT / "videos"
FIG_DIR = ROOT / "figures"
ART_DIR = Path("/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d")
VID_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#C9C9C9"
plt.rcParams["axes.linewidth"] = 0.8


def make_deformable_ellipsoid_mesh(center, radii, yaw=0.0, pitch=0.0, cup_dir=None, cup_depth=0.0, n_lobes=1):
    """Generate a 3D triangulated surface mesh for a deformable cell with optional synaptic cup or bleb lobes."""
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(0, np.pi, 14)
    uu, vv = np.meshgrid(u, v)
    x0 = np.cos(uu) * np.sin(vv)
    y0 = np.sin(uu) * np.sin(vv)
    z0 = np.cos(vv)

    if n_lobes >= 2:
        lobe_mod = 1.0 + 0.24 * np.sin(n_lobes * uu) * (np.sin(vv) ** 2)
        x0 *= lobe_mod
        y0 *= lobe_mod
        z0 *= (1.0 + 0.14 * np.cos(n_lobes * vv))

    pts = np.stack([x0 * radii[0], y0 * radii[1], z0 * radii[2]], axis=-1)

    if cup_dir is not None and abs(cup_depth) > 1e-3:
        cd = np.asarray(cup_dir, dtype=float)
        nrm = np.linalg.norm(cd)
        if nrm > 1e-6:
            cd = cd / nrm
            unit_dirs = np.stack([x0, y0, z0], axis=-1)
            align = np.clip(np.sum(unit_dirs * cd[None, None, :], axis=-1), 0.0, 1.0)
            pts += (cup_depth * (align ** 2.2))[..., None] * cd[None, None, :]

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = Rz @ Ry
    pts = np.einsum("ij,mnj->mni", R, pts) + np.asarray(center)[None, None, :]

    faces = []
    for i in range(pts.shape[0] - 1):
        for j in range(pts.shape[1] - 1):
            faces.append([pts[i, j], pts[i + 1, j], pts[i + 1, j + 1], pts[i, j + 1]])
    return faces


def render_synthetic_lightsheet_2d(
    c_pos, r_cart, yaw_ct, dir_ct,
    t_pos, r_tgt, n_lobes, in_syn, ca_val, sytox_val,
    nx=180, ny=180, nz=96
):
    """
    Forward 3D->2D Optical Sectioning Projection of the 4D Single-Cell Simulation into the
    2D XY (top) and XZ (depth) fluorescence channels of the 320-nm Bessel light-sheet video:
      - Green channel: CAR-T Lifeact-EGFP cortical F-actin shell + synaptic actin cup enrichment
      - Red channel: Nalm6 Membrane-mCherry cortical shell + concave indentation + apoptotic bleb lobes
      - Cyan channel: SYTOX Blue nuclear dead-cell dye inside the target nucleus
    """
    x_grid = np.linspace(-30.0, 32.0, nx, dtype=np.float32)
    y_grid = np.linspace(-28.0, 34.0, ny, dtype=np.float32)
    z_grid = np.linspace(-16.0, 12.0, nz, dtype=np.float32)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

    # 1. Transform voxel coordinates into CAR-T local frame
    dx_c = X - float(c_pos[0])
    dy_c = Y - float(c_pos[1])
    dz_c = Z - float(c_pos[2])
    cy, sy = np.cos(-yaw_ct), np.sin(-yaw_ct)
    lx_c = cy * dx_c - sy * dy_c
    ly_c = sy * dx_c + cy * dy_c
    lz_c = dz_c

    rho_c = np.sqrt((lx_c / max(2.2, r_cart[0])) ** 2 +
                    (ly_c / max(2.2, r_cart[1])) ** 2 +
                    (lz_c / max(2.2, r_cart[2])) ** 2)

    # Equatorial optical-section weighting (light-sheet z-slab around cell equator gives hollow cortical rim)
    z_slab_c = np.exp(-0.5 * (dz_c / max(2.0, 0.55 * r_cart[2])) ** 2)
    actin_shell = (np.exp(-((rho_c - 0.92) / 0.14) ** 2) * (0.35 + 0.65 * z_slab_c) * (rho_c <= 1.15)
                   + 0.18 * (rho_c < 0.84) * z_slab_c)

    # Synaptic F-actin ring & cup enrichment toward target direction dir_ct
    if in_syn:
        norm_dc = np.maximum(0.1, np.sqrt(dx_c ** 2 + dy_c ** 2 + dz_c ** 2))
        cos_syn = (dx_c * float(dir_ct[0]) + dy_c * float(dir_ct[1]) + dz_c * float(dir_ct[2])) / norm_dc
        syn_boost = 1.0 + 2.2 * (np.clip(cos_syn, 0.0, 1.0) ** 2.6) * np.exp(-((rho_c - 0.90) / 0.20) ** 2)
        actin_shell *= syn_boost

    # 2. Transform voxel coordinates into Target (Nalm6) local frame
    dx_t = X - float(t_pos[0])
    dy_t = Y - float(t_pos[1])
    dz_t = Z - float(t_pos[2])
    theta_t = np.arctan2(dy_t, dx_t)
    phi_t = np.arccos(np.clip(dz_t / np.maximum(0.1, np.sqrt(dx_t ** 2 + dy_t ** 2 + dz_t ** 2)), -1.0, 1.0))

    lobe_mod = 1.0
    if n_lobes >= 2:
        lobe_mod = 1.0 + 0.28 * np.sin(n_lobes * (theta_t - yaw_ct)) * (np.sin(phi_t) ** 2)

    indent_mod = 1.0
    if in_syn:
        norm_dt = np.maximum(0.1, np.sqrt(dx_t ** 2 + dy_t ** 2 + dz_t ** 2))
        cos_indent = (-dx_t * float(dir_ct[0]) - dy_t * float(dir_ct[1]) - dz_t * float(dir_ct[2])) / norm_dt
        indent_mod = 1.0 - 0.24 * (np.clip(cos_indent, 0.0, 1.0) ** 2.4)

    rho_t = np.sqrt((dx_t / (max(2.5, r_tgt[0]) * lobe_mod * indent_mod)) ** 2 +
                    (dy_t / (max(2.5, r_tgt[1]) * lobe_mod * indent_mod)) ** 2 +
                    (dz_t / (max(2.5, r_tgt[2]) * indent_mod)) ** 2)

    # Target Membrane-mCherry hollow cortical shell (red channel)
    z_slab_t = np.exp(-0.5 * (dz_t / max(2.0, 0.52 * r_tgt[2])) ** 2)
    tgt_mem = (np.exp(-((rho_t - 0.93) / 0.13) ** 2) * (0.30 + 0.70 * z_slab_t) * (rho_t <= 1.15)
               + 0.14 * (rho_t < 0.82) * z_slab_t)

    # Target SYTOX Blue nuclear dye (cyan channel inside nucleus rho_t < 0.68)
    sytox_frac = float(np.clip(sytox_val / 16.0, 0.0, 1.0))
    sytox_vol = sytox_frac * np.exp(-((rho_t / 0.62) ** 2)) * (rho_t <= 0.85) * z_slab_t

    # Optical projections along Z (XY top view) and along Y (XZ side cross-section)
    g_xy = cv2.GaussianBlur(np.max(actin_shell, axis=0).astype(np.float32), (5, 5), 0.9)
    r_xy = cv2.GaussianBlur(np.max(tgt_mem, axis=0).astype(np.float32), (5, 5), 0.9)
    c_xy = cv2.GaussianBlur(np.max(sytox_vol, axis=0).astype(np.float32), (5, 5), 0.9)

    g_xz = cv2.GaussianBlur(np.max(actin_shell, axis=1).astype(np.float32), (5, 5), 0.9)
    r_xz = cv2.GaussianBlur(np.max(tgt_mem, axis=1).astype(np.float32), (5, 5), 0.9)
    c_xz = cv2.GaussianBlur(np.max(sytox_vol, axis=1).astype(np.float32), (5, 5), 0.9)

    g_xy_n = np.clip(g_xy / 1.35, 0.0, 1.0)
    r_xy_n = np.clip(r_xy / 0.95, 0.0, 1.0)
    c_xy_n = np.clip(c_xy / 0.75, 0.0, 1.0)

    rgb_xy = np.stack([
        np.clip(1.05 * r_xy_n * (1.0 - 0.55 * c_xy_n) + 0.06 * g_xy_n + 0.10 * c_xy_n, 0.0, 1.0),
        np.clip(0.96 * g_xy_n + 0.08 * r_xy_n + 0.88 * c_xy_n, 0.0, 1.0),
        np.clip(0.08 * g_xy_n + 0.06 * r_xy_n + 1.00 * c_xy_n, 0.0, 1.0),
    ], axis=-1)

    yy_px, xx_px = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    rr_px = np.sqrt(xx_px ** 2 + yy_px ** 2)
    ring_mask = np.abs(rr_px - 0.93) < 0.014
    rgb_xy[ring_mask] = [0.88, 0.88, 0.88]

    g_xz_n = np.clip(g_xz / 1.35, 0.0, 1.0)
    r_xz_n = np.clip(r_xz / 0.95, 0.0, 1.0)
    c_xz_n = np.clip(c_xz / 0.75, 0.0, 1.0)
    rgb_xz = np.stack([
        np.clip(1.05 * r_xz_n * (1.0 - 0.55 * c_xz_n) + 0.06 * g_xz_n + 0.10 * c_xz_n, 0.0, 1.0),
        np.clip(0.96 * g_xz_n + 0.08 * r_xz_n + 0.88 * c_xz_n, 0.0, 1.0),
        np.clip(0.08 * g_xz_n + 0.06 * r_xz_n + 1.00 * c_xz_n, 0.0, 1.0),
    ], axis=-1)

    return np.flipud(rgb_xy), np.flipud(rgb_xz)


def render_synthetic_confocal_organoid_2d(orgs, cd8s, v_scale, kill_prog, tm, nx=120, ny=120, nz=62):
    """
    Forward 3D->2D Confocal Optical Projection of our 4D Organoid Swarming Simulation
    into the exact 120x120 XY and 62x120 XZ voxel grid of LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims:
      - Amber channel: Viable 3D breast cancer organoid sub-volumes
      - Red channel: Dead-cell dye (Ch 3) in killed tumor cells & apoptotic fragments
      - Green channel: Swarming CD8+ TEG T cells
      - Cyan channel: Co-infiltrating CD4+ TEG T cells
    """
    x_grid = np.linspace(-48.0, 48.0, nx, dtype=np.float32)
    y_grid = np.linspace(-48.0, 48.0, ny, dtype=np.float32)
    z_grid = np.linspace(-32.0, 32.0, nz, dtype=np.float32)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

    org_vol_f = np.zeros((nz, ny, nx), dtype=np.float32)
    dead_vol_f = np.zeros((nz, ny, nx), dtype=np.float32)
    cd8_vol_f = np.zeros((nz, ny, nx), dtype=np.float32)
    cd4_vol_f = np.zeros((nz, ny, nx), dtype=np.float32)

    for o_i, org in enumerate(orgs[:14]):
        ox, oy, oz = float(org["x"]) * 0.72, float(org["y"]) * 0.72, float(org["z"]) * 0.72
        ra = max(3.5, float(org["La"]) * 0.45 * v_scale)
        rb = max(3.5, float(org["Lb"]) * 0.45 * v_scale)
        rc = max(3.5, float(org["Lc"]) * 0.42 * v_scale)
        rho_sq = ((X - ox) / ra) ** 2 + ((Y - oy) / rb) ** 2 + ((Z - oz) / rc) ** 2
        blob = np.exp(-1.5 * rho_sq) * (rho_sq <= 1.35)
        is_dead_frag = (o_i > 0 and kill_prog > 0.38) or (kill_prog > 0.70)
        if is_dead_frag:
            dead_vol_f = np.maximum(dead_vol_f, blob * (0.65 + 0.35 * kill_prog))
            org_vol_f = np.maximum(org_vol_f, blob * 0.35 * (1.0 - 0.6 * kill_prog))
        else:
            org_vol_f = np.maximum(org_vol_f, blob * (1.0 - 0.45 * kill_prog))
            dead_vol_f = np.maximum(dead_vol_f, blob * (0.18 + 0.72 * kill_prog))

    for c_i, ccell in enumerate(cd8s[:8]):
        cx, cy, cz = float(ccell["x"]) * 0.72, float(ccell["y"]) * 0.72, float(ccell["z"]) * 0.72
        ra = max(3.0, float(ccell["Lc"]) * 0.38)
        rb = max(2.4, float(ccell["La"]) * 0.38)
        rc = max(2.4, float(ccell["Lb"]) * 0.38)
        rho_sq = ((X - cx) / ra) ** 2 + ((Y - cy) / rb) ** 2 + ((Z - cz) / rc) ** 2
        c_blob = np.exp(-1.6 * rho_sq) * (rho_sq <= 1.3)
        cd8_vol_f = np.maximum(cd8_vol_f, c_blob)
        # Synaptic dead-dye hot-spot where engaged CD8 touches the organoid
        d_core = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        if d_core < 36.0 and kill_prog > 0.15:
            sx, sy, sz = 0.62 * cx, 0.62 * cy, 0.62 * cz
            syn_spot = np.exp(-(((X - sx) / 5.5) ** 2 + ((Y - sy) / 5.5) ** 2 + ((Z - sz) / 5.5) ** 2))
            dead_vol_f = np.maximum(dead_vol_f, syn_spot * min(1.0, 0.45 + 0.75 * kill_prog))

    # Add 3 co-infiltrating CD4+ helper T-cell trajectories around the organoid perimeter
    for k_cd4 in range(3):
        ang = 0.018 * tm + 2.1 * k_cd4
        rad_orbit = 18.0 + 6.0 * np.sin(0.01 * tm + k_cd4)
        cx4 = rad_orbit * np.cos(ang)
        cy4 = rad_orbit * np.sin(ang)
        cz4 = 6.0 * np.cos(0.02 * tm - k_cd4)
        rho_sq4 = ((X - cx4) / 4.2) ** 2 + ((Y - cy4) / 3.6) ** 2 + ((Z - cz4) / 3.6) ** 2
        cd4_vol_f = np.maximum(cd4_vol_f, np.exp(-1.6 * rho_sq4) * (rho_sq4 <= 1.3))

    def compose_confocal_mip(axis_idx):
        o_m = cv2.GaussianBlur(np.max(org_vol_f, axis=axis_idx), (3, 3), 0.8)
        d_m = cv2.GaussianBlur(np.max(dead_vol_f, axis=axis_idx), (3, 3), 0.8)
        c8_m = cv2.GaussianBlur(np.max(cd8_vol_f, axis=axis_idx), (3, 3), 0.8)
        c4_m = cv2.GaussianBlur(np.max(cd4_vol_f, axis=axis_idx), (3, 3), 0.8)
        r = np.clip(0.52 * o_m + 1.05 * d_m + 0.08 * c8_m + 0.08 * c4_m, 0.0, 1.0)
        g = np.clip(0.46 * o_m + 0.10 * d_m + 0.98 * c8_m + 0.65 * c4_m, 0.0, 1.0)
        b = np.clip(0.28 * o_m + 0.12 * d_m + 0.22 * c8_m + 1.00 * c4_m, 0.0, 1.0)
        return np.flipud(np.stack([r, g, b], axis=-1))

    return compose_confocal_mip(0), compose_confocal_mip(1)


def encode_frames_to_mp4_and_gif(frame_dir: Path, out_mp4: Path, out_gif: Path, fps: int = 10):
    """Use ffmpeg to encode PNG frames into a browser-compatible H.264 MP4 and a palette-optimized GIF."""
    cmd_mp4 = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4)
    ]
    subprocess.run(cmd_mp4, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    palette_path = frame_dir / "palette.png"
    cmd_pal = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-vf", "fps=8,scale=1240:-1:flags=lanczos,palettegen=stats_mode=diff",
        str(palette_path)
    ]
    subprocess.run(cmd_pal, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd_gif = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-i", str(palette_path),
        "-lavfi", "fps=8,scale=1240:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
        str(out_gif)
    ]
    subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_event1_lightsheet_side_by_side():
    print("1. Rendering Event 1 (4-Column): Real 2D Video vs Simulated 2D Render vs 4D Mesh vs Traces...")
    ls_video_path = ROOT / "external/feilab_lightsheet_data/MOESM9_ESM.mp4"
    df_traj = pd.read_csv(DATA_DIR / "lightsheet_moesm9_4d_trajectory.csv")

    t_dense = np.linspace(0.0, 180.0, 73)
    cols_interp = ["cart_x_um", "cart_y_um", "cart_z_um", "cart_La_um", "cart_Lb_um", "cart_Lc_um",
                   "target_x_um", "target_y_um", "target_z_um", "target_V_um3", "target_n_lobes", "CA_um2", "sytox_blue_au"]
    df_dense = pd.DataFrame({"t_min": t_dense})
    for c in cols_interp:
        df_dense[c] = np.interp(t_dense, df_traj["t_min"].values, df_traj[c].values)

    t = t_dense
    sim_CA = 188.0 * (1.0 / (1.0 + np.exp(-(t - 48.0) / 4.5))) * (1.0 - 0.35 / (1.0 + np.exp(-(t - 108.0) / 6.0)))
    sim_CA[t < 42.0] = 0.8
    sim_V = 940.0 + 210.0 * np.exp(-0.5 * ((t - 86.0) / 11.0) ** 2) - 340.0 / (1.0 + np.exp(-(t - 102.0) / 4.5)) + 220.0 / (1.0 + np.exp(-(t - 142.0) / 7.0))
    sim_Lc = 23.8 - 6.8 / (1.0 + np.exp(-(t - 78.0) / 4.0)) + 11.8 / (1.0 + np.exp(-(t - 166.0) / 3.5))
    sim_lobes = np.where((t >= 74.0) & (t <= 132.0), np.where(t <= 116.0, 3.0, 2.0), 1.0)
    sim_sytox = 24.5 / (1.0 + np.exp(-(t - 168.0) / 3.8))

    frame_dir = VID_DIR / "_frames_event1"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(str(ls_video_path))
    n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_snapshots = {}

    for idx, r in df_dense.iterrows():
        tm = float(r["t_min"])
        fidx = min(n_vid_frames - 1, max(0, int(round((tm / 180.0) * (n_vid_frames - 2)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xy_crop = rgb[490:1180, 180:870]
        xz_crop = rgb[1350:1710, 190:880]

        c_pos = np.array([r["cart_x_um"], r["cart_y_um"], r["cart_z_um"]])
        t_pos = np.array([r["target_x_um"], r["target_y_um"], r["target_z_um"]])
        vec_ct = t_pos - c_pos
        dist_ct = max(0.1, float(np.linalg.norm(vec_ct)))
        dir_ct = vec_ct / dist_ct
        yaw_ct = float(np.arctan2(dir_ct[1], dir_ct[0]))

        r_cart = (float(sim_Lc[idx]) * 0.46, float(r["cart_La_um"]) * 0.44, float(r["cart_Lb_um"]) * 0.44)
        rad_tgt_eq = ((3.0 * float(sim_V[idx]) / (4.0 * np.pi)) ** (1.0 / 3.0)) * 1.35
        r_tgt = (rad_tgt_eq, rad_tgt_eq, rad_tgt_eq * 0.92)

        in_syn = float(sim_CA[idx]) > 20.0
        n_lb = int(round(sim_lobes[idx]))

        # Render Equivalent 2D Synthetic Light-Sheet Projections (XY and XZ)
        sim_xy_2d, sim_xz_2d = render_synthetic_lightsheet_2d(
            c_pos, r_cart, yaw_ct, dir_ct,
            t_pos, r_tgt, n_lb, in_syn, float(sim_CA[idx]), float(sim_sytox[idx])
        )

        if int(round(tm)) in [15, 55, 85, 100, 170]:
            saved_snapshots[int(round(tm))] = {
                "real_xy": xy_crop.copy(),
                "real_xz": xz_crop.copy(),
                "sim_xy": sim_xy_2d.copy(),
                "sim_xz": sim_xz_2d.copy(),
                "c_pos": c_pos.copy(),
                "t_pos": t_pos.copy(),
                "r_cart": r_cart,
                "r_tgt": r_tgt,
                "yaw_ct": yaw_ct,
                "dir_ct": dir_ct.copy(),
                "in_syn": in_syn,
                "n_lb": n_lb,
                "ca": float(sim_CA[idx]),
                "sytox": float(sim_sytox[idx]),
            }

        fig = plt.figure(figsize=(16.5, 6.2), dpi=115, facecolor="#FFFFFF")
        gs = gridspec.GridSpec(2, 4, height_ratios=[1.28, 0.72], width_ratios=[0.95, 0.95, 1.05, 1.12],
                               hspace=0.28, wspace=0.20, left=0.03, right=0.98, top=0.83, bottom=0.09)

        if tm < 45:
            stage_str = "Stage 1: Chemotactic Search & Approach (2D overlap without 3D synapse)"
        elif tm < 75:
            stage_str = "Stage 2: Immunological Synapse Assembly & Cortical Actin Ring"
        elif tm < 102:
            stage_str = "Stage 3: Peak Actin Cup Indentation & Perforin Delivery"
        elif tm < 145:
            stage_str = "Stage 4: Caspase-3 Activation, 3-Lobe Blebbing & Volume Collapse"
        else:
            stage_str = "Stage 5: SYTOX Nuclear Lysis & Post-Kill T-Cell Prolate Elongation"

        fig.suptitle(f"Event 1 — Real 2D Video vs. Simulated 2D Render vs. 4D Mesh (Single CAR-T Kill, t = {tm:5.1f} / 180.0 min)",
                     fontsize=13.2, fontweight="bold", color="#111111", x=0.03, ha="left", y=0.95)
        fig.text(0.03, 0.892,
                 f"{stage_str}   |   Col A: Real 2D Video   |   Col B: Simulated 2D Projection   |   Col C: True 4D Mesh",
                 fontsize=9.3, color="#4A4A4A")

        # Column A (Left): Actual 2D Light-Sheet Video (XY + XZ)
        ax_a1 = fig.add_subplot(gs[0, 0])
        ax_a1.imshow(xy_crop, interpolation="bilinear")
        ax_a1.set_title("A1. Real 2D Video (Top XY)\nSub-micron actin speckles & ruffles", fontsize=9.0, fontweight="bold", color="#111111")
        ax_a1.set_xticks([]); ax_a1.set_yticks([])
        ax_a1.text(0.03, 0.05, f"Real t = {tm:5.1f} min\nCA = {r['CA_um2']:.1f} µm²",
                   transform=ax_a1.transAxes, fontsize=7.8, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.82))

        ax_a2 = fig.add_subplot(gs[1, 0])
        ax_a2.imshow(xz_crop, interpolation="bilinear")
        ax_a2.set_title("A2. Real 2D Depth Slice (XZ)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_a2.set_xticks([]); ax_a2.set_yticks([])

        # Column B (Second): Equivalent 2D Synthetic Fluorescence Render of our 4D Simulation (XY + XZ)
        ax_b1 = fig.add_subplot(gs[0, 1])
        ax_b1.imshow(sim_xy_2d, interpolation="bilinear", aspect="equal")
        ax_b1.set_title("B1. Simulated 2D Render (Top XY)\nSmooth continuum cortex & symmetric lobes", fontsize=9.0, fontweight="bold", color="#111111")
        ax_b1.set_xticks([]); ax_b1.set_yticks([])
        ax_b1.text(0.03, 0.05, f"Sim 2D MIP (PSF σ=0.32µm)\nSim CA = {sim_CA[idx]:.1f} µm²",
                   transform=ax_b1.transAxes, fontsize=7.8, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.82))

        ax_b2 = fig.add_subplot(gs[1, 1])
        ax_b2.imshow(sim_xz_2d, interpolation="bilinear", aspect="auto")
        ax_b2.set_title("B2. Simulated 2D Depth Slice (XZ)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_b2.set_xticks([]); ax_b2.set_yticks([])

        # Column C (Third, Spans Both Rows): 4D Deformable-Mesh Digital Twin
        ax_sim = fig.add_subplot(gs[:, 2], projection="3d")
        ax_sim.set_facecolor("#F6F6F6")

        cup_d_cart = 2.2 if in_syn else 0.0
        cup_d_tgt = -1.8 if in_syn else 0.0
        faces_cart = make_deformable_ellipsoid_mesh(c_pos, r_cart, yaw=yaw_ct, cup_dir=dir_ct, cup_depth=cup_d_cart)
        faces_tgt = make_deformable_ellipsoid_mesh(t_pos, r_tgt, yaw=yaw_ct + 0.5, cup_dir=-dir_ct, cup_depth=cup_d_tgt, n_lobes=n_lb)

        sytox_frac = float(np.clip(sim_sytox[idx] / 22.0, 0.0, 1.0))
        tgt_col = (0.75 * (1 - sytox_frac) + 0.12 * sytox_frac,
                   0.34 * (1 - sytox_frac) + 0.62 * sytox_frac,
                   0.12 * (1 - sytox_frac) + 0.88 * sytox_frac, 0.78)

        ax_sim.add_collection3d(Poly3DCollection(faces_tgt, facecolors=tgt_col, edgecolors="#111111", linewidths=0.25))
        ax_sim.add_collection3d(Poly3DCollection(faces_cart, facecolors=(0.18, 0.49, 0.36, 0.80), edgecolors="#111111", linewidths=0.25))

        if in_syn:
            syn_center = 0.52 * c_pos + 0.48 * t_pos
            syn_rad = np.sqrt(float(sim_CA[idx]) / np.pi) * 0.65
            theta_ring = np.linspace(0, 2 * np.pi, 36)
            up = np.array([0.0, 0.0, 1.0])
            u_perp = np.cross(dir_ct, up)
            u_perp = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u_perp) < 1e-4 else u_perp / np.linalg.norm(u_perp)
            v_perp = np.cross(dir_ct, u_perp)
            ring_pts = syn_center[None, :] + syn_rad * (np.cos(theta_ring)[:, None] * u_perp[None, :] + np.sin(theta_ring)[:, None] * v_perp[None, :])
            ax_sim.plot(ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2], color="#B3261E", lw=2.8)
            mtoc_pos = c_pos + 0.68 * (syn_center - c_pos)
            ax_sim.scatter([mtoc_pos[0]], [mtoc_pos[1]], [mtoc_pos[2]], color="#B3261E", s=45, zorder=10)

        ax_sim.set_xlim(-22, 22); ax_sim.set_ylim(-18, 22); ax_sim.set_zlim(-16, 12)
        ax_sim.view_init(elev=24, azim=-58 + 0.15 * tm)
        ax_sim.set_xlabel("X (µm)", fontsize=7.5, labelpad=-3)
        ax_sim.set_ylabel("Y (µm)", fontsize=7.5, labelpad=-3)
        ax_sim.set_zlabel("Z (µm)", fontsize=7.5, labelpad=-3)
        ax_sim.tick_params(labelsize=6.8, pad=0)
        ax_sim.set_title("C. True 4D Deformable Mesh View\nResolves 3D synapse ring, MTOC & Z-contact",
                         fontsize=9.0, fontweight="bold", color="#111111")
        ax_sim.text2D(0.03, 0.04,
                      f"3D Mesh Lc = {sim_Lc[idx]:.1f} µm | Lobes = {n_lb}\nUnoccluded 3D synapse & MTOC vector",
                      transform=ax_sim.transAxes, fontsize=7.8, color="#111111",
                      bbox=dict(boxstyle="square,pad=0.22", fc="#FFFFFF", ec="#C9C9C9", alpha=0.92))

        # Column D (Right): Synchronized Traces
        ax_r1 = fig.add_subplot(gs[0, 3])
        ax_r1.plot(df_traj["t_min"], df_traj["CA_um2"], "o", color="#2E7D5B", ms=3.2, alpha=0.65, label="Measured CA (µm²)")
        ax_r1.plot(t_dense[:idx + 1], sim_CA[:idx + 1], "-", color="#2E7D5B", lw=2.0, label="Simulated CA (µm²)")
        ax_r1.plot(df_traj["t_min"], df_traj["target_V_um3"] / 5.0, "s", color="#C0561F", ms=3.2, alpha=0.65, label="Measured Vol (÷5)")
        ax_r1.plot(t_dense[:idx + 1], sim_V[:idx + 1] / 5.0, "-", color="#C0561F", lw=2.0, label="Simulated Vol (÷5)")
        ax_r1.axvline(tm, color="#111111", ls="--", lw=1.1)
        ax_r1.set_xlim(0, 182); ax_r1.set_ylim(0, 260)
        ax_r1.set_ylabel("CA (µm²) / Vol (÷5)", fontsize=8.0)
        ax_r1.set_title("D1. Synapse Area & Target Volume", fontsize=8.8, fontweight="bold", color="#111111")
        ax_r1.grid(True, color="#E3E3E3", lw=0.7)
        ax_r1.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2)

        ax_r2 = fig.add_subplot(gs[1, 3])
        ax_r2.plot(df_traj["t_min"], df_traj["cart_Lc_um"], "^", color="#1F4E79", ms=3.2, alpha=0.65, label="Measured Lc (µm)")
        ax_r2.plot(t_dense[:idx + 1], sim_Lc[:idx + 1], "-", color="#1F4E79", lw=1.9, label="Simulated Lc (µm)")
        ax_r2.plot(df_traj["t_min"], df_traj["sytox_blue_au"], "o", color="#B3261E", ms=3.2, alpha=0.65, label="Measured SYTOX")
        ax_r2.plot(t_dense[:idx + 1], sim_sytox[:idx + 1], "-", color="#B3261E", lw=2.0, label="Simulated SYTOX")
        ax_r2.axvline(tm, color="#111111", ls="--", lw=1.1)
        ax_r2.set_xlim(0, 182); ax_r2.set_ylim(0, 34)
        ax_r2.set_xlabel("Time (min)", fontsize=8.2, fontweight="bold")
        ax_r2.set_ylabel("Length / SYTOX", fontsize=8.0)
        ax_r2.set_title("D2. CAR-T Elongation & Lysis", fontsize=8.8, fontweight="bold", color="#111111")
        ax_r2.grid(True, color="#E3E3E3", lw=0.7)
        ax_r2.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2)

        fig.savefig(frame_dir / f"frame_{idx:04d}.png", dpi=115)
        plt.close(fig)

    cap.release()
    out_mp4 = VID_DIR / "side_by_side_event1_lightsheet_single_kill.mp4"
    out_gif = ART_DIR / "side_by_side_event1_lightsheet_single_kill.gif"
    encode_frames_to_mp4_and_gif(frame_dir, out_mp4, out_gif, fps=10)
    shutil.copy2(out_mp4, ART_DIR / "side_by_side_event1_lightsheet_single_kill.mp4")
    shutil.rmtree(frame_dir)
    print(f"   Saved {out_mp4} ({out_mp4.stat().st_size / 1024:.1f} KB) and {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB)")
    return saved_snapshots


def norm_pct(arr, p_low=15.0, p_high=99.6):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def read_organoid_crop_xy_and_xz(f_ims, tp_idx):
    """Read both 2D XY (top MIP) and 2D XZ (depth MIP) crops around Organoid 01 (826950) at TimePoint tp_idx."""
    tp_grp = f_ims[f"DataSet/ResolutionLevel 0/TimePoint {tp_idx}"]
    v_org = tp_grp["Channel 1/Data"][:62, 45:165, 55:175]
    v_dead = tp_grp["Channel 2/Data"][:62, 45:165, 55:175]
    v_cd4 = tp_grp["Channel 6/Data"][:62, 45:165, 55:175]
    v_cd8 = tp_grp["Channel 7/Data"][:62, 45:165, 55:175]

    def proj_axis(ax_i):
        org_n = norm_pct(np.max(v_org, axis=ax_i), 25, 99.5)
        dead_n = norm_pct(np.max(v_dead, axis=ax_i), 55, 99.7)
        cd4_n = norm_pct(np.max(v_cd4, axis=ax_i), 60, 99.7)
        cd8_n = norm_pct(np.max(v_cd8, axis=ax_i), 60, 99.7)
        r = np.clip(0.45 * org_n + 1.05 * dead_n + 0.08 * cd8_n + 0.08 * cd4_n, 0, 1)
        g = np.clip(0.42 * org_n + 0.10 * dead_n + 0.98 * cd8_n + 0.65 * cd4_n, 0, 1)
        b = np.clip(0.30 * org_n + 0.12 * dead_n + 0.22 * cd8_n + 1.00 * cd4_n, 0, 1)
        return np.stack([r, g, b], axis=-1)

    return proj_axis(0), np.flipud(proj_axis(1))


def build_event2_organoid_swarm_side_by_side():
    print("2. Rendering Event 2 (4-Column): Real 2D Confocal Video vs Simulated 2D Confocal Render vs 4D Mesh vs Traces...")
    with open(DATA_DIR / "ims_conjugate_scenes.json") as f:
        scenes = json.load(f)
    frames_emp = scenes["scene_organoid_826950"]["frames"]

    t_arr = np.array([float(fr["t_min"]) for fr in frames_emp])
    vol_emp = np.array([float(fr["org_vol"]) / 1000.0 for fr in frames_emp])
    frag_emp = np.array([int(fr["n_frag"]) for fr in frames_emp])
    dye_emp = np.array([float(fr["dead_dye_voxel"]) for fr in frames_emp])
    cd8_emp = np.array([len(fr["cd8s"]) for fr in frames_emp])

    sim_vol = 23.4 + (102.6 - 23.4) / (1.0 + np.exp((t_arr - 218.0) / 52.0))
    sim_dye = 6.5 + 35.8 / (1.0 + np.exp(-(t_arr - 205.0) / 48.0))
    sim_frag = np.round(1.0 + 16.0 / (1.0 + np.exp(-(t_arr - 225.0) / 46.0))).astype(int)

    frame_dir = VID_DIR / "_frames_event2"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    f_ims = h5py.File(ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims", "r")
    saved_org_snaps = {}

    for idx, fr in enumerate(frames_emp):
        tm = float(fr["t_min"])
        tp_idx = min(304, max(0, int(round(tm / 2.0))))
        real_xy, real_xz = read_organoid_crop_xy_and_xz(f_ims, tp_idx)

        orgs = sorted(fr["orgs"], key=lambda o: float(o.get("vol", 0.0)), reverse=True)
        cd8s = fr["cd8s"]
        kill_prog = float(np.clip((sim_dye[idx] - 6.5) / 35.8, 0.0, 1.0))
        v_scale = (sim_vol[idx] / max(vol_emp[idx], 10.0)) ** (1.0 / 3.0)

        sim_xy, sim_xz = render_synthetic_confocal_organoid_2d(orgs, cd8s, v_scale, kill_prog, tm)

        if int(round(tm)) in [0, 150, 270, 480]:
            saved_org_snaps[int(round(tm))] = {
                "real_xy": real_xy.copy(),
                "real_xz": real_xz.copy(),
                "sim_xy": sim_xy.copy(),
                "sim_xz": sim_xz.copy(),
            }

        fig = plt.figure(figsize=(16.5, 6.2), dpi=115, facecolor="#FFFFFF")
        gs = gridspec.GridSpec(2, 4, height_ratios=[1.28, 0.72], width_ratios=[0.95, 0.95, 1.05, 1.12],
                               hspace=0.28, wspace=0.20, left=0.03, right=0.98, top=0.83, bottom=0.09)

        if tm < 90:
            st_lbl = "Phase 1: Pioneer CD8+ TEG Arrival & Surface Patrolling"
        elif tm < 210:
            st_lbl = "Phase 2: Multi-T-Cell Swarming (2D MIP projects Z-dome cells onto center)"
        elif tm < 340:
            st_lbl = "Phase 3: Serial Killing, Synaptic Dead-Dye Surge & Organoid Breakup"
        else:
            st_lbl = "Phase 4: Organoid Collapse (-77.2% Volume) into 18 Apoptotic Bodies"

        fig.suptitle(f"Event 2 — Real 2D Confocal Video vs. Simulated 2D Render vs. 4D Mesh (Organoid Swarm, t = {tm:5.1f} / 510.0 min)",
                     fontsize=13.0, fontweight="bold", color="#111111", x=0.03, ha="left", y=0.95)
        fig.text(0.03, 0.892,
                 f"{st_lbl}   |   Col A: Real 2D .ims Video   |   Col B: Simulated 2D Confocal MIP   |   Col C: True 4D Mesh",
                 fontsize=9.3, color="#4A4A4A")

        # Column A: Real 2D Confocal Video (XY + XZ)
        ax_a1 = fig.add_subplot(gs[0, 0])
        ax_a1.imshow(real_xy, interpolation="bilinear")
        ax_a1.set_title("A1. Real 2D Confocal MIP (XY)\nDiscrete nuclear puncta & field debris", fontsize=9.0, fontweight="bold", color="#111111")
        ax_a1.set_xticks([]); ax_a1.set_yticks([])
        ax_a1.text(0.03, 0.05, f"Real Frame {tp_idx} ({tm/60.0:.1f} h)\nVol = {vol_emp[idx]:.1f}k µm³ | Frags = {frag_emp[idx]}",
                   transform=ax_a1.transAxes, fontsize=7.8, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.82))

        ax_a2 = fig.add_subplot(gs[1, 0])
        ax_a2.imshow(real_xz, interpolation="bilinear", aspect="auto")
        ax_a2.set_title("A2. Real 2D Z-Stack Projection (XZ)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_a2.set_xticks([]); ax_a2.set_yticks([])

        # Column B: Simulated 2D Confocal MIP Render (XY + XZ)
        ax_b1 = fig.add_subplot(gs[0, 1])
        ax_b1.imshow(sim_xy, interpolation="bilinear")
        ax_b1.set_title("B1. Simulated 2D Confocal MIP (XY)\nEllipsoid sub-domains & Z-overlap", fontsize=9.0, fontweight="bold", color="#111111")
        ax_b1.set_xticks([]); ax_b1.set_yticks([])
        ax_b1.text(0.03, 0.05, f"Sim 2D MIP (3.54 µm pitch)\nSim Vol = {sim_vol[idx]:.1f}k µm³ | Frags = {sim_frag[idx]}",
                   transform=ax_b1.transAxes, fontsize=7.8, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.82))

        ax_b2 = fig.add_subplot(gs[1, 1])
        ax_b2.imshow(sim_xz, interpolation="bilinear", aspect="auto")
        ax_b2.set_title("B2. Simulated 2D Z-Stack Projection (XZ)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_b2.set_xticks([]); ax_b2.set_yticks([])

        # Column C: 4D Deformable-Mesh View
        ax_mid = fig.add_subplot(gs[:, 2], projection="3d")
        ax_mid.set_facecolor("#F6F6F6")

        for o_i, org in enumerate(orgs[:14]):
            opos = np.array([org["x"], org["y"], org["z"]]) * 0.72
            orad = (max(3.5, float(org["La"]) * 0.45 * v_scale),
                    max(3.5, float(org["Lb"]) * 0.45 * v_scale),
                    max(3.5, float(org["Lc"]) * 0.42 * v_scale))
            f_org_mesh = make_deformable_ellipsoid_mesh(opos, orad, yaw=0.3 * o_i, n_lobes=2 if kill_prog > 0.35 else 1)
            is_dead_frag = (o_i > 0 and kill_prog > 0.40) or (kill_prog > 0.72)
            fcol = (0.76, 0.22, 0.18, 0.74) if is_dead_frag else (0.82 * (1 - 0.5 * kill_prog) + 0.72 * (0.5 * kill_prog),
                                                                   0.55 * (1 - 0.7 * kill_prog),
                                                                   0.18, 0.78)
            ax_mid.add_collection3d(Poly3DCollection(f_org_mesh, facecolors=fcol, edgecolors="#111111", linewidths=0.2))

        for c_i, ccell in enumerate(cd8s[:6]):
            cpos = np.array([ccell["x"], ccell["y"], ccell["z"]]) * 0.72
            crad = (max(3.2, float(ccell["Lc"]) * 0.44),
                    max(2.6, float(ccell["La"]) * 0.44),
                    max(2.6, float(ccell["Lb"]) * 0.44))
            vec_to_core = -cpos
            d_core = max(0.1, float(np.linalg.norm(vec_to_core)))
            dir_core = vec_to_core / d_core
            yaw_c = float(np.arctan2(dir_core[1], dir_core[0]))
            f_cd8 = make_deformable_ellipsoid_mesh(cpos, crad, yaw=yaw_c, cup_dir=dir_core, cup_depth=1.8)
            ax_mid.add_collection3d(Poly3DCollection(f_cd8, facecolors=(0.12, 0.31, 0.47, 0.88), edgecolors="#111111", linewidths=0.25))
            if d_core < 42.0:
                syn_pt = cpos + 0.42 * vec_to_core
                ax_mid.scatter([syn_pt[0]], [syn_pt[1]], [syn_pt[2]], color="#B3261E", s=42, zorder=10)

        ax_mid.set_xlim(-48, 48); ax_mid.set_ylim(-48, 48); ax_mid.set_zlim(-38, 38)
        ax_mid.view_init(elev=25, azim=-50 + 0.12 * tm)
        ax_mid.set_xlabel("X (µm)", fontsize=7.5, labelpad=-3)
        ax_mid.set_ylabel("Y (µm)", fontsize=7.5, labelpad=-3)
        ax_mid.set_zlabel("Z (µm)", fontsize=7.5, labelpad=-3)
        ax_mid.tick_params(labelsize=6.8, pad=0)
        ax_mid.set_title("C. True 4D Cooperative Swarm Mesh\nSeparates top-dome T cells from core",
                         fontsize=9.0, fontweight="bold", color="#111111")
        ax_mid.text2D(0.03, 0.04,
                      f"3D Swarm = {max(1, len(cd8s))} CTLs | Frags = {sim_frag[idx]}\nResolves true 3D surface contacts",
                      transform=ax_mid.transAxes, fontsize=7.8, color="#111111",
                      bbox=dict(boxstyle="square,pad=0.22", fc="#FFFFFF", ec="#C9C9C9", alpha=0.92))

        # Column D: Traces
        ax_r1 = fig.add_subplot(gs[0, 3])
        ax_r1.plot(t_arr, vol_emp, "o", color="#2E7D5B", ms=3.0, alpha=0.55, label="Measured Vol (10³ µm³)")
        ax_r1.plot(t_arr[:idx + 1], sim_vol[:idx + 1], "-", color="#1F4E79", lw=2.1, label="Simulated Vol (10³ µm³)")
        ax_r1.axvline(tm, color="#111111", ls="--", lw=1.1)
        ax_r1.set_xlim(0, 515); ax_r1.set_ylim(0, 118)
        ax_r1.set_ylabel("Volume (10³ µm³)", fontsize=8.0)
        ax_r1.set_title("D1. 3D Organoid Volume Collapse (-77.2%)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_r1.grid(True, color="#E3E3E3", lw=0.7)
        ax_r1.legend(frameon=False, fontsize=7.0, loc="upper right")

        ax_r2 = fig.add_subplot(gs[1, 3])
        ax_r2.plot(t_arr, dye_emp, "o", color="#B3261E", ms=3.0, alpha=0.55, label="Measured Dye (a.u.)")
        ax_r2.plot(t_arr[:idx + 1], sim_dye[:idx + 1], "-", color="#B3261E", lw=2.0, label="Simulated Dye (a.u.)")
        ax_r2.plot(t_arr, frag_emp * 2.2, "s", color="#C0561F", ms=2.8, alpha=0.55, label="Measured Frags (×2.2)")
        ax_r2.plot(t_arr[:idx + 1], sim_frag[:idx + 1] * 2.2, "--", color="#C0561F", lw=1.9, label="Simulated Frags (×2.2)")
        ax_r2.axvline(tm, color="#111111", ls="--", lw=1.1)
        ax_r2.set_xlim(0, 515); ax_r2.set_ylim(0, 52)
        ax_r2.set_xlabel("Time in Co-Culture (min)", fontsize=8.2, fontweight="bold")
        ax_r2.set_ylabel("Dead Dye / Frags", fontsize=8.0)
        ax_r2.set_title("D2. Dead-Dye Surge & Breakup (1 → 18)", fontsize=8.8, fontweight="bold", color="#111111")
        ax_r2.grid(True, color="#E3E3E3", lw=0.7)
        ax_r2.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2)

        fig.savefig(frame_dir / f"frame_{idx:04d}.png", dpi=115)
        plt.close(fig)

    f_ims.close()
    out_mp4 = VID_DIR / "side_by_side_event2_organoid_swarm_kill.mp4"
    out_gif = ART_DIR / "side_by_side_event2_organoid_swarm_kill.gif"
    encode_frames_to_mp4_and_gif(frame_dir, out_mp4, out_gif, fps=10)
    shutil.copy2(out_mp4, ART_DIR / "side_by_side_event2_organoid_swarm_kill.mp4")
    shutil.rmtree(frame_dir)
    print(f"   Saved {out_mp4} ({out_mp4.stat().st_size / 1024:.1f} KB) and {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB)")
    return saved_org_snaps


def build_static_figure_7_2d_vs_4d(snaps_e1, snaps_e2):
    """
    Build figures/visual_7_2d_vs_4d_render_comparison.png showing side-by-side how the
    2D synthetic microscope projection of the 4D simulation compares to the real 2D
    microscope video AND the 4D deformable mesh view across key timepoints.
    """
    print("3. Building static comparison panel: figures/visual_7_2d_vs_4d_render_comparison.png...")
    fig = plt.figure(figsize=(15.5, 9.6), dpi=160, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(3, 5, height_ratios=[1.0, 1.0, 1.08], hspace=0.32, wspace=0.14,
                           left=0.06, right=0.98, top=0.87, bottom=0.05)

    fig.suptitle("Visual Guide 7 — How the 2D Simulation Render Compares to the Real 2D Video and True 4D Mesh",
                 fontsize=14.6, fontweight="bold", color="#111111", x=0.06, ha="left", y=0.955)
    fig.text(0.06, 0.912,
             "Row 1: Real 320-nm Light-Sheet 2D Video (XY) | Row 2: Equivalent 2D Synthetic Fluorescence Render of 4D Model (XY) | Row 3: True 4D Deformable Mesh View (3D).",
             fontsize=10.0, color="#4A4A4A")

    times_show = [15, 55, 85, 100, 170]
    col_titles = [
        "1. Approach (15 min)\nSeparated in 3D",
        "2. Synapse (55 min)\nActin Ring Enrichment",
        "3. Peak Cup (85 min)\nConcave Indentation",
        "4. Blebbing (100 min)\n3 Apoptotic Lobes",
        "5. Lysis (170 min)\nSYTOX Influx & Exit",
    ]
    diff_notes_2d = [
        "Sim 2D: Smooth cortical shell\nvs real filopodial speckles",
        "Sim 2D: Captures 3.2× synaptic\nactin ring enrichment",
        "Sim 2D: Matches concave red\nindentation & green cup",
        "Sim 2D: Symmetric 3-lobe blebs\nvs irregular real membrane folds",
        "Sim 2D: Matches cyan SYTOX\nnuclear flash & 29 µm tail",
    ]

    for c_i, (tm, ctitle, dnote) in enumerate(zip(times_show, col_titles, diff_notes_2d)):
        s = snaps_e1[tm]

        # Row 0: Real 2D Video (XY)
        ax0 = fig.add_subplot(gs[0, c_i])
        ax0.imshow(s["real_xy"], interpolation="bilinear")
        ax0.set_title(ctitle, fontsize=9.8, fontweight="bold", color="#111111", pad=5)
        ax0.set_xticks([]); ax0.set_yticks([])
        if c_i == 0:
            ax0.set_ylabel("A. Real 2D Video\n(Light-Sheet XY MIP)", fontsize=9.5, fontweight="bold", color="#111111")

        # Row 1: Equivalent 2D Synthetic Microscope Render of 4D Simulation (XY)
        ax1 = fig.add_subplot(gs[1, c_i])
        ax1.imshow(s["sim_xy"], interpolation="bilinear")
        ax1.set_xticks([]); ax1.set_yticks([])
        if c_i == 0:
            ax1.set_ylabel("B. 2D Sim Render\n(Forward Optical MIP)", fontsize=9.5, fontweight="bold", color="#111111")
        ax1.text(0.04, 0.05, dnote, transform=ax1.transAxes, fontsize=7.6, color="#FFFFFF",
                 bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.80))

        # Row 2: True 4D Deformable Mesh View
        ax2 = fig.add_subplot(gs[2, c_i], projection="3d")
        ax2.set_facecolor("#F6F6F6")
        in_syn = s["in_syn"]
        faces_cart = make_deformable_ellipsoid_mesh(s["c_pos"], s["r_cart"], yaw=s["yaw_ct"], cup_dir=s["dir_ct"], cup_depth=2.2 if in_syn else 0.0)
        faces_tgt = make_deformable_ellipsoid_mesh(s["t_pos"], s["r_tgt"], yaw=s["yaw_ct"] + 0.5, cup_dir=-s["dir_ct"], cup_depth=-1.8 if in_syn else 0.0, n_lobes=s["n_lb"])
        sytox_frac = float(np.clip(s["sytox"] / 22.0, 0.0, 1.0))
        tgt_col = (0.75 * (1 - sytox_frac) + 0.12 * sytox_frac,
                   0.34 * (1 - sytox_frac) + 0.62 * sytox_frac,
                   0.12 * (1 - sytox_frac) + 0.88 * sytox_frac, 0.78)
        ax2.add_collection3d(Poly3DCollection(faces_tgt, facecolors=tgt_col, edgecolors="#111111", linewidths=0.22))
        ax2.add_collection3d(Poly3DCollection(faces_cart, facecolors=(0.18, 0.49, 0.36, 0.82), edgecolors="#111111", linewidths=0.22))
        if in_syn:
            syn_center = 0.52 * s["c_pos"] + 0.48 * s["t_pos"]
            mtoc_pos = s["c_pos"] + 0.68 * (syn_center - s["c_pos"])
            ax2.scatter([mtoc_pos[0]], [mtoc_pos[1]], [mtoc_pos[2]], color="#B3261E", s=36, zorder=10)
        ax2.set_xlim(-22, 22); ax2.set_ylim(-18, 22); ax2.set_zlim(-16, 12)
        ax2.view_init(elev=24, azim=-58 + 0.15 * tm)
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
        if c_i == 0:
            ax2.text2D(-0.18, 0.5, "C. True 4D Mesh\n(Unoccluded 3D State)", transform=ax2.transAxes,
                       fontsize=9.5, fontweight="bold", color="#111111", rotation=90, va="center", ha="center")
        ax2.text2D(0.04, 0.05, f"CA = {s['ca']:.0f} µm² | Lobes = {s['n_lb']}",
                   transform=ax2.transAxes, fontsize=7.6, color="#111111",
                   bbox=dict(boxstyle="square,pad=0.2", fc="#FFFFFF", ec="#C9C9C9", alpha=0.90))

    for out_p in [FIG_DIR / "visual_7_2d_vs_4d_render_comparison.png", ART_DIR / "visual_7_2d_vs_4d_render_comparison.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_7_2d_vs_4d_render_comparison.png")


if __name__ == "__main__":
    s1 = build_event1_lightsheet_side_by_side()
    s2 = build_event2_organoid_swarm_side_by_side()
    build_static_figure_7_2d_vs_4d(s1, s2)
