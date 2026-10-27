#!/usr/bin/env python3
"""
Generate three factual raw-data primer figures for a physicist from measured microscopy files:
  1. `figures/primer_01_lightsheet_chamber_timeline.png`
     Orthogonal max-intensity projections (XY and XZ) per channel across 6 timepoints in
     representative chamber `v01_6_40_120_0-6_hd1_MIP_ROI_39` (opened directly from `.h5` `/raw`),
     with detector ADU ranges, wavelength/fluorophore labels, timestamps, and physical scale bars.
  2. `figures/primer_02_segmentation_overlay.png`
     Same chamber (`v01_6_40_120_0-6_hd1_MIP_ROI_39`, `01677.h5`, t = 60 min) showing raw
     channels (XY and XZ MIPs + central Z/Y slices) alongside the 3D U-Net segmentation mask
     (`01677_predictions.h5`, labels 0..4) overlaid as contours and as discrete class maps.
  3. `figures/primer_03_dataset_scale.png`
     Direct physical comparison across the three microscopy datasets (`ScienceDB 19235` light-sheet,
     `Zenodo 8328441` TEG + breast cancer organoid confocal `.ims`, and `Exp28` CAR-T + B-ALL `.ims`)
     using exact numbers read from `data/raw_data_inventory.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import binary_erosion

from paths import REPO_ROOT, DATA_DIR, FIGURES_DIR as FIG_DIR
from constants import VOXEL_UM, OPTICAL_FWHM_UM
CHAMBERS_DIR = DATA_DIR / "scidb_19235_chambers"
INVENTORY_JSON = DATA_DIR / "raw_data_inventory.json"

# House style tokens (light mode)
PAPER = "#FFFFFF"
PAPER_2 = "#F6F6F6"
INK = "#111111"
TEXT = "#1F1F1F"
MUTED = "#5C5C5C"
RULE = "#E3E3E3"
LINE = "#C9C9C9"
ACCENT = "#0B5FD1"
DANGER = "#B3261E"

PLOT_1 = "#1F4E79"  # deep blue
PLOT_2 = "#C0561F"  # rust orange
PLOT_3 = "#2E7D5B"  # muted green
PLOT_4 = "#6B4C9A"  # muted purple

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

# Voxel pitches documented in repo
DX_RECON_UM = VOXEL_UM        # sqrt(0.0385 um^2) from feilab summary.m line 18
DX_FWHM_UM = OPTICAL_FWHM_UM  # Optical FWHM cited in data_dictionary.md / data_registry.csv


def norm_percentile(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((img.astype(np.float32) - vmin) / max(1.0, float(vmax - vmin)), 0.0, 1.0)


def plot_primer_01_timeline() -> Path:
    """
    Generate `figures/primer_01_lightsheet_chamber_timeline.png`.
    Uses chamber `v01_6_40_120_0-6_hd1_MIP_ROI_39` which has 8 full 3D `.h5` raw stacks + masks.
    Shows 6 `.h5` volumes spanning the interaction:
      - 00237 (Visit 1, t = 0 min, Sweep 3/6)
      - 00240 (Visit 1, t = 0 min, Sweep 6/6: Ch 2 switches from 561 nm mApple-Mem to 637 nm SiR-tubulin)
      - 01677 (Visit 7, t = 60 min, Sweep 3/6)
      - 02637 (Visit 11, t = 100 min, Sweep 3/6)
      - 03117 (Visit 13, t = 120 min, Sweep 3/6)
      - 04797 (Visit 20, t = 190 min, Sweep 3/6)
    """
    ch_dir = CHAMBERS_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_39"
    stems = ["00237", "00240", "01677", "02637", "03117", "04797"]

    # Load all 6 raw 3D volumes (3, 190, 256, 256) uint16 and masks (190, 256, 256) uint8
    volumes = []
    masks = []
    meta = []
    for s in stems:
        with h5py.File(ch_dir / f"{s}.h5", "r") as f_raw:
            raw = f_raw["raw"][:]  # shape (3, 190, 256, 256) uint16
        with h5py.File(ch_dir / f"{s}_predictions.h5", "r") as f_pred:
            pred = f_pred["predictions"][:]  # shape (190, 256, 256) uint8
        idx = int(s)
        visit_idx = (idx - 1) // 240  # 0..19 (40 ROIs * 6 sweeps = 240 volumes/visit)
        sweep_idx = (idx - 1) % 6 + 1
        t_min = visit_idx * 10
        volumes.append(raw)
        masks.append(pred)
        meta.append({
            "stem": s,
            "visit": visit_idx + 1,
            "sweep": sweep_idx,
            "t_min": t_min,
        })

    nz, nx, ny = volumes[0].shape[1], volumes[0].shape[2], volumes[0].shape[3]

    # Fixed display ranges in raw detector ADU per channel so intensity changes over time are honest
    # Baseline camera offset is ~390 ADU (Std ~2.6 ADU in reconsitution_app_eng.mlapp)
    ch_configs = [
        {
            "ch_idx": 0,
            "title": "Ch 0: 405 nm\nSYTOX Blue\n(dead-cell dye)",
            "subtitle": "Display:\n[388, 412] ADU\n(thr > 398 ADU)",
            "vmin": 388.0,
            "vmax": 412.0,
        },
        {
            "ch_idx": 1,
            "title": "Ch 1: 488 nm\nCAR-T F-actin\n(Lifeact-EGFP)",
            "subtitle": "Display:\n[388, 455] ADU",
            "vmin": 388.0,
            "vmax": 455.0,
        },
        {
            "ch_idx": 2,
            "title": "Ch 2: 561 / 637 nm\nNalm6 mApple-Mem\n(Swp 6: SiR-tub)",
            "subtitle": "Display:\n[395, 1100] ADU\n(Swp 6: [395, 520])",
            "vmin": 395.0,
            "vmax": 1100.0,
        },
    ]

    fig = plt.figure(figsize=(16.5, 11.8), dpi=200)
    # 3 channel blocks x 2 projections (XY and XZ) = 6 rows, 6 timepoint columns
    gs = GridSpec(
        6, 6, figure=fig,
        height_ratios=[1.0, 190.0 / 256.0, 1.0, 190.0 / 256.0, 1.0, 190.0 / 256.0],
        hspace=0.22, wspace=0.08,
        left=0.14, right=0.98, top=0.89, bottom=0.07,
    )

    fig.text(
        0.14, 0.955,
        "ScienceDB 19235 Raw Light-Sheet Stacks: Orthogonal Projections Across 6 Timepoints",
        fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.14, 0.930,
        "Chamber: v01_6_40_120_0-6_hd1 / MIP_ROI_39 (Group 1, Untreated high-killing) | "
        "HDF5 /raw shape: (3, 190, 256, 256) uint16 | HDF5 attrs: {} (empty)",
        fontsize=9.5, color=MUTED,
    )

    for col_i, (raw, pred, m) in enumerate(zip(volumes, masks, meta)):
        for ch_b, cfg in enumerate(ch_configs):
            c_idx = cfg["ch_idx"]
            vol_c = raw[c_idx]
            # In h5py C-order, raw has shape (3, Z=190, X=256, Y=256):
            # axis 0 of vol_c is Z (MIPxy), axis 1 of vol_c is Y (MIPxz)
            mip_xy = vol_c.max(axis=0)  # (256, 256)
            mip_xz = vol_c.max(axis=1)  # (190, 256)

            vmin = cfg["vmin"]
            vmax = 520.0 if (c_idx == 2 and m["sweep"] == 6) else cfg["vmax"]

            row_xy = ch_b * 2
            row_xz = ch_b * 2 + 1

            ax_xy = fig.add_subplot(gs[row_xy, col_i])
            ax_xz = fig.add_subplot(gs[row_xz, col_i])

            ax_xy.imshow(norm_percentile(mip_xy, vmin, vmax), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_xz.imshow(norm_percentile(mip_xz, vmin, vmax), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

            # Annotate measured raw uint16 max ADU in corner
            ax_xy.text(
                0.03, 0.95, f"XY  max={int(vol_c.max())} ADU",
                transform=ax_xy.transAxes, ha="left", va="top",
                fontsize=7.2, color="#FFFFFF",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="#111111", alpha=0.65, edgecolor="none"),
            )
            ax_xz.text(
                0.03, 0.93, f"XZ (Z={nz})",
                transform=ax_xz.transAxes, ha="left", va="top",
                fontsize=7.2, color="#FFFFFF",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="#111111", alpha=0.65, edgecolor="none"),
            )

            # Column headers on top row
            if row_xy == 0:
                sweep_note = "Sweep 6/6 (637nm MTOC)" if m["sweep"] == 6 else f"Sweep {m['sweep']}/6 (561nm Mem)"
                ax_xy.set_title(
                    f"t = {m['t_min']} min (Visit {m['visit']}/20)\n{m['stem']}.h5 | {sweep_note}",
                    fontsize=8.5, fontweight="bold", color=INK, pad=6,
                )

            # Row labels on left column
            if col_i == 0:
                ax_xy.set_ylabel("XY (256x256)", fontsize=8, fontweight="bold", color=INK)
                ax_xz.set_ylabel("XZ (190x256)", fontsize=8, fontweight="bold", color=INK)
                fig.text(
                    0.012,
                    ax_xy.get_position().y1 - 0.01,
                    f"{cfg['title']}\n{cfg['subtitle']}",
                    fontsize=8.2, fontweight="bold", color=INK, va="top", ha="left",
                )

            # Scale bar on first panel (both 0.1962 um/vox from summary.m and 0.320 um cited FWHM)
            if row_xy == 0 and col_i == 0:
                bar_um = 10.0
                px_recon = bar_um / DX_RECON_UM  # 50.96 px
                px_fwhm = bar_um / DX_FWHM_UM    # 31.25 px
                y_bar = 236
                x0 = 16
                ax_xy.plot([x0, x0 + px_recon], [y_bar, y_bar], color="#FFFFFF", lw=2.4)
                ax_xy.text(
                    x0, y_bar - 8,
                    "10 um (51 px @ 0.196 um/px)",
                    fontsize=6.6, color="#FFFFFF", fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="#111111", alpha=0.65, edgecolor="none"),
                )
                ax_xy.plot([x0, x0 + px_fwhm], [y_bar + 12, y_bar + 12], color="#CCCCCC", lw=1.6, ls="--")

            ax_xy.set_xticks([])
            ax_xy.set_yticks([])
            ax_xz.set_xticks([])
            ax_xz.set_yticks([])

    fig.text(
        0.14, 0.015,
        "Voxel scale note: HDF5 headers contain no voxel-size attribute (attrs = {}). "
        "Solid scale bar (51.0 px = 10 um) uses dx = 0.1962 um/vox from summary.m (sqrt(0.0385 um^2/vox));\n"
        "dashed scale bar (31.3 px = 10 um) shows dx = 0.320 um cited in data_dictionary.md (optical lateral FWHM).",
        fontsize=8.2, color=MUTED,
    )

    out_path = FIG_DIR / "primer_01_lightsheet_chamber_timeline.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_primer_02_segmentation_overlay() -> Path:
    """
    Generate `figures/primer_02_segmentation_overlay.png`.
    Shows chamber `v01_6_40_120_0-6_hd1_MIP_ROI_39` at timepoint `01677` (t = 60 min, Visit 7/20, Sweep 3/6):
      - Row 1: Orthogonal XY Max-Intensity Projections (`Z` collapsed)
      - Row 2: Orthogonal XZ Max-Intensity Projections (`Y` collapsed)
      - Row 3: Central XY Slice (`z = z_center`) showing non-occluded interior vs cortical/membrane shell labels
    Columns:
      1. Raw Ch 0 (405 nm SYTOX Blue)
      2. Raw Ch 1 (488 nm CAR-T Lifeact-EGFP)
      3. Raw Ch 2 (561 nm Nalm6 mApple-Mem)
      4. Raw Composite + 3D U-Net Label Contours Overlaid
      5. Discrete 3D U-Net Mask Labels (`0..4`)
    """
    ch_dir = CHAMBERS_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_39"
    stem = "01677"
    with h5py.File(ch_dir / f"{stem}.h5", "r") as f_raw:
        raw = f_raw["raw"][:]  # (3, 190, 256, 256) uint16
    with h5py.File(ch_dir / f"{stem}_predictions.h5", "r") as f_pred:
        pred = f_pred["predictions"][:]  # (190, 256, 256) uint8

    nz, nx, ny = pred.shape
    vals, cnts = np.unique(pred, return_counts=True)
    vox_counts = {int(v): int(c) for v, c in zip(vals, cnts)}

    # Find centroid Z of the interacting pair for the central slice row
    zyx_cells = np.argwhere(pred > 0)
    z_mid = int(np.median(zyx_cells[:, 0]))

    # House-style data palette for discrete labels 0..4
    label_colors = [
        "#F6F6F6",  # 0: Background
        "#8FBCE6",  # 1: CAR-T body (light blue)
        "#E9A271",  # 2: Target interior (light orange)
        "#1F4E79",  # 3: CAR-T dense F-actin (deep blue, --plot-1)
        "#C0561F",  # 4: Target membrane (rust orange, --plot-2)
    ]
    cmap_labels = ListedColormap(label_colors)
    norm_labels = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_labels.N)

    fig = plt.figure(figsize=(16.2, 10.6), dpi=200)
    gs = GridSpec(
        3, 5, figure=fig,
        height_ratios=[1.0, 190.0 / 256.0, 1.0],
        hspace=0.24, wspace=0.10,
        left=0.07, right=0.98, top=0.84, bottom=0.14,
    )

    fig.text(
        0.07, 0.945,
        "3D U-Net Semantic Segmentation Overlay on Raw Light-Sheet Data",
        fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.07, 0.918,
        f"Chamber: v01_6_40_120_0-6_hd1 / MIP_ROI_39 | File: {stem}.h5 & {stem}_predictions.h5 "
        f"(Visit 7/20, t = 60 min, Sweep 3/6) | Shape: (190, 256, 256)",
        fontsize=9.5, color=MUTED,
    )

    views = [
        ("XY Max Projection\n(max over Z=0..189)", "mip_xy"),
        ("XZ Max Projection\n(max over Y=0..255)", "mip_xz"),
        (f"Central XY Slice (z = {z_mid})\n(shows interior labels 1 & 2)", "slice_xy"),
    ]

    col_titles = [
        "Raw Ch 0: 405 nm\n(SYTOX Blue, [388, 412] ADU)",
        "Raw Ch 1: 488 nm\n(Lifeact-EGFP, [388, 455] ADU)",
        "Raw Ch 2: 561 nm\n(mApple-Mem, [395, 1100] ADU)",
        "Raw 488+561 Composite\n+ U-Net Label Contours",
        "U-Net `/predictions` Mask\n(Discrete uint8 Labels 0..4)",
    ]

    for row_i, (row_label, view_mode) in enumerate(views):
        if view_mode == "mip_xy":
            c0 = norm_percentile(raw[0].max(axis=0), 388.0, 412.0)
            c1 = norm_percentile(raw[1].max(axis=0), 388.0, 455.0)
            c2 = norm_percentile(raw[2].max(axis=0), 395.0, 1100.0)
            lbl_2d = pred.max(axis=0)
            m_c1 = (pred == 1).any(axis=0)
            m_c2 = (pred == 2).any(axis=0)
            m_c3 = (pred == 3).any(axis=0)
            m_c4 = (pred == 4).any(axis=0)
        elif view_mode == "mip_xz":
            c0 = norm_percentile(raw[0].max(axis=1), 388.0, 412.0)
            c1 = norm_percentile(raw[1].max(axis=1), 388.0, 455.0)
            c2 = norm_percentile(raw[2].max(axis=1), 395.0, 1100.0)
            lbl_2d = pred.max(axis=1)
            m_c1 = (pred == 1).any(axis=1)
            m_c2 = (pred == 2).any(axis=1)
            m_c3 = (pred == 3).any(axis=1)
            m_c4 = (pred == 4).any(axis=1)
        else:
            c0 = norm_percentile(raw[0, z_mid], 388.0, 412.0)
            c1 = norm_percentile(raw[1, z_mid], 388.0, 455.0)
            c2 = norm_percentile(raw[2, z_mid], 395.0, 750.0)
            lbl_2d = pred[z_mid]
            m_c1 = lbl_2d == 1
            m_c2 = lbl_2d == 2
            m_c3 = lbl_2d == 3
            m_c4 = lbl_2d == 4

        for col_i in range(5):
            ax = fig.add_subplot(gs[row_i, col_i])
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9.0, fontweight="bold", color=INK, pad=6)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=8.5, fontweight="bold", color=INK)

            if col_i == 0:
                ax.imshow(c0, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            elif col_i == 1:
                ax.imshow(c1, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            elif col_i == 2:
                ax.imshow(c2, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            elif col_i == 3:
                # Grayscale composite of 488 + 561 with crisp color-coded label contours
                comp = np.clip(0.55 * c1 + 0.75 * c2, 0.0, 1.0)
                ax.imshow(comp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                for mask_2d, col_hex, lw in [
                    (m_c4, PLOT_2, 1.4),
                    (m_c2, "#E9A271", 1.0),
                    (m_c3, "#7FB3FF", 1.4),
                    (m_c1, "#FFFFFF", 0.9),
                ]:
                    if mask_2d.any():
                        edge = mask_2d & (~binary_erosion(mask_2d, iterations=1))
                        yy, xx = np.where(edge)
                        ax.scatter(xx, yy, s=0.6, c=col_hex, marker="s", linewidths=0)
            else:
                ax.imshow(lbl_2d, cmap=cmap_labels, norm=norm_labels, interpolation="nearest")

            if row_i == 0 and col_i == 0:
                px_recon = 10.0 / DX_RECON_UM
                ax.plot([16, 16 + px_recon], [236, 236], color="#FFFFFF", lw=2.4)
                ax.text(
                    16, 226, "10 um (0.196 um/px)",
                    fontsize=6.8, color="#FFFFFF", fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="#111111", alpha=0.65, edgecolor="none"),
                )

            ax.set_xticks([])
            ax.set_yticks([])

    # Legend naming each label from repo documentation (`step2_instance_seg.m` lines 38-41)
    legend_specs = [
        (0, label_colors[0], f"Label 0: Background (outside cells) — {vox_counts.get(0, 0):,d} voxels [implicit in step2_instance_seg.m]"),
        (1, label_colors[1], f"Label 1: CAR-T cell body (`t_mask = I==1`) — {vox_counts.get(1, 0):,d} voxels [step2_instance_seg.m line 38]"),
        (2, label_colors[2], f"Label 2: Target Nalm6 interior (`b_mask = I==2`) — {vox_counts.get(2, 0):,d} voxels [step2_instance_seg.m line 39]"),
        (3, label_colors[3], f"Label 3: CAR-T dense F-actin (`actin = I==3`) — {vox_counts.get(3, 0):,d} voxels [step2_instance_seg.m line 40]"),
        (4, label_colors[4], f"Label 4: Target Nalm6 membrane (`mem = I==4`) — {vox_counts.get(4, 0):,d} voxels [step2_instance_seg.m line 41]"),
    ]
    patches = [
        mpatches.Patch(facecolor=col, edgecolor=INK, linewidth=0.8, label=txt)
        for _, col, txt in legend_specs
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=2,
        frameon=True,
        facecolor=PAPER_2,
        edgecolor=LINE,
        fontsize=8.3,
        title="Documented 3D U-Net Label Meanings (`external/feilab_CART_analysis/main/analysis/step2_instance_seg.m` lines 38-41; all 5 labels 0..4 documented)",
        title_fontsize=8.6,
    )

    out_path = FIG_DIR / "primer_02_segmentation_overlay.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_primer_03_dataset_scale() -> Path:
    """
    Generate `figures/primer_03_dataset_scale.png`.
    Reads `data/raw_data_inventory.json` and compares the 3 datasets across:
      1. Voxel pitch (dx, dy, dz in um)
      2. 1:1 true-scale Field of View schematic (XY and XZ in um)
      3. Frame / sampling interval (s)
      4. Total recording duration (h)
      5. Sample size (cells / tracks / 3D surface events)
    """
    with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
        inv = json.load(f)

    ds_list = inv["dataset_scale_comparison"]
    d_ls, d_org, d_exp28 = ds_list[0], ds_list[1], ds_list[2]

    fig = plt.figure(figsize=(16.0, 9.4), dpi=200)
    gs = GridSpec(
        2, 3, figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.11, right=0.98, top=0.86, bottom=0.09,
    )

    fig.text(
        0.11, 0.945,
        "Physical Scale & Sampling Comparison Across the Three Raw Microscopy Datasets",
        fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.11, 0.915,
        "All values read from data/raw_data_inventory.json (data/scidb_19235_chambers/, "
        "data/ims_metadata_20190626.json, data/ims_metadata_exp28.json)",
        fontsize=9.5, color=MUTED,
    )

    names = [
        "ScienceDB 19235\nLight-Sheet\n(CAR-T + Nalm6)",
        "Zenodo 8328441\nConfocal Organoid\n(TEG + BC PDO)",
        "Zenodo 8328441\nConfocal Exp28\n(CAR-T + B-ALL)",
    ]
    colors = [PLOT_1, PLOT_2, PLOT_3]
    y_pos = np.arange(len(names))

    # Panel A: Voxel Pitch (dx, dy, dz in um)
    ax_a = fig.add_subplot(gs[0, 0])
    dx_vals = [
        d_ls["voxel_size_um"]["dx"],
        d_org["voxel_size_um"]["dx"],
        d_exp28["voxel_size_um"]["dx"],
    ]
    dz_vals = [
        d_ls["voxel_size_um"]["dz"],
        d_org["voxel_size_um"]["dz"],
        d_exp28["voxel_size_um"]["dz"],
    ]
    h_bar = 0.32
    ax_a.barh(y_pos - h_bar / 2, dx_vals, height=h_bar, color=PLOT_1, edgecolor=INK, lw=0.7, label="Lateral dx = dy (um)")
    ax_a.barh(y_pos + h_bar / 2, dz_vals, height=h_bar, color=PLOT_2, edgecolor=INK, lw=0.7, label="Axial dz (um)")
    # Also mark the 0.320 um optical FWHM on ScienceDB bar
    ax_a.plot([0.320], [y_pos[0] - h_bar / 2], marker="d", color=INK, markersize=5, label="Optical FWHM (0.320 um)")
    for i, (vx, vz) in enumerate(zip(dx_vals, dz_vals)):
        x_txt = 0.40 if i == 0 else vx + 0.08
        extra = " (0.320 FWHM)" if i == 0 else ""
        ax_a.text(x_txt, y_pos[i] - h_bar / 2, f"{vx:.3f} um{extra}", va="center", fontsize=8.0, color=TEXT)
        ax_a.text(vz + 0.08, y_pos[i] + h_bar / 2, f"{vz:.3f} um", va="center", fontsize=8.0, color=TEXT)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names, fontsize=8.5, fontweight="bold")
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 4.9)
    ax_a.set_xlabel("Voxel size (um)", fontsize=9, fontweight="bold")
    ax_a.set_title("A. Voxel Pitch (dx, dy vs dz)", fontsize=10, fontweight="bold", loc="left")
    ax_a.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor=PAPER_2, edgecolor=LINE)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: True-Scale 2D Field-of-View Schematic (XY and XZ in um)
    ax_b = fig.add_subplot(gs[0, 1])
    fov_specs = [
        ("Zenodo Confocal Organoid\n708.5 x 708.5 x 65.1 um", d_org["fov_um_XYZ"][0], d_org["fov_um_XYZ"][1], PLOT_2, "-"),
        ("Zenodo Confocal Exp28\n466.0 x 465.9 x 27.3 um", d_exp28["fov_um_XYZ"][0], d_exp28["fov_um_XYZ"][1], PLOT_3, "--"),
        ("ScienceDB Light-Sheet (at 0.320 um)\n81.9 x 81.9 x 60.8 um", d_ls["fov_um_XYZ_at_0320"][0], d_ls["fov_um_XYZ_at_0320"][1], PLOT_1, ":"),
        ("ScienceDB Light-Sheet (at 0.196 um)\n50.2 x 50.2 x 37.3 um", d_ls["fov_um_XYZ_at_01962"][0], d_ls["fov_um_XYZ_at_01962"][1], PLOT_1, "-"),
    ]
    for lbl, wx, wy, col, ls in fov_specs:
        rect = mpatches.Rectangle((0, 0), wx, wy, fill=False, edgecolor=col, linewidth=1.8, linestyle=ls, label=lbl)
        ax_b.add_patch(rect)
    ax_b.set_xlim(-20, 780)
    ax_b.set_ylim(-20, 780)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("X physical extent (um)", fontsize=9, fontweight="bold")
    ax_b.set_ylabel("Y physical extent (um)", fontsize=9, fontweight="bold")
    ax_b.set_title("B. 1:1 True-Scale XY Field of View", fontsize=10, fontweight="bold", loc="left")
    ax_b.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor=PAPER_2, edgecolor=LINE)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: Frame Interval (s, log scale)
    ax_c = fig.add_subplot(gs[0, 2])
    cadence_labels = [
        "ScienceDB Burst\n(6 sweeps/visit)",
        "ScienceDB Visit\n(20 visits/chamber)",
        "Zenodo Organoid\n(669 frames)",
        "Zenodo Exp28\n(480 frames)",
    ]
    cadence_s = [
        d_ls["frame_interval_s_burst"],
        d_ls["visit_interval_s"],
        d_org["frame_interval_s"],
        d_exp28["frame_interval_s"],
    ]
    cadence_cols = [PLOT_1, PLOT_1, PLOT_2, PLOT_3]
    yc = np.arange(len(cadence_labels))
    ax_c.barh(yc, cadence_s, height=0.52, color=cadence_cols, edgecolor=INK, lw=0.7)
    ax_c.set_xscale("log")
    for i, val in enumerate(cadence_s):
        ax_c.text(val * 1.15, yc[i], f"{val:.1f} s ({val/60.0:.2f} min)", va="center", fontsize=8.0, color=TEXT)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(cadence_labels, fontsize=8.2, fontweight="bold")
    ax_c.invert_yaxis()
    ax_c.set_xlim(1.0, 4500.0)
    ax_c.set_xlabel("Sampling interval (s, log scale)", fontsize=9, fontweight="bold")
    ax_c.set_title("C. Temporal Cadence (Frame Interval)", fontsize=10, fontweight="bold", loc="left")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D: Total Recording Duration (h)
    ax_d = fig.add_subplot(gs[1, 0])
    dur_h = [
        d_ls["total_duration_h"],
        d_org["total_duration_h"],
        d_exp28["total_duration_h"],
    ]
    ax_d.barh(y_pos, dur_h, height=0.48, color=colors, edgecolor=INK, lw=0.7)
    for i, val in enumerate(dur_h):
        note = " (20 visits; 3-5 h cohort)" if i == 0 else f" ({[120, 669, 480][i]} frames)"
        ax_d.text(val + 0.5, y_pos[i], f"{val:.2f} h{note}", va="center", fontsize=8.0, color=TEXT)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(names, fontsize=8.5, fontweight="bold")
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 34)
    ax_d.set_xlabel("Total recording duration (hours)", fontsize=9, fontweight="bold")
    ax_d.set_title("D. Total Recording Span (Hours)", fontsize=10, fontweight="bold", loc="left")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # Panel E: Tracked Cells / Chambers / Events
    ax_e = fig.add_subplot(gs[1, 1])
    event_labels = [
        "ScienceDB Curated\nSingle-Cell Pairs",
        "ScienceDB Deposited\nMicrochambers",
        "Zenodo Well 4\nCD8 + CD4 Tracks",
        "Zenodo All 6 Wells\nTotal T-Cell Tracks",
        "Zenodo Well 4\nTracked Organoids",
        "Zenodo Exp28\nCD4 + CD8 Tracks",
    ]
    event_counts = [
        d_ls["n_cells_or_events"],                           # 419
        d_ls["n_deposited_chambers"],                        # 502
        d_org["n_cells_or_events"] + d_org["n_cd4_tracks_in_ims"],  # 1171 + 668 = 1839
        d_org["n_all_6_wells_tracks"],                       # 6931
        d_org["n_tracked_organoids"],                        # 21
        d_exp28["n_cells_or_events"],                        # 195
    ]
    event_cols = [PLOT_1, PLOT_1, PLOT_2, PLOT_2, PLOT_2, PLOT_3]
    ye = np.arange(len(event_labels))
    ax_e.barh(ye, event_counts, height=0.55, color=event_cols, edgecolor=INK, lw=0.7)
    ax_e.set_xscale("log")
    for i, val in enumerate(event_counts):
        ax_e.text(val * 1.15, ye[i], f"{val:,d}", va="center", fontsize=8.0, color=TEXT)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels(event_labels, fontsize=7.8, fontweight="bold")
    ax_e.invert_yaxis()
    ax_e.set_xlim(10, 35000)
    ax_e.set_xlabel("Count (log scale)", fontsize=9, fontweight="bold")
    ax_e.set_title("E. Sample Size (Cells / Chambers / Tracks)", fontsize=10, fontweight="bold", loc="left")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # Panel F: Quantitative Summary Table
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    ax_f.set_title("F. Inventory Summary Table", fontsize=10, fontweight="bold", loc="left")
    table_rows = [
        ["Metric", "ScienceDB 19235", "Zenodo Organoid", "Zenodo Exp28"],
        ["Grid (X,Y,Z)", "256x256x166..190", "200x200x62", "512x512x30"],
        ["dx, dy (um)", "0.196 (0.32 FWHM)", "3.5425", "0.910"],
        ["dz (um)", "0.196 (0.32 FWHM)", "1.0500", "0.910"],
        ["FOV XY (um)", "50.2x50.2 (81.9)", "708.5 x 708.5", "466.0 x 465.9"],
        ["FOV Z (um)", "32.6..37.3 (60.8)", "65.1", "27.3"],
        ["Channels", "3 (405/488/561)", "9 (RS1-4, masks)", "7 (4 raw, 3 mask)"],
        ["Cadence", "2.5s / 10m visit", "119.7 s (2.0m)", "168.6 s (2.81m)"],
        ["Frames / Dur.", "120 / 3.3h (3-5h)", "669 / 22.21 h", "480 / 22.48 h"],
        ["Cells / Events", "419 pairs (502)", "1,171 CD8; 21 PDO", "195 CAR-T; 114k"],
    ]
    tbl = ax_f.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        colWidths=[0.22, 0.28, 0.25, 0.25],
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(PAPER_2)
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(PAPER)
            cell.set_text_props(color=TEXT)

    out_path = FIG_DIR / "primer_03_dataset_scale.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    if not INVENTORY_JSON.exists():
        raise FileNotFoundError(
            f"Missing {INVENTORY_JSON}; run `python3 src/inventory_raw_data.py` first."
        )
    p1 = plot_primer_01_timeline()
    p2 = plot_primer_02_segmentation_overlay()
    p3 = plot_primer_03_dataset_scale()
    for p in [p1, p2, p3]:
        print(f"Wrote figure: {p.relative_to(REPO_ROOT)} ({p.stat().st_size:,d} bytes)")


if __name__ == "__main__":
    main()
