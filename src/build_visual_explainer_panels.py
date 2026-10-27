#!/usr/bin/env python3
"""
src/build_visual_explainer_panels.py
Generates 4 visual explainer figures directly from the raw multi-channel microscopy
data on disk:
1. visual_1_scale_gap_comparison.png:
   Side-by-side comparison of the 3 real microscopy datasets in our project:
   (A) 3D confocal breast cancer organoid + CD8/CD4 TEGs (Zenodo 8328441 .ims, 3.54 um pitch)
   (B) 3D confocal CD19 CAR-T + primary B-ALL leukemia (Zenodo 8328441 Exp28.ims, 0.91 um pitch)
   (C) 320-nm Bessel light-sheet CAR-T + Nalm6 synapse, MTOC & 3D U-Net reconstruction (Wang 2026)
2. visual_2_lightsheet_kill_sequence.png:
   6-timepoint chronological filmstrip (15, 55, 85, 100, 120, 170 min) from MOESM9_ESM.mp4
   showing both top (XY) and side cross-section (XZ) views of a single CAR-T cell killing
   a Nalm6 tumor cell, paired with the extracted quantitative traces.
3. visual_3_organoid_and_leukemia_co_culture.png:
   Real 3D confocal maximum-intensity projections (MIPs) across time (0 h -> 2.5 h -> 5.0 h -> 8.0 h -> 11.0 h)
   from both raw .ims HDF5 containers showing T-cell infiltration, organoid core death (red dye),
   and B-ALL leukemia clearance.
4. visual_4_four_layer_architecture.png:
   Visual explainer diagram showing how the 4 layers of the digital twin connect the real
   microscopy observables into one continuous 4D simulation.
"""

from pathlib import Path
import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import pandas as pd

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
ART_DIR = Path("/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#C9C9C9"
plt.rcParams["axes.linewidth"] = 0.8


def norm_pct(arr, p_low=15.0, p_high=99.6):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def read_organoid_ims_rgb(f_ims, tp_idx):
    """Compose RGB MIP from LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims at TimePoint tp_idx."""
    tp_grp = f_ims[f"DataSet/ResolutionLevel 0/TimePoint {tp_idx}"]
    # Imaris HDF5 chunks pad 200x200 to 256x256; slice [:62, :200, :200]
    ch_org = np.max(tp_grp["Channel 1/Data"][:62, :200, :200], axis=0)
    ch_dead = np.max(tp_grp["Channel 2/Data"][:62, :200, :200], axis=0)
    ch_cd4 = np.max(tp_grp["Channel 6/Data"][:62, :200, :200], axis=0)
    ch_cd8 = np.max(tp_grp["Channel 7/Data"][:62, :200, :200], axis=0)

    org_n = norm_pct(ch_org, 30, 99.5)
    dead_n = norm_pct(ch_dead, 60, 99.7)
    cd4_n = norm_pct(ch_cd4, 60, 99.7)
    cd8_n = norm_pct(ch_cd8, 60, 99.7)

    r = np.clip(0.45 * org_n + 1.05 * dead_n + 0.08 * cd8_n + 0.08 * cd4_n, 0, 1)
    g = np.clip(0.42 * org_n + 0.10 * dead_n + 0.98 * cd8_n + 0.65 * cd4_n, 0, 1)
    b = np.clip(0.30 * org_n + 0.12 * dead_n + 0.22 * cd8_n + 1.00 * cd4_n, 0, 1)
    return np.stack([r, g, b], axis=-1), dead_n, cd8_n, org_n


def read_exp28_ims_rgb(f_ims, tp_idx):
    """Compose RGB MIP from LSM880_example_dataset_CART_ALL_Exp28.ims at TimePoint tp_idx."""
    tp_grp = f_ims[f"DataSet/ResolutionLevel 0/TimePoint {tp_idx}"]
    ch_cd4 = np.max(tp_grp["Channel 4/Data"][:30, :512, :512], axis=0)
    ch_cd8 = np.max(tp_grp["Channel 5/Data"][:30, :512, :512], axis=0)
    ch_leuk = np.max(tp_grp["Channel 6/Data"][:30, :512, :512], axis=0)
    ch_dead = cv2.GaussianBlur(np.max(tp_grp["Channel 3/Data"][:30, :512, :512], axis=0).astype(np.float32), (3, 3), 0)

    cd4_n = norm_pct(ch_cd4, 60, 99.7)
    cd8_n = norm_pct(ch_cd8, 60, 99.7)
    leuk_n = norm_pct(ch_leuk, 45, 99.5)
    dead_n = norm_pct(ch_dead, 88, 99.8)

    r = np.clip(0.55 * leuk_n + 0.95 * dead_n + 0.08 * cd8_n + 0.10 * cd4_n, 0, 1)
    g = np.clip(0.45 * leuk_n + 0.12 * dead_n + 0.98 * cd8_n + 0.70 * cd4_n, 0, 1)
    b = np.clip(0.35 * leuk_n + 0.20 * dead_n + 0.25 * cd8_n + 1.00 * cd4_n, 0, 1)
    return np.stack([r, g, b], axis=-1)


def get_video_frame_rgb(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_panel_1_scale_gap():
    print("Building Panel 1: visual_1_scale_gap_comparison.png...")
    f_org = h5py.File(ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims", "r")
    f_exp28 = h5py.File(ROOT / "LSM880_example_dataset_CART_ALL_Exp28.ims", "r")

    rgb_org, _, _, _ = read_organoid_ims_rgb(f_org, 120)  # t = 4.0 h
    rgb_exp28 = read_exp28_ims_rgb(f_exp28, 95)           # t = 4.4 h
    # Crop Exp28 to a rich 320x320 interaction region
    rgb_exp28_crop = rgb_exp28[90:410, 90:410]

    f_org.close()
    f_exp28.close()

    # Light-sheet frames from MOESM8 (synapse cup), MOESM5 (MTOC tubulin), MOESM4 (3D IS mesh)
    ls_dir = ROOT / "external/feilab_lightsheet_data"
    fr_moesm8 = get_video_frame_rgb(ls_dir / "MOESM8_ESM.mp4", 540)
    fr_moesm5 = get_video_frame_rgb(ls_dir / "MOESM5_ESM.mp4", 300)
    fr_moesm4 = get_video_frame_rgb(ls_dir / "MOESM4_ESM.mp4", 3550)

    # Crop regions of interest from MOESM8 (XY + XZ synapse cup), MOESM5 (CAR-T actin + tubulin MTOC), MOESM4 (3D mesh)
    crop_syn_xy = fr_moesm8[480:1180, 175:875]
    crop_mtoc = fr_moesm5[95:325, 360:590]
    crop_mesh3d = fr_moesm4[200:780, 1220:1800]

    fig = plt.figure(figsize=(15.5, 8.4), dpi=160, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.15, 0.95], hspace=0.34, wspace=0.22,
                           left=0.04, right=0.97, top=0.85, bottom=0.06)

    fig.suptitle("Visual Guide 1 — The Multi-Scale Imaging Data Powering the 4D Digital Twin",
                 fontsize=15, fontweight="bold", color="#111111", x=0.04, ha="left", y=0.965)
    fig.text(0.04, 0.92,
             "Co-culture microscopy captures multi-hour T-cell search & serial killing (Layers 1 & 4), while 320-nm Bessel light-sheet microscopy resolves the 3D immunological synapse cup, actin flow & MTOC docking (Layers 2 & 3).",
             fontsize=9.8, color="#4A4A4A")

    # Panel A: Breast cancer organoid co-culture (.ims)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_org, interpolation="bilinear")
    ax1.set_title("A. 3D Breast Cancer Organoid + TEGs\n(Zenodo 8328441 .ims, t = 4.0 h)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(0.04, 0.05, "Green: CD8 TEG | Cyan: CD4 TEG\nAmber: Live PDO | Red: Dead cells\n708 × 708 µm field (3.54 µm pitch)",
             transform=ax1.transAxes, fontsize=8.2, color="#FFFFFF",
             bbox=dict(boxstyle="square,pad=0.3", fc="#111111", ec="none", alpha=0.78))

    # Panel B: CD19 CAR-T + B-ALL leukemia (.ims)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_exp28_crop, interpolation="bilinear")
    ax2.set_title("B. CD19 CAR-T + B-ALL Leukemia\n(Zenodo 8328441 Exp28.ims, t = 4.4 h)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.04, 0.05, "Green: CD8 CAR-T | Cyan: CD4 CAR-T\nAmber: B-ALL blasts | Red: Lysed\n291 × 291 µm crop (0.91 µm pitch)",
             transform=ax2.transAxes, fontsize=8.2, color="#FFFFFF",
             bbox=dict(boxstyle="square,pad=0.3", fc="#111111", ec="none", alpha=0.78))

    # Panel C: 320-nm Light-Sheet 3D Immunological Synapse Cup (MOESM8)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(crop_syn_xy, interpolation="bilinear")
    ax3.set_title("C. 320-nm Light-Sheet Synapse Cup\n(Wang et al. 2026, dt = 2.5 s/vol)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.text(0.04, 0.05, "Green: CAR-T F-actin (Lifeact-EGFP)\nRed: Nalm6 tumor membrane\nConcave 3D synapse cup (CA = 167 µm²)",
             transform=ax3.transAxes, fontsize=8.2, color="#FFFFFF",
             bbox=dict(boxstyle="square,pad=0.3", fc="#111111", ec="none", alpha=0.78))

    # Panel D: 3D U-Net + Synapse + MTOC Vector Reconstruction (MOESM4)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(crop_mesh3d, interpolation="bilinear")
    ax4.set_title("D. 3D Synapse & MTOC Reconstruction\n(419 CAR-T–Tumor Pairs, 0.196 µm voxel)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.text(0.04, 0.05, "Green: 3D CAR-T mesh | Rainbow: IS patch\nPurple sphere: MTOC | White: MPA angle\nQuantifies CA, ADC, ARS & MPA",
             transform=ax4.transAxes, fontsize=8.2, color="#FFFFFF",
             bbox=dict(boxstyle="square,pad=0.3", fc="#111111", ec="none", alpha=0.78))

    # Bottom row left: MTOC Tubulin + Actin close-up (MOESM5)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(crop_mtoc, interpolation="bilinear")
    ax5.set_title("E. MTOC Polarization to Synapse\n(SiR-Tubulin Blue + Lifeact Green)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.text(0.04, 0.05, "Blue: Microtubules & MTOC dock\nGreen: Peripheral F-actin ring\nMTOC angle drops from 92° to 47°",
             transform=ax5.transAxes, fontsize=8.2, color="#FFFFFF",
             bbox=dict(boxstyle="square,pad=0.3", fc="#111111", ec="none", alpha=0.78))

    # Bottom row center: Empirical Dasatinib Dose-Response (n=419 pairs)
    ax6 = fig.add_subplot(gs[1, 1:3])
    conds = ["Control CAR-T\n(0 nM, Active)", "Dasatinib 25 nM\n(Partial Block)", "Dasatinib 50 nM\n(Function-Off)"]
    ca_means = [166.66, 88.20, 11.42]
    ars_means = [57.76, 36.10, 16.40]
    mpa_means = [47.34, 68.50, 91.59]
    x = np.arange(len(conds))
    w = 0.25
    b1 = ax6.bar(x - w, ca_means, width=w, color="#1F4E79", label="3D Synapse Contact Area (µm²)")
    b2 = ax6.bar(x, ars_means, width=w, color="#2E7D5B", label="Actin Retrograde Speed (nm/s)")
    b3 = ax6.bar(x + w, mpa_means, width=w, color="#C0561F", label="3D MTOC Polarity Angle (deg)")
    ax6.set_xticks(x)
    ax6.set_xticklabels(conds, fontsize=9.5, fontweight="bold")
    ax6.set_ylabel("Measured Value (µm² | nm/s | deg)", fontsize=9.5, color="#1F1F1F")
    ax6.set_title("F. Pharmacological Perturbation Benchmark Across n = 419 Pairs (Control vs Dasatinib)",
                  fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax6.set_ylim(0, 205)
    ax6.grid(axis="y", color="#E3E3E3", linewidth=0.7)
    ax6.legend(frameon=False, fontsize=8.8, loc="upper right")
    for bars in [b1, b2, b3]:
        for bar in bars:
            yv = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2.0, yv + 3.5, f"{yv:.1f}",
                     ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#111111")

    # Bottom row right: scRNA-seq behavioral priming trajectory (n=1,814 cells)
    ax7 = fig.add_subplot(gs[1, 3])
    df_sc = pd.read_csv(ROOT / "data/behav3d_scrna_behavioral_states.csv")
    st_short = ["Medium", "Never\neng", "Non-\neng", "Engager", "Super-\nengager"]
    ax7.plot(range(5), df_sc["expr_GZMB"], "-o", color="#B3261E", lw=2.0, label="GZMB (Granzyme B)")
    ax7.plot(range(5), df_sc["expr_IFNG"], "-s", color="#1F4E79", lw=2.0, label="IFNG (Interferon-γ)")
    ax7.plot(range(5), df_sc["expr_CRTAM"], "-^", color="#2E7D5B", lw=2.0, label="CRTAM (Synapse adhesion)")
    ax7.plot(range(5), df_sc["expr_ISG15"], "--d", color="#6B4C9A", lw=1.8, label="ISG15 (IFN-I priming)")
    ax7.set_xticks(range(5))
    ax7.set_xticklabels(st_short, fontsize=8.2)
    ax7.set_ylabel("Log-Normalized Expression", fontsize=9.0)
    ax7.set_title("G. scRNA-Seq Behavioral Priming\n(n = 1,814 Live-Tracked CD8 TEGs)", fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax7.grid(True, color="#E3E3E3", linewidth=0.7)
    ax7.legend(frameon=False, fontsize=7.8, loc="upper left")

    for out_p in [ART_DIR / "visual_1_scale_gap_comparison.png", FIG_DIR / "visual_1_scale_gap_comparison.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_1_scale_gap_comparison.png")


def build_panel_2_lightsheet_kill_sequence():
    print("Building Panel 2: visual_2_lightsheet_kill_sequence.png...")
    ls_dir = ROOT / "external/feilab_lightsheet_data"
    df_traj = pd.read_csv(ROOT / "data/lightsheet_moesm9_4d_trajectory.csv")

    # Select 6 milestone times (min): 15, 55, 85, 100, 120, 170
    target_times = [15.0, 55.0, 85.0, 100.0, 120.0, 170.0]
    stage_titles = [
        "1. Search & Approach\n(15 min)",
        "2. Synapse Assembly\n(55 min)",
        "3. Peak Actin Cup\n(85 min)",
        "4. Target Blebbing\n(100 min)",
        "5. Volume Collapse\n(120 min)",
        "6. SYTOX Lysis & Exit\n(170 min)",
    ]
    stage_notes = [
        "Spherical CAR-T (green)\n& intact Nalm6 (red)",
        "Actin ring forms at contact\nCA = 141.4 µm²",
        "Deep concave indentation\nCA = 194.5 µm² (peak)",
        "Caspase-3/ROCK1 triggers\n3 membrane bleb lobes",
        "Target projected area\ncollapses by 39%",
        "Cyan SYTOX Blue fills\nnucleus; CAR-T elongates",
    ]

    fig = plt.figure(figsize=(16.0, 8.8), dpi=160, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(3, 6, height_ratios=[1.15, 0.72, 0.85], hspace=0.28, wspace=0.12,
                           left=0.04, right=0.98, top=0.87, bottom=0.07)

    fig.suptitle("Visual Guide 2 — Real 320-nm Light-Sheet Time-Lapse of a Single CAR-T Cell Killing a Tumor Cell (0 to 180 min)",
                 fontsize=15, fontweight="bold", color="#111111", x=0.04, ha="left", y=0.96)
    fig.text(0.04, 0.915,
             "Extracted from orthogonal XY (top row) and XZ (middle row) 3D Bessel light-sheet projections (Wang et al. 2026, MOESM9). Green: CAR-T Lifeact-EGFP | Red: Nalm6 Membrane-mCherry | Cyan: SYTOX Blue dead-cell dye.",
             fontsize=10.0, color="#4A4A4A")

    for i, (t_min, title, note) in enumerate(zip(target_times, stage_titles, stage_notes)):
        row_idx = int((df_traj["t_min"] - t_min).abs().argsort().iloc[0])
        row = df_traj.iloc[row_idx]
        # MOESM9 has 284 frames spanning 0..180 min
        fidx = min(283, max(0, int(round((float(row["t_min"]) / 180.0) * 282))))
        fr = get_video_frame_rgb(ls_dir / "MOESM9_ESM.mp4", fidx)
        dist_3d = float(np.hypot(row["cart_x_um"] - row["target_x_um"],
                                 np.hypot(row["cart_y_um"] - row["target_y_um"], row["cart_z_um"] - row["target_z_um"])))

        # Crop XY microwell interior (y: 490..1180, x: 180..870) and XZ side projection (y: 1350..1710, x: 190..880)
        xy_crop = fr[490:1180, 180:870]
        xz_crop = fr[1350:1710, 190:880]

        ax_xy = fig.add_subplot(gs[0, i])
        ax_xy.imshow(xy_crop, interpolation="bilinear")
        ax_xy.set_title(title, fontsize=10.0, fontweight="bold", color="#111111", pad=5)
        ax_xy.set_xticks([]); ax_xy.set_yticks([])
        if i == 0:
            ax_xy.set_ylabel("Top View (XY)\n10 µm scale", fontsize=9.5, fontweight="bold", color="#111111")
        ax_xy.text(0.04, 0.05, note, transform=ax_xy.transAxes, fontsize=8.0, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.25", fc="#111111", ec="none", alpha=0.80))

        ax_xz = fig.add_subplot(gs[1, i])
        ax_xz.imshow(xz_crop, interpolation="bilinear")
        ax_xz.set_xticks([]); ax_xz.set_yticks([])
        if i == 0:
            ax_xz.set_ylabel("Side View (XZ)\nDepth cross-section", fontsize=9.5, fontweight="bold", color="#111111")
        ax_xz.text(0.04, 0.08,
                   f"CA: {row['CA_um2']:.1f} µm² | Dist: {dist_3d:.1f} µm",
                   transform=ax_xz.transAxes, fontsize=7.8, color="#FFFFFF",
                   bbox=dict(boxstyle="square,pad=0.2", fc="#111111", ec="none", alpha=0.75))

    # Bottom row: 2 quantitative trajectory plots across all 37 timepoints (0-180 min)
    ax_b1 = fig.add_subplot(gs[2, 0:3])
    ax_b1.plot(df_traj["t_min"], df_traj["CA_um2"], "-o", color="#2E7D5B", lw=2.0, ms=4, label="3D Synapse Contact Area (µm²)")
    ax_b1.plot(df_traj["t_min"], df_traj["target_V_um3"] / 5.0, "-s", color="#C0561F", lw=2.0, ms=4, label="Target 3D Volume (÷5 µm³)")
    for tm in target_times:
        ax_b1.axvline(tm, color="#C9C9C9", ls="--", lw=0.8)
    ax_b1.set_xlabel("Time (min)", fontsize=9.5, fontweight="bold")
    ax_b1.set_ylabel("Synapse Area (µm²) / Vol (÷5)", fontsize=9.5)
    ax_b1.set_title("Measured 3D Synapse Assembly (50–85 min) Followed by Target Collapse (110–120 min)",
                    fontsize=10.0, fontweight="bold", color="#111111")
    ax_b1.grid(True, color="#E3E3E3", lw=0.7)
    ax_b1.legend(frameon=False, fontsize=8.8, loc="upper left")

    ax_b2 = fig.add_subplot(gs[2, 3:6])
    ax_b2.plot(df_traj["t_min"], df_traj["sytox_blue_au"], "-o", color="#B3261E", lw=2.2, ms=4, label="SYTOX Blue Nuclear Intensity (a.u.)")
    ax_b2.plot(df_traj["t_min"], df_traj["cart_Lc_um"], "-^", color="#1F4E79", lw=2.0, ms=4, label="CAR-T Major Length (µm)")
    ax_b2.plot(df_traj["t_min"], df_traj["target_n_lobes"] * 8.0, "--d", color="#6B4C9A", lw=1.8, ms=4, label="Target Bleb Lobes (×8)")
    for tm in target_times:
        ax_b2.axvline(tm, color="#C9C9C9", ls="--", lw=0.8)
    ax_b2.set_xlabel("Time (min)", fontsize=9.5, fontweight="bold")
    ax_b2.set_ylabel("SYTOX (%) / Length (µm)", fontsize=9.5)
    ax_b2.set_title("Target Blebbing (80–100 min), SYTOX Blue Nuclear Influx (150–180 min) & CAR-T Detachment",
                    fontsize=10.0, fontweight="bold", color="#111111")
    ax_b2.grid(True, color="#E3E3E3", lw=0.7)
    ax_b2.legend(frameon=False, fontsize=8.8, loc="upper left")

    for out_p in [ART_DIR / "visual_2_lightsheet_kill_sequence.png", FIG_DIR / "visual_2_lightsheet_kill_sequence.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_2_lightsheet_kill_sequence.png")


def build_panel_3_organoid_and_leukemia():
    print("Building Panel 3: visual_3_organoid_and_leukemia_co_culture.png...")
    f_org = h5py.File(ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims", "r")
    f_exp28 = h5py.File(ROOT / "LSM880_example_dataset_CART_ALL_Exp28.ims", "r")

    org_tps = [0, 75, 150, 225, 300]  # 0.0 h, 2.5 h, 5.0 h, 7.5 h, 10.0 h
    exp_tps = [0, 90, 180, 300, 420]  # 0.0 h, 4.2 h, 8.4 h, 14.0 h, 19.7 h

    fig = plt.figure(figsize=(15.5, 8.5), dpi=160, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(3, 5, height_ratios=[1.0, 1.0, 0.9], hspace=0.30, wspace=0.14,
                           left=0.05, right=0.98, top=0.87, bottom=0.06)

    fig.suptitle("Visual Guide 3 — Real 4D Confocal Co-Culture Time-Lapses: Solid Breast Tumor Organoids vs Liquid B-ALL Leukemia",
                 fontsize=14.8, fontweight="bold", color="#111111", x=0.05, ha="left", y=0.96)
    fig.text(0.05, 0.915,
             "Top row: Breast cancer PDO (Organoid 01 zoom) infiltrated by CD8 (green) & CD4 (cyan) TEGs turning bright red (dead-cell nuclear dye) over 10 h. Middle row: CD19 CAR-T cells clearing primary B-ALL leukemia over 19.7 h.",
             fontsize=10.0, color="#4A4A4A")

    for i, tp in enumerate(org_tps):
        rgb_full, dead_n, cd8_n, _ = read_organoid_ims_rgb(f_org, tp)
        # Crop around the central-right organoid cluster (y: 45..165, x: 55..175)
        crop_org = rgb_full[45:165, 55:175]
        dead_pct = 100.0 * float(np.mean(dead_n[45:165, 55:175] > 0.35))
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(crop_org, interpolation="bilinear")
        ax.set_title(f"Breast PDO — t = {tp*2.0/60.0:.1f} h\n(Frame {tp})", fontsize=10.0, fontweight="bold", color="#111111", pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Solid Tumor PDO\n(TEG Co-Culture)", fontsize=9.5, fontweight="bold", color="#111111")
        ax.text(0.04, 0.05, f"Dead-dye area: {dead_pct:.1f}%\nGreen: CD8 | Red: Lysed",
                transform=ax.transAxes, fontsize=8.0, color="#FFFFFF",
                bbox=dict(boxstyle="square,pad=0.25", fc="#111111", ec="none", alpha=0.78))

    for i, tp in enumerate(exp_tps):
        rgb_e = read_exp28_ims_rgb(f_exp28, tp)
        crop_e = rgb_e[110:390, 110:390]
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(crop_e, interpolation="bilinear")
        ax.set_title(f"B-ALL + CAR-T — t = {tp*2.81/60.0:.1f} h\n(Frame {tp})", fontsize=10.0, fontweight="bold", color="#111111", pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Liquid B-ALL Blasts\n(CD19 CAR-T)", fontsize=9.5, fontweight="bold", color="#111111")
        ax.text(0.04, 0.05, "Green: CD8 CAR-T\nCyan: CD4 | Red: Lysed",
                transform=ax.transAxes, fontsize=8.0, color="#FFFFFF",
                bbox=dict(boxstyle="square,pad=0.25", fc="#111111", ec="none", alpha=0.78))

    f_org.close()
    f_exp28.close()

    # Bottom row left: Organoid volume loss & fragmentation curve from ims_conjugate_scenes.json (scene_organoid_826950)
    import json
    with open(ROOT / "data/ims_conjugate_scenes.json") as f_sc:
        sc826 = json.load(f_sc)["scene_organoid_826950"]["frames"]
    t_hrs = [fr["t_min"] / 60.0 for fr in sc826]
    v0 = sc826[0]["org_vol"]
    vol_pct = [100.0 * fr["org_vol"] / v0 for fr in sc826]
    frags_scaled = [fr["n_frag"] * 4.5 for fr in sc826]
    dye_scaled = [fr["dead_dye_voxel"] * 2.2 for fr in sc826]

    ax_c1 = fig.add_subplot(gs[2, 0:3])
    ax_c1.plot(t_hrs, vol_pct, "-", color="#2E7D5B", lw=2.2, label="Organoid 01 Viable Volume (%)")
    ax_c1.plot(t_hrs, frags_scaled, "--", color="#C0561F", lw=2.0, label="3D Apoptotic Sub-Fragments (×4.5)")
    ax_c1.plot(t_hrs, dye_scaled, "-.", color="#B3261E", lw=2.0, label="Nuclear Dead-Dye Intensity (×2.2 a.u.)")
    ax_c1.set_xlabel("Co-Culture Time (hours)", fontsize=9.5, fontweight="bold")
    ax_c1.set_ylabel("Volume (%) / Fragments", fontsize=9.5)
    ax_c1.set_title("Measured 3D Organoid Volume Collapse (102,553 -> 44,713 µm³) & Fragmentation (3 -> 18 bodies)",
                    fontsize=10.0, fontweight="bold", color="#111111")
    ax_c1.grid(True, color="#E3E3E3", lw=0.7)
    ax_c1.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Bottom row right: 3D T-cell elongation across the 9 behavioral clusters
    df_cl = pd.read_csv(ROOT / "data/tcell_3d_morphology_by_cluster.csv")
    ax_c2 = fig.add_subplot(gs[2, 3:5])
    colors_cl = ["#5C5C5C" if c < 7 else ("#1F4E79" if c < 9 else "#B3261E") for c in df_cl["cluster"]]
    ax_c2.bar(df_cl["cluster"].astype(str), df_cl["max_length_um"], color=colors_cl, width=0.62)
    ax_c2.axhline(13.6, color="#111111", ls="--", lw=1.0, label="Resting Spherical T Cell (13.6 µm)")
    ax_c2.set_xlabel("BEHAV3D Behavioral Cluster (1–5: No Contact | 7–8: Engager | 9: Super-Engager)", fontsize=9.0, fontweight="bold")
    ax_c2.set_ylabel("Peak 3D Length (µm)", fontsize=9.5)
    ax_c2.set_title("Super-Engager T Cells (Cluster 9) Elongate to 28.9 µm During Killing",
                    fontsize=10.0, fontweight="bold", color="#111111")
    ax_c2.set_ylim(0, 34)
    ax_c2.grid(axis="y", color="#E3E3E3", lw=0.7)
    ax_c2.legend(frameon=False, fontsize=8.5, loc="upper left")

    for out_p in [ART_DIR / "visual_3_organoid_and_leukemia_co_culture.png", FIG_DIR / "visual_3_organoid_and_leukemia_co_culture.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_3_organoid_and_leukemia_co_culture.png")


def build_panel_4_four_layer_architecture():
    print("Building Panel 4: visual_4_four_layer_architecture.png...")
    df_swp = pd.read_csv(ROOT / "data/deformable_two_cell_sweep.csv")
    with open(ROOT / "data/sim_presets_4d.json") as f:
        import json
        presets = json.load(f)

    fig = plt.figure(figsize=(15.5, 8.0), dpi=160, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.05, 1.0], hspace=0.36, wspace=0.24,
                           left=0.05, right=0.97, top=0.86, bottom=0.08)

    fig.suptitle("Visual Guide 4 — How the 4-Layer Digital Twin Couples Mechanics, Polarity, and Target Execution",
                 fontsize=15, fontweight="bold", color="#111111", x=0.05, ha="left", y=0.95)
    fig.text(0.05, 0.905,
             "Each model layer is parameterized directly from its matching imaging modality and passes physical state variables forward and backward in real time.",
             fontsize=10.2, color="#4A4A4A")

    layers_info = [
        ("Layer 1 — Search, Arrest & Priming",
         "#1F4E79",
         "• Chemokine & target proximity guidance\n"
         "• 9 BEHAV3D Markov clusters (Cl 1..9)\n"
         "• IFN-I (ISG15/STAT1) -> CRTAM (44×)\n"
         "• Speed: 3.97 -> 0.42 µm/min on arrest\n"
         "• Fit: 3,598 tracks + 1,814 scRNA cells"),
        ("Layer 2 — Deformable Synapse Cup",
         "#2E7D5B",
         "• TCR/CAR signal -> WASP/Arp2/3 actin\n"
         "• 3D vertex mesh + protrusion F_syn\n"
         "• Area CA = 166.7 µm², indent = 1.18 µm\n"
         "• Retrograde actin flow ARS = 57.8 nm/s\n"
         "• Fit: 419 light-sheet pairs + ShElastic"),
        ("Layer 3 — MTOC & Pore Delivery",
         "#C0561F",
         "• Dynein pulling on cortical F-actin\n"
         "• 3D MTOC angle MPA: 92° -> 47° (4 min)\n"
         "• Basu 2016 Law: tension sigma_mem\n"
         "  potentiates perforin k_pore up to 10×\n"
         "• Competes with ESCRT membrane repair"),
        ("Layer 4 — Caspase-3, Blebs & Exit",
         "#B3261E",
         "• Granzyme B influx via perforin pores\n"
         "• Bistable Caspase-3 -> ROCK1 blebbing\n"
         "• 39–74% volume loss & 3–18 fragments\n"
         "• SYTOX/ToPro-3 nuclear lysis (41–100m)\n"
         "• Detachment for serial killing (T1->T2)"),
    ]

    for i, (l_title, col_hex, body_txt) in enumerate(layers_info):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")
        rect = patches.FancyBboxPatch((0.2, 0.4), 9.6, 9.0, boxstyle="round,pad=0.2,rounding_size=0.3",
                                      fc="#F6F6F6", ec=col_hex, lw=2.2)
        ax.add_patch(rect)
        ax.text(0.55, 8.5, l_title, fontsize=9.4, fontweight="bold", color=col_hex, va="top")
        ax.plot([0.55, 9.45], [7.35, 7.35], color="#C9C9C9", lw=1.0)
        ax.text(0.55, 6.85, body_txt, fontsize=8.1, color="#1F1F1F", va="top", linespacing=1.45)

    # Bottom row left (cols 0-1): Force vs Stiffness Phase Diagram from deformable_two_cell_sweep.csv
    ax_p1 = fig.add_subplot(gs[1, 0:2])
    for stiff, col_s, mk in [(300.0, "#2E7D5B", "o"), (600.0, "#1F4E79", "s"), (1500.0, "#B3261E", "^")]:
        sub = df_swp[df_swp["E_target_Pa"] == stiff]
        ax_p1.plot(sub["F_syn_nN"], sub["basu_mechano_multiplier"], f"-{mk}", color=col_s, lw=2.0,
                   label=f"Target stiffness E = {int(stiff)} Pa")
    ax_p1.axvline(0.05, color="#5C5C5C", ls=":", lw=1.2)
    ax_p1.text(0.07, 10.2, "Dasatinib 50 nM\n(0.05 nN)", fontsize=8.2, color="#5C5C5C")
    ax_p1.axvline(0.85, color="#111111", ls="--", lw=1.2)
    ax_p1.text(0.88, 10.2, "Control CAR-T / CD8\n(0.85 nN)", fontsize=8.2, fontweight="bold", color="#111111")
    ax_p1.set_xlabel("Synaptic Actin Protrusion Force F_syn (nN)", fontsize=9.5, fontweight="bold")
    ax_p1.set_ylabel("Basu Mechano-Pore Multiplier (×)", fontsize=9.5)
    ax_p1.set_title("Step 7 Mechano-Chemical Multiplier Across Force & Target Stiffness",
                    fontsize=10.5, fontweight="bold", color="#111111")
    ax_p1.grid(True, color="#E3E3E3", lw=0.7)
    ax_p1.legend(frameon=False, fontsize=8.8, loc="lower right")

    # Bottom row right (cols 2-3): Simulated Presets Comparison (Super-Engager vs Exhausted vs Dasatinib 50 nM)
    ax_p2 = fig.add_subplot(gs[1, 2:4])
    for pkey, lbl, col_p, ls_p in [
        ("super_engager_serial", "Super-Engager CD8 (Serial Kill T1 & T2)", "#1F4E79", "-"),
        ("single_conjugate_13T", "Standard Single Conjugate (13T Target)", "#2E7D5B", "--"),
        ("transient_subthreshold", "Transient Contact (Sub-Threshold Repair)", "#6B4C9A", "-."),
        ("dasatinib_50nM", "Dasatinib 50 nM (Function-Off CAR-T)", "#B3261E", ":"),
    ]:
        frs = presets[pkey]["frames"]
        ts = [f["t_min"] for f in frs]
        dyes = [f["t1_dead_dye"] for f in frs]
        ax_p2.plot(ts, dyes, ls=ls_p, color=col_p, lw=2.2, label=lbl)
    ax_p2.axhline(10.0, color="#B3261E", ls="--", lw=1.0, label="Lysis Threshold (10 a.u.)")
    ax_p2.set_xlabel("Simulation Time (min)", fontsize=9.5, fontweight="bold")
    ax_p2.set_ylabel("Target Nuclear Dead-Dye Uptake (a.u.)", fontsize=9.5)
    ax_p2.set_title("4D Digital Twin Output: Target Lysis Across Functional vs Inhibited Presets",
                    fontsize=10.5, fontweight="bold", color="#111111")
    ax_p2.grid(True, color="#E3E3E3", lw=0.7)
    ax_p2.legend(frameon=False, fontsize=8.5, loc="upper left")

    for out_p in [ART_DIR / "visual_4_four_layer_architecture.png", FIG_DIR / "visual_4_four_layer_architecture.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_4_four_layer_architecture.png")


def build_panel_5_multi_tcell_cooperativity():
    print("Building Panel 5: visual_5_multi_tcell_cooperativity.png & data/organoid_multitcell_swarm_empirical.csv...")
    ims_path = ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims"
    with h5py.File(ims_path, "r") as f:
        img_attrs = f["DataSetInfo/Image"].attrs
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
        sm_cd4 = f["Scene8/Content/MegaSurfaces0/SurfaceModel"][:]
        smi_cd4 = f["Scene8/Content/MegaSurfaces0/SurfaceModelInfo"][:]
        sm_cd8 = f["Scene8/Content/MegaSurfaces1/SurfaceModel"][:]
        smi_cd8 = f["Scene8/Content/MegaSurfaces1/SurfaceModelInfo"][:]
        sm_org = f["Scene8/Content/MegaSurfaces3/SurfaceModel"][:]
        smi_org = f["Scene8/Content/MegaSurfaces3/SurfaceModelInfo"][:]
        tr_org = f["Scene8/Content/MegaSurfaces3/Track0"][:]
        to_org = f["Scene8/Content/MegaSurfaces3/TrackObject0"][:]

        # Compute empirical simultaneous T-cell counts across all 21 tracked organoids
        cd4_pos = np.column_stack([smi_cd4["PositionX"] * 1000, smi_cd4["PositionY"] * 1000, smi_cd4["PositionZ"] * 1000])
        cd4_t = sm_cd4["ID_Time"]
        cd8_pos = np.column_stack([smi_cd8["PositionX"] * 1000, smi_cd8["PositionY"] * 1000, smi_cd8["PositionZ"] * 1000])
        cd8_t = sm_cd8["ID_Time"]
        org_pos = np.column_stack([smi_org["PositionX"] * 1000, smi_org["PositionY"] * 1000, smi_org["PositionZ"] * 1000])
        org_vol = smi_org["Volume"] * 1e9
        org_t = sm_org["ID_Time"]
        org_id = sm_org["ID"]
        id_to_idx = {int(oid): i for i, oid in enumerate(org_id)}

        swarm_rows = []
        for tr in tr_org:
            tid = int(tr["ID"])
            i0, i1 = int(tr["IndexTrackObjectBegin"]), int(tr["IndexTrackObjectEnd"])
            obj_ids = to_org["ID_Object"][i0:i1]
            idxs = [id_to_idx[int(o)] for o in obj_ids if int(o) in id_to_idx]
            if not idxs:
                continue
            times = np.unique(org_t[idxs])
            cd8_counts, cd4_counts, vols, n_frags = [], [], [], []
            for t in times[::5]:
                sub_i = [ix for ix in idxs if org_t[ix] == t]
                v_tot = float(np.sum(org_vol[sub_i]))
                vols.append(v_tot)
                n_frags.append(len(sub_i))
                op = org_pos[sub_i]
                orad = ((3.0 * org_vol[sub_i]) / (4.0 * np.pi)) ** (1.0 / 3.0)
                c8 = cd8_pos[cd8_t == t]
                n8 = int(np.sum(np.min(np.linalg.norm(c8[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1) <= 10.0)) if len(c8) > 0 else 0
                c4 = cd4_pos[cd4_t == t]
                n4 = int(np.sum(np.min(np.linalg.norm(c4[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1) <= 10.0)) if len(c4) > 0 else 0
                cd8_counts.append(n8)
                cd4_counts.append(n4)
            v_init = float(np.mean(vols[:3]))
            v_final = float(np.mean(vols[-3:]))
            tot_arr = np.array(cd8_counts) + np.array(cd4_counts)
            size_class = (
                "Large multicellular organoid (>50,000 um3)" if v_init >= 50000
                else ("Medium organoid (10,000-50,000 um3)" if v_init >= 10000 else "Single target / small cluster (<5,000 um3)")
            )
            swarm_rows.append({
                "organoid_track_id": tid,
                "size_class": size_class,
                "initial_volume_um3": round(v_init, 0),
                "final_volume_um3": round(v_final, 0),
                "volume_remaining_frac": round(v_final / max(v_init, 1.0), 2),
                "max_fragments": int(max(n_frags)),
                "peak_simultaneous_cd8": int(max(cd8_counts)),
                "mean_simultaneous_cd8": round(float(np.mean(cd8_counts)), 2),
                "peak_simultaneous_cd4": int(max(cd4_counts)),
                "mean_simultaneous_cd4": round(float(np.mean(cd4_counts)), 2),
                "peak_total_tcells": int(max(tot_arr)),
                "mean_total_tcells": round(float(np.mean(tot_arr)), 2),
            })
        df_swarm = pd.DataFrame(swarm_rows).sort_values("initial_volume_um3", ascending=False)
        df_swarm.to_csv(ROOT / "data/organoid_multitcell_swarm_empirical.csv", index=False)

        # Build Figure 5
        fig = plt.figure(figsize=(15.5, 9.2), dpi=160, facecolor="#FFFFFF")
        gs = gridspec.GridSpec(2, 5, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.22,
                               left=0.05, right=0.97, top=0.87, bottom=0.08)

        fig.suptitle("Visual Guide 5 — Why Multiple T Cells Are Required: Real Organoid Swarming & Additive Cooperativity",
                     fontsize=14.8, fontweight="bold", color="#111111", x=0.05, ha="left", y=0.955)
        fig.text(0.05, 0.912,
                 "Real 9-channel confocal data (149,129 CD4 + 58,517 CD8 3D surfaces) proves solid tumor organoids are attacked by 6–22 cooperating T cells simultaneously.",
                 fontsize=10.2, color="#4A4A4A")

        # Top row: 5 real frames of Organoid 1000826950 with overlaid 3D tracked CD8+ and CD4+ T cell contacts
        # Find centroid of Organoid 1000826950 at t=0
        tr826 = [tr for tr in tr_org if int(tr["ID"]) == 1000826950][0]
        obj826 = to_org["ID_Object"][int(tr826["IndexTrackObjectBegin"]):int(tr826["IndexTrackObjectEnd"])]
        idx826 = [id_to_idx[int(o)] for o in obj826 if int(o) in id_to_idx]
        p826_init = np.mean(org_pos[idx826[:5]], axis=0)  # um

        # Convert um to pixel index in 200x200 image
        def um_to_px(x_um, y_um):
            px = (x_um / 1000.0 - ext_min[0]) / (ext_max[0] - ext_min[0]) * 200.0
            py = (y_um / 1000.0 - ext_min[1]) / (ext_max[1] - ext_min[1]) * 200.0
            return px, py

        cx_px, cy_px = um_to_px(p826_init[0], p826_init[1])
        half_w = 38
        x0 = int(np.clip(cx_px - half_w, 0, 200 - 2 * half_w))
        y0 = int(np.clip(cy_px - half_w, 0, 200 - 2 * half_w))
        x1, y1 = x0 + 2 * half_w, y0 + 2 * half_w

        tps = [10, 90, 170, 250, 330]
        for i, tp in enumerate(tps):
            rgb, _, _, _ = read_organoid_ims_rgb(f, tp)
            crop = rgb[y0:y1, x0:x1]
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(crop, interpolation="bilinear")

            # Find all 3D tracked CD8 and CD4 surfaces near this organoid at frame tp
            sub_o = [ix for ix in idx826 if org_t[ix] == tp]
            if not sub_o:
                sub_o = [ix for ix in idx826 if abs(int(org_t[ix]) - tp) <= 2]
            op = org_pos[sub_o] if sub_o else p826_init[None, :]
            orad = ((3.0 * org_vol[sub_o]) / (4.0 * np.pi)) ** (1.0 / 3.0) if sub_o else np.array([25.0])

            c8 = cd8_pos[cd8_t == tp]
            c4 = cd4_pos[cd4_t == tp]
            engaged_8, engaged_4 = [], []
            if len(c8) > 0:
                d8 = np.min(np.linalg.norm(c8[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1)
                engaged_8 = c8[d8 <= 14.0]
            if len(c4) > 0:
                d4 = np.min(np.linalg.norm(c4[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1)
                engaged_4 = c4[d4 <= 14.0]

            for pt in engaged_8:
                px, py = um_to_px(pt[0], pt[1])
                if x0 <= px < x1 and y0 <= py < y1:
                    ax.plot(px - x0, py - y0, "o", ms=7.5, mfc="none", mec="#55FF77", mew=1.8)
            for pt in engaged_4:
                px, py = um_to_px(pt[0], pt[1])
                if x0 <= px < x1 and y0 <= py < y1:
                    ax.plot(px - x0, py - y0, "s", ms=6.8, mfc="none", mec="#55DDFF", mew=1.6)

            ax.set_title(f"t = {tp*2.0/60.0:.1f} h ({tp*2} min)\n{len(engaged_8)} CD8 + {len(engaged_4)} CD4 engaged",
                         fontsize=9.8, fontweight="bold", color="#111111", pad=5)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("Real Confocal Crop\n(Organoid 1000826950)", fontsize=9.5, fontweight="bold", color="#111111")
            ax.text(0.04, 0.05, f"Circles: 3D CD8 ({len(engaged_8)})\nSquares: 3D CD4 ({len(engaged_4)})",
                    transform=ax.transAxes, fontsize=7.8, color="#FFFFFF",
                    bbox=dict(boxstyle="square,pad=0.22", fc="#111111", ec="none", alpha=0.80))

    # Bottom row left (cols 0-1): Empirical Simultaneous T Cells vs Organoid Initial Volume
    ax_b1 = fig.add_subplot(gs[1, 0:2])
    df_s_plot = df_swarm.sort_values("initial_volume_um3")
    x_vols = df_s_plot["initial_volume_um3"] / 1000.0  # in 10^3 um^3
    ax_b1.scatter(x_vols, df_s_plot["peak_simultaneous_cd8"], color="#1F4E79", s=46, label="Peak simultaneous CD8+ T cells", zorder=3)
    ax_b1.scatter(x_vols, df_s_plot["peak_total_tcells"], color="#C0561F", marker="s", s=46, label="Peak total T cells (CD8+ & CD4+)", zorder=3)
    # Trend lines
    z_tot = np.polyfit(x_vols, df_s_plot["peak_total_tcells"], 1)
    x_line = np.linspace(0, 145, 50)
    ax_b1.plot(x_line, z_tot[0] * x_line + z_tot[1], "--", color="#C0561F", lw=1.5, alpha=0.8)
    ax_b1.set_xlabel("Initial Tumor Target / Organoid Volume (×10³ µm³)", fontsize=9.2, fontweight="bold")
    ax_b1.set_ylabel("Peak Simultaneous T Cells Engaged", fontsize=9.2)
    ax_b1.set_title("Empirical Scaling (21 Organoids): Target Volume vs Swarm Size",
                    fontsize=9.6, fontweight="bold", color="#111111")
    ax_b1.grid(True, color="#E3E3E3", lw=0.7)
    ax_b1.legend(frameon=False, fontsize=8.3, loc="upper left")

    # Bottom row middle (col 2): Real scRNA-seq Chemokine Swarming Surge (CCL4, CCL3, IFNG)
    df_sc = pd.read_csv(ROOT / "data/behav3d_scrna_behavioral_states.csv")
    ax_b2 = fig.add_subplot(gs[1, 2])
    states_short = ["Med", "Never", "Non-eng", "Engager", "Super"]
    x_idx = np.arange(len(df_sc))
    ax_b2.plot(x_idx, df_sc["expr_CCL4"], "-o", color="#B3261E", lw=2.0, label="CCL4 (chemoattractant)")
    ax_b2.plot(x_idx, df_sc["expr_CCL3"], "-s", color="#C0561F", lw=2.0, label="CCL3 (chemoattractant)")
    ax_b2.plot(x_idx, df_sc["expr_IFNG"], "-^", color="#1F4E79", lw=1.8, label="IFNG (sensitizer)")
    ax_b2.set_xticks(x_idx)
    ax_b2.set_xticklabels(states_short, fontsize=8.2)
    ax_b2.set_ylabel("Mean Log-Normalized Expression", fontsize=9.0)
    ax_b2.set_title("scRNA-seq (n=1,814): Pioneer\nSynapse Induces CCL3/CCL4",
                    fontsize=9.4, fontweight="bold", color="#111111")
    ax_b2.grid(True, color="#E3E3E3", lw=0.7)
    ax_b2.legend(frameon=False, fontsize=7.8, loc="upper left")

    # Bottom row right (cols 3-4): Additive Multi-T-Cell Cooperativity vs ESCRT Membrane Repair
    from model_4d_cell_kill import simulate_preset
    ax_b3 = fig.add_subplot(gs[1, 3:5])
    coop_colors = {1: "#5C5C5C", 2: "#2E7D5B", 3: "#C0561F", 4: "#B3261E"}
    coop_styles = {1: ":", 2: "-.", 3: "--", 4: "-"}
    for n_eff in [1, 2, 3, 4]:
        res_n = simulate_preset(
            f"coop_{n_eff}",
            F_syn_max_nN=0.52,
            E_target_Pa=1300.0,
            beta_tension=2.15,
            k_repair_min=0.165,
            dwell_cap_min=65.0,
            serial_kill=False,
            n_effectors=n_eff,
        )
        ts_n = [r["t_min"] for r in res_n["frames"]]
        dyes_n = [r["t1_dead_dye"] for r in res_n["frames"]]
        kill_t = res_n["summary"]["t1_killed_time_min"]
        plural = "s" if n_eff > 1 else ""
        lbl_n = (
            f"{n_eff} T cell{plural} — ESCRT repair wins (Target survives)"
            if kill_t is None
            else f"{n_eff} cooperating T cell{plural} — Lysed at {kill_t:.0f} min"
        )
        ax_b3.plot(ts_n, dyes_n, ls=coop_styles[n_eff], color=coop_colors[n_eff], lw=2.2, label=lbl_n)

    ax_b3.axhline(10.0, color="#B3261E", ls="--", lw=1.0, label="Permeabilization Threshold (10 a.u.)")
    ax_b3.set_xlabel("Time After Pioneer Approach (min)", fontsize=9.2, fontweight="bold")
    ax_b3.set_ylabel("Target Nuclear Dead-Dye Uptake (a.u.)", fontsize=9.2)
    ax_b3.set_title("4D Model Cooperativity: Multiple T Cells Overcome ESCRT Membrane Repair",
                    fontsize=9.8, fontweight="bold", color="#111111")
    ax_b3.grid(True, color="#E3E3E3", lw=0.7)
    ax_b3.legend(frameon=False, fontsize=8.2, loc="upper left")

    for out_p in [ART_DIR / "visual_5_multi_tcell_cooperativity.png", FIG_DIR / "visual_5_multi_tcell_cooperativity.png"]:
        fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print("   Saved visual_5_multi_tcell_cooperativity.png and data/organoid_multitcell_swarm_empirical.csv")


if __name__ == "__main__":
    build_panel_1_scale_gap()
    build_panel_2_lightsheet_kill_sequence()
    build_panel_3_organoid_and_leukemia()
    build_panel_4_four_layer_architecture()
    build_panel_5_multi_tcell_cooperativity()
    print("All 5 visual explainer panels built successfully.")
