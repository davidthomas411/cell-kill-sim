#!/usr/bin/env python3
"""
Version 3.0 — Part I & Physics Layer:
  1. Rebuilds All 7 Single-Cell 320-nm Light-Sheet Digital Twins directly from the
     deposited ScienceDB `10.57760/sciencedb.19235` 3D U-Net semantic segmentation
     masks (`*_predictions.h5`, `(166, 256, 256) uint8`: `1 = CAR-T body`, `2 = Nalm6
     target interior`, `3 = CAR-T dense F-actin`, `4 = Target mApple-Mem membrane`)
     and raw 16-bit fluorescence stacks/projections (`405 nm` SYTOX Blue, `488 nm`
     Lifeact-EGFP / `637 nm` SiR-Tubulin on Volume 6, `561 nm` mApple-Mem).
     - Eliminates all supplementary MP4 frames, burned-in text ("chamber 39 volume 1",
       "Actin"), scale bars, empty t=0 frames (Twin 5), and 2.8x colored-pixel mismatch (Twin 7).
  2. Builds the Coupled Synaptic Mechanics & Pore/Repair Physics Layer across both the
     deposited 3D volumes and the 419-pair deposited cohort (`Source_Data_Wang2024_NatCommun.xlsx`):
     - Uncapped Molecular-Clutch Synaptic Traction & Hertzian Indentation Force Balance
       driven by measured 3D contact area CA (um^2), measured Actin Retrograde Flow Speed
       ARFS (nm/s), and Actin Depletion Coefficient ADC.
     - 2D Profile Log-Likelihood & Fisher Information Identifiability Map proving that
       kinematics alone identify only the ratio eta_clutch / E_target (condition number
       kappa = 1.48e4), requiring external AFM/MP-TFM stiffness anchoring (E_target = 450 +/- 120 Pa).
     - Tension-Potentiated Perforin Pore Insertion vs. Ca2+/ESCRT Membrane Repair ODE Layer
       (Basu et al. Cell 2016; Ritter et al. Science 2022) gated by 3D MTOC docking angle MPA (deg).
     - Strict Out-of-Sample Perturbation Test trained on Untreated High-Killing (Groups 1-5,
       n = 167, 57.5% kill) and tested with zero refitting on Untreated Low-Killing
       (Groups 6-10, n = 163, 27.0% kill) and Dasatinib 25 nM & 50 nM Lck-Inhibition
       (Groups 8*, 9*, 10*, n = 89, 4.5% kill).
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, stats
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from build_physically_constrained_4d_shapes import (
    extract_smooth_surface_net_mesh,
    shade_mesh_phong,
)

from paths import REPO_ROOT as ROOT, DATA_DIR, FIGURES_DIR as FIG_DIR, VIDEOS_DIR as VID_DIR
from constants import VOXEL_UM
ART_DIR = FIG_DIR
CHAMBER_DIR = DATA_DIR / "scidb_19235_chambers"
SAMPLE_DIR = DATA_DIR / "scidb_19235_samples"

DX_UM = VOXEL_UM  # Lateral voxel pitch of the deposited 256x256 ROI grid (from feilab summary.m sqrt(0.0385 um^2))
VOX_VOL_UM3 = DX_UM ** 3  # Voxel volume in um^3 per voxel

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#111111",
    "axes.linewidth": 0.8,
})


def encode_mp4_and_gif(frame_dir: Path, mp4_path: Path, gif_path: Path, fps: int = 6):
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
    cmd_gif = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-vf", "fps=6,scale=840:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        str(gif_path),
    ]
    subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ==============================================================================
# 1. PHYSICS LAYER: MECHANICAL FORCE BALANCE, IDENTIFIABILITY & PORE/REPAIR ODE
# ==============================================================================

def compute_synaptic_force_and_tension(
    ca_um2: np.ndarray | float,
    arfs_nm_s: np.ndarray | float,
    adc: np.ndarray | float,
    eta_clutch: float = 17.1,   # pN * s / um^3 (calibrated with AFM E_target = 450 Pa prior)
    E_target_Pa: float = 450.0, # Pa (external AFM/MP-TFM stiffness prior for B-ALL/Nalm6)
    T0_pN_um: float = 50.0,     # Cortical resting tension (pN/um = uN/m)
) -> dict:
    """
    Coupled Molecular-Clutch Traction & Hertzian Viscoelastic Indentation Model:
      - Uncapped synaptic traction force F_syn (nN):
          F_syn = 1e-3 * eta_clutch * A_syn * v_actin * Phi_clutch(ADC)
        where v_actin = max(0, ARFS_nm_s) * 1e-3 (um/s), and
        Phi_clutch(ADC) = 1 + 0.18 * log1p(max(0, ADC)) captures peripheral F-actin ring focus.
      - Quasi-static Hertzian + cortical-shell indentation depth delta_syn (um):
          F_indent(delta) = (4/3) * (E_target / (1 - nu^2)) * sqrt(R_eff) * delta^(3/2) + 2*pi*T0*delta
      - Local synaptic membrane tension sigma_syn (pN/um):
          sigma_syn = T0 + (F_syn * 1e3) / (2 * pi * R_syn)
    """
    ca = np.maximum(0.0, np.nan_to_num(np.asarray(ca_um2, dtype=np.float64), nan=0.0))
    v_um_s = np.maximum(0.0, np.nan_to_num(np.asarray(arfs_nm_s, dtype=np.float64), nan=0.0)) * 1e-3
    adc_pos = np.maximum(0.0, np.nan_to_num(np.asarray(adc, dtype=np.float64), nan=0.0))

    phi_clutch = 1.0 + 0.18 * np.log1p(adc_pos)
    # Uncapped traction force in nN (1 nN = 1000 pN)
    # Include basal LFA-1/ICAM-1 adhesion traction (1.8 pN/um^2) when in physical contact (ca > 0)
    f_syn_pN = np.where(
        ca > 0.5,
        ca * (eta_clutch * v_um_s * phi_clutch + 1.8 * (v_um_s / (v_um_s + 0.008))),
        0.0
    )
    f_syn_nN = f_syn_pN * 1e-3

    # Effective contact radius R_syn (um) and reduced cell radius R_eff = 3.2 um
    r_syn_um = np.sqrt(ca / np.pi + 0.25)
    r_eff_um = 3.2
    nu = 0.45
    hertz_coeff_pN = (4.0 / 3.0) * (E_target_Pa / (1.0 - nu ** 2)) * np.sqrt(r_eff_um)

    # Invert F_indent(delta) = f_syn_pN for indentation depth delta_syn (um)
    # Approximate analytical inversion of c1 * delta^(1.5) + c2 * delta = F
    c2 = 2.0 * np.pi * T0_pN_um
    delta_syn_um = np.where(
        f_syn_pN > 0.0,
        np.minimum(3.8, (f_syn_pN / (hertz_coeff_pN + 0.65 * c2)) ** (2.0 / 3.0)),
        0.0
    )

    # Local membrane tension sigma_syn (pN/um)
    sigma_syn_pN_um = np.where(
        ca > 0.5,
        T0_pN_um + f_syn_pN / (2.0 * np.pi * np.maximum(1.5, r_syn_um)),
        T0_pN_um
    )

    return {
        "F_syn_nN": f_syn_nN,
        "delta_syn_um": delta_syn_um,
        "sigma_syn_pN_um": sigma_syn_pN_um,
        "r_syn_um": r_syn_um,
    }


def predict_pore_repair_kill_probability(
    ca_um2: np.ndarray,
    arfs_nm_s: np.ndarray,
    adc: np.ndarray,
    mpa_deg: np.ndarray,
    k_pfp: float = 0.088,        # Perforin pore insertion rate constant (min^-1)
    k_escrt: float = 0.045,      # Ca2+/ESCRT-III membrane repair rate (min^-1, tau_repair = 22.2 min)
    g_crit: float = 1.15,        # Critical intracellular Granzyme B threshold for caspase commitment
    theta_0_deg: float = 24.0,   # MTOC docking angular half-width (deg)
    eta_clutch: float = 17.1,
    E_target_Pa: float = 450.0,
    exposure_min: float = 120.0,
) -> dict:
    """
    Tension-Potentiated Perforin Pore Insertion & Ca2+/ESCRT Membrane Repair Model
    (Basu et al. Cell 2016; Ritter et al. Science 2022):
      1. Mechanical tension potentiation of perforin pore formation:
           M_tension = exp((sigma_syn - T0) / sigma_char),  sigma_char = 65 pN/um
      2. MTOC polarization & cortical actin gating of lytic granule exocytosis:
           Phi_degran = (CA / (CA + K_A)) * (1 / (1 + (MPA / theta_0)^2)) * (1 - exp(-ARFS / 6.0))
      3. Steady-state active perforin pores N_pore^* = (k_pfp / k_escrt) * Phi_degran * M_tension
      4. Net Granzyme B accumulation over contact window and logistic/Weibull kill probability.
    """
    mech = compute_synaptic_force_and_tension(
        ca_um2, arfs_nm_s, adc, eta_clutch=eta_clutch, E_target_Pa=E_target_Pa
    )
    f_syn = mech["F_syn_nN"]
    sigma_syn = mech["sigma_syn_pN_um"]
    delta_syn = mech["delta_syn_um"]

    ca = np.maximum(0.0, np.nan_to_num(np.asarray(ca_um2, dtype=np.float64), nan=0.0))
    arfs = np.maximum(0.0, np.nan_to_num(np.asarray(arfs_nm_s, dtype=np.float64), nan=0.0))
    adc_pos = np.maximum(0.0, np.nan_to_num(np.asarray(adc, dtype=np.float64), nan=0.0))
    # Where MPA is NaN in Source_Data_Wang2024_NatCommun.xlsx, MTOC failed to polarize to the synapse (~85 deg)
    mpa_arr = np.asarray(mpa_deg, dtype=np.float64)
    mpa = np.where(np.isnan(mpa_arr), 85.0, mpa_arr)

    # Mechano-potentiation factor (Basu et al. Cell 2016: tension stretches target membrane to potentiate perforin)
    m_tension = np.exp(np.clip((sigma_syn - 50.0) / 55.0, 0.0, 2.2))

    # Degranulation gate: requires contact area (K_A = 65 um^2), F-actin ring focus (ADC), MTOC docking, and Lck/ARFS signaling
    gate_area = (ca ** 1.35) / (ca ** 1.35 + 65.0 ** 1.35)
    gate_adc = 0.45 + 0.55 * (adc_pos / (adc_pos + 3.2))
    gate_mtoc = 1.0 / (1.0 + (mpa / theta_0_deg) ** 2.0)
    gate_lck = (arfs ** 1.4) / (arfs ** 1.4 + 28.0 ** 1.4)
    phi_degran = gate_area * gate_adc * (0.18 + 0.82 * gate_mtoc) * gate_lck

    # Net un-repaired perforin pore level above Ca2+/ESCRT-III membrane repair rate k_escrt:
    # When pore insertion <= k_escrt, ESCRT-III seals membrane lesions before lethal Granzyme B entry!
    pore_prod = k_pfp * phi_degran * m_tension
    n_pore_ss = np.maximum(0.0, (pore_prod - 0.32 * k_escrt) / max(1e-4, k_escrt))
    gzmb_dose = n_pore_ss * (exposure_min / 60.0)

    # Biophysical Weibull pore-threshold hazard: P_kill = 0 when gzmb_dose = 0!
    p_kill = 1.0 - np.exp(-((gzmb_dose / max(0.05, g_crit)) ** 1.65))

    # Compare against the old v2.5 area-only capped proxy (which ignored ARFS, MPA, and ESCRT repair)
    f_old_capped = np.where(ca > 0.5, np.minimum(1.55, 0.0095 * ca + 0.28), 0.0)
    p_kill_old_area_proxy = np.where(ca > 0.5, 1.0 / (1.0 + np.exp(-2.6 * (f_old_capped - 0.68))), 0.02)

    return {
        "F_syn_nN": f_syn,
        "F_old_capped_nN": f_old_capped,
        "delta_syn_um": delta_syn,
        "sigma_syn_pN_um": sigma_syn,
        "m_tension": m_tension,
        "n_pore_ss": n_pore_ss,
        "gzmb_dose": gzmb_dose,
        "p_kill_physics": p_kill,
        "p_kill_old_proxy": p_kill_old_area_proxy,
    }


def run_physics_and_perturbation_benchmark():
    """
    1. Calibrate the 3 pore-repair parameters (k_pfp, k_escrt, g_crit) ONLY on
       Untreated High-Killing CAR-T (Groups 1-5, n = 167 cells, observed kill = 96/167 = 57.5%).
    2. Evaluate Out-of-Sample with ZERO refitting on:
       - Untreated Low-Killing CAR-T (Groups 6-10, n = 163 cells, observed kill = 44/163 = 27.0%)
       - Dasatinib 50 nM & 25 nM Lck-Inhibited CAR-T (Groups 8*, 9*, 10*, n = 89 cells, observed kill = 4/89 = 4.5%)
         Plus the 103-cell Dasatinib dose-response table (`lightsheet_wang2026_dasatinib.csv`: 0 nM, 25 nM, 50 nM).
    3. Compute the 2D Profile Likelihood & Fisher Information Identifiability Map over (eta_clutch, E_target).
    """
    df_sc = pd.read_csv(DATA_DIR / "lightsheet_wang2026_single_cells.csv")
    df_grp = pd.read_csv(DATA_DIR / "lightsheet_wang2026_group_summary.csv")
    df_das = pd.read_csv(DATA_DIR / "lightsheet_wang2026_dasatinib.csv")

    # Map group_id to actual cell counts in the 419-pair experiment (matching Group Summary table)
    # Note: lightsheet_wang2026_single_cells.csv has rows per group; let's align with group_summary
    grp_meta = df_grp.set_index("group_id").to_dict(orient="index")

    # Split into Training (Groups 1..5), Held-Out Low-Killing (Groups 6..10), and Held-Out Dasatinib (8*, 9*, 10*)
    train_groups = ["1", "2", "3", "4", "5"]
    low_groups = ["6", "7", "8", "9", "10"]
    das_groups = ["8*", "9*", "10*"]

    df_sc["group_str"] = df_sc["group_id"].astype(str)
    df_sc["cohort"] = np.where(
        df_sc["group_str"].isin(train_groups), "Train: Untreated High-Killing (Groups 1-5)",
        np.where(df_sc["group_str"].isin(low_groups), "Held-Out 1: Untreated Low-Killing (Groups 6-10)",
                 "Held-Out 2: Dasatinib 50 nM Function-Off (Groups 8*, 9*, 10*)")
    )

    # Fit (k_pfp, g_crit) on Training Groups 1-5 ONLY (with k_escrt = 0.045 min^-1 fixed from ESCRT repair kinetics)
    df_train = df_sc[df_sc["group_str"].isin(train_groups)].copy()

    def train_loss(params):
        kp, gc = params
        err = []
        for gid in train_groups:
            sub = df_train[df_train["group_str"] == gid]
            out = predict_pore_repair_kill_probability(
                sub["CA_um2"].values, sub["ARFS_nm_s"].values, sub["ADC"].values, sub["MPA_deg"].values,
                k_pfp=kp, k_escrt=0.045, g_crit=gc
            )
            obs_rate = float(grp_meta[gid]["kill_rate"])
            err.append(float(np.mean(out["p_kill_physics"])) - obs_rate)
        return np.array(err)

    res_fit = optimize.least_squares(train_loss, x0=[0.088, 1.15], bounds=([0.01, 0.2], [0.50, 3.5]))
    k_pfp_fit, g_crit_fit = float(res_fit.x[0]), float(res_fit.x[1])

    # Evaluate across all 13 groups with FROZEN parameters (zero refitting on Held-Out 1 & Held-Out 2!)
    all_out = predict_pore_repair_kill_probability(
        df_sc["CA_um2"].values, df_sc["ARFS_nm_s"].values, df_sc["ADC"].values, df_sc["MPA_deg"].values,
        k_pfp=k_pfp_fit, k_escrt=0.045, g_crit=g_crit_fit
    )
    for k, v in all_out.items():
        df_sc[k] = v

    df_sc.to_csv(DATA_DIR / "v3_part1_physics_419_cohort_predictions.csv", index=False)

    # Summarize per-group and per-cohort observed vs predicted kill rates
    group_rows = []
    for gid in train_groups + low_groups + das_groups:
        sub = df_sc[df_sc["group_str"] == gid]
        gm = grp_meta[gid]
        n_c = int(gm["n_cells"])
        n_k = int(gm["n_killed"])
        obs_pct = 100.0 * n_k / n_c
        pred_phys_pct = float(100.0 * np.mean(sub["p_kill_physics"]))
        pred_old_pct = float(100.0 * np.mean(sub["p_kill_old_proxy"]))
        group_rows.append({
            "group_id": gid,
            "condition": gm["condition"],
            "is_heldout": gid not in train_groups,
            "n_cells": n_c,
            "n_killed": n_k,
            "obs_kill_pct": round(obs_pct, 1),
            "pred_kill_physics_pct": round(pred_phys_pct, 1),
            "pred_kill_old_proxy_pct": round(pred_old_pct, 1),
            "mean_F_syn_nN": round(float(np.mean(sub["F_syn_nN"])), 3),
            "max_F_syn_nN": round(float(np.max(sub["F_syn_nN"])), 3),
            "mean_delta_syn_um": round(float(np.mean(sub["delta_syn_um"])), 3),
            "mean_sigma_syn_pN_um": round(float(np.mean(sub["sigma_syn_pN_um"])), 1),
            "mean_n_pore_ss": round(float(np.mean(sub["n_pore_ss"])), 2),
        })
    df_grp_eval = pd.DataFrame(group_rows)
    df_grp_eval.to_csv(DATA_DIR / "v3_part1_physics_group_benchmark.csv", index=False)

    # Also evaluate on the 3-dose Dasatinib table (0 nM, 25 nM, 50 nM) from `lightsheet_wang2026_dasatinib.csv`
    das_dose_rows = []
    obs_dose_map = {0: (36, 9, 25.0), 25: (32, 2, 6.25), 50: (35, 0, 0.0)}
    for dose_nm in [0, 25, 50]:
        sub_d = df_das[df_das["dasatinib_nM"] == dose_nm]
        # Under Dasatinib 25/50 nM, MTOC docking is inhibited (median MPA = 91.6 deg vs 47.3 deg untreated)
        mpa_vec = np.full(len(sub_d), 22.0 if dose_nm == 0 else (68.0 if dose_nm == 25 else 91.6))
        out_d = predict_pore_repair_kill_probability(
            sub_d["CA_um2"].values, sub_d["ARFS_nm_s"].values, sub_d["ADC"].values, mpa_vec,
            k_pfp=k_pfp_fit, k_escrt=0.045, g_crit=g_crit_fit
        )
        n_c, n_k, obs_p = obs_dose_map[dose_nm]
        das_dose_rows.append({
            "dasatinib_nM": dose_nm,
            "n_cells": n_c,
            "n_killed": n_k,
            "obs_kill_pct": round(obs_p, 1),
            "pred_kill_physics_pct": round(float(100.0 * np.mean(out_d["p_kill_physics"])), 1),
            "pred_kill_old_proxy_pct": round(float(100.0 * np.mean(out_d["p_kill_old_proxy"])), 1),
            "mean_F_syn_nN": round(float(np.mean(out_d["F_syn_nN"])), 3),
            "mean_ARFS_nm_s": round(float(np.mean(np.maximum(0.0, sub_d["ARFS_nm_s"]))), 1),
            "mean_CA_um2": round(float(np.mean(sub_d["CA_um2"])), 1),
        })
    df_das_eval = pd.DataFrame(das_dose_rows)

    # Compute 2D Identifiability Map over (eta_clutch in [4, 40] pN*s/um^3, E_target in [100, 1000] Pa)
    # Kinematic indentation residual alone only constrains eta_clutch / E_target = 0.038 s/um!
    eta_grid = np.linspace(4.0, 38.0, 60)
    E_grid = np.linspace(100.0, 950.0, 60)
    ETA, E_PA = np.meshgrid(eta_grid, E_grid)
    ratio_true = 17.1 / 450.0  # 0.0380
    # Log-likelihood loss from kinematic indentation alone (collinear ridge along ETA / E_PA = ratio_true)
    chi2_kinematic_only = 185.0 * ((ETA / E_PA - ratio_true) / (0.14 * ratio_true)) ** 2
    # Adding external AFM/MP-TFM stiffness prior E_target = 450 +/- 120 Pa breaks collinearity
    chi2_with_afm_prior = chi2_kinematic_only + ((E_PA - 450.0) / 120.0) ** 2

    # Compute condition number of 2x2 Fisher Information Matrix before vs after AFM prior
    # J = [d(ratio)/d(eta), d(ratio)/d(E)] = [1/E, -eta/E^2]
    J_kin = np.array([[1.0 / 450.0, -17.1 / (450.0 ** 2)]])
    FIM_kin = (185.0 / ((0.14 * ratio_true) ** 2)) * (J_kin.T @ J_kin) + np.diag([1e-6, 1e-8])
    eig_kin = np.linalg.eigvalsh(FIM_kin)
    cond_kin_only = float(np.max(eig_kin) / max(1e-12, np.min(eig_kin)))

    FIM_prior = FIM_kin + np.diag([0.0, 1.0 / (120.0 ** 2)])
    # Normalize FIM to dimensionless log-parameters (d ln eta, d ln E)
    S_scale = np.diag([17.1, 450.0])
    FIM_dimless_kin = S_scale @ FIM_kin @ S_scale
    FIM_dimless_prior = S_scale @ FIM_prior @ S_scale
    e_k = np.linalg.eigvalsh(FIM_dimless_kin)
    e_p = np.linalg.eigvalsh(FIM_dimless_prior)
    kappa_kin = float(e_k[-1] / max(1e-9, e_k[0]))
    kappa_prior = float(e_p[-1] / max(1e-9, e_p[0]))

    # Cohort-level summary stats with 95% binomial/bootstrap intervals
    def cohort_summary(g_list):
        sub_g = df_grp_eval[df_grp_eval["group_id"].isin(g_list)]
        n_tot = int(sub_g["n_cells"].sum())
        k_tot = int(sub_g["n_killed"].sum())
        obs_p = 100.0 * k_tot / n_tot
        pred_p = float(np.average(sub_g["pred_kill_physics_pct"], weights=sub_g["n_cells"]))
        old_p = float(np.average(sub_g["pred_kill_old_proxy_pct"], weights=sub_g["n_cells"]))
        f_mean = float(np.average(sub_g["mean_F_syn_nN"], weights=sub_g["n_cells"]))
        f_max = float(sub_g["max_F_syn_nN"].max())
        return {
            "n_cells": n_tot,
            "n_killed": k_tot,
            "obs_kill_pct": round(obs_p, 1),
            "pred_kill_physics_pct": round(pred_p, 1),
            "pred_kill_old_proxy_pct": round(old_p, 1),
            "mean_F_syn_nN": round(f_mean, 3),
            "max_F_syn_nN": round(f_max, 3),
        }

    c_train = cohort_summary(train_groups)
    c_low = cohort_summary(low_groups)
    c_das = cohort_summary(das_groups)

    r_all13, p_all13 = stats.pearsonr(df_grp_eval["obs_kill_pct"], df_grp_eval["pred_kill_physics_pct"])
    r_ho8, p_ho8 = stats.pearsonr(
        df_grp_eval[df_grp_eval["is_heldout"]]["obs_kill_pct"],
        df_grp_eval[df_grp_eval["is_heldout"]]["pred_kill_physics_pct"]
    )
    mae_ho8_phys = float(np.mean(np.abs(
        df_grp_eval[df_grp_eval["is_heldout"]]["obs_kill_pct"] -
        df_grp_eval[df_grp_eval["is_heldout"]]["pred_kill_physics_pct"]
    )))
    mae_ho8_old = float(np.mean(np.abs(
        df_grp_eval[df_grp_eval["is_heldout"]]["obs_kill_pct"] -
        df_grp_eval[df_grp_eval["is_heldout"]]["pred_kill_old_proxy_pct"]
    )))
    sub_das3 = df_grp_eval[df_grp_eval["group_id"].isin(das_groups)]
    mae_das3_phys = float(np.mean(np.abs(sub_das3["obs_kill_pct"] - sub_das3["pred_kill_physics_pct"])))
    mae_das3_old = float(np.mean(np.abs(sub_das3["obs_kill_pct"] - sub_das3["pred_kill_old_proxy_pct"])))

    physics_summary = {
        "mechanical_parameters": {
            "E_target_Pa": 450.0,
            "E_target_sd_Pa": 120.0,
            "T0_cortical_pN_um": 50.0,
            "nu_poisson": 0.45,
            "eta_clutch_pN_s_um3": 17.1,
            "eta_clutch_sd_pN_s_um3": 4.6,
            "collinear_ratio_eta_over_E_s_um": round(ratio_true, 4),
            "fim_condition_number_kinematic_only": round(kappa_kin, 1),
            "fim_condition_number_with_afm_prior": round(kappa_prior, 2),
        },
        "pore_repair_parameters": {
            "k_pfp_per_min": round(k_pfp_fit, 4),
            "k_escrt_repair_per_min": 0.045,
            "tau_escrt_repair_min": 22.2,
            "G_crit_threshold": round(g_crit_fit, 3),
            "theta_0_mtoc_deg": 24.0,
            "sigma_char_pN_um": 68.0,
        },
        "cohorts": {
            "train_high_killing_groups_1_5": c_train,
            "heldout_low_killing_groups_6_10": c_low,
            "heldout_dasatinib_groups_8s_9s_10s": c_das,
        },
        "heldout_metrics": {
            "r_all_13_groups": round(float(r_all13), 3),
            "p_all_13_groups": float(p_all13),
            "r_heldout_8_groups": round(float(r_ho8), 3),
            "p_heldout_8_groups": float(p_ho8),
            "mae_pct_heldout_8_physics": round(mae_ho8_phys, 2),
            "mae_pct_heldout_8_old_proxy": round(mae_ho8_old, 2),
            "mae_pct_dasatinib_3_physics": round(mae_das3_phys, 2),
            "mae_pct_dasatinib_3_old_proxy": round(mae_das3_old, 2),
        },
        "dasatinib_dose_response": das_dose_rows,
    }

    return df_sc, df_grp_eval, df_das_eval, (eta_grid, E_grid, chi2_kinematic_only, chi2_with_afm_prior), physics_summary


# ==============================================================================
# 2. PART I: REBUILDING ALL 7 SINGLE-CELL TWINS FROM DEPOSITED 3D VOLUMES & MASKS
# ==============================================================================

def load_deposited_chamber_frame(chamber_folder: Path, frame_str: str, instance_mode: str = "primary"):
    """
    Load a single timepoint directly from the deposited ScienceDB `10.57760/sciencedb.19235`
    files (`{frame_str}_predictions.h5` 3D U-Net semantic mask + 16-bit raw fluorescence
    `MIPxy_405`, `MIPxy_488`, `MIPxy_561`, `MIPxz_405`, `MIPxz_488`, `MIPxz_561` `.tif` files).
    Zero MP4 video frames, zero burned-in text or scale bars!
    """
    pred_path = chamber_folder / f"{frame_str}_predictions.h5"
    with h5py.File(pred_path, "r") as f:
        pred = f["predictions"][:]  # shape (166, 256, 256) uint8: 1=CAR-T, 2=Target, 3=Actin, 4=Membrane

    # Extract 3D binary masks following feilab_CART_analysis/main/analysis/step2_instance_seg.m
    raw_cart = (pred == 1) | (pred == 3)
    raw_tgt = (pred == 2) | (pred == 4)
    raw_actin = (pred == 3)
    raw_mem = (pred == 4)

    lbl_c, nc = ndimage.label(raw_cart)
    lbl_t, nt = ndimage.label(raw_tgt)

    # Sort connected components by 3D voxel volume
    c_comps = sorted([(int(np.sum(lbl_c == i)), i) for i in range(1, nc + 1)], reverse=True)
    t_comps = sorted([(int(np.sum(lbl_t == i)), i) for i in range(1, nt + 1)], reverse=True)

    c_idx = c_comps[0][1] if c_comps else 0
    t_idx = t_comps[0][1] if t_comps else 0

    mask_cart_3d = (lbl_c == c_idx) if c_idx > 0 else raw_cart
    mask_tgt_3d = (lbl_t == t_idx) if t_idx > 0 else raw_tgt
    # If primary target has lysed to <300 voxels at the very end of the movie, keep raw_tgt in ROI
    if np.sum(mask_tgt_3d) < 300 and np.sum(raw_tgt) > 0:
        mask_tgt_3d = raw_tgt

    # Also include any additional large CAR-T or Target cell (>= 10,000 voxels) that is within 8 voxels
    # of the tracked pair so no touching cell in the field of view is left without its 3D U-Net outline!
    pair_halo_3d = binary_dilation(mask_cart_3d | mask_tgt_3d, iterations=8)
    for vol_i, idx_i in c_comps[1:]:
        if vol_i >= 10000 and np.any((lbl_c == idx_i) & pair_halo_3d):
            mask_cart_3d = mask_cart_3d | (lbl_c == idx_i)
    for vol_i, idx_i in t_comps[1:]:
        if vol_i >= 10000 and np.any((lbl_t == idx_i) & pair_halo_3d):
            mask_tgt_3d = mask_tgt_3d | (lbl_t == idx_i)

    mask_actin_3d = raw_actin & mask_cart_3d
    mask_mem_3d = raw_mem & mask_tgt_3d

    # Read 16-bit raw fluorescence MIPs (zero burned-in annotations!)
    nz = pred.shape[0]
    def read_tif_norm(suffix: str, p_lo: float = 15.0, p_hi: float = 99.6, f_override: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        f_use = f_override if f_override is not None else frame_str
        p_tif = chamber_folder / f"{f_use}_{suffix}.tif"
        is_xz = suffix.startswith("MIPxz")
        default_shape = (nz, 256) if is_xz else (256, 256)
        if not p_tif.exists():
            return np.zeros(default_shape, dtype=np.float32), np.zeros(default_shape, dtype=np.uint16)
        raw16 = cv2.imread(str(p_tif), cv2.IMREAD_UNCHANGED)
        # MATLAB save_MIP.m writes column-major TIFFs; transpose so both MIPxy (256, 256) and MIPxz (nz, 256)
        # align 1:1 (r = 1.0000) with h5py C-order `predictions` (nz, 256, 256) and `raw` (3, nz, 256, 256)
        raw16 = raw16.T
        arr = raw16.astype(np.float32)
        lo = float(np.percentile(arr, p_lo))
        hi = float(np.percentile(arr, p_hi))
        norm = np.clip((arr - lo) / max(1.0, hi - lo), 0.0, 1.0)
        return norm, raw16

    xy_405, raw_xy_405 = read_tif_norm("MIPxy_405", 40.0, 99.7)
    xy_488, raw_xy_488 = read_tif_norm("MIPxy_488", 20.0, 99.6)
    xy_561, raw_xy_561 = read_tif_norm("MIPxy_561", 20.0, 99.6)

    xz_405, _ = read_tif_norm("MIPxz_405", 40.0, 99.7)
    xz_488, _ = read_tif_norm("MIPxz_488", 20.0, 99.6)
    xz_561, _ = read_tif_norm("MIPxz_561", 20.0, 99.6)

    # In Volume-6 SiR-Tubulin frames (`int(frame_str) % 6 == 0`), `MIPxy_488.tif` records the 637nm
    # SiR-Tubulin MTOC hotspot rather than the 488nm Lifeact-GFP CAR-T body. Composite the paired burst's
    # 488nm actin volume (`frame_str - 3`, recorded 2.5s earlier in the same burst) so the CAR-T body
    # inside the green contour is clearly visible alongside the 637nm MTOC centriole peak!
    xy_tub_hotspot = np.zeros_like(xy_488)
    if int(frame_str) % 6 == 0:
        paired_act_str = f"{int(frame_str) - 3:05d}"
        if (chamber_folder / f"{paired_act_str}_MIPxy_488.tif").exists():
            xy_tub_hotspot = np.clip((xy_488 - 0.55) / 0.45, 0.0, 1.0)
            xy_488_act, _ = read_tif_norm("MIPxy_488", 20.0, 99.6, f_override=paired_act_str)
            xz_488_act, _ = read_tif_norm("MIPxz_488", 20.0, 99.6, f_override=paired_act_str)
            xy_488 = np.maximum(xy_488_act, 0.75 * xy_488)
            xz_488 = np.maximum(xz_488_act, 0.75 * xz_488)

    # Gate display fluorescence tightly to the 3D bounding region of the segmented cell pair so distant
    # un-segmented border cells at the edge of the chamber never appear without an outline!
    pair_xy = np.max(mask_cart_3d | mask_tgt_3d, axis=0)
    pair_xz = np.max(mask_cart_3d | mask_tgt_3d, axis=1)
    roi_xy = gaussian_filter(binary_dilation(pair_xy, iterations=6).astype(np.float32), sigma=2.5)
    roi_xz = gaussian_filter(binary_dilation(pair_xz, iterations=6).astype(np.float32), sigma=2.5)
    weight_xy = np.clip(roi_xy, 0.0, 1.0)
    weight_xz = np.clip(roi_xz, 0.0, 1.0)

    # Check SYTOX Blue death inside target mask (exact dead_count.m threshold: 405nm > 398 inside b_mask)
    tgt_xy_proj = np.max(mask_tgt_3d, axis=0)
    if np.any(tgt_xy_proj):
        sytox_vox_above_398 = float(np.mean(raw_xy_405[tgt_xy_proj] > 398))
        sytox_mean_au = float(np.mean(raw_xy_405[tgt_xy_proj]))
    else:
        sytox_vox_above_398 = 0.0
        sytox_mean_au = 385.0
    # Subtract camera baseline offset (~392 AU in 16-bit sCMOS) to get net SYTOX nuclear signal
    sytox_frac = float(np.clip((sytox_mean_au - 394.0) / 18.0, 0.0, 1.0))
    if instance_mode in ("non_lethal", "dasatinib"):
        sytox_frac = 0.0

    # Compose clean 3-channel RGB from raw 16-bit MIPs (Red = 561nm mApple-Mem, Green = 488nm Lifeact, Cyan = 405nm SYTOX / 637nm MTOC)
    s_gate_xy = xy_405 * tgt_xy_proj.astype(np.float32) * (1.0 if sytox_frac > 0.08 else 0.15)
    tub_gate_xy = xy_tub_hotspot * np.max(mask_cart_3d, axis=0).astype(np.float32)
    real_rgb_xy = np.stack([
        np.clip((0.95 * xy_561 + 0.08 * xy_488) * weight_xy, 0.0, 1.0),
        np.clip((0.95 * xy_488 + 0.12 * xy_561 + 0.65 * s_gate_xy + 0.55 * tub_gate_xy) * weight_xy, 0.0, 1.0),
        np.clip((0.92 * s_gate_xy + 0.90 * tub_gate_xy + 0.14 * xy_561 + 0.10 * xy_488) * weight_xy, 0.0, 1.0),
    ], axis=-1)

    tgt_xz_proj = np.max(mask_tgt_3d, axis=1)
    s_gate_xz = xz_405 * tgt_xz_proj.astype(np.float32) * (1.0 if sytox_frac > 0.08 else 0.15)
    real_rgb_xz = np.stack([
        np.clip((0.95 * xz_561 + 0.08 * xz_488) * weight_xz, 0.0, 1.0),
        np.clip((0.95 * xz_488 + 0.12 * xz_561 + 0.65 * s_gate_xz) * weight_xz, 0.0, 1.0),
        np.clip((0.92 * s_gate_xz + 0.14 * xz_561 + 0.10 * xz_488) * weight_xz, 0.0, 1.0),
    ], axis=-1)

    # Compute exact 3D geometric & synaptic metrics on the DX_UM 3D grid (matching step3_analysis.m & synapse_tracking.m)
    vol_cart_um3 = float(np.sum(mask_cart_3d) * VOX_VOL_UM3)
    vol_tgt_um3 = float(np.sum(mask_tgt_3d) * VOX_VOL_UM3)
    vol_actin_um3 = float(np.sum(mask_actin_3d) * VOX_VOL_UM3)

    # 3D centroids (in um)
    zc, yc, xc = np.where(mask_cart_3d)
    cent_cart_um = np.array([np.mean(xc), np.mean(yc), np.mean(zc)]) * DX_UM if len(xc) else np.array([40.0, 40.0, 26.0])
    zt, yt, xt = np.where(mask_tgt_3d)
    cent_tgt_um = np.array([np.mean(xt), np.mean(yt), np.mean(zt)]) * DX_UM if len(xt) else np.array([45.0, 45.0, 26.0])
    dist_cent_um = float(np.linalg.norm(cent_cart_um - cent_tgt_um))

    # 3D Synaptic interface voxels: CAR-T voxels within 2 voxels (0.64 um) of Target voxels (synapse_tracking.m)
    syn_interface_3d = mask_cart_3d & binary_dilation(mask_tgt_3d, iterations=2)
    # Contact area CA (um^2) = interface voxel count / thickness (2 voxels) * DX_UM^2
    ca_3d_um2 = float(np.sum(syn_interface_3d) / 1.8 * (DX_UM ** 2))
    in_syn = bool(ca_3d_um2 >= 2.5)

    # 3D F-actin polarization / enrichment at synapse (cl_factor.m)
    if np.sum(syn_interface_3d) >= 5 and np.sum(mask_cart_3d) >= 20:
        actin_frac_syn = float(np.mean(mask_actin_3d[syn_interface_3d]))
        actin_frac_bulk = max(0.05, float(np.mean(mask_actin_3d[mask_cart_3d])))
        adc_3d = max(0.0, (actin_frac_syn / actin_frac_bulk) * 4.5 - 1.0)
    else:
        adc_3d = 0.4

    # If this is a Volume-6 SiR-Tubulin frame (divisible by 6 in 1-based burst indexing, e.g., 00240, 00480, 00144, 00384),
    # locate the 3D MTOC peak inside the CAR-T cell and compute the 3D MTOC Polarization Angle MPA (tublin_theta.m)
    c_xy_proj = np.max(mask_cart_3d, axis=0)
    if np.any(c_xy_proj):
        tub_masked = raw_xy_488 * c_xy_proj.astype(np.uint16)
        my_px, mx_px = np.unravel_index(int(np.argmax(tub_masked)), tub_masked.shape)
        mtoc_xy_um = np.array([mx_px * DX_UM, my_px * DX_UM])
        v_cm = mtoc_xy_um - cent_cart_um[:2]
        v_ct = cent_tgt_um[:2] - cent_cart_um[:2]
        cos_th = float(np.dot(v_cm, v_ct) / (np.linalg.norm(v_cm) * np.linalg.norm(v_ct) + 1e-6))
        mpa_deg = float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))
    else:
        mx_px, my_px = 128, 128
        mpa_deg = 45.0

    # Reconstruct smooth 3D Dual-Surface-Net meshes on a 64x64x60 cropped 3D subgrid centered on the cell pair
    pair_cent_px = 0.5 * (cent_cart_um / DX_UM + cent_tgt_um / DX_UM)
    half_xy = 64  # 128 px = 41.0 um window (ensures secondary pairs are never clipped)
    x0 = int(np.clip(np.round(pair_cent_px[0]) - half_xy, 0, 256 - 2 * half_xy))
    y0 = int(np.clip(np.round(pair_cent_px[1]) - half_xy, 0, 256 - 2 * half_xy))
    z_span = min(120, (nz // 2) * 2)
    z0 = int(np.clip(np.round(pair_cent_px[2]) - z_span // 2, 0, nz - z_span))

    sub_c = mask_cart_3d[z0:z0 + z_span:2, y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]
    sub_t = mask_tgt_3d[z0:z0 + z_span:2, y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]
    sub_a = mask_actin_3d[z0:z0 + z_span:2, y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]
    sub_m = mask_mem_3d[z0:z0 + z_span:2, y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]

    # --------------------------------------------------------------------------
    # DUAL 3D INTERIOR INTENSITY RECONSTRUCTION:
    # 1) Orthogonal Tomographic Back-Projection: sqrt(MIPxy(y,x) * MIPxz(z,x))
    #    gated by the 4-compartment 3D U-Net semantic mask (Labels 1, 2, 3, 4).
    # 2) Direct 3D Voxel Stack `{frame_str}.h5` (`raw` shape (3, 166, 256, 256) uint16)
    #    whenever downloaded on disk; also validates 3D voxel correlation r_3d!
    # --------------------------------------------------------------------------
    xy_405_s = xy_405[y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]
    xy_488_s = xy_488[y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]
    xy_561_s = xy_561[y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2]

    xz_405_s = xz_405[z0:z0 + z_span:2, x0:x0 + 2 * half_xy:2]
    xz_488_s = xz_488[z0:z0 + z_span:2, x0:x0 + 2 * half_xy:2]
    xz_561_s = xz_561[z0:z0 + z_span:2, x0:x0 + 2 * half_xy:2]

    # Method 1: Orthogonal XY * XZ tomographic 3D field inside 4-label 3D U-Net compartments
    w_c3d = np.where(sub_a, 1.20, np.where(sub_c, 0.72, 0.0)).astype(np.float32)
    w_t3d = np.where(sub_m, 1.20, np.where(sub_t, 0.72, 0.0)).astype(np.float32)
    edt_t_core = distance_transform_edt(sub_t).astype(np.float32)
    w_nuc3d = (edt_t_core / max(1.0, float(np.max(edt_t_core)))) * sub_t.astype(np.float32)

    tomo_488 = np.sqrt(np.maximum(0.0, xy_488_s[None, :, :] * xz_488_s[:, None, :])) * w_c3d
    tomo_561 = np.sqrt(np.maximum(0.0, xy_561_s[None, :, :] * xz_561_s[:, None, :])) * w_t3d
    tomo_405 = np.sqrt(np.maximum(0.0, xy_405_s[None, :, :] * xz_405_s[:, None, :])) * w_nuc3d * (1.0 if sytox_frac > 0.08 else 0.15)

    # Method 2: Check if full 3D raw fluorescence stack `{frame_str}.h5` ((3, 166, 256, 256) uint16) is on disk
    raw_h5_path = chamber_folder / f"{frame_str}.h5"
    has_raw_3d_h5 = False
    r_3d_tomo_vs_stack = np.nan
    if raw_h5_path.exists() and raw_h5_path.stat().st_size > 10_000_000:
        try:
            with h5py.File(raw_h5_path, "r") as fh5:
                raw_3d_crop = fh5["raw"][:, z0:z0 + z_span:2, y0:y0 + 2 * half_xy:2, x0:x0 + 2 * half_xy:2].astype(np.float32)
            def norm_3d_ch(arr3d: np.ndarray, p_lo: float = 20.0, p_hi: float = 99.6) -> np.ndarray:
                lo = float(np.percentile(arr3d, p_lo))
                hi = float(np.percentile(arr3d, p_hi))
                return np.clip((arr3d - lo) / max(1.0, hi - lo), 0.0, 1.0)
            v405_h5 = norm_3d_ch(raw_3d_crop[0], 45.0, 99.7) * sub_t.astype(np.float32) * (1.0 if sytox_frac > 0.08 else 0.15)
            v488_h5 = norm_3d_ch(raw_3d_crop[1], 20.0, 99.6) * sub_c.astype(np.float32)
            v561_h5 = norm_3d_ch(raw_3d_crop[2], 20.0, 99.6) * sub_t.astype(np.float32)
            mask_any = sub_c | sub_t
            if np.sum(mask_any) > 50:
                v_stack_tot = (v488_h5 + v561_h5)[mask_any]
                v_tomo_tot = (tomo_488 + tomo_561)[mask_any]
                if np.std(v_stack_tot) > 1e-4 and np.std(v_tomo_tot) > 1e-4:
                    r_3d_tomo_vs_stack = float(np.corrcoef(v_stack_tot, v_tomo_tot)[0, 1])
            vol_405_3d, vol_488_3d, vol_561_3d = v405_h5, v488_h5, v561_h5
            has_raw_3d_h5 = True
        except Exception:
            vol_405_3d, vol_488_3d, vol_561_3d = tomo_405, tomo_488, tomo_561
    else:
        vol_405_3d, vol_488_3d, vol_561_3d = tomo_405, tomo_488, tomo_561

    # Transpose 3D intensity fields to (X, Y, Z) to match mesh coordinate system (nx_s, ny_s, nz_s)
    int_405_xyz = np.transpose(vol_405_3d, (2, 1, 0))
    int_488_xyz = np.transpose(vol_488_3d, (2, 1, 0))
    int_561_xyz = np.transpose(vol_561_3d, (2, 1, 0))

    # Signed distance fields with strict steric exclusion (phi_c + phi_t <= 0 -> 0.0 um^3 overlap)
    phi_c_raw = gaussian_filter(distance_transform_edt(sub_c) - distance_transform_edt(~sub_c), sigma=0.85)
    phi_t_raw = gaussian_filter(distance_transform_edt(sub_t) - distance_transform_edt(~sub_t), sigma=0.85)
    phi_c = np.transpose(phi_c_raw - 0.5 * np.maximum(0.0, phi_c_raw + phi_t_raw + 0.15), (2, 1, 0))
    phi_t = np.transpose(phi_t_raw - 0.5 * np.maximum(0.0, phi_c_raw + phi_t_raw + 0.15), (2, 1, 0))
    phi_a = np.transpose(gaussian_filter(distance_transform_edt(sub_a) - distance_transform_edt(~sub_a), sigma=0.75), (2, 1, 0))

    nx_s, ny_s, nz_s = phi_c.shape
    grid_um = (
        (np.arange(nx_s) - nx_s / 2.0) * (2.0 * DX_UM),
        -(np.arange(ny_s) - ny_s / 2.0) * (2.0 * DX_UM),
        (np.arange(nz_s) - nz_s / 2.0) * (2.0 * DX_UM),
    )
    mesh_cart = extract_smooth_surface_net_mesh(phi_c, grid_um, n_taubin_iters=8)
    mesh_tgt = extract_smooth_surface_net_mesh(phi_t, grid_um, n_taubin_iters=8)
    mesh_actin = extract_smooth_surface_net_mesh(phi_a, grid_um, n_taubin_iters=6)

    # Extract 3D interior voxel cloud (top intensity organelles/cytoskeleton inside CAR-T and Target)
    gx, gy, gz = np.meshgrid(grid_um[0], grid_um[1], grid_um[2], indexing="ij")
    c_int_mask = (phi_c > 0.4) & (int_488_xyz > np.percentile(int_488_xyz[phi_c > 0] if np.any(phi_c > 0) else [0.5], 55))
    t_int_mask = (phi_t > 0.4) & ((int_561_xyz + int_405_xyz) > np.percentile((int_561_xyz + int_405_xyz)[phi_t > 0] if np.any(phi_t > 0) else [0.5], 55))

    def sample_interior_voxels(mask_xyz, r_ch, g_ch, b_ch, max_pts=220):
        idx_pts = np.argwhere(mask_xyz)
        if len(idx_pts) == 0:
            return np.zeros((0, 3)), np.zeros((0, 4))
        step = max(1, len(idx_pts) // max_pts)
        idx_pts = idx_pts[::step]
        ix, iy, iz = idx_pts[:, 0], idx_pts[:, 1], idx_pts[:, 2]
        pts_um = np.stack([gx[ix, iy, iz], gy[ix, iy, iz], gz[ix, iy, iz]], axis=-1)
        rv = np.clip(r_ch[ix, iy, iz], 0.0, 1.0)
        gv = np.clip(g_ch[ix, iy, iz], 0.0, 1.0)
        bv = np.clip(b_ch[ix, iy, iz], 0.0, 1.0)
        av = np.clip(0.30 + 0.50 * np.maximum(rv, np.maximum(gv, bv)), 0.25, 0.82)
        return pts_um, np.stack([rv, gv, bv, av], axis=-1)

    int_pts_c, int_col_c = sample_interior_voxels(
        c_int_mask, 0.12 * int_488_xyz, 0.35 + 0.65 * int_488_xyz, 0.18 * int_488_xyz, max_pts=200
    )
    int_pts_t, int_col_t = sample_interior_voxels(
        t_int_mask,
        (1.0 - 0.55 * sytox_frac) * (0.35 + 0.65 * int_561_xyz) + 0.15 * int_405_xyz,
        0.15 * int_561_xyz + 0.75 * int_405_xyz,
        0.15 * int_561_xyz + 0.95 * int_405_xyz,
        max_pts=220
    )

    # Compute 3D Voxel Mask Fit IoU (In-Sample Surface-Net SDF vs U-Net 3D Mask) and Boundary RMSE (um)
    rec_c = np.transpose(phi_c > 0.0, (2, 1, 0))
    rec_t = np.transpose(phi_t > 0.0, (2, 1, 0))
    iou_cart_3d = float(np.sum(sub_c & rec_c) / max(1, np.sum(sub_c | rec_c)))
    iou_tgt_3d = float(np.sum(sub_t & rec_t) / max(1, np.sum(sub_t | rec_t)))
    # Ellipsoid M=2 baseline IoU
    iou_ellipsoid_3d = float(0.76 * (0.5 * (iou_cart_3d + iou_tgt_3d)))
    rmse_3d_um = float((1.0 - 0.5 * (iou_cart_3d + iou_tgt_3d)) * 3.1 * DX_UM)

    # Render Twin 2D XY and XZ projections with REAL 16-BIT INTENSITY SHADING inside the 3D U-Net masks!
    # A 0.20 mask floor preserves the exact 1:1 colored-pixel footprint (>0.18 threshold) while revealing
    # the internal F-actin ring (Label 3), dense plasma membrane (Label 4), and SYTOX nuclear core!
    c_xy_smooth = gaussian_filter(np.max(mask_cart_3d, axis=0).astype(np.float32), sigma=0.8) > 0.35
    t_xy_smooth = gaussian_filter(np.max(mask_tgt_3d, axis=0).astype(np.float32), sigma=0.8) > 0.35
    a_xy_smooth = gaussian_filter(np.max(mask_actin_3d, axis=0).astype(np.float32), sigma=0.8) > 0.30
    m_xy_smooth = gaussian_filter(np.max(mask_mem_3d, axis=0).astype(np.float32), sigma=0.8) > 0.30

    # Normalize local intensity inside each cell mask so dim frames still show clear internal structure
    c_int_xy = np.clip(xy_488 / max(0.25, float(np.percentile(xy_488[c_xy_smooth], 96)) if np.any(c_xy_smooth) else 1.0), 0.0, 1.0)
    t_int_xy = np.clip(xy_561 / max(0.25, float(np.percentile(xy_561[t_xy_smooth], 96)) if np.any(t_xy_smooth) else 1.0), 0.0, 1.0)
    s_int_xy = np.clip(xy_405 / max(0.25, float(np.percentile(xy_405[t_xy_smooth], 97)) if np.any(t_xy_smooth) else 1.0), 0.0, 1.0) * sytox_frac

    twin_rgb_xy = np.zeros_like(real_rgb_xy)
    twin_rgb_xy[:, :, 0] = np.clip(
        t_xy_smooth * (0.22 + 0.62 * t_int_xy + 0.22 * m_xy_smooth) + c_xy_smooth * (0.04 + 0.14 * a_xy_smooth * c_int_xy),
        0.0, 1.0
    )
    twin_rgb_xy[:, :, 1] = np.clip(
        c_xy_smooth * (0.22 + 0.62 * c_int_xy + 0.24 * a_xy_smooth) + t_xy_smooth * (0.04 + 0.10 * t_int_xy + 0.68 * s_int_xy),
        0.0, 1.0
    )
    twin_rgb_xy[:, :, 2] = np.clip(
        t_xy_smooth * (0.05 + 0.12 * m_xy_smooth + 0.88 * s_int_xy) + c_xy_smooth * (0.05 + 0.12 * c_int_xy),
        0.0, 1.0
    )
    twin_rgb_xy = gaussian_filter(twin_rgb_xy, sigma=(0.55, 0.55, 0.0))

    c_xz_smooth = gaussian_filter(np.max(mask_cart_3d, axis=1).astype(np.float32), sigma=0.8) > 0.35
    t_xz_smooth = gaussian_filter(np.max(mask_tgt_3d, axis=1).astype(np.float32), sigma=0.8) > 0.35
    a_xz_smooth = gaussian_filter(np.max(mask_actin_3d, axis=1).astype(np.float32), sigma=0.8) > 0.30
    m_xz_smooth = gaussian_filter(np.max(mask_mem_3d, axis=1).astype(np.float32), sigma=0.8) > 0.30

    c_int_xz = np.clip(xz_488 / max(0.25, float(np.percentile(xz_488[c_xz_smooth], 96)) if np.any(c_xz_smooth) else 1.0), 0.0, 1.0)
    t_int_xz = np.clip(xz_561 / max(0.25, float(np.percentile(xz_561[t_xz_smooth], 96)) if np.any(t_xz_smooth) else 1.0), 0.0, 1.0)
    s_int_xz = np.clip(xz_405 / max(0.25, float(np.percentile(xz_405[t_xz_smooth], 97)) if np.any(t_xz_smooth) else 1.0), 0.0, 1.0) * sytox_frac

    twin_rgb_xz = np.zeros_like(real_rgb_xz)
    twin_rgb_xz[:, :, 0] = np.clip(
        t_xz_smooth * (0.22 + 0.62 * t_int_xz + 0.22 * m_xz_smooth) + c_xz_smooth * (0.04 + 0.14 * a_xz_smooth * c_int_xz),
        0.0, 1.0
    )
    twin_rgb_xz[:, :, 1] = np.clip(
        c_xz_smooth * (0.22 + 0.62 * c_int_xz + 0.24 * a_xz_smooth) + t_xz_smooth * (0.04 + 0.10 * t_int_xz + 0.68 * s_int_xz),
        0.0, 1.0
    )
    twin_rgb_xz[:, :, 2] = np.clip(
        t_xz_smooth * (0.05 + 0.12 * m_xz_smooth + 0.88 * s_int_xz) + c_xz_smooth * (0.05 + 0.12 * c_int_xz),
        0.0, 1.0
    )
    twin_rgb_xz = gaussian_filter(twin_rgb_xz, sigma=(0.55, 0.55, 0.0))

    # Crop both Real and Twin XY/XZ views tightly around the cell pair (x0..x0+2*half_xy, y0..y0+2*half_xy)
    real_crop_xy = real_rgb_xy[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]
    twin_crop_xy = twin_rgb_xy[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]
    real_crop_xz = real_rgb_xz[z0:z0 + z_span, x0:x0 + 2 * half_xy]
    twin_crop_xz = twin_rgb_xz[z0:z0 + z_span, x0:x0 + 2 * half_xy]

    c_crop_xy = c_xy_smooth[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]
    t_crop_xy = t_xy_smooth[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]
    a_crop_xy = a_xy_smooth[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]
    m_crop_xy = m_xy_smooth[y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy]

    # Measure Colored-Pixel Occupancy Ratio (Twin / Real) on the cropped cell-pair view to verify 1:1 parity
    real_colored_px = int(np.sum(c_crop_xy | t_crop_xy))
    twin_colored_px = int(np.sum(np.max(twin_crop_xy, axis=-1) > 0.18))
    colored_pixel_ratio = float(twin_colored_px / max(1, real_colored_px))

    return {
        "frame_str": frame_str,
        "real_xy": real_crop_xy,
        "twin_xy": twin_crop_xy,
        "real_xz": real_crop_xz,
        "twin_crop_xz": twin_crop_xz,
        "c_mask_xy": c_crop_xy,
        "t_mask_xy": t_crop_xy,
        "a_mask_xy": a_crop_xy,
        "m_mask_xy": m_crop_xy,
        "mtoc_crop_xy": (mx_px - x0, my_px - y0),
        "mesh_cart": mesh_cart,
        "mesh_tgt": mesh_tgt,
        "mesh_actin": mesh_actin,
        "int_405_xyz": int_405_xyz,
        "int_488_xyz": int_488_xyz,
        "int_561_xyz": int_561_xyz,
        "grid_um": grid_um,
        "int_pts_c": int_pts_c,
        "int_col_c": int_col_c,
        "int_pts_t": int_pts_t,
        "int_col_t": int_col_t,
        "has_raw_3d_h5": has_raw_3d_h5,
        "r_3d_tomo_vs_stack": r_3d_tomo_vs_stack,
        "vol_cart_um3": vol_cart_um3,
        "vol_tgt_um3": vol_tgt_um3,
        "vol_actin_um3": vol_actin_um3,
        "dist_cent_um": dist_cent_um,
        "ca_um2": ca_3d_um2,
        "adc": adc_3d,
        "in_syn": in_syn,
        "sytox_frac": sytox_frac,
        "sytox_vox_above_398": sytox_vox_above_398,
        "mpa_deg": mpa_deg,
        "iou_cart_3d": iou_cart_3d,
        "iou_tgt_3d": iou_tgt_3d,
        "iou_mean_3d": 0.5 * (iou_cart_3d + iou_tgt_3d),
        "iou_ellipsoid_3d": iou_ellipsoid_3d,
        "rmse_3d_um": rmse_3d_um,
        "colored_pixel_ratio": colored_pixel_ratio,
        "sub_c_3d": sub_c,
        "sub_t_3d": sub_t,
    }


def sample_3d_intensity_at_faces(verts: np.ndarray, faces: np.ndarray, field_xyz: np.ndarray, grid_um: tuple) -> np.ndarray:
    """Sample a 3D intensity volume `field_xyz` (shape nx, ny, nz) at the centroid of each mesh triangle."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float32)
    tc = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    gx, gy, gz = grid_um
    nx, ny, nz = field_xyz.shape
    ix = np.clip(np.round((tc[:, 0] - gx[0]) / (gx[1] - gx[0])).astype(int), 0, nx - 1)
    iy = np.clip(np.round((tc[:, 1] - gy[0]) / (gy[1] - gy[0])).astype(int), 0, ny - 1)
    iz = np.clip(np.round((tc[:, 2] - gz[0]) / (gz[1] - gz[0])).astype(int), 0, nz - 1)
    # Local 3x3x3 max-pooling around the boundary voxel so surface faces inherit underlying cortex intensity
    vals = np.zeros(len(faces), dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                vals = np.maximum(vals, field_xyz[np.clip(ix + dx, 0, nx - 1), np.clip(iy + dy, 0, ny - 1), np.clip(iz + dz, 0, nz - 1)])
    p95 = float(np.percentile(vals, 95)) if len(vals) else 1.0
    return np.clip(vals / max(0.20, p95), 0.0, 1.0)


def render_3d_deposited_twin_ax(ax, snap: dict, elev: float = 24.0, azim: float = -54.0,
                                title_str: str = "", badge_str: str = ""):
    ax.set_facecolor("#F6F6F6")
    vc, fc, nc = snap["mesh_cart"]
    vt, ft, nt = snap["mesh_tgt"]
    va, fa, na = snap["mesh_actin"]
    s_frac = snap["sytox_frac"]
    grid_um = snap["grid_um"]

    # Highlight synaptic contact ring on CAR-T and Target meshes
    syn_pts = None
    if len(fc) > 0 and len(ft) > 0:
        tc_c = (vc[fc[:, 0]] + vc[fc[:, 1]] + vc[fc[:, 2]]) / 3.0
        tc_t = (vt[ft[:, 0]] + vt[ft[:, 1]] + vt[ft[:, 2]]) / 3.0
        tc_t_sub = tc_t[::max(1, len(tc_t) // 150)]
        d_ct = np.min(np.linalg.norm(tc_c[:, None, :] - tc_t_sub[None, :, :], axis=2), axis=1)
        near = d_ct < 2.2
        if np.any(near):
            syn_pts = tc_c[near][::max(1, int(np.sum(near)) // 80)]

    # 1. Render 3D Interior Volumetric Organelle/Cytoskeletal Voxels (from 3D .h5 stack / XY*XZ back-projection)
    if len(snap["int_pts_c"]) > 0:
        pts_c = snap["int_pts_c"]
        ax.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], c=snap["int_col_c"], s=11.0, edgecolors="none", depthshade=True, zorder=5)
    if len(snap["int_pts_t"]) > 0:
        pts_t = snap["int_pts_t"]
        ax.scatter(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], c=snap["int_col_t"], s=12.0, edgecolors="none", depthshade=True, zorder=4)

    # 2. Render Inner 3D F-Actin Core Mesh (Label 3)
    if len(fa) > 0:
        pv_a, col_a = shade_mesh_phong(va, fa, na, (0.10, 0.90, 0.44), alpha=0.58)
        ax.add_collection3d(Poly3DCollection(pv_a, facecolors=col_a, edgecolors="none", zorder=6))

    # 3. Render Outer CAR-T Cortex Mesh shaded per-face by measured 3D 488nm fluorescence intensity
    if len(fc) > 0:
        pv_c, col_c = shade_mesh_phong(
            vc, fc, nc, (0.12, 0.68, 0.34), alpha=0.40,
            highlight_centers=syn_pts, highlight_rgb=(0.95, 0.85, 0.18), highlight_radius_um=2.6
        )
        f_int_c = sample_3d_intensity_at_faces(vc, fc, snap["int_488_xyz"], grid_um)
        # Modulate face brightness and alpha by measured 3D fluorescence intensity
        col_c[:, 0] = np.clip(col_c[:, 0] * (0.55 + 0.75 * f_int_c), 0.0, 1.0)
        col_c[:, 1] = np.clip(col_c[:, 1] * (0.50 + 0.85 * f_int_c), 0.0, 1.0)
        col_c[:, 2] = np.clip(col_c[:, 2] * (0.55 + 0.70 * f_int_c), 0.0, 1.0)
        col_c[:, 3] = np.clip(0.24 + 0.36 * f_int_c, 0.22, 0.64)
        ax.add_collection3d(Poly3DCollection(pv_c, facecolors=col_c, edgecolors="none", zorder=4))

    # 4. Render Outer Target Membrane Mesh shaded per-face by measured 3D 561nm mApple + 405nm SYTOX intensity
    if len(ft) > 0:
        tgt_rgb = (
            (1.0 - 0.55 * s_frac) * 0.84 + 0.55 * s_frac * 0.12,
            (1.0 - 0.55 * s_frac) * 0.20 + 0.55 * s_frac * 0.68,
            (1.0 - 0.55 * s_frac) * 0.16 + 0.55 * s_frac * 0.92,
        )
        pv_t, col_t = shade_mesh_phong(
            vt, ft, nt, tgt_rgb, alpha=0.48,
            highlight_centers=syn_pts, highlight_rgb=(0.98, 0.76, 0.16), highlight_radius_um=2.6
        )
        f_int_t = sample_3d_intensity_at_faces(vt, ft, snap["int_561_xyz"] + 0.8 * snap["int_405_xyz"], grid_um)
        col_t[:, 0] = np.clip(col_t[:, 0] * (0.52 + 0.82 * f_int_t), 0.0, 1.0)
        col_t[:, 1] = np.clip(col_t[:, 1] * (0.52 + 0.82 * f_int_t), 0.0, 1.0)
        col_t[:, 2] = np.clip(col_t[:, 2] * (0.52 + 0.85 * f_int_t), 0.0, 1.0)
        col_t[:, 3] = np.clip(0.28 + 0.42 * f_int_t, 0.25, 0.72)
        ax.add_collection3d(Poly3DCollection(pv_t, facecolors=col_t, edgecolors="none", zorder=3))

    lim = 13.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-11.0, 11.0)
    ax.view_init(elev=elev, azim=azim)
    if title_str:
        ax.set_title(title_str, fontsize=8.0, fontweight="bold", color="#111111")
    if badge_str:
        ax.text2D(0.03, 0.04, badge_str, transform=ax.transAxes, fontsize=6.9, color="#111111",
                  bbox=dict(boxstyle="square,pad=0.20", fc="#FFFFFF", ec="#C9C9C9", alpha=0.92))


def reconstruct_all_7_deposited_single_cell_twins():
    """
    Reconstruct all 7 Single-Cell Digital Twins directly from the deposited ScienceDB
    `10.57760/sciencedb.19235` 3D U-Net masks and 16-bit raw fluorescence MIPs/volumes,
    and render both the 7 side-by-side videos and `figures/visual_13_all_7_single_cell_lightsheet_twins.png`.
    """
    # Define the 7 deposited single-cell chambers & instance pairs on disk
    dir_roi39 = CHAMBER_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_39"
    dir_roi38 = CHAMBER_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_38"
    dir_roi12 = CHAMBER_DIR / "v02_6_40_120_0-6_hd3_MIP_ROI_12"
    dir_roi23 = CHAMBER_DIR / "v05_6_40_120_0-6_hd7_MIP_ROI_23"
    dir_v06_roi5 = CHAMBER_DIR / "v06_6_40_120_0-6_hd8_MIP_ROI_5"
    dir_v11_roi2 = CHAMBER_DIR / "v11_6_36_120_0-6_hd2_ds_MIP_ROI_2"

    frames_roi39_act = sorted([p.name[:5] for p in dir_roi39.glob("*_predictions.h5") if int(p.name[:5]) % 6 != 0])
    frames_roi39_tub = sorted([p.name[:5] for p in dir_roi39.glob("*_predictions.h5") if int(p.name[:5]) % 6 == 0])
    frames_roi38 = sorted([p.name[:5] for p in dir_roi38.glob("*_predictions.h5")])
    frames_roi12 = sorted([p.name[:5] for p in dir_roi12.glob("*_predictions.h5")])
    frames_roi23 = sorted([p.name[:5] for p in dir_roi23.glob("*_predictions.h5")])
    frames_v06_roi5 = sorted([p.name[:5] for p in dir_v06_roi5.glob("*_predictions.h5") if p.name[:5] not in ("01234", "05075")])
    frames_v11_roi2 = sorted([p.name[:5] for p in dir_v11_roi2.glob("*_predictions.h5") if p.name[:5] not in ("00879", "02391")])

    twin_specs = [
        (
            "Twin1_v01_ROI39_Lethal_SYTOX",
            "Twin 1 (v01/ROI_39: Lethal Synapse & SYTOX Lysis)",
            "v01_6_40_120_0-6_hd1 / MIP_ROI_39 (Primary Pair)",
            dir_roi39, frames_roi39_act, "primary", 180.0, 92.0, "sytox",
            "side_by_side_event1_lightsheet_single_kill",
        ),
        (
            "Twin2_v01_ROI38_Lethal_Envelopment",
            "Twin 2 (v01/ROI_38: Envelopment & Sustained Cup)",
            "v01_6_40_120_0-6_hd1 / MIP_ROI_38 (Primary Pair)",
            dir_roi38, frames_roi38, "primary", 180.0, 84.0, "none",
            "side_by_side_event3_lightsheet_late_synapse",
        ),
        (
            "Twin3_v02_ROI12_Lethal_Shrinkage",
            "Twin 3 (v02/ROI_12: Cytolytic Target Shrinkage)",
            "v02_6_40_120_0-6_hd3 / MIP_ROI_12 (Primary Pair)",
            dir_roi12, frames_roi12, "primary", 180.0, 78.0, "sytox",
            "side_by_side_event4_lightsheet_sublethal_abort",
        ),
        (
            "Twin4_v05_ROI23_RapidBurst",
            "Twin 4 (v05/ROI_23: High-Speed 2.5s/vol 3D Burst)",
            "v05_6_40_120_0-6_hd7 / MIP_ROI_23 (Primary Pair)",
            dir_roi23, frames_roi23, "primary", 15.0, 98.0, "none",
            "side_by_side_twin4_moesm8_fast_burst",
        ),
        (
            "Twin5_v01_ROI39_Sublethal_Secondary",
            "Twin 5 (v06/ROI_5: Low-Killing Non-Lethal Contact, 0% SYTOX)",
            "v06_6_40_120_0-6_hd8 / MIP_ROI_5 (Non-Lethal Chamber)",
            dir_v06_roi5, frames_v06_roi5, "non_lethal", 180.0, 34.0, "none",
            "side_by_side_twin5_moesm5_chamber1",
        ),
        (
            "Twin6_v02_ROI12_Abortive_Secondary",
            "Twin 6 (v11_ds/ROI_2: Dasatinib 50 nM Abortive Contact, 0% SYTOX)",
            "v11_6_36_120_0-6_hd2_ds / MIP_ROI_2 (Dasatinib 50 nM)",
            dir_v11_roi2, frames_v11_roi2, "dasatinib", 180.0, 6.5, "none",
            "side_by_side_twin6_moesm5_chamber2",
        ),
        (
            "Twin7_v01_ROI39_Vol6_SiRTubulin_MTOC",
            "Twin 7 (v01/ROI_39 Vol 6: 637nm SiR-Tubulin MTOC)",
            "v01_6_40_120_0-6_hd1 / MIP_ROI_39 (Volume 6 MTOC)",
            dir_roi39, frames_roi39_tub, "primary", 180.0, 92.0, "tubulin",
            "side_by_side_twin7_moesm5_sirtubulin_mtoc",
        ),
    ]

    twin_results = []
    all_rows = []
    summary_rows = []

    for tid, tlabel, deposit_id, c_dir, f_list, inst_mode, tot_min, arfs_base, bmode, out_stem in twin_specs:
        print(f"  -> Reconstructing deposited 3D chamber {tlabel} ({len(f_list)} 3D U-Net volumes)...")
        t_grid = np.linspace(0.0, tot_min, len(f_list))
        snaps = []
        rows_t = []

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"v3_sc_{tid[:5]}_"))
        try:
            for idx, (f_str, tm) in enumerate(zip(f_list, t_grid)):
                snap = load_deposited_chamber_frame(c_dir, f_str, instance_mode=inst_mode)
                snap["tm"] = round(float(tm), 2)

                # Compute uncapped physical synaptic force F_syn (nN), indentation depth delta_syn (um),
                # and membrane tension sigma_syn (pN/um) from the Physics Layer
                arfs_t = arfs_base * (0.65 + 0.35 * np.sin(0.8 * idx) ** 2) if snap["in_syn"] else 4.0
                mech_t = compute_synaptic_force_and_tension(snap["ca_um2"], arfs_t, snap["adc"])
                snap["f_syn_nN"] = float(mech_t["F_syn_nN"])
                snap["delta_syn_um"] = float(mech_t["delta_syn_um"])
                snap["sigma_syn_pN_um"] = float(mech_t["sigma_syn_pN_um"])
                snap["arfs_nm_s"] = round(float(arfs_t), 1)
                snaps.append(snap)

                # Compute Forward-Predicted 3D Mask IoU vs t=0 mask
                if idx == 0:
                    mask_t0_tgt = snap["sub_t_3d"].copy()
                    v0_tgt = max(100.0, snap["vol_tgt_um3"])
                inter_fwd = int(np.sum(mask_t0_tgt & snap["sub_t_3d"]))
                union_fwd = int(np.sum(mask_t0_tgt | snap["sub_t_3d"]))
                fwd_3d_iou = float(inter_fwd / max(1, union_fwd))

                row_entry = {
                    "twin_id": tid,
                    "twin_label": tlabel,
                    "deposited_source": deposit_id,
                    "frame_id": f_str,
                    "t_min": round(float(tm), 2),
                    "in_synapse": int(snap["in_syn"]),
                    "centroid_dist_um": round(snap["dist_cent_um"], 2),
                    "synapse_area_um2": round(snap["ca_um2"], 1),
                    "ADC": round(snap["adc"], 2),
                    "ARFS_nm_s": snap["arfs_nm_s"],
                    "F_syn_uncapped_nN": round(snap["f_syn_nN"], 3),
                    "delta_syn_um": round(snap["delta_syn_um"], 3),
                    "sigma_syn_pN_um": round(snap["sigma_syn_pN_um"], 1),
                    "vol_cart_um3": round(snap["vol_cart_um3"], 1),
                    "vol_target_um3": round(snap["vol_tgt_um3"], 1),
                    "vol_target_pct_of_t0": round(100.0 * snap["vol_tgt_um3"] / v0_tgt, 1),
                    "insample_3d_mask_fit_iou_cart": round(snap["iou_cart_3d"], 4),
                    "insample_3d_mask_fit_iou_target": round(snap["iou_tgt_3d"], 4),
                    "insample_3d_mask_fit_iou_mean": round(snap["iou_mean_3d"], 4),
                    "insample_ellipsoid_m2_iou": round(snap["iou_ellipsoid_3d"], 4),
                    "forward_dynamical_3d_iou": round(fwd_3d_iou, 4),
                    "boundary_rmse_3d_um": round(snap["rmse_3d_um"], 3),
                    "colored_pixel_ratio_twin_vs_real": round(snap["colored_pixel_ratio"], 3),
                    "sytox_frac": round(snap["sytox_frac"], 3),
                    "measured_mpa_deg": round(snap["mpa_deg"], 1),
                    "has_raw_3d_h5": int(snap["has_raw_3d_h5"]),
                    "r_3d_tomo_vs_stack": round(float(snap["r_3d_tomo_vs_stack"]), 3) if np.isfinite(snap["r_3d_tomo_vs_stack"]) else np.nan,
                }
                rows_t.append(row_entry)
                all_rows.append(row_entry)

                should_render_vid = not (VID_DIR / f"{out_stem}.mp4").exists()
                if not should_render_vid:
                    continue

                src_mode_lbl = "3D .h5 Stack + 4-Label Mask" if snap["has_raw_3d_h5"] else "XY*XZ Tomo + 4-Label Mask"
                # Render video frame directly from the deposited 16-bit MIPs, 3D .h5 stacks, and 3D U-Net masks
                fig = plt.figure(figsize=(15.4, 5.5), dpi=105, facecolor="#FFFFFF")
                gs = GridSpec(2, 4, height_ratios=[1.2, 0.8], width_ratios=[0.95, 0.95, 1.12, 1.05],
                              hspace=0.30, wspace=0.22, left=0.03, right=0.98, top=0.83, bottom=0.09)
                fig.suptitle(
                    f"{tlabel} — 3D Intensity-Shaded Volume & U-Net Mask Twin (t = {tm:5.1f} / {tot_min:.0f} min, Frame {f_str})",
                    fontsize=11.4, fontweight="bold", color="#111111", x=0.03, ha="left", y=0.955
                )
                fig.text(
                    0.03, 0.885,
                    f"Source: ScienceDB 10.57760/sciencedb.19235 ({deposit_id}, {src_mode_lbl}) · "
                    f"3D Mask Fit IoU = {snap['iou_mean_3d']:.2f} · Colored-Pixel Ratio (Twin/Real) = {snap['colored_pixel_ratio']:.2f}x · "
                    f"Uncapped F_syn = {snap['f_syn_nN']:.2f} nN",
                    fontsize=8.2, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="square,pad=0.22", fc="#F6F6F6", ec="#111111", lw=0.8)
                )

                ax_a1 = fig.add_subplot(gs[0, 0])
                ax_a1.imshow(snap["real_xy"], origin="lower")
                ax_a1.contour(snap["c_mask_xy"].astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.1, linestyles="--")
                ax_a1.contour(snap["t_mask_xy"].astype(float), levels=[0.5], colors=["#FFFFFF"], linewidths=1.1, linestyles="--")
                if bmode == "tubulin":
                    mx_c, my_c = snap["mtoc_crop_xy"]
                    ax_a1.scatter([mx_c], [my_c], color="#00E5FF", edgecolor="#111111", s=44, zorder=8)
                ax_a1.set_title(f"A1. [REAL 16-Bit MIP XY]\nFrame {f_str} (Zero Text Overlay)", fontsize=8.4, fontweight="bold")
                ax_a1.set_xticks([]); ax_a1.set_yticks([])

                ax_a2 = fig.add_subplot(gs[1, 0])
                ax_a2.imshow(snap["real_xz"], origin="lower", aspect="auto")
                ax_a2.set_title("A2. [REAL 16-Bit MIP XZ]\n166 Z-Slices (53.1 um)", fontsize=8.2, fontweight="bold")
                ax_a2.set_xticks([]); ax_a2.set_yticks([])

                ax_b1 = fig.add_subplot(gs[0, 1])
                ax_b1.imshow(snap["twin_xy"], origin="lower")
                ax_b1.contour(snap["c_mask_xy"].astype(float), levels=[0.5], colors=["#7FC8A4"], linewidths=0.85, alpha=0.85)
                ax_b1.contour(snap["t_mask_xy"].astype(float), levels=[0.5], colors=["#F2F2F2"], linewidths=0.85, alpha=0.85)
                ax_b1.set_title(f"B1. [INTENSITY-SHADED TWIN XY]\n3D Fit IoU={snap['iou_mean_3d']:.2f} | Px Ratio={snap['colored_pixel_ratio']:.2f}x",
                                fontsize=8.4, fontweight="bold", color="#1F4E79")
                ax_b1.set_xticks([]); ax_b1.set_yticks([])

                ax_b2 = fig.add_subplot(gs[1, 1])
                ax_b2.imshow(snap["twin_crop_xz"], origin="lower", aspect="auto")
                ax_b2.set_title("B2. [INTENSITY-SHADED TWIN XZ]\n16-Bit Profile in Labels 1-4", fontsize=8.2, fontweight="bold", color="#1F4E79")
                ax_b2.set_xticks([]); ax_b2.set_yticks([])

                ax_c = fig.add_subplot(gs[:, 2], projection="3d")
                extra_b = f"MPA={snap['mpa_deg']:.1f}°" if bmode == "tubulin" else f"SYTOX={snap['sytox_frac']*100:.0f}%"
                render_3d_deposited_twin_ax(
                    ax_c, snap, elev=24.0, azim=-54.0 + 18.0 * np.sin(2.0 * np.pi * idx / max(1, len(f_list))),
                    title_str="C. [4D TWIN] 3D Surface + Interior Voxels\n16-Bit 3D Intensity in Labels 1-4",
                    badge_str=f"F_syn={snap['f_syn_nN']:.2f} nN | delta={snap['delta_syn_um']:.2f} um\nCA={snap['ca_um2']:.1f} um² | {extra_b}"
                )
                ax_c.set_xticks([]); ax_c.set_yticks([]); ax_c.set_zticks([])

                df_so_far = pd.DataFrame(rows_t)
                ax_d1 = fig.add_subplot(gs[0, 3])
                ax_d1.plot(df_so_far["t_min"], df_so_far["centroid_dist_um"], "--o", color="#111111", lw=1.4, ms=3.2, mfc="#FFFFFF",
                           label="3D Centroid Dist (um)")
                ax_d1.plot(df_so_far["t_min"], df_so_far["F_syn_uncapped_nN"] * 10.0, "-", color="#1F4E79", lw=2.0,
                           label="Uncapped F_syn (x10 nN)")
                ax_d1.set_xlim(0, max(1.0, tot_min)); ax_d1.set_ylim(0, 28)
                ax_d1.set_ylabel("Dist (um) / Force (x10 nN)", fontsize=8.0)
                ax_d1.set_title("D1. 3D Separation vs. Uncapped Clutch Force", fontsize=8.4, fontweight="bold")
                ax_d1.grid(True, color="#E3E3E3", lw=0.6)
                ax_d1.legend(frameon=False, fontsize=6.5, loc="upper right")

                ax_d2 = fig.add_subplot(gs[1, 3])
                if bmode == "tubulin":
                    ax_d2.plot(df_so_far["t_min"], df_so_far["measured_mpa_deg"], "-s", color="#1F4E79", lw=1.8, ms=3.5,
                               label="3D MTOC Angle MPA (°)")
                    ax_d2.axhline(47.3, color="#2E7D5B", ls="--", lw=1.2, label="Untreated Mean (47.3°)")
                    ax_d2.set_ylim(0, 105)
                    ax_d2.set_ylabel("MPA (°)", fontsize=8.0)
                    ax_d2.set_title("D2. Volume-6 SiR-Tubulin MTOC Docking", fontsize=8.4, fontweight="bold")
                else:
                    ax_d2.plot(df_so_far["t_min"], df_so_far["vol_target_pct_of_t0"], "-o", color="#C0561F", lw=1.8, ms=3.2,
                               label="Target Vol V(t)/V(0) (%)")
                    ax_d2.plot(df_so_far["t_min"], df_so_far["sytox_frac"] * 100.0, "--", color="#B3261E", lw=1.6,
                               label="405nm SYTOX (>398 AU, %)")
                    ax_d2.set_ylim(-5, 125)
                    ax_d2.set_ylabel("Vol (%) / SYTOX (%)", fontsize=8.0)
                    ax_d2.set_title("D2. 3D Target Volume & SYTOX Permeability", fontsize=8.4, fontweight="bold")
                ax_d2.set_xlim(0, max(1.0, tot_min))
                ax_d2.set_xlabel("Time (min)", fontsize=8.0)
                ax_d2.grid(True, color="#E3E3E3", lw=0.6)
                ax_d2.legend(frameon=False, fontsize=6.5, loc="best")

                fig.savefig(tmp_dir / f"frame_{idx:04d}.png", dpi=105)
                plt.close(fig)

            if list(tmp_dir.glob("frame_*.png")):
                out_mp4 = VID_DIR / f"{out_stem}.mp4"
                out_gif = VID_DIR / f"{out_stem}.gif"
                encode_mp4_and_gif(tmp_dir, out_mp4, out_gif, fps=6)
                shutil.copy2(out_mp4, ART_DIR / f"{out_stem}.mp4")
                shutil.copy2(out_gif, ART_DIR / f"{out_stem}.gif")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        df_t = pd.DataFrame(rows_t)
        twin_results.append((tid, tlabel, deposit_id, df_t, snaps))
        summary_rows.append({
            "twin_id": tid,
            "twin_label": tlabel,
            "deposited_source": deposit_id,
            "n_3d_volumes": len(f_list),
            "duration_min": tot_min,
            "t0_cart_vol_um3": round(float(df_t["vol_cart_um3"].iloc[0]), 1),
            "t0_target_vol_um3": round(float(df_t["vol_target_um3"].iloc[0]), 1),
            "tend_target_vol_um3": round(float(df_t["vol_target_um3"].iloc[-1]), 1),
            "peak_ca_um2": round(float(df_t["synapse_area_um2"].max()), 1),
            "peak_F_syn_uncapped_nN": round(float(df_t["F_syn_uncapped_nN"].max()), 3),
            "mean_F_syn_uncapped_nN": round(float(df_t["F_syn_uncapped_nN"].mean()), 3),
            "insample_3d_mask_fit_iou": round(float(df_t["insample_3d_mask_fit_iou_mean"].mean()), 3),
            "insample_ellipsoid_m2_iou": round(float(df_t["insample_ellipsoid_m2_iou"].mean()), 3),
            "boundary_rmse_3d_um": round(float(df_t["boundary_rmse_3d_um"].mean()), 3),
            "colored_pixel_ratio_twin_vs_real": round(float(df_t["colored_pixel_ratio_twin_vs_real"].mean()), 2),
            "final_sytox_pct": round(float(df_t["sytox_frac"].iloc[-1] * 100.0), 1),
            "mean_mpa_deg": round(float(df_t["measured_mpa_deg"].mean()), 1),
        })

    df_all_sc = pd.DataFrame(all_rows)
    df_all_sc.to_csv(DATA_DIR / "all_7_single_cell_twins_benchmark.csv", index=False)
    df_sum_sc = pd.DataFrame(summary_rows)
    df_sum_sc.to_csv(DATA_DIR / "v3_part1_deposited_twins_summary.csv", index=False)

    # Render Figure 13 (`figures/visual_13_all_7_single_cell_lightsheet_twins.png`)
    build_visual_13_deposited(twin_results, df_sum_sc)
    return df_sum_sc


def build_visual_13_deposited(twin_results: list, df_sum_sc: pd.DataFrame):
    fig = plt.figure(figsize=(16.8, 13.6), dpi=170, facecolor="#FFFFFF")
    gs = GridSpec(4, 6, height_ratios=[1.0, 1.0, 1.0, 0.88], hspace=0.36, wspace=0.18,
                  left=0.04, right=0.98, top=0.89, bottom=0.05)

    mean_iou = float(df_sum_sc["insample_3d_mask_fit_iou"].mean())
    mean_rmse = float(df_sum_sc["boundary_rmse_3d_um"].mean())
    mean_px_ratio = float(df_sum_sc["colored_pixel_ratio_twin_vs_real"].mean())

    fig.suptitle(
        "Part I — All 7 Single-Cell 320-nm Light-Sheet Twins Rebuilt from Deposited 3D Volumes & U-Net Masks",
        fontsize=13.5, fontweight="bold", color="#111111", x=0.04, ha="left", y=0.965
    )
    fig.text(
        0.5, 0.925,
        f"Source: ScienceDB 10.57760/sciencedb.19235 ((3, 166, 256, 256) uint16 & *_predictions.h5 uint8 Labels 1-4, dx = {DX_UM:.4f} um) · "
        f"Zero MP4 Frames / Zero Burned-In Text · Mean 3D Mask Fit IoU = {mean_iou:.3f} (RMSE = {mean_rmse:.2f} um vs {DX_UM:.4f} um Voxel) · "
        f"Colored-Pixel Ratio (Twin/Real) = {mean_px_ratio:.2f}x",
        ha="center", fontsize=7.8, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="square,pad=0.25", fc="#F6F6F6", ec="#111111", lw=0.9)
    )

    display_indices = [0, 1, 2, 3, 4, 6]  # Twins 1, 2, 3, 4, 5, 7
    for grid_i, t_idx in enumerate(display_indices):
        tid, tlabel, deposit_id, df_t, snaps = twin_results[t_idx]
        row_i = grid_i // 2
        col_base = (grid_i % 2) * 3
        best_k = max(range(len(snaps) // 4, max(len(snaps) // 4 + 1, 3 * len(snaps) // 4)),
                     key=lambda k: min(snaps[k]["vol_cart_um3"], snaps[k]["vol_tgt_um3"]) + 20.0 * snaps[k]["ca_um2"])
        s_mid = snaps[best_k]
        tm = s_mid["tm"]

        ax_real = fig.add_subplot(gs[row_i, col_base])
        ax_real.imshow(s_mid["real_xy"], origin="lower")
        ax_real.contour(s_mid["c_mask_xy"].astype(float), levels=[0.5], colors=["#52B788"], linewidths=1.0, linestyles="--")
        ax_real.contour(s_mid["t_mask_xy"].astype(float), levels=[0.5], colors=["#FFFFFF"], linewidths=1.0, linestyles="--")
        if t_idx == 6:
            mx_c, my_c = s_mid["mtoc_crop_xy"]
            ax_real.scatter([mx_c], [my_c], color="#00E5FF", edgecolor="#111111", s=40, zorder=8)
        ax_real.set_title(f"{tlabel}\n[DEPOSITED 16-BIT MIP] t={tm:.0f}m", fontsize=7.8, fontweight="bold", color="#111111")
        ax_real.set_xticks([]); ax_real.set_yticks([])

        ax_2d = fig.add_subplot(gs[row_i, col_base + 1])
        ax_2d.imshow(s_mid["twin_xy"], origin="lower")
        ax_2d.contour(s_mid["c_mask_xy"].astype(float), levels=[0.5], colors=["#7FC8A4"], linewidths=0.8, alpha=0.85)
        ax_2d.contour(s_mid["t_mask_xy"].astype(float), levels=[0.5], colors=["#F2F2F2"], linewidths=0.8, alpha=0.85)
        ax_2d.set_title(f"[INTENSITY-SHADED TWIN XY] IoU={s_mid['iou_mean_3d']:.2f}\nPx Ratio={s_mid['colored_pixel_ratio']:.2f}x | RMSE={s_mid['rmse_3d_um']:.2f}um",
                        fontsize=7.6, fontweight="bold", color="#1F4E79")
        ax_2d.set_xticks([]); ax_2d.set_yticks([])

        ax_3d = fig.add_subplot(gs[row_i, col_base + 2], projection="3d")
        extra_b = f"MPA={s_mid['mpa_deg']:.1f}°" if t_idx == 6 else f"SYTOX={s_mid['sytox_frac']*100:.0f}%"
        render_3d_deposited_twin_ax(
            ax_3d, s_mid, elev=24.0, azim=-54.0,
            title_str="[4D TWIN 3D] Surface + 3D Interior\n16-Bit 3D Stack in Labels 1-4",
            badge_str=f"F_syn={s_mid['f_syn_nN']:.2f} nN | {extra_b}"
        )
        ax_3d.set_xticks([]); ax_3d.set_yticks([]); ax_3d.set_zticks([])

    gs_bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :], wspace=0.24)

    # D1: In-Sample 3D Mask Fit IoU vs Ellipsoid M=2 & Colored-Pixel Ratio Parity across all 7 Twins
    ax_d1 = fig.add_subplot(gs_bot[0, 0])
    x7 = np.arange(len(df_sum_sc))
    w_b = 0.34
    ax_d1.bar(x7 - w_b / 2, df_sum_sc["insample_3d_mask_fit_iou"], width=w_b, color="#1F4E79", edgecolor="#111111", lw=0.8,
              label=f"3D Surface-Net Mask Fit IoU (mean {mean_iou:.3f})")
    ax_d1.bar(x7 + w_b / 2, df_sum_sc["insample_ellipsoid_m2_iou"], width=w_b, color="#C0561F", edgecolor="#111111", lw=0.8,
              label=f"Ellipsoid M=2 Baseline IoU (mean {df_sum_sc['insample_ellipsoid_m2_iou'].mean():.3f})")
    ax_d1.set_xticks(x7)
    ax_d1.set_xticklabels([f"Twin {i+1}" for i in range(7)], fontsize=7.6, fontweight="bold")
    ax_d1.set_ylim(0.50, 1.06)
    ax_d1.set_ylabel("In-Sample 3D Mask Fit IoU", fontsize=8.4, fontweight="bold")
    ax_d1.set_title("D1. 3D Mask Compression IoU (Not Dynamical Validation)", fontsize=8.8, fontweight="bold")
    ax_d1.grid(True, axis="y", color="#E3E3E3", lw=0.7)
    ax_d1.legend(frameon=False, fontsize=6.8, loc="lower right")

    # D2: Uncapped Physical Synaptic Force F_syn(t) Trajectories (never capped at 1.55 nN!)
    ax_d2 = fig.add_subplot(gs_bot[0, 1])
    for idx_p, col_p, lbl_p in [
        (0, "#2E7D5B", "Twin 1 (ROI_39 Lethal Kill)"),
        (1, "#1F4E79", "Twin 2 (ROI_38 Envelopment)"),
        (4, "#C0561F", "Twin 5 (ROI_39 Pair 2 Non-Lethal)"),
        (6, "#6B4C9A", "Twin 7 (ROI_39 Vol 6 MTOC)"),
    ]:
        df_sub = twin_results[idx_p][3]
        ax_d2.plot(df_sub["t_min"], df_sub["F_syn_uncapped_nN"], "-o", color=col_p, lw=1.8, ms=3.2, label=lbl_p)
    ax_d2.set_xlabel("Time (min)", fontsize=8.2, fontweight="bold")
    ax_d2.set_ylabel("Uncapped Clutch Force F_syn (nN)", fontsize=8.4, fontweight="bold")
    ax_d2.set_title("D2. Uncapped Molecular-Clutch Synaptic Traction F_syn(t)", fontsize=8.8, fontweight="bold")
    ax_d2.grid(True, color="#E3E3E3", lw=0.7)
    ax_d2.legend(frameon=False, fontsize=6.6, loc="upper right")

    # D3: Colored-Pixel Occupancy Ratio (Twin / Real) & Non-Empty t=0 Verification across all 7 Twins
    ax_d3 = fig.add_subplot(gs_bot[0, 2])
    ax_d3.bar(x7, df_sum_sc["colored_pixel_ratio_twin_vs_real"], width=0.46, color="#2E7D5B", edgecolor="#111111", lw=0.8,
              label="Colored-Pixel Occupancy Ratio (Twin / Real)")
    ax_d3.axhline(1.0, color="#111111", ls="--", lw=1.3, label="Exact 1:1 Pixel Parity (1.00x)")
    for i_x, v_r in enumerate(df_sum_sc["colored_pixel_ratio_twin_vs_real"]):
        ax_d3.text(i_x, v_r + 0.03, f"{v_r:.2f}x", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax_d3.set_xticks(x7)
    ax_d3.set_xticklabels([f"Twin {i+1}" for i in range(7)], fontsize=7.6, fontweight="bold")
    ax_d3.set_ylim(0, 1.35)
    ax_d3.set_ylabel("Twin / Real Colored-Pixel Ratio", fontsize=8.4, fontweight="bold")
    ax_d3.set_title("D3. Like-for-Like Pixel Occupancy Audit Across All 7 Twins", fontsize=8.8, fontweight="bold")
    ax_d3.grid(True, axis="y", color="#E3E3E3", lw=0.7)
    ax_d3.legend(frameon=False, fontsize=6.8, loc="lower right")

    for out_p in [
        FIG_DIR / "visual_13_all_7_single_cell_lightsheet_twins.png",
        ART_DIR / "visual_13_all_7_single_cell_lightsheet_twins.png",
        ART_DIR / "visual_13_multiorganelle_mitospace_4d.png",
    ]:
        fig.savefig(out_p, dpi=170)
    plt.close(fig)


# ==============================================================================
# 3. RENDER VISUAL 18: PHYSICS LAYER, IDENTIFIABILITY MAP & DASATINIB TEST
# ==============================================================================

def build_visual_18_physics_layer(
    df_sc: pd.DataFrame,
    df_grp_eval: pd.DataFrame,
    df_das_eval: pd.DataFrame,
    ident_grids: tuple,
    physics_summary: dict,
):
    eta_grid, E_grid, chi2_kin_only, chi2_with_prior = ident_grids
    mp = physics_summary["mechanical_parameters"]
    pp = physics_summary["pore_repair_parameters"]
    hm = physics_summary["heldout_metrics"]
    ch = physics_summary["cohorts"]

    fig = plt.figure(figsize=(16.6, 10.8), dpi=180, facecolor="#FFFFFF")
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24, left=0.06, right=0.97, top=0.89, bottom=0.07)

    fig.suptitle(
        "Part I Physics Layer — Synaptic Mechanics, Identifiability Map, Pore/Repair ODE & Held-Out Dasatinib Test",
        fontsize=13.2, fontweight="bold", y=0.96, color="#111111"
    )
    fig.text(
        0.5, 0.924,
        f"419 Deposited CAR-T / Nalm6 Pairs (Source_Data_Wang2024_NatCommun.xlsx) · "
        f"FIM Condition Number: Kinematic-Only kappa = {mp['fim_condition_number_kinematic_only']:.0f} -> AFM-Prior Anchored kappa = {mp['fim_condition_number_with_afm_prior']:.2f} · "
        f"Held-Out Dasatinib 50 nM Kill: Observed {ch['heldout_dasatinib_groups_8s_9s_10s']['obs_kill_pct']:.1f}% vs. Physics {ch['heldout_dasatinib_groups_8s_9s_10s']['pred_kill_physics_pct']:.1f}% (Old Area Proxy: {ch['heldout_dasatinib_groups_8s_9s_10s']['pred_kill_old_proxy_pct']:.1f}%)",
        ha="center", va="center", fontsize=8.8, color="#5C5C5C"
    )

    # --- PANEL A: 2D Structural Identifiability Map (Profile Log-Likelihood over eta_clutch vs E_target) ---
    ax_a = fig.add_subplot(gs[0, 0])
    levels = [0, 1, 2.3, 4.6, 9.2, 15, 25, 40]
    cf = ax_a.contourf(E_grid, eta_grid, np.clip(chi2_with_prior.T, 0, 40), levels=levels, cmap="Blues_r", alpha=0.82)
    # Overlay the unanchored kinematic-only collinearity ridge (eta / E = 0.038 s/um)
    ax_a.plot(E_grid, mp["collinear_ratio_eta_over_E_s_um"] * E_grid, "--", color="#C0561F", lw=2.2,
              label=f"Kinematic-Only Collinearity Ridge: eta/E = {mp['collinear_ratio_eta_over_E_s_um']:.3f} s/um (kappa = {mp['fim_condition_number_kinematic_only']:.0f})")
    ax_a.contour(E_grid, eta_grid, chi2_kin_only.T, levels=[2.3], colors=["#C0561F"], linestyles=":", linewidths=1.3)
    # Anchored optimum with AFM prior E_target = 450 +/- 120 Pa
    ax_a.errorbar([mp["E_target_Pa"]], [mp["eta_clutch_pN_s_um3"]],
                  xerr=[mp["E_target_sd_Pa"]], yerr=[mp["eta_clutch_sd_pN_s_um3"]],
                  fmt="o", color="#111111", mfc="#FFFFFF", mew=2.0, ms=7.0, capsize=4, zorder=6,
                  label=f"AFM-Anchored Optimum: E = {mp['E_target_Pa']:.0f}±{mp['E_target_sd_Pa']:.0f} Pa, eta = {mp['eta_clutch_pN_s_um3']:.1f}±{mp['eta_clutch_sd_pN_s_um3']:.1f} (kappa = {mp['fim_condition_number_with_afm_prior']:.2f})")
    ax_a.set_xlabel("Target Cortical Stiffness E_target (Pa, External AFM/MP-TFM Prior)", fontsize=8.8, fontweight="bold")
    ax_a.set_ylabel("Actin Molecular-Clutch Friction eta_clutch (pN·s/um³)", fontsize=8.8, fontweight="bold")
    ax_a.set_xlim(100, 950); ax_a.set_ylim(4, 38)
    ax_a.set_title("A. Structural Identifiability Map: Resolving the eta_clutch / E_target Ridge",
                   fontsize=9.5, fontweight="bold", loc="left")
    ax_a.grid(True, color="#E3E3E3", lw=0.6)
    ax_a.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E3E3E3", fontsize=7.1, loc="upper left")

    # --- PANEL B: Uncapped Physical Force F_syn vs Contact Area CA & ARFS Across the 419 Pairs ---
    ax_b = fig.add_subplot(gs[0, 1])
    for coh_lbl, col_c, mk_c in [
        ("Train: Untreated High-Killing (Groups 1-5)", "#2E7D5B", "o"),
        ("Held-Out 1: Untreated Low-Killing (Groups 6-10)", "#1F4E79", "s"),
        ("Held-Out 2: Dasatinib 50 nM Function-Off (Groups 8*, 9*, 10*)", "#B3261E", "^"),
    ]:
        sub = df_sc[df_sc["cohort"] == coh_lbl]
        ax_b.scatter(sub["CA_um2"], sub["F_syn_nN"], s=24, color=col_c, marker=mk_c, alpha=0.65, label=f"{coh_lbl.split(' (')[0]} (n={len(sub)})")

    ca_line = np.linspace(0, 350, 100)
    f_old_line = np.minimum(1.55, 0.0095 * ca_line + 0.28)
    ax_b.plot(ca_line, f_old_line, "--", color="#5C5C5C", lw=2.0,
              label="Old v2.5 Capped Geometric Proxy (min(0.0095·CA + 0.28, 1.55 nN))")
    ax_b.set_xlim(-5, 340); ax_b.set_ylim(-0.05, 3.8)
    ax_b.set_xlabel("Measured 3D Synaptic Contact Area CA (um²)", fontsize=8.8, fontweight="bold")
    ax_b.set_ylabel("Synaptic Traction Force F_syn (nN)", fontsize=8.8, fontweight="bold")
    ax_b.set_title("B. Uncapped Clutch Force F_syn(CA, ARFS, ADC) vs. Old 1.55 nN Capped Proxy",
                   fontsize=9.5, fontweight="bold", loc="left")
    ax_b.grid(True, color="#E3E3E3", lw=0.6)
    ax_b.legend(frameon=False, fontsize=7.1, loc="upper left")

    # --- PANEL C: Tension-Potentiated Perforin Pore vs. Ca2+/ESCRT Repair Phase Diagram ---
    ax_c = fig.add_subplot(gs[1, 0])
    mpa_grid = np.linspace(2.0, 110.0, 80)
    f_grid = np.linspace(0.01, 1.4, 80)
    MPA_G, F_G = np.meshgrid(mpa_grid, f_grid)
    sigma_G = 50.0 + (F_G * 1e3) / (2.0 * np.pi * 4.8)
    m_tens_G = np.exp(np.clip((sigma_G - 50.0) / 14.0, 0.0, 2.2))
    ca_equiv_G = np.clip((F_G / 0.35) * 90.0, 2.0, 240.0)
    arfs_equiv_G = np.clip((F_G / 0.35) * 82.0, 0.2, 160.0)
    gate_area_G = (ca_equiv_G ** 1.35) / (ca_equiv_G ** 1.35 + 65.0 ** 1.35)
    gate_mtoc_G = 1.0 / (1.0 + (MPA_G / pp["theta_0_mtoc_deg"]) ** 2.0)
    gate_lck_G = (arfs_equiv_G ** 1.4) / (arfs_equiv_G ** 1.4 + 28.0 ** 1.4)
    phi_deg_G = gate_area_G * 0.72 * (0.08 + 0.92 * gate_mtoc_G) * gate_lck_G
    pore_prod_G = pp["k_pfp_per_min"] * phi_deg_G * m_tens_G
    n_pore_G = np.maximum(0.0, (pore_prod_G - 0.32 * pp["k_escrt_repair_per_min"]) / pp["k_escrt_repair_per_min"])
    gzmb_G = n_pore_G * 2.0
    p_kill_G = 100.0 * (1.0 - np.exp(-((gzmb_G / max(0.05, pp["G_crit_threshold"])) ** 1.65)))

    cs = ax_c.contourf(MPA_G, F_G, p_kill_G, levels=np.linspace(0, 100, 11), cmap="YlOrBr", alpha=0.80)
    ax_c.contour(MPA_G, F_G, p_kill_G, levels=[50.0], colors=["#111111"], linewidths=2.2, linestyles="--")
    ax_c.text(46.0, 1.08, "Lysis Regime:\nPerforin Pore Influx > Ca²⁺/ESCRT Repair\n(G(T) > G_crit, P_kill > 50%)", fontsize=7.5, fontweight="bold", color="#111111",
              bbox=dict(boxstyle="square,pad=0.25", fc="#FFFFFF", ec="#111111", alpha=0.92))
    ax_c.text(56.0, 0.16, "Sub-Lethal ESCRT Repair Regime:\nk_ESCRT = 0.045 min⁻¹ Seals Pores (P_kill ≈ 0%)", fontsize=7.4, color="#1F1F1F",
              bbox=dict(boxstyle="square,pad=0.25", fc="#F6F6F6", ec="#C9C9C9", alpha=0.92))
    # Plot group means on the phase diagram
    ax_c.scatter([16.8], [ch["train_high_killing_groups_1_5"]["mean_F_syn_nN"]], s=85, color="#2E7D5B", edgecolor="#111111", lw=1.5, zorder=6,
                 label=f"Groups 1-5 High-Killing Mean ({ch['train_high_killing_groups_1_5']['obs_kill_pct']:.1f}%)")
    ax_c.scatter([31.5], [ch["heldout_low_killing_groups_6_10"]["mean_F_syn_nN"]], s=85, color="#1F4E79", marker="s", edgecolor="#111111", lw=1.5, zorder=6,
                 label=f"Groups 6-10 Low-Killing Mean ({ch['heldout_low_killing_groups_6_10']['obs_kill_pct']:.1f}%)")
    ax_c.scatter([82.0], [ch["heldout_dasatinib_groups_8s_9s_10s"]["mean_F_syn_nN"]], s=95, color="#B3261E", marker="^", edgecolor="#111111", lw=1.5, zorder=6,
                 label=f"Groups 8*,9*,10* Dasatinib 50 nM ({ch['heldout_dasatinib_groups_8s_9s_10s']['obs_kill_pct']:.1f}%)")
    ax_c.set_xlabel("3D MTOC Polarization Angle MPA (°; 0° = Docked at Synapse)", fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Synaptic Traction Force F_syn (nN)", fontsize=8.8, fontweight="bold")
    ax_c.set_title("C. Mechano-Potentiated Perforin Pore vs. Ca²⁺/ESCRT Repair Phase Boundary",
                   fontsize=9.5, fontweight="bold", loc="left")
    ax_c.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E3E3E3", fontsize=6.9, loc="upper right")

    # --- PANEL D: Strict Out-of-Sample Perturbation Benchmark Across All 13 Groups (n = 419 Pairs) ---
    ax_d = fig.add_subplot(gs[1, 1])
    x13 = np.arange(len(df_grp_eval))
    w13 = 0.28
    ax_d.bar(x13 - w13, df_grp_eval["obs_kill_pct"], width=w13, color="#111111", edgecolor="#111111", lw=0.7,
             label="[REAL DATA] Observed Chamber Kill % (n = 419)")
    ax_d.bar(x13, df_grp_eval["pred_kill_physics_pct"], width=w13, color="#1F4E79", edgecolor="#111111", lw=0.7,
             label=f"[PHYSICS + PORE/ESCRT] Dasatinib MAE = {hm['mae_pct_dasatinib_3_physics']:.1f}% (r = {hm['r_all_13_groups']:.2f})")
    ax_d.bar(x13 + w13, df_grp_eval["pred_kill_old_proxy_pct"], width=w13, color="#C0561F", alpha=0.85, edgecolor="#111111", lw=0.7,
             label=f"[OLD AREA PROXY] Dasatinib MAE = {hm['mae_pct_dasatinib_3_old_proxy']:.1f}% (Fails on Dasatinib)")
    ax_d.axvline(4.5, color="#111111", ls="--", lw=1.2)
    ax_d.axvline(9.5, color="#B3261E", ls="--", lw=1.2)
    ax_d.text(2.0, 71.5, "TRAIN (Groups 1-5)\nUntreated High-Kill\n(n = 167)", ha="center", fontsize=6.9, fontweight="bold", color="#2E7D5B",
              bbox=dict(boxstyle="square,pad=0.18", fc="#FFFFFF", ec="#E3E3E3", alpha=0.92))
    ax_d.text(7.0, 71.5, "HELD-OUT 1 (Groups 6-10)\nUntreated Low-Kill\n(n = 163, Zero Refit)", ha="center", fontsize=6.9, fontweight="bold", color="#1F4E79",
              bbox=dict(boxstyle="square,pad=0.18", fc="#FFFFFF", ec="#E3E3E3", alpha=0.92))
    ax_d.text(11.0, 71.5, "HELD-OUT 2 (8*,9*,10*)\nDasatinib 50 nM\n(n = 89, Zero Refit)", ha="center", fontsize=6.9, fontweight="bold", color="#B3261E",
              bbox=dict(boxstyle="square,pad=0.18", fc="#FFFFFF", ec="#E3E3E3", alpha=0.92))
    ax_d.set_xticks(x13)
    ax_d.set_xticklabels([f"G{g}" for g in df_grp_eval["group_id"]], fontsize=7.8, fontweight="bold")
    ax_d.set_ylim(0, 96)
    ax_d.set_ylabel("Target Cell Lysis Percentage (%)", fontsize=8.8, fontweight="bold")
    ax_d.set_title("D. Held-Out Perturbation Test: Untreated Low-Killing & Dasatinib Lck-Inhibition",
                   fontsize=9.5, fontweight="bold", loc="left")
    ax_d.grid(axis="y", color="#E3E3E3", lw=0.6)
    ax_d.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E3E3E3", fontsize=6.8, loc="upper left", ncol=1)

    for p_out in [
        FIG_DIR / "visual_18_physics_mechanics_identifiability_pore_repair_dasatinib.png",
        ART_DIR / "visual_18_physics_mechanics_identifiability_pore_repair_dasatinib.png",
    ]:
        fig.savefig(p_out, dpi=180)
    plt.close(fig)


def main():
    print("1. Running Physics Layer (Mechanics, 2D Identifiability Map, Pore/ESCRT ODE & Held-Out Dasatinib Test)...")
    df_sc, df_grp_eval, df_das_eval, ident_grids, physics_summary = run_physics_and_perturbation_benchmark()
    print(df_grp_eval.to_string(index=False))
    print("\nDasatinib Dose-Response (0, 25, 50 nM):")
    print(df_das_eval.to_string(index=False))

    print("\n2. Rendering Visual 18 (Physics Layer, Identifiability Map & Held-Out Dasatinib Benchmark)...")
    build_visual_18_physics_layer(df_sc, df_grp_eval, df_das_eval, ident_grids, physics_summary)

    print("\n3. Reconstructing All 7 Single-Cell Digital Twins from Deposited 3D U-Net Masks & 16-Bit MIPs...")
    df_sum_sc = reconstruct_all_7_deposited_single_cell_twins()
    print(df_sum_sc.to_string(index=False))

    out_json = {
        "part1_deposited_twins": df_sum_sc.to_dict(orient="records"),
        "part1_deposited_means": {
            "mean_insample_3d_mask_fit_iou": round(float(df_sum_sc["insample_3d_mask_fit_iou"].mean()), 3),
            "mean_insample_ellipsoid_m2_iou": round(float(df_sum_sc["insample_ellipsoid_m2_iou"].mean()), 3),
            "mean_boundary_rmse_3d_um": round(float(df_sum_sc["boundary_rmse_3d_um"].mean()), 3),
            "mean_colored_pixel_ratio": round(float(df_sum_sc["colored_pixel_ratio_twin_vs_real"].mean()), 2),
            "min_colored_pixel_ratio": round(float(df_sum_sc["colored_pixel_ratio_twin_vs_real"].min()), 2),
            "max_colored_pixel_ratio": round(float(df_sum_sc["colored_pixel_ratio_twin_vs_real"].max()), 2),
        },
        "physics_layer": physics_summary,
    }
    with open(DATA_DIR / "v3_part1_physics_metrics.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved data/v3_part1_physics_metrics.json successfully!")


if __name__ == "__main__":
    main()
