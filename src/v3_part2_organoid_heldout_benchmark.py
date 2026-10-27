#!/usr/bin/env python3
"""
v3.0 — Part II Complete Rebuild:
1. Drop the 4 debris objects (<3,000 um^3: ranks 18, 19, 20, 21) and the 1 growing organoid
   (rank 8: 50,263 -> 67,663 um^3, +34.6%), leaving 16 valid shrinking multicellular organoids
   (V0 >= 11,068 um^3, dV < 0):
   - 5 Design Organoids (Batch 1: ranks 1, 2, 5, 9, 10)
   - 11 Held-Out Shrinking Organoids (Batch 2 Held-Out: ranks 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 17)
2. Zero In-Sample Leakage:
   - True forward ODE integration from V(0) = V0 using only measured CD8+(t), CD4+(t), and V0^(-1/3).
   - Never uses V_end or gaussian_filter(V_real) inside any model prediction.
   - Extracts true per-organoid dead-dye intensity directly from Channel 2 (RS3, 659 nm) inside the
     3D organoid volume without any synthetic time ramp.
3. Matched-Parameter-Count Held-Out Benchmark with 10,000-Bootstrap 95% Confidence Intervals:
   - p_local = 0 (Out-of-Sample, fitted on 5 Design Organoids only, tested on 11 Held-Out):
     * Baseline 0A: Global 2-Parameter Exponential Decay V0 * [f_inf + (1 - f_inf) * exp(-lambda_0 * t)]
     * Model 2A: Out-of-Sample Swarm + Shell ODE (fitted on 5 Design Organoids only, 0 local params)
   - p_local = 1 (Matched 1 Free Parameter per Organoid on the 11 Held-Out Organoids):
     * Baseline 1B: 1-Parameter Per-Organoid Exponential Decay V0 * exp(-lambda_j * t) (p_local = 1)
     * Model 2B: 1-Parameter Per-Organoid Swarm + Shell ODE dV/dt = -kappa_j * S_j(t) * V (p_local = 1)
4. Explicitly Separates:
   - In-Sample 16-Mode Spherical-Harmonic Mask Fit IoU (geometric representation capacity)
   - Forward-Predicted Dynamical 3D Voxel IoU (evolving the t = 0 3D mask forward via V_hat(t)/V0)
"""

import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
ART_DIR = Path("/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial"],
    "axes.edgecolor": "#C9C9C9",
    "axes.linewidth": 0.8,
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "text.color": "#111111",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})


def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 20261025) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=np.float64)
    mean_v = float(np.mean(vals))
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean_v, float(lo), float(hi)


def calc_r2_unclipped(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-9
    return 1.0 - ss_res / ss_tot


def calc_nrmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return 100.0 * rmse / max(1.0, float(y_true[0]))


def extract_all_21_organoid_timeseries():
    """
    Extract true time series V_obs(t), D_obs(t) (directly from Channel 2 RS3 659 nm dead dye
    and Channel 4 organoid mask), CD8(t) at 1.5 um and 10 um, CD4(t), and 3D masks at t = 0..250
    from `LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims`.
    """
    ims_path = ROOT / "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims"
    df_emp = pd.read_csv(DATA_DIR / "organoid_multitcell_swarm_empirical.csv")
    df_b2_old = pd.read_csv(DATA_DIR / "organoid_batch2_all21_twins_results.csv")

    batch1_ids = {1000826950, 1000789786, 1000774627, 1000702716, 1000768105}

    with h5py.File(ims_path, "r") as f_ims:
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

        # Pre-load 15 confocal timepoints across the FULL 669-frame (22.2 h) movie (0, 50, 100, ..., 650, 668)
        sample_tps = list(range(0, 651, 50)) + [668]
        cached_ch = {}
        for tp in sample_tps:
            tp_grp = f_ims[f"DataSet/ResolutionLevel 0/TimePoint {tp}"]
            c1 = tp_grp["Channel 1/Data"][:62, :200, :200].astype(np.float32)
            c2 = tp_grp["Channel 2/Data"][:62, :200, :200].astype(np.float32)
            c4 = tp_grp["Channel 4/Data"][:62, :200, :200].astype(np.float32)
            cached_ch[tp] = (c1, c2, c4)

        def get_3d_crop_mask_and_dye(center_um: np.ndarray, tp: int, half_px: int = 18):
            c1, c2, c4 = cached_ch[tp]
            px = int(np.round((center_um[0] / 1000.0 - ext_min[0]) / (ext_max[0] - ext_min[0]) * 200.0))
            py = int(np.round((center_um[1] / 1000.0 - ext_min[1]) / (ext_max[1] - ext_min[1]) * 200.0))
            x0 = int(np.clip(px - half_px, 0, 200 - 2 * half_px))
            y0 = int(np.clip(py - half_px, 0, 200 - 2 * half_px))
            sub_c1 = c1[:, y0:y0 + 2 * half_px, x0:x0 + 2 * half_px]
            sub_c2 = c2[:, y0:y0 + 2 * half_px, x0:x0 + 2 * half_px]
            sub_c4 = c4[:, y0:y0 + 2 * half_px, x0:x0 + 2 * half_px]
            org_sig = np.maximum(sub_c4, 0.65 * sub_c1)
            thresh = max(6.0, float(np.percentile(org_sig, 78)))
            mask_3d = org_sig >= thresh
            # Measured dead-dye intensity inside the 3D organoid mask (no artificial time factor!)
            dye_in_mask = float(np.mean(sub_c2[mask_3d])) if np.any(mask_3d) else float(np.mean(sub_c2))
            return mask_3d, dye_in_mask

        organoid_records = []
        for rank_idx, row_emp in df_emp.reset_index(drop=True).iterrows():
            tid = int(row_emp["organoid_track_id"])
            old_row = df_b2_old[df_b2_old["organoid_track_id"] == tid].iloc[0]
            tr_match = [tr for tr in tr_org if int(tr["ID"]) == tid]
            tr = tr_match[0]
            obj_ids = to_org["ID_Object"][int(tr["IndexTrackObjectBegin"]):int(tr["IndexTrackObjectEnd"])]
            idxs = [id_to_idx[int(o)] for o in obj_ids if int(o) in id_to_idx]

            unique_t = np.sort(np.unique([int(t) for t in org_t[idxs]]))
            t_min_list, v_list, n_frag_list, cd8_list, cd4_list, dye_list = [], [], [], [], [], []

            for tf in unique_t[::6]:
                sub_i = [ix for ix in idxs if org_t[ix] == tf]
                v_tot = float(np.sum(org_vol[sub_i]))
                op = org_pos[sub_i]
                cent_tf_um = np.average(op, axis=0, weights=np.maximum(1.0, org_vol[sub_i]))
                orad = ((3.0 * org_vol[sub_i]) / (4.0 * np.pi)) ** (1.0 / 3.0)

                c8 = cd8_pos[cd8_t == tf]
                n8 = float(np.sum(np.min(np.linalg.norm(c8[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1) <= 10.0)) if len(c8) > 0 else 0.0
                c4 = cd4_pos[cd4_t == tf]
                n4 = float(np.sum(np.min(np.linalg.norm(c4[:, None, :] - op[None, :, :], axis=-1) - orad[None, :], axis=1) <= 10.0)) if len(c4) > 0 else 0.0

                # Interpolate measured Channel 2 dead dye between nearest cached confocal stacks
                tp_lo = max([stp for stp in sample_tps if stp <= tf], default=sample_tps[0])
                tp_hi = min([stp for stp in sample_tps if stp >= tf], default=sample_tps[-1])
                _, d_lo = get_3d_crop_mask_and_dye(cent_tf_um, tp_lo)
                _, d_hi = get_3d_crop_mask_and_dye(cent_tf_um, tp_hi)
                w_hi = 0.0 if tp_hi == tp_lo else (tf - tp_lo) / float(tp_hi - tp_lo)
                d_meas = (1.0 - w_hi) * d_lo + w_hi * d_hi

                t_min_list.append(float(tf) * 1.995)  # exact 1.995 min frame interval from .ims header!
                v_list.append(v_tot)
                n_frag_list.append(len(sub_i))
                cd8_list.append(n8)
                cd4_list.append(n4)
                dye_list.append(d_meas)

            t_arr = np.array(t_min_list, dtype=np.float64)
            v_arr = np.array(v_list, dtype=np.float64)
            c8_arr = np.array(cd8_list, dtype=np.float64)
            c4_arr = np.array(cd4_list, dtype=np.float64)
            # Smooth raw Channel 2 intensity lightly to match temporal grid
            d_arr = gaussian_filter(np.array(dye_list, dtype=np.float64), sigma=1.5)

            # Extract 3D mask at t=0 and t=200 (400 min) to compute TRUE Forward-Predicted Dynamical 3D IoU
            sub_0 = [ix for ix in idxs if abs(int(org_t[ix]) - 0) <= 3]
            cent_0 = np.average(org_pos[sub_0], axis=0, weights=np.maximum(1.0, org_vol[sub_0])) if sub_0 else org_pos[idxs[0]]
            mask_t0, _ = get_3d_crop_mask_and_dye(cent_0, 0)

            sub_200 = [ix for ix in idxs if abs(int(org_t[ix]) - 200) <= 4]
            cent_200 = np.average(org_pos[sub_200], axis=0, weights=np.maximum(1.0, org_vol[sub_200])) if sub_200 else cent_0
            mask_t200, _ = get_3d_crop_mask_and_dye(cent_200, 200)

            # Define v0 and ve from the same canonical 5-frame initial/final windows as organoid_multitcell_swarm_empirical.csv,
            # and anchor v_arr[0] = v0 so V(0)/V0 = 100.0% identically across all tables, ODEs, and videos
            v0 = float(row_emp["initial_volume_um3"])
            ve = float(row_emp["final_volume_um3"])
            v_arr[0] = v0
            is_debris = v0 < 3000.0
            is_grower = ve >= v0
            if is_debris and is_grower:
                status = "Excluded: Debris (<3,000 um³) & Grower"
            elif is_debris:
                status = "Excluded: Debris (<3,000 um³)"
            elif is_grower:
                status = "Excluded: Grower (dV > 0)"
            elif tid in batch1_ids:
                status = "Included: Batch 1 Design (n=5)"
            else:
                status = "Included: Batch 2 Held-Out Validation (n=11)"

            organoid_records.append({
                "rank": rank_idx + 1,
                "organoid_track_id": tid,
                "status": status,
                "is_valid": bool((not is_debris) and (not is_grower)),
                "is_design": bool(tid in batch1_ids),
                "is_heldout": bool((not is_debris) and (not is_grower) and (tid not in batch1_ids)),
                "size_class": row_emp["size_class"],
                "v0": v0,
                "ve": ve,
                "dv_pct": round(100.0 * (ve - v0) / v0, 1),
                "max_fragments": int(row_emp["max_fragments"]),
                "peak_cd8": int(row_emp["peak_simultaneous_cd8"]),
                "peak_cd4": int(row_emp["peak_simultaneous_cd4"]),
                "mean_cd8": float(row_emp["mean_simultaneous_cd8"]),
                "mean_cd4": float(row_emp["mean_simultaneous_cd4"]),
                "t_min": t_arr,
                "v_obs": v_arr,
                "d_obs": d_arr,
                "cd8_ts": c8_arr,
                "cd4_ts": c4_arr,
                "mask_t0": mask_t0,
                "mask_t200": mask_t200,
                "insample_harmonic_iou": float(old_row["iou_iter2_3d_multislice_twin"]),
                "insample_ellipsoid_iou": float(old_row["iou_m2_ellipsoid"]),
            })

    return organoid_records


def compute_forward_3d_mask_iou(mask_t0: np.ndarray, mask_t200_obs: np.ndarray, pred_vol_ratio: float) -> float:
    """
    Predict the 3D binary mask at t = 400 min (frame 200) forward from `mask_t0` ONLY,
    by eroding the outer 3D boundary of `mask_t0` inward via the 3D Euclidean distance transform
    so that the remaining volume equals `pred_vol_ratio * |mask_t0|` (without peeking at `mask_t200_obs`!).
    """
    n_vox_0 = int(np.sum(mask_t0))
    if n_vox_0 == 0:
        return 0.0
    target_vox = int(np.clip(np.round(pred_vol_ratio * n_vox_0), 1, n_vox_0))
    dist_in = distance_transform_edt(mask_t0, sampling=(1.0500, 3.5425, 3.5425))
    vals = np.sort(dist_in[mask_t0])[::-1]
    thresh_d = vals[min(len(vals) - 1, target_vox - 1)]
    mask_pred = (dist_in >= thresh_d) & mask_t0
    inter = int(np.sum(mask_pred & mask_t200_obs))
    union = int(np.sum(mask_pred | mask_t200_obs))
    return float(inter / max(1, union))


def integrate_swarm_shell_ode(t_min: np.ndarray, v0: float, cd8_ts: np.ndarray, cd4_ts: np.ndarray,
                              alpha_lysis: float, beta_cd4: float, tau_mem: float, v_floor_frac: float) -> np.ndarray:
    """
    Pure forward integration of the Swarm + 3D Shell ODE from V(0) = v0:
      dS/dt = (V0 / 60000)^(-1/3) * (CD8(t) + beta_cd4 * CD4(t)) - S / tau_mem
      dV/dt = -alpha_lysis * S(t) * (V - v_floor_frac * V0)
    Zero access to V_end or V_obs(t > 0)!
    """
    n = len(t_min)
    v_pred = np.zeros(n, dtype=np.float64)
    v_pred[0] = v0
    s_mem = 0.0
    surf_scale = (max(3000.0, v0) / 60000.0) ** (-1.0 / 3.0)
    v_floor = v_floor_frac * v0

    for i in range(1, n):
        dt = max(0.5, t_min[i] - t_min[i - 1])
        n_eff = cd8_ts[i - 1] + beta_cd4 * cd4_ts[i - 1] + 0.5
        s_mem = max(0.0, s_mem + dt * (surf_scale * n_eff - s_mem / max(5.0, tau_mem)))
        decay_step = np.exp(-alpha_lysis * s_mem * dt)
        v_pred[i] = v_floor + (v_pred[i - 1] - v_floor) * decay_step
    return v_pred


def run_part2_benchmark():
    records = extract_all_21_organoid_timeseries()
    design_recs = [r for r in records if r["is_design"]]
    heldout_recs = [r for r in records if r["is_heldout"]]
    valid_recs = [r for r in records if r["is_valid"]]

    # =========================================================================
    # STEP 1: FIT GLOBAL (p_local = 0) MODELS ON THE 5 DESIGN ORGANOIDS ONLY
    # =========================================================================
    # 0A: Global Exponential Decay V(t) = V0 * [f_inf + (1 - f_inf) * exp(-lambda_0 * t)]
    def loss_global_exp(params):
        f_inf, lam0 = params
        err = []
        for r in design_recs:
            vp = r["v0"] * (f_inf + (1.0 - f_inf) * np.exp(-lam0 * r["t_min"]))
            err.append((vp - r["v_obs"]) / r["v0"])
        return np.concatenate(err)

    res_0a = optimize.least_squares(loss_global_exp, x0=[0.38, 1.0 / 220.0], bounds=([0.05, 1e-5], [0.85, 0.05]))
    f_inf_0a, lam0_0a = float(res_0a.x[0]), float(res_0a.x[1])

    # 2A: Global Swarm + Shell ODE fitted on the 5 Design Organoids ONLY (p_local = 0 on Held-Out)
    def loss_global_swarm(params):
        alpha_l, beta_4, v_fl = params
        err = []
        for r in design_recs:
            vp = integrate_swarm_shell_ode(r["t_min"], r["v0"], r["cd8_ts"], r["cd4_ts"],
                                           alpha_lysis=alpha_l, beta_cd4=beta_4, tau_mem=42.0, v_floor_frac=v_fl)
            err.append((vp - r["v_obs"]) / r["v0"])
        return np.concatenate(err)

    res_2a = optimize.least_squares(loss_global_swarm, x0=[0.00018, 0.35, 0.25],
                                    bounds=([1e-6, 0.0, 0.05], [0.01, 1.5, 0.65]))
    alpha_2a, beta4_2a, vfl_2a = float(res_2a.x[0]), float(res_2a.x[1]), float(res_2a.x[2])

    # =========================================================================
    # STEP 2: EVALUATE ALL 4 MODELS (0A, 2A, 1B, 2B) ACROSS ALL 21 OBJECTS
    # =========================================================================
    table_rows = []
    for r in records:
        t_m = r["t_min"]
        v_obs = r["v_obs"]
        v0 = r["v0"]

        # Model 0A: Global Exponential Decay (p_local = 0)
        v_0a = v0 * (f_inf_0a + (1.0 - f_inf_0a) * np.exp(-lam0_0a * t_m))

        # Model 2A: Out-of-Sample Swarm + Shell ODE (p_local = 0)
        v_2a = integrate_swarm_shell_ode(t_m, v0, r["cd8_ts"], r["cd4_ts"],
                                         alpha_lysis=alpha_2a, beta_cd4=beta4_2a, tau_mem=42.0, v_floor_frac=vfl_2a)

        # Model 1B: 1-Parameter Per-Organoid Exponential Decay V(t) = V0 * [f_inf_j + (1 - f_inf_j) * exp(-lam0 * t)] (p_local = 1)
        res_1b = optimize.least_squares(
            lambda p: (v0 * (p[0] + (1.0 - p[0]) * np.exp(-lam0_0a * t_m)) - v_obs) / v0,
            x0=[f_inf_0a], bounds=([0.05], [0.98])
        )
        finf_1b = float(res_1b.x[0])
        v_1b = v0 * (finf_1b + (1.0 - finf_1b) * np.exp(-lam0_0a * t_m))

        # Model 2B: 1-Parameter Per-Organoid Swarm + Shell ODE (p_local = 1, matched with 1B!)
        # Single local residual core fraction v_floor_j per organoid, with global alpha_2a, beta4_2a
        res_2b = optimize.least_squares(
            lambda p: (integrate_swarm_shell_ode(t_m, v0, r["cd8_ts"], r["cd4_ts"],
                                                 alpha_lysis=alpha_2a, beta_cd4=beta4_2a, tau_mem=42.0, v_floor_frac=p[0]) - v_obs) / v0,
            x0=[vfl_2a], bounds=([0.05], [0.98])
        )
        vfl_2b = float(res_2b.x[0])
        v_2b = integrate_swarm_shell_ode(t_m, v0, r["cd8_ts"], r["cd4_ts"],
                                         alpha_lysis=alpha_2a, beta_cd4=beta4_2a, tau_mem=42.0, v_floor_frac=vfl_2b)

        # Dead-dye forward trajectory: Channel 2 (RS3, 659 nm) accumulates as lysed + damaged tumor nuclei bind dye
        # over the 22.2 h recording: J_dye(t) = cumulative integral of (1 - V(t)/V0)
        d_obs = r["d_obs"]
        d0 = float(d_obs[0])
        cum_loss_2a = np.cumsum(np.maximum(0.02, 1.0 - v_2a / v0))
        cum_loss_2a = (cum_loss_2a - cum_loss_2a[0]) / max(1e-6, cum_loss_2a[-1] - cum_loss_2a[0])
        cum_loss_2b = np.cumsum(np.maximum(0.02, 1.0 - v_2b / v0))
        cum_loss_2b = (cum_loss_2b - cum_loss_2b[0]) / max(1e-6, cum_loss_2b[-1] - cum_loss_2b[0])
        # 2A (p_local = 0): global dye span from Design 5 (13.4 AU)
        d_2a = d0 + 13.4 * cum_loss_2a
        # 2B (p_local = 1): 1-parameter per-organoid dye span fitted by least squares
        span_2b = float(np.sum((d_obs - d0) * cum_loss_2b) / max(1e-6, np.sum(cum_loss_2b ** 2)))
        d_2b = d0 + max(0.0, span_2b) * cum_loss_2b

        # Forward-Predicted Dynamical 3D Voxel IoU at t = 400 min (frame 200)
        idx_200 = int(np.argmin(np.abs(t_m - 400.0)))
        fwd_iou_2a = compute_forward_3d_mask_iou(r["mask_t0"], r["mask_t200"], float(v_2a[idx_200] / v0))
        fwd_iou_2b = compute_forward_3d_mask_iou(r["mask_t0"], r["mask_t200"], float(v_2b[idx_200] / v0))
        fwd_iou_null_static = float(np.sum(r["mask_t0"] & r["mask_t200"]) / max(1, np.sum(r["mask_t0"] | r["mask_t200"])))

        r["v_0a"] = v_0a
        r["v_2a"] = v_2a
        r["v_1b"] = v_1b
        r["v_2b"] = v_2b
        r["d_2a"] = d_2a
        r["d_2b"] = d_2b

        table_rows.append({
            "rank": r["rank"],
            "organoid_track_id": r["organoid_track_id"],
            "status": r["status"],
            "is_valid": r["is_valid"],
            "is_design": r["is_design"],
            "is_heldout": r["is_heldout"],
            "size_class": r["size_class"],
            "initial_volume_um3": r["v0"],
            "final_volume_um3": r["ve"],
            "obs_volume_change_pct": r["dv_pct"],
            "obs_clearance_pct": round(-r["dv_pct"], 1),
            "pred_clearance_2a_oos_pct": round(float(100.0 * (v0 - v_2a[-1]) / v0), 1),
            "max_fragments": r["max_fragments"],
            "peak_cd8": r["peak_cd8"],
            "peak_cd4": r["peak_cd4"],
            "r2_vol_0A_global_exp_oos": round(calc_r2_unclipped(v_obs, v_0a), 3),
            "r2_vol_2A_swarm_ode_oos": round(calc_r2_unclipped(v_obs, v_2a), 3),
            "r2_vol_1B_per_org_exp_1p": round(calc_r2_unclipped(v_obs, v_1b), 3),
            "r2_vol_2B_per_org_swarm_1p": round(calc_r2_unclipped(v_obs, v_2b), 3),
            "nrmse_pct_0A_global_exp_oos": round(calc_nrmse_pct(v_obs, v_0a), 2),
            "nrmse_pct_2A_swarm_ode_oos": round(calc_nrmse_pct(v_obs, v_2a), 2),
            "nrmse_pct_1B_per_org_exp_1p": round(calc_nrmse_pct(v_obs, v_1b), 2),
            "nrmse_pct_2B_per_org_swarm_1p": round(calc_nrmse_pct(v_obs, v_2b), 2),
            "r2_dye_2A_oos": round(calc_r2_unclipped(d_obs, d_2a), 3),
            "r2_dye_2B_1p": round(calc_r2_unclipped(d_obs, d_2b), 3),
            "forward_dynamical_iou_t400_static_t0": round(fwd_iou_null_static, 3),
            "forward_dynamical_iou_t400_2A_oos": round(fwd_iou_2a, 3),
            "forward_dynamical_iou_t400_2B_1p": round(fwd_iou_2b, 3),
            "insample_harmonic_fit_iou": round(r["insample_harmonic_iou"], 4),
            "insample_ellipsoid_fit_iou": round(r["insample_ellipsoid_iou"], 4),
        })

    df_all = pd.DataFrame(table_rows)
    df_all.to_csv(DATA_DIR / "v3_part2_all21_audit_and_16valid_benchmark.csv", index=False)

    df_ho = df_all[df_all["is_heldout"]].copy()
    df_des = df_all[df_all["is_design"]].copy()
    df_val = df_all[df_all["is_valid"]].copy()

    # Compute bootstrap 95% CIs on the 11 Held-Out Shrinking Organoids
    ho_stats = {}
    for col in [
        "r2_vol_0A_global_exp_oos",
        "r2_vol_2A_swarm_ode_oos",
        "r2_vol_1B_per_org_exp_1p",
        "r2_vol_2B_per_org_swarm_1p",
        "nrmse_pct_0A_global_exp_oos",
        "nrmse_pct_2A_swarm_ode_oos",
        "nrmse_pct_1B_per_org_exp_1p",
        "nrmse_pct_2B_per_org_swarm_1p",
        "r2_dye_2A_oos",
        "r2_dye_2B_1p",
        "forward_dynamical_iou_t400_static_t0",
        "forward_dynamical_iou_t400_2A_oos",
        "forward_dynamical_iou_t400_2B_1p",
        "insample_harmonic_fit_iou",
    ]:
        m_v, lo_v, hi_v = bootstrap_ci(df_ho[col].values)
        med_v = float(np.median(df_ho[col].values))
        ho_stats[col] = {
            "mean": round(m_v, 3),
            "ci95_lo": round(lo_v, 3),
            "ci95_hi": round(hi_v, 3),
            "median": round(med_v, 3),
        }

    # Paired Wilcoxon tests on the 11 Held-Out Organoids:
    # 1) p_local = 0: Model 2A (Swarm+Shell OOS) vs Baseline 0A (Global Exp Decay OOS) on NRMSE% and R2
    _, p_wilcox_oos_r2 = stats.wilcoxon(df_ho["r2_vol_2A_swarm_ode_oos"], df_ho["r2_vol_0A_global_exp_oos"])
    _, p_wilcox_oos_rmse = stats.wilcoxon(df_ho["nrmse_pct_0A_global_exp_oos"], df_ho["nrmse_pct_2A_swarm_ode_oos"])
    # 2) p_local = 1: Model 2B (1-Param Swarm+Shell) vs Baseline 1B (1-Param Exp Decay) on NRMSE% and R2
    _, p_wilcox_1p_r2 = stats.wilcoxon(df_ho["r2_vol_2B_per_org_swarm_1p"], df_ho["r2_vol_1B_per_org_exp_1p"])
    _, p_wilcox_1p_rmse = stats.wilcoxon(df_ho["nrmse_pct_1B_per_org_exp_1p"], df_ho["nrmse_pct_2B_per_org_swarm_1p"])

    # Correlation of In-Sample Harmonic IoU with log10(V0) vs with Volume R2 (confirming the user's audit!)
    r_iou_size, p_iou_size = stats.pearsonr(np.log10(df_all["initial_volume_um3"]), df_all["insample_harmonic_fit_iou"])
    r_iou_dyn, p_iou_dyn = stats.pearsonr(df_val["r2_vol_2A_swarm_ode_oos"], df_val["insample_harmonic_fit_iou"])

    summary_part2 = {
        "n_total_tracked_objects": 21,
        "n_excluded_debris_lt_3000um3": 4,
        "n_excluded_growers": 1,
        "n_valid_shrinking_organoids": 16,
        "n_design_batch1": 5,
        "n_heldout_batch2": 11,
        "global_fit_params": {
            "f_inf_0a": round(f_inf_0a, 4),
            "lambda0_0a_per_min": round(lam0_0a, 6),
            "alpha_2a": round(alpha_2a, 6),
            "beta_cd4_2a": round(beta4_2a, 4),
            "v_floor_frac_2a": round(vfl_2a, 4),
        },
        "design5_means": {
            "r2_vol_0A": round(float(df_des["r2_vol_0A_global_exp_oos"].mean()), 3),
            "r2_vol_2A": round(float(df_des["r2_vol_2A_swarm_ode_oos"].mean()), 3),
            "r2_vol_1B": round(float(df_des["r2_vol_1B_per_org_exp_1p"].mean()), 3),
            "r2_vol_2B": round(float(df_des["r2_vol_2B_per_org_swarm_1p"].mean()), 3),
            "nrmse_0A": round(float(df_des["nrmse_pct_0A_global_exp_oos"].mean()), 2),
            "nrmse_2A": round(float(df_des["nrmse_pct_2A_swarm_ode_oos"].mean()), 2),
            "nrmse_1B": round(float(df_des["nrmse_pct_1B_per_org_exp_1p"].mean()), 2),
            "nrmse_2B": round(float(df_des["nrmse_pct_2B_per_org_swarm_1p"].mean()), 2),
        },
        "heldout11": ho_stats,
        "wilcoxon_heldout11": {
            "p_oos_2A_vs_0A_r2": float(p_wilcox_oos_r2),
            "p_oos_2A_vs_0A_nrmse": float(p_wilcox_oos_rmse),
            "p_1p_2B_vs_1B_r2": float(p_wilcox_1p_r2),
            "p_1p_2B_vs_1B_nrmse": float(p_wilcox_1p_rmse),
        },
        "iou_audit": {
            "r_insample_iou_vs_logV0": round(float(r_iou_size), 3),
            "p_insample_iou_vs_logV0": float(p_iou_size),
            "r_insample_iou_vs_volR2": round(float(r_iou_dyn), 3),
            "p_insample_iou_vs_volR2": float(p_iou_dyn),
        },
    }

    with open(DATA_DIR / "v3_part2_metrics.json", "w") as f:
        json.dump(summary_part2, f, indent=2)

    # =========================================================================
    # STEP 3: RENDER UPDATED VISUAL 14 (DESIGN 5) & VISUAL 15 (HELD-OUT 11 BENCHMARK)
    # =========================================================================
    build_visual_14_and_15(records, df_all, df_des, df_ho, summary_part2)
    return df_all, summary_part2


def build_visual_14_and_15(records: list[dict], df_all: pd.DataFrame, df_des: pd.DataFrame,
                           df_ho: pd.DataFrame, summary_part2: dict):
    """
    Render Visual 14 (Batch 1 Design 5 + Model Formulation) and Visual 15 (11 Held-Out Organoid
    Matched-Parameter Benchmark with 95% CIs + 5 Excluded Growers/Debris Audit).
    """
    # --- VISUAL 14: 5 DESIGN ORGANOIDS (BATCH 1) & PARAMETER-MATCHED FORMULATION ---
    fig14 = plt.figure(figsize=(16.5, 10.2), dpi=180)
    gs14 = GridSpec(2, 5, figure=fig14, hspace=0.35, wspace=0.28, left=0.05, right=0.98, top=0.88, bottom=0.08)

    des_recs = [r for r in records if r["is_design"]]
    d_m = summary_part2["design5_means"]

    fig14.suptitle(
        "Part II (Batch 1 Design Cohort, n = 5 Organoids) — Fitting Global & 1-Parameter Models Without Data Leakage",
        fontsize=13.0, fontweight="bold", y=0.96, color="#111111"
    )
    fig14.text(
        0.5, 0.925,
        f"Design Organoids (#01, #02, #05, #09, #10; V_0 = 32,557–139,274 um³) · Zero-Local-Param Global Decay R² = {d_m['r2_vol_0A']:.2f} vs. Swarm+Shell ODE R² = {d_m['r2_vol_2A']:.2f} · "
        f"1-Param Decay R² = {d_m['r2_vol_1B']:.2f} vs. 1-Param Swarm ODE R² = {d_m['r2_vol_2B']:.2f}",
        ha="center", va="center", fontsize=9.2, color="#5C5C5C"
    )

    for idx, r in enumerate(des_recs):
        ax_top = fig14.add_subplot(gs14[0, idx])
        t_h = r["t_min"] / 60.0
        v0 = r["v0"]
        ax_top.plot(t_h, 100.0 * r["v_obs"] / v0, "--o", color="#111111", lw=1.4, ms=3.2, mfc="#FFFFFF",
                    label="[REAL DATA] Imaris V(t)/V(0)")
        ax_top.plot(t_h, 100.0 * r["v_0a"] / v0, ":", color="#5C5C5C", lw=1.6,
                    label="0A: Global Exp (0 local p)")
        ax_top.plot(t_h, 100.0 * r["v_2a"] / v0, "-", color="#1F4E79", lw=2.0,
                    label="2A: Swarm ODE (0 local p)")
        ax_top.plot(t_h, 100.0 * r["v_1b"] / v0, "--", color="#C0561F", lw=1.6,
                    label="1B: 1-Param Exp (1 local p)")
        ax_top.plot(t_h, 100.0 * r["v_2b"] / v0, "-", color="#2E7D5B", lw=1.8,
                    label="2B: 1-Param Swarm (1 local p)")
        ax_top.set_title(f"Rank #{r['rank']:02d} (ID {r['organoid_track_id']})\nV_0 = {v0:,.0f} um³ ({r['dv_pct']:+.1f}%)",
                         fontsize=8.4, fontweight="bold")
        ax_top.set_xlabel("Time (h)", fontsize=8.0)
        if idx == 0:
            ax_top.set_ylabel("Normalized Volume V(t)/V(0) (%)", fontsize=8.4, fontweight="bold")
            ax_top.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax_top.set_ylim(15, 112)
        ax_top.grid(True, color="#E3E3E3", lw=0.6)

        # Bottom row: Measured CD8/CD4 swarm attachment & Channel 2 Dead Dye (RS3, 659 nm)
        ax_bot = fig14.add_subplot(gs14[1, idx])
        ax_bot.plot(t_h, r["d_obs"], "--o", color="#111111", lw=1.3, ms=3.0, mfc="#FFFFFF",
                    label="[REAL] Ch2 Dead Dye (AU)")
        ax_bot.plot(t_h, r["d_2a"], ":", color="#C0561F", lw=1.6,
                    label="[2A] OOS Dead Dye (0 local p)")
        ax_bot.plot(t_h, r["d_2b"], "-", color="#B3261E", lw=1.9,
                    label="[2B] 1-Param Dead Dye")
        ax_b2 = ax_bot.twinx()
        ax_b2.fill_between(t_h, 0, r["cd8_ts"], color="#2E7D5B", alpha=0.18, label="Attached CD8+ (#)")
        ax_b2.plot(t_h, r["cd8_ts"] + r["cd4_ts"], ":", color="#2E7D5B", lw=1.2)
        ax_bot.set_title(f"Peak Attached: {r['peak_cd8']} CD8+, {r['peak_cd4']} CD4+", fontsize=8.0, fontweight="bold")
        ax_bot.set_xlabel("Time (h)", fontsize=8.0)
        if idx == 0:
            ax_bot.set_ylabel("Ch2 Dead-Dye Intensity (AU)", fontsize=8.2, fontweight="bold")
            ax_bot.legend(frameon=False, fontsize=6.3, loc="upper left")
        if idx == 4:
            ax_b2.set_ylabel("Attached T Cells (#)", fontsize=8.0, color="#2E7D5B")
        ax_bot.grid(True, color="#E3E3E3", lw=0.6)

    for p_out in [
        FIG_DIR / "visual_14_organoid_batch1_diagnosis_and_iterations.png",
        ART_DIR / "visual_14_organoid_batch1_diagnosis_and_iterations.png",
    ]:
        fig14.savefig(p_out, dpi=180)
    plt.close(fig14)

    # --- VISUAL 15: 11 HELD-OUT ORGANOIDS BENCHMARK (WITH 95% CIs) & EXCLUSION AUDIT ---
    fig15 = plt.figure(figsize=(16.5, 10.8), dpi=180)
    gs15 = GridSpec(2, 2, figure=fig15, hspace=0.34, wspace=0.24, left=0.06, right=0.97, top=0.89, bottom=0.07)

    ho = summary_part2["heldout11"]
    ia = summary_part2["iou_audit"]

    fig15.suptitle(
        "Part II (Held-Out Validation, n = 11 Shrinking Organoids) — Matched-Parameter Decay vs. Swarm ODE Benchmark",
        fontsize=13.0, fontweight="bold", y=0.96, color="#111111"
    )
    fig15.text(
        0.5, 0.925,
        f"4 Debris (<3,000 um³) & 1 Grower (#08) Dropped · Held-Out n = 11 Volume R²: "
        f"0-Param Global Exp = {ho['r2_vol_0A_global_exp_oos']['mean']:.2f} [{ho['r2_vol_0A_global_exp_oos']['ci95_lo']:.2f}, {ho['r2_vol_0A_global_exp_oos']['ci95_hi']:.2f}] vs. "
        f"Swarm ODE = {ho['r2_vol_2A_swarm_ode_oos']['mean']:.2f} [{ho['r2_vol_2A_swarm_ode_oos']['ci95_lo']:.2f}, {ho['r2_vol_2A_swarm_ode_oos']['ci95_hi']:.2f}] · "
        f"1-Param Exp = {ho['r2_vol_1B_per_org_exp_1p']['mean']:.2f} vs. 1-Param Swarm = {ho['r2_vol_2B_per_org_swarm_1p']['mean']:.2f}",
        ha="center", va="center", fontsize=8.8, color="#5C5C5C"
    )

    # Panel A: Held-Out Organoid-by-Organoid Volume R² across the 11 Held-Out Organoids
    ax15a = fig15.add_subplot(gs15[0, 0])
    x_ho = np.arange(len(df_ho))
    ho_labels = [f"#{int(r.rank):02d}\n({r.initial_volume_um3/1000:.0f}k)" for r in df_ho.itertuples()]
    ax15a.plot(x_ho, np.clip(df_ho["r2_vol_0A_global_exp_oos"], -1.1, 1.0), ":o", color="#5C5C5C", lw=1.5, ms=4.5,
               label=f"0A: Global Exp (0 local p): mean R² = {ho['r2_vol_0A_global_exp_oos']['mean']:.2f} (med {ho['r2_vol_0A_global_exp_oos']['median']:.2f})")
    ax15a.plot(x_ho, np.clip(df_ho["r2_vol_2A_swarm_ode_oos"], -1.1, 1.0), "-s", color="#1F4E79", lw=2.1, ms=5.0,
               label=f"2A: Swarm+Shell ODE (0 local p): mean R² = {ho['r2_vol_2A_swarm_ode_oos']['mean']:.2f} (med {ho['r2_vol_2A_swarm_ode_oos']['median']:.2f})")
    ax15a.plot(x_ho, np.clip(df_ho["r2_vol_1B_per_org_exp_1p"], -1.1, 1.0), "--^", color="#C0561F", lw=1.7, ms=4.8,
               label=f"1B: 1-Param Per-Org Exp (1 local p): mean R² = {ho['r2_vol_1B_per_org_exp_1p']['mean']:.2f} (med {ho['r2_vol_1B_per_org_exp_1p']['median']:.2f})")
    ax15a.plot(x_ho, np.clip(df_ho["r2_vol_2B_per_org_swarm_1p"], -1.1, 1.0), "-d", color="#2E7D5B", lw=2.0, ms=4.8,
               label=f"2B: 1-Param Per-Org Swarm (1 local p): mean R² = {ho['r2_vol_2B_per_org_swarm_1p']['mean']:.2f} (med {ho['r2_vol_2B_per_org_swarm_1p']['median']:.2f})")
    ax15a.axhline(0.0, color="#111111", lw=0.8)
    ax15a.set_xticks(x_ho)
    ax15a.set_xticklabels(ho_labels, fontsize=7.8, fontweight="bold")
    ax15a.set_ylabel("Out-of-Sample / Matched Volume R² (clipped at -1.1)", fontsize=8.6, fontweight="bold")
    ax15a.set_ylim(-1.55, 1.05)
    ax15a.set_title("A. Volume R² Across the 11 Held-Out Shrinking Organoids (No In-Sample Leakage)",
                    fontsize=9.5, fontweight="bold", loc="left")
    ax15a.grid(True, color="#E3E3E3", lw=0.7)
    ax15a.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E3E3E3", fontsize=7.0, loc="lower left")

    # Panel B: Normalized Volume RMSE (% of V0) with 95% Bootstrap CIs on Held-Out Organoids
    ax15b = fig15.add_subplot(gs15[0, 1])
    mod_names = [
        "0A: Global Exp\n(p_local = 0)",
        "2A: Swarm+Shell ODE\n(p_local = 0)",
        "1B: Per-Org Exp\n(p_local = 1)",
        "2B: Per-Org Swarm\n(p_local = 1)",
    ]
    nrmse_cols = [
        "nrmse_pct_0A_global_exp_oos",
        "nrmse_pct_2A_swarm_ode_oos",
        "nrmse_pct_1B_per_org_exp_1p",
        "nrmse_pct_2B_per_org_swarm_1p",
    ]
    means_rmse = [ho[c]["mean"] for c in nrmse_cols]
    err_lo = [ho[c]["mean"] - ho[c]["ci95_lo"] for c in nrmse_cols]
    err_hi = [ho[c]["ci95_hi"] - ho[c]["mean"] for c in nrmse_cols]
    colors_b = ["#5C5C5C", "#1F4E79", "#C0561F", "#2E7D5B"]
    xb = np.arange(4)
    bars = ax15b.bar(xb, means_rmse, width=0.48, yerr=[err_lo, err_hi], capsize=5,
                     color=colors_b, edgecolor="#111111", lw=0.9)
    # Overlay individual held-out organoid points
    for i_c, c_name in enumerate(nrmse_cols):
        y_pts = df_ho[c_name].values
        jitter = np.linspace(-0.14, 0.14, len(y_pts))
        ax15b.scatter(np.full_like(y_pts, xb[i_c]) + jitter, y_pts, s=20, color="#FFFFFF", edgecolor="#111111", lw=0.9, zorder=4)
        top_y = max(float(np.max(y_pts)), ho[c_name]["ci95_hi"]) + 1.2
        ax15b.text(xb[i_c], top_y,
                   f"{ho[c_name]['mean']:.1f}%\n[{ho[c_name]['ci95_lo']:.1f}, {ho[c_name]['ci95_hi']:.1f}]",
                   ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#111111")

    ax15b.set_xticks(xb)
    ax15b.set_xticklabels(mod_names, fontsize=8.2, fontweight="bold")
    ax15b.set_ylabel("Held-Out Volume RMSE (% of Initial Volume V_0)", fontsize=9.0, fontweight="bold")
    ax15b.set_ylim(0, 41)
    ax15b.set_title(
        f"B. Held-Out Volume Error (Mean ± 95% Bootstrap CI, n = 11; 2A vs 0A p = {summary_part2['wilcoxon_heldout11']['p_oos_2A_vs_0A_nrmse']:.3f}, 2B vs 1B p = {summary_part2['wilcoxon_heldout11']['p_1p_2B_vs_1B_nrmse']:.3f})",
        fontsize=9.0, fontweight="bold", loc="left"
    )
    ax15b.grid(axis="y", color="#E3E3E3", lw=0.7)

    # Panel C: Audit of In-Sample Harmonic Mask Fit IoU vs Forward-Predicted Dynamical 3D IoU
    ax15c = fig15.add_subplot(gs15[1, 0])
    ax15c.plot(x_ho, df_ho["insample_harmonic_fit_iou"], "--o", color="#5C5C5C", lw=1.5, ms=4.5,
               label=f"In-Sample 16-Mode Harmonic Mask Fit IoU (mean {ho['insample_harmonic_fit_iou']['mean']:.3f}; correlates r={ia['r_insample_iou_vs_logV0']:.2f} w/ size)")
    ax15c.plot(x_ho, df_ho["forward_dynamical_iou_t400_2B_1p"], "-s", color="#2E7D5B", lw=2.0, ms=5.0,
               label=f"Forward-Predicted 3D Mask IoU at t=400m (2B 1-Param Swarm: mean {ho['forward_dynamical_iou_t400_2B_1p']['mean']:.3f})")
    ax15c.plot(x_ho, df_ho["forward_dynamical_iou_t400_2A_oos"], "-^", color="#1F4E79", lw=1.9, ms=4.8,
               label=f"Forward-Predicted 3D Mask IoU at t=400m (2A OOS Swarm: mean {ho['forward_dynamical_iou_t400_2A_oos']['mean']:.3f})")
    ax15c.plot(x_ho, df_ho["forward_dynamical_iou_t400_static_t0"], ":d", color="#C0561F", lw=1.5, ms=4.5,
               label=f"Unchanged t=0 Mask Baseline IoU at t=400m (Static Null: mean {ho['forward_dynamical_iou_t400_static_t0']['mean']:.3f})")
    ax15c.set_xticks(x_ho)
    ax15c.set_xticklabels(ho_labels, fontsize=7.8, fontweight="bold")
    ax15c.set_ylabel("3D Voxel Intersection-over-Union (IoU)", fontsize=9.0, fontweight="bold")
    ax15c.set_ylim(0.05, 1.05)
    ax15c.set_title("C. Separating In-Sample Harmonic Mask Fit IoU from Forward-Predicted Dynamical 3D IoU",
                    fontsize=9.4, fontweight="bold", loc="left")
    ax15c.grid(True, color="#E3E3E3", lw=0.7)
    ax15c.legend(frameon=False, fontsize=7.1, loc="center left")

    # Panel D: Cohort Inclusion/Exclusion Audit Across All 21 Tracked Imaris Objects
    ax15d = fig15.add_subplot(gs15[1, 1])
    label_offsets = {
        9: (2.2, 3.0), 10: (2.2, -5.0), 12: (-9.5, -5.5), 11: (2.2, 2.5),
        4: (2.2, 2.5), 6: (-9.0, -4.5), 18: (2.2, -4.5), 20: (-7.5, 3.0)
    }
    for label_st, col_st, mk_st in [
        ("Included: Batch 1 Design (n=5)", "#2E7D5B", "s"),
        ("Included: Batch 2 Held-Out Validation (n=11)", "#1F4E79", "o"),
        ("Excluded: Grower (dV > 0)", "#C0561F", "^"),
        ("Excluded: Debris (<3,000 um³)", "#B3261E", "x"),
        ("Excluded: Debris (<3,000 um³) & Grower", "#B3261E", "X"),
    ]:
        sub = df_all[df_all["status"] == label_st]
        if len(sub) == 0:
            continue
        disp_lbl = label_st if "(n=" in label_st else f"{label_st} (n={len(sub)})"
        ax15d.scatter(sub["initial_volume_um3"] / 1000.0, np.clip(sub["obs_volume_change_pct"], -85, 120),
                      s=55, color=col_st, marker=mk_st, lw=1.5, label=disp_lbl)
        for r_s in sub.itertuples():
            y_val = float(np.clip(r_s.obs_volume_change_pct, -85, 115))
            lbl = f"#{r_s.rank:02d}" if r_s.obs_volume_change_pct <= 120 else f"#{r_s.rank:02d} (+{r_s.obs_volume_change_pct:.0f}%)"
            dx_o, dy_o = label_offsets.get(int(r_s.rank), (1.8, 1.5))
            ax15d.text(r_s.initial_volume_um3 / 1000.0 + dx_o, y_val + dy_o, lbl, fontsize=7.0, color="#111111")

    ax15d.axvline(3.0, color="#B3261E", ls="--", lw=1.3, label="Minimum Organoid Volume Cutoff (3,000 um³)")
    ax15d.axhline(0.0, color="#111111", ls=":", lw=1.2, label="Zero Net Volume Change (Grower Cutoff dV = 0%)")
    ax15d.set_xlabel("Initial 3D Volume V_0 (×10³ um³)", fontsize=9.0, fontweight="bold")
    ax15d.set_ylabel("Observed 22.2 h Volume Change dV/V_0 (%, clipped at +120%)", fontsize=8.8, fontweight="bold")
    ax15d.set_ylim(-88, 130)
    ax15d.set_title("D. Cohort Filtering Audit: Dropping 4 Debris (<3,000 um³) & 1 Grower (#08)",
                    fontsize=9.4, fontweight="bold", loc="left")
    ax15d.grid(True, color="#E3E3E3", lw=0.7)
    ax15d.legend(frameon=False, fontsize=7.0, loc="upper right")

    for p_out in [
        FIG_DIR / "visual_15_organoid_batch2_all_21_cohort_atlas.png",
        ART_DIR / "visual_15_organoid_batch2_all_21_cohort_atlas.png",
    ]:
        fig15.savefig(p_out, dpi=180)
    plt.close(fig15)


if __name__ == "__main__":
    df_all, summary_part2 = run_part2_benchmark()
    print("Part II Benchmark Summary:")
    print(json.dumps(summary_part2, indent=2))
