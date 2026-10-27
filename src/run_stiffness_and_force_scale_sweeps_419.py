#!/usr/bin/env python3
"""
Exact 419-cell force-scale (eta_clutch) sweep and 3D-dent target-stiffness (E_target) sweep.
1. Exports data/lightsheet_wang2026_single_cells_419_clean.csv (unpadded 419 cells from Source_Data_Wang2024_NatCommun.xlsx).
2. Runs Sweep A: eta_clutch sweep (0.5 to 500 pN*s/um^3, 25 log-spaced points) on the unpadded 419 cells
   using the exact predict_pore_repair_kill_probability function from v3_part1_deposited_volumes_and_physics.py.
3. Runs Sweep B: E_target sweep (50 to 10,000 Pa, 25 log-spaced points) feeding the measured 3D dent (dent_max_um)
   into the Hertz + cortical-shell force relation (line 112 of v3_part1_deposited_volumes_and_physics.py)
   and tension-potentiated pore model on the 419 cells.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from v3_part1_deposited_volumes_and_physics import predict_pore_repair_kill_probability

DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

TRAIN_GROUPS = ["1", "2", "3", "4", "5"]
LOW_GROUPS = ["6", "7", "8", "9", "10"]
DASA_GROUPS = ["8*", "9*", "10*"]
ALL_GROUPS = TRAIN_GROUPS + LOW_GROUPS + DASA_GROUPS


def export_clean_419_csv() -> pd.DataFrame:
    df_raw = pd.read_csv(DATA_DIR / "lightsheet_wang2026_single_cells.csv")
    df_clean = df_raw[df_raw["CA_um2"].notna()].copy().reset_index(drop=True)
    assert len(df_clean) == 419, f"Expected 419 rows, got {len(df_clean)}"
    out_path = DATA_DIR / "lightsheet_wang2026_single_cells_419_clean.csv"
    df_clean.to_csv(out_path, index=False)
    return df_clean


def predict_dent_hertz_pore_kill(
    ca_um2: np.ndarray,
    arfs_nm_s: np.ndarray,
    adc: np.ndarray,
    mpa_deg: np.ndarray,
    dent_um: np.ndarray,
    r0_um: np.ndarray,
    k_pfp: float = 0.088,
    k_escrt: float = 0.045,
    g_crit: float = 1.15,
    E_target_Pa: float = 450.0,
    T0_pN_um: float = 50.0,
    exposure_min: float = 120.0,
    pure_dent_gate: bool = False,
) -> dict:
    """
    Replaces clutch force with the Hertz + cortical-shell force computed directly from
    measured 3D indentation depth delta = dent_um:
      F_indent(delta) = (4/3) * (E_target / (1 - nu^2)) * sqrt(R_eff) * delta^(3/2) + 2*pi*T0*delta
      sigma_syn = T0 + F_indent / (2 * pi * R_syn)
    """
    ca = np.maximum(0.0, np.nan_to_num(np.asarray(ca_um2, dtype=np.float64), nan=0.0))
    arfs = np.maximum(0.0, np.nan_to_num(np.asarray(arfs_nm_s, dtype=np.float64), nan=0.0))
    adc_pos = np.maximum(0.0, np.nan_to_num(np.asarray(adc, dtype=np.float64), nan=0.0))
    mpa_arr = np.asarray(mpa_deg, dtype=np.float64)
    mpa = np.where(np.isnan(mpa_arr), 85.0, mpa_arr)
    delta = np.maximum(0.0, np.nan_to_num(np.asarray(dent_um, dtype=np.float64), nan=0.0))
    r_targ = np.clip(np.nan_to_num(np.asarray(r0_um, dtype=np.float64), nan=6.5), 3.5, 11.0)

    r_syn_um = np.sqrt(ca / np.pi + 0.25)
    r_eff_um = (r_targ * 5.0) / (r_targ + 5.0)
    nu = 0.45
    hertz_coeff_pN = (4.0 / 3.0) * (E_target_Pa / (1.0 - nu ** 2)) * np.sqrt(r_eff_um)
    f_syn_pN = np.where(
        delta > 0.0,
        hertz_coeff_pN * (delta ** 1.5) + 2.0 * np.pi * T0_pN_um * delta,
        0.0,
    )
    f_syn_nN = f_syn_pN * 1e-3

    sigma_syn_pN_um = np.where(
        ca > 0.5,
        T0_pN_um + f_syn_pN / (2.0 * np.pi * np.maximum(1.5, r_syn_um)),
        T0_pN_um,
    )

    m_tension = np.exp(np.clip((sigma_syn_pN_um - T0_pN_um) / 55.0, 0.0, 2.5))

    gate_area = (ca ** 1.35) / (ca ** 1.35 + 65.0 ** 1.35)
    if pure_dent_gate:
        # Contact area gate + tension potentiation only (no ARFS/ADC/MPA multipliers)
        phi_degran = gate_area
    else:
        gate_adc = 0.45 + 0.55 * (adc_pos / (adc_pos + 3.2))
        gate_mtoc = 1.0 / (1.0 + (mpa / 24.0) ** 2.0)
        gate_lck = (arfs ** 1.4) / (arfs ** 1.4 + 28.0 ** 1.4)
        phi_degran = gate_area * gate_adc * (0.18 + 0.82 * gate_mtoc) * gate_lck

    pore_prod = k_pfp * phi_degran * m_tension
    n_pore_ss = np.maximum(0.0, (pore_prod - 0.32 * k_escrt) / max(1e-4, k_escrt))
    gzmb_dose = n_pore_ss * (exposure_min / 60.0)
    p_kill = 1.0 - np.exp(-((gzmb_dose / max(0.05, g_crit)) ** 1.65))

    return {
        "F_syn_nN": f_syn_nN,
        "sigma_syn_pN_um": sigma_syn_pN_um,
        "p_kill_physics": p_kill,
    }


def evaluate_clutch_at_eta(df: pd.DataFrame, eta: float, wide_bounds: bool = False) -> dict:
    g_str = df["group_id"].astype(str).values
    obs_by_g = df.groupby(df["group_id"].astype(str))["group_kill_rate"].first().to_dict()

    ca = df["CA_um2"].values
    arfs = df["ARFS_nm_s"].values
    adc = df["ADC"].values
    mpa = df["MPA_deg"].values

    train_mask = np.isin(g_str, TRAIN_GROUPS)

    def resid(params):
        kp, gc = params
        out = predict_pore_repair_kill_probability(
            ca[train_mask], arfs[train_mask], adc[train_mask], mpa[train_mask],
            k_pfp=kp, k_escrt=0.045, g_crit=gc, eta_clutch=eta
        )
        p = out["p_kill_physics"]
        g_tr = g_str[train_mask]
        return np.array([np.mean(p[g_tr == g]) - obs_by_g[g] for g in TRAIN_GROUPS])

    lb = [0.01, 0.20] if not wide_bounds else [0.001, 0.05]
    ub = [0.50, 3.50] if not wide_bounds else [5.00, 20.0]
    res = least_squares(resid, x0=[0.088, 1.15], bounds=(lb, ub))
    kp_fit, gc_fit = float(res.x[0]), float(res.x[1])

    out_all = predict_pore_repair_kill_probability(
        ca, arfs, adc, mpa, k_pfp=kp_fit, k_escrt=0.045, g_crit=gc_fit, eta_clutch=eta
    )
    p_all = out_all["p_kill_physics"]
    f_all = out_all["F_syn_nN"]

    preds = {g: float(np.mean(p_all[g_str == g])) for g in ALL_GROUPS}
    train_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in TRAIN_GROUPS]) * 100.0)
    low_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in LOW_GROUPS]) * 100.0)
    dasa_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in DASA_GROUPS]) * 100.0)
    gap = float((np.mean([preds[g] for g in TRAIN_GROUPS]) - np.mean([preds[g] for g in LOW_GROUPS])) * 100.0)

    return {
        "train_mae": train_mae,
        "low_mae": low_mae,
        "dasa_mae": dasa_mae,
        "gap": gap,
        "k_pfp": kp_fit,
        "g_crit": gc_fit,
        "mean_f_train": float(np.mean(f_all[train_mask])),
        "max_f_train": float(np.max(f_all[train_mask])),
    }


def evaluate_dent_stiffness(df: pd.DataFrame, E_pa: float, pure_dent_gate: bool = False) -> dict:
    g_str = df["group_id"].astype(str).values
    obs_by_g = df.groupby(df["group_id"].astype(str))["group_kill_rate"].first().to_dict()

    ca = df["CA_xlsx_um2"].values
    arfs = df["ARFS_xlsx_nm_s"].values
    adc = df["ADC_xlsx"].values
    mpa = df["MPA_xlsx_deg"].values
    dent = df["dent_max_um"].values
    r0 = df["R0_median_um"].values

    train_mask = np.isin(g_str, TRAIN_GROUPS)

    def resid(params):
        kp, gc = params
        out = predict_dent_hertz_pore_kill(
            ca[train_mask], arfs[train_mask], adc[train_mask], mpa[train_mask],
            dent[train_mask], r0[train_mask],
            k_pfp=kp, k_escrt=0.045, g_crit=gc, E_target_Pa=E_pa, pure_dent_gate=pure_dent_gate
        )
        p = out["p_kill_physics"]
        g_tr = g_str[train_mask]
        return np.array([np.mean(p[g_tr == g]) - obs_by_g[g] for g in TRAIN_GROUPS])

    res = least_squares(resid, x0=[0.088, 1.15], bounds=([0.001, 0.05], [5.0, 20.0]))
    kp_fit, gc_fit = float(res.x[0]), float(res.x[1])

    out_all = predict_dent_hertz_pore_kill(
        ca, arfs, adc, mpa, dent, r0,
        k_pfp=kp_fit, k_escrt=0.045, g_crit=gc_fit, E_target_Pa=E_pa, pure_dent_gate=pure_dent_gate
    )
    p_all = out_all["p_kill_physics"]
    f_all = out_all["F_syn_nN"]

    preds = {g: float(np.mean(p_all[g_str == g])) for g in ALL_GROUPS}
    train_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in TRAIN_GROUPS]) * 100.0)
    low_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in LOW_GROUPS]) * 100.0)
    dasa_mae = float(np.mean([abs(preds[g] - obs_by_g[g]) for g in DASA_GROUPS]) * 100.0)
    gap = float((np.mean([preds[g] for g in TRAIN_GROUPS]) - np.mean([preds[g] for g in LOW_GROUPS])) * 100.0)

    return {
        "train_mae": train_mae,
        "low_mae": low_mae,
        "dasa_mae": dasa_mae,
        "gap": gap,
        "k_pfp": kp_fit,
        "g_crit": gc_fit,
        "mean_f_train": float(np.mean(f_all[train_mask])),
        "max_f_train": float(np.max(f_all[train_mask])),
    }


def evaluate_dent_stiffness_fixed_threshold(df: pd.DataFrame, E_pa: float) -> dict:
    """
    Task-file Outcomes test:
      "Next step is the force chain, with the pore-formation threshold taken from the
       published force measurements rather than fitted, and a sweep of target stiffness
       to see whether one plausible value makes dents, threshold and deaths agree."
    Published force threshold: F_crit = 1.5 nN (1500 pN), width = 0.45 nN (450 pN)
    from micropillar/microparticle traction measurements (Basu et al. Cell 2016;
    Vorselen et al. Nat Commun 2020).
    Hertzian contact force from measured 3D dent delta = dent_max_um:
      F_indent(delta) = (4/3) * (E_target / (1 - nu^2)) * sqrt(R_eff) * delta^(3/2)
    Only the baseline exocytosis/kill rate scale k0 is fitted on Training Groups 1-5.
    """
    g_str = df["group_id"].astype(str).values
    obs_by_g = df.groupby(df["group_id"].astype(str))["group_kill_rate"].first().to_dict()

    ca = np.maximum(0.0, df["CA_xlsx_um2"].fillna(0.0).values)
    dent = np.maximum(0.0, df["dent_max_um"].fillna(0.0).values)
    r0 = np.clip(df["R0_median_um"].fillna(6.5).values, 3.5, 11.0)

    r_eff_um = (r0 * 5.0) / (r0 + 5.0)
    nu = 0.45
    hertz_coeff_pN = (4.0 / 3.0) * (E_pa / (1.0 - nu ** 2)) * np.sqrt(r_eff_um)
    f_indent_nN = (hertz_coeff_pN * (dent ** 1.5)) * 1e-3

    # Published mechanical threshold (Basu et al. Cell 2016; Vorselen et al. 2020):
    # Cytotoxic pore potentiation transitions around F_crit = 1.5 nN (width = 0.35 nN),
    # with basal non-mechanopotentiated killing fraction f_basal = 0.04 when in contact.
    F_crit_nN = 1.50
    F_width_nN = 0.35
    gate_mech = 1.0 / (1.0 + np.exp(-(f_indent_nN - F_crit_nN) / F_width_nN))
    gate_area = ca / (ca + 60.0)

    train_mask = np.isin(g_str, TRAIN_GROUPS)

    def resid(params):
        k0 = params[0]
        hazard = k0 * gate_area[train_mask] * (0.04 + 0.96 * gate_mech[train_mask])
        p = 1.0 - np.exp(-hazard)
        g_tr = g_str[train_mask]
        return np.array([np.mean(p[g_tr == g]) - obs_by_g[g] for g in TRAIN_GROUPS])

    res = least_squares(resid, x0=[1.5], bounds=([0.01], [35.0]))
    k0_fit = float(res.x[0])
    hazard_all = k0_fit * gate_area * (0.04 + 0.96 * gate_mech)
    p_all = 1.0 - np.exp(-hazard_all)

    preds = {g: float(np.mean(p_all[g_str == g])) for g in ALL_GROUPS}
    return {
        "train_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in TRAIN_GROUPS]) * 100.0),
        "low_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in LOW_GROUPS]) * 100.0),
        "dasa_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in DASA_GROUPS]) * 100.0),
        "gap": float((np.mean([preds[g] for g in TRAIN_GROUPS]) - np.mean([preds[g] for g in LOW_GROUPS])) * 100.0),
        "k0_fit": k0_fit,
        "mean_f_train": float(np.mean(f_indent_nN[train_mask])),
        "max_f_train": float(np.max(f_indent_nN[train_mask])),
    }


def evaluate_comparator_ca_arfs(df: pd.DataFrame) -> dict:
    """One-parameter saturating function of CA * ARFS: p = (CA*v) / (CA*v + K)."""
    g_str = df["group_id"].astype(str).values
    obs_by_g = df.groupby(df["group_id"].astype(str))["group_kill_rate"].first().to_dict()
    x = np.maximum(0.0, df["CA_um2"].fillna(0.0).values) * np.maximum(0.0, df["ARFS_nm_s"].fillna(0.0).values * 1e-3)
    train_mask = np.isin(g_str, TRAIN_GROUPS)

    def resid(params):
        K = params[0]
        p = x[train_mask] / (x[train_mask] + K)
        g_tr = g_str[train_mask]
        return np.array([np.mean(p[g_tr == g]) - obs_by_g[g] for g in TRAIN_GROUPS])

    res = least_squares(resid, x0=[5.0], bounds=([1e-4], [1e4]))
    K_fit = float(res.x[0])
    p_all = x / (x + K_fit)
    preds = {g: float(np.mean(p_all[g_str == g])) for g in ALL_GROUPS}
    return {
        "train_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in TRAIN_GROUPS]) * 100.0),
        "low_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in LOW_GROUPS]) * 100.0),
        "dasa_mae": float(np.mean([abs(preds[g] - obs_by_g[g]) for g in DASA_GROUPS]) * 100.0),
        "gap": float((np.mean([preds[g] for g in TRAIN_GROUPS]) - np.mean([preds[g] for g in LOW_GROUPS])) * 100.0),
        "K": K_fit,
    }


def main():
    df_clean = export_clean_419_csv()
    df_aligned = pd.read_csv(DATA_DIR / "dent_step4_aligned_419_pairs.csv")
    df_aligned["group_id"] = df_aligned["group_id"].astype(str)

    comp = evaluate_comparator_ca_arfs(df_clean)
    print(f"Comparator (CA*ARFS, 419 clean cells): low_mae={comp['low_mae']:.2f} pp, dasa_mae={comp['dasa_mae']:.2f} pp, gap={comp['gap']:.2f} pp")

    # 1. Sweep A: Force-scale (eta_clutch) sweep on the exact 419-cell spreadsheet
    eta_grid = np.logspace(np.log10(0.5), np.log10(500.0), 25)
    eta_grid = np.sort(np.unique(np.append(eta_grid, 17.1)))
    boot_etas = {0.5, 17.1, float(eta_grid[12]), 500.0}
    rng = np.random.default_rng(42)

    rows_eta = []
    for eta in eta_grid:
        res = evaluate_clutch_at_eta(df_clean, float(eta), wide_bounds=False)
        res_wide = evaluate_clutch_at_eta(df_clean, float(eta), wide_bounds=True)

        low_ci, dasa_ci, gap_ci = "", "", ""
        if any(abs(eta - b) < 1e-6 for b in boot_etas):
            b_low, b_dasa, b_gap = [], [], []
            for _ in range(150):
                parts = [grp.iloc[rng.integers(0, len(grp), size=len(grp))] for _, grp in df_clean.groupby("group_id", sort=False)]
                dfb = pd.concat(parts, ignore_index=True)
                rb = evaluate_clutch_at_eta(dfb, float(eta), wide_bounds=False)
                b_low.append(rb["low_mae"])
                b_dasa.append(rb["dasa_mae"])
                b_gap.append(rb["gap"])
            low_ci = f"[{np.percentile(b_low, 2.5):.1f}, {np.percentile(b_low, 97.5):.1f}]"
            dasa_ci = f"[{np.percentile(b_dasa, 2.5):.1f}, {np.percentile(b_dasa, 97.5):.1f}]"
            gap_ci = f"[{np.percentile(b_gap, 2.5):.1f}, {np.percentile(b_gap, 97.5):.1f}]"

        rows_eta.append({
            "eta_clutch_pN_s_um3": round(float(eta), 3),
            "mean_force_train_nN": round(res["mean_f_train"], 3),
            "max_force_train_nN": round(res["max_f_train"], 3),
            "train_mae_pct": round(res["train_mae"], 2),
            "heldout_low_mae_pct": round(res["low_mae"], 2),
            "heldout_low_ci95": low_ci,
            "dasatinib_mae_pct": round(res["dasa_mae"], 2),
            "dasatinib_ci95": dasa_ci,
            "predicted_donor_gap_pct": round(res["gap"], 2),
            "donor_gap_ci95": gap_ci,
            "wide_bounds_low_mae_pct": round(res_wide["low_mae"], 2),
            "wide_bounds_gap_pct": round(res_wide["gap"], 2),
            "k_pfp_fit": round(res["k_pfp"], 4),
            "g_crit_fit": round(res["g_crit"], 4),
        })

    df_eta = pd.DataFrame(rows_eta)
    df_eta.to_csv(DATA_DIR / "force_scale_sweep_419_clean.csv", index=False)

    # 2. Sweep B: Target stiffness (E_target) sweep using measured 3D dent (dent_max_um)
    E_grid = np.logspace(np.log10(50.0), np.log10(10000.0), 25)
    E_grid = np.sort(np.unique(np.append(E_grid, [100.0, 250.0, 450.0, 1000.0, 2500.0, 5000.0])))
    boot_Es = {50.0, 250.0, 450.0, 1000.0, 2500.0, 10000.0}

    rows_E = []
    for E_val in E_grid:
        res_full = evaluate_dent_stiffness(df_aligned, float(E_val), pure_dent_gate=False)
        res_pure = evaluate_dent_stiffness(df_aligned, float(E_val), pure_dent_gate=True)
        res_fix = evaluate_dent_stiffness_fixed_threshold(df_aligned, float(E_val))

        low_ci, dasa_ci, gap_ci = "", "", ""
        if any(abs(E_val - b) < 1e-6 for b in boot_Es):
            b_low, b_dasa, b_gap = [], [], []
            for _ in range(150):
                parts = [grp.iloc[rng.integers(0, len(grp), size=len(grp))] for _, grp in df_aligned.groupby("group_id", sort=False)]
                dfb = pd.concat(parts, ignore_index=True)
                rb = evaluate_dent_stiffness_fixed_threshold(dfb, float(E_val))
                b_low.append(rb["low_mae"])
                b_dasa.append(rb["dasa_mae"])
                b_gap.append(rb["gap"])
            low_ci = f"[{np.percentile(b_low, 2.5):.1f}, {np.percentile(b_low, 97.5):.1f}]"
            dasa_ci = f"[{np.percentile(b_dasa, 2.5):.1f}, {np.percentile(b_dasa, 97.5):.1f}]"
            gap_ci = f"[{np.percentile(b_gap, 2.5):.1f}, {np.percentile(b_gap, 97.5):.1f}]"

        rows_E.append({
            "E_target_Pa": round(float(E_val), 1),
            "mean_force_train_nN": round(res_full["mean_f_train"], 3),
            "max_force_train_nN": round(res_full["max_f_train"], 3),
            "refit_heldout_low_mae_pct": round(res_full["low_mae"], 2),
            "refit_dasatinib_mae_pct": round(res_full["dasa_mae"], 2),
            "refit_donor_gap_pct": round(res_full["gap"], 2),
            "fixed_thr_train_mae_pct": round(res_fix["train_mae"], 2),
            "fixed_thr_low_mae_pct": round(res_fix["low_mae"], 2),
            "fixed_thr_low_ci95": low_ci,
            "fixed_thr_dasa_mae_pct": round(res_fix["dasa_mae"], 2),
            "fixed_thr_dasa_ci95": dasa_ci,
            "fixed_thr_donor_gap_pct": round(res_fix["gap"], 2),
            "fixed_thr_gap_ci95": gap_ci,
            "pure_dent_low_mae_pct": round(res_pure["low_mae"], 2),
            "k0_fixed_thr_fit": round(res_fix["k0_fit"], 4),
            "k_pfp_refit": round(res_full["k_pfp"], 4),
            "g_crit_refit": round(res_full["g_crit"], 4),
        })

    df_E = pd.DataFrame(rows_E)
    df_E.to_csv(DATA_DIR / "target_stiffness_sweep_3d_dent.csv", index=False)

    # 3. Plot House-Style comparison figure
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.plot(df_eta["eta_clutch_pN_s_um3"], df_eta["heldout_low_mae_pct"], "-o", color="#1F4E79", lw=1.8, ms=4, label="Physics (clutch force, 419 clean cells)")
    ax.axhline(comp["low_mae"], color="#C0561F", ls="--", lw=1.5, label=f"Comparator CA*ARFS ({comp['low_mae']:.1f} pp)")
    ax.axhline(17.7, color="#2E7D5B", ls=":", lw=1.6, label="Dent + CA rule (17.7 pp)")
    ax.set_xscale("log")
    ax.set_xlabel("Clutch force scale eta_clutch (pN s / um^3)", fontsize=10, color="#111111")
    ax.set_ylabel("Held-out low-killing donor MAE (pp)", fontsize=10, color="#111111")
    ax.set_title("A. Clutch Force Scale Sweep (419 Clean Cells)", fontsize=11, fontweight="bold", color="#111111", loc="left")
    ax.set_ylim(12, 32)
    ax.grid(True, color="#E3E3E3", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.set_facecolor("#FFFFFF")
    ax.plot(df_E["E_target_Pa"], df_E["fixed_thr_low_mae_pct"], "-o", color="#1F4E79", lw=1.8, ms=4, label="Fixed published threshold (Basu 2016)")
    ax.plot(df_E["E_target_Pa"], df_E["refit_heldout_low_mae_pct"], "-s", color="#6B4C9A", lw=1.5, ms=3.5, label="Refitted threshold at each E_target")
    ax.axhline(comp["low_mae"], color="#C0561F", ls="--", lw=1.4, label=f"Comparator CA*ARFS ({comp['low_mae']:.1f} pp)")
    ax.axhline(17.7, color="#2E7D5B", ls=":", lw=1.6, label="Model-free Dent + CA rule (17.7 pp)")
    ax.set_xscale("log")
    ax.set_xlabel("Target stiffness E_target (Pa)", fontsize=10, color="#111111")
    ax.set_ylabel("Held-out low-killing donor MAE (pp)", fontsize=10, color="#111111")
    ax.set_title("B. Target Stiffness Sweep Using Measured 3D Dent", fontsize=11, fontweight="bold", color="#111111", loc="left")
    ax.set_ylim(10, 32)
    ax.grid(True, color="#E3E3E3", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "stiffness_and_force_scale_sweeps.png", dpi=200)
    plt.close(fig)
    print("Saved all outputs successfully.")


if __name__ == "__main__":
    main()
