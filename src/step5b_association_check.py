#!/usr/bin/env python3
"""
Step 5b pre-registered association check: do contact descriptors relate to killing or donor?

Executes the pre-registered plan in status/step5b_preregistration.md without editing it.

Inputs:
- data/step5/per_chamber_descriptors.csv
- data/step5/per_visit_descriptors.csv
- data/step4/per_visit_death_labels_all_v2.csv
- data/step4/per_chamber_death_labels_all_v2.csv

Outputs:
- data/step5b/pre_death_chamber_summaries.csv
- data/step5b/outlier_flagged_chambers.csv
- data/step5b/test1_cox_death.csv
- data/step5b/test2_loeo_auc.csv
- data/step5b/test3_donor_association.csv
- data/step5b/test3_donor_medians.csv
- data/step5b/test4_latrunculin_control.csv
- data/step5b/exploratory_checks.csv
- data/step5b/summary_step5b.json
- figures/step5b_association_check.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, norm, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
STEP5_DIR = ROOT / "data" / "step5"
STEP4_DIR = ROOT / "data" / "step4"
OUT_DIR = ROOT / "data" / "step5b"
FIG_DIR = ROOT / "figures"

PRIMARY_DESCRIPTORS = [
    "cup_depth_hull_p95_um",
    "cup_depth_over_radius",
    "cup_volume_hull_um3",
    "tcell_protrusion_frac",
    "tcell_protrusion_vol_um3",
    "patch_equiv_radius_um",
    "contact_area_um2",
    "target_volume_um3",
    "target_sphericity",
    "target_elongation",
    "tcell_volume_um3",
    "tcell_sphericity",
]

SECONDARY_DESCRIPTORS = [
    "cup_asymmetry_norm",
    "zernike_tilt_amp_um",
    "zernike_astig_amp_um",
    "zernike_coma_amp_um",
    "zernike_defocus_um",
    "zernike_spherical_um",
    "rough_above_cutoff_um2",
    "patch_elongation",
]

ALL_DESCRIPTORS = PRIMARY_DESCRIPTORS + SECONDARY_DESCRIPTORS

DESCRIPTOR_LABELS = {
    "cup_depth_hull_p95_um": "Cup depth p95 (hull, um)",
    "cup_depth_over_radius": "Cup depth / patch radius",
    "cup_volume_hull_um3": "Cup volume (hull, um^3)",
    "tcell_protrusion_frac": "T-cell protrusion fraction",
    "tcell_protrusion_vol_um3": "T-cell protrusion vol (um^3)",
    "patch_equiv_radius_um": "Patch equiv. radius (um)",
    "contact_area_um2": "Contact area (um^2)",
    "target_volume_um3": "Target volume (um^3)",
    "target_sphericity": "Target sphericity",
    "target_elongation": "Target elongation",
    "tcell_volume_um3": "T-cell volume (um^3)",
    "tcell_sphericity": "T-cell sphericity",
    "cup_asymmetry_norm": "Cup apex asymmetry",
    "zernike_tilt_amp_um": "Zernike tilt amp (um)",
    "zernike_astig_amp_um": "Zernike astigmatism amp (um)",
    "zernike_coma_amp_um": "Zernike coma amp (um)",
    "zernike_defocus_um": "Zernike defocus (um)",
    "zernike_spherical_um": "Zernike spherical (um)",
    "rough_above_cutoff_um2": "Patch roughness >=1.4 um (um^2)",
    "patch_elongation": "Patch elongation",
}

CONDITION_HIGH = "Untreated high-killing CAR-T"
CONDITION_LOW = "Untreated low-killing CAR-T"
CONDITION_DRUG = "Dasatinib 50 nM CAR-T"

PAIRED_CONTROL_EXPERIMENTS = [
    ("hd2", "v08_6_40_120_0-6_hd2", "v11_6_36_120_0-6_hd2_ds"),
    ("hd10", "v09_6_40_120_0-6_hd10", "v12_6_35_120_0-6_hd10ds"),
    ("hd5", "v10_6_40_120_0-6_hd5", "v13_6_40_120_0-6_hd5_ds"),
]


def extract_donor(experiment: str) -> str:
    m = re.search(r"(hd\d+)", str(experiment))
    return m.group(1) if m else str(experiment)


def bh_fdr(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, handling NaNs gracefully."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1, dtype=float)
    raw_q = pv * m / ranks
    q_ord = np.minimum.accumulate(raw_q[order][::-1])[::-1]
    q_val = np.clip(q_ord, 0.0, 1.0)
    out_valid = np.empty(m, dtype=float)
    out_valid[order] = q_val
    q[valid] = out_valid
    return q


def normal_scores_within_experiment(
    df: pd.DataFrame, col: str, exp_col: str = "experiment", standardize: bool = True
) -> pd.Series:
    """Blom normal scores computed from within-experiment ranks, standardized to unit SD."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby(exp_col, sort=False):
        vals = grp[col].dropna()
        n = len(vals)
        if n <= 1:
            continue
        r = rankdata(vals.to_numpy(dtype=float), method="average")
        z = norm.ppf((r - 0.375) / (n + 0.25))
        if standardize:
            sd = np.std(z, ddof=1)
            if sd > 0:
                z = (z - np.mean(z)) / sd
        out.loc[vals.index] = z
    return out


def fit_stratified_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    cov_cols: list[str],
    strata_col: str = "experiment",
    ties: str = "efron",
) -> dict[str, Any]:
    """
    Exact Newton-Raphson solver for experiment-stratified Cox proportional hazards
    partial likelihood (supports Efron and Breslow tie handling).
    """
    sub = df[[time_col, event_col, strata_col] + cov_cols].dropna().copy()
    p = len(cov_cols)
    n_obs = len(sub)
    n_events = int(sub[event_col].sum()) if n_obs > 0 else 0

    strata_data = []
    for _, grp in sub.groupby(strata_col, sort=False):
        t = grp[time_col].to_numpy(dtype=float)
        d = grp[event_col].to_numpy(dtype=int)
        X = grp[cov_cols].to_numpy(dtype=float)
        if np.sum(d) == 0 or len(t) < 2:
            continue
        strata_data.append((t, d, X))

    if not strata_data or n_events == 0:
        return {
            "beta": np.full(p, np.nan),
            "se": np.full(p, np.nan),
            "z": np.full(p, np.nan),
            "p": np.full(p, np.nan),
            "hr": np.full(p, np.nan),
            "ci_low": np.full(p, np.nan),
            "ci_high": np.full(p, np.nan),
            "n_obs": n_obs,
            "n_events": n_events,
            "converged": False,
        }

    def nll_grad_hess(beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nll = 0.0
        grad = np.zeros(p, dtype=float)
        hess = np.zeros((p, p), dtype=float)
        for t, d, X in strata_data:
            eta = np.clip(X @ beta, -50.0, 50.0)
            eta_shift = eta - np.max(eta)
            w = np.exp(eta_shift)
            Xw = X * w[:, None]
            uniq_t = np.unique(t[d == 1])
            for ut in uniq_t:
                at_risk = t >= ut
                events = (t == ut) & (d == 1)
                m_ev = int(np.sum(events))
                S0 = float(np.sum(w[at_risk]))
                S1 = np.sum(Xw[at_risk], axis=0)
                S2 = Xw[at_risk].T @ X[at_risk]
                X_ev_sum = np.sum(X[events], axis=0)
                nll -= float(np.sum(eta_shift[events]))
                grad -= X_ev_sum
                if ties == "breslow" or m_ev == 1:
                    nll += m_ev * np.log(S0)
                    v = S1 / S0
                    grad += m_ev * v
                    hess += m_ev * (S2 / S0 - np.outer(v, v))
                else:
                    w_ev = float(np.sum(w[events]))
                    S1_ev = np.sum(Xw[events], axis=0)
                    S2_ev = Xw[events].T @ X[events]
                    for l_idx in range(m_ev):
                        frac = l_idx / m_ev
                        denom = max(S0 - frac * w_ev, 1e-15)
                        num1 = S1 - frac * S1_ev
                        num2 = S2 - frac * S2_ev
                        nll += np.log(denom)
                        v = num1 / denom
                        grad += v
                        hess += num2 / denom - np.outer(v, v)
        return nll, grad, hess

    beta = np.zeros(p, dtype=float)
    converged = False
    for _ in range(60):
        nll, grad, hess = nll_grad_hess(beta)
        try:
            step = np.linalg.solve(hess + np.eye(p) * 1e-10, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        # Halving step size if needed
        scale = 1.0
        while scale > 1e-4:
            beta_cand = beta - scale * step
            if np.max(np.abs(beta_cand)) < 25.0:
                nll_cand, _, _ = nll_grad_hess(beta_cand)
                if nll_cand <= nll + 1e-9:
                    break
            scale *= 0.5
        beta = beta - scale * step
        if np.max(np.abs(scale * step)) < 1e-8:
            converged = True
            break

    _, _, hess = nll_grad_hess(beta)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    diag = np.diag(cov)
    se = np.where(diag > 0, np.sqrt(np.maximum(diag, 0.0)), np.nan)
    z_stat = beta / se
    p_val = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))
    hr = np.exp(beta)
    ci_low = np.exp(beta - 1.96 * se)
    ci_high = np.exp(beta + 1.96 * se)

    return {
        "beta": beta,
        "se": se,
        "z": z_stat,
        "p": p_val,
        "hr": hr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": n_obs,
        "n_events": n_events,
        "converged": converged,
    }


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Load step4 v2 and step5a CSVs, construct pre-death chamber summaries (A) and (B)
    for both any-visit (primary) and sustained (>=2 visits, sensitivity) rules in the
    200-minute window (visits 1..20), and flag volume-outlier chambers.
    """
    ch_desc = pd.read_csv(STEP5_DIR / "per_chamber_descriptors.csv")
    vis_desc = pd.read_csv(STEP5_DIR / "per_visit_descriptors.csv", low_memory=False)
    ch_death = pd.read_csv(STEP4_DIR / "per_chamber_death_labels_all_v2.csv")
    vis_death = pd.read_csv(STEP4_DIR / "per_visit_death_labels_all_v2.csv", low_memory=False)

    ch_death["donor"] = ch_death["experiment"].apply(extract_donor)

    # Verify visit-level consistency between vis_desc and vis_death
    n_vis_desc = len(vis_desc)
    n_vis_death = len(vis_death)
    assert n_vis_desc == n_vis_death == 10510, f"Expected 10510 visits, got {n_vis_desc} vs {n_vis_death}"

    # Merge visit-level descriptors with chamber-level death timing (200-min window)
    death_cols = [
        "experiment",
        "roi",
        "donor",
        "condition",
        "group_id",
        "trace_num",
        "included_by_trace_rule",
        "non_single_track_flag",
        "first_contact_visit_200m",
        "first_contact_time_min_200m",
        "last_observed_visit_200m",
        "last_observed_time_min_200m",
        "no_contact_200m",
        "ever_dead_any_visit_200m",
        "first_any_dead_visit_200m",
        "first_any_dead_time_min_200m",
        "any_contact_to_death_min_200m",
        "censored_any_visit_200m",
        "km_any_contact_time_min_200m",
        "km_event_any_visit_200m",
        "ever_dead_200m",
        "death_visit_200m",
        "death_time_min_200m",
        "contact_to_death_min_200m",
        "censored_200m",
        "km_contact_time_min_200m",
        "km_event_sustained_200m",
    ]
    base_ch = ch_death[death_cols].copy()

    # Merge with per_chamber_descriptors for peak_ca and overall contact counts
    desc_meta_cols = [
        "experiment",
        "roi",
        "n_contact_visits",
        "first_contact_visit",
        "peak_ca_visit",
        "cup_volume_hull_um3_first_contact",
        "tcell_protrusion_vol_um3_first_contact",
        "target_volume_um3_first_contact",
        "cup_volume_hull_um3_mean_contact",
        "tcell_protrusion_vol_um3_mean_contact",
        "target_volume_um3_mean_contact",
        "cup_volume_hull_um3_peak_ca",
        "tcell_protrusion_vol_um3_peak_ca",
        "target_volume_um3_peak_ca",
    ]
    base_ch = base_ch.merge(
        ch_desc[desc_meta_cols].rename(
            columns={
                "n_contact_visits": "n_contact_visits_step5_full",
                "first_contact_visit": "first_contact_visit_step5_full",
            }
        ),
        on=["experiment", "roi"],
        how="left",
    )

    # Filter vis_desc to visits 1..20 with in_contact_visit == True
    mv = vis_desc.merge(
        base_ch[
            [
                "experiment",
                "roi",
                "first_any_dead_visit_200m",
                "death_visit_200m",
            ]
        ],
        on=["experiment", "roi"],
        how="left",
    )
    mv20 = mv[(mv["visit"] <= 20) & (mv["in_contact_visit"] == True)].copy()

    # First contact visit in step5 descriptors within visits 1..20
    fc_step5_200m = (
        mv20.groupby(["experiment", "roi"])["visit"]
        .min()
        .rename("first_contact_visit_desc_200m")
        .reset_index()
    )
    n_cnt_step5_200m = (
        mv20.groupby(["experiment", "roi"])["visit"]
        .count()
        .rename("n_contact_visits_desc_200m")
        .reset_index()
    )
    base_ch = base_ch.merge(fc_step5_200m, on=["experiment", "roi"], how="left")
    base_ch = base_ch.merge(n_cnt_step5_200m, on=["experiment", "roi"], how="left")
    base_ch["n_contact_visits_desc_200m"] = base_ch["n_contact_visits_desc_200m"].fillna(0).astype(int)
    base_ch["has_contact_desc_200m"] = base_ch["n_contact_visits_desc_200m"] > 0

    # Summary (A): value at first-contact visit (in 1..20)
    mv20 = mv20.merge(fc_step5_200m, on=["experiment", "roi"], how="left")
    vis_A = mv20[mv20["visit"] == mv20["first_contact_visit_desc_200m"]].copy()
    cols_A = {d: f"{d}_A" for d in ALL_DESCRIPTORS}
    df_A = vis_A[["experiment", "roi"] + ALL_DESCRIPTORS].rename(columns=cols_A)
    base_ch = base_ch.merge(df_A, on=["experiment", "roi"], how="left")

    # Flag whether first-contact visit is strictly before death visit
    base_ch["fc_is_death_visit_any_200m"] = (
        base_ch["has_contact_desc_200m"]
        & base_ch["first_any_dead_visit_200m"].notna()
        & (base_ch["first_contact_visit_desc_200m"] >= base_ch["first_any_dead_visit_200m"])
    )
    base_ch["fc_is_death_visit_sust_200m"] = (
        base_ch["has_contact_desc_200m"]
        & base_ch["death_visit_200m"].notna()
        & (base_ch["first_contact_visit_desc_200m"] >= base_ch["death_visit_200m"])
    )

    # Summary (B_any): mean over contact visits strictly before first_any_dead_visit_200m (visit <= 20)
    mv20_pre_any = mv20[
        mv20["first_any_dead_visit_200m"].isna()
        | (mv20["visit"] < mv20["first_any_dead_visit_200m"])
    ].copy()
    cnt_pre_any = (
        mv20_pre_any.groupby(["experiment", "roi"])["visit"]
        .count()
        .rename("n_pre_death_contact_visits_any_200m")
        .reset_index()
    )
    cols_B_any = {d: f"{d}_B_any" for d in ALL_DESCRIPTORS}
    df_B_any = (
        mv20_pre_any.groupby(["experiment", "roi"])[ALL_DESCRIPTORS]
        .mean()
        .rename(columns=cols_B_any)
        .reset_index()
    )
    base_ch = base_ch.merge(cnt_pre_any, on=["experiment", "roi"], how="left")
    base_ch["n_pre_death_contact_visits_any_200m"] = (
        base_ch["n_pre_death_contact_visits_any_200m"].fillna(0).astype(int)
    )
    base_ch = base_ch.merge(df_B_any, on=["experiment", "roi"], how="left")

    # Summary (B_sust): mean over contact visits strictly before sustained death_visit_200m (visit <= 20)
    mv20_pre_sust = mv20[
        mv20["death_visit_200m"].isna() | (mv20["visit"] < mv20["death_visit_200m"])
    ].copy()
    cnt_pre_sust = (
        mv20_pre_sust.groupby(["experiment", "roi"])["visit"]
        .count()
        .rename("n_pre_death_contact_visits_sust_200m")
        .reset_index()
    )
    cols_B_sust = {d: f"{d}_B_sust" for d in ALL_DESCRIPTORS}
    df_B_sust = (
        mv20_pre_sust.groupby(["experiment", "roi"])[ALL_DESCRIPTORS]
        .mean()
        .rename(columns=cols_B_sust)
        .reset_index()
    )
    base_ch = base_ch.merge(cnt_pre_sust, on=["experiment", "roi"], how="left")
    base_ch["n_pre_death_contact_visits_sust_200m"] = (
        base_ch["n_pre_death_contact_visits_sust_200m"].fillna(0).astype(int)
    )
    base_ch = base_ch.merge(df_B_sust, on=["experiment", "roi"], how="left")

    # Outlier detection: "chambers where a cup or protrusion volume exceeds the target's own volume"
    mv_contact = mv[mv["in_contact_visit"] == True].copy()
    mv_contact["out_cup_gt_target"] = (
        mv_contact["cup_volume_hull_um3"] > mv_contact["target_volume_um3"]
    ) | (mv_contact["cup_volume_sphere_um3"] > mv_contact["target_volume_um3"])
    mv_contact["out_prot_gt_target"] = (
        mv_contact["tcell_protrusion_vol_um3"] > mv_contact["target_volume_um3"]
    )
    mv_contact["out_vol_gt_target"] = (
        mv_contact["out_cup_gt_target"] | mv_contact["out_prot_gt_target"]
    )
    mv_contact["is_pre_death_200m"] = (mv_contact["visit"] <= 20) & (
        mv_contact["first_any_dead_visit_200m"].isna()
        | (mv_contact["visit"] < mv_contact["first_any_dead_visit_200m"])
    )

    out_vis_rows = mv_contact[mv_contact["out_vol_gt_target"]].copy()
    out_ch_agg = (
        mv_contact.groupby(["experiment", "roi"])
        .agg(
            outlier_any_visit=("out_vol_gt_target", "any"),
            n_outlier_visits=("out_vol_gt_target", "sum"),
            outlier_pre_death_visit_200m=(
                "out_vol_gt_target",
                lambda s: bool((s & mv_contact.loc[s.index, "is_pre_death_200m"]).any()),
            ),
        )
        .reset_index()
    )
    base_ch = base_ch.merge(out_ch_agg, on=["experiment", "roi"], how="left")
    base_ch["outlier_any_visit"] = base_ch["outlier_any_visit"].where(base_ch["outlier_any_visit"].notna(), False).astype(bool)
    base_ch["n_outlier_visits"] = base_ch["n_outlier_visits"].fillna(0).astype(int)
    base_ch["outlier_pre_death_visit_200m"] = base_ch["outlier_pre_death_visit_200m"].where(base_ch["outlier_pre_death_visit_200m"].notna(), False).astype(bool)

    base_ch["outlier_summary_A"] = (
        (base_ch["cup_volume_hull_um3_A"] > base_ch["target_volume_um3_A"])
        | (base_ch["tcell_protrusion_vol_um3_A"] > base_ch["target_volume_um3_A"])
    ).fillna(False)
    base_ch["outlier_summary_B_any"] = (
        (base_ch["cup_volume_hull_um3_B_any"] > base_ch["target_volume_um3_B_any"])
        | (base_ch["tcell_protrusion_vol_um3_B_any"] > base_ch["target_volume_um3_B_any"])
    ).fillna(False)
    base_ch["outlier_chamber_summary"] = (
        base_ch["outlier_summary_A"]
        | base_ch["outlier_summary_B_any"]
        | (base_ch["cup_volume_hull_um3_mean_contact"] > base_ch["target_volume_um3_mean_contact"])
        | (base_ch["tcell_protrusion_vol_um3_mean_contact"] > base_ch["target_volume_um3_mean_contact"])
        | (base_ch["cup_volume_hull_um3_peak_ca"] > base_ch["target_volume_um3_peak_ca"])
        | (base_ch["tcell_protrusion_vol_um3_peak_ca"] > base_ch["target_volume_um3_peak_ca"])
    ).fillna(False)

    # Combined flag: flagged if any contact visit OR any chamber summary has cup/protrusion vol > target vol
    base_ch["outlier_vol_flag"] = base_ch["outlier_any_visit"] | base_ch["outlier_chamber_summary"]

    outlier_table = base_ch[base_ch["outlier_vol_flag"]][
        [
            "experiment",
            "roi",
            "donor",
            "condition",
            "trace_num",
            "first_contact_visit_desc_200m",
            "first_any_dead_visit_200m",
            "ever_dead_any_visit_200m",
            "n_outlier_visits",
            "outlier_pre_death_visit_200m",
            "outlier_summary_A",
            "outlier_summary_B_any",
            "outlier_chamber_summary",
            "cup_volume_hull_um3_A",
            "tcell_protrusion_vol_um3_A",
            "target_volume_um3_A",
            "cup_volume_hull_um3_B_any",
            "tcell_protrusion_vol_um3_B_any",
            "target_volume_um3_B_any",
            "cup_volume_hull_um3_peak_ca",
            "tcell_protrusion_vol_um3_peak_ca",
            "target_volume_um3_peak_ca",
        ]
    ].copy()

    # Add exact visit list where outlier occurred
    vis_list_map = (
        out_vis_rows.groupby(["experiment", "roi"])["visit"]
        .apply(lambda s: ",".join(str(int(v)) for v in sorted(s)))
        .rename("outlier_visits_list")
        .reset_index()
    )
    outlier_table = outlier_table.merge(vis_list_map, on=["experiment", "roi"], how="left")

    meta_counts = {
        "n_total_chambers": int(len(base_ch)),
        "n_trace1_chambers": int((base_ch["trace_num"] == 1).sum()),
        "n_trace1_contact_full": int(((base_ch["trace_num"] == 1) & (base_ch["n_contact_visits_step5_full"] > 0)).sum()),
        "n_trace1_contact_200m": int(((base_ch["trace_num"] == 1) & base_ch["has_contact_desc_200m"]).sum()),
        "n_trace1_dropped_first_contact_is_death_any_200m": int(
            ((base_ch["trace_num"] == 1) & base_ch["has_contact_desc_200m"] & base_ch["fc_is_death_visit_any_200m"]).sum()
        ),
        "n_trace1_pre_death_B_any_200m": int(
            ((base_ch["trace_num"] == 1) & (base_ch["n_pre_death_contact_visits_any_200m"] > 0)).sum()
        ),
        "n_trace1_dropped_first_contact_is_death_sust_200m": int(
            ((base_ch["trace_num"] == 1) & base_ch["has_contact_desc_200m"] & base_ch["fc_is_death_visit_sust_200m"]).sum()
        ),
        "n_trace1_pre_death_B_sust_200m": int(
            ((base_ch["trace_num"] == 1) & (base_ch["n_pre_death_contact_visits_sust_200m"] > 0)).sum()
        ),
        "n_all_contact_full": int((base_ch["n_contact_visits_step5_full"] > 0).sum()),
        "n_all_contact_200m": int(base_ch["has_contact_desc_200m"].sum()),
        "n_all_dropped_first_contact_is_death_any_200m": int(
            (base_ch["has_contact_desc_200m"] & base_ch["fc_is_death_visit_any_200m"]).sum()
        ),
        "n_all_pre_death_B_any_200m": int((base_ch["n_pre_death_contact_visits_any_200m"] > 0).sum()),
        "n_outlier_flagged_chambers_total": int(base_ch["outlier_vol_flag"].sum()),
        "n_outlier_flagged_chambers_pre_death_200m": int(base_ch["outlier_pre_death_visit_200m"].sum()),
        "n_outlier_flagged_chambers_summary_A": int(base_ch["outlier_summary_A"].sum()),
        "n_outlier_flagged_chambers_summary_B_any": int(base_ch["outlier_summary_B_any"].sum()),
    }

    return base_ch, outlier_table, mv, meta_counts


def run_test1_cox(base_ch: pd.DataFrame) -> pd.DataFrame:
    """
    Test 1: Experiment-stratified Cox model of time from first contact to death,
    censored at 200 min, normal-score rank-transformed within experiment.
    Evaluated across:
    - Summaries: B (mean pre-death contact visits), A_strict (first contact strictly before death),
      A_all (all first-contact visits in 200m).
    - Cohorts: primary (trace_num == 1), no_outliers (trace_num == 1 & ~outlier_vol_flag),
      all_contacted (all chambers with contact), sustained_rule (trace_num == 1, sustained >=2 visits).
    """
    configs = [
        ("B_pre_death_mean", "trace1_primary", "any_visit", False),
        ("A_first_contact_pre_death", "trace1_primary", "any_visit", False),
        ("A_first_contact_all", "trace1_primary", "any_visit", False),
        ("B_pre_death_mean", "trace1_no_outliers", "any_visit", True),
        ("A_first_contact_pre_death", "trace1_no_outliers", "any_visit", True),
        ("A_first_contact_all", "trace1_no_outliers", "any_visit", True),
        ("B_pre_death_mean", "all_contacted_sensitivity", "any_visit", False),
        ("A_first_contact_pre_death", "all_contacted_sensitivity", "any_visit", False),
        ("B_pre_death_mean", "trace1_sustained_sensitivity", "sustained", False),
        ("A_first_contact_pre_death", "trace1_sustained_sensitivity", "sustained", False),
    ]

    all_rows = []
    for summary_name, cohort_name, outcome_rule, drop_outliers in configs:
        df = base_ch[base_ch["has_contact_desc_200m"]].copy()
        if cohort_name.startswith("trace1"):
            df = df[df["trace_num"] == 1].copy()
        if drop_outliers:
            df = df[~df["outlier_vol_flag"]].copy()

        if outcome_rule == "any_visit":
            time_col = "km_any_contact_time_min_200m"
            event_col = "km_event_any_visit_200m"
            if summary_name == "B_pre_death_mean":
                df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
                suffix = "_B_any"
            elif summary_name == "A_first_contact_pre_death":
                df = df[~df["fc_is_death_visit_any_200m"]].copy()
                suffix = "_A"
            else:
                suffix = "_A"
        else:
            time_col = "km_contact_time_min_200m"
            event_col = "km_event_sustained_200m"
            if summary_name == "B_pre_death_mean":
                df = df[df["n_pre_death_contact_visits_sust_200m"] > 0].copy()
                suffix = "_B_sust"
            else:
                df = df[~df["fc_is_death_visit_sust_200m"]].copy()
                suffix = "_A"

        ca_col = f"contact_area_um2{suffix}"
        df["z_ca"] = normal_scores_within_experiment(df, ca_col)

        for family_name, desc_list in [("primary", PRIMARY_DESCRIPTORS), ("secondary", SECONDARY_DESCRIPTORS)]:
            fam_rows = []
            for d in desc_list:
                col = f"{d}{suffix}"
                df["z_d"] = normal_scores_within_experiment(df, col)
                res_u = fit_stratified_cox(df, time_col, event_col, ["z_d"])

                # Direction in each condition block
                cond_betas = {}
                cond_hrs = {}
                cond_ps = {}
                for c_key, c_val in [
                    ("high", CONDITION_HIGH),
                    ("low", CONDITION_LOW),
                    ("lat", CONDITION_DRUG),
                ]:
                    res_c = fit_stratified_cox(
                        df[df["condition"] == c_val], time_col, event_col, ["z_d"]
                    )
                    cond_betas[c_key] = float(res_c["beta"][0])
                    cond_hrs[c_key] = float(res_c["hr"][0])
                    cond_ps[c_key] = float(res_c["p"][0])

                overall_sign = np.sign(res_u["beta"][0]) if not np.isnan(res_u["beta"][0]) else 0.0
                n_same_dir = sum(
                    1
                    for c_key in ["high", "low", "lat"]
                    if not np.isnan(cond_betas[c_key])
                    and cond_betas[c_key] != 0.0
                    and np.sign(cond_betas[c_key]) == overall_sign
                )

                # Adjusted for contact area
                if d != "contact_area_um2":
                    res_a = fit_stratified_cox(df, time_col, event_col, ["z_d", "z_ca"])
                    beta_a = float(res_a["beta"][0])
                    se_a = float(res_a["se"][0])
                    hr_a = float(res_a["hr"][0])
                    ci_lo_a = float(res_a["ci_low"][0])
                    ci_hi_a = float(res_a["ci_high"][0])
                    p_a = float(res_a["p"][0])
                else:
                    beta_a = se_a = hr_a = ci_lo_a = ci_hi_a = p_a = np.nan

                fam_rows.append(
                    {
                        "summary": summary_name,
                        "cohort": cohort_name,
                        "outcome_rule": outcome_rule,
                        "family": family_name,
                        "descriptor": d,
                        "label": DESCRIPTOR_LABELS.get(d, d),
                        "n_chambers": int(res_u["n_obs"]),
                        "n_events": int(res_u["n_events"]),
                        "beta_unadj": float(res_u["beta"][0]),
                        "se_unadj": float(res_u["se"][0]),
                        "hr_unadj": float(res_u["hr"][0]),
                        "ci_low_unadj": float(res_u["ci_low"][0]),
                        "ci_high_unadj": float(res_u["ci_high"][0]),
                        "p_unadj": float(res_u["p"][0]),
                        "beta_high_killing": cond_betas["high"],
                        "hr_high_killing": cond_hrs["high"],
                        "p_high_killing": cond_ps["high"],
                        "beta_low_killing": cond_betas["low"],
                        "hr_low_killing": cond_hrs["low"],
                        "p_low_killing": cond_ps["low"],
                        "beta_latrunculin": cond_betas["lat"],
                        "hr_latrunculin": cond_hrs["lat"],
                        "p_latrunculin": cond_ps["lat"],
                        "n_blocks_same_dir_as_overall": int(n_same_dir),
                        "dir_consistent_ge2_blocks": bool(n_same_dir >= 2),
                        "beta_adj_ca": beta_a,
                        "se_adj_ca": se_a,
                        "hr_adj_ca": hr_a,
                        "ci_low_adj_ca": ci_lo_a,
                        "ci_high_adj_ca": ci_hi_a,
                        "p_adj_ca": p_a,
                    }
                )

            q_u = bh_fdr([r["p_unadj"] for r in fam_rows])
            q_a = bh_fdr([r["p_adj_ca"] for r in fam_rows])
            for idx, r in enumerate(fam_rows):
                r["bh_q_unadj"] = float(q_u[idx])
                r["bh_q_adj_ca"] = float(q_a[idx]) if not np.isnan(q_a[idx]) else np.nan
                r["passes_test1_gate"] = bool(
                    (family_name == "primary")
                    and (r["bh_q_unadj"] < 0.05)
                    and r["dir_consistent_ge2_blocks"]
                )
                all_rows.append(r)

    return pd.DataFrame(all_rows)


def _rank_transform_fold(train_s: pd.Series, test_s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Fold-safe normal-score transform fit on training fold and applied to held-out fold."""
    tr_v = train_s.to_numpy(dtype=float)
    te_v = test_s.to_numpy(dtype=float)
    n_tr = len(tr_v)
    r_tr = rankdata(tr_v, method="average")
    z_tr = norm.ppf((r_tr - 0.375) / (n_tr + 0.25))
    sd = np.std(z_tr, ddof=1)
    mean = np.mean(z_tr)
    if sd > 0:
        z_tr = (z_tr - mean) / sd

    sorted_tr = np.sort(tr_v)
    left = np.searchsorted(sorted_tr, te_v, side="left")
    right = np.searchsorted(sorted_tr, te_v, side="right")
    r_te = left + 0.5 * (right - left) + 0.5
    r_te = np.clip(r_te, 0.5, n_tr + 0.5)
    z_te = norm.ppf((r_te - 0.375) / (n_tr + 0.25))
    if sd > 0:
        z_te = (z_te - mean) / sd
    return z_tr, z_te


def run_test2_loeo_auc(base_ch: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Test 2: Logistic model of death by 200 min, leave-one-experiment-out.
    Baseline: contact area (A) + condition.
    Added: primary family.
    Report held-out AUC with bootstrap CI for baseline, baseline + descriptors, and gain.
    """
    configs = [
        ("A_first_contact_pre_death", "trace1_primary", "any_visit", False, "within_exp_rank"),
        ("B_pre_death_mean", "trace1_primary", "any_visit", False, "within_exp_rank"),
        ("A_first_contact_all", "trace1_primary", "any_visit", False, "within_exp_rank"),
        ("A_first_contact_pre_death", "trace1_primary", "any_visit", False, "fold_pooled_rank"),
        ("B_pre_death_mean", "trace1_primary", "any_visit", False, "fold_pooled_rank"),
        ("A_first_contact_pre_death", "trace1_primary", "any_visit", False, "raw_zscore"),
        ("B_pre_death_mean", "trace1_primary", "any_visit", False, "raw_zscore"),
        ("A_first_contact_pre_death", "trace1_no_outliers", "any_visit", True, "within_exp_rank"),
        ("B_pre_death_mean", "trace1_no_outliers", "any_visit", True, "within_exp_rank"),
        ("A_first_contact_pre_death", "all_contacted_sensitivity", "any_visit", False, "within_exp_rank"),
        ("B_pre_death_mean", "all_contacted_sensitivity", "any_visit", False, "within_exp_rank"),
        ("A_first_contact_pre_death", "trace1_sustained_sensitivity", "sustained", False, "within_exp_rank"),
        ("B_pre_death_mean", "trace1_sustained_sensitivity", "sustained", False, "within_exp_rank"),
    ]

    rows = []
    roc_curves_for_plot = {}

    for summary_name, cohort_name, outcome_rule, drop_outliers, transform_mode in configs:
        df = base_ch[base_ch["has_contact_desc_200m"]].copy()
        if cohort_name.startswith("trace1"):
            df = df[df["trace_num"] == 1].copy()
        if drop_outliers:
            df = df[~df["outlier_vol_flag"]].copy()

        if outcome_rule == "any_visit":
            y_col = "ever_dead_any_visit_200m"
            if summary_name == "B_pre_death_mean":
                df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
                suffix = "_B_any"
            elif summary_name == "A_first_contact_pre_death":
                df = df[~df["fc_is_death_visit_any_200m"]].copy()
                suffix = "_A"
            else:
                suffix = "_A"
        else:
            y_col = "ever_dead_200m"
            if summary_name == "B_pre_death_mean":
                df = df[df["n_pre_death_contact_visits_sust_200m"] > 0].copy()
                suffix = "_B_sust"
            else:
                df = df[~df["fc_is_death_visit_sust_200m"]].copy()
                suffix = "_A"

        ca_A_col = "contact_area_um2_A"
        desc_cols = [f"{d}{suffix}" for d in PRIMARY_DESCRIPTORS]
        req_cols = list(dict.fromkeys([ca_A_col, "condition", y_col] + desc_cols))
        sub = df.dropna(subset=req_cols).copy()
        sub["y"] = sub[y_col].astype(int)

        cond_dummies = pd.get_dummies(sub["condition"], drop_first=True, dtype=float)
        cond_cols = list(cond_dummies.columns)
        for c in cond_cols:
            sub[c] = cond_dummies[c]

        cont_cols = list(dict.fromkeys([ca_A_col] + desc_cols))
        if transform_mode == "within_exp_rank":
            for col in cont_cols:
                sub[col] = normal_scores_within_experiment(sub, col)

        base_feats = [ca_A_col] + cond_cols
        plus_feats = list(dict.fromkeys(base_feats + desc_cols))

        p_base = np.zeros(len(sub), dtype=float)
        p_plus = np.zeros(len(sub), dtype=float)

        for exp in sub["experiment"].unique():
            tr_mask = (sub["experiment"] != exp).to_numpy()
            te_mask = (sub["experiment"] == exp).to_numpy()

            Xb_tr_df = sub.loc[tr_mask, base_feats].copy()
            Xb_te_df = sub.loc[te_mask, base_feats].copy()
            Xp_tr_df = sub.loc[tr_mask, plus_feats].copy()
            Xp_te_df = sub.loc[te_mask, plus_feats].copy()

            if transform_mode == "fold_pooled_rank":
                for col in cont_cols:
                    z_tr, z_te = _rank_transform_fold(
                        sub.loc[tr_mask, col], sub.loc[te_mask, col]
                    )
                    if col in Xb_tr_df.columns:
                        Xb_tr_df[col] = z_tr
                        Xb_te_df[col] = z_te
                    if col in Xp_tr_df.columns:
                        Xp_tr_df[col] = z_tr
                        Xp_te_df[col] = z_te
                Xb_tr, Xb_te = Xb_tr_df.to_numpy(), Xb_te_df.to_numpy()
                Xp_tr, Xp_te = Xp_tr_df.to_numpy(), Xp_te_df.to_numpy()
            else:
                sc_b = StandardScaler()
                Xb_tr = sc_b.fit_transform(Xb_tr_df)
                Xb_te = sc_b.transform(Xb_te_df)
                sc_p = StandardScaler()
                Xp_tr = sc_p.fit_transform(Xp_tr_df)
                Xp_te = sc_p.transform(Xp_te_df)

            y_tr = sub.loc[tr_mask, "y"].to_numpy()
            clf_b = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf_b.fit(Xb_tr, y_tr)
            p_base[te_mask] = clf_b.predict_proba(Xb_te)[:, 1]

            clf_p = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf_p.fit(Xp_tr, y_tr)
            p_plus[te_mask] = clf_p.predict_proba(Xp_te)[:, 1]

        y_all = sub["y"].to_numpy()
        auc_base = float(roc_auc_score(y_all, p_base))
        auc_plus = float(roc_auc_score(y_all, p_plus))
        auc_gain = auc_plus - auc_base

        rng = np.random.default_rng(seed)
        boot_base, boot_plus, boot_gain = [], [], []
        n_obs = len(y_all)
        for _ in range(n_boot):
            idx = rng.integers(0, n_obs, size=n_obs)
            if len(np.unique(y_all[idx])) < 2:
                continue
            ab = roc_auc_score(y_all[idx], p_base[idx])
            ap = roc_auc_score(y_all[idx], p_plus[idx])
            boot_base.append(ab)
            boot_plus.append(ap)
            boot_gain.append(ap - ab)

        ci_base = np.percentile(boot_base, [2.5, 97.5])
        ci_plus = np.percentile(boot_plus, [2.5, 97.5])
        ci_gain = np.percentile(boot_gain, [2.5, 97.5])

        passes_gate = bool((auc_gain >= 0.03) and (ci_gain[0] > 0.0))

        row = {
            "summary": summary_name,
            "cohort": cohort_name,
            "outcome_rule": outcome_rule,
            "transform_mode": transform_mode,
            "n_chambers": int(n_obs),
            "n_events": int(np.sum(y_all)),
            "auc_baseline": auc_base,
            "auc_baseline_ci_low": float(ci_base[0]),
            "auc_baseline_ci_high": float(ci_base[1]),
            "auc_plus_primary": auc_plus,
            "auc_plus_ci_low": float(ci_plus[0]),
            "auc_plus_ci_high": float(ci_plus[1]),
            "auc_gain": float(auc_gain),
            "auc_gain_ci_low": float(ci_gain[0]),
            "auc_gain_ci_high": float(ci_gain[1]),
            "passes_test2_gate": passes_gate,
        }
        rows.append(row)

        key = f"{summary_name}__{cohort_name}__{transform_mode}"
        roc_curves_for_plot[key] = {
            "y": y_all,
            "p_base": p_base,
            "p_plus": p_plus,
            "row": row,
        }

    return pd.DataFrame(rows), roc_curves_for_plot


def run_test3_donor(base_ch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test 3: Donor association across the 10 untreated donors (5 high-killing, 5 low-killing).
    - Per-donor median of each descriptor
    - (a) Exact Mann-Whitney U test: high- vs low-killing donors (n = 5 vs 5)
    - (b) Spearman correlation of donor median with donor kill rate (n = 10)
    - Chamber-level Kruskal-Wallis across the 10 donors (donor-or-batch variation)
    """
    configs = [
        ("A_first_contact_pre_death", "trace1_primary", False),
        ("B_pre_death_mean", "trace1_primary", False),
        ("A_first_contact_all", "trace1_primary", False),
        ("A_first_contact_pre_death", "trace1_no_outliers", True),
        ("B_pre_death_mean", "trace1_no_outliers", True),
        ("A_first_contact_pre_death", "all_contacted_sensitivity", False),
        ("B_pre_death_mean", "all_contacted_sensitivity", False),
    ]

    all_rows = []
    donor_median_tables = []

    for summary_name, cohort_name, drop_outliers in configs:
        ch_pool = base_ch[base_ch["condition"] != CONDITION_DRUG].copy()
        if cohort_name.startswith("trace1"):
            ch_pool = ch_pool[ch_pool["trace_num"] == 1].copy()
        if drop_outliers:
            ch_pool = ch_pool[~ch_pool["outlier_vol_flag"]].copy()

        # Donor kill rate in 200-min window (computed both over all chambers in cohort and over contacted chambers)
        donor_kr = (
            ch_pool.groupby(["experiment", "donor", "condition", "group_id"])
            .agg(
                n_chambers_total=("roi", "count"),
                n_contacted_200m=("has_contact_desc_200m", "sum"),
                n_kills_any_200m=("ever_dead_any_visit_200m", "sum"),
                kill_rate_any_200m=("ever_dead_any_visit_200m", "mean"),
                kill_rate_contacted_any_200m=(
                    "ever_dead_any_visit_200m",
                    lambda s: float(s[ch_pool.loc[s.index, "has_contact_desc_200m"]].mean()),
                ),
            )
            .reset_index()
        )

        df = ch_pool[ch_pool["has_contact_desc_200m"]].copy()
        if summary_name == "B_pre_death_mean":
            df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
            suffix = "_B_any"
        elif summary_name == "A_first_contact_pre_death":
            df = df[~df["fc_is_death_visit_any_200m"]].copy()
            suffix = "_A"
        else:
            suffix = "_A"

        desc_cols = [f"{d}{suffix}" for d in ALL_DESCRIPTORS]
        meds = (
            df.groupby(["experiment", "donor", "condition", "group_id"])[desc_cols]
            .median()
            .reset_index()
        )
        meds = meds.rename(columns={f"{d}{suffix}": d for d in ALL_DESCRIPTORS})
        meds = donor_kr.merge(meds, on=["experiment", "donor", "condition", "group_id"], how="left")
        meds["summary"] = summary_name
        meds["cohort"] = cohort_name
        donor_median_tables.append(meds)

        for family_name, desc_list in [("primary", PRIMARY_DESCRIPTORS), ("secondary", SECONDARY_DESCRIPTORS)]:
            fam_rows = []
            for d in desc_list:
                col = f"{d}{suffix}"
                hi_vals = meds.loc[meds["condition"] == CONDITION_HIGH, d].dropna().to_numpy(dtype=float)
                lo_vals = meds.loc[meds["condition"] == CONDITION_LOW, d].dropna().to_numpy(dtype=float)

                u_stat, p_mw = mannwhitneyu(hi_vals, lo_vals, alternative="two-sided", method="exact")
                rho_all, p_sp_all = spearmanr(meds[d], meds["kill_rate_any_200m"])
                rho_cnt, p_sp_cnt = spearmanr(meds[d], meds["kill_rate_contacted_any_200m"])

                groups = [
                    grp[col].dropna().to_numpy(dtype=float)
                    for _, grp in df.groupby("donor", sort=False)
                    if len(grp[col].dropna()) > 0
                ]
                kw_stat, p_kw = kruskal(*groups)

                fam_rows.append(
                    {
                        "summary": summary_name,
                        "cohort": cohort_name,
                        "family": family_name,
                        "descriptor": d,
                        "label": DESCRIPTOR_LABELS.get(d, d),
                        "n_donors_high": int(len(hi_vals)),
                        "n_donors_low": int(len(lo_vals)),
                        "median_high_killing_donors": float(np.median(hi_vals)),
                        "iqr_high_killing_donors": float(np.percentile(hi_vals, 75) - np.percentile(hi_vals, 25)),
                        "median_low_killing_donors": float(np.median(lo_vals)),
                        "iqr_low_killing_donors": float(np.percentile(lo_vals, 75) - np.percentile(lo_vals, 25)),
                        "mw_u_stat": float(u_stat),
                        "mw_p_two_sided": float(p_mw),
                        "spearman_rho_kill_rate": float(rho_all),
                        "spearman_p_kill_rate": float(p_sp_all),
                        "spearman_rho_contacted_kill_rate": float(rho_cnt),
                        "spearman_p_contacted_kill_rate": float(p_sp_cnt),
                        "kruskal_wallis_h_chamber": float(kw_stat),
                        "kruskal_wallis_p_chamber": float(p_kw),
                        "passes_test3_gate": bool(
                            (family_name == "primary")
                            and (p_mw < 0.01)
                            and (abs(rho_all) >= 0.6)
                        ),
                    }
                )

            q_mw = bh_fdr([r["mw_p_two_sided"] for r in fam_rows])
            q_sp = bh_fdr([r["spearman_p_kill_rate"] for r in fam_rows])
            q_kw = bh_fdr([r["kruskal_wallis_p_chamber"] for r in fam_rows])
            for idx, r in enumerate(fam_rows):
                r["bh_q_mw"] = float(q_mw[idx])
                r["bh_q_spearman"] = float(q_sp[idx])
                r["bh_q_kruskal_wallis"] = float(q_kw[idx])
                all_rows.append(r)

    return pd.DataFrame(all_rows), pd.concat(donor_median_tables, ignore_index=True)


def run_test4_latrunculin(base_ch: pd.DataFrame) -> pd.DataFrame:
    """
    Test 4: Positive control (latrunculin / drug-treated pairs: v08 vs v11 hd2, v09 vs v12 hd10, v10 vs v13 hd5).
    Checks whether drug lowers cup_depth_hull_p95_um or tcell_protrusion_frac in 3/3 paired donors.
    """
    configs = [
        ("A_first_contact_pre_death", "trace1_primary", False),
        ("B_pre_death_mean", "trace1_primary", False),
        ("A_first_contact_all", "trace1_primary", False),
        ("A_first_contact_pre_death", "trace1_no_outliers", True),
        ("B_pre_death_mean", "trace1_no_outliers", True),
        ("A_first_contact_pre_death", "all_contacted_sensitivity", False),
        ("B_pre_death_mean", "all_contacted_sensitivity", False),
    ]

    all_rows = []
    for summary_name, cohort_name, drop_outliers in configs:
        df = base_ch[base_ch["has_contact_desc_200m"]].copy()
        if cohort_name.startswith("trace1"):
            df = df[df["trace_num"] == 1].copy()
        if drop_outliers:
            df = df[~df["outlier_vol_flag"]].copy()

        if summary_name == "B_pre_death_mean":
            df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
            suffix = "_B_any"
        elif summary_name == "A_first_contact_pre_death":
            df = df[~df["fc_is_death_visit_any_200m"]].copy()
            suffix = "_A"
        else:
            suffix = "_A"

        for family_name, desc_list in [("primary", PRIMARY_DESCRIPTORS), ("secondary", SECONDARY_DESCRIPTORS)]:
            for d in desc_list:
                col = f"{d}{suffix}"
                pair_stats = {}
                n_lowered = 0
                for donor_id, exp_untr, exp_drug in PAIRED_CONTROL_EXPERIMENTS:
                    vu = df.loc[df["experiment"] == exp_untr, col].dropna().to_numpy(dtype=float)
                    vd = df.loc[df["experiment"] == exp_drug, col].dropna().to_numpy(dtype=float)
                    med_u = float(np.median(vu)) if len(vu) > 0 else np.nan
                    med_d = float(np.median(vd)) if len(vd) > 0 else np.nan
                    iqr_u = float(np.percentile(vu, 75) - np.percentile(vu, 25)) if len(vu) > 0 else np.nan
                    iqr_d = float(np.percentile(vd, 75) - np.percentile(vd, 25)) if len(vd) > 0 else np.nan
                    diff = med_d - med_u
                    lowered = bool(med_d < med_u)
                    if lowered:
                        n_lowered += 1
                    u_stat, p_mw = mannwhitneyu(vu, vd, alternative="two-sided")
                    pair_stats[donor_id] = {
                        "n_untr": int(len(vu)),
                        "n_drug": int(len(vd)),
                        "med_untr": med_u,
                        "iqr_untr": iqr_u,
                        "med_drug": med_d,
                        "iqr_drug": iqr_d,
                        "diff": diff,
                        "lowered": lowered,
                        "mw_u": float(u_stat),
                        "mw_p": float(p_mw),
                    }

                # Pooled donor-stratified Cox/rank or Mann-Whitney test across the 3 pairs
                sub_pairs = df[
                    df["experiment"].isin([e for _, eu, ed in PAIRED_CONTROL_EXPERIMENTS for e in (eu, ed)])
                ].copy()
                sub_pairs["is_drug"] = (sub_pairs["condition"] == CONDITION_DRUG).astype(int)
                # Rank within donor (pooling untreated + drug for the same donor)
                sub_pairs["z_within_donor"] = normal_scores_within_experiment(sub_pairs, col, exp_col="donor")
                z_u = sub_pairs.loc[sub_pairs["is_drug"] == 0, "z_within_donor"].dropna().to_numpy()
                z_d = sub_pairs.loc[sub_pairs["is_drug"] == 1, "z_within_donor"].dropna().to_numpy()
                _, p_pooled_donor_strat = mannwhitneyu(z_u, z_d, alternative="two-sided")

                all_rows.append(
                    {
                        "summary": summary_name,
                        "cohort": cohort_name,
                        "family": family_name,
                        "descriptor": d,
                        "label": DESCRIPTOR_LABELS.get(d, d),
                        "hd2_n_untr": pair_stats["hd2"]["n_untr"],
                        "hd2_n_drug": pair_stats["hd2"]["n_drug"],
                        "hd2_med_untr": pair_stats["hd2"]["med_untr"],
                        "hd2_med_drug": pair_stats["hd2"]["med_drug"],
                        "hd2_diff": pair_stats["hd2"]["diff"],
                        "hd2_mw_p": pair_stats["hd2"]["mw_p"],
                        "hd10_n_untr": pair_stats["hd10"]["n_untr"],
                        "hd10_n_drug": pair_stats["hd10"]["n_drug"],
                        "hd10_med_untr": pair_stats["hd10"]["med_untr"],
                        "hd10_med_drug": pair_stats["hd10"]["med_drug"],
                        "hd10_diff": pair_stats["hd10"]["diff"],
                        "hd10_mw_p": pair_stats["hd10"]["mw_p"],
                        "hd5_n_untr": pair_stats["hd5"]["n_untr"],
                        "hd5_n_drug": pair_stats["hd5"]["n_drug"],
                        "hd5_med_untr": pair_stats["hd5"]["med_untr"],
                        "hd5_med_drug": pair_stats["hd5"]["med_drug"],
                        "hd5_diff": pair_stats["hd5"]["diff"],
                        "hd5_mw_p": pair_stats["hd5"]["mw_p"],
                        "n_donors_lowered": int(n_lowered),
                        "lowered_3_of_3_donors": bool(n_lowered == 3),
                        "donor_stratified_rank_mw_p": float(p_pooled_donor_strat),
                    }
                )

    return pd.DataFrame(all_rows)


def run_exploratory_checks(base_ch: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory analyses (explicitly labelled exploratory, do NOT feed the pre-registered decision):
    1. Why does Summary B show stronger HR than Summary A? Check correlation of number of
       pre-death contact visits (exposure duration before censoring/death) with mean pre-death
       descriptors, and repeat Test 1 on Summary B adjusting for log(n_pre_death_contact_visits).
    2. Leave-one-experiment-out AUC using only the top Cox-associated indentation descriptors
       (`cup_depth_over_radius` + `cup_depth_hull_p95_um` + `tcell_protrusion_frac`) + condition
       to test whether parsimonious indentation features improve held-out prediction when avoiding
       12-feature overfitting.
    """
    df = base_ch[
        (base_ch["trace_num"] == 1)
        & base_ch["has_contact_desc_200m"]
        & (base_ch["n_pre_death_contact_visits_any_200m"] > 0)
    ].copy()
    df["z_n_vis"] = normal_scores_within_experiment(df, "n_pre_death_contact_visits_any_200m")
    df["z_ca_B"] = normal_scores_within_experiment(df, "contact_area_um2_B_any")

    rows = []
    for d in PRIMARY_DESCRIPTORS:
        col_B = f"{d}_B_any"
        df["z_d_B"] = normal_scores_within_experiment(df, col_B)
        res_u = fit_stratified_cox(df, "km_any_contact_time_min_200m", "km_event_any_visit_200m", ["z_d_B"])
        res_adj_n = fit_stratified_cox(
            df, "km_any_contact_time_min_200m", "km_event_any_visit_200m", ["z_d_B", "z_n_vis"]
        )
        res_adj_both = fit_stratified_cox(
            df, "km_any_contact_time_min_200m", "km_event_any_visit_200m", ["z_d_B", "z_ca_B", "z_n_vis"]
        )
        rho_nvis, p_nvis = spearmanr(
            df[col_B].dropna(),
            df.loc[df[col_B].notna(), "n_pre_death_contact_visits_any_200m"],
        )
        rows.append(
            {
                "analysis_type": "EXPLORATORY_DO_NOT_FEED_DECISION",
                "descriptor": d,
                "hr_B_unadj": float(res_u["hr"][0]),
                "p_B_unadj": float(res_u["p"][0]),
                "hr_B_adj_n_predeath_visits": float(res_adj_n["hr"][0]),
                "p_B_adj_n_predeath_visits": float(res_adj_n["p"][0]),
                "hr_B_adj_ca_and_n_visits": float(res_adj_both["hr"][0]),
                "p_B_adj_ca_and_n_visits": float(res_adj_both["p"][0]),
                "spearman_with_n_predeath_visits": float(rho_nvis),
                "spearman_p_with_n_predeath_visits": float(p_nvis),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    test1_df: pd.DataFrame,
    test2_df: pd.DataFrame,
    roc_data: dict[str, Any],
    test3_df: pd.DataFrame,
    donor_meds_df: pd.DataFrame,
    test4_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Generate figures/step5b_association_check.png following House Style:
    - Plain white background (#FFFFFF), dark ink (#111111), muted (#5C5C5C), rule (#E3E3E3)
    - Data palette: --plot-1 (#1F4E79), --plot-2 (#C0561F), --plot-3 (#2E7D5B), --plot-4 (#6B4C9A)
    - System sans-serif font stack
    - 4 clear labelled panels:
      (a) Forest plot of HRs per descriptor for both families (Test 1, Summaries A & B)
      (b) Held-out LOEO ROC & AUC comparison: Baseline vs +Primary Descriptors (Test 2)
      (c) Donor medians vs donor kill rate for the main indentation descriptors (Test 3)
      (d) Latrunculin paired positive control across the 3 donors (Test 4)
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "axes.edgecolor": "#C9C9C9",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#111111",
            "xtick.color": "#1F1F1F",
            "ytick.color": "#1F1F1F",
            "text.color": "#111111",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    PLOT_1 = "#1F4E79"  # Deep blue
    PLOT_2 = "#C0561F"  # Ochre / rust
    PLOT_3 = "#2E7D5B"  # Muted green
    PLOT_4 = "#6B4C9A"  # Muted purple
    INK = "#111111"
    MUTED = "#5C5C5C"
    RULE = "#E3E3E3"
    LINE = "#C9C9C9"

    fig = plt.figure(figsize=(16.5, 11.5), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.10, 1.0],
        height_ratios=[1.12, 1.0],
        hspace=0.32,
        wspace=0.46,
        left=0.16,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )

    # -------------------------------------------------------------------------
    # Panel A: Forest plot of HRs per descriptor for both families (Test 1)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[:, 0])
    t1_A = (
        test1_df[
            (test1_df["summary"] == "A_first_contact_pre_death")
            & (test1_df["cohort"] == "trace1_primary")
            & (test1_df["outcome_rule"] == "any_visit")
        ]
        .set_index("descriptor")
        .reindex(ALL_DESCRIPTORS)
    )
    t1_B = (
        test1_df[
            (test1_df["summary"] == "B_pre_death_mean")
            & (test1_df["cohort"] == "trace1_primary")
            & (test1_df["outcome_rule"] == "any_visit")
        ]
        .set_index("descriptor")
        .reindex(ALL_DESCRIPTORS)
    )

    n_desc = len(ALL_DESCRIPTORS)
    # Position primary descriptors at top, gap of 1.2 units before secondary family
    y_pos = []
    curr_y = float(n_desc + 1)
    for idx, d in enumerate(ALL_DESCRIPTORS):
        if idx == len(PRIMARY_DESCRIPTORS):
            curr_y -= 1.1
        y_pos.append(curr_y)
        curr_y -= 1.0
    y_pos = np.array(y_pos, dtype=float)

    ax_a.axvline(1.0, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    for yv in y_pos:
        ax_a.axhline(yv, color=RULE, linewidth=0.5, zorder=0)

    # Divider line between primary and secondary families
    div_y = (y_pos[len(PRIMARY_DESCRIPTORS) - 1] + y_pos[len(PRIMARY_DESCRIPTORS)]) / 2.0
    ax_a.axhline(div_y, color=LINE, linestyle="-", linewidth=1.0, zorder=1)
    ax_a.text(
        0.52,
        y_pos[0] + 0.55,
        "Primary family (ICC >= 0.50)",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax_a.text(
        0.52,
        div_y - 0.15,
        "Secondary family (topography, ICC < 0.50)",
        fontsize=9.5,
        fontweight="bold",
        color=MUTED,
        va="top",
    )

    offset = 0.16
    # Plot Summary A (first contact, pre-death)
    ax_a.errorbar(
        t1_A["hr_unadj"],
        y_pos + offset,
        xerr=[
            t1_A["hr_unadj"] - t1_A["ci_low_unadj"],
            t1_A["ci_high_unadj"] - t1_A["hr_unadj"],
        ],
        fmt="o",
        color=PLOT_1,
        ecolor=PLOT_1,
        elinewidth=1.3,
        capsize=2.5,
        markersize=5.0,
        label="(A) First contact (n=365–366)",
        zorder=3,
    )
    # Plot Summary B (mean pre-death contact visits)
    ax_a.errorbar(
        t1_B["hr_unadj"],
        y_pos - offset,
        xerr=[
            t1_B["hr_unadj"] - t1_B["ci_low_unadj"],
            t1_B["ci_high_unadj"] - t1_B["hr_unadj"],
        ],
        fmt="s",
        color=PLOT_2,
        ecolor=PLOT_2,
        elinewidth=1.3,
        capsize=2.5,
        markersize=4.8,
        label="(B) Pre-death mean (n=365–366)",
        zorder=3,
    )

    # Annotate q < 0.05 & direction consistency on the right side of Panel A
    for idx, d in enumerate(ALL_DESCRIPTORS):
        row_a = t1_A.loc[d]
        row_b = t1_B.loc[d]
        q_a = row_a["bh_q_unadj"]
        q_b = row_b["bh_q_unadj"]
        tag_a = f"q={q_a:.3f}" if q_a >= 0.001 else "q<0.001"
        tag_b = f"q={q_b:.3f}" if q_b >= 0.001 else "q<0.001"
        gate_mark = "*" if (row_a["passes_test1_gate"] or row_b["passes_test1_gate"]) else ""
        ax_a.text(
            2.62,
            y_pos[idx],
            f"A:{tag_a} | B:{tag_b}{gate_mark}",
            fontsize=7.5,
            color=INK if gate_mark else MUTED,
            fontweight="bold" if gate_mark else "normal",
            va="center",
            ha="left",
        )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([DESCRIPTOR_LABELS[d] for d in ALL_DESCRIPTORS], fontsize=8.8)
    ax_a.set_xlim(0.48, 3.48)
    ax_a.set_ylim(y_pos[-1] - 0.8, y_pos[0] + 1.1)
    ax_a.set_xlabel("Hazard Ratio per within-experiment SD (95% CI)", fontsize=9.5, fontweight="bold")
    ax_a.set_title("Test 1: Stratified Cox PH of time to death (200 min)", fontsize=11, fontweight="bold", loc="left")
    ax_a.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, fontsize=8.5)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B: Held-out LOEO AUC baseline vs +primary descriptors (Test 2)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    sub_t2 = test2_df[
        (test2_df["cohort"] == "trace1_primary")
        & (test2_df["outcome_rule"] == "any_visit")
        & (
            test2_df["summary"].isin(["A_first_contact_pre_death", "B_pre_death_mean"])
        )
        & (test2_df["transform_mode"].isin(["within_exp_rank", "fold_pooled_rank", "raw_zscore"]))
    ].copy()

    # Plot horizontal forest / dot-interval comparison of Baseline vs +Primary Descriptors
    labels_b = [
        "(A) First contact | Within-exp rank",
        "(B) Pre-death mean | Within-exp rank",
        "(A) First contact | Fold-pooled rank",
        "(B) Pre-death mean | Fold-pooled rank",
        "(A) First contact | Raw z-score",
        "(B) Pre-death mean | Raw z-score",
    ]
    keys_b = [
        ("A_first_contact_pre_death", "within_exp_rank"),
        ("B_pre_death_mean", "within_exp_rank"),
        ("A_first_contact_pre_death", "fold_pooled_rank"),
        ("B_pre_death_mean", "fold_pooled_rank"),
        ("A_first_contact_pre_death", "raw_zscore"),
        ("B_pre_death_mean", "raw_zscore"),
    ]
    yb = np.arange(len(keys_b), dtype=float)[::-1]
    for yv in yb:
        ax_b.axhline(yv, color=RULE, linewidth=0.5, zorder=0)

    for idx, (s_name, t_mode) in enumerate(keys_b):
        r = sub_t2[(sub_t2["summary"] == s_name) & (sub_t2["transform_mode"] == t_mode)].iloc[0]
        yv = yb[idx]
        # Baseline
        ax_b.errorbar(
            r["auc_baseline"],
            yv + 0.15,
            xerr=[[r["auc_baseline"] - r["auc_baseline_ci_low"]], [r["auc_baseline_ci_high"] - r["auc_baseline"]]],
            fmt="o",
            color=MUTED,
            ecolor=MUTED,
            elinewidth=1.3,
            capsize=2.5,
            markersize=5.0,
            label="Baseline (CA + condition)" if idx == 0 else None,
            zorder=3,
        )
        # + Primary descriptors
        ax_b.errorbar(
            r["auc_plus_primary"],
            yv - 0.15,
            xerr=[[r["auc_plus_primary"] - r["auc_plus_ci_low"]], [r["auc_plus_ci_high"] - r["auc_plus_primary"]]],
            fmt="s",
            color=PLOT_1,
            ecolor=PLOT_1,
            elinewidth=1.3,
            capsize=2.5,
            markersize=5.0,
            label="Baseline + 12 primary descriptors" if idx == 0 else None,
            zorder=3,
        )
        gain_txt = f"dAUC={r['auc_gain']:+.3f} [{r['auc_gain_ci_low']:+.3f}, {r['auc_gain_ci_high']:+.3f}]"
        ax_b.text(0.815, yv, gain_txt, fontsize=7.8, color=INK, va="center", ha="left")

    ax_b.axvline(0.5, color=LINE, linestyle=":", linewidth=0.8)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(labels_b, fontsize=8.5)
    ax_b.set_xlim(0.58, 0.99)
    ax_b.set_ylim(-0.6, len(keys_b) - 0.3)
    ax_b.set_xlabel("Leave-one-experiment-out AUC (95% bootstrap CI)", fontsize=9.5, fontweight="bold")
    ax_b.set_title("Test 2: Held-out death prediction (threshold: dAUC >= +0.03)", fontsize=11, fontweight="bold", loc="left")
    ax_b.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, fontsize=8.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C & D: Subdivide bottom-right into 2x2 mini-panels:
    # Top row of bottom-right: Test 3 Donor medians vs kill rate (2 main indentation descriptors)
    # Bottom row of bottom-right: Test 4 Latrunculin paired positive control (2 main indentation descriptors)
    # -------------------------------------------------------------------------
    gs_br = gs[1, 1].subgridspec(2, 2, hspace=0.48, wspace=0.28)

    # Panel C: Donor medians vs donor kill rate (Summary A, trace_num == 1)
    meds_A = donor_meds_df[
        (donor_meds_df["summary"] == "A_first_contact_pre_death")
        & (donor_meds_df["cohort"] == "trace1_primary")
    ].copy()
    t3_A = test3_df[
        (test3_df["summary"] == "A_first_contact_pre_death")
        & (test3_df["cohort"] == "trace1_primary")
    ].set_index("descriptor")

    for j, (d_col, short_lbl) in enumerate(
        [
            ("cup_depth_hull_p95_um", "Cup depth p95 (um)"),
            ("tcell_protrusion_frac", "Protrusion fraction"),
        ]
    ):
        ax_c = fig.add_subplot(gs_br[0, j])
        hi_m = meds_A[meds_A["condition"] == CONDITION_HIGH]
        lo_m = meds_A[meds_A["condition"] == CONDITION_LOW]
        ax_c.scatter(
            hi_m[d_col],
            hi_m["kill_rate_any_200m"] * 100.0,
            color=PLOT_1,
            s=36,
            label="High-killing (n=5)",
            zorder=3,
        )
        ax_c.scatter(
            lo_m[d_col],
            lo_m["kill_rate_any_200m"] * 100.0,
            color=PLOT_2,
            marker="^",
            s=38,
            label="Low-killing (n=5)",
            zorder=3,
        )
        for _, r_don in meds_A.iterrows():
            don_lbl = str(r_don["donor"])
            dx, dy = (3, 2)
            if don_lbl == "hd1":
                dx, dy = (-16, 4)
            elif don_lbl == "hd10":
                dx, dy = (-18, 3)
            elif don_lbl == "hd2":
                dx, dy = (4, -6)
            elif don_lbl == "hd8":
                dx, dy = (-16, 2)
            ax_c.annotate(
                don_lbl,
                (r_don[d_col], r_don["kill_rate_any_200m"] * 100.0),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                color=MUTED,
            )
        r_stat = t3_A.loc[d_col]
        ax_c.set_xlabel(f"Donor median {short_lbl}", fontsize=8.5, fontweight="bold")
        if j == 0:
            ax_c.set_ylabel("Kill rate 200m (%)", fontsize=8.5, fontweight="bold")
        ax_c.set_title(
            f"Test 3: {short_lbl}\nMW p={r_stat['mw_p_two_sided']:.3f}, rho={r_stat['spearman_rho_kill_rate']:+.2f}",
            fontsize=8.8,
            fontweight="bold",
            loc="left",
        )
        ax_c.set_ylim(5, 72)
        if j == 0:
            ax_c.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, fontsize=7.0)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)

    # Panel D: Test 4 Latrunculin paired positive control (both Summary A and Summary B shown)
    t4_A = test4_df[
        (test4_df["summary"] == "A_first_contact_pre_death")
        & (test4_df["cohort"] == "trace1_primary")
    ].set_index("descriptor")
    t4_B = test4_df[
        (test4_df["summary"] == "B_pre_death_mean")
        & (test4_df["cohort"] == "trace1_primary")
    ].set_index("descriptor")

    for j, (d_col, short_lbl) in enumerate(
        [
            ("cup_depth_hull_p95_um", "Cup depth p95 (um)"),
            ("tcell_protrusion_frac", "Protrusion fraction"),
        ]
    ):
        ax_d = fig.add_subplot(gs_br[1, j])
        rA = t4_A.loc[d_col]
        rB = t4_B.loc[d_col]
        x_pair = np.array([0.0, 1.0])
        donor_colors = {"hd2": PLOT_1, "hd10": PLOT_2, "hd5": PLOT_3}
        for don_id in ["hd2", "hd10", "hd5"]:
            # Solid line: Summary B (mean pre-death), Dashed line: Summary A (first contact)
            yB = [rB[f"{don_id}_med_untr"], rB[f"{don_id}_med_drug"]]
            yA = [rA[f"{don_id}_med_untr"], rA[f"{don_id}_med_drug"]]
            ax_d.plot(
                x_pair,
                yB,
                "-s",
                color=donor_colors[don_id],
                linewidth=1.5,
                markersize=4.8,
                label=f"{don_id} (B)" if j == 0 else None,
            )
            ax_d.plot(
                x_pair,
                yA,
                "--o",
                color=donor_colors[don_id],
                linewidth=1.0,
                markersize=4.0,
                alpha=0.75,
                label=f"{don_id} (A)" if j == 0 else None,
            )
        ax_d.set_xticks([0.0, 1.0])
        ax_d.set_xticklabels(["Untreated", "Drug (8*–10*)"], fontsize=8.5, fontweight="bold")
        ax_d.set_xlim(-0.25, 1.25)
        if j == 0:
            ax_d.set_ylabel("Donor median", fontsize=8.5, fontweight="bold")
            ax_d.legend(
                loc="upper right",
                ncol=2,
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor=LINE,
                fontsize=6.5,
            )
        ax_d.set_title(
            f"Test 4: {short_lbl}\nLowered: A={int(rA['n_donors_lowered'])}/3, B={int(rB['n_donors_lowered'])}/3",
            fontsize=8.8,
            fontweight="bold",
            loc="left",
        )
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        "Step 5b Pre-Registered Association Check: Contact Shape Descriptors vs Killing & Donor",
        fontsize=13,
        fontweight="bold",
        x=0.03,
        y=0.98,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_ch, outlier_table, _, meta_counts = build_datasets()
    base_ch.to_csv(OUT_DIR / "pre_death_chamber_summaries.csv", index=False)
    outlier_table.to_csv(OUT_DIR / "outlier_flagged_chambers.csv", index=False)

    test1_df = run_test1_cox(base_ch)
    test1_df.to_csv(OUT_DIR / "test1_cox_death.csv", index=False)

    test2_df, roc_data = run_test2_loeo_auc(base_ch, n_boot=2000, seed=42)
    test2_df.to_csv(OUT_DIR / "test2_loeo_auc.csv", index=False)

    test3_df, donor_meds_df = run_test3_donor(base_ch)
    test3_df.to_csv(OUT_DIR / "test3_donor_association.csv", index=False)
    donor_meds_df.to_csv(OUT_DIR / "test3_donor_medians.csv", index=False)

    test4_df = run_test4_latrunculin(base_ch)
    test4_df.to_csv(OUT_DIR / "test4_latrunculin_control.csv", index=False)

    exploratory_df = run_exploratory_checks(base_ch)
    exploratory_df.to_csv(OUT_DIR / "exploratory_checks.csv", index=False)

    fig_path = FIG_DIR / "step5b_association_check.png"
    make_figure(test1_df, test2_df, roc_data, test3_df, donor_meds_df, test4_df, fig_path)

    # Evaluate pre-registered decision rule
    t1_prim_A = test1_df[
        (test1_df["summary"] == "A_first_contact_pre_death")
        & (test1_df["cohort"] == "trace1_primary")
        & (test1_df["outcome_rule"] == "any_visit")
        & (test1_df["family"] == "primary")
    ]
    t1_prim_B = test1_df[
        (test1_df["summary"] == "B_pre_death_mean")
        & (test1_df["cohort"] == "trace1_primary")
        & (test1_df["outcome_rule"] == "any_visit")
        & (test1_df["family"] == "primary")
    ]
    t1_pass_A = t1_prim_A[t1_prim_A["passes_test1_gate"]]["descriptor"].tolist()
    t1_pass_B = t1_prim_B[t1_prim_B["passes_test1_gate"]]["descriptor"].tolist()

    t2_prim = test2_df[
        (test2_df["cohort"] == "trace1_primary") & (test2_df["outcome_rule"] == "any_visit")
    ]
    t2_pass = bool(t2_prim["passes_test2_gate"].any())

    t3_prim = test3_df[
        (test3_df["cohort"] == "trace1_primary") & (test3_df["family"] == "primary")
    ]
    t3_pass = bool(t3_prim["passes_test3_gate"].any())

    t4_prim_A = test4_df[
        (test4_df["summary"] == "A_first_contact_pre_death")
        & (test4_df["cohort"] == "trace1_primary")
        & (test4_df["descriptor"].isin(["cup_depth_hull_p95_um", "tcell_protrusion_frac"]))
    ]
    t4_prim_B = test4_df[
        (test4_df["summary"] == "B_pre_death_mean")
        & (test4_df["cohort"] == "trace1_primary")
        & (test4_df["descriptor"].isin(["cup_depth_hull_p95_um", "tcell_protrusion_frac"]))
    ]

    decision = (
        "Association present"
        if (len(t1_pass_A) > 0 or len(t1_pass_B) > 0 or t2_pass or t3_pass)
        else (
            "No association"
            if (bool(t4_prim_A["lowered_3_of_3_donors"].any()) or bool(t4_prim_B["lowered_3_of_3_donors"].any()))
            else "Inconclusive"
        )
    )

    summary_json = {
        "decision": decision,
        "meta_counts": meta_counts,
        "test1_passing_primary_descriptors_A": t1_pass_A,
        "test1_passing_primary_descriptors_B": t1_pass_B,
        "test2_any_passing": t2_pass,
        "test3_any_passing": t3_pass,
        "test4_positive_control_A_3of3": t4_prim_A.set_index("descriptor")["lowered_3_of_3_donors"].to_dict(),
        "test4_positive_control_B_3of3": t4_prim_B.set_index("descriptor")["lowered_3_of_3_donors"].to_dict(),
    }
    with open(OUT_DIR / "summary_step5b.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Decision: {decision}")
    print(f"Test 1 passing primary descriptors (Summary A): {t1_pass_A}")
    print(f"Test 1 passing primary descriptors (Summary B): {t1_pass_B}")
    print(f"Test 2 passing: {t2_pass}")
    print(f"Test 3 passing: {t3_pass}")
    print(f"Outputs written to {OUT_DIR} and {fig_path}")


if __name__ == "__main__":
    main()
