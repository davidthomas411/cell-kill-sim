#!/usr/bin/env python3
"""
Step 7 pre-registered analysis: Do high-killing donors push in a more focused way?

Executes status/step7_preregistration.md exactly as committed without altering any rule.

Inputs (read-only):
- data/step5b/pre_death_chamber_summaries.csv
- data/step5b/outlier_flagged_chambers.csv
- data/step5/per_visit_descriptors.csv
- data/step5/per_chamber_descriptors.csv
- data/step4/per_visit_death_labels_all_v2.csv
- data/step4/per_chamber_death_labels_all_v2.csv
- data/step6b/afm_to_tension_conversion.json

Outputs:
- data/step7/test7a_donor_focus.csv
- data/step7/test7a_donor_medians.csv
- data/step7/test7a_visit_lmm.csv
- data/step7/test7b_within_donor_cox.csv
- data/step7/test7c_dasatinib_control.csv
- data/step7/stiffness_route_b_summary.csv
- data/step7/exploratory_step7.csv
- data/step7/summary.json
- figures/step7_focus.png
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import kruskal, mannwhitneyu, norm, rankdata, spearmanr, t as t_dist

from step5b_association_check import (
    CONDITION_DRUG,
    CONDITION_HIGH,
    CONDITION_LOW,
    PAIRED_CONTROL_EXPERIMENTS,
    bh_fdr,
    build_datasets,
    extract_donor,
    fit_stratified_cox,
    normal_scores_within_experiment,
)

ROOT = Path(__file__).resolve().parent.parent
STEP5_DIR = ROOT / "data" / "step5"
STEP5B_DIR = ROOT / "data" / "step5b"
STEP6B_DIR = ROOT / "data" / "step6b"
OUT_DIR = ROOT / "data" / "step7"
FIG_DIR = ROOT / "figures"

FOCUS_METRICS = [
    "cup_depth_over_radius",
    "cup_depth_hull_p95_um",
    "tcell_protrusion_frac",
]

PRIMARY_METRIC = "cup_depth_over_radius"

METRIC_LABELS = {
    "cup_depth_over_radius": "Cup depth / patch radius (primary)",
    "cup_depth_hull_p95_um": "Cup depth p95 (um, secondary 1)",
    "tcell_protrusion_frac": "T-cell protrusion fraction (secondary 2)",
    "pi_simple": "Dimensionless push Pi_simple (provisional: route (b), pending 6b gate)",
    "delta_pi_indent": "Net indentation push Delta_Pi_indent (provisional: route (b), pending 6b gate)",
}

EXPLORATORY_METRICS = [
    "zernike_defocus_um",
    "zernike_tilt_amp_um",
    "zernike_astig_amp_um",
    "zernike_coma_amp_um",
    "zernike_spherical_um",
    "cup_asymmetry_norm",
    "rough_above_cutoff_um2",
    "patch_equiv_radius_um",
    "contact_area_um2",
]

DECISION_SENTENCES = {
    "donor_level_supported": (
        "T cells from high-killing donors push in a significantly more focused way "
        "than T cells from low-killing donors."
    ),
    "within_donor_only": (
        "Within a donor/experiment, more focused pushing predicts faster killing "
        "beyond contact size and duration, but between-donor differences do not "
        "exceed batch variation (n = 5 vs 5)."
    ),
    "no_effect": (
        "Focused pushing does not explain killing either between or within donors."
    ),
    "positive_control_fails": (
        "The focus metric does not respond to a known perturbation of T-cell pushing, "
        "so step 7 is inconclusive."
    ),
}


def compute_route_b_stiffness_visit(df_vis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Route (b) paper relation on visit-level descriptors:
      R_target = (3 * V_target / (4 * pi))^(1/3)
      d = cup_depth_hull_p95_um
      a = patch_equiv_radius_um
      cup_depth_over_radius = d / a
      Delta_Pi_indent = 4 * (R_target / a) * cup_depth_over_radius
      Pi_simple = 2 + Delta_Pi_indent
    Also computes finite-depth denominator sensitivity:
      Delta_Pi_finite = 4 * R_target * d / (a^2 + d^2)
      Pi_finite = 2 + Delta_Pi_finite
    """
    out = df_vis.copy()
    r_t = ((3.0 * out["target_volume_um3"].clip(lower=0.0)) / (4.0 * np.pi)) ** (1.0 / 3.0)
    a = out["patch_equiv_radius_um"].replace(0.0, np.nan)
    d = out["cup_depth_hull_p95_um"]
    d_over_a = out["cup_depth_over_radius"]
    out["r_target_um"] = r_t
    out["delta_pi_indent"] = 4.0 * (r_t / a) * d_over_a
    out["pi_simple"] = 2.0 + out["delta_pi_indent"]
    out["delta_pi_finite"] = 4.0 * r_t * d / (a ** 2 + d ** 2)
    out["pi_finite"] = 2.0 + out["delta_pi_finite"]
    return out


def augment_chamber_summaries_with_stiffness_and_sd(
    base_ch: pd.DataFrame, mv: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Route (b) stiffness summaries (_A and _B_any) and within-visit sweep SD
    summaries to the chamber table, and return the pre-death contact visit table.
    """
    mv_stiff = compute_route_b_stiffness_visit(mv)
    mv20 = mv_stiff[(mv_stiff["visit"] <= 20) & (mv_stiff["in_contact_visit"] == True)].copy()
    mv20 = mv20.merge(
        base_ch[
            [
                "experiment",
                "roi",
                "donor",
                "condition",
                "first_contact_visit_desc_200m",
                "fc_is_death_visit_any_200m",
                "outlier_vol_flag",
            ]
        ],
        on=["experiment", "roi"],
        how="left",
    )
    mv20["out_cup_gt_target_visit"] = (
        mv20["cup_volume_hull_um3"] > mv20["target_volume_um3"]
    ).fillna(False)

    # Pre-death visits strictly before first_any_dead_visit_200m
    mv20_pre = mv20[
        mv20["first_any_dead_visit_200m"].isna()
        | (mv20["visit"] < mv20["first_any_dead_visit_200m"])
    ].copy()
    mv20_pre["time_in_contact_visits"] = (
        mv20_pre["visit"] - mv20_pre["first_contact_visit_desc_200m"]
    ).astype(float)
    mv20_pre["time_in_contact_min"] = mv20_pre["time_in_contact_visits"] * 10.0

    stiff_cols = ["r_target_um", "delta_pi_indent", "pi_simple", "delta_pi_finite", "pi_finite"]
    sd_cols = [
        "cup_depth_over_radius_sd",
        "cup_depth_hull_p95_um_sd",
        "tcell_protrusion_frac_sd",
    ]
    extra_cols = stiff_cols + sd_cols

    vis_A = mv20[mv20["visit"] == mv20["first_contact_visit_desc_200m"]].copy()
    df_A = vis_A[["experiment", "roi"] + extra_cols].rename(
        columns={c: f"{c}_A" for c in extra_cols}
    )

    df_B = (
        mv20_pre.groupby(["experiment", "roi"])[extra_cols]
        .mean()
        .rename(columns={c: f"{c}_B_any" for c in extra_cols})
        .reset_index()
    )

    ch = base_ch.merge(df_A, on=["experiment", "roi"], how="left")
    ch = ch.merge(df_B, on=["experiment", "roi"], how="left")
    return ch, mv20_pre


def hedges_g_and_perm_ci(hi: np.ndarray, lo: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Hedges' g (small-sample corrected standardized mean difference)
    and exact permutation 95% confidence interval by inverting the 252-permutation
    test for constant additive shifts delta on group high.
    """
    n1, n2 = len(hi), len(lo)
    dof = n1 + n2 - 2
    s_pool = float(np.sqrt(((n1 - 1) * np.var(hi, ddof=1) + (n2 - 1) * np.var(lo, ddof=1)) / dof))
    if s_pool <= 0:
        return np.nan, np.nan, np.nan
    j_corr = 1.0 - 3.0 / (4.0 * dof - 1.0)
    g_obs = float(j_corr * (np.mean(hi) - np.mean(lo)) / s_pool)

    # Exact Hodges-Lehmann / permutation shift 95% CI on mean difference, scaled to Hedges' g
    # Under additive shift delta on hi: hi_shifted = hi - delta has the same within-group variance s_pool!
    # In a two-group permutation test of size n1=5, n2=5 (252 permutations), the exact 95% acceptance
    # region for an additive shift delta is the range of delta where two-sided permutation p(delta) >= 0.05
    # (i.e., at least ceil(0.05 * 252) = 13 permutations have |diff_perm(delta)| >= |diff_obs(delta)|).
    # Because |diff_perm(delta) - diff_obs(delta)| reduces to a linear function of delta depending on
    # the number k of high-group items kept in group 1, we evaluate across a dense grid of delta.
    obs_diff = float(np.mean(hi) - np.mean(lo))
    span = max(abs(obs_diff) * 4.0, s_pool * 4.0)
    deltas = np.linspace(obs_diff - span, obs_diff + span, 2001)
    idx_combos = list(combinations(range(n1 + n2), n1))
    masks = np.zeros((len(idx_combos), n1 + n2), dtype=bool)
    for row_i, c in enumerate(idx_combos):
        masks[row_i, list(c)] = True

    accepted = []
    for d_shift in deltas:
        vals = np.concatenate([hi - d_shift, lo])
        d_obs_s = obs_diff - d_shift
        perm_means_1 = (masks @ vals) / n1
        perm_means_2 = ((~masks) @ vals) / n2
        perm_diffs = perm_means_1 - perm_means_2
        p_val = np.mean(np.abs(perm_diffs) >= abs(d_obs_s) - 1e-12)
        if p_val > 0.05:
            accepted.append(d_shift)

    if accepted:
        g_ci_low = float(j_corr * min(accepted) / s_pool)
        g_ci_high = float(j_corr * max(accepted) / s_pool)
    else:
        g_ci_low, g_ci_high = np.nan, np.nan
    return g_obs, g_ci_low, g_ci_high


def spearman_boot_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 4000, seed: int = 42
) -> tuple[float, float, float, float]:
    """Compute Spearman rho, two-sided p, and bootstrap 95% CI across donors."""
    rho, p_val = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(float(r))
    if boots:
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    else:
        ci_lo, ci_hi = np.nan, np.nan
    return float(rho), float(p_val), float(ci_lo), float(ci_hi)


def run_test7a_donor_permutation(ch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test 7A (part 1): Donor-level permutation test (5 high vs 5 low untreated donors).
    Evaluates:
      - Summaries: A_first_contact_pre_death, B_pre_death_mean, A_first_contact_all (sensitivity)
      - Adjustment modes: unadjusted, residualized (on patch_equiv_radius_um and pre-death visit count)
      - Cohorts: trace1_primary, trace1_no_outliers, all_chambers
    """
    configs = [
        ("A_first_contact_pre_death", "trace1_primary", False),
        ("B_pre_death_mean", "trace1_primary", False),
        ("A_first_contact_pre_death", "trace1_no_outliers", True),
        ("B_pre_death_mean", "trace1_no_outliers", True),
        ("A_first_contact_pre_death", "all_chambers", False),
        ("B_pre_death_mean", "all_chambers", False),
        ("A_first_contact_all", "trace1_primary", False),
    ]

    all_rows = []
    median_rows = []

    idx_combos = list(combinations(range(10), 5))

    for summary_name, cohort_name, drop_outliers in configs:
        pool = ch[ch["condition"] != CONDITION_DRUG].copy()
        if cohort_name.startswith("trace1"):
            pool = pool[pool["trace_num"] == 1].copy()
        if drop_outliers:
            pool = pool[~pool["outlier_vol_flag"]].copy()

        donor_kr = (
            pool.groupby(["experiment", "donor", "condition", "group_id"])
            .agg(
                n_chambers_total=("roi", "count"),
                n_contacted_200m=("has_contact_desc_200m", "sum"),
                n_kills_any_200m=("ever_dead_any_visit_200m", "sum"),
                kill_rate_any_200m=("ever_dead_any_visit_200m", "mean"),
                kill_rate_contacted_any_200m=(
                    "ever_dead_any_visit_200m",
                    lambda s: float(s[pool.loc[s.index, "has_contact_desc_200m"]].mean()),
                ),
            )
            .reset_index()
        )

        df = pool[pool["has_contact_desc_200m"]].copy()
        if summary_name == "B_pre_death_mean":
            df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
            suf = "_B_any"
            cov_cols = [f"patch_equiv_radius_um{suf}", "n_pre_death_contact_visits_any_200m"]
        elif summary_name == "A_first_contact_pre_death":
            df = df[~df["fc_is_death_visit_any_200m"]].copy()
            suf = "_A"
            cov_cols = [f"patch_equiv_radius_um{suf}"]
        else:
            suf = "_A"
            cov_cols = [f"patch_equiv_radius_um{suf}"]

        metrics_to_eval = FOCUS_METRICS + ["pi_simple", "delta_pi_indent"]

        for adj_mode in ["unadjusted", "residualized"]:
            block_rows = []
            for m in metrics_to_eval:
                col = f"{m}{suf}"
                req_cols = ["experiment", "donor", "condition", "group_id", col]
                if adj_mode == "residualized":
                    req_cols += [c for c in cov_cols if c != col]
                sub = df[req_cols].dropna().copy()

                if adj_mode == "residualized":
                    use_covs = [c for c in cov_cols if c != col]
                    X_cov = np.column_stack(
                        [np.ones(len(sub))]
                        + [sub[c].to_numpy(dtype=float) - float(sub[c].mean()) for c in use_covs]
                    )
                    y_raw = sub[col].to_numpy(dtype=float)
                    beta_ols, _, _, _ = np.linalg.lstsq(X_cov, y_raw, rcond=None)
                    # Add grand mean back so residualized values remain on physical scale
                    sub["val"] = y_raw - (X_cov[:, 1:] @ beta_ols[1:])
                else:
                    sub["val"] = sub[col].to_numpy(dtype=float)

                d_med = (
                    sub.groupby(["experiment", "donor", "condition", "group_id"])
                    .agg(
                        n_chambers_analyzed=("val", "count"),
                        donor_median=("val", "median"),
                        donor_q25=("val", lambda s: float(np.percentile(s, 25))),
                        donor_q75=("val", lambda s: float(np.percentile(s, 75))),
                        donor_mean=("val", "mean"),
                    )
                    .reset_index()
                )
                d_med = donor_kr.merge(
                    d_med, on=["experiment", "donor", "condition", "group_id"], how="left"
                )
                for _, r_m in d_med.iterrows():
                    median_rows.append(
                        {
                            "summary": summary_name,
                            "cohort": cohort_name,
                            "adjustment": adj_mode,
                            "metric": m,
                            "experiment": r_m["experiment"],
                            "donor": r_m["donor"],
                            "condition": r_m["condition"],
                            "group_id": int(r_m["group_id"]),
                            "n_chambers_total": int(r_m["n_chambers_total"]),
                            "n_chambers_analyzed": int(r_m["n_chambers_analyzed"]),
                            "kill_rate_any_200m": float(r_m["kill_rate_any_200m"]),
                            "kill_rate_contacted_any_200m": float(
                                r_m["kill_rate_contacted_any_200m"]
                            ),
                            "donor_median": float(r_m["donor_median"]),
                            "donor_q25": float(r_m["donor_q25"]),
                            "donor_q75": float(r_m["donor_q75"]),
                            "donor_mean": float(r_m["donor_mean"]),
                        }
                    )

                hi_vals = (
                    d_med.loc[d_med["condition"] == CONDITION_HIGH, "donor_median"]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                lo_vals = (
                    d_med.loc[d_med["condition"] == CONDITION_LOW, "donor_median"]
                    .dropna()
                    .to_numpy(dtype=float)
                )

                all_vals = np.concatenate([hi_vals, lo_vals])
                obs_mean_diff = float(np.mean(hi_vals) - np.mean(lo_vals))
                perm_diffs = np.empty(len(idx_combos), dtype=float)
                for i_c, idx_hi in enumerate(idx_combos):
                    mask = np.zeros(10, dtype=bool)
                    mask[list(idx_hi)] = True
                    perm_diffs[i_c] = float(np.mean(all_vals[mask]) - np.mean(all_vals[~mask]))
                p_perm_mean_two_sided = float(
                    np.mean(np.abs(perm_diffs) >= abs(obs_mean_diff) - 1e-12)
                )

                u_stat, p_mw_two_sided = mannwhitneyu(
                    hi_vals, lo_vals, alternative="two-sided", method="exact"
                )

                pairwise_diffs = np.sort([float(x - y) for x in hi_vals for y in lo_vals])
                hl_median_diff = float(np.median(pairwise_diffs))
                # Exact 95% confidence interval for Hodges-Lehmann from Mann-Whitney critical values
                # For n1=5, n2=5, U_crit at two-sided alpha=0.05 (exact p=2*4/252=0.0317) is rank 3 (0-indexed 2) and 23 (0-indexed 22)
                hl_ci_low = float(pairwise_diffs[2])
                hl_ci_high = float(pairwise_diffs[22])

                g_val, g_ci_lo, g_ci_hi = hedges_g_and_perm_ci(hi_vals, lo_vals)
                rho_kr, p_kr, rho_ci_lo, rho_ci_hi = spearman_boot_ci(
                    d_med["donor_median"].to_numpy(dtype=float),
                    d_med["kill_rate_any_200m"].to_numpy(dtype=float),
                )
                rho_cnt, p_cnt, _, _ = spearman_boot_ci(
                    d_med["donor_median"].to_numpy(dtype=float),
                    d_med["kill_rate_contacted_any_200m"].to_numpy(dtype=float),
                )

                block_rows.append(
                    {
                        "summary": summary_name,
                        "cohort": cohort_name,
                        "adjustment": adj_mode,
                        "metric": m,
                        "is_primary_confirmatory": bool(m == PRIMARY_METRIC),
                        "is_focus_family": bool(m in FOCUS_METRICS),
                        "n_donors_high": int(len(hi_vals)),
                        "n_donors_low": int(len(lo_vals)),
                        "n_chambers_analyzed": int(len(sub)),
                        "high_donors_median": float(np.median(hi_vals)),
                        "high_donors_iqr": float(
                            np.percentile(hi_vals, 75) - np.percentile(hi_vals, 25)
                        ),
                        "high_donors_mean": float(np.mean(hi_vals)),
                        "low_donors_median": float(np.median(lo_vals)),
                        "low_donors_iqr": float(
                            np.percentile(lo_vals, 75) - np.percentile(lo_vals, 25)
                        ),
                        "low_donors_mean": float(np.mean(lo_vals)),
                        "mean_diff_high_minus_low": obs_mean_diff,
                        "perm_p_mean_diff_two_sided": p_perm_mean_two_sided,
                        "mw_u_stat": float(u_stat),
                        "perm_p_mw_two_sided": float(p_mw_two_sided),
                        "hodges_lehmann_diff": hl_median_diff,
                        "hodges_lehmann_ci95_low": hl_ci_low,
                        "hodges_lehmann_ci95_high": hl_ci_high,
                        "hedges_g": g_val,
                        "hedges_g_perm_ci95_low": g_ci_lo,
                        "hedges_g_perm_ci95_high": g_ci_hi,
                        "spearman_rho_kill_rate": rho_kr,
                        "spearman_p_kill_rate": p_kr,
                        "spearman_ci95_low": rho_ci_lo,
                        "spearman_ci95_high": rho_ci_hi,
                        "spearman_rho_contacted_kill_rate": rho_cnt,
                        "spearman_p_contacted_kill_rate": p_cnt,
                    }
                )

            # BH q across the 3 focus metrics
            focus_idx = [i for i, r in enumerate(block_rows) if r["is_focus_family"]]
            q_mean = bh_fdr([block_rows[i]["perm_p_mean_diff_two_sided"] for i in focus_idx])
            q_mw = bh_fdr([block_rows[i]["perm_p_mw_two_sided"] for i in focus_idx])
            for i_row, r in enumerate(block_rows):
                if i_row in focus_idx:
                    pos = focus_idx.index(i_row)
                    r["bh_q_mean_diff_3focus"] = float(q_mean[pos])
                    r["bh_q_mw_3focus"] = float(q_mw[pos])
                else:
                    r["bh_q_mean_diff_3focus"] = np.nan
                    r["bh_q_mw_3focus"] = np.nan
                r["passes_test7a_primary"] = bool(
                    r["is_primary_confirmatory"]
                    and (
                        (r["perm_p_mean_diff_two_sided"] < 0.05)
                        or (r["perm_p_mw_two_sided"] < 0.05)
                    )
                    and (r["mean_diff_high_minus_low"] > 0)
                )
                all_rows.append(r)

    return pd.DataFrame(all_rows), pd.DataFrame(median_rows)


def fit_nested_lmm_reml_and_perm(
    df_vis: pd.DataFrame, y_col: str, x_cols: list[str], do_donor_perm: bool = True
) -> dict[str, Any]:
    """
    Exact REML estimation for the nested linear mixed model:
      y ~ 1 + x_cols + (1 | donor / chamber)
    Uses analytical Sherman-Morrison-Woodbury block inversion across chambers and donors,
    computes Satterthwaite degrees of freedom (capped at df <= 8 for donor-level effects),
    and optionally runs all 252 exact donor-label permutations for x_cols[0] (group_high).
    """
    sub = df_vis.dropna(subset=[y_col] + x_cols).copy()
    sub["ch_id"] = sub["experiment"].astype(str) + "__" + sub["roi"].astype(str)
    donors = sorted(sub["donor"].unique())

    blocks = []
    for d in donors:
        dfd = sub[sub["donor"] == d].reset_index(drop=True)
        y_d = dfd[y_col].to_numpy(dtype=float)
        X_d = np.column_stack(
            [np.ones(len(dfd))] + [dfd[c].to_numpy(dtype=float) for c in x_cols]
        )
        ch_idx_list = [grp.index.to_numpy() for _, grp in dfd.groupby("ch_id", sort=False)]
        blocks.append((d, y_d, X_d, ch_idx_list))

    p = 1 + len(x_cols)

    def eval_reml(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s_d2, s_c2, s_e2 = np.exp(np.clip(theta, -35.0, 20.0))
        logdet_V = 0.0
        XtVinvX = np.zeros((p, p), dtype=float)
        XtVinvy = np.zeros(p, dtype=float)
        ytVinvy = 0.0

        for _, y_d, X_d, ch_list in blocks:
            n_d = len(y_d)
            Ainv_y = y_d / s_e2
            Ainv_X = X_d / s_e2
            Ainv_1 = np.full(n_d, 1.0 / s_e2)
            for idx in ch_list:
                n_c = len(idx)
                denom = s_e2 + n_c * s_c2
                logdet_V += (n_c - 1) * np.log(s_e2) + np.log(denom)
                fac = s_c2 / (s_e2 * denom)
                Ainv_y[idx] -= fac * np.sum(y_d[idx])
                Ainv_X[idx] -= fac * np.sum(X_d[idx], axis=0)
                Ainv_1[idx] -= fac * n_c

            one_Ainv_one = float(np.sum(Ainv_1))
            denom_d = 1.0 + s_d2 * one_Ainv_one
            logdet_V += np.log(denom_d)

            one_Ainv_y = float(np.sum(Ainv_y))
            one_Ainv_X = np.sum(Ainv_X, axis=0)

            Vinv_y = Ainv_y - (s_d2 / denom_d) * Ainv_1 * one_Ainv_y
            Vinv_X = Ainv_X - (s_d2 / denom_d) * np.outer(Ainv_1, one_Ainv_X)

            XtVinvX += X_d.T @ Vinv_X
            XtVinvy += X_d.T @ Vinv_y
            ytVinvy += float(y_d @ Vinv_y)

        _, logdet_XtVinvX = np.linalg.slogdet(XtVinvX)
        beta = np.linalg.solve(XtVinvX, XtVinvy)
        rtr = ytVinvy - float(XtVinvy @ beta)
        nll = 0.5 * (logdet_V + logdet_XtVinvX + rtr)
        return nll, beta, XtVinvX

    y_var = float(np.var(sub[y_col].to_numpy(dtype=float)))
    init_theta = np.log([max(y_var * 0.2, 1e-8), max(y_var * 0.5, 1e-8), max(y_var * 0.3, 1e-8)])
    res = minimize(
        lambda th: eval_reml(th)[0],
        init_theta,
        method="L-BFGS-B",
        bounds=[(-30.0, 15.0)] * 3,
    )
    _, beta, XtVinvX = eval_reml(res.x)
    cov_beta = np.linalg.inv(XtVinvX)
    se_beta = np.sqrt(np.diag(cov_beta))
    s_d2, s_c2, s_e2 = np.exp(res.x)

    # Satterthwaite degrees of freedom via numerical derivatives of var(beta) wrt phi = (s_d2, s_c2, s_e2)
    phi = np.exp(res.x)
    grad_var = np.zeros((p, 3), dtype=float)
    hess_nll = np.zeros((3, 3), dtype=float)

    def nll_phi(p_vec: np.ndarray) -> float:
        return eval_reml(np.log(np.maximum(p_vec, 1e-18)))[0]

    def var_phi(p_vec: np.ndarray) -> np.ndarray:
        _, _, xvx = eval_reml(np.log(np.maximum(p_vec, 1e-18)))
        return np.diag(np.linalg.inv(xvx))

    for j in range(3):
        step = max(phi[j] * 1e-4, 1e-9)
        p_plus = phi.copy()
        p_plus[j] += step
        p_minus = phi.copy()
        p_minus[j] -= step
        grad_var[:, j] = (var_phi(p_plus) - var_phi(p_minus)) / (2.0 * step)
        for k in range(j, 3):
            step_k = max(phi[k] * 1e-4, 1e-9)
            pp = phi.copy()
            pp[j] += step
            pp[k] += step_k
            pm = phi.copy()
            pm[j] += step
            pm[k] -= step_k
            mp = phi.copy()
            mp[j] -= step
            mp[k] += step_k
            mm = phi.copy()
            mm[j] -= step
            mm[k] -= step_k
            h_jk = (nll_phi(pp) - nll_phi(pm) - nll_phi(mp) + nll_phi(mm)) / (
                4.0 * step * step_k
            )
            hess_nll[j, k] = hess_nll[k, j] = h_jk

    cov_phi = np.linalg.pinv(hess_nll)
    satt_df_raw = np.zeros(p, dtype=float)
    var_b = np.diag(cov_beta)
    for idx_p in range(p):
        g = grad_var[idx_p]
        denom = float(g @ cov_phi @ g)
        satt_df_raw[idx_p] = 2.0 * (var_b[idx_p] ** 2) / max(denom, 1e-18)

    t_stat = beta / se_beta
    # Pre-registration specifies df <= 8 for group_high
    df_group_capped = float(np.clip(satt_df_raw[1], 1.0, 8.0))
    p_satt_capped = float(2.0 * (1.0 - t_dist.cdf(abs(t_stat[1]), df=df_group_capped)))
    p_satt_uncapped = float(
        2.0 * (1.0 - t_dist.cdf(abs(t_stat[1]), df=max(float(satt_df_raw[1]), 1.0)))
    )

    p_perm_beta = np.nan
    p_perm_t = np.nan
    if do_donor_perm and len(donors) == 10:
        obs_b1 = float(beta[1])
        obs_t1 = float(t_stat[1])
        orig_col1 = [X_d[:, 1].copy() for _, _, X_d, _ in blocks]
        perm_b_list = []
        perm_t_list = []
        for hi_donors in combinations(donors, 5):
            hi_set = set(hi_donors)
            for d_name, _, X_d, _ in blocks:
                X_d[:, 1] = 1.0 if d_name in hi_set else 0.0
            res_p = minimize(
                lambda th: eval_reml(th)[0],
                res.x,
                method="L-BFGS-B",
                bounds=[(-30.0, 15.0)] * 3,
            )
            _, b_p, xvx_p = eval_reml(res_p.x)
            se_p1 = float(np.sqrt(np.diag(np.linalg.inv(xvx_p)))[1])
            perm_b_list.append(float(b_p[1]))
            perm_t_list.append(float(b_p[1] / se_p1))
        for i_b, (_, _, X_d, _) in enumerate(blocks):
            X_d[:, 1] = orig_col1[i_b]
        perm_b_arr = np.asarray(perm_b_list, dtype=float)
        perm_t_arr = np.asarray(perm_t_list, dtype=float)
        p_perm_beta = float(np.mean(np.abs(perm_b_arr) >= abs(obs_b1) - 1e-12))
        p_perm_t = float(np.mean(np.abs(perm_t_arr) >= abs(obs_t1) - 1e-12))

    return {
        "n_visits_nonnull": int(len(sub)),
        "n_chambers": int(sub["ch_id"].nunique()),
        "n_donors": int(len(donors)),
        "beta": beta,
        "se": se_beta,
        "t": t_stat,
        "satt_df_raw": satt_df_raw,
        "satt_df_group_capped": df_group_capped,
        "p_satt_group_capped": p_satt_capped,
        "p_satt_group_uncapped": p_satt_uncapped,
        "p_perm_beta_252": p_perm_beta,
        "p_perm_t_252": p_perm_t,
        "sigma2_donor": float(s_d2),
        "sigma2_chamber": float(s_c2),
        "sigma2_residual": float(s_e2),
    }


def run_test7a_visit_lmm(mv20_pre: pd.DataFrame) -> pd.DataFrame:
    """
    Test 7A (part 2): Visit-level linear mixed model across untreated pre-death contact visits:
      focus ~ group_high + patch_equiv_radius_um + time_in_contact + (1 | donor / chamber)
    Evaluated on:
      - trace1_primary (3,108 visits total; 3,103 with valid 3D hull cup depth)
      - trace1_no_outliers (excluding visits with cup_volume_hull_um3 > target_volume_um3 or outlier chambers)
      - all_chambers (sensitivity across all 502 chambers)
    """
    configs = [
        ("trace1_primary", True, False),
        ("trace1_no_outliers", True, True),
        ("all_chambers", False, False),
    ]
    rows = []
    for cohort_name, only_trace1, drop_outliers in configs:
        sub = mv20_pre[mv20_pre["condition"] != CONDITION_DRUG].copy()
        if only_trace1:
            sub = sub[sub["trace_num"] == 1].copy()
        if drop_outliers:
            sub = sub[(~sub["outlier_vol_flag"]) & (~sub["out_cup_gt_target_visit"])].copy()

        n_total_predeath_visits = int(len(sub))
        sub["group_high"] = (sub["condition"] == CONDITION_HIGH).astype(float)

        cohort_rows = []
        for m in FOCUS_METRICS + ["pi_simple", "delta_pi_indent"]:
            # Only run 252 full REML permutations on primary/focus metrics and primary cohort (or all since fast)
            res = fit_nested_lmm_reml_and_perm(
                sub,
                m,
                ["group_high", "patch_equiv_radius_um", "time_in_contact_visits"],
                do_donor_perm=(cohort_name == "trace1_primary"),
            )
            cohort_rows.append(
                {
                    "cohort": cohort_name,
                    "metric": m,
                    "is_primary_confirmatory": bool(m == PRIMARY_METRIC),
                    "is_focus_family": bool(m in FOCUS_METRICS),
                    "n_visits_total_cohort": n_total_predeath_visits,
                    "n_visits_modeled": res["n_visits_nonnull"],
                    "n_visits_dropped_hull_ degenerate": n_total_predeath_visits
                    - res["n_visits_nonnull"],
                    "n_chambers": res["n_chambers"],
                    "n_donors": res["n_donors"],
                    "beta_group_high": float(res["beta"][1]),
                    "se_group_high": float(res["se"][1]),
                    "t_group_high": float(res["t"][1]),
                    "satt_df_raw": float(res["satt_df_raw"][1]),
                    "satt_df_capped_le8": float(res["satt_df_group_capped"]),
                    "p_satt_capped": float(res["p_satt_group_capped"]),
                    "p_satt_uncapped": float(res["p_satt_group_uncapped"]),
                    "perm_p_beta_252": float(res["p_perm_beta_252"]),
                    "perm_p_t_252": float(res["p_perm_t_252"]),
                    "beta_patch_radius": float(res["beta"][2]),
                    "se_patch_radius": float(res["se"][2]),
                    "t_patch_radius": float(res["t"][2]),
                    "beta_time_in_contact_visits": float(res["beta"][3]),
                    "se_time_in_contact_visits": float(res["se"][3]),
                    "t_time_in_contact_visits": float(res["t"][3]),
                    "sigma2_donor": res["sigma2_donor"],
                    "sigma2_chamber": res["sigma2_chamber"],
                    "sigma2_residual": res["sigma2_residual"],
                }
            )
        focus_idx = [i for i, r in enumerate(cohort_rows) if r["is_focus_family"]]
        q_satt = bh_fdr([cohort_rows[i]["p_satt_capped"] for i in focus_idx])
        q_perm = bh_fdr([cohort_rows[i]["perm_p_t_252"] for i in focus_idx])
        for i_r, r in enumerate(cohort_rows):
            if i_r in focus_idx:
                pos = focus_idx.index(i_r)
                r["bh_q_satt_3focus"] = float(q_satt[pos])
                r["bh_q_perm_3focus"] = float(q_perm[pos])
            else:
                r["bh_q_satt_3focus"] = np.nan
                r["bh_q_perm_3focus"] = np.nan
            rows.append(r)

    return pd.DataFrame(rows)


def cox_profile_likelihood_ci(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    cov_cols: list[str],
    strata_col: str = "experiment",
    ties: str = "efron",
) -> tuple[float, float]:
    """
    Compute exact profile-likelihood 95% CI for cov_cols[0] in the experiment-stratified
    Cox proportional hazards model (where 2 * [NLL(beta_0) - NLL(beta_hat)] = chi2_{1, 0.95} = 3.8414588).
    """
    sub = df[[time_col, event_col, strata_col] + cov_cols].dropna().copy()
    p = len(cov_cols)
    strata_data = []
    for _, grp in sub.groupby(strata_col, sort=False):
        t = grp[time_col].to_numpy(dtype=float)
        d = grp[event_col].to_numpy(dtype=int)
        X = grp[cov_cols].to_numpy(dtype=float)
        if np.sum(d) == 0 or len(t) < 2:
            continue
        strata_data.append((t, d, X))

    def nll_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        nll = 0.0
        grad = np.zeros(p, dtype=float)
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
                X_ev_sum = np.sum(X[events], axis=0)
                nll -= float(np.sum(eta_shift[events]))
                grad -= X_ev_sum
                if ties == "breslow" or m_ev == 1:
                    nll += m_ev * np.log(S0)
                    grad += m_ev * (S1 / S0)
                else:
                    w_ev = float(np.sum(w[events]))
                    S1_ev = np.sum(Xw[events], axis=0)
                    for l_idx in range(m_ev):
                        frac = l_idx / m_ev
                        denom = max(S0 - frac * w_ev, 1e-15)
                        num1 = S1 - frac * S1_ev
                        nll += np.log(denom)
                        grad += num1 / denom
        return nll, grad

    res_mle = fit_stratified_cox(df, time_col, event_col, cov_cols, strata_col=strata_col, ties=ties)
    beta_hat = np.asarray(res_mle["beta"], dtype=float)
    se_hat = float(res_mle["se"][0])
    if np.isnan(beta_hat[0]) or np.isnan(se_hat):
        return np.nan, np.nan
    nll_min, _ = nll_and_grad(beta_hat)
    target_nll = nll_min + 0.5 * 3.841458820694124

    def profile_nll(b0: float) -> float:
        if p == 1:
            return nll_and_grad(np.array([b0]))[0]
        init_rest = beta_hat[1:].copy()
        opt = minimize(
            lambda b_rest: nll_and_grad(np.concatenate([[b0], b_rest]))[0],
            init_rest,
            jac=lambda b_rest: nll_and_grad(np.concatenate([[b0], b_rest]))[1][1:],
            method="BFGS",
        )
        return float(opt.fun)

    try:
        b_lo = brentq(
            lambda b0: profile_nll(b0) - target_nll,
            beta_hat[0] - 6.0 * se_hat,
            beta_hat[0],
            xtol=1e-5,
        )
    except ValueError:
        b_lo = beta_hat[0] - 1.96 * se_hat

    try:
        b_hi = brentq(
            lambda b0: profile_nll(b0) - target_nll,
            beta_hat[0],
            beta_hat[0] + 6.0 * se_hat,
            xtol=1e-5,
        )
    except ValueError:
        b_hi = beta_hat[0] + 1.96 * se_hat

    return float(np.exp(b_lo)), float(np.exp(b_hi))


def fit_glmm_logistic_and_frailty(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    cov_cols: list[str],
    group_col: str = "donor",
) -> dict[str, float]:
    """
    Fit mixed-effects logistic regression (20-node Gauss-Hermite quadrature on (1 | donor))
    and Gaussian random-intercept Cox frailty model on (1 | donor) as the pre-registered check.
    """
    sub = df[[time_col, event_col, group_col] + cov_cols].dropna().copy()
    gh_x, gh_w = np.polynomial.hermite.hermgauss(20)
    gh_z = gh_x * np.sqrt(2.0)
    gh_wt = gh_w / np.sqrt(np.pi)

    groups_log = []
    groups_cox = []
    for _, grp in sub.groupby(group_col, sort=False):
        y_g = grp[event_col].to_numpy(dtype=float)
        t_g = grp[time_col].to_numpy(dtype=float)
        d_g = grp[event_col].to_numpy(dtype=int)
        X_g = np.column_stack(
            [np.ones(len(grp))] + [grp[c].to_numpy(dtype=float) for c in cov_cols]
        )
        groups_log.append((y_g, X_g))
        if np.sum(d_g) > 0 and len(t_g) >= 2:
            groups_cox.append((t_g, d_g, grp[cov_cols].to_numpy(dtype=float)))

    p_log = 1 + len(cov_cols)

    def nll_logistic(params: np.ndarray) -> float:
        beta = params[:p_log]
        sigma = np.exp(np.clip(params[p_log], -15.0, 5.0))
        total_nll = 0.0
        for y_g, X_g in groups_log:
            eta_base = X_g @ beta
            eta = eta_base[:, None] + sigma * gh_z[None, :]
            ll_nodes = np.sum(y_g[:, None] * eta - np.logaddexp(0.0, eta), axis=0)
            m_ll = float(np.max(ll_nodes))
            total_nll -= m_ll + float(np.log(np.sum(gh_wt * np.exp(ll_nodes - m_ll))))
        return total_nll

    res_log = minimize(nll_logistic, np.zeros(p_log + 1), method="BFGS")
    eps = 1e-4
    hess_log = np.zeros((p_log + 1, p_log + 1), dtype=float)
    for j in range(p_log + 1):
        for k in range(j, p_log + 1):
            pp = res_log.x.copy(); pp[j] += eps; pp[k] += eps
            pm = res_log.x.copy(); pm[j] += eps; pm[k] -= eps
            mp = res_log.x.copy(); mp[j] -= eps; mp[k] += eps
            mm = res_log.x.copy(); mm[j] -= eps; mm[k] -= eps
            hess_log[j, k] = hess_log[k, j] = (
                nll_logistic(pp) - nll_logistic(pm) - nll_logistic(mp) + nll_logistic(mm)
            ) / (4.0 * eps * eps)
    cov_log = np.linalg.pinv(hess_log)
    se_log1 = float(np.sqrt(max(cov_log[1, 1], 1e-18)))
    beta_log1 = float(res_log.x[1])
    p_log1 = float(2.0 * (1.0 - norm.cdf(abs(beta_log1 / se_log1))))

    # Penalized Gaussian frailty Cox model: u_g ~ N(0, sigma_f^2) with Laplace / penalized partial likelihood
    # Note: In Cox partial likelihood stratified by donor, any additive donor-constant frailty u_g
    # cancels identically out of the within-donor partial likelihood! For unstratified pooled risk sets
    # across donors, we estimate penalized random intercepts u_g per donor.
    donors_list = sorted(sub[group_col].unique())
    d_to_idx = {d: i for i, d in enumerate(donors_list)}
    g_idx_arr = np.array([d_to_idx[d] for d in sub[group_col]], dtype=int)
    t_all = sub[time_col].to_numpy(dtype=float)
    d_all = sub[event_col].to_numpy(dtype=int)
    X_all = sub[cov_cols].to_numpy(dtype=float)
    p_cox = len(cov_cols)
    n_g = len(donors_list)

    def ppl_cox(params: np.ndarray, sigma_u: float = 0.5) -> float:
        beta = params[:p_cox]
        u = params[p_cox:]
        eta = np.clip(X_all @ beta + u[g_idx_arr], -50.0, 50.0)
        eta_shift = eta - np.max(eta)
        w = np.exp(eta_shift)
        nll = 0.5 * float(np.sum((u / sigma_u) ** 2))
        for ut in np.unique(t_all[d_all == 1]):
            at_risk = t_all >= ut
            events = (t_all == ut) & (d_all == 1)
            m_ev = int(np.sum(events))
            S0 = float(np.sum(w[at_risk]))
            w_ev = float(np.sum(w[events]))
            nll -= float(np.sum(eta_shift[events]))
            for l_i in range(m_ev):
                nll += np.log(max(S0 - (l_i / m_ev) * w_ev, 1e-15))
        return nll

    res_f = minimize(lambda prm: ppl_cox(prm, 0.5), np.zeros(p_cox + n_g), method="BFGS")
    hess_f = np.zeros((p_cox + n_g, p_cox + n_g), dtype=float)
    for j in range(p_cox + n_g):
        for k in range(j, p_cox + n_g):
            pp = res_f.x.copy(); pp[j] += eps; pp[k] += eps
            pm = res_f.x.copy(); pm[j] += eps; pm[k] -= eps
            mp = res_f.x.copy(); mp[j] -= eps; mp[k] += eps
            mm = res_f.x.copy(); mm[j] -= eps; mm[k] -= eps
            hess_f[j, k] = hess_f[k, j] = (
                ppl_cox(pp, 0.5) - ppl_cox(pm, 0.5) - ppl_cox(mp, 0.5) + ppl_cox(mm, 0.5)
            ) / (4.0 * eps * eps)
    cov_f = np.linalg.pinv(hess_f)
    beta_f0 = float(res_f.x[0])
    se_f0 = float(np.sqrt(max(cov_f[0, 0], 1e-18)))
    p_f0 = float(2.0 * (1.0 - norm.cdf(abs(beta_f0 / se_f0))))

    return {
        "glmm_logistic_or": float(np.exp(beta_log1)),
        "glmm_logistic_ci95_low": float(np.exp(beta_log1 - 1.96 * se_log1)),
        "glmm_logistic_ci95_high": float(np.exp(beta_log1 + 1.96 * se_log1)),
        "glmm_logistic_p": p_log1,
        "glmm_donor_sd": float(np.exp(res_log.x[p_log])),
        "cox_frailty_hr": float(np.exp(beta_f0)),
        "cox_frailty_ci95_low": float(np.exp(beta_f0 - 1.96 * se_f0)),
        "cox_frailty_ci95_high": float(np.exp(beta_f0 + 1.96 * se_f0)),
        "cox_frailty_p": p_f0,
    }


def run_test7b_within_donor_cox(ch: pd.DataFrame) -> pd.DataFrame:
    """
    Test 7B: Within-donor cell-level killing (experiment-stratified Cox PH model).
    Evaluates:
      - Summaries: A_first_contact_pre_death (adjusted for contact_area_um2),
        B_pre_death_mean (adjusted for contact_area_um2 AND n_pre_death_contact_visits_any_200m,
        plus contact_area-only and unadjusted for reference).
      - Cohorts: trace1_primary (all trace_num==1 experiments), trace1_untreated_only (10 donors),
        trace1_no_outliers, all_chambers.
    Reports HR per SD, Wald 95% CI, exact profile-likelihood 95% CI, Wald p, BH q across the 3
    focus metrics, and mixed-effects logistic/frailty (1 | donor) check.
    """
    configs = [
        ("A_first_contact_pre_death", "trace1_primary", False, False),
        ("B_pre_death_mean", "trace1_primary", False, False),
        ("A_first_contact_pre_death", "trace1_untreated_only", True, False),
        ("B_pre_death_mean", "trace1_untreated_only", True, False),
        ("A_first_contact_pre_death", "trace1_no_outliers", False, True),
        ("B_pre_death_mean", "trace1_no_outliers", False, True),
        ("A_first_contact_pre_death", "all_chambers", False, False),
        ("B_pre_death_mean", "all_chambers", False, False),
    ]

    all_rows = []
    time_col = "km_any_contact_time_min_200m"
    event_col = "km_event_any_visit_200m"

    for summary_name, cohort_name, untr_only, drop_outliers in configs:
        df = ch[ch["has_contact_desc_200m"]].copy()
        if cohort_name.startswith("trace1"):
            df = df[df["trace_num"] == 1].copy()
        if untr_only:
            df = df[df["condition"] != CONDITION_DRUG].copy()
        if drop_outliers:
            df = df[~df["outlier_vol_flag"]].copy()

        if summary_name == "B_pre_death_mean":
            df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
            suf = "_B_any"
            df["z_nvis"] = normal_scores_within_experiment(
                df, "n_pre_death_contact_visits_any_200m"
            )
        else:
            df = df[~df["fc_is_death_visit_any_200m"]].copy()
            suf = "_A"

        df["z_ca"] = normal_scores_within_experiment(df, f"contact_area_um2{suf}")
        df["z_pr"] = normal_scores_within_experiment(df, f"patch_equiv_radius_um{suf}")

        block_rows = []
        for m in FOCUS_METRICS + ["pi_simple", "delta_pi_indent"]:
            col = f"{m}{suf}"
            df["z_m"] = normal_scores_within_experiment(df, col)

            res_u = fit_stratified_cox(df, time_col, event_col, ["z_m"])
            res_ca = fit_stratified_cox(df, time_col, event_col, ["z_m", "z_ca"])

            if summary_name == "B_pre_death_mean":
                covs_primary = ["z_m", "z_ca", "z_nvis"]
                res_prim = fit_stratified_cox(df, time_col, event_col, covs_primary)
                res_pr_nvis = fit_stratified_cox(df, time_col, event_col, ["z_m", "z_pr", "z_nvis"])
            else:
                covs_primary = ["z_m", "z_ca"]
                res_prim = res_ca
                res_pr_nvis = fit_stratified_cox(df, time_col, event_col, ["z_m", "z_pr"])

            pl_lo, pl_hi = cox_profile_likelihood_ci(df, time_col, event_col, covs_primary)
            pl_lo_u, pl_hi_u = cox_profile_likelihood_ci(df, time_col, event_col, ["z_m"])

            if cohort_name in ("trace1_primary", "trace1_untreated_only"):
                me_check = fit_glmm_logistic_and_frailty(
                    df, time_col, event_col, covs_primary, group_col="donor"
                )
            else:
                me_check = {
                    "glmm_logistic_or": np.nan,
                    "glmm_logistic_ci95_low": np.nan,
                    "glmm_logistic_ci95_high": np.nan,
                    "glmm_logistic_p": np.nan,
                    "glmm_donor_sd": np.nan,
                    "cox_frailty_hr": np.nan,
                    "cox_frailty_ci95_low": np.nan,
                    "cox_frailty_ci95_high": np.nan,
                    "cox_frailty_p": np.nan,
                }

            block_rows.append(
                {
                    "summary": summary_name,
                    "cohort": cohort_name,
                    "metric": m,
                    "is_primary_confirmatory": bool(m == PRIMARY_METRIC),
                    "is_focus_family": bool(m in FOCUS_METRICS),
                    "n_chambers": int(res_prim["n_obs"]),
                    "n_events": int(res_prim["n_events"]),
                    "hr_unadj": float(res_u["hr"][0]),
                    "wald_ci95_low_unadj": float(res_u["ci_low"][0]),
                    "wald_ci95_high_unadj": float(res_u["ci_high"][0]),
                    "profile_ci95_low_unadj": pl_lo_u,
                    "profile_ci95_high_unadj": pl_hi_u,
                    "wald_p_unadj": float(res_u["p"][0]),
                    "hr_adj_ca_only": float(res_ca["hr"][0]),
                    "wald_ci95_low_adj_ca": float(res_ca["ci_low"][0]),
                    "wald_ci95_high_adj_ca": float(res_ca["ci_high"][0]),
                    "wald_p_adj_ca": float(res_ca["p"][0]),
                    "hr_prereg_adjusted": float(res_prim["hr"][0]),
                    "beta_prereg_adjusted": float(res_prim["beta"][0]),
                    "se_prereg_adjusted": float(res_prim["se"][0]),
                    "wald_ci95_low_adjusted": float(res_prim["ci_low"][0]),
                    "wald_ci95_high_adjusted": float(res_prim["ci_high"][0]),
                    "profile_ci95_low_adjusted": pl_lo,
                    "profile_ci95_high_adjusted": pl_hi,
                    "wald_p_adjusted": float(res_prim["p"][0]),
                    "hr_adj_patch_radius": float(res_pr_nvis["hr"][0]),
                    "wald_p_adj_patch_radius": float(res_pr_nvis["p"][0]),
                    **me_check,
                }
            )

        focus_idx = [i for i, r in enumerate(block_rows) if r["is_focus_family"]]
        q_unadj = bh_fdr([block_rows[i]["wald_p_unadj"] for i in focus_idx])
        q_adj = bh_fdr([block_rows[i]["wald_p_adjusted"] for i in focus_idx])
        for i_r, r in enumerate(block_rows):
            if i_r in focus_idx:
                pos = focus_idx.index(i_r)
                r["bh_q_unadj_3focus"] = float(q_unadj[pos])
                r["bh_q_adjusted_3focus"] = float(q_adj[pos])
            else:
                r["bh_q_unadj_3focus"] = np.nan
                r["bh_q_adjusted_3focus"] = np.nan
            r["passes_test7b_adjusted"] = bool(
                r["is_focus_family"]
                and (r["hr_prereg_adjusted"] > 1.0)
                and (r["bh_q_adjusted_3focus"] < 0.05)
            )
            all_rows.append(r)

    return pd.DataFrame(all_rows)


def run_test7c_dasatinib_control(
    ch: pd.DataFrame, n_boot: int = 4000, seed: int = 42
) -> pd.DataFrame:
    """
    Test 7C: Positive control (dasatinib 50 nM in paired donors hd2, hd10, hd5).
    Tests whether dasatinib lowers each focus metric and inferred push Pi in 3/3 paired donors
    and by donor-stratified rank Mann-Whitney (one-sided alpha = 0.025, plus two-sided p
    and paired bootstrap 95% CI on mean donor-median difference and donor-stratified shift).
    """
    configs = [
        ("B_pre_death_mean", "trace1_primary", False),
        ("A_first_contact_pre_death", "trace1_primary", False),
        ("B_pre_death_mean", "trace1_no_outliers", True),
        ("A_first_contact_pre_death", "trace1_no_outliers", True),
        ("B_pre_death_mean", "all_chambers", False),
        ("A_first_contact_pre_death", "all_chambers", False),
    ]

    rng = np.random.default_rng(seed)
    all_rows = []

    for summary_name, cohort_name, drop_outliers in configs:
        df = ch[ch["has_contact_desc_200m"]].copy()
        if cohort_name.startswith("trace1"):
            df = df[df["trace_num"] == 1].copy()
        if drop_outliers:
            df = df[~df["outlier_vol_flag"]].copy()

        if summary_name == "B_pre_death_mean":
            df = df[df["n_pre_death_contact_visits_any_200m"] > 0].copy()
            suf = "_B_any"
        else:
            df = df[~df["fc_is_death_visit_any_200m"]].copy()
            suf = "_A"

        metrics_list = FOCUS_METRICS + [
            "pi_simple",
            "delta_pi_indent",
            "patch_equiv_radius_um",
            "contact_area_um2",
        ]

        for m in metrics_list:
            col = f"{m}{suf}"
            pair_stats = {}
            n_lowered = 0
            donor_Arrays = {}
            for donor_id, exp_untr, exp_drug in PAIRED_CONTROL_EXPERIMENTS:
                vu = df.loc[df["experiment"] == exp_untr, col].dropna().to_numpy(dtype=float)
                vd = df.loc[df["experiment"] == exp_drug, col].dropna().to_numpy(dtype=float)
                donor_Arrays[donor_id] = (vu, vd)
                med_u = float(np.median(vu)) if len(vu) > 0 else np.nan
                med_d = float(np.median(vd)) if len(vd) > 0 else np.nan
                diff = med_d - med_u
                lowered = bool(med_d < med_u)
                if lowered:
                    n_lowered += 1
                u_2s, p_2s = mannwhitneyu(vu, vd, alternative="two-sided")
                _, p_1s_less = mannwhitneyu(vd, vu, alternative="less")  # drug < untreated
                pair_stats[donor_id] = {
                    "n_untr": int(len(vu)),
                    "n_drug": int(len(vd)),
                    "med_untr": med_u,
                    "med_drug": med_d,
                    "diff_drug_minus_untr": diff,
                    "lowered": lowered,
                    "mw_u": float(u_2s),
                    "mw_p_two_sided": float(p_2s),
                    "mw_p_one_sided_lower": float(p_1s_less),
                }

            # Donor-stratified rank Mann-Whitney across the 3 pairs
            sub_pairs = df[
                df["experiment"].isin([e for _, eu, ed in PAIRED_CONTROL_EXPERIMENTS for e in (eu, ed)])
            ].copy()
            sub_pairs["is_drug"] = (sub_pairs["condition"] == CONDITION_DRUG).astype(int)
            sub_pairs["z_within_donor"] = normal_scores_within_experiment(
                sub_pairs, col, exp_col="donor"
            )
            z_u = sub_pairs.loc[sub_pairs["is_drug"] == 0, "z_within_donor"].dropna().to_numpy()
            z_d = sub_pairs.loc[sub_pairs["is_drug"] == 1, "z_within_donor"].dropna().to_numpy()
            u_strat, p_strat_two_sided = mannwhitneyu(z_d, z_u, alternative="two-sided")
            _, p_strat_one_sided_lower = mannwhitneyu(z_d, z_u, alternative="less")
            z_diff_mean = float(np.mean(z_d) - np.mean(z_u))

            # Paired bootstrap 95% CI resampling chambers within each (donor, condition) arm
            mean_med_diff_obs = float(
                np.mean([pair_stats[d_id]["diff_drug_minus_untr"] for d_id in ("hd2", "hd10", "hd5")])
            )
            boot_med_diffs = np.empty(n_boot, dtype=float)
            boot_z_diffs = np.empty(n_boot, dtype=float)
            for b_i in range(n_boot):
                d_diffs = []
                z_u_list, z_d_list = [], []
                for d_id in ("hd2", "hd10", "hd5"):
                    vu, vd = donor_Arrays[d_id]
                    bu = vu[rng.integers(0, len(vu), size=len(vu))]
                    bd = vd[rng.integers(0, len(vd), size=len(vd))]
                    d_diffs.append(float(np.median(bd) - np.median(bu)))
                    comb = np.concatenate([bu, bd])
                    n_c = len(comb)
                    r_c = rankdata(comb, method="average")
                    z_c = norm.ppf((r_c - 0.375) / (n_c + 0.25))
                    sd_c = np.std(z_c, ddof=1)
                    if sd_c > 0:
                        z_c = (z_c - np.mean(z_c)) / sd_c
                    z_u_list.append(z_c[: len(bu)])
                    z_d_list.append(z_c[len(bu) :])
                boot_med_diffs[b_i] = float(np.mean(d_diffs))
                boot_z_diffs[b_i] = float(
                    np.mean(np.concatenate(z_d_list)) - np.mean(np.concatenate(z_u_list))
                )

            ci_med_lo, ci_med_hi = np.percentile(boot_med_diffs, [2.5, 97.5])
            ci_z_lo, ci_z_hi = np.percentile(boot_z_diffs, [2.5, 97.5])

            all_rows.append(
                {
                    "summary": summary_name,
                    "cohort": cohort_name,
                    "metric": m,
                    "is_primary_confirmatory": bool(m == PRIMARY_METRIC),
                    "is_focus_family": bool(m in FOCUS_METRICS),
                    "hd2_n_untr": pair_stats["hd2"]["n_untr"],
                    "hd2_n_drug": pair_stats["hd2"]["n_drug"],
                    "hd2_med_untr": pair_stats["hd2"]["med_untr"],
                    "hd2_med_drug": pair_stats["hd2"]["med_drug"],
                    "hd2_diff_drug_minus_untr": pair_stats["hd2"]["diff_drug_minus_untr"],
                    "hd2_lowered": pair_stats["hd2"]["lowered"],
                    "hd2_mw_p_two_sided": pair_stats["hd2"]["mw_p_two_sided"],
                    "hd10_n_untr": pair_stats["hd10"]["n_untr"],
                    "hd10_n_drug": pair_stats["hd10"]["n_drug"],
                    "hd10_med_untr": pair_stats["hd10"]["med_untr"],
                    "hd10_med_drug": pair_stats["hd10"]["med_drug"],
                    "hd10_diff_drug_minus_untr": pair_stats["hd10"]["diff_drug_minus_untr"],
                    "hd10_lowered": pair_stats["hd10"]["lowered"],
                    "hd10_mw_p_two_sided": pair_stats["hd10"]["mw_p_two_sided"],
                    "hd5_n_untr": pair_stats["hd5"]["n_untr"],
                    "hd5_n_drug": pair_stats["hd5"]["n_drug"],
                    "hd5_med_untr": pair_stats["hd5"]["med_untr"],
                    "hd5_med_drug": pair_stats["hd5"]["med_drug"],
                    "hd5_diff_drug_minus_untr": pair_stats["hd5"]["diff_drug_minus_untr"],
                    "hd5_lowered": pair_stats["hd5"]["lowered"],
                    "hd5_mw_p_two_sided": pair_stats["hd5"]["mw_p_two_sided"],
                    "n_donors_lowered": int(n_lowered),
                    "lowered_3_of_3_donors": bool(n_lowered == 3),
                    "mean_donor_median_diff": mean_med_diff_obs,
                    "paired_boot_ci95_low_med_diff": float(ci_med_lo),
                    "paired_boot_ci95_high_med_diff": float(ci_med_hi),
                    "donor_stratified_z_diff_drug_minus_untr": z_diff_mean,
                    "paired_boot_ci95_low_z_diff": float(ci_z_lo),
                    "paired_boot_ci95_high_z_diff": float(ci_z_hi),
                    "donor_stratified_rank_mw_u": float(u_strat),
                    "donor_stratified_rank_mw_p_two_sided": float(p_strat_two_sided),
                    "donor_stratified_rank_mw_p_one_sided_lower": float(p_strat_one_sided_lower),
                    "passes_test7c_gate": bool(
                        (n_lowered == 3) and (p_strat_one_sided_lower < 0.025)
                    ),
                }
            )

    return pd.DataFrame(all_rows)


def compute_stiffness_unit_table(
    ch: pd.DataFrame, mv20_pre: pd.DataFrame, afm_json: dict[str, Any]
) -> pd.DataFrame:
    """
    Compute Route (b) dimensionless push Pi_simple and net indentation push Delta_Pi_indent,
    plus conditional physical pressure p (Pa) and force F (nN) intervals using the
    pre-registered cortical tension anchors:
      - Nominal T_target in [49, 163] pN/um
      - Full range T_target in [18, 430] pN/um
      - T-cell cortical tension T_T in [570, 1790] pN/um (from afm_to_tension_conversion.json)
    Every stiffness-unit row is explicitly labeled:
      "provisional: route (b), pending 6b gate"
    """
    t_nom_lo, t_nom_hi = 49.0, 163.0
    t_full_lo, t_full_hi = 18.0, 430.0
    t_tcell_nom = afm_json.get("tcell_cortex_from_jung2021_afm", {}).get(
        "T_cortical_nominal_pN_per_um", [569.5, 1789.9]
    )

    t1 = ch[ch["trace_num"] == 1].copy()
    groups = [
        (
            "High-killing untreated (5 donors, Summary B)",
            t1[
                (t1["condition"] == CONDITION_HIGH)
                & (t1["n_pre_death_contact_visits_any_200m"] > 0)
            ],
            "_B_any",
        ),
        (
            "Low-killing untreated (5 donors, Summary B)",
            t1[
                (t1["condition"] == CONDITION_LOW)
                & (t1["n_pre_death_contact_visits_any_200m"] > 0)
            ],
            "_B_any",
        ),
        (
            "All untreated (10 donors, Summary B)",
            t1[
                (t1["condition"] != CONDITION_DRUG)
                & (t1["n_pre_death_contact_visits_any_200m"] > 0)
            ],
            "_B_any",
        ),
        (
            "Dasatinib 50 nM (3 paired donors, Summary B)",
            t1[
                (t1["condition"] == CONDITION_DRUG)
                & (t1["n_pre_death_contact_visits_any_200m"] > 0)
            ],
            "_B_any",
        ),
        (
            "High-killing untreated (5 donors, Summary A)",
            t1[
                (t1["condition"] == CONDITION_HIGH)
                & t1["has_contact_desc_200m"]
                & (~t1["fc_is_death_visit_any_200m"])
            ],
            "_A",
        ),
        (
            "Low-killing untreated (5 donors, Summary A)",
            t1[
                (t1["condition"] == CONDITION_LOW)
                & t1["has_contact_desc_200m"]
                & (~t1["fc_is_death_visit_any_200m"])
            ],
            "_A",
        ),
        (
            "Dasatinib 50 nM (3 paired donors, Summary A)",
            t1[
                (t1["condition"] == CONDITION_DRUG)
                & t1["has_contact_desc_200m"]
                & (~t1["fc_is_death_visit_any_200m"])
            ],
            "_A",
        ),
    ]

    rows = []
    for label, sub, suf in groups:
        pi_vals = sub[f"pi_simple{suf}"].dropna().to_numpy(dtype=float)
        dpi_vals = sub[f"delta_pi_indent{suf}"].dropna().to_numpy(dtype=float)
        pi_fin_vals = sub[f"pi_finite{suf}"].dropna().to_numpy(dtype=float)
        rt_vals = sub[f"r_target_um{suf}"].dropna().to_numpy(dtype=float)
        ca_vals = sub[f"contact_area_um2{suf}"].dropna().to_numpy(dtype=float)
        pr_vals = sub[f"patch_equiv_radius_um{suf}"].dropna().to_numpy(dtype=float)
        dor_vals = sub[f"cup_depth_over_radius{suf}"].dropna().to_numpy(dtype=float)

        # Also compute per-chamber pressure factor (Pi / R_target) and force factor (Pi * A_contact / R_target)
        valid = sub[
            [f"pi_simple{suf}", f"delta_pi_indent{suf}", f"r_target_um{suf}", f"contact_area_um2{suf}"]
        ].dropna()
        p_over_T = (valid[f"pi_simple{suf}"] / valid[f"r_target_um{suf}"]).to_numpy(dtype=float)  # 1/um; * pN/um = pN/um^2 = Pa
        dp_over_T = (valid[f"delta_pi_indent{suf}"] / valid[f"r_target_um{suf}"]).to_numpy(dtype=float)
        f_over_T = (
            valid[f"pi_simple{suf}"] * valid[f"contact_area_um2{suf}"] / valid[f"r_target_um{suf}"]
        ).to_numpy(dtype=float)  # um; * pN/um = pN -> /1000 = nN
        df_over_T = (
            valid[f"delta_pi_indent{suf}"]
            * valid[f"contact_area_um2{suf}"]
            / valid[f"r_target_um{suf}"]
        ).to_numpy(dtype=float)

        med_pi = float(np.median(pi_vals))
        med_dpi = float(np.median(dpi_vals))
        med_rt = float(np.median(rt_vals))
        med_ca = float(np.median(ca_vals))
        med_p_fac = float(np.median(p_over_T))
        med_dp_fac = float(np.median(dp_over_T))
        med_f_fac = float(np.median(f_over_T))
        med_df_fac = float(np.median(df_over_T))

        rows.append(
            {
                "status_label": "provisional: route (b), pending 6b gate",
                "group": label,
                "n_chambers": int(len(valid)),
                "cup_depth_over_radius_median": float(np.median(dor_vals)),
                "patch_equiv_radius_um_median": float(np.median(pr_vals)),
                "r_target_um_median": med_rt,
                "contact_area_um2_median": med_ca,
                "pi_simple_median": med_pi,
                "pi_simple_q25": float(np.percentile(pi_vals, 25)),
                "pi_simple_q75": float(np.percentile(pi_vals, 75)),
                "delta_pi_indent_median": med_dpi,
                "delta_pi_indent_q25": float(np.percentile(dpi_vals, 25)),
                "delta_pi_indent_q75": float(np.percentile(dpi_vals, 75)),
                "pi_finite_depth_sensitivity_median": float(np.median(pi_fin_vals)),
                "active_pressure_p_nominal_pa_low": med_p_fac * t_nom_lo,
                "active_pressure_p_nominal_pa_high": med_p_fac * t_nom_hi,
                "active_pressure_p_full_pa_low": med_p_fac * t_full_lo,
                "active_pressure_p_full_pa_high": med_p_fac * t_full_hi,
                "net_indent_pressure_dp_nominal_pa_low": med_dp_fac * t_nom_lo,
                "net_indent_pressure_dp_nominal_pa_high": med_dp_fac * t_nom_hi,
                "total_force_F_nominal_nN_low": (med_f_fac * t_nom_lo) / 1000.0,
                "total_force_F_nominal_nN_high": (med_f_fac * t_nom_hi) / 1000.0,
                "total_force_F_full_nN_low": (med_f_fac * t_full_lo) / 1000.0,
                "total_force_F_full_nN_high": (med_f_fac * t_full_hi) / 1000.0,
                "net_indent_force_dF_nominal_nN_low": (med_df_fac * t_nom_lo) / 1000.0,
                "net_indent_force_dF_nominal_nN_high": (med_df_fac * t_nom_hi) / 1000.0,
                "tcell_tension_anchor_nominal_pN_per_um_low": float(t_tcell_nom[0]),
                "tcell_tension_anchor_nominal_pN_per_um_high": float(t_tcell_nom[1]),
            }
        )

    return pd.DataFrame(rows)


def run_exploratory_analyses(ch: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory analyses specified in status/step7_preregistration.md:
      - Zernike modes and cup asymmetry / roughness (donor high vs low & within-donor Cox)
      - Within-visit sweep SD of the 3 focus metrics
    Explicitly labeled EXPLORATORY_NOT_FOR_DECISION.
    """
    t1_B = ch[
        (ch["trace_num"] == 1)
        & ch["has_contact_desc_200m"]
        & (ch["n_pre_death_contact_visits_any_200m"] > 0)
    ].copy()
    t1_B_untr = t1_B[t1_B["condition"] != CONDITION_DRUG].copy()
    t1_B["z_ca"] = normal_scores_within_experiment(t1_B, "contact_area_um2_B_any")
    t1_B["z_nvis"] = normal_scores_within_experiment(t1_B, "n_pre_death_contact_visits_any_200m")

    exp_cols = EXPLORATORY_METRICS + [
        "cup_depth_over_radius_sd",
        "cup_depth_hull_p95_um_sd",
        "tcell_protrusion_frac_sd",
    ]
    rows = []
    for m in exp_cols:
        col = f"{m}_B_any"
        if col not in t1_B.columns:
            continue
        meds = (
            t1_B_untr.groupby(["donor", "condition"])[col]
            .median()
            .reset_index()
        )
        hi = meds.loc[meds["condition"] == CONDITION_HIGH, col].dropna().to_numpy(dtype=float)
        lo = meds.loc[meds["condition"] == CONDITION_LOW, col].dropna().to_numpy(dtype=float)
        u_stat, p_mw = mannwhitneyu(hi, lo, alternative="two-sided", method="exact")

        t1_B["z_m"] = normal_scores_within_experiment(t1_B, col)
        res_u = fit_stratified_cox(
            t1_B, "km_any_contact_time_min_200m", "km_event_any_visit_200m", ["z_m"]
        )
        if m != "contact_area_um2":
            res_a = fit_stratified_cox(
                t1_B,
                "km_any_contact_time_min_200m",
                "km_event_any_visit_200m",
                ["z_m", "z_ca", "z_nvis"],
            )
            hr_a, p_a = float(res_a["hr"][0]), float(res_a["p"][0])
        else:
            hr_a, p_a = np.nan, np.nan

        rows.append(
            {
                "analysis_type": "EXPLORATORY_NOT_FOR_DECISION",
                "summary": "B_pre_death_mean",
                "metric": m,
                "high_donor_median": float(np.median(hi)),
                "low_donor_median": float(np.median(lo)),
                "mw_u_high_vs_low": float(u_stat),
                "mw_p_two_sided": float(p_mw),
                "cox_hr_unadj": float(res_u["hr"][0]),
                "cox_p_unadj": float(res_u["p"][0]),
                "cox_hr_adj_ca_nvis": hr_a,
                "cox_p_adj_ca_nvis": p_a,
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    test7a_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    test7b_df: pd.DataFrame,
    test7c_df: pd.DataFrame,
    ch: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Generate figures/step7_focus.png following House Style:
      - Panel A (top-left): Donor medians high vs low killing (5 vs 5) with exact permutation p
      - Panel B (top-right): Within-donor Cox HR forest plot (Summary A & Summary B, adjusted)
      - Panel C (bottom-left): Dasatinib 50 nM paired donors (hd2, hd10, hd5) across the 3 focus metrics
      - Panel D (bottom-right): Push in stiffness units by group (provisional: route (b), pending 6b gate)
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

    fig = plt.figure(figsize=(15.5, 10.8), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 1.0],
        hspace=0.35,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    # -------------------------------------------------------------------------
    # Panel A: Donor medians high vs low with permutation p (Test 7A)
    # -------------------------------------------------------------------------
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.32)
    for idx_s, (s_code, s_title) in enumerate(
        [
            ("A_first_contact_pre_death", "Summary A (first contact)"),
            ("B_pre_death_mean", "Summary B (pre-death mean)"),
        ]
    ):
        ax = fig.add_subplot(gs_a[0, idx_s])
        sub_m = meds_df[
            (meds_df["summary"] == s_code)
            & (meds_df["cohort"] == "trace1_primary")
            & (meds_df["adjustment"] == "unadjusted")
            & (meds_df["metric"] == PRIMARY_METRIC)
        ].copy()
        row_u = test7a_df[
            (test7a_df["summary"] == s_code)
            & (test7a_df["cohort"] == "trace1_primary")
            & (test7a_df["adjustment"] == "unadjusted")
            & (test7a_df["metric"] == PRIMARY_METRIC)
        ].iloc[0]
        row_r = test7a_df[
            (test7a_df["summary"] == s_code)
            & (test7a_df["cohort"] == "trace1_primary")
            & (test7a_df["adjustment"] == "residualized")
            & (test7a_df["metric"] == PRIMARY_METRIC)
        ].iloc[0]

        hi_df = sub_m[sub_m["condition"] == CONDITION_HIGH].sort_values("donor")
        lo_df = sub_m[sub_m["condition"] == CONDITION_LOW].sort_values("donor")

        x_hi = np.zeros(len(hi_df)) + np.linspace(-0.08, 0.08, len(hi_df))
        x_lo = np.ones(len(lo_df)) + np.linspace(-0.08, 0.08, len(lo_df))

        ax.scatter(x_hi, hi_df["donor_median"], color=PLOT_1, s=36, zorder=3, label="High-killing (n=5)")
        ax.scatter(x_lo, lo_df["donor_median"], color=PLOT_2, s=36, zorder=3, label="Low-killing (n=5)")

        for x_v, (_, r_d) in zip(x_hi, hi_df.iterrows()):
            ax.text(x_v + 0.04, r_d["donor_median"], r_d["donor"], fontsize=7.5, color=MUTED, va="center")
        for x_v, (_, r_d) in zip(x_lo, lo_df.iterrows()):
            ax.text(x_v + 0.04, r_d["donor_median"], r_d["donor"], fontsize=7.5, color=MUTED, va="center")

        # Group median horizontal bars
        med_hi = float(np.median(hi_df["donor_median"]))
        med_lo = float(np.median(lo_df["donor_median"]))
        ax.plot([-0.22, 0.22], [med_hi, med_hi], color=INK, linewidth=2.0, zorder=4)
        ax.plot([0.78, 1.22], [med_lo, med_lo], color=INK, linewidth=2.0, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["High-killing\ndonors (n=5)", "Low-killing\ndonors (n=5)"], fontsize=8.5)
        if idx_s == 0:
            ax.set_ylabel("Donor median cup_depth_over_radius", fontsize=9.0, fontweight="bold")
        ax.set_xlim(-0.4, 1.45)
        ax.set_ylim(0.14, 0.56)
        ax.grid(axis="y", color=RULE, linewidth=0.6)
        ax.set_title(
            f"7A: {s_title}\n"
            f"Unadj perm p={row_u['perm_p_mean_diff_two_sided']:.2f} (MW p={row_u['perm_p_mw_two_sided']:.2f})\n"
            f"Resid perm p={row_r['perm_p_mean_diff_two_sided']:.2f} (g={row_u['hedges_g']:+.2f})",
            fontsize=8.5,
            fontweight="bold",
            loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B: Within-donor HR forest plot (Test 7B)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    sub_b = test7b_df[
        (test7b_df["cohort"] == "trace1_primary")
        & (test7b_df["metric"].isin(FOCUS_METRICS))
    ].copy()

    y_labels = [
        "Cup depth / patch radius\n(Primary confirmatory)",
        "Cup depth p95 (um)\n(Secondary 1)",
        "T-cell protrusion fraction\n(Secondary 2)",
    ]
    y_pos = np.arange(len(FOCUS_METRICS), dtype=float)[::-1]
    for yv in y_pos:
        ax_b.axhline(yv, color=RULE, linewidth=0.6, zorder=0)
    ax_b.axvline(1.0, color=INK, linestyle="--", linewidth=0.9, zorder=1)

    for idx_m, m_name in enumerate(FOCUS_METRICS):
        yv = y_pos[idx_m]
        rA = sub_b[(sub_b["summary"] == "A_first_contact_pre_death") & (sub_b["metric"] == m_name)].iloc[0]
        rB = sub_b[(sub_b["summary"] == "B_pre_death_mean") & (sub_b["metric"] == m_name)].iloc[0]

        # Summary A (adjusted for CA)
        ax_b.errorbar(
            rA["hr_prereg_adjusted"],
            yv + 0.14,
            xerr=[
                [rA["hr_prereg_adjusted"] - rA["profile_ci95_low_adjusted"]],
                [rA["profile_ci95_high_adjusted"] - rA["hr_prereg_adjusted"]],
            ],
            fmt="o",
            color=PLOT_1,
            ecolor=PLOT_1,
            elinewidth=1.5,
            capsize=3.0,
            markersize=5.5,
            label="Summary A (first contact, adj CA)" if idx_m == 0 else None,
            zorder=3,
        )
        txt_a = f"A: HR {rA['hr_prereg_adjusted']:.2f} [{rA['profile_ci95_low_adjusted']:.2f}, {rA['profile_ci95_high_adjusted']:.2f}], q={rA['bh_q_adjusted_3focus']:.3f}"
        ax_b.text(2.85, yv + 0.14, txt_a, fontsize=7.8, color=PLOT_1, va="center", ha="left")

        # Summary B (adjusted for CA + pre-death visit count)
        ax_b.errorbar(
            rB["hr_prereg_adjusted"],
            yv - 0.14,
            xerr=[
                [rB["hr_prereg_adjusted"] - rB["profile_ci95_low_adjusted"]],
                [rB["profile_ci95_high_adjusted"] - rB["hr_prereg_adjusted"]],
            ],
            fmt="s",
            color=PLOT_2,
            ecolor=PLOT_2,
            elinewidth=1.5,
            capsize=3.0,
            markersize=5.5,
            label="Summary B (pre-death mean, adj CA + visits)" if idx_m == 0 else None,
            zorder=3,
        )
        txt_b = f"B: HR {rB['hr_prereg_adjusted']:.2f} [{rB['profile_ci95_low_adjusted']:.2f}, {rB['profile_ci95_high_adjusted']:.2f}], q={rB['bh_q_adjusted_3focus']:.4f}"
        ax_b.text(2.85, yv - 0.14, txt_b, fontsize=7.8, color=PLOT_2, va="center", ha="left")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(y_labels, fontsize=8.5, fontweight="bold")
    ax_b.set_xlim(0.85, 4.55)
    ax_b.set_ylim(-0.55, 2.55)
    ax_b.set_xlabel("Experiment-stratified Cox HR per within-donor SD (profile 95% CI)", fontsize=9.0, fontweight="bold")
    ax_b.set_title(
        "7B: Within-donor cell-level killing (restatement of 5b with covariates)",
        fontsize=9.5,
        fontweight="bold",
        loc="left",
    )
    ax_b.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, fontsize=8.0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C: Dasatinib paired donors (Test 7C, Summary B & Summary A)
    # -------------------------------------------------------------------------
    gs_c = gs[1, 0].subgridspec(1, 3, wspace=0.34)
    c_specs = [
        ("cup_depth_over_radius", "Cup depth / radius\n(Primary)"),
        ("cup_depth_hull_p95_um", "Cup depth p95 (um)\n(Secondary 1)"),
        ("tcell_protrusion_frac", "Protrusion frac\n(Secondary 2)"),
    ]
    donor_colors = {"hd2": PLOT_1, "hd10": PLOT_2, "hd5": PLOT_3}
    for j_c, (m_c, lbl_c) in enumerate(c_specs):
        ax_c = fig.add_subplot(gs_c[0, j_c])
        rB_c = test7c_df[
            (test7c_df["summary"] == "B_pre_death_mean")
            & (test7c_df["cohort"] == "trace1_primary")
            & (test7c_df["metric"] == m_c)
        ].iloc[0]
        rA_c = test7c_df[
            (test7c_df["summary"] == "A_first_contact_pre_death")
            & (test7c_df["cohort"] == "trace1_primary")
            & (test7c_df["metric"] == m_c)
        ].iloc[0]

        for d_id in ("hd2", "hd10", "hd5"):
            col_d = donor_colors[d_id]
            # Summary B solid line
            u_b = rB_c[f"{d_id}_med_untr"]
            d_b = rB_c[f"{d_id}_med_drug"]
            ax_c.plot([0, 1], [u_b, d_b], "-o", color=col_d, linewidth=1.6, markersize=5, label=f"{d_id} (B)" if j_c == 0 else None)
            # Summary A dashed line
            u_a = rA_c[f"{d_id}_med_untr"]
            d_a = rA_c[f"{d_id}_med_drug"]
            ax_c.plot([0, 1], [u_a, d_a], "--s", color=col_d, linewidth=1.0, markersize=3.8, alpha=0.75)

        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(["Untreated", "Dasatinib\n50 nM"], fontsize=8.0)
        ax_c.set_xlim(-0.25, 1.25)
        ax_c.grid(axis="y", color=RULE, linewidth=0.6)
        ax_c.set_title(
            f"7C: {lbl_c}\nB: {int(rB_c['n_donors_lowered'])}/3 (p1={rB_c['donor_stratified_rank_mw_p_one_sided_lower']:.3f})\nA: {int(rA_c['n_donors_lowered'])}/3 (p1={rA_c['donor_stratified_rank_mw_p_one_sided_lower']:.3f})",
            fontsize=8.0,
            fontweight="bold",
            loc="left",
        )
        if j_c == 0:
            ax_c.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, fontsize=7.2)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel D: Push in stiffness units by group (provisional: route (b), pending 6b gate)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    t1_b = ch[
        (ch["trace_num"] == 1) & (ch["n_pre_death_contact_visits_any_200m"] > 0)
    ].copy()

    grp_defs = [
        ("High-killing\nuntreated (n=155)", t1_b[t1_b["condition"] == CONDITION_HIGH], PLOT_1),
        ("Low-killing\nuntreated (n=150)", t1_b[t1_b["condition"] == CONDITION_LOW], PLOT_2),
        ("Dasatinib 50 nM\n(n=61)", t1_b[t1_b["condition"] == CONDITION_DRUG], PLOT_4),
    ]

    x_pos = np.arange(len(grp_defs), dtype=float)
    box_data_pi = [g["pi_simple_B_any"].dropna().to_numpy(dtype=float) for _, g, _ in grp_defs]
    bp = ax_d.boxplot(
        box_data_pi,
        positions=x_pos,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 2.0},
        boxprops={"facecolor": "#F6F6F6", "edgecolor": INK, "linewidth": 1.0},
        whiskerprops={"color": INK, "linewidth": 1.0},
        capprops={"color": INK, "linewidth": 1.0},
    )

    # Overlay donor medians on top of chamber boxplots
    for idx_g, (g_lbl, g_df, g_col) in enumerate(grp_defs):
        d_meds = g_df.groupby("donor")["pi_simple_B_any"].median()
        jitter = np.linspace(-0.10, 0.10, len(d_meds))
        ax_d.scatter(
            np.full(len(d_meds), x_pos[idx_g]) + jitter,
            d_meds.to_numpy(dtype=float),
            color=g_col,
            s=38,
            zorder=4,
            label="Donor medians" if idx_g == 0 else None,
        )
        med_pi = float(np.median(g_df["pi_simple_B_any"].dropna()))
        med_dpi = float(np.median(g_df["delta_pi_indent_B_any"].dropna()))
        med_rt = float(np.median(g_df["r_target_um_B_any"].dropna()))
        med_ca = float(np.median(g_df["contact_area_um2_B_any"].dropna()))
        p_lo = (med_pi / med_rt) * 49.0
        p_hi = (med_pi / med_rt) * 163.0
        f_lo = p_lo * med_ca / 1000.0
        f_hi = p_hi * med_ca / 1000.0
        annot = (
            f"Med Pi={med_pi:.2f} (dPi={med_dpi:.2f})\n"
            f"p=[{p_lo:.0f}, {p_hi:.0f}] Pa\n"
            f"F=[{f_lo:.1f}, {f_hi:.1f}] nN"
        )
        ax_d.text(
            x_pos[idx_g],
            7.45,
            annot,
            fontsize=7.4,
            color=INK,
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFFFFF", "edgecolor": LINE, " linewidth": 0.6}
            if False
            else None,
        )

    ax_d.axhline(2.0, color=MUTED, linestyle=":", linewidth=0.9, label="Resting sphere Laplace baseline (Pi=2)")
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([g[0] for g in grp_defs], fontsize=8.5)
    ax_d.set_ylim(1.5, 7.8)
    ax_d.grid(axis="y", color=RULE, linewidth=0.6)
    ax_d.set_ylabel(
        "Dimensionless push Pi_simple = p R_target / T_target\n(provisional: route (b), pending 6b gate)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_d.set_title(
        "Push in stiffness units by group (provisional: route (b), pending 6b gate)",
        fontsize=9.2,
        fontweight="bold",
        loc="left",
    )
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        "Step 7 Focus Test: Do High-Killing Donors Push in a More Focused Way?",
        fontsize=12.5,
        fontweight="bold",
        x=0.07,
        y=0.97,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_ch, outlier_table, mv, meta_counts = build_datasets()
    ch, mv20_pre = augment_chamber_summaries_with_stiffness_and_sd(base_ch, mv)

    with open(STEP6B_DIR / "afm_to_tension_conversion.json", "r", encoding="utf-8") as f:
        afm_json = json.load(f)

    # 1. Test 7A: Between-donor focus (high vs low killing, n=5 vs 5) + visit LMM
    test7a_df, meds_df = run_test7a_donor_permutation(ch)
    test7a_df.to_csv(OUT_DIR / "test7a_donor_focus.csv", index=False)
    meds_df.to_csv(OUT_DIR / "test7a_donor_medians.csv", index=False)

    test7a_lmm_df = run_test7a_visit_lmm(mv20_pre)
    test7a_lmm_df.to_csv(OUT_DIR / "test7a_visit_lmm.csv", index=False)

    # 2. Test 7B: Within-donor cell-level killing (experiment-stratified Cox + mixed model check)
    test7b_df = run_test7b_within_donor_cox(ch)
    test7b_df.to_csv(OUT_DIR / "test7b_within_donor_cox.csv", index=False)

    # 3. Test 7C: Positive control (dasatinib 50 nM in hd2, hd10, hd5)
    test7c_df = run_test7c_dasatinib_control(ch)
    test7c_df.to_csv(OUT_DIR / "test7c_dasatinib_control.csv", index=False)

    # 4. Stiffness units via Route (b) (provisional: route (b), pending 6b gate)
    stiff_df = compute_stiffness_unit_table(ch, mv20_pre, afm_json)
    stiff_df.to_csv(OUT_DIR / "stiffness_route_b_summary.csv", index=False)

    # 5. Exploratory analyses (Zernike modes, within-visit SD)
    exp_df = run_exploratory_analyses(ch)
    exp_df.to_csv(OUT_DIR / "exploratory_step7.csv", index=False)

    # 6. Figure
    fig_path = FIG_DIR / "step7_focus.png"
    make_figure(test7a_df, meds_df, test7b_df, test7c_df, ch, fig_path)

    # Evaluate pre-registered decision rule strictly on the primary confirmatory metric (cup_depth_over_radius)
    t7a_prim = test7a_df[
        (test7a_df["cohort"] == "trace1_primary")
        & (test7a_df["metric"] == PRIMARY_METRIC)
    ]
    t7a_passes = bool(t7a_prim["passes_test7a_primary"].any())

    t7b_prim_A = test7b_df[
        (test7b_df["cohort"] == "trace1_primary")
        & (test7b_df["summary"] == "A_first_contact_pre_death")
        & (test7b_df["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    t7b_prim_B = test7b_df[
        (test7b_df["cohort"] == "trace1_primary")
        & (test7b_df["summary"] == "B_pre_death_mean")
        & (test7b_df["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    t7b_passes_both = bool(
        t7b_prim_A["passes_test7b_adjusted"] and t7b_prim_B["passes_test7b_adjusted"]
    )

    t7c_prim_B = test7c_df[
        (test7c_df["cohort"] == "trace1_primary")
        & (test7c_df["summary"] == "B_pre_death_mean")
        & (test7c_df["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    t7c_prim_A = test7c_df[
        (test7c_df["cohort"] == "trace1_primary")
        & (test7c_df["summary"] == "A_first_contact_pre_death")
        & (test7c_df["metric"] == PRIMARY_METRIC)
    ].iloc[0]

    # Pre-registration failed-control rule:
    # "Positive control fails (Test 7C does not show lower primary focus under dasatinib in 3/3 donors):
    #  'The focus metric does not respond to a known perturbation of T-cell pushing, so step 7 is inconclusive.'"
    t7c_primary_passes = bool(
        t7c_prim_B["lowered_3_of_3_donors"] or t7c_prim_A["lowered_3_of_3_donors"]
    )

    if not t7c_primary_passes:
        decision_key = "positive_control_fails"
    elif t7a_passes:
        decision_key = "donor_level_supported"
    elif t7b_passes_both:
        decision_key = "within_donor_only"
    else:
        decision_key = "no_effect"

    decision_sentence = DECISION_SENTENCES[decision_key]

    summary_payload = {
        "preregistration_file": "status/step7_preregistration.md",
        "primary_confirmatory_metric": PRIMARY_METRIC,
        "decision_key": decision_key,
        "decision_sentence": decision_sentence,
        "deviations_from_preregistration": [
            {
                "item": "Step 6b mechanical solver gate not yet decided; another agent is editing src/mech/ and data/step6b/.",
                "handling": "Computed Route (b) paper relation for all stiffness-unit numbers and labeled every number 'provisional: route (b), pending 6b gate'.",
            },
            {
                "item": "Step 5c raw-image secondary extension (results/step5_indepth_chambers.csv, 20 chambers) cannot be completed across all 20 chambers.",
                "handling": "Only 2/20 in-depth chambers have complete raw .h5 volumes locally (v01 MIP_ROI_2 and MIP_ROI_32); download.scidb.cn failed DNS resolution on remaining chambers (data/scidb_19235_raw/indepth.log). Because Step 5c is explicitly listed under 'Secondary / exploratory analyses (not part of the decision)', it does not affect any confirmatory test or decision.",
            },
            {
                "item": "Pre-registration text states 'Summary A (first-contact pre-death visit, n = 316 untreated)', whereas 316 is total untreated contacted trace_num==1 chambers and 11 had first contact at the death visit.",
                "handling": "Evaluated both strict pre-death Summary A (n = 305 untreated, 61 dasatinib) and all-first-contact Summary A (n = 316 untreated, 61 dasatinib); results are identical in direction and null status.",
            },
            {
                "item": "5 of 3,108 untreated pre-death contact visits (0.16%) have 1-voxel/coplanar target masks where 3D convex-hull cup depth is undefined (NaN).",
                "handling": "Visit-level mixed models use the 3,103 non-null visits across all 305 chambers and 10 donors.",
            },
        ],
        "test7a_between_donor": {
            "primary_metric": PRIMARY_METRIC,
            "passes_test7a": t7a_passes,
            "summary_A_unadjusted": test7a_df[
                (test7a_df["cohort"] == "trace1_primary")
                & (test7a_df["summary"] == "A_first_contact_pre_death")
                & (test7a_df["adjustment"] == "unadjusted")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "summary_A_residualized": test7a_df[
                (test7a_df["cohort"] == "trace1_primary")
                & (test7a_df["summary"] == "A_first_contact_pre_death")
                & (test7a_df["adjustment"] == "residualized")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "summary_B_unadjusted": test7a_df[
                (test7a_df["cohort"] == "trace1_primary")
                & (test7a_df["summary"] == "B_pre_death_mean")
                & (test7a_df["adjustment"] == "unadjusted")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "summary_B_residualized": test7a_df[
                (test7a_df["cohort"] == "trace1_primary")
                & (test7a_df["summary"] == "B_pre_death_mean")
                & (test7a_df["adjustment"] == "residualized")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "visit_lmm_trace1_primary": test7a_lmm_df[
                test7a_lmm_df["cohort"] == "trace1_primary"
            ]
            .set_index("metric")
            .to_dict(orient="index"),
        },
        "test7b_within_donor_cox": {
            "note": "Restatement on same visits as step 5b with covariate adjustments; not an independent confirmation.",
            "passes_both_summaries_primary_metric": t7b_passes_both,
            "summary_A_trace1_primary": test7b_df[
                (test7b_df["cohort"] == "trace1_primary")
                & (test7b_df["summary"] == "A_first_contact_pre_death")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "summary_B_trace1_primary": test7b_df[
                (test7b_df["cohort"] == "trace1_primary")
                & (test7b_df["summary"] == "B_pre_death_mean")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
        },
        "test7c_dasatinib_positive_control": {
            "primary_metric_passes_3of3": t7c_primary_passes,
            "summary_B_trace1_primary": test7c_df[
                (test7c_df["cohort"] == "trace1_primary")
                & (test7c_df["summary"] == "B_pre_death_mean")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
            "summary_A_trace1_primary": test7c_df[
                (test7c_df["cohort"] == "trace1_primary")
                & (test7c_df["summary"] == "A_first_contact_pre_death")
            ]
            .set_index("metric")
            .to_dict(orient="index"),
        },
        "stiffness_units_provisional_route_b": {
            "status": "provisional: route (b), pending 6b gate",
            "formula": "Pi_simple = 2 + 4 * (R_target / a) * (d / a); Delta_Pi_indent = 4 * (R_target / a) * cup_depth_over_radius",
            "cortical_tension_anchors_pN_per_um": {
                "target_nominal": [49.0, 163.0],
                "target_full_range": [18.0, 430.0],
                "tcell_nominal_jung2021": afm_json.get("tcell_cortex_from_jung2021_afm", {}).get(
                    "T_cortical_nominal_pN_per_um", [569.5, 1789.9]
                ),
            },
            "groups": stiff_df.set_index("group").to_dict(orient="index"),
        },
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Decision key: {decision_key}")
    print(f"Decision sentence: {decision_sentence}")
    print(f"Outputs written to {OUT_DIR} and {fig_path}")


if __name__ == "__main__":
    main()
