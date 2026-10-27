#!/usr/bin/env python3
"""
Step 3b: Recover chamber-to-paper-row mapping across available experiments in data/step3/<experiment>/.

Method:
  1. Experiment-to-group mapping from V_TO_GROUP in src/run_dent_stiffness_fixes_all_frames.py.
     Paper rows from data/clean_per_cell_419.csv (group_id as string, sorted by cell_index).
  2. Order-preserving alignment (monotone subsequence dynamic programming) of each group's
     n_rows paper rows onto the experiment's n_chambers ROIs in natural ROI index order.
     Cost per (row i, chamber j):
       - Primary (peak visit): |log(our_peak_CA / paper_CA)|
       - Fallback (single visit): min_v |log(visit_v_CA / paper_CA)| + FALLBACK_PENALTY
       - Zero-CA handling: 0 if both are 0 (or fallback to a zero visit with FALLBACK_PENALTY),
         otherwise capped at COST_CAP.
       - Capped at COST_CAP so a single outlier row cannot distort the global alignment.
  3. Per-row ambiguity via forward-backward DP:
       margin[i] = min_{k != j*(i)} T[i, k] - T[i, j*(i)],
     where T[i, j] is the optimal total alignment cost constrained to assign row i to chamber j.
     Labels:
       - firm:      peak-visit match within 0.5% and margin >= AMBIGUOUS_MARGIN
       - probable:  fallback visit within 5%, or peak-visit match within 0.5-5%, and margin >= AMBIGUOUS_MARGIN
       - ambiguous: best alternative chamber has margin < AMBIGUOUS_MARGIN (and match within 5%)
       - unmatched: percentage difference > 5% at matched visit
  4. Chance baseline per group (500 Monte Carlo trials):
       - Shuffled across the 419-cell paper pool (excluding current group, or drawn without replacement
         from all 419 cells) and log-jittered draws; reports expected number of firm matches by chance.
  5. Writes:
       - results/chamber_to_row_map.csv
       - results/chamber_to_row_summary.csv
       - figures/step3_contact_area_reproduction.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from run_dent_stiffness_fixes_all_frames import GROUP_ORDER, V_TO_GROUP

# Alignment & classification parameters
FALLBACK_PENALTY: float = 0.015   # log-cost penalty (~1.5%) when using a non-peak visit
COST_CAP: float = 0.25            # maximum per-row log cost (~28.4%) so one bad row cannot drive alignment
FIRM_PCT_TOL: float = 0.5         # <= 0.5% error at peak visit required for "firm"
PROBABLE_PCT_TOL: float = 5.0     # <= 5.0% error required for "probable" / "ambiguous" (else "unmatched")
AMBIGUOUS_MARGIN: float = 0.005   # log-cost margin threshold (~0.5%) below which a row is "ambiguous"
N_CHANCE_TRIALS: int = 500        # Monte Carlo trials per group for chance baseline
RNG_SEED: int = 42


def compute_cost_matrix(
    paper_ca: np.ndarray,
    peak_ca: np.ndarray,
    visit_ca: np.ndarray,
    fallback_penalty: float = FALLBACK_PENALTY,
    cost_cap: float = COST_CAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (n_rows, n_chambers) alignment cost matrix and visit-selection metadata.

    Returns:
      cost:         (n_rows, n_chambers) capped log cost
      matched_v:    (n_rows, n_chambers) 1-based visit index selected (0 if CA == 0)
      matched_ca:   (n_rows, n_chambers) CA (um^2) of the selected visit (or peak)
      is_fallback:  (n_rows, n_chambers) True if a non-peak visit was chosen
      pct_diff_used:(n_rows, n_chambers) signed percentage difference at matched_ca
    """
    n_rows = len(paper_ca)
    n_ch = len(peak_ca)
    n_visits = visit_ca.shape[1]

    cost = np.full((n_rows, n_ch), cost_cap, dtype=float)
    matched_v = np.zeros((n_rows, n_ch), dtype=int)
    matched_ca = np.zeros((n_rows, n_ch), dtype=float)
    is_fallback = np.zeros((n_rows, n_ch), dtype=bool)
    pct_diff_used = np.zeros((n_rows, n_ch), dtype=float)

    # Precompute 1-based peak visit index per chamber
    peak_v_idx = np.zeros(n_ch, dtype=int)
    for j in range(n_ch):
        if peak_ca[j] > 0 and n_visits > 0:
            peak_v_idx[j] = int(np.argmax(visit_ca[j, :])) + 1
        else:
            peak_v_idx[j] = 0

    for i in range(n_rows):
        p_val = float(paper_ca[i])
        for j in range(n_ch):
            pk_val = float(peak_ca[j])
            v_arr = visit_ca[j, :]

            if p_val <= 0.0:
                if pk_val <= 0.0:
                    cost[i, j] = 0.0
                    matched_v[i, j] = 0
                    matched_ca[i, j] = 0.0
                    is_fallback[i, j] = False
                    pct_diff_used[i, j] = 0.0
                else:
                    # Paper row has CA == 0, but chamber has positive peak CA.
                    # Only an uncontacted chamber (pk_val == 0) is a true zero match;
                    # otherwise assign capped cost.
                    cost[i, j] = cost_cap
                    matched_v[i, j] = peak_v_idx[j]
                    matched_ca[i, j] = pk_val
                    is_fallback[i, j] = False
                    pct_diff_used[i, j] = 100.0
                continue

            # p_val > 0
            c_peak = abs(np.log(pk_val / p_val)) if pk_val > 0.0 else cost_cap

            # Evaluate single-visit fallback across positive visits
            pos_mask = v_arr > 0.0
            if np.any(pos_mask):
                v_costs = np.where(pos_mask, np.abs(np.log(np.maximum(v_arr, 1e-9) / p_val)), np.inf)
                best_k = int(np.argmin(v_costs))
                best_v_val = float(v_arr[best_k])
                c_fb = float(v_costs[best_k]) + fallback_penalty
            else:
                best_k = -1
                best_v_val = 0.0
                c_fb = cost_cap

            if (best_k >= 0) and ((best_k + 1) != peak_v_idx[j]) and (c_fb < c_peak):
                cost[i, j] = min(cost_cap, c_fb)
                matched_v[i, j] = best_k + 1
                matched_ca[i, j] = best_v_val
                is_fallback[i, j] = True
                pct_diff_used[i, j] = (best_v_val - p_val) / p_val * 100.0
            else:
                cost[i, j] = min(cost_cap, c_peak)
                matched_v[i, j] = peak_v_idx[j]
                matched_ca[i, j] = pk_val
                is_fallback[i, j] = False
                pct_diff_used[i, j] = ((pk_val - p_val) / p_val * 100.0) if pk_val > 0.0 else -100.0

    return cost, matched_v, matched_ca, is_fallback, pct_diff_used


def solve_monotone_alignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve strictly increasing subsequence alignment via forward-backward dynamic programming.

    Args:
      cost: (n_rows, n_chambers) matrix

    Returns:
      opt_j:    (n_rows,) 0-based chamber index assigned to each row in the optimal alignment
      T:        (n_rows, n_chambers) min total alignment cost when row i is forced to chamber j
      opt_cost: scalar optimal alignment cost
    """
    n_rows, n_ch = cost.shape
    if n_rows > n_ch:
        raise ValueError(f"Cannot align {n_rows} paper rows onto {n_ch} chambers monotonically.")

    # Forward DP: F[i, j] = min cost to align rows 0..i with row i assigned to chamber j
    F = np.full((n_rows, n_ch), np.inf, dtype=float)
    back = np.full((n_rows, n_ch), -1, dtype=int)
    F[0, : n_ch - n_rows + 1] = cost[0, : n_ch - n_rows + 1]

    for i in range(1, n_rows):
        j_min = i
        j_max = n_ch - (n_rows - 1 - i)
        best_prev_cost = np.inf
        best_prev_k = -1
        for j in range(j_min, j_max):
            k = j - 1
            if F[i - 1, k] < best_prev_cost:
                best_prev_cost = F[i - 1, k]
                best_prev_k = k
            F[i, j] = cost[i, j] + best_prev_cost
            back[i, j] = best_prev_k

    # Backward DP: B[i, j] = min cost to align rows i..n_rows-1 with row i assigned to chamber j
    B = np.full((n_rows, n_ch), np.inf, dtype=float)
    B[-1, n_rows - 1 :] = cost[-1, n_rows - 1 :]

    for i in range(n_rows - 2, -1, -1):
        j_min = i
        j_max = n_ch - (n_rows - 1 - i)
        best_next_cost = np.inf
        for j in range(j_max - 1, j_min - 1, -1):
            k = j + 1
            if B[i + 1, k] < best_next_cost:
                best_next_cost = B[i + 1, k]
            B[i, j] = cost[i, j] + best_next_cost

    # Total minimum cost constrained to pass through (i, j)
    T = F + B - cost
    opt_j = np.zeros(n_rows, dtype=int)
    opt_j[-1] = int(np.argmin(F[-1, :]))
    opt_cost = float(F[-1, opt_j[-1]])
    for i in range(n_rows - 2, -1, -1):
        opt_j[i] = back[i + 1, opt_j[i + 1]]

    return opt_j, T, opt_cost


def classify_row(
    paper_val: float,
    peak_val: float,
    used_val: float,
    used_fb: bool,
    margin: float,
    firm_pct_tol: float = FIRM_PCT_TOL,
    probable_pct_tol: float = PROBABLE_PCT_TOL,
    ambiguous_margin: float = AMBIGUOUS_MARGIN,
) -> str:
    """
    Assign one of {'firm', 'probable', 'ambiguous', 'unmatched'} to an aligned row.
    """
    if paper_val <= 0.0:
        if peak_val <= 0.0:
            return "firm" if margin >= ambiguous_margin else "ambiguous"
        return "unmatched"

    pct_peak = abs((peak_val - paper_val) / paper_val) * 100.0 if peak_val > 0.0 else 100.0
    pct_used = abs((used_val - paper_val) / paper_val) * 100.0 if used_val > 0.0 else 100.0

    if pct_used > probable_pct_tol:
        return "unmatched"
    if margin < ambiguous_margin:
        return "ambiguous"
    if (not used_fb) and (pct_peak <= firm_pct_tol):
        return "firm"
    return "probable"


def align_one_group(
    exp_name: str,
    group_id: str,
    df_paper_group: pd.DataFrame,
    df_chambers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align one group's paper rows onto its experiment's chambers and return row-level DataFrame.
    """
    df_p = df_paper_group.sort_values("cell_index").reset_index(drop=True)
    df_c = df_chambers.sort_values("roi_index").reset_index(drop=True)

    visit_cols = sorted(
        [c for c in df_c.columns if c.startswith("visit_") and c.endswith("_CA_um2")],
        key=lambda c: int(c.split("_")[1]),
    )

    paper_ca = df_p["CA_um2"].to_numpy(dtype=float)
    peak_ca = df_c["CA_um2"].to_numpy(dtype=float)
    visit_ca = df_c[visit_cols].to_numpy(dtype=float)
    rois = df_c["roi"].astype(str).to_numpy()

    cost, matched_v, matched_ca, is_fb, _ = compute_cost_matrix(paper_ca, peak_ca, visit_ca)
    opt_j, T, _ = solve_monotone_alignment(cost)

    rows_out = []
    n_rows = len(df_p)
    for i in range(n_rows):
        j = int(opt_j[i])
        p_val = float(paper_ca[i])
        pk_val = float(peak_ca[j])
        u_val = float(matched_ca[i, j])
        u_v = int(matched_v[i, j])
        fb = bool(is_fb[i, j])

        # Best alternative chamber for row i
        row_T = T[i, :].copy()
        row_T[j] = np.inf
        alt_j = int(np.argmin(row_T))
        margin = float(row_T[alt_j] - T[i, j])

        # If row is ambiguous and the alternative chamber has a smaller peak-visit discrepancy
        # (e.g. Group 1 row 31 where MIP_ROI_35 is the single-T-cell chamber and MIP_ROI_36 is a
        # multi-T-cell excluded chamber with an equally close visit), prefer the lower-index or
        # closer-peak chamber when tied within AMBIGUOUS_MARGIN.
        if margin < AMBIGUOUS_MARGIN:
            # Check if alt_j has a smaller peak discrepancy or fewer multi-T-cell sweeps
            mult_j = int(df_c.loc[j, "multi_tcell_sweeps"]) if "multi_tcell_sweeps" in df_c.columns else 0
            mult_alt = int(df_c.loc[alt_j, "multi_tcell_sweeps"]) if "multi_tcell_sweeps" in df_c.columns else 0
            if (mult_alt < mult_j) or (mult_alt == mult_j and abs(peak_ca[alt_j] - p_val) < abs(pk_val - p_val)):
                j, alt_j = alt_j, j
                pk_val = float(peak_ca[j])
                u_val = float(matched_ca[i, j])
                u_v = int(matched_v[i, j])
                fb = bool(is_fb[i, j])

        label = classify_row(p_val, pk_val, u_val, fb, margin)

        if p_val > 0.0:
            pct_diff_peak = (pk_val - p_val) / p_val * 100.0
            pct_diff_visit = (u_val - p_val) / p_val * 100.0
        else:
            pct_diff_peak = 0.0 if pk_val <= 0.0 else 100.0
            pct_diff_visit = 0.0 if u_val <= 0.0 else 100.0

        matched_visit_str = f"visit_{u_v:02d}" if u_v > 0 else "none"
        if not fb and u_v > 0:
            matched_visit_str = f"visit_{u_v:02d} (peak)"
        elif fb and u_v > 0:
            matched_visit_str = f"visit_{u_v:02d} (fallback:{u_val:.3f})"

        rows_out.append({
            "group_id": str(group_id),
            "cell_index": int(df_p.loc[i, "cell_index"]),
            "experiment": exp_name,
            "roi": rois[j],
            "paper_CA_um2": round(p_val, 3),
            "our_CA_um2": round(pk_val, 3),
            "matched_visit": matched_visit_str,
            "pct_diff": round(pct_diff_peak, 3),
            "label": label,
            "alt_roi": rois[alt_j],
            "margin": round(margin, 5),
            "_matched_visit_CA_um2": round(u_val, 3),
            "_matched_visit_pct_diff": round(pct_diff_visit, 3),
        })

    return pd.DataFrame(rows_out)


def run_chance_baseline(
    n_rows: int,
    df_chambers: pd.DataFrame,
    all_paper_ca: np.ndarray,
    group_paper_ca: np.ndarray,
    n_trials: int = N_CHANCE_TRIALS,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    """
    Vectorized Monte Carlo chance baseline (n_trials = 500):
      1. Shuffled across groups: draw n_rows values without replacement from the 419-cell paper table
         (shuffled across all groups) and run the exact same monotone DP + margin check.
      2. Log-jittered / permuted within group: draw n_rows values with 15% log-normal jitter
         around random permutations of the group's paper values.
    """
    df_c = df_chambers.sort_values("roi_index").reset_index(drop=True)
    visit_cols = sorted(
        [c for c in df_c.columns if c.startswith("visit_") and c.endswith("_CA_um2")],
        key=lambda c: int(c.split("_")[1]),
    )
    peak_ca = df_c["CA_um2"].to_numpy(dtype=float)
    visit_ca = df_c[visit_cols].to_numpy(dtype=float)
    n_ch = len(peak_ca)

    rng = np.random.default_rng(seed)

    def _eval_sample(sample_p: np.ndarray) -> int:
        # Fast vectorized cost matrix for one trial
        p = sample_p[:, None]  # (n_rows, 1)
        pk = peak_ca[None, :]  # (1, n_ch)

        pos_p = p > 0.0
        pos_pk = pk > 0.0
        both_pos = pos_p & pos_pk
        both_zero = (~pos_p) & (~pos_pk)

        c_peak = np.full((n_rows, n_ch), COST_CAP, dtype=float)
        c_peak[both_zero] = 0.0
        c_peak[both_pos] = np.abs(
            np.log(np.maximum(np.broadcast_to(pk, (n_rows, n_ch))[both_pos], 1e-9) / np.broadcast_to(p, (n_rows, n_ch))[both_pos])
        )

        # Fallback visits: (n_rows, n_ch, n_visits)
        v = visit_ca[None, :, :]
        pos_v = v > 0.0
        valid_v = pos_p[:, :, None] & pos_v
        v_costs = np.where(valid_v, np.abs(np.log(np.maximum(v, 1e-9) / np.maximum(p[:, :, None], 1e-9))), np.inf)
        c_fb = np.min(v_costs, axis=2) + FALLBACK_PENALTY

        use_fb = c_fb < c_peak
        cost = np.minimum(COST_CAP, np.where(use_fb, c_fb, c_peak))

        opt_j, T, _ = solve_monotone_alignment(cost)

        firm = 0
        for i in range(n_rows):
            j = int(opt_j[i])
            row_T = T[i, :].copy()
            row_T[j] = np.inf
            margin = float(np.min(row_T) - T[i, j])
            if margin < AMBIGUOUS_MARGIN or use_fb[i, j]:
                continue
            p_val = float(sample_p[i])
            pk_val = float(peak_ca[j])
            if p_val <= 0.0 and pk_val <= 0.0:
                firm += 1
            elif p_val > 0.0 and pk_val > 0.0:
                if abs((pk_val - p_val) / p_val) * 100.0 <= FIRM_PCT_TOL:
                    firm += 1
        return firm

    firms_shuffled = np.zeros(n_trials, dtype=int)
    firms_jittered = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        sample_shuf = rng.choice(all_paper_ca, size=n_rows, replace=False)
        firms_shuffled[t] = _eval_sample(sample_shuf)

        perm_g = rng.permutation(group_paper_ca)
        jitter = np.exp(rng.normal(0.0, 0.15, size=n_rows))
        sample_jit = np.where(perm_g > 0.0, perm_g * jitter, 0.0)
        firms_jittered[t] = _eval_sample(sample_jit)

    return {
        "chance_firm_mean": round(float(np.mean(firms_shuffled)), 3),
        "chance_firm_sd": round(float(np.std(firms_shuffled)), 3),
        "chance_firm_p95": round(float(np.percentile(firms_shuffled, 95)), 1),
        "chance_jitter_firm_mean": round(float(np.mean(firms_jittered)), 3),
        "chance_jitter_firm_sd": round(float(np.std(firms_jittered)), 3),
    }


def make_reproduction_figure(
    df_map: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Redraw figures/step3_contact_area_reproduction.png from the ordered alignment:
      - Left: our vs paper CA (um^2) for all mapped groups, coloured by label, y=x line, log axes.
      - Right: per-group bars of firm / probable / ambiguous / unmatched with chance level marked.
    Applies house-style tokens and typography.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial"],
        "axes.edgecolor": "#111111",
        "axes.linewidth": 0.8,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "text.color": "#111111",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    # House-style data palette
    label_colors = {
        "firm": "#1F4E79",       # --plot-1
        "probable": "#2E7D5B",   # --plot-3
        "ambiguous": "#C0561F",  # --plot-2
        "unmatched": "#B3261E",  # --danger
    }
    label_names = {
        "firm": "Firm (peak <= 0.5%)",
        "probable": "Probable (fallback / 0.5-5%)",
        "ambiguous": "Ambiguous (margin < 0.005)",
        "unmatched": "Unmatched (> 5%)",
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=200)

    # --- Left panel: Our vs Paper CA (log-log) ---
    pos_mask = (df_map["paper_CA_um2"] > 0) & (df_map["our_CA_um2"] > 0)
    df_pos = df_map[pos_mask].copy()
    n_zero_both = int(((df_map["paper_CA_um2"] <= 0) & (df_map["our_CA_um2"] <= 0)).sum())

    if len(df_pos) > 0:
        vmin = min(float(df_pos["paper_CA_um2"].min()), float(df_pos["our_CA_um2"].min())) * 0.8
        vmax = max(float(df_pos["paper_CA_um2"].max()), float(df_pos["our_CA_um2"].max())) * 1.25
    else:
        vmin, vmax = 10.0, 500.0

    ax_left.plot([vmin, vmax], [vmin, vmax], color="#5C5C5C", linestyle="--", linewidth=1.0, label="y = x", zorder=1)

    for lbl in ["firm", "probable", "ambiguous", "unmatched"]:
        sub = df_pos[df_pos["label"] == lbl]
        if len(sub) == 0:
            continue
        cnt_total = int((df_map["label"] == lbl).sum())
        ax_left.scatter(
            sub["paper_CA_um2"],
            sub["our_CA_um2"],
            s=32,
            color=label_colors[lbl],
            alpha=0.9,
            edgecolors="#FFFFFF",
            linewidths=0.4,
            label=f"{label_names[lbl]} (n={cnt_total})",
            zorder=3 if lbl != "firm" else 2,
        )

        # Draw vertical connector to matched fallback visit when different from peak
        if lbl in ("probable", "ambiguous"):
            for _, r in sub.iterrows():
                v_ca = float(r["_matched_visit_CA_um2"])
                if abs(v_ca - float(r["our_CA_um2"])) > 1.0 and v_ca > 0:
                    ax_left.plot(
                        [r["paper_CA_um2"], r["paper_CA_um2"]],
                        [r["our_CA_um2"], v_ca],
                        color=label_colors[lbl],
                        linestyle=":",
                        linewidth=0.9,
                        zorder=2,
                    )
                    ax_left.scatter(
                        [r["paper_CA_um2"]],
                        [v_ca],
                        s=18,
                        facecolors="none",
                        edgecolors=label_colors[lbl],
                        linewidths=1.0,
                        zorder=4,
                    )

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlim(vmin, vmax)
    ax_left.set_ylim(vmin, vmax)
    ax_left.set_xlabel(r"Published contact area ($\mu\mathrm{m}^2$)", fontsize=11, fontweight="600")
    ax_left.set_ylabel(r"Reproduced peak contact area ($\mu\mathrm{m}^2$)", fontsize=11, fontweight="600")
    title_left = "Contact area alignment"
    if n_zero_both > 0:
        title_left += rf" (+{n_zero_both} at 0 $\mu\mathrm{{m}}^2$)"
    ax_left.set_title(title_left, fontsize=12, fontweight="700", loc="left", pad=8)
    ax_left.grid(True, which="major", color="#E3E3E3", linewidth=0.6)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(frameon=False, fontsize=9, loc="upper left")

    # --- Right panel: Per-group stacked bars of labels + chance firm level ---
    groups = df_summary["group_id"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.55 if len(groups) > 1 else 0.35

    bottom = np.zeros(len(groups), dtype=float)
    for lbl in ["firm", "probable", "ambiguous", "unmatched"]:
        vals = df_summary[lbl].to_numpy(dtype=float)
        if np.any(vals > 0) or lbl == "firm":
            ax_right.bar(
                x,
                vals,
                width=width,
                bottom=bottom,
                color=label_colors[lbl],
                label=lbl.capitalize(),
                edgecolor="#FFFFFF",
                linewidth=0.5,
            )
            bottom += vals

    # Mark chance firm level per group
    chance_vals = df_summary["chance_firm_mean"].to_numpy(dtype=float)
    for idx_g, ch_v in enumerate(chance_vals):
        ax_right.plot(
            [x[idx_g] - width * 0.55, x[idx_g] + width * 0.55],
            [ch_v, ch_v],
            color="#111111",
            linestyle="--",
            linewidth=1.5,
            label="Chance firm (mean)" if idx_g == 0 else None,
            zorder=5,
        )

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"Group {g}" for g in groups], fontsize=10)
    if len(groups) == 1:
        ax_right.set_xlim(-0.8, 0.8)
    ax_right.set_ylabel("Number of cells (rows)", fontsize=11, fontweight="600")
    ax_right.set_xlabel("Experimental group", fontsize=11, fontweight="600")
    ax_right.set_title("Per-group alignment classification", fontsize=12, fontweight="700", loc="left", pad=8)
    ax_right.set_ylim(0, max(float(df_summary["n_rows"].max()) * 1.15, 10.0))
    ax_right.grid(True, axis="y", color="#E3E3E3", linewidth=0.6)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    step3_dir = DATA_DIR / "step3"
    paper_csv = DATA_DIR / "clean_per_cell_419.csv"
    df_paper = pd.read_csv(paper_csv, dtype={"group_id": str})
    all_paper_ca = df_paper["CA_um2"].to_numpy(dtype=float)

    # Discover completed experiment subfolders in V_TO_GROUP order
    available_exps: list[tuple[str, str, Path]] = []
    for exp_name, (gid, _cond) in V_TO_GROUP.items():
        ch_csv = step3_dir / exp_name / "per_chamber_contact_area.csv"
        if ch_csv.is_file():
            available_exps.append((exp_name, str(gid), ch_csv))

    # Sort by GROUP_ORDER
    group_rank = {g: idx for idx, g in enumerate(GROUP_ORDER)}
    available_exps.sort(key=lambda item: group_rank.get(item[1], 999))

    if not available_exps:
        print("No completed experiment subfolders found in data/step3/<experiment>/.")
        return

    all_map_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for exp_name, gid, ch_csv in available_exps:
        df_ch = pd.read_csv(ch_csv)
        df_pg = df_paper[df_paper["group_id"] == gid].sort_values("cell_index").reset_index(drop=True)

        df_group_map = align_one_group(exp_name, gid, df_pg, df_ch)
        all_map_dfs.append(df_group_map)

        chance_stats = run_chance_baseline(
            n_rows=len(df_pg),
            df_chambers=df_ch,
            all_paper_ca=all_paper_ca,
            group_paper_ca=df_pg["CA_um2"].to_numpy(dtype=float),
            n_trials=N_CHANCE_TRIALS,
            seed=RNG_SEED + group_rank.get(gid, 0),
        )

        counts = df_group_map["label"].value_counts().to_dict()
        summary_rows.append({
            "group_id": gid,
            "experiment": exp_name,
            "n_rows": int(len(df_pg)),
            "n_chambers": int(len(df_ch)),
            "firm": int(counts.get("firm", 0)),
            "probable": int(counts.get("probable", 0)),
            "ambiguous": int(counts.get("ambiguous", 0)),
            "unmatched": int(counts.get("unmatched", 0)),
            "chance_firm_mean": chance_stats["chance_firm_mean"],
            "chance_firm_sd": chance_stats["chance_firm_sd"],
            "chance_firm_p95": chance_stats["chance_firm_p95"],
            "chance_jitter_firm_mean": chance_stats["chance_jitter_firm_mean"],
        })

    df_map_all = pd.concat(all_map_dfs, ignore_index=True)
    df_summary_all = pd.DataFrame(summary_rows)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_map_cols = [
        "group_id",
        "cell_index",
        "experiment",
        "roi",
        "paper_CA_um2",
        "our_CA_um2",
        "matched_visit",
        "pct_diff",
        "label",
        "alt_roi",
        "margin",
    ]
    map_csv_path = results_dir / "chamber_to_row_map.csv"
    df_map_all[out_map_cols].to_csv(map_csv_path, index=False)

    out_sum_cols = [
        "group_id",
        "n_rows",
        "n_chambers",
        "firm",
        "probable",
        "ambiguous",
        "unmatched",
        "chance_firm_mean",
    ]
    sum_csv_path = results_dir / "chamber_to_row_summary.csv"
    df_summary_all[out_sum_cols].to_csv(sum_csv_path, index=False)

    fig_path = FIGURES_DIR / "step3_contact_area_reproduction.png"
    make_reproduction_figure(df_map_all, df_summary_all, fig_path)

    print(f"Wrote {len(df_map_all)} mapped rows across {len(df_summary_all)} group(s) to {map_csv_path}")
    print(f"Wrote group summary to {sum_csv_path}:")
    print(df_summary_all[out_sum_cols].to_string(index=False))
    print(f"Wrote figure to {fig_path}")


if __name__ == "__main__":
    main()
