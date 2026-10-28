#!/usr/bin/env python3
"""
Step 3b & 3c: Recover chamber-to-paper-row mapping across available experiments in
`data/step3/<experiment>/` (`--source masks_only`) or `data/step3_tracked/<experiment>/`
(`--source tracked`).

Method:
  1. Experiment-to-group mapping from V_TO_GROUP in src/run_dent_stiffness_fixes_all_frames.py.
     Paper rows from data/clean_per_cell_419.csv (group_id as string, sorted by cell_index).
  2. When `--source tracked` is used and `trace_num` is present:
     - If `n_trace1 == sum(trace_num == 1)` is `>= n_rows`, candidate chambers are restricted
       to `trace_num == 1` chambers (matching `step1_setup_analysis.m` line 45 `if trace_num == 1`).
       In the exact bijection case `n_trace1 == n_rows`, every paper row is forced 1-to-1 onto
       the `n_rows` `trace_num == 1` chambers in strictly increasing ROI order (`margin = inf`).
     - If `n_trace1 < n_rows` (e.g., Group 7 where 8 chambers `MIP_ROI_4..9, 37, 38` were not
       deposited in ScienceDB so only 23 `trace_num == 1` chambers exist on disk for 24 paper rows),
       all available chambers are kept as candidates and non-`trace_num == 1` chambers receive an
       exclusion penalty so all `trace_num == 1` chambers are prioritized while allowing the missing
       undeposited slot to be localized.
  3. Order-preserving alignment (monotone subsequence dynamic programming) of each group's
     n_rows paper rows onto the candidate chambers in natural ROI index order.
     Cost per (row i, chamber j):
       - Primary (peak visit): |log(our_peak_CA / paper_CA)|
       - Fallback (single visit): min_v |log(visit_v_CA / paper_CA)| + FALLBACK_PENALTY
       - Zero-CA handling: 0 if both are 0 (or fallback to a zero visit with FALLBACK_PENALTY),
         otherwise capped at COST_CAP.
       - Capped at COST_CAP so a single outlier row cannot distort the global alignment.
  4. Split each row's classification into:
       - `identity_label`: is the chamber assignment forced/firm (`firm`, `probable`, `ambiguous`, `unmatched`)
       - `value_label`: does our CA reproduce the paper value within 0.5% / 5% (`firm`, `probable`, `unmatched`)
       - `label`: combined overall label (`firm`, `probable`, `ambiguous`, `unmatched`)
  5. Writes:
       - `--source masks_only`:
           results/chamber_to_row_map.csv
           results/chamber_to_row_summary.csv
           figures/step3_contact_area_reproduction.png
       - `--source tracked`:
           results/chamber_to_row_map_tracked.csv
           results/chamber_to_row_summary_tracked.csv
           figures/step3_tracked_vs_masks_only.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.paths import DATA_DIR, FIGURES_DIR, REPO_ROOT
except ImportError:
    from paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from run_dent_stiffness_fixes_all_frames import GROUP_ORDER, V_TO_GROUP

# Alignment & classification parameters
FALLBACK_PENALTY: float = 0.015   # log-cost penalty (~1.5%) when using a non-peak visit
COST_CAP: float = 0.25            # maximum per-row log cost (~28.4%) so one bad row cannot drive alignment
TRACE_NEQ1_PENALTY: float = 1.00  # penalty against trace_num != 1 when n_trace1 < n_rows (e.g. Group 7)
FIRM_PCT_TOL: float = 0.5         # <= 0.5% error at peak visit required for "firm" value reproduction
PROBABLE_PCT_TOL: float = 5.0     # <= 5.0% error required for "probable" (else "unmatched")
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

    # Precompute 1-based peak visit index per chamber (ties broken to first positive visit)
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
                if abs(pk_val) <= 1e-12:
                    cost[i, j] = 0.0
                    matched_v[i, j] = 0
                    matched_ca[i, j] = 0.0
                    is_fallback[i, j] = False
                    pct_diff_used[i, j] = 0.0
                else:
                    # Paper row has CA == 0, but chamber has positive peak CA (or is undeposited pk_val < 0).
                    # Only an uncontacted chamber (pk_val == 0) is a true zero match;
                    # otherwise assign capped cost.
                    cost[i, j] = cost_cap
                    matched_v[i, j] = peak_v_idx[j]
                    matched_ca[i, j] = max(0.0, pk_val)
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


def classify_value_label(
    paper_val: float,
    peak_val: float,
    used_val: float,
    used_fb: bool,
    firm_pct_tol: float = FIRM_PCT_TOL,
    probable_pct_tol: float = PROBABLE_PCT_TOL,
) -> str:
    """
    Classify how closely our chamber CA reproduces the published paper value:
      - 'firm':      peak-visit match within <= 0.5% (or both 0)
      - 'probable':  peak-visit match within 0.5-5.0%, or fallback visit within <= 5.0%
      - 'unmatched': best matched visit differs by > 5.0% (or undeposited peak_val < 0)
    """
    if peak_val < 0.0:
        return "unmatched"
    if paper_val <= 0.0:
        return "firm" if abs(peak_val) <= 1e-12 else "unmatched"

    pct_peak = abs((peak_val - paper_val) / paper_val) * 100.0 if peak_val > 0.0 else 100.0
    pct_used = abs((used_val - paper_val) / paper_val) * 100.0 if used_val > 0.0 else 100.0

    if pct_used > probable_pct_tol:
        return "unmatched"
    if (not used_fb) and (pct_peak <= firm_pct_tol):
        return "firm"
    return "probable"


def classify_identity_label(
    value_lbl: str,
    margin: float,
    is_forced_slot: bool = False,
    trace_num_val: int | None = None,
    ambiguous_margin: float = AMBIGUOUS_MARGIN,
) -> str:
    """
    Classify whether the chamber-to-paper-row identity assignment is locked/firm:
      - If `trace_num_val is not None and trace_num_val != 1` (e.g. a fallback chamber when
        an experiment is missing deposited chambers), identity cannot be firm (`unmatched` if
        value is unmatched, else `ambiguous` / `probable`).
      - If `is_forced_slot` is True (e.g. `n_trace1 == n_rows` bijection under `trace_num == 1`,
        or zero spare candidate chambers between two locked `firm` neighbors), the chamber
        identity is forced (`firm`).
      - Otherwise, if `margin < ambiguous_margin`, identity is `ambiguous`.
      - Otherwise (`margin >= ambiguous_margin`), identity is `firm` when `value_lbl == 'firm'`,
        `probable` when `value_lbl == 'probable'`, and `unmatched` when `value_lbl == 'unmatched'`.
    """
    if trace_num_val is not None and trace_num_val != 1:
        if value_lbl == "unmatched":
            return "unmatched"
        return "ambiguous" if margin < ambiguous_margin else "probable"

    if is_forced_slot:
        return "firm"

    if margin < ambiguous_margin:
        return "ambiguous" if value_lbl != "unmatched" else "unmatched"

    return value_lbl


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
    Assign one of {'firm', 'probable', 'ambiguous', 'unmatched'} to an aligned row
    (combined value + margin classification).
    """
    val_lbl = classify_value_label(
        paper_val, peak_val, used_val, used_fb, firm_pct_tol=firm_pct_tol, probable_pct_tol=probable_pct_tol
    )
    if val_lbl == "unmatched":
        return "unmatched"
    if margin < ambiguous_margin:
        return "ambiguous"
    return val_lbl


def align_one_group(
    exp_name: str,
    group_id: str,
    df_paper_group: pd.DataFrame,
    df_chambers: pd.DataFrame,
    restrict_trace_num_eq_1: bool = False,
) -> pd.DataFrame:
    """
    Align one group's paper rows onto its experiment's chambers and return row-level DataFrame.
    When `restrict_trace_num_eq_1=True` and `trace_num` is in `df_chambers`:
      - If `sum(trace_num == 1) >= len(df_paper_group)`, restrict candidates strictly to
        `trace_num == 1` chambers.
      - If `sum(trace_num == 1) < len(df_paper_group)` (e.g., Group 7 with 8 undeposited chambers),
        keep all chambers on disk as candidates and add `TRACE_NEQ1_PENALTY` to chambers with
        `trace_num != 1` so all `trace_num == 1` chambers are strongly preferred.
    """
    df_p = df_paper_group.sort_values("cell_index").reset_index(drop=True)
    df_c_all = df_chambers.sort_values("roi_index").reset_index(drop=True)

    has_trace = "trace_num" in df_c_all.columns
    n_rows = len(df_p)
    n_trace1 = int((df_c_all["trace_num"] == 1).sum()) if has_trace else 0

    restricted_strictly = False
    has_synthetic_missing = False
    if restrict_trace_num_eq_1 and has_trace:
        if n_trace1 >= n_rows:
            df_c = df_c_all[df_c_all["trace_num"] == 1].sort_values("roi_index").reset_index(drop=True)
            restricted_strictly = True
        else:
            # Fewer than n_rows chambers on disk have trace_num == 1 because some ROI folders were
            # not deposited in ScienceDB (e.g., Group 7 / v07_6_39_120_0-6_hd9 where MIP_ROI_4..9
            # and MIP_ROI_37..38 are missing from the deposit).
            # Construct the candidate list of all deposited trace_num == 1 chambers PLUS synthetic
            # placeholder entries for the undeposited ROI indices, sorted by roi_index!
            # Any undeposited placeholder has cost = COST_CAP, so the DP assigns the deposited
            # trace_num == 1 chambers to their exact matching paper rows and places the missing-chamber
            # placeholders at the exact paper rows whose chambers were not deposited.
            existing_indices = set(df_c_all["roi_index"].astype(int).tolist())
            # Infer total ROI count from experiment folder name (e.g. v07_6_39_120... -> 39)
            parts = exp_name.split("_")
            total_rois = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else max(existing_indices) + 1
            missing_indices = [idx for idx in range(total_rois) if idx not in existing_indices]

            df_tr1 = df_c_all[df_c_all["trace_num"] == 1].copy()
            vcols_all = [c for c in df_c_all.columns if c.startswith("visit_") and c.endswith("_CA_um2")]
            synth_rows = []
            for m_idx in missing_indices:
                s_row: dict = {
                    "experiment": exp_name,
                    "roi": f"MIP_ROI_{m_idx} (undeposited)",
                    "roi_index": m_idx,
                    "group_id": group_id,
                    "trace_num": -1,
                    "CA_um2": -1.0,
                    "multi_tcell_sweeps": 0,
                }
                for vc in vcols_all:
                    s_row[vc] = 0.0
                synth_rows.append(s_row)
            if synth_rows:
                df_c = (
                    pd.concat([df_tr1, pd.DataFrame(synth_rows)], ignore_index=True)
                    .sort_values("roi_index")
                    .reset_index(drop=True)
                )
                has_synthetic_missing = True
            else:
                df_c = df_c_all.copy()
    else:
        df_c = df_c_all.copy()

    visit_cols = sorted(
        [c for c in df_c.columns if c.startswith("visit_") and c.endswith("_CA_um2")],
        key=lambda c: int(c.split("_")[1]),
    )

    paper_ca = df_p["CA_um2"].to_numpy(dtype=float)
    peak_ca = df_c["CA_um2"].fillna(0.0).to_numpy(dtype=float)
    visit_ca = df_c[visit_cols].fillna(0.0).to_numpy(dtype=float)
    rois = df_c["roi"].astype(str).to_numpy()
    trace_nums = df_c["trace_num"].to_numpy(dtype=int) if has_trace else np.ones(len(df_c), dtype=int)

    cost, matched_v, matched_ca, is_fb, _ = compute_cost_matrix(paper_ca, peak_ca, visit_ca)

    # If n_trace1 < n_rows under tracked mode:
    # - Add a tiny penalty (1e-4) to synthetic undeposited placeholders (`trace_num == -1`) so that
    #   when an unmatched paper row (cost == COST_CAP) can be assigned either to a real deposited
    #   chamber (e.g. truncated MIP_ROI_36 in Group 7) or to a synthetic undeposited placeholder
    #   (`MIP_ROI_37..38`), the DP prefers the real deposited chamber.
    # - If not using synthetic placeholders, penalize `trace_num != 1` chambers by `TRACE_NEQ1_PENALTY`.
    if restrict_trace_num_eq_1 and has_trace and (not restricted_strictly):
        for j in range(len(df_c)):
            if trace_nums[j] == -1:
                cost[:, j] = cost[:, j] + 1e-4
            elif (not has_synthetic_missing) and trace_nums[j] != 1:
                cost[:, j] = cost[:, j] + TRACE_NEQ1_PENALTY

    opt_j, T, _ = solve_monotone_alignment(cost)

    # First pass: resolve optimal j, margin, and value_label for each row
    n_ch = len(df_c)
    chosen_j = np.zeros(n_rows, dtype=int)
    alt_js = np.zeros(n_rows, dtype=int)
    margins = np.zeros(n_rows, dtype=float)
    val_labels: list[str] = []

    for i in range(n_rows):
        j = int(opt_j[i])
        p_val = float(paper_ca[i])
        pk_val = float(peak_ca[j])
        u_val = float(matched_ca[i, j])
        fb = bool(is_fb[i, j])

        if n_ch == n_rows:
            # Exact bijection (e.g. n_trace1 == n_rows): only 1 valid monotonic path exists
            alt_j = j
            margin = np.inf
        else:
            row_T = T[i, :].copy()
            row_T[j] = np.inf
            alt_j = int(np.argmin(row_T))
            margin = float(row_T[alt_j] - T[i, j])

            if margin < AMBIGUOUS_MARGIN:
                mult_j = int(df_c.loc[j, "multi_tcell_sweeps"]) if "multi_tcell_sweeps" in df_c.columns else 0
                mult_alt = int(df_c.loc[alt_j, "multi_tcell_sweeps"]) if "multi_tcell_sweeps" in df_c.columns else 0
                tr_j = int(trace_nums[j])
                tr_alt = int(trace_nums[alt_j])
                if (tr_alt == 1 and tr_j != 1) or (
                    tr_alt == tr_j
                    and ((mult_alt < mult_j) or (mult_alt == mult_j and abs(peak_ca[alt_j] - p_val) < abs(pk_val - p_val)))
                ):
                    j, alt_j = alt_j, j
                    pk_val = float(peak_ca[j])
                    u_val = float(matched_ca[i, j])
                    fb = bool(is_fb[i, j])

        chosen_j[i] = j
        alt_js[i] = alt_j
        margins[i] = margin
        val_labels.append(classify_value_label(p_val, pk_val, u_val, fb))

    # Determine which rows have a forced slot:
    # Either (a) exact bijection `n_ch == n_rows` (under `trace_num == 1`), OR
    # (b) locked between flanking anchors with zero spare candidate chambers (`j_next - j_prev == i_next - i_prev`)
    forced_slot = np.zeros(n_rows, dtype=bool)
    if restricted_strictly and n_ch == n_rows:
        forced_slot[:] = True
    else:
        # Anchors are rows with margin >= AMBIGUOUS_MARGIN and value_label == 'firm' (or fallback within 0.05%)
        # on a deposited trace_num == 1 chamber
        is_anchor = np.array(
            [
                (margins[i] >= AMBIGUOUS_MARGIN)
                and (
                    val_labels[i] == "firm"
                    or (
                        val_labels[i] == "probable"
                        and paper_ca[i] > 0
                        and abs(matched_ca[i, chosen_j[i]] - paper_ca[i]) / paper_ca[i] * 100.0 <= 0.1
                    )
                )
                and (trace_nums[chosen_j[i]] == 1)
                for i in range(n_rows)
            ],
            dtype=bool,
        )
        for i in range(n_rows):
            if is_anchor[i]:
                forced_slot[i] = True
                continue
            # Find nearest preceding anchor (or virtual boundary at i=-1, j=-1)
            prev_i = -1
            prev_j = -1
            for k in range(i - 1, -1, -1):
                if is_anchor[k]:
                    prev_i = k
                    prev_j = int(chosen_j[k])
                    break
            # Find nearest succeeding anchor (or virtual boundary at i=n_rows, j=n_ch)
            next_i = n_rows
            next_j = n_ch
            for k in range(i + 1, n_rows):
                if is_anchor[k]:
                    next_i = k
                    next_j = int(chosen_j[k])
                    break
            # If the number of candidate chambers strictly between prev_j and next_j equals
            # the number of paper rows strictly between prev_i and next_i, the slot is forced!
            if (next_j - prev_j) == (next_i - prev_i) and (trace_nums[chosen_j[i]] == 1):
                forced_slot[i] = True

    rows_out = []
    for i in range(n_rows):
        j = int(chosen_j[i])
        alt_j = int(alt_js[i])
        margin = float(margins[i])
        p_val = float(paper_ca[i])
        pk_val = float(peak_ca[j])
        u_val = float(matched_ca[i, j])
        u_v = int(matched_v[i, j])
        fb = bool(is_fb[i, j])
        t_num_j = int(trace_nums[j]) if has_trace else None

        val_lbl = val_labels[i]
        id_lbl = classify_identity_label(
            value_lbl=val_lbl,
            margin=margin,
            is_forced_slot=bool(forced_slot[i]),
            trace_num_val=t_num_j,
        )
        # Overall `label`:
        # In masks-only mode (no trace_num restriction), `label` is the standard combined label.
        # In tracked mode, when `is_forced_slot` is True (`margin = inf`), `classify_row` uses `margin = inf`
        # so any row with fallback / 0.5-5% diff is `probable` in value (`value_label`), while `identity_label`
        # records that its chamber identity is `firm` (100% locked by `trace_num == 1` + monotonic order).
        eff_margin = np.inf if forced_slot[i] else margin
        overall_lbl = classify_row(p_val, pk_val, u_val, fb, eff_margin)
        if t_num_j is not None and t_num_j != 1 and overall_lbl == "firm":
            overall_lbl = "probable"

        if pk_val < 0.0:
            pct_diff_peak = -100.0
            pct_diff_visit = -100.0
            matched_visit_str = "undeposited"
        else:
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

        margin_out = round(margin, 5) if np.isfinite(margin) else 999.0
        alt_roi_str = "none (forced)" if (not np.isfinite(margin)) else rois[alt_j]

        row_dict = {
            "group_id": str(group_id),
            "cell_index": int(df_p.loc[i, "cell_index"]),
            "experiment": exp_name,
            "roi": rois[j],
            "trace_num": int(t_num_j) if t_num_j is not None else 1,
            "paper_CA_um2": round(p_val, 3),
            "our_CA_um2": round(max(0.0, pk_val), 3),
            "matched_visit": matched_visit_str,
            "pct_diff": round(pct_diff_peak, 3),
            "identity_label": id_lbl,
            "value_label": val_lbl,
            "label": overall_lbl,
            "alt_roi": alt_roi_str,
            "margin": margin_out,
            "_matched_visit_CA_um2": round(max(0.0, u_val), 3),
            "_matched_visit_pct_diff": round(pct_diff_visit, 3),
        }
        rows_out.append(row_dict)

    return pd.DataFrame(rows_out)


def run_chance_baseline(
    n_rows: int,
    df_chambers: pd.DataFrame,
    all_paper_ca: np.ndarray,
    group_paper_ca: np.ndarray,
    restrict_trace_num_eq_1: bool = False,
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
    df_c_all = df_chambers.sort_values("roi_index").reset_index(drop=True)
    if restrict_trace_num_eq_1 and "trace_num" in df_c_all.columns:
        n_tr1 = int((df_c_all["trace_num"] == 1).sum())
        if n_tr1 >= n_rows:
            df_c = df_c_all[df_c_all["trace_num"] == 1].sort_values("roi_index").reset_index(drop=True)
        else:
            df_c = df_c_all
    else:
        df_c = df_c_all

    visit_cols = sorted(
        [c for c in df_c.columns if c.startswith("visit_") and c.endswith("_CA_um2")],
        key=lambda c: int(c.split("_")[1]),
    )
    peak_ca = df_c["CA_um2"].to_numpy(dtype=float)
    visit_ca = df_c[visit_cols].to_numpy(dtype=float)
    n_ch = len(peak_ca)

    rng = np.random.default_rng(seed)

    def _eval_sample(sample_p: np.ndarray) -> int:
        p = sample_p[:, None]  # (n_rows, 1)
        pk = peak_ca[None, :]  # (1, n_ch)

        pos_p = p > 0.0
        pos_pk = pk > 0.0
        both_pos = pos_p & pos_pk
        both_zero = (~pos_p) & (~pos_pk)

        c_peak = np.full((n_rows, n_ch), COST_CAP, dtype=float)
        c_peak[both_zero] = 0.0
        c_peak[both_pos] = np.abs(
            np.log(
                np.maximum(np.broadcast_to(pk, (n_rows, n_ch))[both_pos], 1e-9)
                / np.broadcast_to(p, (n_rows, n_ch))[both_pos]
            )
        )

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
            if n_ch == n_rows:
                margin = np.inf
            else:
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


def make_before_after_figure(
    df_sum_masks: pd.DataFrame,
    df_sum_tracked: pd.DataFrame,
    df_map_tracked: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Create `figures/step3_tracked_vs_masks_only.png`:
      Per-group firm / probable / ambiguous / unmatched before (masks-only) vs after tracking
      (showing both value reproduction and chamber-identity locking across all 13 groups),
      plus tracked vs published CA scatter.
      Strictly follows user house style.
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

    label_colors = {
        "firm": "#1F4E79",       # --plot-1
        "probable": "#2E7D5B",   # --plot-3
        "ambiguous": "#C0561F",  # --plot-2
        "unmatched": "#B3261E",  # --danger
    }

    fig, (ax_bars, ax_scat) = plt.subplots(
        1, 2, figsize=(12.6, 5.0), dpi=200, gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    groups = df_sum_tracked["group_id"].astype(str).tolist()
    m_indexed = df_sum_masks.set_index(df_sum_masks["group_id"].astype(str))
    t_indexed = df_sum_tracked.set_index(df_sum_tracked["group_id"].astype(str))

    x = np.arange(len(groups), dtype=float)
    w = 0.36

    # Left bar of each pair: Before (Masks-only)
    # Right bar of each pair: After (Tracked)
    b_before = np.zeros(len(groups), dtype=float)
    b_after = np.zeros(len(groups), dtype=float)

    for lbl in ["firm", "probable", "ambiguous", "unmatched"]:
        v_before = np.array(
            [float(m_indexed.loc[g, lbl]) if g in m_indexed.index else 0.0 for g in groups],
            dtype=float,
        )
        v_after = np.array([float(t_indexed.loc[g, lbl]) for g in groups], dtype=float)

        ax_bars.bar(
            x - w / 2.0,
            v_before,
            width=w * 0.92,
            bottom=b_before,
            color=label_colors[lbl],
            alpha=0.50,
            edgecolor="#111111",
            linewidth=0.4,
            label="Masks-only (firm)" if lbl == "firm" else None,
        )
        ax_bars.bar(
            x + w / 2.0,
            v_after,
            width=w * 0.92,
            bottom=b_after,
            color=label_colors[lbl],
            alpha=1.0,
            edgecolor="#FFFFFF",
            linewidth=0.4,
            label=f"{lbl.capitalize()}",
        )
        b_before += v_before
        b_after += v_after

    # Mark single-track count per group with a horizontal tick on the right bar
    if "n_trace_num_eq_1" in df_sum_tracked.columns:
        tr1_vals = df_sum_tracked["n_trace_num_eq_1"].to_numpy(dtype=float)
        for idx_g, tr_v in enumerate(tr1_vals):
            ax_bars.plot(
                [x[idx_g] + w * 0.04, x[idx_g] + w * 0.96],
                [tr_v, tr_v],
                color="#111111",
                linestyle="-",
                linewidth=1.6,
                label="Single-track chambers" if idx_g == 0 else None,
                zorder=5,
            )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(groups, fontsize=9.5)
    ax_bars.set_xlabel("Group (left: masks-only; right: tracked)", fontsize=10.5, fontweight="600")
    ax_bars.set_ylabel("Cells", fontsize=10.5, fontweight="600")
    ax_bars.set_title(
        "Alignment by group",
        fontsize=11,
        fontweight="700",
        loc="left",
        pad=8,
    )
    ax_bars.set_ylim(0, max(float(df_sum_tracked["n_rows"].max()) * 1.18, 10.0))
    ax_bars.grid(True, axis="y", color="#E3E3E3", linewidth=0.6)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")

    # Right panel: Tracked CA vs Published CA scatter
    pos_mask = (df_map_tracked["paper_CA_um2"] > 0) & (df_map_tracked["our_CA_um2"] > 0)
    df_pos = df_map_tracked[pos_mask].copy()

    if len(df_pos) > 0:
        vmin = min(float(df_pos["paper_CA_um2"].min()), float(df_pos["our_CA_um2"].min())) * 0.8
        vmax = max(float(df_pos["paper_CA_um2"].max()), float(df_pos["our_CA_um2"].max())) * 1.25
    else:
        vmin, vmax = 10.0, 500.0

    ax_scat.plot([vmin, vmax], [vmin, vmax], color="#5C5C5C", linestyle="--", linewidth=1.0, label="y = x", zorder=1)

    for lbl in ["firm", "probable", "ambiguous", "unmatched"]:
        sub = df_pos[df_pos["label"] == lbl]
        cnt_total = int((df_map_tracked["label"] == lbl).sum())
        if len(sub) == 0 and cnt_total == 0:
            continue
        ax_scat.scatter(
            sub["paper_CA_um2"],
            sub["our_CA_um2"],
            s=28,
            color=label_colors[lbl],
            alpha=0.9,
            edgecolors="#FFFFFF",
            linewidths=0.35,
            label=f"{lbl.capitalize()} (n={cnt_total})",
            zorder=3 if lbl != "firm" else 2,
        )

    ax_scat.set_xscale("log")
    ax_scat.set_yscale("log")
    ax_scat.set_xlim(vmin, vmax)
    ax_scat.set_ylim(vmin, vmax)
    ax_scat.set_xlabel(r"Published CA ($\mu\mathrm{m}^2$)", fontsize=10.5, fontweight="600")
    ax_scat.set_ylabel(r"Tracked CA ($\mu\mathrm{m}^2$)", fontsize=10.5, fontweight="600")
    ax_scat.set_title("Contact area", fontsize=11, fontweight="700", loc="left", pad=8)
    ax_scat.grid(True, which="major", color="#E3E3E3", linewidth=0.6)
    ax_scat.spines["top"].set_visible(False)
    ax_scat.spines["right"].set_visible(False)
    ax_scat.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align chambers to published paper rows.")
    parser.add_argument(
        "--source",
        choices=["masks_only", "tracked"],
        default="masks_only",
        help="Input source: 'masks_only' (data/step3/) or 'tracked' (data/step3_tracked/).",
    )
    args = parser.parse_args()

    is_tracked = args.source == "tracked"
    base_dir = DATA_DIR / ("step3_tracked" if is_tracked else "step3")
    csv_filename = "per_chamber.csv" if is_tracked else "per_chamber_contact_area.csv"

    paper_csv = DATA_DIR / "clean_per_cell_419.csv"
    df_paper = pd.read_csv(paper_csv, dtype={"group_id": str})
    all_paper_ca = df_paper["CA_um2"].to_numpy(dtype=float)

    available_exps: list[tuple[str, str, Path]] = []
    for exp_name, (gid, _cond) in V_TO_GROUP.items():
        ch_csv = base_dir / exp_name / csv_filename
        if ch_csv.is_file():
            available_exps.append((exp_name, str(gid), ch_csv))

    group_rank = {g: idx for idx, g in enumerate(GROUP_ORDER)}
    available_exps.sort(key=lambda item: group_rank.get(item[1], 999))

    if not available_exps:
        print(f"No completed experiment subfolders found in {base_dir}/<experiment>/{csv_filename}.")
        return

    all_map_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for exp_name, gid, ch_csv in available_exps:
        df_ch = pd.read_csv(ch_csv)
        df_pg = df_paper[df_paper["group_id"] == gid].sort_values("cell_index").reset_index(drop=True)

        df_group_map = align_one_group(
            exp_name=exp_name,
            group_id=gid,
            df_paper_group=df_pg,
            df_chambers=df_ch,
            restrict_trace_num_eq_1=is_tracked,
        )
        all_map_dfs.append(df_group_map)

        chance_stats = run_chance_baseline(
            n_rows=len(df_pg),
            df_chambers=df_ch,
            all_paper_ca=all_paper_ca,
            group_paper_ca=df_pg["CA_um2"].to_numpy(dtype=float),
            restrict_trace_num_eq_1=is_tracked,
            n_trials=N_CHANCE_TRIALS,
            seed=RNG_SEED + group_rank.get(gid, 0),
        )

        counts_lbl = df_group_map["label"].value_counts().to_dict()
        counts_id = df_group_map["identity_label"].value_counts().to_dict()
        counts_val = df_group_map["value_label"].value_counts().to_dict()
        n_tr1 = int((df_ch["trace_num"] == 1).sum()) if "trace_num" in df_ch.columns else int(len(df_ch))

        summary_rows.append({
            "group_id": gid,
            "experiment": exp_name,
            "n_rows": int(len(df_pg)),
            "n_chambers": int(len(df_ch)),
            "n_trace_num_eq_1": n_tr1,
            "identity_firm": int(counts_id.get("firm", 0)),
            "identity_probable": int(counts_id.get("probable", 0)),
            "identity_ambiguous": int(counts_id.get("ambiguous", 0)),
            "identity_unmatched": int(counts_id.get("unmatched", 0)),
            "value_firm": int(counts_val.get("firm", 0)),
            "value_probable": int(counts_val.get("probable", 0)),
            "value_unmatched": int(counts_val.get("unmatched", 0)),
            "firm": int(counts_lbl.get("firm", 0)),
            "probable": int(counts_lbl.get("probable", 0)),
            "ambiguous": int(counts_lbl.get("ambiguous", 0)),
            "unmatched": int(counts_lbl.get("unmatched", 0)),
            "chance_firm_mean": chance_stats["chance_firm_mean"],
            "chance_firm_sd": chance_stats["chance_firm_sd"],
            "chance_firm_p95": chance_stats["chance_firm_p95"],
            "chance_jitter_firm_mean": chance_stats["chance_jitter_firm_mean"],
        })

    df_map_all = pd.concat(all_map_dfs, ignore_index=True)
    df_summary_all = pd.DataFrame(summary_rows)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if is_tracked:
        out_map_cols = [
            "group_id",
            "cell_index",
            "experiment",
            "roi",
            "trace_num",
            "paper_CA_um2",
            "our_CA_um2",
            "matched_visit",
            "pct_diff",
            "identity_label",
            "value_label",
            "label",
            "alt_roi",
            "margin",
        ]
        map_csv_path = results_dir / "chamber_to_row_map_tracked.csv"
        df_map_all[out_map_cols].to_csv(map_csv_path, index=False)

        out_sum_cols = [
            "group_id",
            "n_rows",
            "n_chambers",
            "n_trace_num_eq_1",
            "identity_firm",
            "identity_probable",
            "identity_ambiguous",
            "identity_unmatched",
            "value_firm",
            "value_probable",
            "value_unmatched",
            "firm",
            "probable",
            "ambiguous",
            "unmatched",
            "chance_firm_mean",
        ]
        sum_csv_path = results_dir / "chamber_to_row_summary_tracked.csv"
        df_summary_all[out_sum_cols].to_csv(sum_csv_path, index=False)

        old_sum_path = results_dir / "chamber_to_row_summary.csv"
        df_sum_masks = pd.read_csv(old_sum_path, dtype={"group_id": str}) if old_sum_path.is_file() else pd.DataFrame()
        fig_path = FIGURES_DIR / "step3_tracked_vs_masks_only.png"
        make_before_after_figure(df_sum_masks, df_summary_all, df_map_all, fig_path)
    else:
        out_map_cols = [
            "group_id",
            "cell_index",
            "experiment",
            "roi",
            "paper_CA_um2",
            "our_CA_um2",
            "matched_visit",
            "pct_diff",
            "identity_label",
            "value_label",
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
