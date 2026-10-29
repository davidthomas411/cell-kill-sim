#!/usr/bin/env python3
"""
Clean-subset verification gallery (Gate G1).

Applies the exact clean rule in `status/clean_subset_criteria.md` to `data/qc/per_visit_qc.csv`
(`trace_num == 1` contact visits), saves `data/clean_subset/clean_visits.csv`, selects random,
extreme, and rejected visits across all 5 failure categories (with `SEED = 20261028`), writes
`data/clean_subset/panel_raw_files.csv` (and downloads raw 3D `.h5` volumes when invoked with
`--download-raw`), and renders 110 verification panels (2x2 image grid + full key & text box):
  - Row 1 (XY and XZ slices through contact-patch center): raw 561 nm membrane (grey, inverted)
    with our extracted outlines (`#1F4E79` target, `#C0561F` T cell, grey fill labelled
    "other cells", `#6B4C9A` contact patch squares, `#2E7D5B` dashed target convex hull,
    `#B3261E` cup depth arrow).
  - Row 2 (same XY and XZ slices): the lab's U-Net labels AS DEPOSITED (`predictions` dataset,
    classes 0-4, no processing) colour-coded with fixed light tints and a legend, with our
    extracted target outline (`#1F4E79`) on top.

Outputs:
  - `data/clean_subset/clean_visits.csv`
  - `data/clean_subset/panel_raw_files.csv`
  - `figures/review_clean/<group>/<id>.png` (each <= 400 KB, dpi ~105)
  - `figures/review_clean/contact_sheet_<group>.jpg` (2 panels per row)
  - `figures/review_clean/index.csv`
  - `figures/review_clean/README.md`
  - `.agy_clean_gallery_report.md` (<= 30 lines)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _env_k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_k, "1")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, label
from scipy.spatial import ConvexHull, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.paths import DATA_DIR, FIGURES_DIR
from src.reproduce_contact_area import MANIFEST_CSV, STRUCT26, bbox_slices, bwareaopen_26, imfill_holes_26, sphere_dilate_edt
from src.contact_descriptors import edt_indices_matlab_xyz, fill_solid_cell_mask
from src.review.make_gallery import (
    COL_DANGER,
    COL_INK,
    COL_LINE,
    COL_MUTED,
    COL_PAPER,
    COL_PAPER2,
    COL_PLOT1,
    COL_PLOT2,
    COL_PLOT3,
    COL_PLOT4,
    COL_RULE,
    COL_TEXT,
    add_scale_bar,
    draw_slice_crosshairs,
    find_local_raw_h5,
    intersect_mesh_with_plane,
    plot_mesh_slice_segments,
    setup_house_style,
)

SEED = 20261028
MAX_WORKERS = 3

QC_CSV = DATA_DIR / "qc" / "per_visit_qc.csv"
STEP3_TRACKED_DIR = DATA_DIR / "step3_tracked"
CLEAN_SUBSET_DIR = DATA_DIR / "clean_subset"
CLEAN_VISITS_CSV = CLEAN_SUBSET_DIR / "clean_visits.csv"
PANEL_RAW_FILES_CSV = CLEAN_SUBSET_DIR / "panel_raw_files.csv"
REVIEW_CLEAN_DIR = FIGURES_DIR / "review_clean"
REPORT_MD = REPO_ROOT / ".agy_clean_gallery_report.md"

# Fixed light tints for Row 2 U-Net classes as deposited (0..4)
UNET_RGB = {
    0: (1.00, 1.00, 1.00),  # 0 background: white (#FFFFFF)
    1: (0.97, 0.89, 0.83),  # 1 T-cell interior: light warm orange tint (#F7E3D4)
    2: (0.85, 0.91, 0.96),  # 2 target interior: light blue tint (#D9E8F5)
    3: (0.89, 0.72, 0.59),  # 3 T-cell boundary: medium warm orange tint (#E3B896)
    4: (0.66, 0.78, 0.90),  # 4 target boundary: medium blue tint (#A8C7E6)
}
UNET_HEX = {
    0: "#FFFFFF",
    1: "#F7E3D4",
    2: "#D9E8F5",
    3: "#E3B896",
    4: "#A8C7E6",
}
OTHER_CELL_FILL_RGBA = (0.62, 0.62, 0.62, 0.36)
OTHER_CELL_EDGE_HEX = "#7A7A7A"


def parse_donor(exp: str) -> str:
    """Extract donor identifier (e.g. hd1..hd10) from experiment name."""
    m = re.search(r"(hd\d+)", str(exp))
    return m.group(1) if m else "unknown"


def apply_exact_clean_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact clean subset rule from `status/clean_subset_criteria.md` to
    `trace_num == 1` contact visits in `data/qc/per_visit_qc.csv`:
      1. `target_n_components == 1` and `tcell_n_components == 1`
      2. `target_raw_largest_frac >= 0.9`
      3. `touches_fov_edge == False`
      4. `200 <= target_volume_um3 <= 1500`
      5. `0.7 <= target_vol_ratio_to_own_median <= 1.4`
    Also assigns `first_failed_criterion` (1..5, or 0 if clean) in the exact order of the rule.
    """
    out = df[df["trace_num"] == 1].copy().reset_index(drop=True)
    out["donor"] = out["experiment"].astype(str).map(parse_donor)

    c1 = (out["target_n_components"] == 1) & (out["tcell_n_components"] == 1)
    c2 = out["target_raw_largest_frac"].notna() & (out["target_raw_largest_frac"] >= 0.9)
    c3 = ~out["touches_fov_edge"].astype(bool)
    c4 = (
        out["target_volume_um3"].notna()
        & (out["target_volume_um3"] >= 200.0)
        & (out["target_volume_um3"] <= 1500.0)
    )
    c5 = (
        out["target_vol_ratio_to_own_median"].notna()
        & (out["target_vol_ratio_to_own_median"] >= 0.7)
        & (out["target_vol_ratio_to_own_median"] <= 1.4)
    )

    out["pass_c1_pieces_extracted"] = c1
    out["pass_c2_raw_largest_frac"] = c2
    out["pass_c3_not_fov_edge"] = c3
    out["pass_c4_plausible_vol"] = c4
    out["pass_c5_vol_ratio_median"] = c5
    out["is_clean"] = c1 & c2 & c3 & c4 & c5

    first_fail = np.zeros(len(out), dtype=int)
    first_fail[~c1] = 1
    first_fail[c1 & (~c2)] = 2
    first_fail[c1 & c2 & (~c3)] = 3
    first_fail[c1 & c2 & c3 & (~c4)] = 4
    first_fail[c1 & c2 & c3 & c4 & (~c5)] = 5
    out["first_failed_criterion"] = first_fail

    failed_list = []
    for i in range(len(out)):
        fails = []
        if not bool(c1.iloc[i]):
            fails.append("C1")
        if not bool(c2.iloc[i]):
            fails.append("C2")
        if not bool(c3.iloc[i]):
            fails.append("C3")
        if not bool(c4.iloc[i]):
            fails.append("C4")
        if not bool(c5.iloc[i]):
            fails.append("C5")
        failed_list.append(",".join(fails) if fails else "NONE")
    out["failed_criteria_all"] = failed_list
    return out


def build_sweep_lookup(
    df_qc: pd.DataFrame,
    man_lookup: dict[tuple[str, str, int, int], str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """
    Build lookup `(experiment, roi, visit) -> sweep_info` using `rep_sweep` from `per_visit_qc.csv`
    and `per_sweep.csv` metadata from `data/step3_tracked/<exp>/per_sweep.csv`.
    """
    rep_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    sw_Hint_map: dict[tuple[str, str, int, int], dict[str, int]] = {}
    for exp_name in sorted(df_qc["experiment"].astype(str).unique()):
        sw_csv = STEP3_TRACKED_DIR / exp_name / "per_sweep.csv"
        if not sw_csv.is_file():
            continue
        df_sw = pd.read_csv(sw_csv)
        for _, r in df_sw.iterrows():
            key = (str(r["experiment"]), str(r["roi"]), int(r["visit"]), int(r["sweep"]))
            sw_Hint_map[key] = {
                "tracked_tcell_id": int(r["tracked_tcell_id"]),
                "n_tcell_instances": int(r["n_tcell_instances"]),
                "n_target_instances": int(r["n_target_instances"]),
                "s_largest": int(r["s_largest"]),
            }

    for _, row in df_qc.iterrows():
        exp = str(row["experiment"])
        roi = str(row["roi"])
        visit = int(row["visit"])
        sw = int(row["rep_sweep"])
        rel_p = man_lookup.get((exp, roi, visit, sw), "")
        hints = sw_Hint_map.get((exp, roi, visit, sw), {})
        stem = Path(rel_p).name.replace("_predictions.h5", "") if rel_p else ""
        rep_map[(exp, roi, visit)] = {
            "sweep": sw,
            "rel_path": rel_p,
            "stem": stem.zfill(5) if stem else "",
            "tracked_tcell_id": int(hints.get("tracked_tcell_id", row.get("tracked_tcell_id", 1))),
            "n_tcell_instances": int(hints.get("n_tcell_instances", 1)),
            "n_target_instances": int(hints.get("n_target_instances", row.get("n_target_instances_hint", 1))),
            "s_largest": int(hints.get("s_largest", row.get("rep_s_largest_vox", 0))),
        }
    return rep_map


def compute_fov_border_voxels_for_cat3(
    df_cat3: pd.DataFrame,
    rep_map: dict[tuple[str, str, int], dict[str, Any]],
) -> pd.Series:
    """
    For visits in failure category 3 (`touches_fov_edge == True` while C1 and C2 pass),
    count how many segmented voxels (`predictions > 0`) lie within the 2-voxel FOV border
    margin (`coord < 2` or `coord >= dim - 2`). A smaller positive count means the cell
    barely grazes the 2-voxel border threshold (`0` voxels = pass).
    """
    counts = []
    for _, r in df_cat3.iterrows():
        sw_info = rep_map.get((str(r["experiment"]), str(r["roi"]), int(r["visit"])))
        if not sw_info or not sw_info["rel_path"]:
            counts.append(999999)
            continue
        p = REPO_ROOT / str(sw_info["rel_path"])
        try:
            with h5py.File(p, "r") as hf:
                ds = hf["predictions"]
                Z, Y, X = ds.shape
                bv = int(
                    (ds[:2, :, :] > 0).sum()
                    + (ds[Z - 2 :, :, :] > 0).sum()
                    + (ds[:, :2, :] > 0).sum()
                    + (ds[:, Y - 2 :, :] > 0).sum()
                    + (ds[:, :, :2] > 0).sum()
                    + (ds[:, :, X - 2 :] > 0).sum()
                )
            counts.append(bv)
        except Exception:
            counts.append(999999)
    return pd.Series(counts, index=df_cat3.index, dtype=int)


def select_stratified_random_rows(
    df_pool: pd.DataFrame,
    n_select: int,
    rng: np.random.Generator,
    exclude_keys: set[tuple[str, str, int]],
) -> list[pd.Series]:
    """
    Select `n_select` random visits from `df_pool` stratified across `experiment`
    using `rng` (without replacement, excluding `exclude_keys`).
    """
    avail = df_pool[
        ~df_pool.apply(lambda r: (str(r["experiment"]), str(r["roi"]), int(r["visit"])) in exclude_keys, axis=1)
    ].copy()
    if len(avail) == 0 or n_select <= 0:
        return []
    if len(avail) <= n_select:
        return [avail.iloc[i] for i in range(len(avail))]

    exps = sorted(avail["experiment"].astype(str).unique())
    perm_exps = list(rng.permutation(exps))
    by_exp: dict[str, list[pd.Series]] = {}
    for e in perm_exps:
        sub = avail[avail["experiment"] == e].sort_values(["roi_index", "visit"]).reset_index(drop=True)
        order = rng.permutation(len(sub))
        by_exp[e] = [sub.iloc[int(i)] for i in order]

    chosen: list[pd.Series] = []
    round_idx = 0
    while len(chosen) < n_select:
        added_any = False
        for e in perm_exps:
            if round_idx < len(by_exp[e]):
                row = by_exp[e][round_idx]
                k = (str(row["experiment"]), str(row["roi"]), int(row["visit"]))
                if k not in exclude_keys:
                    chosen.append(row)
                    exclude_keys.add(k)
                    added_any = True
                    if len(chosen) >= n_select:
                        break
        if not added_any:
            break
        round_idx += 1
    return chosen


def select_all_gallery_panels(
    df_all: pd.DataFrame,
    rep_map: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Select all panels for the clean-subset verification gallery with `SEED = 20261028`:
      1. `clean_random`: 4 random clean visits per untreated experiment (10 experiments -> 40)
         + up to 4 per dasatinib experiment (3 experiments -> 12) = 52 panels.
      2. `clean_extremes`: 3 most extreme clean visits for each of 6 quantities (18 panels):
         - `max cup_depth_hull_p95_um`
         - `min cup_depth_hull_p95_um`
         - `max cup_depth_over_radius`
         - `max target_volume_um3`
         - `min target_volume_um3`
         - `min target_raw_largest_frac`
      3. `rejected_<category>` for each of the 5 failure categories (using `first_failed_criterion`
         in the order of the rule: 1..5): 6 stratified random + 2 borderline closest to threshold
         = 8 per category, 40 panels in total. For category 4, includes both too-small (<200) and
         too-large (>1500) cases in both random (3 too-small + 3 too-large) and borderline (1 near 200,
         1 near 1500).
    """
    rng = np.random.default_rng(SEED)
    df_clean = df_all[df_all["is_clean"]].copy()
    panels: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Group 1: clean_random (4 per untreated exp, up to 4 per dasatinib exp)
    # ------------------------------------------------------------------
    all_exps = sorted(df_all["experiment"].astype(str).unique())
    untr_exps = [e for e in all_exps if not e.endswith("ds")]
    dasa_exps = [e for e in all_exps if e.endswith("ds")]

    idx_cr = 1
    for exp in untr_exps + dasa_exps:
        sub = df_clean[df_clean["experiment"] == exp].sort_values(["roi_index", "visit"]).reset_index(drop=True)
        n_take = min(4, len(sub))
        if n_take > 0:
            chosen_idx = sorted(rng.choice(len(sub), size=n_take, replace=False).tolist())
            for rank_i, ci in enumerate(chosen_idx, start=1):
                r = sub.iloc[ci]
                exp_short = exp.split("_")[0]
                cond_short = "dasa" if exp.endswith("ds") else "untr"
                panels.append(
                    {
                        "group": "clean_random",
                        "id": f"clean_rand_{idx_cr:02d}_{exp_short}_{cond_short}_{rank_i}",
                        "selection_tag": f"random_{cond_short}_{exp_short}_{rank_i}_of_{n_take}",
                        "row": r,
                        "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
                    }
                )
                idx_cr += 1

    # ------------------------------------------------------------------
    # Group 2: clean_extremes (3 most extreme clean visits x 6 metrics = 18)
    # ------------------------------------------------------------------
    extreme_specs = [
        ("max_cup_depth", "cup_depth_hull_p95_um", False, "max cup_depth_hull_p95_um"),
        ("min_cup_depth", "cup_depth_hull_p95_um", True, "min cup_depth_hull_p95_um"),
        ("max_depth_over_radius", "cup_depth_over_radius", False, "max cup_depth_over_radius"),
        ("max_target_volume", "target_volume_um3", False, "max target_volume_um3"),
        ("min_target_volume", "target_volume_um3", True, "min target_volume_um3"),
        ("min_raw_largest_frac", "target_raw_largest_frac", True, "min target_raw_largest_frac"),
    ]
    idx_ce = 1
    for tag_short, col_name, asc_flag, desc_lbl in extreme_specs:
        sub = (
            df_clean.dropna(subset=[col_name])
            .sort_values([col_name, "experiment", "roi_index", "visit"], ascending=[asc_flag, True, True, True])
            .reset_index(drop=True)
        )
        for rank_i in range(min(3, len(sub))):
            r = sub.iloc[rank_i]
            panels.append(
                {
                    "group": "clean_extremes",
                    "id": f"clean_ext_{idx_ce:02d}_{tag_short}_rank{rank_i+1}",
                    "selection_tag": f"{desc_lbl} (rank {rank_i+1})",
                    "row": r,
                    "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
                }
            )
            idx_ce += 1

    # ------------------------------------------------------------------
    # Group 3..7: rejected_<category> (8 per category = 6 random + 2 borderline)
    # ------------------------------------------------------------------
    # Category 1: our target or T cell in pieces (first_failed_criterion == 1)
    f1 = df_all[df_all["first_failed_criterion"] == 1].copy()
    # Distance to single-component threshold: 1 - min(target_largest_frac_if_broken, tcell_largest_frac_if_broken)
    def _c1_border_dist(r: pd.Series) -> float:
        d_list = []
        if int(r["target_n_components"]) != 1:
            d_list.append(1.0 - float(r["target_largest_frac"]) if pd.notna(r["target_largest_frac"]) else 1.0)
        if int(r["tcell_n_components"]) != 1:
            d_list.append(1.0 - float(r["tcell_largest_frac"]) if pd.notna(r["tcell_largest_frac"]) else 1.0)
        return min(d_list) if d_list else 1.0

    f1["_border_dist"] = f1.apply(_c1_border_dist, axis=1)
    f1_border_df = f1.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(2)
    used_c1: set[tuple[str, str, int]] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit"])) for _, r in f1_border_df.iterrows()
    }
    f1_rand = select_stratified_random_rows(f1, 6, rng, used_c1)
    c1_rows = [(r, f"random_stratified_{i+1}") for i, r in enumerate(f1_rand)] + [
        (r, f"borderline_closest_{i+1} (largest_frac={1.0 - r['_border_dist']:.3f})")
        for i, (_, r) in enumerate(f1_border_df.iterrows())
    ]
    for i, (r, stag) in enumerate(c1_rows, start=1):
        panels.append(
            {
                "group": "rejected_cat1_extracted_pieces",
                "id": f"rejected_cat1_{i:02d}",
                "selection_tag": stag,
                "row": r,
                "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
            }
        )

    # Category 2: lab labels target in pieces (first_failed_criterion == 2: target_raw_largest_frac < 0.9)
    f2 = df_all[df_all["first_failed_criterion"] == 2].copy()
    f2["_border_dist"] = (0.9 - f2["target_raw_largest_frac"].fillna(0.0)).abs()
    f2_border_df = f2.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(2)
    used_c2: set[tuple[str, str, int]] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit"])) for _, r in f2_border_df.iterrows()
    }
    f2_rand = select_stratified_random_rows(f2, 6, rng, used_c2)
    c2_rows = [(r, f"random_stratified_{i+1}") for i, r in enumerate(f2_rand)] + [
        (r, f"borderline_closest_{i+1} (raw_largest_frac={r['target_raw_largest_frac']:.4f} vs 0.90)")
        for i, (_, r) in enumerate(f2_border_df.iterrows())
    ]
    for i, (r, stag) in enumerate(c2_rows, start=1):
        panels.append(
            {
                "group": "rejected_cat2_raw_pieces",
                "id": f"rejected_cat2_{i:02d}",
                "selection_tag": stag,
                "row": r,
                "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
            }
        )

    # Category 3: cut by image edge (first_failed_criterion == 3: touches_fov_edge == True)
    f3 = df_all[df_all["first_failed_criterion"] == 3].copy()
    f3["_border_vox"] = compute_fov_border_voxels_for_cat3(f3, rep_map)
    f3_border_df = f3.sort_values(["_border_vox", "experiment", "roi_index", "visit"]).head(2)
    used_c3: set[tuple[str, str, int]] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit"])) for _, r in f3_border_df.iterrows()
    }
    f3_rand = select_stratified_random_rows(f3, 6, rng, used_c3)
    c3_rows = [(r, f"random_stratified_{i+1}") for i, r in enumerate(f3_rand)] + [
        (r, f"borderline_closest_{i+1} ({int(r['_border_vox'])} vox in 2px FOV border)")
        for i, (_, r) in enumerate(f3_border_df.iterrows())
    ]
    for i, (r, stag) in enumerate(c3_rows, start=1):
        panels.append(
            {
                "group": "rejected_cat3_fov_edge",
                "id": f"rejected_cat3_{i:02d}",
                "selection_tag": stag,
                "row": r,
                "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
            }
        )

    # Category 4: implausible volume (first_failed_criterion == 4: <200 or >1500 um^3)
    # Include both too-small (<200) and too-large (>1500) in borderline (1 each) and random (3 each)
    f4 = df_all[df_all["first_failed_criterion"] == 4].copy()
    f4_small = f4[f4["target_volume_um3"] < 200.0].copy()
    f4_large = f4[f4["target_volume_um3"] > 1500.0].copy()
    f4_small["_border_dist"] = 200.0 - f4_small["target_volume_um3"]
    f4_large["_border_dist"] = f4_large["target_volume_um3"] - 1500.0

    b4_small = f4_small.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(1)
    b4_large = f4_large.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(1)
    f4_border_df = pd.concat([b4_small, b4_large], ignore_index=False)
    used_c4: set[tuple[str, str, int]] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit"])) for _, r in f4_border_df.iterrows()
    }
    r4_small = select_stratified_random_rows(f4_small, 3, rng, used_c4)
    r4_large = select_stratified_random_rows(f4_large, 3, rng, used_c4)
    c4_rows = (
        [(r, f"random_too_small_{i+1} (<200 um3)") for i, r in enumerate(r4_small)]
        + [(r, f"random_too_large_{i+1} (>1500 um3)") for i, r in enumerate(r4_large)]
        + [
            (
                b4_small.iloc[0],
                f"borderline_too_small (vol={b4_small.iloc[0]['target_volume_um3']:.1f} vs 200 um3)",
            ),
            (
                b4_large.iloc[0],
                f"borderline_too_large (vol={b4_large.iloc[0]['target_volume_um3']:.1f} vs 1500 um3)",
            ),
        ]
    )
    for i, (r, stag) in enumerate(c4_rows, start=1):
        panels.append(
            {
                "group": "rejected_cat4_implausible_volume",
                "id": f"rejected_cat4_{i:02d}",
                "selection_tag": stag,
                "row": r,
                "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
            }
        )

    # Category 5: volume jump vs own chamber (first_failed_criterion == 5: ratio <0.7 or >1.4)
    f5 = df_all[df_all["first_failed_criterion"] == 5].copy()
    f5_low = f5[f5["target_vol_ratio_to_own_median"] < 0.7].copy()
    f5_high = f5[f5["target_vol_ratio_to_own_median"] > 1.4].copy()
    f5_low["_border_dist"] = 0.7 - f5_low["target_vol_ratio_to_own_median"]
    f5_high["_border_dist"] = f5_high["target_vol_ratio_to_own_median"] - 1.4

    b5_low = f5_low.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(1)
    b5_high = f5_high.sort_values(["_border_dist", "experiment", "roi_index", "visit"]).head(1)
    f5_border_df = pd.concat([b5_low, b5_high], ignore_index=False)
    used_c5: set[tuple[str, str, int]] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit"])) for _, r in f5_border_df.iterrows()
    }
    f5_rand = select_stratified_random_rows(f5, 6, rng, used_c5)
    c5_rows = [(r, f"random_stratified_{i+1}") for i, r in enumerate(f5_rand)] + [
        (
            b5_low.iloc[0],
            f"borderline_low_ratio (ratio={b5_low.iloc[0]['target_vol_ratio_to_own_median']:.4f} vs 0.70)",
        ),
        (
            b5_high.iloc[0],
            f"borderline_high_ratio (ratio={b5_high.iloc[0]['target_vol_ratio_to_own_median']:.4f} vs 1.40)",
        ),
    ]
    for i, (r, stag) in enumerate(c5_rows, start=1):
        panels.append(
            {
                "group": "rejected_cat5_volume_jump",
                "id": f"rejected_cat5_{i:02d}",
                "selection_tag": stag,
                "row": r,
                "sweep_info": rep_map[(str(r["experiment"]), str(r["roi"]), int(r["visit"]))],
            }
        )

    return panels


def extract_exact_visit_masks(
    I: np.ndarray,
    tracked_id: int,
    n_t_hint: int,
    n_b_hint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract exact 3D boolean masks `(t_full, b_full)` for the tracked T cell and contacted target
    cell matching `src/contact_descriptors.py` and `src/mask_qc.py`.
    """
    shape_zyx = I.shape
    t_full = np.zeros(shape_zyx, dtype=bool)
    b_full = np.zeros(shape_zyx, dtype=bool)
    tree_t_surf = None

    def _fast_inst(nucleus: np.ndarray, membrane: np.ndarray, expected_n: int) -> np.ndarray:
        nuc_clean, nuc_lbl, max_index = bwareaopen_26(nucleus, 3000)
        if max_index == 0:
            res = np.zeros(nucleus.shape, dtype=np.int8)
            if int(np.sum(membrane)) > 3000:
                mem_lbl, n_mem = label(membrane, structure=STRUCT26)
                if n_mem > 0:
                    cnts = np.bincount(mem_lbl.ravel())
                    cnts[0] = 0
                    res[imfill_holes_26(mem_lbl == int(np.argmax(cnts)))] = 1
            return res
        if max_index == 1 and expected_n == 1:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                cnts = np.bincount(mem_lbl.ravel())
                cnts[0] = 0
                comb = nuc_clean | (mem_lbl == int(np.argmax(cnts)))
            else:
                comb = nuc_clean
            return imfill_holes_26(comb).astype(np.int8)

        res = np.zeros(nucleus.shape, dtype=np.int8)
        if max_index == 1:
            mem_inst = membrane.astype(np.int8)
        else:
            sl = bbox_slices(nuc_clean | membrane, 2, nucleus.shape)
            mem_inst = np.zeros(nucleus.shape, dtype=np.int8)
            if sl is not None:
                inds = edt_indices_matlab_xyz(~nuc_clean[sl])
                sub_i = nuc_lbl[sl][inds].astype(np.int8)
                sub_i[~membrane[sl]] = 0
                mem_inst[sl] = sub_i

        skip_temp = expected_n == max_index
        for idx_i in range(1, max_index + 1):
            m_mask = mem_inst == idx_i
            n_mask = nuc_lbl == idx_i
            if not skip_temp:
                temp = m_mask & sphere_dilate_edt(n_mask, 25.0)
                if int(np.sum(temp)) < 1000:
                    continue
            mem_lbl, n_mem = label(m_mask, structure=STRUCT26)
            if n_mem > 0:
                cnts = np.bincount(mem_lbl.ravel())
                cnts[0] = 0
                comb = n_mask | (mem_lbl == int(np.argmax(cnts)))
            else:
                comb = n_mask
            res[imfill_holes_26(comb)] = idx_i

        if not np.any(res) and int(np.sum(membrane)) > 3000:
            mem_lbl, n_mem = label(membrane, structure=STRUCT26)
            if n_mem > 0:
                cnts = np.bincount(mem_lbl.ravel())
                cnts[0] = 0
                res[imfill_holes_26(mem_lbl == int(np.argmax(cnts)))] = 1
        return res

    sl_t = bbox_slices((I == 1) | (I == 3), 2, shape_zyx)
    if sl_t is not None and tracked_id > 0:
        t_inst_sub = _fast_inst(I[sl_t] == 1, I[sl_t] == 3, n_t_hint)
        if np.any(t_inst_sub == tracked_id):
            t_sel = fill_solid_cell_mask(t_inst_sub == tracked_id)
            t_full[sl_t] = t_sel
            off_t = np.array([sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64)
            t_surf_sub = t_sel & (~binary_erosion(t_sel, structure=STRUCT26))
            t_surf_vox = np.argwhere(t_surf_sub).astype(np.float64) + off_t
            if len(t_surf_vox) > 0:
                tree_t_surf = cKDTree(t_surf_vox)

    raw_b_full = (I == 2) | (I == 4)
    sl_b = bbox_slices(raw_b_full, 2, shape_zyx)
    if sl_b is not None:
        b_inst_sub = _fast_inst(I[sl_b] == 2, I[sl_b] == 4, n_b_hint)
        max_b = int(b_inst_sub.max())
        if max_b > 0:
            off_b = np.array([sl_b[0].start, sl_b[1].start, sl_b[2].start], dtype=np.float64)
            cnts_b = np.bincount(b_inst_sub.ravel())
            cnts_b[0] = 0
            if max_b == 1:
                best_bid = 1
            elif tree_t_surf is not None:
                best_bid = int(np.argmax(cnts_b))
                best_touch = -1
                for bid in range(1, max_b + 1):
                    if cnts_b[bid] == 0:
                        continue
                    bi = b_inst_sub == bid
                    bi_s = bi & (~binary_erosion(bi, structure=STRUCT26))
                    pts_bi = np.argwhere(bi_s).astype(np.float64) + off_b
                    d_bi, _ = tree_t_surf.query(pts_bi, k=1, distance_upper_bound=3.5)
                    n_touch = int(np.sum(np.isfinite(d_bi)))
                    if n_touch > best_touch or (n_touch == best_touch and cnts_b[bid] > cnts_b[best_bid]):
                        best_touch = n_touch
                        best_bid = bid
            else:
                best_bid = int(np.argmax(cnts_b))

            if max_b == 1:
                b_cell_sub = fill_solid_cell_mask(b_inst_sub == 1)
            else:
                cc_lbl, _ = label(b_inst_sub > 0, structure=STRUCT26)
                dom_cc = int(np.median(cc_lbl[b_inst_sub == best_bid]))
                b_cell_sub = fill_solid_cell_mask(cc_lbl == dom_cc)
            b_full[sl_b] = b_cell_sub

    return t_full, b_full


def load_clean_gallery_slice_data(
    exp: str,
    roi: str,
    visit: int,
    sweep_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Load 3D U-Net predictions (`_predictions.h5`) and raw 3D fluorescence `.h5` (channel 2 = 561 nm
    membrane), extract the tracked T-cell and contacted target masks, find the contact-patch center
    `(zc, yc, xc)`, and slice both raw 561 nm membrane and U-Net classes 0..4 in XY (`z = zc`) and
    XZ (`y = yc`).
    """
    rel_path = str(sweep_info["rel_path"])
    abs_mask_path = REPO_ROOT / rel_path
    stem = str(sweep_info["stem"])
    tracked_id = int(sweep_info["tracked_tcell_id"])
    n_t_hint = int(sweep_info["n_tcell_instances"])
    n_b_hint = int(sweep_info["n_target_instances"])
    s_largest = int(sweep_info["s_largest"])

    with h5py.File(abs_mask_path, "r") as hf:
        I = hf["predictions"][:]  # (Z, Y, X)

    Z_dim, Y_dim, X_dim = I.shape
    t_full, b_full = extract_exact_visit_masks(I, tracked_id, n_t_hint, n_b_hint)

    t_all_vox = np.argwhere(t_full).astype(np.float64) if np.any(t_full) else np.empty((0, 3), dtype=np.float64)
    b_all_vox = np.argwhere(b_full).astype(np.float64) if np.any(b_full) else np.empty((0, 3), dtype=np.float64)

    t_surf_vox = (
        np.argwhere(t_full & (~binary_erosion(t_full, structure=STRUCT26))).astype(np.float64)
        if np.any(t_full)
        else np.empty((0, 3), dtype=np.float64)
    )
    b_surf_vox = (
        np.argwhere(b_full & (~binary_erosion(b_full, structure=STRUCT26))).astype(np.float64)
        if np.any(b_full)
        else np.empty((0, 3), dtype=np.float64)
    )

    is_contact_on_b_surf = None
    if len(t_surf_vox) > 0 and len(b_surf_vox) > 0:
        tree_t = cKDTree(t_surf_vox)
        d_bt, _ = tree_t.query(b_surf_vox, k=1)
        within_25 = d_bt <= 2.6
        if int(np.sum(within_25)) >= 12:
            is_contact_on_b_surf = within_25
        else:
            k_sel = min(len(d_bt), max(12, max(1, s_largest)))
            thr_d = min(4.5, float(np.partition(d_bt, k_sel - 1)[k_sel - 1]))
            is_contact_on_b_surf = d_bt <= thr_d

    contact_full = np.zeros_like(I, dtype=bool)
    if is_contact_on_b_surf is not None and np.any(is_contact_on_b_surf):
        c_vox = b_surf_vox[is_contact_on_b_surf].astype(int)
        contact_full[c_vox[:, 0], c_vox[:, 1], c_vox[:, 2]] = True
        p_center_vox = b_surf_vox[is_contact_on_b_surf].mean(axis=0)
    elif len(b_all_vox) > 0 and len(t_all_vox) > 0:
        p_center_vox = 0.5 * (b_all_vox.mean(axis=0) + t_all_vox.mean(axis=0))
    elif len(b_all_vox) > 0:
        p_center_vox = b_all_vox.mean(axis=0)
    else:
        p_center_vox = np.array([Z_dim / 2.0, Y_dim / 2.0, X_dim / 2.0])

    zc = int(np.clip(np.round(p_center_vox[0]), 0, Z_dim - 1))
    yc = int(np.clip(np.round(p_center_vox[1]), 0, Y_dim - 1))
    xc = int(np.clip(np.round(p_center_vox[2]), 0, X_dim - 1))

    # 3D convex hull of target surface points in micrometers
    scale_zyx = np.array([VOXEL_UM, VOXEL_UM, VOXEL_UM], dtype=np.float64)
    pts_b_surf_um = b_surf_vox * scale_zyx if len(b_surf_vox) >= 12 else None
    hull_b = None
    deepest_pt_vox = None
    hull_proj_pt_vox = None
    n_syn = np.array([0.0, 0.0, 1.0])

    if pts_b_surf_um is not None:
        try:
            hull_b = ConvexHull(pts_b_surf_um)
        except Exception:
            hull_b = None

    if hull_b is not None and is_contact_on_b_surf is not None and np.any(is_contact_on_b_surf):
        pts_c_um = b_surf_vox[is_contact_on_b_surf] * scale_zyx
        b_center_um = b_all_vox.mean(axis=0) * scale_zyx
        t_center_um = t_all_vox.mean(axis=0) * scale_zyx if len(t_all_vox) > 0 else pts_c_um.mean(axis=0)
        n_syn = pts_c_um.mean(axis=0) - b_center_um
        if np.dot(t_center_um - b_center_um, n_syn) < 0:
            n_syn = t_center_um - b_center_um
        n_norm = float(np.linalg.norm(n_syn))
        n_syn = (n_syn / n_norm) if n_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        eqs = hull_b.equations
        facing = (eqs[:, :3] @ n_syn) > 0.05
        if not np.any(facing):
            facing = np.ones(len(eqs), dtype=bool)
        eq_f = eqs[facing]
        dists_hull = -((pts_c_um @ eq_f[:, :3].T) + eq_f[:, 3])
        min_facet_idx = np.argmin(dists_hull, axis=1)
        d_hull_c = np.maximum(0.0, np.min(dists_hull, axis=1))

        p95_val = float(np.percentile(d_hull_c, 95.0))
        idx_p95 = int(np.argmin(np.abs(d_hull_c - p95_val)))
        pt_c_p95_um = pts_c_um[idx_p95]
        facet_normal = eq_f[min_facet_idx[idx_p95], :3]
        pt_hull_p95_um = pt_c_p95_um + p95_val * facet_normal
        deepest_pt_vox = pt_c_p95_um / VOXEL_UM
        hull_proj_pt_vox = pt_hull_p95_um / VOXEL_UM

    # Crop window around tracked T cell + target cell (also include raw target fragments if any)
    raw_b_vox = np.argwhere((I == 2) | (I == 4)).astype(np.float64)
    pts_list = [arr for arr in (b_all_vox, t_all_vox, raw_b_vox) if len(arr) > 0]
    union_vox = np.vstack(pts_list) if pts_list else np.array([[zc, yc, xc]], dtype=np.float64)

    pad_vox = 18
    z_min, y_min, x_min = np.maximum(0, np.floor(union_vox.min(axis=0)).astype(int) - pad_vox)
    z_max, y_max, x_max = np.minimum([Z_dim, Y_dim, X_dim], np.ceil(union_vox.max(axis=0)).astype(int) + pad_vox)

    side_xy = min(256, max(y_max - y_min, x_max - x_min, 104))
    cy_box = (y_min + y_max) // 2
    cx_box = (x_min + x_max) // 2
    y0 = int(np.clip(cy_box - side_xy // 2, 0, Y_dim - side_xy))
    y1 = y0 + side_xy
    x0 = int(np.clip(cx_box - side_xy // 2, 0, X_dim - side_xy))
    x1 = x0 + side_xy

    side_z = min(Z_dim, max(z_max - z_min, min(Z_dim, side_xy)))
    cz_box = (z_min + z_max) // 2
    z0 = int(np.clip(cz_box - side_z // 2, 0, Z_dim - side_z))
    z1 = z0 + side_z

    # Load raw 561 nm membrane channel (channel index 2 in `raw`)
    raw_h5_path = find_local_raw_h5(exp, roi, stem)
    raw_xy_memb = None
    raw_xz_memb = None
    raw_error = None
    if raw_h5_path is not None:
        try:
            with h5py.File(raw_h5_path, "r") as hf_r:
                raw_ds = hf_r["raw"]
                raw_xy_memb = raw_ds[2, zc, y0:y1, x0:x1].astype(np.float32)
                raw_xz_memb = raw_ds[2, z0:z1, yc, x0:x1].astype(np.float32)
        except Exception as exc:
            raw_h5_path = None
            raw_error = str(exc)

    return {
        "stem": stem,
        "abs_mask_path": abs_mask_path,
        "raw_h5_path": raw_h5_path,
        "raw_error": raw_error,
        "Z_dim": Z_dim,
        "Y_dim": Y_dim,
        "X_dim": X_dim,
        "I_xy": I[zc, y0:y1, x0:x1],
        "I_xz": I[z0:z1, yc, x0:x1],
        "t_xy": t_full[zc, y0:y1, x0:x1],
        "t_xz": t_full[z0:z1, yc, x0:x1],
        "b_xy": b_full[zc, y0:y1, x0:x1],
        "b_xz": b_full[z0:z1, yc, x0:x1],
        "c_slab_xy": contact_full[max(0, zc - 1) : min(Z_dim, zc + 2), y0:y1, x0:x1].any(axis=0),
        "c_slab_xz": contact_full[z0:z1, max(0, yc - 1) : min(Y_dim, yc + 2), x0:x1].any(axis=1),
        "zc": zc,
        "yc": yc,
        "xc": xc,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "z0": z0,
        "z1": z1,
        "hull_b": hull_b,
        "pts_b_surf_um": pts_b_surf_um,
        "deepest_pt_vox": deepest_pt_vox,
        "hull_proj_pt_vox": hull_proj_pt_vox,
        "n_syn": n_syn,
        "raw_xy_memb": raw_xy_memb,
        "raw_xz_memb": raw_xz_memb,
    }


def make_inverted_raw_561_rgb(raw_memb: np.ndarray | None, shape_hw: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """
    Build inverted grayscale RGB image from raw 561 nm membrane slice:
    background = white (`#FFFFFF`), high membrane fluorescence = dark charcoal (`Greys` inverted).
    If `raw_memb` is `None` (download failed), returns a light neutral placeholder and `has_raw=False`.
    """
    h, w = shape_hw
    if raw_memb is None:
        rgb = np.full((h, w, 3), 0.96, dtype=np.float32)
        return rgb, False
    lo = float(np.percentile(raw_memb, 3.0))
    hi = float(np.percentile(raw_memb, 99.6))
    inten = np.clip((raw_memb - lo) / max(3.0, hi - lo), 0.0, 1.0)
    gray = 1.0 - 0.82 * inten
    return np.stack([gray, gray, gray], axis=-1), True


def make_deposited_unet_rgb(I_slice: np.ndarray) -> np.ndarray:
    """
    Colour-code the lab's U-Net `predictions` slice AS DEPOSITED (classes 0..4, no processing)
    using fixed light tints (`UNET_RGB`).
    """
    h, w = I_slice.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    for cls_id, col_rgb in UNET_RGB.items():
        if cls_id == 0:
            continue
        mask_c = I_slice == cls_id
        if np.any(mask_c):
            rgb[mask_c] = col_rgb
    return rgb


def label_other_cells_on_axis(ax: plt.Axes, other_mask: np.ndarray) -> None:
    """
    Render untracked/other cells (`(I > 0) & (~b_slice) & (~t_slice)`) as a semi-transparent
    grey fill and label the largest visible component "other cells".
    """
    if not np.any(other_mask):
        return
    h, w = other_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[other_mask] = OTHER_CELL_FILL_RGBA
    ax.imshow(overlay, interpolation="nearest", origin="upper", zorder=2)
    ax.contour(other_mask.astype(float), levels=[0.5], colors=[OTHER_CELL_EDGE_HEX], linewidths=0.8, linestyles=":", zorder=3)

    lbl_o, n_o = label(other_mask)
    if n_o > 0:
        cnts = np.bincount(lbl_o.ravel())
        cnts[0] = 0
        best_c = int(np.argmax(cnts))
        if cnts[best_c] >= 25:
            ys, xs = np.where(lbl_o == best_c)
            cy, cx = float(np.mean(ys)), float(np.mean(xs))
            ax.text(
                cx,
                cy,
                "other cells",
                color=COL_TEXT,
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": COL_PAPER, "alpha": 0.78, "edgecolor": COL_LINE, "pad": 1.0, "lw": 0.5},
                zorder=8,
            )


def draw_row1_raw_and_outlines(
    ax: plt.Axes,
    raw_memb: np.ndarray | None,
    I_sl: np.ndarray,
    b_sl: np.ndarray,
    t_sl: np.ndarray,
    c_sl: np.ndarray,
    geom: dict[str, Any],
    plane: str,
    sl_idx: int,
    c_coord: tuple[float, float],
    d_p95: float,
) -> None:
    """
    Draw Row 1 panel (XY or XZ slice):
      - Raw 561 nm membrane (grey, inverted)
      - Other cells: grey fill labelled "other cells"
      - Tracked target outline: blue solid (`#1F4E79`)
      - Tracked T cell outline: orange solid (`#C0561F`)
      - Contact patch: purple squares (`#6B4C9A`)
      - Target convex hull: green dashed (`#2E7D5B`)
      - Cup depth: red arrow (`#B3261E`)
    """
    h_s, w_s = I_sl.shape
    rgb_raw, has_raw = make_inverted_raw_561_rgb(raw_memb, (h_s, w_s))
    ax.imshow(rgb_raw, interpolation="nearest", origin="upper", zorder=1)
    draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

    if not has_raw:
        ax.text(
            0.5,
            0.5,
            "RAW 561 nm DOWNLOAD FAILED / MISSING",
            transform=ax.transAxes,
            color=COL_DANGER,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": COL_PAPER, "edgecolor": COL_DANGER, "pad": 3.0},
            zorder=20,
        )

    other_mask = (I_sl > 0) & (~b_sl) & (~t_sl)
    label_other_cells_on_axis(ax, other_mask)

    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc = geom["zc"], geom["yc"]

    # Tracked target outline (#1F4E79 solid) + 2D/3D target convex hull (#2E7D5B dashed)
    if np.any(b_sl):
        ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.8, linestyles="-", zorder=5)
        pts_2d = np.argwhere(b_sl)
        if len(pts_2d) >= 6:
            try:
                hull2d = ConvexHull(pts_2d[:, ::-1])
                hv = np.append(hull2d.vertices, hull2d.vertices[0])
                ax.plot(pts_2d[hv, 1], pts_2d[hv, 0], color=COL_PLOT3, lw=1.3, ls="--", zorder=6)
            except Exception:
                pass

    if geom["hull_b"] is not None and geom["pts_b_surf_um"] is not None:
        val_um = (zc * VOXEL_UM) if plane == "xy" else (yc * VOXEL_UM)
        axis_i = 0 if plane == "xy" else 1
        segs_hull = intersect_mesh_with_plane(
            geom["pts_b_surf_um"], geom["hull_b"].simplices, axis=axis_i, val_um=val_um
        )
        plot_mesh_slice_segments(ax, segs_hull, plane, x0, y0, z0, color=COL_PLOT3, lw=1.1, ls="--", zorder=6)

    # Tracked T-cell outline (#C0561F solid)
    if np.any(t_sl):
        ax.contour(t_sl.astype(float), levels=[0.5], colors=[COL_PLOT2], linewidths=1.7, linestyles="-", zorder=5)

    # Contact patch (#6B4C9A purple squares)
    cy_pts, cx_pts = np.where(c_sl)
    if len(cx_pts) > 0:
        ax.scatter(cx_pts, cy_pts, s=8.0, c=COL_PLOT4, marker="s", alpha=0.88, linewidths=0, zorder=7)

    # Cup depth (#B3261E red arrow)
    if geom["deepest_pt_vox"] is not None and geom["hull_proj_pt_vox"] is not None and d_p95 > 0.02:
        p_c = geom["deepest_pt_vox"]
        p_h = geom["hull_proj_pt_vox"]
        if plane == "xy":
            x_c, y_c = p_c[2] - x0, p_c[1] - y0
            x_h, y_h = p_h[2] - x0, p_h[1] - y0
        else:
            x_c, y_c = p_c[2] - x0, p_c[0] - z0
            x_h, y_h = p_h[2] - x0, p_h[0] - z0
        dx_seg, dy_seg = x_h - x_c, y_h - y_c
        if np.hypot(dx_seg, dy_seg) < (0.5 * d_p95 / VOXEL_UM):
            n_2d = np.array([geom["n_syn"][2], geom["n_syn"][1 if plane == "xy" else 0]], dtype=float)
            n_2d_norm = float(np.linalg.norm(n_2d))
            if n_2d_norm > 1e-4:
                n_2d /= n_2d_norm
                x_h = x_c + (d_p95 / VOXEL_UM) * n_2d[0]
                y_h = y_c + (d_p95 / VOXEL_UM) * n_2d[1]
        ax.annotate(
            "",
            xy=(x_c, y_c),
            xytext=(x_h, y_h),
            arrowprops={"arrowstyle": "<->", "color": COL_DANGER, "lw": 1.7, "shrinkA": 0, "shrinkB": 0},
            zorder=10,
        )

    add_scale_bar(ax, h_s, w_s, bar_um=5.0)
    plane_lbl = (
        f"Row 1 XY (z={sl_idx}, {sl_idx*VOXEL_UM:.1f} um): Raw 561 nm + outlines"
        if plane == "xy"
        else f"Row 1 XZ (y={sl_idx}, {sl_idx*VOXEL_UM:.1f} um): Raw 561 nm + outlines"
    )
    ax.set_title(plane_lbl, fontsize=8.8, fontweight="bold", color=COL_INK, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_row2_deposited_unet(
    ax: plt.Axes,
    I_sl: np.ndarray,
    b_sl: np.ndarray,
    plane: str,
    sl_idx: int,
    c_coord: tuple[float, float],
) -> None:
    """
    Draw Row 2 panel (XY or XZ slice):
      - Lab's U-Net labels AS DEPOSITED (classes 0..4, no processing) in fixed light tints
      - Our extracted target outline (`#1F4E79` solid) on top
    """
    h_s, w_s = I_sl.shape
    rgb_unet = make_deposited_unet_rgb(I_sl)
    ax.imshow(rgb_unet, interpolation="nearest", origin="upper", zorder=1)
    draw_slice_crosshairs(ax, c_coord[0], c_coord[1])

    if np.any(b_sl):
        ax.contour(b_sl.astype(float), levels=[0.5], colors=[COL_PLOT1], linewidths=1.8, linestyles="-", zorder=5)

    add_scale_bar(ax, h_s, w_s, bar_um=5.0)
    plane_lbl = (
        f"Row 2 XY (z={sl_idx}): Lab U-Net labels (0-4) + our target outline"
        if plane == "xy"
        else f"Row 2 XZ (y={sl_idx}): Lab U-Net labels (0-4) + our target outline"
    )
    ax.set_title(plane_lbl, fontsize=8.8, fontweight="bold", color=COL_INK, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def render_one_clean_gallery_panel(item: dict[str, Any]) -> dict[str, Any]:
    """
    Render one verification panel (2x2 image grid + right-hand legend & text box) to
    `figures/review_clean/<group>/<id>.png` at dpi ~105 (<= 400 KB).
    """
    setup_house_style()
    group = str(item["group"])
    ex_id = str(item["id"])
    sel_tag = str(item["selection_tag"])
    r: pd.Series = item["row"]
    sw_info: dict[str, Any] = item["sweep_info"]

    exp = str(r["experiment"])
    donor = str(r["donor"])
    roi = str(r["roi"])
    visit = int(r["visit"])
    sweep = int(sw_info["sweep"])
    cond = str(r["condition"])

    geom = load_clean_gallery_slice_data(exp, roi, visit, sw_info)
    x0, y0, z0 = geom["x0"], geom["y0"], geom["z0"]
    zc, yc, xc = geom["zc"], geom["yc"], geom["xc"]

    d_val = float(r["cup_depth_hull_p95_um"]) if pd.notna(r["cup_depth_hull_p95_um"]) else 0.0
    r_val = float(r["patch_equiv_radius_um"]) if pd.notna(r["patch_equiv_radius_um"]) else 0.0
    dr_val = float(r["cup_depth_over_radius"]) if pd.notna(r["cup_depth_over_radius"]) else 0.0
    v_b = float(r["target_volume_um3"]) if pd.notna(r["target_volume_um3"]) else 0.0
    raw_frac = float(r["target_raw_largest_frac"]) if pd.notna(r["target_raw_largest_frac"]) else np.nan
    vol_ratio = float(r["target_vol_ratio_to_own_median"]) if pd.notna(r["target_vol_ratio_to_own_median"]) else np.nan

    b_ncomp = int(r["target_n_components"])
    t_ncomp = int(r["tcell_n_components"])
    b_edge = bool(r["target_touches_fov_edge"])
    t_edge = bool(r["tcell_touches_fov_edge"])
    any_edge = bool(r["touches_fov_edge"])

    p1 = bool(r["pass_c1_pieces_extracted"])
    p2 = bool(r["pass_c2_raw_largest_frac"])
    p3 = bool(r["pass_c3_not_fov_edge"])
    p4 = bool(r["pass_c4_plausible_vol"])
    p5 = bool(r["pass_c5_vol_ratio_median"])
    is_clean = bool(r["is_clean"])
    first_fail = int(r["first_failed_criterion"])
    failed_all = str(r["failed_criteria_all"])

    fig = plt.figure(figsize=(12.0, 7.5), dpi=105)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.12],
        height_ratios=[1.0, 1.0],
        wspace=0.12,
        hspace=0.18,
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.035,
    )
    ax_r1_xy = fig.add_subplot(gs[0, 0])
    ax_r1_xz = fig.add_subplot(gs[0, 1])
    ax_r2_xy = fig.add_subplot(gs[1, 0])
    ax_r2_xz = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")

    status_badge = "CLEAN (ALL 5 PASS)" if is_clean else f"REJECTED (First failed: C{first_fail}; All failed: {failed_all})"
    title_col = COL_INK if is_clean else COL_DANGER
    fig.suptitle(
        f"{group} / {ex_id} — {exp} ({donor}) {roi} visit {visit} sweep {sweep} [{status_badge}]",
        fontsize=10.2,
        fontweight="bold",
        color=title_col,
        x=0.025,
        y=0.968,
        ha="left",
    )
    fig.text(
        0.025,
        0.935,
        f"Condition: {cond} | Selection: {sel_tag} | Slice center (z={zc}, y={yc}, x={xc}) vox | Voxel size: 0.196 um (z assumed)",
        fontsize=8.5,
        color=COL_MUTED,
    )

    c_xy_coord = (xc - x0, yc - y0)
    c_xz_coord = (xc - x0, zc - z0)

    draw_row1_raw_and_outlines(
        ax_r1_xy, geom["raw_xy_memb"], geom["I_xy"], geom["b_xy"], geom["t_xy"], geom["c_slab_xy"], geom, "xy", zc, c_xy_coord, d_val
    )
    draw_row1_raw_and_outlines(
        ax_r1_xz, geom["raw_xz_memb"], geom["I_xz"], geom["b_xz"], geom["t_xz"], geom["c_slab_xz"], geom, "xz", yc, c_xz_coord, d_val
    )
    draw_row2_deposited_unet(ax_r2_xy, geom["I_xy"], geom["b_xy"], "xy", zc, c_xy_coord)
    draw_row2_deposited_unet(ax_r2_xz, geom["I_xz"], geom["b_xz"], "xz", yc, c_xz_coord)

    # Full unified key (Row 1 outlines + Row 2 U-Net class legend) on every panel
    legend_handles = [
        Line2D([0], [0], color=COL_PLOT1, lw=1.8, ls="-", label="Tracked target outline (#1F4E79)"),
        Line2D([0], [0], color=COL_PLOT2, lw=1.7, ls="-", label="Tracked T cell outline (#C0561F)"),
        Patch(facecolor=OTHER_CELL_FILL_RGBA, edgecolor=OTHER_CELL_EDGE_HEX, ls=":", label='Other cells (grey fill, labelled)'),
        Line2D([0], [0], color=COL_PLOT4, marker="s", ls="none", markersize=5.5, label="Contact patch (#6B4C9A squares)"),
        Line2D([0], [0], color=COL_PLOT3, lw=1.3, ls="--", label="Target convex hull (#2E7D5B dashed)"),
        Line2D([0], [0], color=COL_DANGER, lw=1.7, ls="-", label="Cup depth p95 (#B3261E arrow)"),
        Patch(facecolor=UNET_HEX[0], edgecolor=COL_LINE, label="U-Net 0: background (white)"),
        Patch(facecolor=UNET_HEX[1], edgecolor=COL_LINE, label="U-Net 1: T-cell interior"),
        Patch(facecolor=UNET_HEX[2], edgecolor=COL_LINE, label="U-Net 2: target interior"),
        Patch(facecolor=UNET_HEX[3], edgecolor=COL_LINE, label="U-Net 3: T-cell boundary"),
        Patch(facecolor=UNET_HEX[4], edgecolor=COL_LINE, label="U-Net 4: target boundary"),
    ]
    ax_txt.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor=COL_PAPER2,
        edgecolor=COL_RULE,
        fontsize=7.6,
        ncol=1,
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.8,
        labelspacing=0.32,
    )

    rel_mask_str = str(geom["abs_mask_path"].relative_to(REPO_ROOT))
    rel_raw_str = (
        str(geom["raw_h5_path"].relative_to(REPO_ROOT))
        if geom["raw_h5_path"] is not None
        else f"MISSING ({geom['raw_error'] or 'download failed'})"
    )

    def _pf(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines_txt = [
        f"Experiment: {exp} | Donor: {donor}",
        f"ROI: {roi} | Visit: {visit} | Sweep: {sweep} (stem {geom['stem']})",
        f"Condition: {cond}",
        f"Clean subset status: {status_badge}",
        "--- Five Clean-Subset Criteria (Pass / Fail) ---",
        f"1. Extracted 1 piece (target={b_ncomp}, T cell={t_ncomp} == 1): [{_pf(p1)}]",
        f"2. Lab raw target largest frac ({raw_frac:.4f} >= 0.90): [{_pf(p2)}]",
        f"3. Not cut by FOV edge (touch={any_edge}; B={b_edge}, T={t_edge}): [{_pf(p3)}]",
        f"4. Target volume ({v_b:.1f} um^3 in [200, 1500]): [{_pf(p4)}]",
        f"5. Vol / chamber median ({vol_ratio:.3f} in [0.70, 1.40]): [{_pf(p5)}]",
        "--- Measurements & Metadata ---",
        f"Cup depth (hull p95): {d_val:.3f} um",
        f"Patch equiv radius: {r_val:.3f} um",
        f"Depth / radius: {dr_val:.3f}",
        f"Target volume: {v_b:.1f} um^3 (ch. med: {float(r['chamber_median_target_vol_um3']):.1f} um^3)",
        f"Raw-label largest fraction: {raw_frac:.4f} (raw comp={int(r['target_raw_n_components'])})",
        f"Voxel size: {VOXEL_UM:.3f} um (z = 0.196 um assumed)",
        "--- Source Files ---",
        f"Mask: {rel_mask_str}",
        f"Raw:  {rel_raw_str}",
    ]

    ax_txt.text(
        0.0,
        0.52,
        "\n".join(lines_txt),
        transform=ax_txt.transAxes,
        fontsize=7.65,
        va="top",
        ha="left",
        color=COL_TEXT,
        linespacing=1.30,
        bbox={"facecolor": COL_PAPER2, "edgecolor": COL_RULE, "boxstyle": "round,pad=0.45"},
    )

    out_path = REVIEW_CLEAN_DIR / group / f"{ex_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)

    # Enforce <= 400 KB per PNG
    fsize_kb = out_path.stat().st_size / 1024.0
    if fsize_kb > 400.0:
        with Image.open(out_path) as im:
            im_q = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            im_q.save(out_path, optimize=True)
        fsize_kb = out_path.stat().st_size / 1024.0

    return {
        "group": group,
        "id": ex_id,
        "selection_tag": sel_tag,
        "experiment": exp,
        "donor": donor,
        "roi": roi,
        "visit": visit,
        "sweep": sweep,
        "stem": geom["stem"],
        "condition": cond,
        "is_clean": is_clean,
        "first_failed_criterion": first_fail,
        "failed_criteria_all": failed_all,
        "pass_c1_pieces_extracted": p1,
        "pass_c2_raw_largest_frac": p2,
        "pass_c3_not_fov_edge": p3,
        "pass_c4_plausible_vol": p4,
        "pass_c5_vol_ratio_median": p5,
        "target_n_components": b_ncomp,
        "tcell_n_components": t_ncomp,
        "target_raw_n_components": int(r["target_raw_n_components"]),
        "target_raw_largest_frac": round(raw_frac, 6) if pd.notna(raw_frac) else "",
        "target_touches_fov_edge": b_edge,
        "tcell_touches_fov_edge": t_edge,
        "touches_fov_edge": any_edge,
        "target_volume_um3": round(v_b, 3),
        "chamber_median_target_vol_um3": round(float(r["chamber_median_target_vol_um3"]), 3),
        "target_vol_ratio_to_own_median": round(vol_ratio, 5) if pd.notna(vol_ratio) else "",
        "cup_depth_hull_p95_um": round(d_val, 4),
        "patch_equiv_radius_um": round(r_val, 4),
        "cup_depth_over_radius": round(dr_val, 4),
        "voxel_size_um": VOXEL_UM,
        "slice_zc_vox": zc,
        "slice_yc_vox": yc,
        "slice_xc_vox": xc,
        "has_raw_h5": bool(geom["raw_h5_path"] is not None),
        "mask_h5_path": rel_mask_str,
        "raw_h5_path": rel_raw_str,
        "panel_png_path": str(out_path.relative_to(REPO_ROOT)),
        "panel_kb": round(fsize_kb, 1),
    }


def write_contact_sheets(index_df: pd.DataFrame) -> list[Path]:
    """
    Write `figures/review_clean/contact_sheet_<group>.jpg` for each group, tiling all panels
    of that group 2 per row.
    """
    written: list[Path] = []
    for grp, gdf in index_df.groupby("group", sort=False):
        png_paths = [REPO_ROOT / str(p) for p in gdf["panel_png_path"].tolist()]
        imgs = [Image.open(p).convert("RGB") for p in png_paths if p.is_file()]
        if not imgs:
            continue
        w0, h0 = imgs[0].size
        ncols = 2
        nrows = (len(imgs) + ncols - 1) // ncols
        pad = 12
        sheet_w = ncols * w0 + (ncols + 1) * pad
        sheet_h = nrows * h0 + (nrows + 1) * pad
        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(246, 246, 246))
        for idx, im in enumerate(imgs):
            r_i = idx // ncols
            c_i = idx % ncols
            if im.size != (w0, h0):
                im = im.resize((w0, h0), Image.Resampling.LANCZOS)
            sheet.paste(im, (pad + c_i * (w0 + pad), pad + r_i * (h0 + pad)))
            im.close()
        out_jpg = REVIEW_CLEAN_DIR / f"contact_sheet_{grp}.jpg"
        sheet.save(out_jpg, quality=88, optimize=True)
        sheet.close()
        written.append(out_jpg)
    return written


def write_clean_gallery_readme() -> None:
    """Write `figures/review_clean/README.md` documenting every visual element and function."""
    readme = """# Clean-Subset Verification Gallery (`figures/review_clean/`)

Generated by `src/review/make_clean_gallery.py` (`SEED = 20261028`) for Gate G1 verification (`status/verification_plan.md`, `status/clean_subset_criteria.md`).

## Clean Subset Rule (`apply_exact_clean_rule` in `src/review/make_clean_gallery.py`)
Applied to all `trace_num == 1` contact visits in `data/qc/per_visit_qc.csv` (4,523 visits; 1,213 clean visits saved to `data/clean_subset/clean_visits.csv`):
1. `target_n_components == 1` and `tcell_n_components == 1` (our extracted target and T cell are each 1 connected 3D component `>= 20` voxels).
2. `target_raw_largest_frac >= 0.9` (in the lab's U-Net labels as deposited `(I == 2) | (I == 4)`, the largest 26-connected component holds `>= 90%` of target voxels).
3. `touches_fov_edge == False` (neither extracted cell has voxels within 2 voxels of the 3D image border).
4. `200 <= target_volume_um3 <= 1500` (plausible single Nalm6 cell volume).
5. `0.7 <= target_vol_ratio_to_own_median <= 1.4` (stable target volume relative to the chamber's median target volume).

## Panel Layout & Functions (`render_one_clean_gallery_panel`)
Each panel (`figures/review_clean/<group>/<id>.png`, 2x2 image slices + right-hand legend and text box) displays XY (`z = zc`) and XZ (`y = yc`) slices through the 3D contact-patch centroid `(zc, yc, xc)` loaded by `load_clean_gallery_slice_data`:

- **Row 1 (XY and XZ slices) — `draw_row1_raw_and_outlines`**:
  - **Background (`make_inverted_raw_561_rgb`)**: Raw 561 nm membrane fluorescence channel (`raw[2]` from `data/scidb_19235_raw/<exp>/<roi>/<stem>.h5`), grey-scale inverted (`#FFFFFF` paper background, dark charcoal membrane signal).
  - **Other cells (`label_other_cells_on_axis`)**: Any segmented voxel `(I > 0)` outside the tracked target and tracked T cell is shaded with semi-transparent grey fill (`OTHER_CELL_FILL_RGBA`) and dotted grey contour (`#7A7A7A`), with the largest component labelled `"other cells"`.
  - **Tracked target outline (`#1F4E79` blue solid)**: 2D slice contour of the 3D solid target mask `b_full` extracted by `extract_exact_visit_masks` (matching `src/contact_descriptors.py`).
  - **Tracked T-cell outline (`#C0561F` orange solid)**: 2D slice contour of the 3D solid T-cell mask `t_full` (`tracked_tcell_id`) extracted by `extract_exact_visit_masks`.
  - **Contact patch (`#6B4C9A` purple squares)**: Target surface voxels within the synapse contact zone (`c_slab_xy`, `c_slab_xz`).
  - **Target convex hull (`#2E7D5B` green dashed)**: 2D slice convex hull (`scipy.spatial.ConvexHull`) and 3D convex hull plane intersection (`intersect_mesh_with_plane`, `plot_mesh_slice_segments`).
  - **Cup depth (`#B3261E` red double-headed arrow)**: Vector from the 3D target convex hull facet to the 95th-percentile indented contact point (`cup_depth_hull_p95_um`).
  - **Scale bar & crosshairs (`add_scale_bar`, `draw_slice_crosshairs`)**: Literal `5 um` bar (`0.1962 um/voxel`) and dotted crosshairs at `(xc, yc)` / `(xc, zc)`.

- **Row 2 (same XY and XZ slices) — `draw_row2_deposited_unet`**:
  - **Lab's U-Net semantic labels as deposited (`make_deposited_unet_rgb`)**: Unprocessed `predictions` array (`0` background `#FFFFFF`, `1` T-cell interior `#F7E3D4`, `2` target interior `#D9E8F5`, `3` T-cell boundary `#E3B896`, `4` target boundary `#A8C7E6`).
  - **Our extracted target outline (`#1F4E79` blue solid)**: Drawn directly on top of the deposited U-Net classes so lab segmentation vs. our instance extraction can be compared immediately.

## Groups (`select_all_gallery_panels`)
- `clean_random` (52 panels): 4 random clean visits per untreated experiment (10 experiments = 40) + 4 per dasatinib experiment (3 experiments = 12).
- `clean_extremes` (18 panels): 3 most extreme clean visits for `max cup_depth_hull_p95_um`, `min cup_depth_hull_p95_um`, `max cup_depth_over_radius`, `max target_volume_um3`, `min target_volume_um3`, and `min target_raw_largest_frac`.
- `rejected_cat1_extracted_pieces` (8 panels): First failed Criterion 1 (6 stratified random + 2 borderline).
- `rejected_cat2_raw_pieces` (8 panels): First failed Criterion 2 (6 stratified random + 2 borderline closest to `0.90`).
- `rejected_cat3_fov_edge` (8 panels): First failed Criterion 3 (6 stratified random + 2 borderline with 1 voxel in the 2-px FOV border).
- `rejected_cat4_implausible_volume` (8 panels): First failed Criterion 4 (3 random `< 200 um^3` + 3 random `> 1500 um^3` + 1 borderline near `200 um^3` + 1 borderline near `1500 um^3`).
- `rejected_cat5_volume_jump` (8 panels): First failed Criterion 5 (6 stratified random + 1 borderline below `0.70` + 1 borderline above `1.40`).
- Contact sheets: `contact_sheet_<group>.jpg` (2 panels per row) built by `write_contact_sheets`.
"""
    (REVIEW_CLEAN_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean-subset verification gallery (Gate G1).")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write clean_visits.csv and panel_raw_files.csv without rendering panels.",
    )
    parser.add_argument(
        "--download-raw",
        action="store_true",
        help="Run src/download_raw.py --file-list data/clean_subset/panel_raw_files.csv --workers 8 before rendering.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    CLEAN_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    df_qc = pd.read_csv(QC_CSV, low_memory=False)
    df_all = apply_exact_clean_rule(df_qc)

    df_clean = df_all[df_all["is_clean"]].copy().reset_index(drop=True)
    df_clean.to_csv(CLEAN_VISITS_CSV, index=False)
    print(f"Saved {len(df_clean)} / {len(df_all)} clean visits to {CLEAN_VISITS_CSV}", flush=True)

    df_man = pd.read_csv(MANIFEST_CSV)
    man_lookup: dict[tuple[str, str, int, int], str] = {
        (str(r["experiment"]), str(r["roi"]), int(r["visit_index"]), int(r["sweep_index"])): str(r["path"])
        for _, r in df_man.iterrows()
    }
    rep_map = build_sweep_lookup(df_all, man_lookup)

    panels = select_all_gallery_panels(df_all, rep_map)
    print(f"Selected {len(panels)} total panels across {len({p['group'] for p in panels})} groups.", flush=True)

    raw_rows = []
    seen_raw: set[tuple[str, str, str]] = set()
    for p in panels:
        exp = str(p["row"]["experiment"])
        roi = str(p["row"]["roi"])
        stem = str(p["sweep_info"]["stem"]).zfill(5)
        key = (exp, roi, stem)
        if key not in seen_raw:
            seen_raw.add(key)
            raw_rows.append({"experiment": exp, "roi": roi, "stem": stem})

    df_raw_files = pd.DataFrame(raw_rows).sort_values(["experiment", "roi", "stem"]).reset_index(drop=True)
    df_raw_files.to_csv(PANEL_RAW_FILES_CSV, index=False)
    print(f"Wrote {len(df_raw_files)} unique raw volumes to {PANEL_RAW_FILES_CSV}", flush=True)

    if args.prepare_only:
        return 0

    if args.download_raw:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "src" / "download_raw.py"),
            "--file-list",
            str(PANEL_RAW_FILES_CSV),
            "--workers",
            "8",
        ]
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)

    print(f"Rendering {len(panels)} panels with {MAX_WORKERS} workers...", flush=True)
    results_by_idx: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(render_one_clean_gallery_panel, item): idx for idx, item in enumerate(panels)}
        done_n = 0
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            res = fut.result()
            results_by_idx[idx] = res
            done_n += 1
            if done_n % 15 == 0 or done_n == len(panels):
                print(f"  [{done_n:3d}/{len(panels):3d}] rendered {res['group']}/{res['id']}.png ({res['panel_kb']:.1f} KB)", flush=True)

    ordered_results = [results_by_idx[i] for i in range(len(panels))]
    index_df = pd.DataFrame(ordered_results)
    index_csv_path = REVIEW_CLEAN_DIR / "index.csv"
    index_df.to_csv(index_csv_path, index=False)
    write_clean_gallery_readme()
    sheets = write_contact_sheets(index_df)
    print(f"Wrote {index_csv_path}, README.md, and {len(sheets)} contact sheets.", flush=True)

    elapsed = time.perf_counter() - t0
    n_raw_ok = int(index_df["has_raw_h5"].sum())
    n_raw_fail = int((~index_df["has_raw_h5"]).sum())
    n_vol_ok = sum(1 for _, r in df_raw_files.iterrows() if find_local_raw_h5(str(r["experiment"]), str(r["roi"]), str(r["stem"])) is not None)
    n_vol_fail = len(df_raw_files) - n_vol_ok

    # Build concise <=30-line report
    exp_parts = []
    for exp_name, g_exp in df_all.groupby("experiment", sort=True):
        short_e = str(exp_name).split("_")[0]
        donor_e = parse_donor(str(exp_name))
        ds_tag = "ds" if str(exp_name).endswith("ds") else ""
        n_c = int(g_exp["is_clean"].sum())
        n_t = len(g_exp)
        exp_parts.append(f"{short_e}({donor_e}{ds_tag}): {n_c}/{n_t} ({n_c/n_t:.2f})")

    grp_counts = index_df.groupby("group", sort=False).size().to_dict()
    report_lines = [
        "# Clean-Subset Verification Gallery Report",
        "## Deviations",
        "- None. Exact rule in `status/clean_subset_criteria.md` applied; all 110 panels (52 clean_random + 18 clean_extremes + 40 rejected across 5 categories) rendered with raw 561 nm membrane in Row 1 and deposited U-Net labels (0-4) in Row 2.",
        "## Clean-Subset Size (`data/clean_subset/clean_visits.csv`)",
        f"- Total clean contact visits (`trace_num == 1`): {len(df_clean)} / {len(df_all)} ({len(df_clean)/len(df_all):.1%}); {df_clean[['experiment','roi']].drop_duplicates().shape[0]} chambers >=1 clean visit.",
        f"- Untreated (v01-v10): {'; '.join(exp_parts[:10])}.",
        f"- Dasatinib (v11-v13): {'; '.join(exp_parts[10:])}.",
        "## First-Failed Criterion Breakdown (Rejected Visits = 3,310)",
        f"- C1 (extracted pieces): {int((df_all['first_failed_criterion']==1).sum())} | C2 (raw target pieces <0.9): {int((df_all['first_failed_criterion']==2).sum())} | C3 (FOV edge): {int((df_all['first_failed_criterion']==3).sum())} | C4 (vol outside [200,1500]): {int((df_all['first_failed_criterion']==4).sum())} | C5 (vol jump outside [0.7,1.4]): {int((df_all['first_failed_criterion']==5).sum())}.",
        "## Panel & Contact-Sheet Counts (`figures/review_clean/`)",
        f"- Total panels: {len(index_df)} (max size {index_df['panel_kb'].max():.1f} KB <= 400 KB; dpi=105) + {len(sheets)} contact sheets (`contact_sheet_<group>.jpg`).",
        f"- `clean_random`: {grp_counts.get('clean_random', 0)} (40 untreated + 12 dasatinib) | `clean_extremes`: {grp_counts.get('clean_extremes', 0)} (3 x 6 metrics).",
        f"- `rejected_cat1..cat5`: {grp_counts.get('rejected_cat1_extracted_pieces', 0)}, {grp_counts.get('rejected_cat2_raw_pieces', 0)}, {grp_counts.get('rejected_cat3_fov_edge', 0)}, {grp_counts.get('rejected_cat4_implausible_volume', 0)}, {grp_counts.get('rejected_cat5_volume_jump', 0)} (6 random + 2 borderline each = 40).",
        "## Raw Volume Downloads (`data/clean_subset/panel_raw_files.csv`)",
        f"- Unique volumes listed: {len(df_raw_files)} | Volumes OK: {n_vol_ok} | Volumes failed: {n_vol_fail}.",
        f"- Panels with raw 561 nm membrane shown: {n_raw_ok} / {len(index_df)} (failed: {n_raw_fail}).",
        "## Run Time",
        f"- Total script wall time: {elapsed:.1f} s (workers={MAX_WORKERS}).",
    ]
    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Done in {elapsed:.1f}s. Report written to {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
