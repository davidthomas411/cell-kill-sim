#!/usr/bin/env python3
"""Chamber movie pipeline for visual review.

Builds two clips per chamber from 3D raw light-sheet volumes (`data/scidb_19235_raw/<exp>/<roi>/<stem>.h5`,
dataset `raw` of shape `(3, z, y, x)` uint16) and U-Net masks (`data/scidb_19235_masks/<exp>/<roi>/<stem>_predictions.h5`,
dataset `predictions` with classes 0..4):
  (a) `visits`: one sweep per visit across all downloaded visits (10 min apart)
  (b) `burst`: within-visit sweep burst across all downloaded sweeps of a chosen visit

Visual encoding (preserving `src/review/make_chamber_movie.py` look):
  - Channel 0 (405 nm): death dye -> yellow above 404 ADU after sigma=3.0 2D Gaussian filter
  - Channel 1 (488 nm): T cell -> green (fixed display range across chamber movie)
  - Channel 2 (561 nm): target cells -> magenta (fixed display range across chamber movie)
  - Display denoise: light 3D Gaussian filter (sigma = 0.7 vox) on display channels only
  - Cropping: union 3D bounding box of tracked cells over the movie plus 5 um (26 vox) margin
  - Outlines: tracked target (blue), tracked T cell (orange) using the exact extraction from
    `src/review/make_gallery.py`, and other cells (thin grey); `--raw-masks` outlines raw U-Net classes.
  - Strip chart under top view (XY, max over z) and side view (XZ, max over y):
    per-visit `cup_depth_hull_p95_um`, `contact_area_um2`, `target_volume_um3` from
    `data/step5/per_visit_descriptors.csv` & `data/qc/per_visit_qc.csv`, and `dead_visit`
    from `data/step4/per_visit_death_labels_all_v2.csv`, with a vertical cursor at the current visit.
  - Outputs per clip: H.264 MP4 (`yuv420p`, `crf 24`, `faststart`, `2 fps`), animated WebP fallback,
    poster frame JPEG (`_poster.jpg`), and `figures/movies/index.csv`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import os
import subprocess
import sys
import tempfile
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
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.paths import DATA_DIR, FIGURES_DIR
from src.reproduce_contact_area import STRUCT26, bbox_slices, bwareaopen_26, imfill_holes_26, sphere_dilate_edt
from src.contact_descriptors import edt_indices_matlab_xyz, fill_solid_cell_mask

RAW_DIR = DATA_DIR / "scidb_19235_raw"
MANIFEST_CSV = RAW_DIR / "manifest.csv"
MASKS_DIR = DATA_DIR / "scidb_19235_masks"
STEP3_TRACKED_DIR = DATA_DIR / "step3_tracked"
STEP4_VISIT_CSV = DATA_DIR / "step4" / "per_visit_death_labels_all_v2.csv"
STEP4_CHAMBER_CSV = DATA_DIR / "step4" / "per_chamber_death_labels_all_v2.csv"
STEP5_VISIT_CSV = DATA_DIR / "step5" / "per_visit_descriptors.csv"
STEP5_SWEEP_CSV = DATA_DIR / "step5" / "per_sweep_descriptors.csv"
QC_VISIT_CSV = DATA_DIR / "qc" / "per_visit_qc.csv"
MOVIES_DIR = FIGURES_DIR / "movies"
INDEX_CSV = MOVIES_DIR / "index.csv"

ZOOM_W = 2
MARGIN_UM = 5.0
MARGIN_VOX = int(math.ceil(MARGIN_UM / VOXEL_UM))  # 26 voxels
DENOISE_SIGMA_VOX = 0.7
FPS = 2

# Outline colors matching prototype and review conventions
COL_TARGET_OUTLINE = (90, 160, 255)   # blue tracked target
COL_TCELL_OUTLINE = (255, 150, 40)    # orange tracked T cell
COL_OTHER_OUTLINE = (150, 150, 150)   # thin grey other cells


def _load_font(size: int = 11, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if not bold
        else [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Menlo.ttc",
        ]
    )
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def extract_tracked_and_other_masks(
    I: np.ndarray,
    tracked_id: int,
    n_t_hint: int,
    n_b_hint: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D boolean masks `(t_full, b_full, other_full)` for the tracked T cell,
    contacted target cell, and other cells in the chamber using the exact instance
    extraction from `src/review/make_gallery.py` / `src/contact_descriptors.py`.
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
                mem_lbl, n_mem = ndi.label(membrane, structure=STRUCT26)
                if n_mem > 0:
                    cnts = np.bincount(mem_lbl.ravel())
                    cnts[0] = 0
                    res[imfill_holes_26(mem_lbl == int(np.argmax(cnts)))] = 1
            return res
        if max_index == 1 and expected_n == 1:
            mem_lbl, n_mem = ndi.label(membrane, structure=STRUCT26)
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
            mem_lbl, n_mem = ndi.label(m_mask, structure=STRUCT26)
            if n_mem > 0:
                cnts = np.bincount(mem_lbl.ravel())
                cnts[0] = 0
                comb = n_mask | (mem_lbl == int(np.argmax(cnts)))
            else:
                comb = n_mask
            res[imfill_holes_26(comb)] = idx_i

        if not np.any(res) and int(np.sum(membrane)) > 3000:
            mem_lbl, n_mem = ndi.label(membrane, structure=STRUCT26)
            if n_mem > 0:
                cnts = np.bincount(mem_lbl.ravel())
                cnts[0] = 0
                res[imfill_holes_26(mem_lbl == int(np.argmax(cnts)))] = 1
        return res

    sl_t = bbox_slices((I == 1) | (I == 3), 2, shape_zyx)
    if sl_t is not None and tracked_id > 0:
        t_inst_sub = _fast_inst(I[sl_t] == 1, I[sl_t] == 3, n_t_hint)
        if not np.any(t_inst_sub == tracked_id) and int(t_inst_sub.max()) > 0:
            # Fallback for sweep 6 or untracked sweep: pick largest T-cell instance
            cnts_t = np.bincount(t_inst_sub.ravel())
            cnts_t[0] = 0
            t_pick = int(np.argmax(cnts_t))
        else:
            t_pick = tracked_id
        if np.any(t_inst_sub == t_pick):
            t_sel = fill_solid_cell_mask(t_inst_sub == t_pick)
            t_full[sl_t] = t_sel
            off_t = np.array([sl_t[0].start, sl_t[1].start, sl_t[2].start], dtype=np.float64)
            t_surf_sub = t_sel & (~ndi.binary_erosion(t_sel, structure=STRUCT26))
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
                    bi_s = bi & (~ndi.binary_erosion(bi, structure=STRUCT26))
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
                cc_lbl, _ = ndi.label(b_inst_sub > 0, structure=STRUCT26)
                dom_cc = int(np.median(cc_lbl[b_inst_sub == best_bid]))
                b_cell_sub = fill_solid_cell_mask(cc_lbl == dom_cc)
            b_full[sl_b] = b_cell_sub

    # Other cells in chamber: U-Net foreground outside a 2-voxel buffer of tracked cells
    tracked_union = b_full | t_full
    if np.any(tracked_union):
        excl = ndi.binary_dilation(tracked_union, iterations=2)
        other_full = (I > 0) & (~excl)
    else:
        other_full = I > 0

    return t_full, b_full, other_full


def outline_2d(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask & (~ndi.binary_erosion(mask))


def load_metadata_tables() -> dict[str, Any]:
    """Load manifest, step5 descriptors, QC flags, step4 v2 death calls, and step3 sweep hints."""
    man_rows = list(csv.DictReader(open(MANIFEST_CSV, "r", encoding="utf-8")))
    df_s5_v = pd.read_csv(STEP5_VISIT_CSV, low_memory=False)
    df_s5_v = df_s5_v[df_s5_v["trace_num"] == 1].copy()

    df_qc_v = pd.read_csv(QC_VISIT_CSV, low_memory=False)
    df_qc_v = df_qc_v[df_qc_v["trace_num"] == 1].copy()

    df_s4_v = pd.read_csv(STEP4_VISIT_CSV, low_memory=False)
    df_s4_v = df_s4_v[df_s4_v["trace_num"] == 1].copy()

    df_s4_c = pd.read_csv(STEP4_CHAMBER_CSV, low_memory=False)
    df_s4_c = df_s4_c[df_s4_c["trace_num"] == 1].copy()

    df_s5_sw = pd.read_csv(STEP5_SWEEP_CSV, low_memory=False)
    df_s5_sw = df_s5_sw[df_s5_sw["trace_num"] == 1].copy()

    # Build per-sweep tracking hint lookup from data/step3_tracked/<exp>/per_sweep.csv
    sw_hints: dict[tuple[str, str, int, int], dict[str, int]] = {}
    for exp_dir in sorted(STEP3_TRACKED_DIR.iterdir()):
        sw_csv = exp_dir / "per_sweep.csv"
        if not sw_csv.is_file():
            continue
        df_sw = pd.read_csv(sw_csv)
        df_sw = df_sw[df_sw["trace_num"] == 1]
        for _, r in df_sw.iterrows():
            key = (str(r["experiment"]), str(r["roi"]), int(r["visit"]), int(r["sweep"]))
            sw_hints[key] = {
                "tracked_tcell_id": int(r["tracked_tcell_id"]),
                "n_tcell_instances": int(r["n_tcell_instances"]),
                "n_target_instances": int(r["n_target_instances"]),
            }

    return {
        "man_rows": man_rows,
        "df_s5_v": df_s5_v,
        "df_qc_v": df_qc_v,
        "df_s4_v": df_s4_v,
        "df_s4_c": df_s4_c,
        "df_s5_sw": df_s5_sw,
        "sw_hints": sw_hints,
    }


def find_eligible_chambers(man_rows: list[dict[str, str]]) -> list[tuple[str, str, bool]]:
    """
    Return all `(exp, roi, has_all_visits)` chambers that have at least one sweep per visit
    fully downloaded (`ok == 'True'` in `manifest.csv` and `.h5` file exists on disk).
    """
    by_ch: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in man_rows:
        by_ch.setdefault((r["experiment"], r["roi"]), []).append(r)

    eligible: list[tuple[str, str, bool]] = []
    for (exp, roi), rlist in sorted(
        by_ch.items(),
        key=lambda kv: (kv[0][0], int(kv[0][1].split("_")[-1]) if kv[0][1].split("_")[-1].isdigit() else kv[0][1]),
    ):
        all_visits = sorted({int(r["visit_index"]) for r in rlist})
        if not all_visits:
            continue
        ok_visits = sorted(
            {
                int(r["visit_index"])
                for r in rlist
                if r.get("ok") == "True" and Path(r["path"]).is_file()
            }
        )
        if len(ok_visits) == len(all_visits):
            eligible.append((exp, roi, True))
    return eligible


def choose_burst_visit(
    exp: str,
    roi: str,
    man_ch_rows: list[dict[str, str]],
    df_qc_ch: pd.DataFrame,
    df_s5_ch: pd.DataFrame,
    override_visit: int | None = None,
) -> int:
    """
    Choose the visit for the within-visit sweep burst clip (b):
    prefers clean contact visits with multiple downloaded sweeps and maximum `cup_depth_hull_p95_um`.
    """
    sweeps_by_v: dict[int, list[int]] = {}
    for r in man_ch_rows:
        if r.get("ok") == "True" and Path(r["path"]).is_file():
            v = int(r["visit_index"])
            sweeps_by_v.setdefault(v, []).append(int(r["sweep_index"]))

    multi_visits = {v for v, sws in sweeps_by_v.items() if len(sws) >= 2}
    if not multi_visits:
        multi_visits = set(sweeps_by_v.keys())

    if override_visit is not None and override_visit in multi_visits:
        return override_visit

    if len(df_qc_ch) > 0:
        sub = df_qc_ch[df_qc_ch["visit"].isin(multi_visits)].copy()
        if len(sub) > 0:
            clean = sub[
                (~sub["flag_fragmented"].astype(bool))
                & (~sub["flag_edge"].astype(bool))
                & (sub["target_volume_um3"].between(200.0, 1500.0))
                & (sub["target_vol_ratio_to_own_median"].between(0.7, 1.4))
            ]
            pool = clean if len(clean) > 0 else sub
            pool_valid = pool.dropna(subset=["cup_depth_hull_p95_um"])
            if len(pool_valid) > 0:
                best_row = pool_valid.sort_values("cup_depth_hull_p95_um", ascending=False).iloc[0]
                return int(best_row["visit"])

    if len(df_s5_ch) > 0:
        sub5 = df_s5_ch[df_s5_ch["visit"].isin(multi_visits)].dropna(subset=["cup_depth_hull_p95_um"])
        if len(sub5) > 0:
            return int(sub5.sort_values("cup_depth_hull_p95_um", ascending=False).iloc[0]["visit"])

    return min(multi_visits) if multi_visits else 1


def draw_strip_chart(
    dr: ImageDraw.ImageDraw,
    x0_box: int,
    y0_box: int,
    w_box: int,
    h_box: int,
    n_visits: int,
    visit_metrics: dict[int, dict[str, Any]],
    current_visit: int,
    death_visit_sustained: int | None,
    sweep_metrics: dict[str, Any] | None,
    font_sm: ImageFont.ImageFont,
    font_b: ImageFont.ImageFont,
) -> None:
    """
    Draw the 4-row per-visit strip chart under the XY and XZ views:
      1. `cup_depth_hull_p95_um` (um)
      2. `contact_area_um2` (um^2)
      3. `target_volume_um3` (um^3)
      4. `dead_visit` (Step 4 v2 per-visit death call)
    with a vertical cursor at `current_visit`.
    """
    # Background panel for strip chart
    dr.rectangle([x0_box, y0_box, x0_box + w_box - 1, y0_box + h_box - 1], fill=(14, 14, 18), outline=(55, 55, 65))

    pad_l = 118
    pad_r = 108
    pad_t = 22
    pad_b = 22
    plot_x0 = x0_box + pad_l
    plot_x1 = x0_box + w_box - pad_r
    plot_w = max(60, plot_x1 - plot_x0)

    # Title bar inside strip chart
    cur_m = visit_metrics.get(current_visit, {})
    qc_state = cur_m.get("qc_label", "no contact")
    hdr_txt = f"Per-visit tracked target trajectory (cursor: visit {current_visit}, {qc_state})"
    if sweep_metrics is not None:
        sw_d = sweep_metrics.get("cup_depth_hull_p95_um")
        sw_ca = sweep_metrics.get("contact_area_um2")
        sw_v = sweep_metrics.get("target_volume_um3")
        d_s = f"{sw_d:.2f}" if (sw_d is not None and not np.isnan(sw_d)) else "—"
        ca_s = f"{sw_ca:.1f}" if (sw_ca is not None and not np.isnan(sw_ca)) else "—"
        v_s = f"{sw_v:.0f}" if (sw_v is not None and not np.isnan(sw_v)) else "—"
        hdr_txt = f"Per-visit trajectory (visit {current_visit})  |  Current sweep: depth={d_s} um, CA={ca_s} um2, vol={v_s} um3"
    dr.text((x0_box + 8, y0_box + 4), hdr_txt, fill=(220, 220, 230), font=font_b)

    def _vx(v: int) -> int:
        if n_visits <= 1:
            return plot_x0 + plot_w // 2
        return int(round(plot_x0 + (v - 1) * plot_w / (n_visits - 1)))

    tracks = [
        ("cup_depth_hull_p95_um", "Cup depth (um)", (110, 185, 255), "{:.2f} um", 0.0, 4.0),
        ("contact_area_um2", "Contact (um2)", (255, 165, 70), "{:.1f} um2", 0.0, 140.0),
        ("target_volume_um3", "Target vol (um3)", (130, 220, 165), "{:.0f} um3", 0.0, 1400.0),
    ]

    avail_h = h_box - pad_t - pad_b
    row_h = avail_h // 4
    cursor_x = _vx(current_visit)

    # Draw vertical gridlines every 5 visits and vertical cursor line
    for v in range(1, n_visits + 1):
        vx = _vx(v)
        if v == 1 or v % 5 == 0 or v == n_visits:
            dr.line([(vx, y0_box + pad_t), (vx, y0_box + h_box - pad_b)], fill=(34, 34, 42), width=1)

    # Sustained death visit vertical dashed indicator if present
    if death_visit_sustained is not None and 1 <= death_visit_sustained <= n_visits:
        dx = _vx(death_visit_sustained)
        for yy in range(y0_box + pad_t, y0_box + h_box - pad_b, 6):
            dr.line([(dx, yy), (dx, min(yy + 3, y0_box + h_box - pad_b))], fill=(255, 110, 100), width=1)

    # Vertical cursor line at current visit
    dr.line([(cursor_x, y0_box + pad_t - 2), (cursor_x, y0_box + h_box - pad_b + 2)], fill=(255, 255, 255), width=2)

    for idx, (col_key, lbl, col_rgb, fmt_str, default_lo, default_hi) in enumerate(tracks):
        ry0 = y0_box + pad_t + idx * row_h + 3
        ry1 = y0_box + pad_t + (idx + 1) * row_h - 4
        rh = max(12, ry1 - ry0)

        vals = [
            float(visit_metrics[v][col_key])
            for v in range(1, n_visits + 1)
            if v in visit_metrics
            and visit_metrics[v].get(col_key) is not None
            and not np.isnan(float(visit_metrics[v][col_key]))
        ]
        v_lo = 0.0
        v_hi = max(default_hi, max(vals) * 1.12) if vals else default_hi
        if v_hi <= v_lo:
            v_hi = v_lo + 1.0

        # Horizontal baseline and midline
        dr.line([(plot_x0, ry1), (plot_x1, ry1)], fill=(50, 50, 62), width=1)
        dr.line([(plot_x0, (ry0 + ry1) // 2), (plot_x1, (ry0 + ry1) // 2)], fill=(28, 28, 36), width=1)

        # Label on left
        dr.text((x0_box + 8, ry0 + (rh - 12) // 2), lbl, fill=col_rgb, font=font_sm)

        # Plot polyline and markers
        pts: list[tuple[int, int, int, bool]] = []
        for v in range(1, n_visits + 1):
            val = visit_metrics.get(v, {}).get(col_key)
            if val is None or np.isnan(float(val)):
                continue
            fval = float(val)
            frac = np.clip((fval - v_lo) / (v_hi - v_lo), 0.0, 1.0)
            vy = int(round(ry1 - frac * rh))
            vx = _vx(v)
            is_clean = bool(visit_metrics.get(v, {}).get("is_clean_qc", True))
            pts.append((v, vx, vy, is_clean))

        for i in range(len(pts) - 1):
            if pts[i + 1][0] - pts[i][0] <= 2:
                dr.line([(pts[i][1], pts[i][2]), (pts[i + 1][1], pts[i + 1][2])], fill=col_rgb, width=2)
            else:
                dr.line([(pts[i][1], pts[i][2]), (pts[i + 1][1], pts[i + 1][2])], fill=(80, 80, 95), width=1)

        for v, vx, vy, is_clean in pts:
            r_dot = 4 if v == current_visit else 2
            if v == current_visit:
                dr.ellipse([vx - r_dot - 1, vy - r_dot - 1, vx + r_dot + 1, vy + r_dot + 1], fill=(255, 255, 255))
                dr.ellipse([vx - r_dot + 1, vy - r_dot + 1, vx + r_dot - 1, vy + r_dot - 1], fill=col_rgb)
            elif is_clean:
                dr.ellipse([vx - r_dot, vy - r_dot, vx + r_dot, vy + r_dot], fill=col_rgb)
            else:
                # QC-flagged contact visit shown as hollow grey-ringed marker
                dr.ellipse([vx - r_dot, vy - r_dot, vx + r_dot, vy + r_dot], fill=(20, 20, 25), outline=(150, 150, 160))

        # Current visit numeric readout on right
        cur_val = cur_m.get(col_key)
        if cur_val is not None and not np.isnan(float(cur_val)):
            val_txt = fmt_str.format(float(cur_val))
        else:
            val_txt = "—"
        dr.text((plot_x1 + 8, ry0 + (rh - 12) // 2), val_txt, fill=(240, 240, 245), font=font_b)

    # Row 4: Death call (`dead_visit` from step4 *_v2)
    ry0 = y0_box + pad_t + 3 * row_h + 4
    ry1 = y0_box + pad_t + 4 * row_h - 4
    rh = max(10, ry1 - ry0)
    dr.text((x0_box + 8, ry0 + (rh - 12) // 2), "Death call (v2)", fill=(245, 215, 90), font=font_sm)

    step_w = max(3, plot_w // max(1, n_visits) - 2)
    for v in range(1, n_visits + 1):
        vx = _vx(v)
        is_dead = bool(visit_metrics.get(v, {}).get("dead_visit", False))
        has_art = bool(visit_metrics.get(v, {}).get("artifact_405_band", False))
        bx0 = vx - step_w // 2
        bx1 = vx + step_w // 2
        if is_dead:
            fill_c = (240, 200, 50)
            out_c = (255, 240, 150) if v == current_visit else (190, 150, 30)
        elif has_art:
            fill_c = (120, 60, 60)
            out_c = (180, 90, 90)
        else:
            fill_c = (42, 55, 48)
            out_c = (255, 255, 255) if v == current_visit else (70, 85, 78)
        dr.rectangle([bx0, ry0 + 2, bx1, ry1 - 2], fill=fill_c, outline=out_c)

    cur_dead = bool(cur_m.get("dead_visit", False))
    dead_str = "DEAD (405+)" if cur_dead else "ALIVE"
    dead_col = (255, 215, 70) if cur_dead else (175, 215, 185)
    dr.text((plot_x1 + 8, ry0 + (rh - 12) // 2), dead_str, fill=dead_col, font=font_b)

    # X-axis visit / time ticks at bottom
    axis_y = y0_box + h_box - pad_b + 4
    for v in range(1, n_visits + 1):
        if v == 1 or v % 5 == 0 or v == n_visits:
            vx = _vx(v)
            lbl_v = f"v{v} ({v*10}m)" if (v == 1 or v == n_visits or v == 10) else f"v{v}"
            dr.text((vx - 10, axis_y), lbl_v, fill=(175, 175, 185), font=font_sm)


def render_clip_frames(
    exp: str,
    roi: str,
    clip_name: str,
    frame_rows: list[dict[str, str]],
    crop_box: tuple[int, int, int, int, int, int],
    lo: list[float],
    hi: list[float],
    masks_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    raw_cache: dict[str, np.ndarray],
    visit_metrics: dict[int, dict[str, Any]],
    sweep_metrics_map: dict[tuple[int, int], dict[str, Any]],
    n_visits_total: int,
    condition_str: str,
    fate_str: str,
    death_visit_sustained: int | None,
    use_raw_masks: bool,
) -> list[Image.Image]:
    """Render all frames of a clip (`visits` or `burst`) as PIL RGB images with even dimensions."""
    z0, z1, y0, y1, x0, x1 = crop_box
    exp_short = exp.split("_")[0]

    font_hdr = _load_font(13, bold=True)
    font_sub = _load_font(11, bold=False)
    font_b = _load_font(11, bold=True)
    font_sm = _load_font(10, bold=False)

    def _norm(img: np.ndarray, c: int) -> np.ndarray:
        return np.clip((img - lo[c]) / max(1.0, hi[c] - lo[c]), 0.0, 1.0)

    def _composite_and_outline(
        raw_crop: np.ndarray,
        pred_crop: np.ndarray,
        t_crop: np.ndarray,
        b_crop: np.ndarray,
        other_crop: np.ndarray,
        axis: int,
    ) -> Image.Image:
        # Channels 1 and 2 use 3D Gaussian denoised display volume (`raw_crop[1]`, `raw_crop[2]`)
        t_mip = _norm(raw_crop[1].max(axis), 1)  # channel 1: T cell -> green
        g_mip = _norm(raw_crop[2].max(axis), 2)  # channel 2: target cells -> magenta
        # Channel 0: death dye -> yellow above 404 ADU after sigma=3.0 2D Gaussian filter
        d_mip = np.clip((ndi.gaussian_filter(raw_crop[0].max(axis), 3.0) - 404.0) / 20.0, 0.0, 1.0)
        rgb = np.stack([g_mip + d_mip, t_mip + 0.8 * d_mip, g_mip], axis=-1)
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        if use_raw_masks:
            tgt_proj = ((pred_crop == 2) | (pred_crop == 4)).max(axis)
            tc_proj = ((pred_crop == 1) | (pred_crop == 3)).max(axis)
            rgb_u8[outline_2d(tgt_proj)] = COL_TARGET_OUTLINE
            rgb_u8[outline_2d(tc_proj)] = COL_TCELL_OUTLINE
            im = Image.fromarray(rgb_u8).resize(
                (rgb_u8.shape[1] * ZOOM_W, rgb_u8.shape[0] * ZOOM_W), Image.NEAREST
            )
            return im

        # Extracted tracked target (blue) and tracked T cell (orange) at 2px, other cells thin 1px grey
        tgt_proj = b_crop.max(axis)
        tc_proj = t_crop.max(axis)
        other_proj = other_crop.max(axis) & (~tgt_proj) & (~tc_proj)

        rgb_u8[outline_2d(tgt_proj)] = COL_TARGET_OUTLINE
        rgb_u8[outline_2d(tc_proj)] = COL_TCELL_OUTLINE
        im = Image.fromarray(rgb_u8).resize(
            (rgb_u8.shape[1] * ZOOM_W, rgb_u8.shape[0] * ZOOM_W), Image.NEAREST
        )
        if np.any(other_proj):
            im_arr = np.array(im)
            other_up = np.repeat(np.repeat(other_proj, ZOOM_W, axis=0), ZOOM_W, axis=1)
            thin_other = outline_2d(other_up)
            im_arr[thin_other] = COL_OTHER_OUTLINE
            im = Image.fromarray(im_arr)
        return im

    frames: list[Image.Image] = []
    is_burst = clip_name.startswith("burst")

    for r in frame_rows:
        stem = str(r["stem"]).zfill(5)
        v = int(r["visit_index"])
        s = int(r["sweep_index"])
        raw_crop = raw_cache[stem]
        pred_crop, t_crop, b_crop, other_crop = masks_cache[stem]

        panel_xy = _composite_and_outline(raw_crop, pred_crop, t_crop, b_crop, other_crop, axis=0)
        panel_xz = _composite_and_outline(raw_crop, pred_crop, t_crop, b_crop, other_crop, axis=1)

        header_h = 42
        gap_views = 8
        strip_h = 186
        footer_h = 36

        min_canvas_w = 540
        canvas_w = max(panel_xy.width + 16, min_canvas_w)
        if canvas_w % 2 == 1:
            canvas_w += 1
        canvas_h = header_h + panel_xy.height + gap_views + panel_xz.height + 8 + strip_h + footer_h
        if canvas_h % 2 == 1:
            canvas_h += 1

        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        x_pad = (canvas_w - panel_xy.width) // 2
        y_xy = header_h
        y_xz = header_h + panel_xy.height + gap_views
        y_strip = y_xz + panel_xz.height + 8
        y_foot = y_strip + strip_h

        canvas.paste(panel_xy, (x_pad, y_xy))
        canvas.paste(panel_xz, (x_pad, y_xz))

        dr = ImageDraw.Draw(canvas)

        # Header lines
        t_min = v * 10
        sw_sfx = " (6th vol: tubulin ch2)" if s == 6 else f" (+{(s-1)*15}s)"
        mode_lbl = f"within-visit burst (visit {v})" if is_burst else "all visits (10 min intervals)"
        dr.text(
            (8, 4),
            f"{exp_short} {roi}  |  visit {v}/{n_visits_total}  t = {t_min} min  |  sweep {s}/6{sw_sfx}  [{mode_lbl}]",
            fill=(255, 255, 255),
            font=font_hdr,
        )
        qc_lbl = visit_metrics.get(v, {}).get("qc_label", "no contact")
        dr.text(
            (8, 22),
            f"{condition_str}  |  200m outcome: {fate_str}  |  visit {v} QC: {qc_lbl}  |  crop: union bbox + 5 um",
            fill=(195, 195, 205),
            font=font_sub,
        )

        # View labels
        dr.text((x_pad + 6, y_xy + 4), "top view (XY, max over z)", fill=(220, 220, 220), font=font_b)
        dr.text((x_pad + 6, y_xz + 4), "side view (XZ, max over y)", fill=(220, 220, 220), font=font_b)

        # 5 um scale bar on both XY and XZ panels
        px_5um = int(round((5.0 / VOXEL_UM) * ZOOM_W))
        for y_panel_bot in (y_xy + panel_xy.height, y_xz + panel_xz.height):
            sb_x1 = x_pad + panel_xy.width - 10
            sb_x0 = sb_x1 - px_5um
            sb_y = y_panel_bot - 10
            dr.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=(255, 255, 255), width=3)
            dr.text((sb_x0 + 4, sb_y - 14), "5 um", fill=(255, 255, 255), font=font_sm)

        # Strip chart under the views
        sw_m = sweep_metrics_map.get((v, s)) if is_burst else None
        draw_strip_chart(
            dr=dr,
            x0_box=6,
            y0_box=y_strip,
            w_box=canvas_w - 12,
            h_box=strip_h,
            n_visits=n_visits_total,
            visit_metrics=visit_metrics,
            current_visit=v,
            death_visit_sustained=death_visit_sustained,
            sweep_metrics=sw_m,
            font_sm=font_sm,
            font_b=font_b,
        )

        # Footer stating current channel understanding and outline key
        dr.text(
            (8, y_foot + 3),
            "Channels (current understanding): ch0 (405)=death dye (>404 ADU, yellow) | ch1 (488)=T cell (green) | ch2 (561)=target cells (magenta)",
            fill=(215, 215, 215),
            font=font_sm,
        )
        if use_raw_masks:
            out_txt = "Outlines (--raw-masks): raw U-Net target classes 2|4 (blue), raw U-Net T-cell classes 1|3 (orange)"
        else:
            out_txt = "Outlines (make_gallery extraction): tracked target (blue), tracked T cell (orange), other cells (thin grey)"
        dr.text((8, y_foot + 18), out_txt, fill=(185, 185, 195), font=font_sm)

        frames.append(canvas)

    return frames


def encode_mp4_webp_poster(
    frames: list[Image.Image],
    mp4_path: Path,
    webp_path: Path,
    poster_path: Path,
    fps: int = FPS,
) -> tuple[int, int, int]:
    """
    Encode `frames` to:
      - H.264 MP4 (`yuv420p`, `crf 24`, `faststart`, `2 fps`)
      - Animated WebP fallback (targeting `<= 1 MB`)
      - Poster frame JPEG (`_poster.jpg`) from the midpoint/peak frame
    Returns `(mp4_bytes, webp_bytes, poster_bytes)`.
    """
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Save poster frame JPEG (use middle frame or frame 0)
    poster_idx = min(len(frames) - 1, len(frames) // 2)
    frames[poster_idx].save(poster_path, format="JPEG", quality=85, optimize=True)

    # Encode MP4 via ffmpeg
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for i, im in enumerate(frames):
            im.save(td_p / f"frame_{i:03d}.png", format="PNG")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(td_p / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Encode animated WebP fallback (target <= 1 MB)
    duration_ms = int(round(1000.0 / fps))
    for q in (78, 65, 50, 38):
        frames[0].save(
            webp_path,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            quality=q,
            method=4,
        )
        if webp_path.stat().st_size <= 1_000_000:
            break

    return (
        int(mp4_path.stat().st_size),
        int(webp_path.stat().st_size),
        int(poster_path.stat().st_size),
    )


def process_one_chamber(
    exp: str,
    roi: str,
    has_all_visits: bool,
    man_ch_rows: list[dict[str, str]],
    df_s5_ch: pd.DataFrame,
    df_qc_ch: pd.DataFrame,
    df_s4_ch: pd.DataFrame,
    df_s4_c_ch: pd.DataFrame,
    df_s5_sw_ch: pd.DataFrame,
    sw_hints_ch: dict[tuple[int, int], dict[str, int]],
    use_raw_masks: bool = False,
    override_burst_visit: int | None = None,
) -> list[dict[str, Any]]:
    """Build both `(a) visits` and `(b) burst` clips for a single `(exp, roi)` chamber."""
    t0_ch = time.perf_counter()
    exp_short = exp.split("_")[0]

    # Sort manifest rows
    ok_rows = [
        r
        for r in man_ch_rows
        if r.get("ok") == "True" and Path(r["path"]).is_file()
    ]
    ok_rows.sort(key=lambda r: (int(r["visit_index"]), int(r["sweep_index"])))

    # Select 1 sweep per visit for clip (a): sweep 1 if ok, else first ok sweep of that visit
    by_visit: dict[int, list[dict[str, str]]] = {}
    for r in ok_rows:
        by_visit.setdefault(int(r["visit_index"]), []).append(r)

    visit_rows: list[dict[str, str]] = []
    for v in sorted(by_visit.keys()):
        v_list = by_visit[v]
        sw1 = [r for r in v_list if int(r["sweep_index"]) == 1]
        visit_rows.append(sw1[0] if sw1 else v_list[0])

    # Choose visit for within-visit sweep burst clip (b)
    burst_visit = choose_burst_visit(
        exp=exp,
        roi=roi,
        man_ch_rows=ok_rows,
        df_qc_ch=df_qc_ch,
        df_s5_ch=df_s5_ch,
        override_visit=override_burst_visit,
    )
    burst_rows = by_visit.get(burst_visit, [visit_rows[0]])

    # Collect unique stems needed across both clips
    needed_rows_by_stem: dict[str, dict[str, str]] = {}
    for r in visit_rows + burst_rows:
        needed_rows_by_stem[str(r["stem"]).zfill(5)] = r

    # Step 1: Load 3D masks and extract tracked T cell + tracked target + other cells for all needed stems,
    # and determine the union 3D bounding box of tracked cells over the movie + 5 um margin.
    full_masks_by_stem: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    zs, ys, xs = [], [], []
    shape_zyx = (190, 256, 256)

    for stem, r in needed_rows_by_stem.items():
        v = int(r["visit_index"])
        s = int(r["sweep_index"])
        mask_path = MASKS_DIR / exp / roi / f"{stem}_predictions.h5"
        with h5py.File(mask_path, "r") as hf:
            I = hf["predictions"][:]
        shape_zyx = I.shape

        # Lookup tracking hints (for sweep 6 fallback to sweep 5 or visit 1 hint)
        hint = (
            sw_hints_ch.get((v, s))
            or sw_hints_ch.get((v, 5))
            or sw_hints_ch.get((v, 1))
            or {"tracked_tcell_id": 1, "n_tcell_instances": 1, "n_target_instances": 1}
        )
        t_full, b_full, other_full = extract_tracked_and_other_masks(
            I,
            tracked_id=int(hint["tracked_tcell_id"]),
            n_t_hint=int(hint["n_tcell_instances"]),
            n_b_hint=int(hint["n_target_instances"]),
        )
        full_masks_by_stem[stem] = (I, t_full, b_full, other_full)

        u_mask = (I > 0) if use_raw_masks else (t_full | b_full)
        if np.any(u_mask):
            zz, yy, xx = np.where(u_mask)
            zs.extend([int(zz.min()), int(zz.max())])
            ys.extend([int(yy.min()), int(yy.max())])
            xs.extend([int(xx.min()), int(xx.max())])

    Z_dim, Y_dim, X_dim = shape_zyx
    if zs and ys and xs:
        z0 = max(0, min(zs) - MARGIN_VOX)
        z1 = min(Z_dim, max(zs) + 1 + MARGIN_VOX)
        y0 = max(0, min(ys) - MARGIN_VOX)
        y1 = min(Y_dim, max(ys) + 1 + MARGIN_VOX)
        x0 = max(0, min(xs) - MARGIN_VOX)
        x1 = min(X_dim, max(xs) + 1 + MARGIN_VOX)
    else:
        z0, z1, y0, y1, x0, x1 = 0, Z_dim, 0, Y_dim, 0, X_dim

    # Ensure even crop spans
    if (z1 - z0) % 2 == 1:
        if z1 < Z_dim:
            z1 += 1
        elif z0 > 0:
            z0 -= 1
    if (y1 - y0) % 2 == 1:
        if y1 < Y_dim:
            y1 += 1
        elif y0 > 0:
            y0 -= 1
    if (x1 - x0) % 2 == 1:
        if x1 < X_dim:
            x1 += 1
        elif x0 > 0:
            x0 -= 1

    crop_box = (z0, z1, y0, y1, x0, x1)

    # Crop masks and free full-volume masks
    masks_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for stem, (I, t_full, b_full, other_full) in full_masks_by_stem.items():
        masks_cache[stem] = (
            I[z0:z1, y0:y1, x0:x1],
            t_full[z0:z1, y0:y1, x0:x1],
            b_full[z0:z1, y0:y1, x0:x1],
            other_full[z0:z1, y0:y1, x0:x1],
        )
    del full_masks_by_stem

    # Step 2: Load cropped raw volumes and apply light 3D Gaussian denoise (sigma = 0.7 vox) on display channels
    raw_cache: dict[str, np.ndarray] = {}
    for stem, r in needed_rows_by_stem.items():
        with h5py.File(r["path"], "r") as hf_r:
            raw_c = hf_r["raw"][:, z0:z1, y0:y1, x0:x1].astype(np.float32)
        # Light 3D gaussian denoise (sigma = 0.7 vox) on display channels
        raw_c[1] = ndi.gaussian_filter(raw_c[1], DENOISE_SIGMA_VOX)
        raw_c[2] = ndi.gaussian_filter(raw_c[2], DENOISE_SIGMA_VOX)
        raw_cache[stem] = raw_c

    # Fixed per-channel display range across the chamber's movie frames (exclude sweep 6 tubulin from ch2 hi)
    ref_stems = [str(r["stem"]).zfill(5) for r in visit_rows]
    lo = [
        0.0,
        float(np.percentile(np.concatenate([raw_cache[s][1].max(0).ravel() for s in ref_stems]), 1.0)),
        float(np.percentile(np.concatenate([raw_cache[s][2].max(0).ravel() for s in ref_stems]), 1.0)),
    ]
    hi = [
        1.0,
        float(np.percentile(np.concatenate([raw_cache[s][1].max(0).ravel() for s in ref_stems]), 99.8)),
        float(np.percentile(np.concatenate([raw_cache[s][2].max(0).ravel() for s in ref_stems]), 99.8)),
    ]

    # Step 3: Build per-visit and per-sweep metric dictionaries for the strip chart
    n_visits_total = max(int(r["visit_index"]) for r in man_ch_rows)
    visit_metrics: dict[int, dict[str, Any]] = {v: {} for v in range(1, n_visits_total + 1)}

    for _, r in df_s5_ch.iterrows():
        v = int(r["visit"])
        visit_metrics.setdefault(v, {}).update(
            {
                "in_contact_visit": bool(r.get("in_contact_visit", False)),
                "cup_depth_hull_p95_um": r.get("cup_depth_hull_p95_um"),
                "contact_area_um2": r.get("contact_area_um2"),
                "target_volume_um3": r.get("target_volume_um3"),
                "qc_label": "contact" if bool(r.get("in_contact_visit", False)) else "no contact",
                "is_clean_qc": False,
            }
        )

    for _, r in df_qc_ch.iterrows():
        v = int(r["visit"])
        frag = bool(r.get("flag_fragmented", False))
        edge = bool(r.get("flag_edge", False))
        v_ok = pd.notna(r.get("target_volume_um3")) and (200.0 <= float(r["target_volume_um3"]) <= 1500.0)
        r_ok = pd.notna(r.get("target_vol_ratio_to_own_median")) and (
            0.7 <= float(r["target_vol_ratio_to_own_median"]) <= 1.4
        )
        is_clean = (not frag) and (not edge) and v_ok and r_ok
        flags = []
        if frag:
            flags.append("frag")
        if edge:
            flags.append("edge")
        if not v_ok:
            flags.append("vol")
        if not r_ok:
            flags.append("jump")
        qc_lbl = "clean" if is_clean else f"flagged ({','.join(flags)})"
        visit_metrics.setdefault(v, {}).update(
            {
                "cup_depth_hull_p95_um": r.get("cup_depth_hull_p95_um"),
                "contact_area_um2": r.get("contact_area_um2"),
                "target_volume_um3": r.get("target_volume_um3"),
                "is_clean_qc": is_clean,
                "qc_label": qc_lbl,
            }
        )

    for _, r in df_s4_ch.iterrows():
        v = int(r["visit"])
        visit_metrics.setdefault(v, {}).update(
            {
                "dead_visit": bool(r.get("dead_visit", False)),
                "artifact_405_band": bool(r.get("artifact_405_band", False)),
            }
        )

    sweep_metrics_map: dict[tuple[int, int], dict[str, Any]] = {}
    for _, r in df_s5_sw_ch.iterrows():
        sweep_metrics_map[(int(r["visit"]), int(r["sweep"]))] = {
            "cup_depth_hull_p95_um": r.get("cup_depth_hull_p95_um"),
            "contact_area_um2": r.get("contact_area_um2"),
            "target_volume_um3": r.get("target_volume_um3"),
        }

    condition_str = "CAR-T"
    fate_str = "Survived"
    death_visit_sustained: int | None = None
    if len(df_s4_c_ch) > 0:
        rc = df_s4_c_ch.iloc[0]
        condition_str = str(rc.get("condition", "CAR-T"))
        ever_d = bool(rc.get("ever_dead_200m", rc.get("ever_dead", False)))
        dv = rc.get("death_visit_200m", rc.get("death_visit"))
        if ever_d and pd.notna(dv):
            death_visit_sustained = int(float(dv))
            fate_str = f"Killed (visit {death_visit_sustained}, {death_visit_sustained*10} min)"
        elif bool(rc.get("ever_dead_any_visit_200m", False)):
            fadv = rc.get("first_any_dead_visit_200m")
            fate_str = f"Single-visit 405+ (v{int(float(fadv))})" if pd.notna(fadv) else "Single-visit 405+"
        else:
            fate_str = "Survived (<=200 min)"

    results: list[dict[str, Any]] = []
    sfx_raw = "_rawmasks" if use_raw_masks else ""

    # Clip (a): visits
    clip_a_frames = render_clip_frames(
        exp=exp,
        roi=roi,
        clip_name="visits",
        frame_rows=visit_rows,
        crop_box=crop_box,
        lo=lo,
        hi=hi,
        masks_cache=masks_cache,
        raw_cache=raw_cache,
        visit_metrics=visit_metrics,
        sweep_metrics_map=sweep_metrics_map,
        n_visits_total=n_visits_total,
        condition_str=condition_str,
        fate_str=fate_str,
        death_visit_sustained=death_visit_sustained,
        use_raw_masks=use_raw_masks,
    )
    stem_a = f"{exp_short}_{roi}_visits{sfx_raw}"
    mp4_a = MOVIES_DIR / f"{stem_a}.mp4"
    webp_a = MOVIES_DIR / f"{stem_a}.webp"
    poster_a = MOVIES_DIR / f"{stem_a}_poster.jpg"
    sz_mp4_a, sz_webp_a, sz_jpg_a = encode_mp4_webp_poster(clip_a_frames, mp4_a, webp_a, poster_a, fps=FPS)
    results.append(
        {
            "exp": exp,
            "roi": roi,
            "clip": "visits",
            "path": str(mp4_a.relative_to(REPO_ROOT)),
            "size": sz_mp4_a,
            "n_frames": len(clip_a_frames),
            "has_all_visits": has_all_visits,
            "_webp_path": str(webp_a.relative_to(REPO_ROOT)),
            "_webp_size": sz_webp_a,
            "_poster_path": str(poster_a.relative_to(REPO_ROOT)),
            "_poster_size": sz_jpg_a,
            "_burst_visit": burst_visit,
            "_elapsed_s": round(time.perf_counter() - t0_ch, 2),
        }
    )

    # Clip (b): within-visit sweep burst
    clip_b_name = f"burst_v{burst_visit:02d}"
    clip_b_frames = render_clip_frames(
        exp=exp,
        roi=roi,
        clip_name=clip_b_name,
        frame_rows=burst_rows,
        crop_box=crop_box,
        lo=lo,
        hi=hi,
        masks_cache=masks_cache,
        raw_cache=raw_cache,
        visit_metrics=visit_metrics,
        sweep_metrics_map=sweep_metrics_map,
        n_visits_total=n_visits_total,
        condition_str=condition_str,
        fate_str=fate_str,
        death_visit_sustained=death_visit_sustained,
        use_raw_masks=use_raw_masks,
    )
    stem_b = f"{exp_short}_{roi}_{clip_b_name}{sfx_raw}"
    mp4_b = MOVIES_DIR / f"{stem_b}.mp4"
    webp_b = MOVIES_DIR / f"{stem_b}.webp"
    poster_b = MOVIES_DIR / f"{stem_b}_poster.jpg"
    sz_mp4_b, sz_webp_b, sz_jpg_b = encode_mp4_webp_poster(clip_b_frames, mp4_b, webp_b, poster_b, fps=FPS)
    results.append(
        {
            "exp": exp,
            "roi": roi,
            "clip": clip_b_name,
            "path": str(mp4_b.relative_to(REPO_ROOT)),
            "size": sz_mp4_b,
            "n_frames": len(clip_b_frames),
            "has_all_visits": has_all_visits,
            "_webp_path": str(webp_b.relative_to(REPO_ROOT)),
            "_webp_size": sz_webp_b,
            "_poster_path": str(poster_b.relative_to(REPO_ROOT)),
            "_poster_size": sz_jpg_b,
            "_burst_visit": burst_visit,
            "_elapsed_s": round(time.perf_counter() - t0_ch, 2),
        }
    )

    return results


def _worker_star(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return process_one_chamber(*args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="movies.py",
        description="Build chamber review movies (visits and within-visit sweep burst) with strip chart.",
    )
    parser.add_argument("positional", nargs="*", help="Optional <experiment> <roi> positional filter.")
    parser.add_argument("--experiment", type=str, default=None, help="Filter to a single experiment.")
    parser.add_argument("--roi", type=str, default=None, help="Filter to a single ROI.")
    parser.add_argument(
        "--raw-masks",
        action="store_true",
        help="Outline raw U-Net classes (2|4 target, 1|3 T cell) instead of extracted tracked instances.",
    )
    parser.add_argument(
        "--burst-visit",
        type=int,
        default=None,
        help="Override chosen visit index (1-based) for the within-visit sweep burst clip.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel chamber workers.")
    args = parser.parse_args(argv)

    exp_filter = args.experiment
    roi_filter = args.roi
    if len(args.positional) >= 2:
        exp_filter = args.positional[0]
        roi_filter = args.positional[1]
    elif len(args.positional) == 1:
        exp_filter = args.positional[0]

    t0 = time.perf_counter()
    meta = load_metadata_tables()
    man_rows = meta["man_rows"]
    eligible = find_eligible_chambers(man_rows)

    if exp_filter is not None:
        eligible = [c for c in eligible if c[0] == exp_filter or c[0].startswith(exp_filter)]
    if roi_filter is not None:
        eligible = [c for c in eligible if c[1] == roi_filter]

    if not eligible:
        print("No eligible chambers found matching filter.", file=sys.stderr)
        return 1

    MOVIES_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Any, ...]] = []
    for exp, roi, has_all in eligible:
        man_ch = [r for r in man_rows if r["experiment"] == exp and r["roi"] == roi]
        s5_ch = meta["df_s5_v"][(meta["df_s5_v"]["experiment"] == exp) & (meta["df_s5_v"]["roi"] == roi)].copy()
        qc_ch = meta["df_qc_v"][(meta["df_qc_v"]["experiment"] == exp) & (meta["df_qc_v"]["roi"] == roi)].copy()
        s4_ch = meta["df_s4_v"][(meta["df_s4_v"]["experiment"] == exp) & (meta["df_s4_v"]["roi"] == roi)].copy()
        s4_c_ch = meta["df_s4_c"][(meta["df_s4_c"]["experiment"] == exp) & (meta["df_s4_c"]["roi"] == roi)].copy()
        s5_sw_ch = meta["df_s5_sw"][(meta["df_s5_sw"]["experiment"] == exp) & (meta["df_s5_sw"]["roi"] == roi)].copy()
        sw_h_ch = {
            (v, s): h
            for (e_k, r_k, v, s), h in meta["sw_hints"].items()
            if e_k == exp and r_k == roi
        }
        tasks.append(
            (
                exp,
                roi,
                has_all,
                man_ch,
                s5_ch,
                qc_ch,
                s4_ch,
                s4_c_ch,
                s5_sw_ch,
                sw_h_ch,
                args.raw_masks,
                args.burst_visit,
            )
        )

    all_rows: list[dict[str, Any]] = []
    n_workers = min(max(1, args.workers), len(tasks))
    if n_workers == 1:
        for t in tasks:
            res = _worker_star(t)
            all_rows.extend(res)
            for r in res:
                print(
                    f"[DONE] {r['exp']} {r['roi']} ({r['clip']}): "
                    f"MP4={r['size']/1024:.1f} KB, WebP={r['_webp_size']/1024:.1f} KB, "
                    f"frames={r['n_frames']}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fut_map = {pool.submit(_worker_star, t): (t[0], t[1]) for t in tasks}
            for fut in as_completed(fut_map):
                res = fut.result()
                all_rows.extend(res)
                for r in res:
                    print(
                        f"[DONE] {r['exp']} {r['roi']} ({r['clip']}): "
                        f"MP4={r['size']/1024:.1f} KB, WebP={r['_webp_size']/1024:.1f} KB, "
                        f"frames={r['n_frames']}",
                        flush=True,
                    )

    # Sort output rows deterministically by chamber and clip order (visits first, then burst)
    all_rows.sort(
        key=lambda r: (
            r["exp"],
            int(r["roi"].split("_")[-1]) if r["roi"].split("_")[-1].isdigit() else 0,
            0 if r["clip"] == "visits" else 1,
        )
    )

    # Write figures/movies/index.csv with exact required columns when running default batch
    if not args.raw_masks and exp_filter is None and roi_filter is None:
        fieldnames = ["exp", "roi", "clip", "path", "size", "n_frames", "has_all_visits"]
        with open(INDEX_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"Wrote {INDEX_CSV.relative_to(REPO_ROOT)} ({len(all_rows)} clips).", flush=True)

    elapsed = time.perf_counter() - t0
    total_mp4 = sum(int(r["size"]) for r in all_rows)
    total_webp = sum(int(r["_webp_size"]) for r in all_rows)
    print(
        f"Completed {len(all_rows)} clips across {len(eligible)} chambers in {elapsed:.1f}s | "
        f"Total MP4={total_mp4/1e6:.2f} MB, Total WebP={total_webp/1e6:.2f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
