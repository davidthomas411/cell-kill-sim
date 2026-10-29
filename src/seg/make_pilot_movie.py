"""Build 4-column candidate comparison movies and per-visit JPEG frames for all 8 Pilot P2 chambers.

Conventions (same layout as P1, updated for P2):
- 4 columns: C0 (Lab deposited), C1 (Lab repaired), C1s (C1 + signal check), C3 (PlantSeg 3D fixed).
- Each column shows XY max projection (top row) and XZ max projection (bottom row).
- Raw composite: green ch1 (T-cell Lifeact), magenta ch2 (target mApple-Mem),
  yellow ch0 (SYTOX death dye above 404 ADU after sigma=3 Gaussian), fixed display range per chamber.
- Cropped to chamber union of cells + 5 um (26 voxels at 0.1962 um/voxel).
- Outlines of projected 3D masks: tracked target in blue (90, 160, 255),
  tracked T cell in orange (255, 150, 40), other instances in thin grey (155, 155, 155).
- Prints precision/recall (P/R) of the tracked target and T cell under each panel.
- Canvas width: 1200 px (even height).
- Saves zero-padded frames ONLY: figures/pilot/frames/<exp>_<roi>/<visit:02d>.jpg
- Encodes H.264 MP4 (< 1.5 MB, libx264, crf 28-30, 2 fps): figures/pilot/<exp>_<roi>_candidates.mp4
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reproduce_contact_area import STRUCT26
from src.seg.candidates import PILOT4_CHAMBERS, PILOT_CHAMBERS, VOXEL_UM

CANDIDATES = [
    ("C0", "C0: Lab deposited"),
    ("C1", "C1: Lab repaired"),
    ("C1s", "C1s: C1 + signal check"),
    ("C3", "C3: PlantSeg 3D"),
]


def outline_2d(mask_2d: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Inner boundary outline of a 2D boolean mask with given pixel thickness."""
    if not np.any(mask_2d):
        return np.zeros_like(mask_2d, dtype=bool)
    return mask_2d & (~ndi.binary_erosion(mask_2d, iterations=thickness))


def resize_bool_nearest(mask_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize of a 2D boolean array to (height, width)."""
    im = Image.fromarray(mask_2d.astype(np.uint8) * 255)
    im_up = im.resize((width, height), Image.NEAREST)
    return np.asarray(im_up) > 127


def load_font(size: int = 13) -> ImageFont.ImageFont:
    for p in [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_chamber_movie(
    short_exp: str = "v01",
    exp: str = "v01_6_40_120_0-6_hd1",
    roi: str = "MIP_ROI_32",
) -> Path:
    manifest_path = REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv"
    with open(manifest_path, "r") as f:
        all_rows = list(csv.DictReader(f))

    by_visit: Dict[int, List[dict]] = {}
    for r in all_rows:
        if r["experiment"] == exp and r["roi"] == roi and r["ok"] == "True" and r["sweep_index"] != "6":
            by_visit.setdefault(int(r["visit_index"]), []).append(r)

    rows = []
    for v in sorted(by_visit.keys()):
        vlist = by_visit[v]
        s1 = [r for r in vlist if r["sweep_index"] == "1"]
        if s1:
            rows.append(s1[0])
        else:
            s2 = [r for r in vlist if r["sweep_index"] == "2"]
            rows.append(s2[0] if s2 else sorted(vlist, key=lambda x: int(x["sweep_index"]))[0])

    # Load per-chamber metrics for precision/recall annotations
    metrics_csv = REPO_ROOT / "data" / "pilot_seg" / f"{short_exp}_{roi}_p2_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = REPO_ROOT / "data" / "pilot_seg" / "pilot2_metrics.csv"
    pr_lookup: Dict[Tuple[int, str], Tuple[float, float, float, float]] = {}
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        mdf_sub = mdf[(mdf["experiment"] == exp) & (mdf["roi"] == roi)]
        for _, mr in mdf_sub.iterrows():
            pr_lookup[(int(mr["visit"]), str(mr["candidate"]))] = (
                float(mr["target_precision"]),
                float(mr["target_recall"]),
                float(mr["tcell_precision"]),
                float(mr["tcell_recall"]),
            )

    # Load raw volumes, predictions, and compute chamber-wide union cell crop (+ 5 um = 26 px)
    frames_raw: List[np.ndarray] = []
    stems: List[str] = []
    z_min, y_min, x_min = 10**6, 10**6, 10**6
    z_max, y_max, x_max = 0, 0, 0

    for r in rows:
        stem = f"{int(r['stem']):05d}"
        stems.append(stem)
        with h5py.File(REPO_ROOT / r["path"], "r") as f_raw:
            raw = f_raw["raw"][:].astype(np.float32)
        with h5py.File(
            REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5",
            "r",
        ) as f_pred:
            pred = f_pred["predictions"][:]
        frames_raw.append(raw)

        lab, _ = ndi.label(pred > 0, structure=STRUCT26)
        cnt = np.bincount(lab.ravel())
        keep = np.where(cnt >= 2000)[0]
        keep = keep[keep > 0]
        clean = np.isin(lab, keep) if keep.size > 0 else (pred > 0)
        nz = np.argwhere(clean)
        if nz.size > 0:
            lo_zyx = nz.min(axis=0)
            hi_zyx = nz.max(axis=0)
            z_min = min(z_min, int(lo_zyx[0]))
            y_min = min(y_min, int(lo_zyx[1]))
            x_min = min(x_min, int(lo_zyx[2]))
            z_max = max(z_max, int(hi_zyx[0]))
            y_max = max(y_max, int(hi_zyx[1]))
            x_max = max(x_max, int(hi_zyx[2]))

    _, Z, Y, X = frames_raw[0].shape
    pad = 26  # 26 * 0.1962 = 5.10 um
    z0, z1 = max(0, z_min - pad), min(Z, z_max + pad + 1)
    y0, y1 = max(0, y_min - pad), min(Y, y_max + pad + 1)
    x0, x1 = max(0, x_min - pad), min(X, x_max + pad + 1)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    print(f"[{short_exp} {roi}] Chamber crop (cells + 5 um): Z=[{z0}:{z1}], Y=[{y0}:{y1}], X=[{x0}:{x1}]")

    # Fixed display range across the whole chamber (clipping camera read-noise floor at p65)
    lo = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 65.0))
        for c in range(3)
    ]
    hi = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 99.8))
        for c in range(3)
    ]

    def norm(img: np.ndarray, c: int) -> np.ndarray:
        return np.clip((img - lo[c]) / max(1.0, hi[c] - lo[c]), 0.0, 1.0)

    def composite(raw_crop: np.ndarray, axis: int) -> np.ndarray:
        t = norm(raw_crop[1].max(axis), 1)  # ch1: T cell -> green
        g = norm(raw_crop[2].max(axis), 2)  # ch2: target -> magenta
        # ch0: SYTOX death dye above 404 ADU after sigma=3 Gaussian
        d = np.clip((ndi.gaussian_filter(raw_crop[0].max(axis), 3.0) - 404.0) / 20.0, 0.0, 1.0)
        rgb = np.stack([g + d, t + 0.8 * d, g], axis=-1)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Fixed 1200 px canvas width geometry
    total_w = 1200
    left_label_w = 28
    right_pad = 12
    col_gap = 8
    row_gap = 8
    top_header_h = 46
    pr_panel_h = 38
    bottom_footer_h = 26

    panel_w = (total_w - left_label_w - right_pad - 3 * col_gap) // 4  # 284 px
    x_span = max(1, x1 - x0)
    y_span = max(1, y1 - y0)
    z_span = max(1, z1 - z0)
    scale_factor = float(panel_w) / float(x_span)
    top_h = max(32, int(round(y_span * scale_factor)))
    side_h = max(32, int(round(z_span * scale_factor)))

    total_h = top_header_h + top_h + row_gap + side_h + pr_panel_h + bottom_footer_h
    if total_h % 2 != 0:
        total_h += 1

    frames_dir = REPO_ROOT / "figures" / "pilot" / "frames" / f"{short_exp}_{roi}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Remove any accidental unpadded files if present in subfolder
    for unpad in range(1, 10):
        p_unpad = frames_dir / f"{unpad}.jpg"
        if p_unpad.exists():
            p_unpad.unlink()

    movie_path = REPO_ROOT / "figures" / "pilot" / f"{short_exp}_{roi}_candidates.mp4"

    font_title = load_font(14)
    font_col = load_font(13)
    font_small = load_font(12)

    for r, raw_full, stem in zip(rows, frames_raw, stems):
        visit = int(r["visit_index"])
        sweep = int(r["sweep_index"])
        raw_crop = raw_full[:, z0:z1, y0:y1, x0:x1]

        base_top = composite(raw_crop, axis=0)   # (Y_crop, X_crop, 3)
        base_side = composite(raw_crop, axis=1)  # (Z_crop, X_crop, 3)

        col_panels: List[Tuple[Image.Image, Image.Image]] = []
        for cand_code, _ in CANDIDATES:
            if cand_code == "C3":
                npz_path = REPO_ROOT / "data" / "pilot_seg" / "C3_fixed" / exp / roi / f"{stem}.npz"
                if not npz_path.exists():
                    npz_path = REPO_ROOT / "data" / "pilot_seg" / "C3" / exp / roi / f"{stem}.npz"
            else:
                npz_path = REPO_ROOT / "data" / "pilot_seg" / cand_code / exp / roi / f"{stem}.npz"

            data = np.load(npz_path)
            t_inst = data["t_inst"][crop_sl]
            b_inst = data["b_inst"][crop_sl]
            t_mask = data["t_mask"][crop_sl] > 0
            b_mask = data["b_mask"][crop_sl] > 0
            tid = int(data["tracked_t_id"])
            bid = int(data["tracked_b_id"])

            rendered_views: List[Image.Image] = []
            for axis, base_rgb, target_h in [(0, base_top, top_h), (1, base_side, side_h)]:
                rgb_up = np.array(
                    Image.fromarray(base_rgb).resize((panel_w, target_h), Image.NEAREST)
                )

                # Other instances: thin 1px grey outline on panel
                for inst_arr, tr_id in [(b_inst, bid), (t_inst, tid)]:
                    for uid in np.unique(inst_arr):
                        if uid == 0 or int(uid) == tr_id:
                            continue
                        proj_o = (inst_arr == uid).max(axis=axis)
                        if np.any(proj_o):
                            proj_o_up = resize_bool_nearest(proj_o, panel_w, target_h)
                            rgb_up[outline_2d(proj_o_up, thickness=1)] = (155, 155, 155)

                # Tracked target: 2px blue outline of projected 3D mask
                proj_b = b_mask.max(axis=axis)
                if np.any(proj_b):
                    proj_b_up = resize_bool_nearest(proj_b, panel_w, target_h)
                    rgb_up[outline_2d(proj_b_up, thickness=2)] = (90, 160, 255)

                # Tracked T cell: 2px orange outline of projected 3D mask
                proj_t = t_mask.max(axis=axis)
                if np.any(proj_t):
                    proj_t_up = resize_bool_nearest(proj_t, panel_w, target_h)
                    rgb_up[outline_2d(proj_t_up, thickness=2)] = (255, 150, 40)

                rendered_views.append(Image.fromarray(rgb_up))

            col_panels.append((rendered_views[0], rendered_views[1]))

        canvas = Image.new("RGB", (total_w, total_h), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)

        # Top header: chamber + visit + time
        dr.text(
            (left_label_w, 6),
            f"{short_exp} {roi}  |  Visit {visit:02d}/20  (t = {(visit - 1) * 10} min, sweep {sweep})",
            fill=(242, 242, 242),
            font=font_title,
        )

        # Row labels on left
        dr.text((4, top_header_h + 6), "XY", fill=(200, 200, 200), font=font_small)
        dr.text((4, top_header_h + top_h + row_gap + 6), "XZ", fill=(200, 200, 200), font=font_small)

        pr_y0 = top_header_h + top_h + row_gap + side_h + 4
        for c_idx, (cand_code, col_label) in enumerate(CANDIDATES):
            x_off = left_label_w + c_idx * (panel_w + col_gap)
            dr.text((x_off, 26), col_label, fill=(242, 242, 242), font=font_col)
            top_im, side_im = col_panels[c_idx]
            canvas.paste(top_im, (x_off, top_header_h))
            canvas.paste(side_im, (x_off, top_header_h + top_h + row_gap))

            b_p, b_r, t_p, t_r = pr_lookup.get((visit, cand_code), (0.0, 0.0, 0.0, 0.0))
            dr.text(
                (x_off + 2, pr_y0),
                f"Target: P={b_p:.2f}  R={b_r:.2f}",
                fill=(130, 190, 255),
                font=font_small,
            )
            dr.text(
                (x_off + 2, pr_y0 + 16),
                f"T cell: P={t_p:.2f}  R={t_r:.2f}",
                fill=(255, 175, 85),
                font=font_small,
            )

        # 5 um scale bar on right-most XY panel
        px_5um = max(10, int(round((5.0 / VOXEL_UM) * scale_factor)))
        sb_x1 = left_label_w + 4 * panel_w + 3 * col_gap - 10
        sb_x0 = sb_x1 - px_5um
        sb_y = top_header_h + top_h - 10
        dr.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=(255, 255, 255), width=3)
        dr.text((sb_x0, sb_y - 15), "5 um", fill=(255, 255, 255), font=font_small)

        # Footer legend
        footer_y = pr_y0 + pr_panel_h
        dr.text(
            (left_label_w, footer_y),
            "Raw: green ch1 (T cell) | magenta ch2 (target) | yellow ch0 (SYTOX >404 ADU)    "
            "Outlines: blue tracked target | orange tracked T cell | grey other instances",
            fill=(205, 205, 205),
            font=font_small,
        )

        # Save zero-padded <visit:02d>.jpg ONLY (no duplicate unpadded files)
        jpg_path_pad = frames_dir / f"{visit:02d}.jpg"
        canvas.save(jpg_path_pad, format="JPEG", quality=82)

    # Encode H.264 MP4 (1200 px wide, crf 28, 2 fps, <= 1.5 MB)
    for crf_val in ("28", "30", "32"):
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(frames_dir / "%02d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            crf_val,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size_mb = movie_path.stat().st_size / (1024 * 1024)
        if size_mb <= 1.48:
            break
    print(f"Saved {movie_path} ({size_mb:.3f} MB, crf={crf_val}, {total_w}x{total_h})")
    return movie_path


def main() -> None:
    for short_exp, exp, roi in PILOT_CHAMBERS:
        render_chamber_movie(short_exp=short_exp, exp=exp, roi=roi)


CANDIDATES_P3 = [
    ("C0", "C0: Lab deposited"),
    ("C3", "C3: PlantSeg 3D (P2)"),
    ("C4", "C4: PlantSeg + channel ID (P3)"),
]


def render_chamber_movie_p3(
    short_exp: str = "v01",
    exp: str = "v01_6_40_120_0-6_hd1",
    roi: str = "MIP_ROI_32",
) -> Path:
    """Render 3-column (C0, C3, C4) comparison movie and frames for Pilot P3."""
    manifest_path = REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv"
    with open(manifest_path, "r") as f:
        all_rows = list(csv.DictReader(f))

    by_visit: Dict[int, List[dict]] = {}
    for r in all_rows:
        if r["experiment"] == exp and r["roi"] == roi and r["ok"] == "True" and r["sweep_index"] != "6":
            by_visit.setdefault(int(r["visit_index"]), []).append(r)

    rows = []
    for v in sorted(by_visit.keys()):
        vlist = by_visit[v]
        s1 = [r for r in vlist if r["sweep_index"] == "1"]
        if s1:
            rows.append(s1[0])
        else:
            s2 = [r for r in vlist if r["sweep_index"] == "2"]
            rows.append(s2[0] if s2 else sorted(vlist, key=lambda x: int(x["sweep_index"]))[0])

    metrics_csv = REPO_ROOT / "data" / "pilot_seg" / f"{short_exp}_{roi}_p3_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = REPO_ROOT / "data" / "pilot_seg" / "pilot3_metrics.csv"

    score_lookup: Dict[Tuple[int, str], Dict[str, object]] = {}
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        mdf_sub = mdf[(mdf["experiment"] == exp) & (mdf["roi"] == roi)]
        for _, mr in mdf_sub.iterrows():
            score_lookup[(int(mr["visit"]), str(mr["candidate"]))] = {
                "tracked_t_id": int(mr["tracked_t_id"]),
                "tracked_b_id": int(mr["tracked_b_id"]),
                "tcell_track_id": int(mr["tcell_track_id"]),
                "target_track_id": int(mr["target_track_id"]),
                "b_surf": float(mr["target_surf_score"]),
                "b_miss": float(mr["target_missed_2vox"]),
                "b_phan": float(mr["target_phantom_frac"]),
                "t_surf": float(mr["tcell_surf_score"]),
                "t_miss": float(mr["tcell_missed_2vox"]),
                "t_phan": float(mr["tcell_phantom_frac"]),
            }

    frames_raw: List[np.ndarray] = []
    stems: List[str] = []
    z_min, y_min, x_min = 10**6, 10**6, 10**6
    z_max, y_max, x_max = 0, 0, 0

    for r in rows:
        stem = f"{int(r['stem']):05d}"
        stems.append(stem)
        with h5py.File(REPO_ROOT / r["path"], "r") as f_raw:
            raw = f_raw["raw"][:].astype(np.float32)
        with h5py.File(
            REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5",
            "r",
        ) as f_pred:
            pred = f_pred["predictions"][:]
        frames_raw.append(raw)

        lab, _ = ndi.label(pred > 0, structure=STRUCT26)
        cnt = np.bincount(lab.ravel())
        keep = np.where(cnt >= 2000)[0]
        keep = keep[keep > 0]
        clean = np.isin(lab, keep) if keep.size > 0 else (pred > 0)
        npz_c4 = REPO_ROOT / "data" / "pilot_seg" / "C4" / exp / roi / f"{stem}.npz"
        if npz_c4.exists():
            with np.load(npz_c4) as z4:
                clean = clean | (z4["t_inst"] > 0) | (z4["b_inst"] > 0)
        nz = np.argwhere(clean)
        if nz.size > 0:
            lo_zyx = nz.min(axis=0)
            hi_zyx = nz.max(axis=0)
            z_min = min(z_min, int(lo_zyx[0]))
            y_min = min(y_min, int(lo_zyx[1]))
            x_min = min(x_min, int(lo_zyx[2]))
            z_max = max(z_max, int(hi_zyx[0]))
            y_max = max(y_max, int(hi_zyx[1]))
            x_max = max(x_max, int(hi_zyx[2]))

    _, Z, Y, X = frames_raw[0].shape
    pad = 26
    z0, z1 = max(0, z_min - pad), min(Z, z_max + pad + 1)
    y0, y1 = max(0, y_min - pad), min(Y, y_max + pad + 1)
    x0, x1 = max(0, x_min - pad), min(X, x_max + pad + 1)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    print(f"[{short_exp} {roi}] P3 crop (cells + 5 um): Z=[{z0}:{z1}], Y=[{y0}:{y1}], X=[{x0}:{x1}]")

    lo = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 65.0))
        for c in range(3)
    ]
    hi = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 99.8))
        for c in range(3)
    ]

    def norm(img: np.ndarray, c: int) -> np.ndarray:
        return np.clip((img - lo[c]) / max(1.0, hi[c] - lo[c]), 0.0, 1.0)

    def composite(raw_crop: np.ndarray, axis: int) -> np.ndarray:
        t = norm(raw_crop[1].max(axis), 1)
        g = norm(raw_crop[2].max(axis), 2)
        d = np.clip((ndi.gaussian_filter(raw_crop[0].max(axis), 3.0) - 404.0) / 20.0, 0.0, 1.0)
        rgb = np.stack([g + d, t + 0.8 * d, g], axis=-1)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    total_w = 1200
    left_label_w = 28
    right_pad = 12
    col_gap = 10
    row_gap = 8
    top_header_h = 46
    pr_panel_h = 40
    bottom_footer_h = 26

    panel_w = (total_w - left_label_w - right_pad - 2 * col_gap) // 3  # 380 px
    x_span = max(1, x1 - x0)
    y_span = max(1, y1 - y0)
    z_span = max(1, z1 - z0)
    scale_factor = float(panel_w) / float(x_span)
    top_h = max(32, int(round(y_span * scale_factor)))
    side_h = max(32, int(round(z_span * scale_factor)))

    total_h = top_header_h + top_h + row_gap + side_h + pr_panel_h + bottom_footer_h
    if total_h % 2 != 0:
        total_h += 1

    p3_dir = REPO_ROOT / "figures" / "pilot" / "p3"
    frames_dir = p3_dir / "frames" / f"{short_exp}_{roi}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = p3_dir / f"{short_exp}_{roi}.mp4"

    font_title = load_font(14)
    font_col = load_font(13)
    font_small = load_font(12)

    for r, raw_full, stem in zip(rows, frames_raw, stems):
        visit = int(r["visit_index"])
        sweep = int(r["sweep_index"])
        raw_crop = raw_full[:, z0:z1, y0:y1, x0:x1]

        base_top = composite(raw_crop, axis=0)
        base_side = composite(raw_crop, axis=1)

        col_panels: List[Tuple[Image.Image, Image.Image]] = []
        for cand_code, _ in CANDIDATES_P3:
            if cand_code == "C3":
                npz_path = REPO_ROOT / "data" / "pilot_seg" / "C3_fixed" / exp / roi / f"{stem}.npz"
                if not npz_path.exists():
                    npz_path = REPO_ROOT / "data" / "pilot_seg" / "C3" / exp / roi / f"{stem}.npz"
            else:
                npz_path = REPO_ROOT / "data" / "pilot_seg" / cand_code / exp / roi / f"{stem}.npz"

            data = np.load(npz_path)
            t_inst = data["t_inst"][crop_sl]
            b_inst = data["b_inst"][crop_sl]

            info = score_lookup.get((visit, cand_code))
            if info is not None:
                tid = int(info["tracked_t_id"])
                bid = int(info["tracked_b_id"])
            else:
                tid = int(data["tracked_t_id"])
                bid = int(data["tracked_b_id"])

            t_mask = (t_inst == tid) if tid > 0 else np.zeros_like(t_inst, dtype=bool)
            b_mask = (b_inst == bid) if bid > 0 else np.zeros_like(b_inst, dtype=bool)

            rendered_views: List[Image.Image] = []
            for axis, base_rgb, target_h in [(0, base_top, top_h), (1, base_side, side_h)]:
                rgb_up = np.array(
                    Image.fromarray(base_rgb).resize((panel_w, target_h), Image.NEAREST)
                )

                for inst_arr, tr_id in [(b_inst, bid), (t_inst, tid)]:
                    for uid in np.unique(inst_arr):
                        if uid == 0 or int(uid) == tr_id:
                            continue
                        proj_o = (inst_arr == uid).max(axis=axis)
                        if np.any(proj_o):
                            proj_o_up = resize_bool_nearest(proj_o, panel_w, target_h)
                            rgb_up[outline_2d(proj_o_up, thickness=1)] = (155, 155, 155)

                proj_b = b_mask.max(axis=axis)
                if np.any(proj_b):
                    proj_b_up = resize_bool_nearest(proj_b, panel_w, target_h)
                    rgb_up[outline_2d(proj_b_up, thickness=2)] = (90, 160, 255)

                proj_t = t_mask.max(axis=axis)
                if np.any(proj_t):
                    proj_t_up = resize_bool_nearest(proj_t, panel_w, target_h)
                    rgb_up[outline_2d(proj_t_up, thickness=2)] = (255, 150, 40)

                rendered_views.append(Image.fromarray(rgb_up))

            col_panels.append((rendered_views[0], rendered_views[1]))

        canvas = Image.new("RGB", (total_w, total_h), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)

        dr.text(
            (left_label_w, 6),
            f"{short_exp} {roi}  |  Visit {visit:02d}/20  (t = {(visit - 1) * 10} min, sweep {sweep})",
            fill=(242, 242, 242),
            font=font_title,
        )

        dr.text((4, top_header_h + 6), "XY", fill=(200, 200, 200), font=font_small)
        dr.text((4, top_header_h + top_h + row_gap + 6), "XZ", fill=(200, 200, 200), font=font_small)

        pr_y0 = top_header_h + top_h + row_gap + side_h + 4
        for c_idx, (cand_code, col_label) in enumerate(CANDIDATES_P3):
            x_off = left_label_w + c_idx * (panel_w + col_gap)
            dr.text((x_off, 26), col_label, fill=(242, 242, 242), font=font_col)
            top_im, side_im = col_panels[c_idx]
            canvas.paste(top_im, (x_off, top_header_h))
            canvas.paste(side_im, (x_off, top_header_h + top_h + row_gap))

            info = score_lookup.get(
                (visit, cand_code),
                {
                    "target_track_id": 0,
                    "tcell_track_id": 0,
                    "b_surf": 0.0,
                    "b_miss": 0.0,
                    "b_phan": 0.0,
                    "t_surf": 0.0,
                    "t_miss": 0.0,
                    "t_phan": 0.0,
                },
            )
            dr.text(
                (x_off + 2, pr_y0),
                f"Target (tr={int(info['target_track_id'])}): surf={float(info['b_surf']):.2f}  miss={float(info['b_miss']):.2f}  phan={float(info['b_phan']):.2f}",
                fill=(130, 190, 255),
                font=font_small,
            )
            dr.text(
                (x_off + 2, pr_y0 + 16),
                f"T cell (tr={int(info['tcell_track_id'])}): surf={float(info['t_surf']):.2f}  miss={float(info['t_miss']):.2f}  phan={float(info['t_phan']):.2f}",
                fill=(255, 175, 85),
                font=font_small,
            )

        px_5um = max(10, int(round((5.0 / VOXEL_UM) * scale_factor)))
        sb_x1 = left_label_w + 3 * panel_w + 2 * col_gap - 10
        sb_x0 = sb_x1 - px_5um
        sb_y = top_header_h + top_h - 10
        dr.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=(255, 255, 255), width=3)
        dr.text((sb_x0, sb_y - 15), "5 um", fill=(255, 255, 255), font=font_small)

        footer_y = pr_y0 + pr_panel_h
        dr.text(
            (left_label_w, footer_y),
            "Raw: green ch1 (T cell) | magenta ch2 (target) | yellow ch0 (SYTOX >404 ADU)    "
            "Outlines: blue tracked target | orange tracked T cell | grey other instances",
            fill=(205, 205, 205),
            font=font_small,
        )

        jpg_path_pad = frames_dir / f"{visit:02d}.jpg"
        canvas.save(jpg_path_pad, format="JPEG", quality=82)

    for crf_val in ("26", "28", "30", "32"):
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(frames_dir / "%02d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            crf_val,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size_mb = movie_path.stat().st_size / (1024 * 1024)
        if size_mb <= 1.48:
            break
    print(f"Saved {movie_path} ({size_mb:.3f} MB, crf={crf_val}, {total_w}x{total_h})")
    return movie_path


def main_p3() -> None:
    for short_exp, exp, roi in PILOT_CHAMBERS:
        render_chamber_movie_p3(short_exp=short_exp, exp=exp, roi=roi)


CANDIDATES_P4 = [
    ("C0", "C0: Lab deposited"),
    ("C4", "C4: PlantSeg + channel ID (P3)"),
]


def render_chamber_movie_p4(
    short_exp: str = "v06",
    exp: str = "v06_6_40_120_0-6_hd8",
    roi: str = "MIP_ROI_28",
) -> Path:
    """Render 2-column (C0, C4) comparison movie and frames for held-out Pilot P4 chambers."""
    manifest_path = REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv"
    with open(manifest_path, "r") as f:
        all_rows = list(csv.DictReader(f))

    by_visit: Dict[int, List[dict]] = {}
    for r in all_rows:
        if r["experiment"] == exp and r["roi"] == roi and r["ok"] == "True" and r["sweep_index"] != "6":
            by_visit.setdefault(int(r["visit_index"]), []).append(r)

    rows = []
    for v in sorted(by_visit.keys()):
        vlist = by_visit[v]
        s1 = [r for r in vlist if r["sweep_index"] == "1"]
        if s1:
            rows.append(s1[0])
        else:
            s2 = [r for r in vlist if r["sweep_index"] == "2"]
            rows.append(s2[0] if s2 else sorted(vlist, key=lambda x: int(x["sweep_index"]))[0])

    n_visits = len(rows)
    metrics_csv = REPO_ROOT / "data" / "pilot_seg" / f"{short_exp}_{roi}_p4_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = REPO_ROOT / "data" / "pilot_seg" / "pilot4_metrics.csv"

    score_lookup: Dict[Tuple[int, str], Dict[str, object]] = {}
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        mdf_sub = mdf[(mdf["experiment"] == exp) & (mdf["roi"] == roi)]
        for _, mr in mdf_sub.iterrows():
            score_lookup[(int(mr["visit"]), str(mr["candidate"]))] = {
                "tracked_t_id": int(mr["tracked_t_id"]),
                "tracked_b_id": int(mr["tracked_b_id"]),
                "tcell_track_id": int(mr["tcell_track_id"]),
                "target_track_id": int(mr["target_track_id"]),
                "b_surf": float(mr["target_surf_score"]),
                "b_miss": float(mr["target_missed_2vox"]),
                "b_phan": float(mr["target_phantom_frac"]),
                "t_surf": float(mr["tcell_surf_score"]),
                "t_miss": float(mr["tcell_missed_2vox"]),
                "t_phan": float(mr["tcell_phantom_frac"]),
            }

    frames_raw: List[np.ndarray] = []
    stems: List[str] = []
    z_min, y_min, x_min = 10**6, 10**6, 10**6
    z_max, y_max, x_max = 0, 0, 0

    for r in rows:
        stem = f"{int(r['stem']):05d}"
        stems.append(stem)
        with h5py.File(REPO_ROOT / r["path"], "r") as f_raw:
            raw = f_raw["raw"][:].astype(np.float32)
        with h5py.File(
            REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5",
            "r",
        ) as f_pred:
            pred = f_pred["predictions"][:]
        frames_raw.append(raw)

        lab, _ = ndi.label(pred > 0, structure=STRUCT26)
        cnt = np.bincount(lab.ravel())
        keep = np.where(cnt >= 2000)[0]
        keep = keep[keep > 0]
        clean = np.isin(lab, keep) if keep.size > 0 else (pred > 0)
        npz_c4 = REPO_ROOT / "data" / "pilot_seg" / "C4" / exp / roi / f"{stem}.npz"
        if npz_c4.exists():
            with np.load(npz_c4) as z4:
                clean = clean | (z4["t_inst"] > 0) | (z4["b_inst"] > 0)
        nz = np.argwhere(clean)
        if nz.size > 0:
            lo_zyx = nz.min(axis=0)
            hi_zyx = nz.max(axis=0)
            z_min = min(z_min, int(lo_zyx[0]))
            y_min = min(y_min, int(lo_zyx[1]))
            x_min = min(x_min, int(lo_zyx[2]))
            z_max = max(z_max, int(hi_zyx[0]))
            y_max = max(y_max, int(hi_zyx[1]))
            x_max = max(x_max, int(hi_zyx[2]))

    _, Z, Y, X = frames_raw[0].shape
    pad = 26
    z0, z1 = max(0, z_min - pad), min(Z, z_max + pad + 1)
    y0, y1 = max(0, y_min - pad), min(Y, y_max + pad + 1)
    x0, x1 = max(0, x_min - pad), min(X, x_max + pad + 1)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    print(f"[{short_exp} {roi}] P4 crop (cells + 5 um): Z=[{z0}:{z1}], Y=[{y0}:{y1}], X=[{x0}:{x1}]")

    lo = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 65.0))
        for c in range(3)
    ]
    hi = [
        float(np.percentile(np.concatenate([raw[c, z0:z1, y0:y1, x0:x1].max(0).ravel() for raw in frames_raw]), 99.8))
        for c in range(3)
    ]

    def norm(img: np.ndarray, c: int) -> np.ndarray:
        return np.clip((img - lo[c]) / max(1.0, hi[c] - lo[c]), 0.0, 1.0)

    def composite(raw_crop: np.ndarray, axis: int) -> np.ndarray:
        t = norm(raw_crop[1].max(axis), 1)
        g = norm(raw_crop[2].max(axis), 2)
        d = np.clip((ndi.gaussian_filter(raw_crop[0].max(axis), 3.0) - 404.0) / 20.0, 0.0, 1.0)
        rgb = np.stack([g + d, t + 0.8 * d, g], axis=-1)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    n_cols = len(CANDIDATES_P4)
    total_w = 960
    left_label_w = 28
    right_pad = 12
    col_gap = 10
    row_gap = 8
    top_header_h = 46
    pr_panel_h = 40
    bottom_footer_h = 26

    panel_w = (total_w - left_label_w - right_pad - (n_cols - 1) * col_gap) // n_cols  # 455 px
    x_span = max(1, x1 - x0)
    y_span = max(1, y1 - y0)
    z_span = max(1, z1 - z0)
    scale_factor = float(panel_w) / float(x_span)
    top_h = max(32, int(round(y_span * scale_factor)))
    side_h = max(32, int(round(z_span * scale_factor)))

    total_h = top_header_h + top_h + row_gap + side_h + pr_panel_h + bottom_footer_h
    if total_h % 2 != 0:
        total_h += 1

    p4_dir = REPO_ROOT / "figures" / "pilot" / "p4"
    frames_dir = p4_dir / "frames" / f"{short_exp}_{roi}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = p4_dir / f"{short_exp}_{roi}.mp4"

    font_title = load_font(14)
    font_col = load_font(13)
    font_small = load_font(12)

    for r, raw_full, stem in zip(rows, frames_raw, stems):
        visit = int(r["visit_index"])
        sweep = int(r["sweep_index"])
        raw_crop = raw_full[:, z0:z1, y0:y1, x0:x1]

        base_top = composite(raw_crop, axis=0)
        base_side = composite(raw_crop, axis=1)

        col_panels: List[Tuple[Image.Image, Image.Image]] = []
        for cand_code, _ in CANDIDATES_P4:
            npz_path = REPO_ROOT / "data" / "pilot_seg" / cand_code / exp / roi / f"{stem}.npz"
            data = np.load(npz_path)
            t_inst = data["t_inst"][crop_sl]
            b_inst = data["b_inst"][crop_sl]

            info = score_lookup.get((visit, cand_code))
            if info is not None:
                tid = int(info["tracked_t_id"])
                bid = int(info["tracked_b_id"])
            else:
                tid = int(data["tracked_t_id"])
                bid = int(data["tracked_b_id"])

            t_mask = (t_inst == tid) if tid > 0 else np.zeros_like(t_inst, dtype=bool)
            b_mask = (b_inst == bid) if bid > 0 else np.zeros_like(b_inst, dtype=bool)

            rendered_views: List[Image.Image] = []
            for axis, base_rgb, target_h in [(0, base_top, top_h), (1, base_side, side_h)]:
                rgb_up = np.array(
                    Image.fromarray(base_rgb).resize((panel_w, target_h), Image.NEAREST)
                )

                for inst_arr, tr_id in [(b_inst, bid), (t_inst, tid)]:
                    for uid in np.unique(inst_arr):
                        if uid == 0 or int(uid) == tr_id:
                            continue
                        proj_o = (inst_arr == uid).max(axis=axis)
                        if np.any(proj_o):
                            proj_o_up = resize_bool_nearest(proj_o, panel_w, target_h)
                            rgb_up[outline_2d(proj_o_up, thickness=1)] = (155, 155, 155)

                proj_b = b_mask.max(axis=axis)
                if np.any(proj_b):
                    proj_b_up = resize_bool_nearest(proj_b, panel_w, target_h)
                    rgb_up[outline_2d(proj_b_up, thickness=2)] = (90, 160, 255)

                proj_t = t_mask.max(axis=axis)
                if np.any(proj_t):
                    proj_t_up = resize_bool_nearest(proj_t, panel_w, target_h)
                    rgb_up[outline_2d(proj_t_up, thickness=2)] = (255, 150, 40)

                rendered_views.append(Image.fromarray(rgb_up))

            col_panels.append((rendered_views[0], rendered_views[1]))

        canvas = Image.new("RGB", (total_w, total_h), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)

        dr.text(
            (left_label_w, 6),
            f"{short_exp} {roi}  |  Visit {visit:02d}/{n_visits:02d}  (t = {(visit - 1) * 10} min, sweep {sweep})",
            fill=(242, 242, 242),
            font=font_title,
        )

        dr.text((4, top_header_h + 6), "XY", fill=(200, 200, 200), font=font_small)
        dr.text((4, top_header_h + top_h + row_gap + 6), "XZ", fill=(200, 200, 200), font=font_small)

        pr_y0 = top_header_h + top_h + row_gap + side_h + 4
        for c_idx, (cand_code, col_label) in enumerate(CANDIDATES_P4):
            x_off = left_label_w + c_idx * (panel_w + col_gap)
            dr.text((x_off, 26), col_label, fill=(242, 242, 242), font=font_col)
            top_im, side_im = col_panels[c_idx]
            canvas.paste(top_im, (x_off, top_header_h))
            canvas.paste(side_im, (x_off, top_header_h + top_h + row_gap))

            info = score_lookup.get(
                (visit, cand_code),
                {
                    "target_track_id": 0,
                    "tcell_track_id": 0,
                    "b_surf": 0.0,
                    "b_miss": 0.0,
                    "b_phan": 0.0,
                    "t_surf": 0.0,
                    "t_miss": 0.0,
                    "t_phan": 0.0,
                },
            )
            dr.text(
                (x_off + 2, pr_y0),
                f"Target (tr={int(info['target_track_id'])}): surf={float(info['b_surf']):.2f}  miss={float(info['b_miss']):.2f}  phan={float(info['b_phan']):.2f}",
                fill=(130, 190, 255),
                font=font_small,
            )
            dr.text(
                (x_off + 2, pr_y0 + 16),
                f"T cell (tr={int(info['tcell_track_id'])}): surf={float(info['t_surf']):.2f}  miss={float(info['t_miss']):.2f}  phan={float(info['t_phan']):.2f}",
            fill=(255, 175, 85),
                font=font_small,
            )

        px_5um = max(10, int(round((5.0 / VOXEL_UM) * scale_factor)))
        sb_x1 = left_label_w + n_cols * panel_w + (n_cols - 1) * col_gap - 10
        sb_x0 = sb_x1 - px_5um
        sb_y = top_header_h + top_h - 10
        dr.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=(255, 255, 255), width=3)
        dr.text((sb_x0, sb_y - 15), "5 um", fill=(255, 255, 255), font=font_small)

        footer_y = pr_y0 + pr_panel_h
        dr.text(
            (left_label_w, footer_y),
            "Raw: green ch1 (T cell) | magenta ch2 (target) | yellow ch0 (SYTOX >404 ADU)    "
            "Outlines: blue tracked target | orange tracked T cell | grey other instances",
            fill=(205, 205, 205),
            font=font_small,
        )

        jpg_path_pad = frames_dir / f"{visit:02d}.jpg"
        canvas.save(jpg_path_pad, format="JPEG", quality=82)

    for crf_val in ("26", "28", "30", "32", "34", "36"):
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(frames_dir / "%02d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            crf_val,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size_mb = movie_path.stat().st_size / (1024 * 1024)
        if size_mb <= 1.48:
            break
    print(f"Saved {movie_path} ({size_mb:.3f} MB, crf={crf_val}, {total_w}x{total_h})")
    return movie_path


def main_p4() -> None:
    for short_exp, exp, roi in PILOT4_CHAMBERS:
        render_chamber_movie_p4(short_exp=short_exp, exp=exp, roi=roi)


if __name__ == "__main__":
    main_p4()
