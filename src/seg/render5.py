"""Render 2-column Phase 5 comparison movies and frames (Old C4 tracking vs New track5).

Reuses drawing helpers from src/seg/make_pilot_movie.py (outline_2d, resize_bool_nearest, load_font).
Left column:  C4 instances with OLD tracked IDs from the .npz (and P3/P4 metrics CSV).
Right column: <source> instances ('C4' default, 'C5' supported) with NEW track5 IDs.
Under each right panel prints: target track ID, T-cell track ID, contact area um^2, target volume um^3 (no score text).
Outputs:
  figures/pilot/p5/<short_exp>_<roi>.mp4 (< 1.5 MB)
  figures/pilot/p5/frames/<short_exp>_<roi>/<visit:02d>.jpg
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from src.reproduce_contact_area import STRUCT26
from src.seg.candidates import (
    FACE_AREA_UM2,
    PILOT4_CHAMBERS,
    PILOT_CHAMBERS,
    REPO_ROOT,
    VOXEL_UM,
    VOXEL_VOL_UM3,
    count_interface_faces,
)
from src.seg.make_pilot_movie import load_font, outline_2d, resize_bool_nearest
from src.seg.track5 import ALL_20_CHAMBERS, load_chamber_visits, track_chamber


def _load_old_track_lookup(exp: str, roi: str) -> Dict[int, Dict[str, object]]:
    """Load old P3/P4 track IDs for C4 if available."""
    out: Dict[int, Dict[str, object]] = {}
    for csv_name in ("pilot3_metrics.csv", "pilot4_metrics.csv"):
        p = REPO_ROOT / "data" / "pilot_seg" / csv_name
        if not p.exists():
            continue
        mdf = pd.read_csv(p)
        sub = mdf[(mdf["experiment"] == exp) & (mdf["roi"] == roi) & (mdf["candidate"] == "C4")]
        for _, r in sub.iterrows():
            out[int(r["visit"])] = {
                "tracked_t_id": int(r["tracked_t_id"]),
                "tracked_b_id": int(r["tracked_b_id"]),
                "tcell_track_id": int(r["tcell_track_id"]),
                "target_track_id": int(r["target_track_id"]),
                "contact_area_um2": float(r["contact_area_um2"]),
                "target_vol_um3": float(r["target_vol_um3"]),
            }
    return out


def render_chamber_movie_p5(
    short_exp: str = "v01",
    exp: str = "v01_6_40_120_0-6_hd1",
    roi: str = "MIP_ROI_2",
    source: str = "C4",
    out_dir: Path | str | None = None,
) -> Path:
    """Render 2-column (C4 old tracking vs <source> track5) movie and frames for one chamber."""
    t0 = time.perf_counter()
    chamber_id = f"{short_exp}_{roi}"
    rows = load_chamber_visits(exp, roi)
    n_visits = len(rows)

    p5_csv = REPO_ROOT / "data" / "pilot_seg" / "p5_tracks" / f"{chamber_id}.csv"
    if p5_csv.exists():
        p5_df = pd.read_csv(p5_csv)
        if "source" in p5_df.columns and str(p5_df["source"].iloc[0]) != source:
            p5_df = track_chamber(short_exp, exp, roi, source=source)
    else:
        p5_df = track_chamber(short_exp, exp, roi, source=source)

    p5_lookup: Dict[int, Dict[str, object]] = {}
    for _, r in p5_df.iterrows():
        p5_lookup[int(r["visit"])] = r.to_dict()

    old_lookup = _load_old_track_lookup(exp, roi)

    frames_raw: List[np.ndarray] = []
    stems: List[str] = []
    z_min, y_min, x_min = 10**6, 10**6, 10**6
    z_max, y_max, x_max = 0, 0, 0

    for r in rows:
        stem = str(r["stem"])
        stems.append(stem)
        with h5py.File(REPO_ROOT / str(r["rel_path"]), "r") as f_raw:
            raw = f_raw["raw"][:].astype(np.float32)
        frames_raw.append(raw)

        clean = np.zeros(raw.shape[1:], dtype=bool)
        for src_code in set(["C4", source]):
            npz_p = REPO_ROOT / "data" / "pilot_seg" / src_code / exp / roi / f"{stem}.npz"
            if npz_p.exists():
                with np.load(npz_p) as zf:
                    clean = clean | (zf["t_inst"] > 0) | (zf["b_inst"] > 0)
        if not np.any(clean):
            pred_p = REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5"
            if pred_p.exists():
                with h5py.File(pred_p, "r") as fp:
                    pred = fp["predictions"][:]
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
    pad = 26
    z0, z1 = max(0, z_min - pad), min(Z, z_max + pad + 1)
    y0, y1 = max(0, y_min - pad), min(Y, y_max + pad + 1)
    x0, x1 = max(0, x_min - pad), min(X, x_max + pad + 1)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

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

    cols = [
        ("C4", "C4: Old tracked IDs (P3/P4)", False),
        (source, f"{source}: New track5 IDs (P5)", True),
    ]
    n_cols = len(cols)
    total_w = 960
    left_label_w = 28
    right_pad = 12
    col_gap = 10
    row_gap = 8
    top_header_h = 46
    pr_panel_h = 40
    bottom_footer_h = 26

    panel_w = (total_w - left_label_w - right_pad - (n_cols - 1) * col_gap) // n_cols
    x_span = max(1, x1 - x0)
    y_span = max(1, y1 - y0)
    z_span = max(1, z1 - z0)
    scale_factor = float(panel_w) / float(x_span)
    top_h = max(32, int(round(y_span * scale_factor)))
    side_h = max(32, int(round(z_span * scale_factor)))

    total_h = top_header_h + top_h + row_gap + side_h + pr_panel_h + bottom_footer_h
    if total_h % 2 != 0:
        total_h += 1

    p5_dir = Path(out_dir) if out_dir is not None else (REPO_ROOT / "figures" / "pilot" / "p5")
    if not p5_dir.is_absolute():
        p5_dir = REPO_ROOT / p5_dir
    frames_dir = p5_dir / "frames" / chamber_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = p5_dir / f"{chamber_id}.mp4"

    font_title = load_font(14)
    font_col = load_font(13)
    font_small = load_font(12)

    for r, raw_full, stem in zip(rows, frames_raw, stems):
        visit = int(r["visit"])
        sweep = int(r["sweep"])
        raw_crop = raw_full[:, z0:z1, y0:y1, x0:x1]

        base_top = composite(raw_crop, axis=0)
        base_side = composite(raw_crop, axis=1)

        col_panels: List[Tuple[Image.Image, Image.Image]] = []
        col_texts: List[Tuple[str, str]] = []

        for src_code, _, is_new in cols:
            npz_path = REPO_ROOT / "data" / "pilot_seg" / src_code / exp / roi / f"{stem}.npz"
            with np.load(npz_path) as data:
                t_inst_full = data["t_inst"]
                b_inst_full = data["b_inst"]
                npz_tid = int(data["tracked_t_id"])
                npz_bid = int(data["tracked_b_id"])

            if is_new:
                info = p5_lookup[visit]
                tid = int(info["tracked_t_id"])
                bid = int(info["tracked_b_id"])
                t_tr = int(info["tcell_track_id"])
                b_tr = int(info["target_track_id"])
                contact_um2 = float(info["contact_area_um2"])
                b_vol_um3 = float(info["target_vol_um3"])
            else:
                tid = npz_tid
                bid = npz_bid
                old_info = old_lookup.get(visit)
                if old_info is not None:
                    t_tr = int(old_info["tcell_track_id"])
                    b_tr = int(old_info["target_track_id"])
                    contact_um2 = float(old_info["contact_area_um2"])
                    b_vol_um3 = float(old_info["target_vol_um3"])
                else:
                    t_tr = tid
                    b_tr = bid
                    tm_f = (t_inst_full == tid) if tid > 0 else np.zeros_like(t_inst_full, dtype=bool)
                    bm_f = (b_inst_full == bid) if bid > 0 else np.zeros_like(b_inst_full, dtype=bool)
                    faces = count_interface_faces(tm_f, bm_f) if (tid > 0 and bid > 0) else 0
                    contact_um2 = round(float(faces) * FACE_AREA_UM2, 3)
                    b_vol_um3 = round(int(bm_f.sum()) * VOXEL_VOL_UM3, 3)

            t_inst = t_inst_full[crop_sl]
            b_inst = b_inst_full[crop_sl]
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
            line1 = f"Target tr={b_tr}  |  T cell tr={t_tr}"
            line2 = f"Contact: {contact_um2:.1f} um^2  |  Target vol: {b_vol_um3:.1f} um^3"
            col_texts.append((line1, line2))

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
        for c_idx, (_, col_label, _) in enumerate(cols):
            x_off = left_label_w + c_idx * (panel_w + col_gap)
            dr.text((x_off, 26), col_label, fill=(242, 242, 242), font=font_col)
            top_im, side_im = col_panels[c_idx]
            canvas.paste(top_im, (x_off, top_header_h))
            canvas.paste(side_im, (x_off, top_header_h + top_h + row_gap))

            l1, l2 = col_texts[c_idx]
            dr.text((x_off + 2, pr_y0), l1, fill=(130, 190, 255), font=font_small)
            dr.text((x_off + 2, pr_y0 + 16), l2, fill=(255, 175, 85), font=font_small)

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

    dt = time.perf_counter() - t0
    print(f"Saved {movie_path} ({size_mb:.3f} MB, crf={crf_val}, {total_w}x{total_h}, {dt:.2f}s)")
    return movie_path


def render_all_p5(source: str = "C4", out_dir: Path | str | None = None) -> None:
    """Render all 20 chambers for Phase 5."""
    for short_exp, exp, roi in ALL_20_CHAMBERS:
        render_chamber_movie_p5(short_exp=short_exp, exp=exp, roi=roi, source=source, out_dir=out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Phase 5 comparison movies.")
    parser.add_argument("--source", type=str, default="C4", help="Instance source directory (C4 or C5).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for movies and frames.")
    args = parser.parse_args()
    render_all_p5(source=args.source, out_dir=args.out_dir)
