"""Build 4-column candidate comparison movie and per-visit JPEG frames for v01 MIP_ROI_32.

Follows display conventions of src/review/make_chamber_movie.py:
- Raw composite: green ch1 (T-cell Lifeact), magenta ch2 (target mApple-Mem),
  yellow ch0 (SYTOX death dye above 404 ADU after sigma=3 Gaussian), fixed display range.
- Cropped to cells + 5 um (25 voxels at 0.1962 um/voxel).
- 4 columns (C0..C3), each column showing top view (max over z) and side view (max over y).
- Outlines of PROJECTED 3D masks: tracked target in blue (90, 160, 255),
  tracked T cell in orange (255, 150, 40), other instances in thin grey (150, 150, 150).
- Saves figures/pilot/frames/<visit>.jpg (quality 80) and
  figures/pilot/v01_MIP_ROI_32_candidates.mp4 (H.264, crf 24, 2 fps, <= 3 MB).
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reproduce_contact_area import STRUCT26
VOXEL_UM = 0.1962
PAD_UM = 5.0
PAD_PX = int(round(PAD_UM / VOXEL_UM))  # 25 voxels (~4.91 um; use 26 for >= 5.0 um)

CANDIDATES = [
    ("C0", "C0: Lab deposited"),
    ("C1", "C1: Lab repaired"),
    ("C2", "C2: Cellpose-SAM 3D"),
    ("C3", "C3: PlantSeg 3D"),
]


def outline_2d(mask_2d: np.ndarray) -> np.ndarray:
    """1-pixel inner boundary outline of a 2D boolean mask."""
    if not np.any(mask_2d):
        return np.zeros_like(mask_2d, dtype=bool)
    return mask_2d & (~ndi.binary_erosion(mask_2d))


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


def main(exp: str = "v01_6_40_120_0-6_hd1", roi: str = "MIP_ROI_32") -> None:
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

    # Load raw volumes, predictions, and compute chamber-wide union cell crop (+ 5 um = 26 px)
    frames_raw: List[np.ndarray] = []
    stems: List[str] = []
    z_min, y_min, x_min = 10**6, 10**6, 10**6
    z_max, y_max, x_max = 0, 0, 0

    for r in rows:
        stem = f"{int(r['stem']):05d}"
        stems.append(stem)
        raw = h5py.File(REPO_ROOT / r["path"], "r")["raw"][:].astype(np.float32)
        pred = h5py.File(
            REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5",
            "r",
        )["predictions"][:]
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
    print(f"Chamber crop (cells + 5 um): Z=[{z0}:{z1}], Y=[{y0}:{y1}], X=[{x0}:{x1}]")

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

    scale = 2
    col_gap = 8
    row_gap = 8
    top_header_h = 46
    bottom_footer_h = 28
    left_label_w = 24

    frames_dir = REPO_ROOT / "figures" / "pilot" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = REPO_ROOT / "figures" / "pilot" / "v01_MIP_ROI_32_candidates.mp4"

    font_title = load_font(14)
    font_col = load_font(13)
    font_small = load_font(12)

    for idx, (r, raw_full, stem) in enumerate(zip(rows, frames_raw, stems)):
        visit = int(r["visit_index"])
        sweep = int(r["sweep_index"])
        raw_crop = raw_full[:, z0:z1, y0:y1, x0:x1]

        base_top = composite(raw_crop, axis=0)   # (Y_crop, X_crop, 3)
        base_side = composite(raw_crop, axis=1)  # (Z_crop, X_crop, 3)

        col_panels: List[Tuple[Image.Image, Image.Image]] = []
        for cand_code, _ in CANDIDATES:
            npz_path = REPO_ROOT / "data" / "pilot_seg" / cand_code / exp / roi / f"{stem}.npz"
            data = np.load(npz_path)
            t_inst = data["t_inst"][crop_sl]
            b_inst = data["b_inst"][crop_sl]
            t_mask = data["t_mask"][crop_sl] > 0
            b_mask = data["b_mask"][crop_sl] > 0
            tid = int(data["tracked_t_id"])
            bid = int(data["tracked_b_id"])

            rendered_views: List[Image.Image] = []
            for axis, base_rgb in [(0, base_top), (1, base_side)]:
                rgb_up = np.array(
                    Image.fromarray(base_rgb).resize(
                        (base_rgb.shape[1] * scale, base_rgb.shape[0] * scale),
                        Image.NEAREST,
                    )
                )

                # Other instances: thin 1px grey outline on upscaled panel
                for inst_arr, tr_id in [(b_inst, bid), (t_inst, tid)]:
                    for uid in np.unique(inst_arr):
                        if uid == 0 or int(uid) == tr_id:
                            continue
                        proj_o = (inst_arr == uid).max(axis=axis)
                        proj_o_up = np.repeat(np.repeat(proj_o, scale, axis=0), scale, axis=1)
                        rgb_up[outline_2d(proj_o_up)] = (155, 155, 155)

                # Tracked target: 2px blue outline of projected 3D mask
                proj_b = b_mask.max(axis=axis)
                if np.any(proj_b):
                    out_b = outline_2d(proj_b)
                    out_b_up = np.repeat(np.repeat(out_b, scale, axis=0), scale, axis=1)
                    rgb_up[out_b_up] = (90, 160, 255)

                # Tracked T cell: 2px orange outline of projected 3D mask
                proj_t = t_mask.max(axis=axis)
                if np.any(proj_t):
                    out_t = outline_2d(proj_t)
                    out_t_up = np.repeat(np.repeat(out_t, scale, axis=0), scale, axis=1)
                    rgb_up[out_t_up] = (255, 150, 40)

                rendered_views.append(Image.fromarray(rgb_up))

            col_panels.append((rendered_views[0], rendered_views[1]))

        panel_w = col_panels[0][0].width
        top_h = col_panels[0][0].height
        side_h = col_panels[0][1].height

        total_w = left_label_w + 4 * panel_w + 3 * col_gap + 12
        total_h = top_header_h + top_h + row_gap + side_h + bottom_footer_h
        # Ensure even dimensions for H.264 yuv420p
        if total_w % 2 != 0:
            total_w += 1
        if total_h % 2 != 0:
            total_h += 1

        canvas = Image.new("RGB", (total_w, total_h), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)

        # Top header: chamber + visit + time
        dr.text(
            (left_label_w, 6),
            f"v01 {roi}  |  Visit {visit:02d}/20  (t = {(visit - 1) * 10} min, sweep {sweep})",
            fill=(242, 242, 242),
            font=font_title,
        )

        # Row labels on left
        dr.text((4, top_header_h + 6), "XY", fill=(200, 200, 200), font=font_small)
        dr.text((4, top_header_h + top_h + row_gap + 6), "XZ", fill=(200, 200, 200), font=font_small)

        for c_idx, (_, col_label) in enumerate(CANDIDATES):
            x_off = left_label_w + c_idx * (panel_w + col_gap)
            dr.text((x_off, 26), col_label, fill=(242, 242, 242), font=font_col)
            top_im, side_im = col_panels[c_idx]
            canvas.paste(top_im, (x_off, top_header_h))
            canvas.paste(side_im, (x_off, top_header_h + top_h + row_gap))

        # 5 um scale bar on right-most XY panel
        px_5um = int(round(5.0 / VOXEL_UM * scale))
        sb_x1 = left_label_w + 4 * panel_w + 3 * col_gap - 10
        sb_x0 = sb_x1 - px_5um
        sb_y = top_header_h + top_h - 10
        dr.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=(255, 255, 255), width=3)
        dr.text((sb_x0, sb_y - 15), "5 um", fill=(255, 255, 255), font=font_small)

        # Footer legend
        footer_y = top_header_h + top_h + row_gap + side_h + 6
        dr.text(
            (left_label_w, footer_y),
            "Raw: green ch1 (T cell) | magenta ch2 (target) | yellow ch0 (SYTOX >404 ADU)    "
            "Outlines: blue tracked target | orange tracked T cell | grey other instances",
            fill=(210, 210, 210),
            font=font_small,
        )

        # Save both <visit>.jpg and zero-padded <visit:02d>.jpg at quality 80
        jpg_path_pad = frames_dir / f"{visit:02d}.jpg"
        jpg_path_raw = frames_dir / f"{visit}.jpg"
        canvas.save(jpg_path_pad, format="JPEG", quality=80)
        if jpg_path_raw != jpg_path_pad:
            canvas.save(jpg_path_raw, format="JPEG", quality=80)

    # Encode H.264 MP4 (crf 24, 2 fps) from zero-padded frames 01..20
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
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie_path),
    ]
    subprocess.run(cmd, check=True)
    size_mb = movie_path.stat().st_size / (1024 * 1024)
    print(f"Saved {movie_path} ({size_mb:.3f} MB)")


if __name__ == "__main__":
    main()
