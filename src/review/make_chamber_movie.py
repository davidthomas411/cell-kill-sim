"""Prototype chamber movie: one frame per visit (sweep 1), raw 3-channel composite as
maximum-intensity projections (top view XY and side view XZ), with the lab's U-Net
mask projections outlined. Usage: python3 src/review/make_chamber_movie.py <experiment> <roi>"""
import csv, sys, collections
import h5py, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

exp, roi = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open("data/scidb_19235_raw/manifest.csv"))
        if r["experiment"] == exp and r["roi"] == roi and r["ok"] == "True" and r["sweep_index"] == "1"]
rows.sort(key=lambda r: int(r["visit_index"]))

def load(r):
    raw = h5py.File(r["path"])["raw"][:].astype(np.float32)          # (c, z, y, x)
    pred = h5py.File(f"data/scidb_19235_masks/{exp}/{roi}/{r['stem']}_predictions.h5")["predictions"][:]
    return raw, pred

frames_raw = [load(r) for r in rows]
# fixed per-channel display range across the whole movie, so brightness changes are real
lo = [np.percentile(np.concatenate([f[0][c].max(0).ravel() for f in frames_raw]), 1) for c in range(3)]
hi = [np.percentile(np.concatenate([f[0][c].max(0).ravel() for f in frames_raw]), 99.8) for c in range(3)]

def norm(img, c): return np.clip((img - lo[c]) / max(1.0, hi[c] - lo[c]), 0, 1)

def composite(raw, axis):
    t = norm(raw[1].max(axis), 1)      # channel 1: T cell -> green
    g = norm(raw[2].max(axis), 2)      # channel 2: target cells -> magenta
    # channel 0: death dye -> yellow; smoothed like src/death_labels.py and shown only above its 404 ADU threshold
    d = np.clip((ndi.gaussian_filter(raw[0].max(axis), 3.0) - 404.0) / 20.0, 0, 1)
    rgb = np.stack([g + d, t + 0.8 * d, g], -1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def outline(mask):
    return mask & ~ndi.binary_erosion(mask)

W = 2
frames = []
for (raw, pred), r in zip(frames_raw, rows):
    panels = []
    for axis in (0, 1):                                   # 0: top view (MIP over z), 1: side view (MIP over y)
        rgb = composite(raw, axis)
        tgt = ((pred == 2) | (pred == 4)).max(axis); tc = ((pred == 1) | (pred == 3)).max(axis)
        rgb[outline(tgt)] = (90, 160, 255); rgb[outline(tc)] = (255, 150, 40)
        im = Image.fromarray(rgb).resize((rgb.shape[1] * W, rgb.shape[0] * W), Image.NEAREST)
        panels.append(im)
    h = panels[0].height + panels[1].height + 40
    canvas = Image.new("RGB", (panels[0].width, h + 30), (0, 0, 0))
    canvas.paste(panels[0], (0, 30)); canvas.paste(panels[1], (0, 40 + panels[0].height))
    dr = ImageDraw.Draw(canvas)
    v = int(r["visit_index"])
    dr.text((6, 6), f"{exp.split('_')[0]} {roi}  visit {v}  t = {(v-1)*10} min  (sweep 1)", fill=(255, 255, 255))
    dr.text((6, 30 + panels[0].height + 4), "side view (XZ)", fill=(200, 200, 200))
    dr.text((6, 34), "top view (XY)", fill=(200, 200, 200))
    px = 5.0 / 0.1962 * W                                  # 5 um scale bar
    y0 = 30 + panels[0].height - 12
    dr.line([(canvas.width - 10 - px, y0), (canvas.width - 10, y0)], fill=(255, 255, 255), width=3)
    dr.text((canvas.width - 10 - px, y0 - 14), "5 um", fill=(255, 255, 255))
    dr.text((6, h + 6), "green T cell | magenta targets | yellow death dye | outlines: blue target mask, orange T-cell mask",
            fill=(220, 220, 220))
    frames.append(canvas)
out = f"figures/movies/{exp.split('_')[0]}_{roi}.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=600, loop=0)
print(out, len(frames))
