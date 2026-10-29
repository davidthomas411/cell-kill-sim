"""Mask figures for the guide (docs/guide/img/f4-f7), drawn on the raw 3-colour image.
Run in the seg env: conda run -n seg python tools/guide_mask_figures.py"""
import sys
from pathlib import Path
import h5py, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from src.seg.candidates import run_c0

IMG = R / "docs/guide/img"; IMG.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
BLUE, ORANGE = (90, 170, 255), (255, 160, 40)
F = 2  # upscale

def load(exp, roi, stem):
    raw = h5py.File(R / f"data/scidb_19235_raw/{exp}/{roi}/{stem}.h5")["raw"][:].astype(np.float32)
    pred = h5py.File(R / f"data/scidb_19235_masks/{exp}/{roi}/{stem}_predictions.h5")["predictions"][:]
    return raw, pred

def tracked_pair(pred):
    """Lab masks as deposited (C0); the pair of T cell and target with the largest contact."""
    t_inst, b_inst, _ = run_c0(pred)
    best = (0, None, None)
    for t in np.unique(t_inst)[1:]:
        tm = t_inst == t; near = ndi.binary_dilation(tm, iterations=3)
        for b in np.unique(b_inst)[1:]:
            bm = b_inst == b
            n = int((near & bm).sum())  # the lab masks leave a gap of a few voxels between cells
            if n > best[0] or best[1] is None: best = (n, tm, bm)
    return best[1], best[2]

def rgb(raw, take):
    def n(a, lo, hi): return np.clip((a - lo) / (hi - lo), 0, 1)
    t = n(take(raw[1]), 395, 460); g = n(take(raw[2]), 400, 560)
    d = np.clip((ndi.gaussian_filter(take(raw[0]), 3.0) - 404.0) / 20.0, 0, 1)
    return np.clip(np.stack([g + d, t + 0.8 * d, g], -1), 0, 1)

def outline(m): return m & ~ndi.binary_erosion(m)

def panel(img, masks, box, title, fill=None):
    """img HxWx3 float; masks list of (bool 2D, colour); box (y0,y1,x0,x1); fill = (mask, colour) shaded."""
    a = img.copy()
    if fill is not None:
        m, c = fill; a[m] = 0.55 * a[m] + 0.45 * np.array(c) / 255
    for m, c in masks:
        o = ndi.binary_dilation(outline(m)); a[o] = np.array(c) / 255
    y0, y1, x0, x1 = box
    im = Image.fromarray((a[y0:y1, x0:x1] * 255).astype(np.uint8))
    im = im.resize((im.width * F, im.height * F), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    dr.rectangle(dr.textbbox((10, 8), title, font=FONT), fill=(0, 0, 0)); dr.text((10, 8), title, font=FONT, fill=(255, 255, 255))
    px = 5 / 0.1962 * F; y = im.height - 18
    dr.line([(im.width - 16 - px, y), (im.width - 16, y)], fill=(255, 255, 255), width=4)
    dr.text((im.width - 16 - px / 2, y - 16), "5 µm", font=SMALL, fill=(255, 255, 255), anchor="mm")
    return im

def bbox(m2d, pad, shape):
    ys, xs = np.nonzero(m2d)
    return (max(0, ys.min() - pad), min(shape[0], ys.max() + pad), max(0, xs.min() - pad), min(shape[1], xs.max() + pad))

def row(ims, name):
    h = max(i.height for i in ims); out = Image.new("RGB", (sum(i.width for i in ims) + 6 * (len(ims) - 1), h), (255, 255, 255))
    x = 0
    for i in ims: out.paste(i, (x, 0)); x += i.width + 6
    out.save(IMG / name, quality=84); print("wrote", name, out.size)

def views(exp, roi, stem, name, slice_panel=False):
    raw, pred = load(exp, roi, stem)
    tm, bm = tracked_pair(pred)
    both = tm | bm
    top = rgb(raw, lambda a: a.max(0)); side = rgb(raw, lambda a: a.max(1))
    ims = [panel(top, [(bm.any(0), BLUE), (tm.any(0), ORANGE)], bbox(both.any(0), 25, top.shape), "seen from above"),
           panel(side, [(bm.any(1), BLUE), (tm.any(1), ORANGE)], bbox(both.any(1), 25, side.shape), "seen from the side")]
    if slice_panel:  # a cut through the middle of the target, mask shaded
        z = int(np.round(ndi.center_of_mass(bm)[0]))
        sl = rgb(raw, lambda a: a[z])
        ims.append(panel(sl, [(bm[z], BLUE)], bbox(both.any(0), 25, top.shape), "cut through the middle", fill=(bm[z], BLUE)))
    row(ims, name)

if __name__ == "__main__":
    views("v01_6_40_120_0-6_hd1", "MIP_ROI_5", "00991", "f4_good_mask.jpg")
    views("v12_6_35_120_0-6_hd10ds", "MIP_ROI_33", "00830", "f5_hollow.jpg", slice_panel=True)
    views("v01_6_40_120_0-6_hd1", "MIP_ROI_32", "01153", "f6_merged.jpg")  # visit 5
    views("v10_6_40_120_0-6_hd5", "MIP_ROI_23", "04219", "f7_real_cup.jpg")
