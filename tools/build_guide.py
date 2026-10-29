"""Build the plain-language project guide (docs/guide/) from lab/guide/guide.md plus
figures made from real data. Usage: python3 tools/build_guide.py"""
import csv, subprocess
from pathlib import Path
import h5py, numpy as np, markdown
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

R = Path(__file__).resolve().parents[1]; OUT = R / "docs/guide"; IMG = OUT / "img"; IMG.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
EXP = "v01_6_40_120_0-6_hd1"

def stems(roi):
    rows = [r for r in csv.DictReader(open(R / "data/scidb_19235_raw/manifest.csv"))
            if r["experiment"] == EXP and r["roi"] == roi and r["ok"] == "True" and r["sweep_index"] == "1"]
    return {int(r["visit_index"]): r["stem"] for r in rows}

def vol(roi, stem):
    raw = h5py.File(R / f"data/scidb_19235_raw/{EXP}/{roi}/{stem}.h5")["raw"][:].astype(np.float32)
    pred = h5py.File(R / f"data/scidb_19235_masks/{EXP}/{roi}/{stem}_predictions.h5")["predictions"][:]
    return raw, pred

def composite(raw, axis=0, project=True, z=None):
    take = (lambda a: a.max(axis)) if project else (lambda a: a[z])
    def n(a, lo, hi): return np.clip((a - lo) / (hi - lo), 0, 1)
    t = n(take(raw[1]), 395, 460); g = n(take(raw[2]), 400, 560)
    d = np.clip((ndi.gaussian_filter(take(raw[0]), 3.0) - 404.0) / 20.0, 0, 1)
    rgb = np.stack([g + d, t + 0.8 * d, g], -1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

def big(im, f=3): return im.resize((im.width * f, im.height * f), Image.NEAREST)

def label(im, items, title=None):
    dr = ImageDraw.Draw(im)
    for (x, y, tx, ty, text, col) in items:
        dr.line([(tx, ty), (x, y)], fill=col, width=3); dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
        dr.rectangle(dr.textbbox((tx, ty), text, font=FONT, anchor="mm"), fill=(0, 0, 0)); dr.text((tx, ty), text, font=FONT, fill=col, anchor="mm")
    if title: dr.text((12, 10), title, font=FONT, fill=(255, 255, 255))
    return im

def scalebar(im, f=3):
    dr = ImageDraw.Draw(im); px = 5 / 0.1962 * f; y = im.height - 22
    dr.line([(im.width - 20 - px, y), (im.width - 20, y)], fill=(255, 255, 255), width=4)
    dr.text((im.width - 20 - px / 2, y - 18), "5 µm", font=SMALL, fill=(255, 255, 255), anchor="mm")
    return im

def save(im, name): im.convert("RGB").save(IMG / name, quality=82); return name

def crop_panel(src, box, name, scale=1.0):
    im = Image.open(R / src).convert("RGB").crop(box)
    if scale != 1.0: im = im.resize((int(im.width * scale), int(im.height * scale)))
    return save(im, name)

def figures():
    s2 = stems("MIP_ROI_2")
    raw, pred = vol("MIP_ROI_2", s2[13])
    # 1. one moment, annotated (top view of a whole volume)
    im = big(composite(raw).crop((40, 60, 200, 230)))
    im = label(im, [(270, 300, 150, 30, "tumour cell (magenta)", (255, 120, 255)),
                    (360, 140, 380, 80, "T cell (green)", (120, 255, 120)),
                    (150, 140, 175, 470, "neighbour, out of focus", (220, 220, 220))])
    save(scalebar(im), "f1_one_moment.jpg")
    # 2. time strip
    tiles = []
    for v in (1, 5, 9, 13, 17, 20):
        r, _ = vol("MIP_ROI_2", s2[v]); t = big(composite(r).crop((40, 60, 200, 230)), 2)
        ImageDraw.Draw(t).text((10, 8), f"{(v - 1) * 10} min", font=FONT, fill=(255, 255, 255)); tiles.append(t)
    strip = Image.new("RGB", (tiles[0].width * 3 + 8, tiles[0].height * 2 + 4), (255, 255, 255))
    for k, t in enumerate(tiles): strip.paste(t, ((k % 3) * (t.width + 4), (k // 3) * (t.height + 4)))
    save(strip, "f2_time_strip.jpg")
    # 3. slice vs projection, and top vs side view
    zc = int(np.argmax(((pred == 2) | (pred == 4)).sum((1, 2))))
    a = big(composite(raw, project=False, z=zc).crop((40, 60, 200, 230)), 2)
    b = big(composite(raw).crop((40, 60, 200, 230)), 2)
    c = big(composite(raw, axis=1).crop((40, 40, 200, 160)), 2)
    for t, txt in ((a, "one slice"), (b, "top view (all slices)"), (c, "side view (all slices)")):
        ImageDraw.Draw(t).text((10, 8), txt, font=FONT, fill=(255, 255, 255))
    row = Image.new("RGB", (a.width + b.width + c.width + 8, max(a.height, c.height)), (255, 255, 255))
    row.paste(a, (0, 0)); row.paste(b, (a.width + 4, 0)); row.paste(c, (a.width + b.width + 8, 0))
    save(row, "f3_slice_vs_projection.jpg")
    # 4-7. real gallery panels (cropped to the image area)
    crop_panel("figures/review_clean/clean_random/clean_rand_03_v01_untr_3.png", (0, 30, 770, 787), "f4_good_mask.jpg")
    crop_panel("figures/review_clean/clean_random/clean_rand_47_v12_dasa_3.png", (0, 30, 770, 787), "f5_hollow.jpg")
    crop_panel("figures/review_clean/clean_extremes/clean_ext_04_min_cup_depth_rank1.png", (0, 30, 770, 787), "f6_merged.jpg")
    crop_panel("figures/review_clean/clean_random/clean_rand_39_v10_untr_3.png", (0, 30, 770, 787), "f7_real_cup.jpg")
    crop_panel("figures/review/cup_validation/b_dent_R5_a3_d2.png", (0, 0, 1100, 450), "f8_cup_test.jpg")
    crop_panel("figures/review/death_labels/death_killed_04.png", (0, 0, 900, 560), "f9_death.jpg")
    crop_panel("figures/review/fit_reachable/fit_reach_03.png", (0, 0, 900, 470), "f10_model.jpg")
    for name in ("v01_MIP_ROI_2", "v01_MIP_ROI_32"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(R / f"figures/movies/{name}.gif"), "-movflags", "+faststart",
                        "-pix_fmt", "yuv420p", "-vf", "fps=2,crop=iw:ih*0.55:0:0,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-crf", "26", str(IMG / f"{name}.mp4")], check=True)

def page():
    body = markdown.markdown(open(R / "lab/guide/guide.md").read(), extensions=["tables", "attr_list", "md_in_html"])
    css = """:root{--paper:#fbfbfa;--ink:#1d2430;--muted:#5a6473;--rule:#dfe3e8;--accent:#1f4e79;--note:#eef3f8;
    --serif:"Source Serif 4",Georgia,serif;--sans:"Source Sans 3","Segoe UI",system-ui,sans-serif}
    @media (prefers-color-scheme:dark){:root{color-scheme:dark;--paper:#12161c;--ink:#e6eaf0;--muted:#a0aab8;--rule:#2b3340;--accent:#8fb8e6;--note:#1a2230}}
    body{background:var(--paper);color:var(--ink);font:17px/1.6 var(--sans);margin:0;padding:2rem max(16px,env(safe-area-inset-left,0px)) 4rem}
    main{max-width:46rem;margin:0 auto} h1,h2,h3{font-family:var(--serif);line-height:1.2;text-wrap:balance}
    h1{font-size:2.3rem;margin:.2rem 0 1rem} h2{font-size:1.6rem;margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule)}
    figure{margin:1.5rem 0} figure img,figure video{width:100%;height:auto;border-radius:4px;background:#000}
    figcaption{font-size:.92rem;color:var(--muted);margin-top:.4rem} a{color:var(--accent)}
    table{border-collapse:collapse;width:100%;font-size:.95rem} td,th{border-bottom:1px solid var(--rule);padding:.4rem .5rem;text-align:left;vertical-align:top}
    .note{background:var(--note);border-radius:6px;padding:.8rem 1rem} .toc{font-size:.95rem}"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Cell-Kill-Sim: Start Here</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=Source+Sans+3:wght@400;600&display=swap">
<style>{css}</style></head><body><main>{body}</main></body></html>"""
    (OUT / "index.html").write_text(html)

if __name__ == "__main__":
    figures(); page(); print("built", OUT)
