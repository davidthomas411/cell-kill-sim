#!/usr/bin/env python3
"""Static site generator for the Cell-Kill-Sim lab record system (GitHub Pages).

Generates plain HTML/CSS into docs/ with:
  - docs/.nojekyll
  - docs/style.css (single CSS file, House Style light/dark via prefers-color-scheme)
  - docs/index.html (Gates G0-G5 status from lab files, galleries, notebook summary)
  - docs/notebook/index.html + docs/notebook/<id>.html
  - docs/protocols/index.html + docs/protocols/<stem>.html
  - docs/registry.html (dataset registry table with status labels)
  - docs/handbook/index.html + docs/handbook/<stem>.html
  - docs/galleries/index.html + docs/galleries/<gallery_id>.html
    with JPEG thumbnails (sips max width 1100 px, quality 70) in docs/galleries/<id>/
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

try:
    import markdown as _md_mod
except ImportError:
    _md_mod = None

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from lab import (  # noqa: E402
    GALLERIES_CSV,
    HANDBOOK_DIR,
    NOTEBOOK_DIR,
    PROTOCOLS_DIR,
    REGISTRY_CSV,
    REPO_ROOT,
    VALID_VERDICTS,
    list_notebook_entries,
    list_protocols,
    load_galleries,
    load_registry,
    load_reviews,
    parse_markdown_with_front_matter,
)
from review_app import (  # noqa: E402
    _HIDDEN_INDEX_COLUMNS,
    _human_label,
    _load_gallery_index,
    _summarize_reviews_by_panel,
)

DOCS_DIR: Path = REPO_ROOT / "docs"
MAX_JPEG_WIDTH: int = 1100
JPEG_QUALITY: int = 70

SITE_CSS = """/* house-style static site stylesheet (light/dark via prefers-color-scheme) */
:root{
  --paper:#FFFFFF; --paper-2:#F6F6F6;
  --ink:#111111; --text:#1F1F1F; --muted:#5C5C5C;
  --rule:#E3E3E3; --line:#C9C9C9;
  --accent:#0B5FD1; --danger:#B3261E;
  --plot-1:#1F4E79; --plot-2:#C0561F; --plot-3:#2E7D5B; --plot-4:#6B4C9A;
  color-scheme:light;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#121212; --paper-2:#1C1C1C; --ink:#F2F2F2; --text:#E6E6E6; --muted:#A8A8A8;
    --rule:#303030; --line:#4A4A4A; --accent:#7FB3FF; --danger:#FF8A7A;
    --plot-1:#8FBCE6; --plot-2:#E9A271; --plot-3:#7FC8A4; --plot-4:#B79BE0;
    color-scheme:dark;
  }
}
*,*::before,*::after{box-sizing:border-box}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{
  margin:0;background:var(--paper);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:17px;line-height:1.55;
}
h1,h2,h3{color:var(--ink);font-weight:700;letter-spacing:-.01em;line-height:1.2}
h1{font-size:30px;margin:0 0 10px}
h2{font-size:22px;margin:36px 0 10px}
h3{font-size:18px;margin:24px 0 8px}
p{margin:8px 0}
ul,ol{margin:8px 0;padding-left:22px}
li{margin:4px 0}
a{color:var(--accent)}
code,pre{
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:14px;
}
code{background:var(--paper-2);padding:1px 4px;border-radius:3px}
pre{
  background:var(--paper-2);border:1px solid var(--rule);border-radius:3px;
  padding:10px 14px;overflow-x:auto;line-height:1.45;
}
pre code{background:transparent;padding:0}
.meta,.muted{color:var(--muted);font-size:14px}
.label{color:var(--muted);font-size:13px;font-weight:600}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* header and navigation */
.top{border-bottom:1px solid var(--rule)}
.top .in{
  max-width:1080px;margin:0 auto;padding:14px 28px;
  display:flex;align-items:baseline;gap:24px;flex-wrap:wrap;
}
.brand{font-size:18px;font-weight:700;color:var(--ink);text-decoration:none;white-space:nowrap}
.tabs{display:flex;gap:22px;margin-left:auto;align-items:baseline;flex-wrap:wrap}
.tabs a{
  font-size:16px;text-decoration:none;color:var(--text);
  padding-bottom:3px;border-bottom:2px solid transparent;
}
.tabs a:hover{color:var(--ink)}
.tabs a.on{color:var(--ink);border-bottom-color:var(--ink);font-weight:600}

/* layout */
.wrap{
  max-width:1080px;margin:0 auto;padding:0 28px;
  display:grid;grid-template-columns:220px minmax(0,1fr);gap:40px;
}
.wrap.one{grid-template-columns:minmax(0,1fr)}
main{padding:28px 0 96px;min-width:0;max-width:760px}
.wrap.one main{max-width:1024px}
.side{
  padding:28px 0 40px 16px;position:sticky;top:0;height:100vh;
  overflow:auto;font-size:15px;
}
.side .section{margin:18px 0 4px;font-weight:700;color:var(--ink)}
.side a{
  display:flex;gap:10px;padding:3px 0;color:var(--text);
  text-decoration:none;line-height:1.35;position:relative;
}
.side a:hover{text-decoration:underline}
.side a.on{color:var(--ink);font-weight:600}
.side a.on::before{
  content:"";position:absolute;left:-14px;top:9px;
  width:6px;height:6px;border-radius:50%;background:var(--ink);
}

/* buttons and controls */
.btn{
  display:inline-flex;align-items:center;gap:8px;
  border:1px solid var(--ink);border-radius:4px;padding:6px 12px;
  font:inherit;font-size:14px;font-weight:600;line-height:1.2;
  background:transparent;color:var(--ink);cursor:pointer;text-decoration:none;
}
.btn:hover{background:var(--paper-2)}
.btn.primary{background:var(--ink);color:var(--paper)}
select,input[type=text],input[type=search]{
  font:inherit;font-size:14px;color:var(--text);background:var(--paper);
  border:1px solid var(--line);border-radius:4px;padding:6px 10px;max-width:100%;
}

/* tables: hairlines, no fills */
table.data,main table{
  border-collapse:collapse;width:100%;font-size:15px;
  font-variant-numeric:tabular-nums;margin:12px 0 20px;
}
table.data th,table.data td,main table th,main table td{
  padding:7px 12px 7px 0;text-align:left;vertical-align:top;
  border-top:1px solid var(--rule);
}
table.data th,main table th{
  font-size:13px;color:var(--muted);font-weight:600;
  border-top:0;border-bottom:1px solid var(--ink);
}
.tblwrap{overflow-x:auto;max-width:100%}

/* status labels (plain high-contrast text/hairline border, no garish fills) */
.status-pill{
  display:inline-block;font-size:13px;font-weight:600;line-height:1.2;
  padding:2px 6px;border:1px solid var(--line);border-radius:3px;
  color:var(--ink);background:transparent;white-space:nowrap;
}
.status-pill.st-verified,.status-pill.st-signed,.status-pill.st-reviewed,.status-pill.st-ok{
  border-color:var(--ink);color:var(--ink);font-weight:700;
}
.status-pill.st-unverified,.status-pill.st-draft,.status-pill.st-unsure{
  border-color:var(--line);color:var(--muted);
}
.status-pill.st-rejected,.status-pill.st-problem{
  border-color:var(--danger);color:var(--danger);font-weight:700;
}

/* gallery panel blocks */
.filter-bar{
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:12px 0;border-bottom:1px solid var(--rule);margin-bottom:20px;
}
.filter-bar label{font-size:13px;font-weight:600;color:var(--muted)}
.panel-card{
  border-top:1px solid var(--ink);padding:20px 0 28px;margin-top:20px;
}
.panel-head{
  display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;flex-wrap:wrap;margin-bottom:10px;
}
.panel-head h2{margin:0;font-size:20px}
.fig{
  border:1px solid var(--rule);border-radius:3px;padding:8px;
  background:var(--paper);max-width:100%;margin:8px 0 14px;
}
.fig img{display:block;max-width:100%;height:auto;width:100%}
.panel-grid{
  display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;
  align-items:start;
}
.disagree-note{
  color:var(--danger);font-weight:700;font-size:14px;margin:4px 0;
}

@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

@media (max-width:900px){
  .top .in{padding:12px 16px;gap:10px}
  .tabs{margin-left:0;width:100%;gap:14px;overflow-x:auto}
  .tabs a{white-space:nowrap}
  .wrap{grid-template-columns:1fr;padding:0 16px;gap:0}
  .side{display:none}
  main{padding:16px 0 72px;max-width:100%}
  h1{font-size:25px}
  .panel-grid{grid-template-columns:1fr;gap:12px}
}
"""


def _esc(val: Any) -> str:
    return html.escape(str(val if val is not None else ""))


def _status_pill(status: str) -> str:
    s = (status or "").strip()
    slug = re.sub(r"[^a-z0-9_-]+", "-", s.lower())
    return f'<span class="status-pill st-{_esc(slug)}">{_esc(s)}</span>'


def _render_md(md_text: str) -> str:
    """Convert Markdown text to HTML, rewriting internal .md links to .html."""
    # Rewrite relative links to .md files into .html links
    rewritten = re.sub(
        r"(\]\([^)#]+)\.md(#[^)]*)?\)",
        lambda m: f"{m.group(1)}.html{m.group(2) or ''})",
        md_text,
    )
    if _md_mod is not None:
        return _md_mod.markdown(
            rewritten,
            extensions=["tables", "fenced_code"],
        )
    # Fallback minimal converter if python-markdown is unavailable
    lines = rewritten.splitlines()
    out: list[str] = []
    in_pre = False
    for line in lines:
        if line.startswith("```"):
            if not in_pre:
                out.append("<pre><code>")
                in_pre = True
            else:
                out.append("</code></pre>")
                in_pre = False
            continue
        if in_pre:
            out.append(_esc(line))
            continue
        if line.startswith("### "):
            out.append(f"<h3>{_esc(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_esc(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_esc(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{_esc(line)}</p>")
    if in_pre:
        out.append("</code></pre>")
    return "\n".join(out)


def _page_shell(
    title: str,
    active_tab: str,
    body_html: str,
    root_prefix: str = "",
    sidebar_html: str | None = None,
) -> str:
    """Wrap page content in the standard House Style header and layout."""
    nav_items = [
        ("guide", "Start here", f"{root_prefix}guide/index.html"),
        ("gates", "Gates", f"{root_prefix}index.html"),
        ("notebook", "Notebook", f"{root_prefix}notebook/index.html"),
        ("protocols", "Protocols", f"{root_prefix}protocols/index.html"),
        ("registry", "Registry", f"{root_prefix}registry.html"),
        ("galleries", "Galleries", f"{root_prefix}galleries/index.html"),
        ("handbook", "Handbook", f"{root_prefix}handbook/index.html"),
    ]
    tabs_html = "\n".join(
        f'<a href="{href}" class="{"on" if key == active_tab else ""}">{label}</a>'
        for key, label, href in nav_items
    )
    wrap_cls = "wrap" if sidebar_html else "wrap one"
    side_block = f'<aside class="side">{sidebar_html}</aside>' if sidebar_html else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="{root_prefix}style.css">
</head>
<body>
<header class="top">
  <div class="in">
    <a class="brand" href="{root_prefix}index.html">Cell-Kill-Sim Lab</a>
    <nav class="tabs" aria-label="Primary">{tabs_html}</nav>
  </div>
</header>
<div class="{wrap_cls}">
  {side_block}
  <main>
    {body_html}
  </main>
</div>
</body>
</html>
"""


def _png_dimensions(png_path: Path) -> tuple[int, int] | None:
    """Read (width, height) from PNG IHDR chunk."""
    try:
        with open(png_path, "rb") as f:
            header = f.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", header[16:24])
            return int(w), int(h)
    except OSError:
        pass
    return None


def _convert_png_to_jpeg(
    src_png: Path, dst_jpg: Path, max_width: int = MAX_JPEG_WIDTH, quality: int = JPEG_QUALITY
) -> bool:
    """Convert src_png to dst_jpg using macOS sips (max width 1100 px, quality 70)."""
    if not src_png.is_file():
        return False
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    # Skip reconversion if dst_jpg is newer than src_png and non-empty
    try:
        if (
            dst_jpg.is_file()
            and dst_jpg.stat().st_size > 0
            and dst_jpg.stat().st_mtime >= src_png.stat().st_mtime
        ):
            return True
    except OSError:
        pass

    dims = _png_dimensions(src_png)
    cmd = [
        "sips",
        "-s",
        "format",
        "jpeg",
        "-s",
        "formatOptions",
        str(quality),
    ]
    if dims is None or dims[0] > max_width:
        cmd.extend(["--resampleWidth", str(max_width)])
    cmd.extend([str(src_png), "--out", str(dst_jpg)])
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return res.returncode == 0 and dst_jpg.is_file()


def _is_gallery_safe_to_read(gallery_row: dict[str, str]) -> bool:
    """Avoid touching figures/review_clean/ while another job is still writing it."""
    g_path = gallery_row.get("path", "").strip()
    if "review_clean" in g_path:
        # The clean-gallery job writes .agy_clean_gallery_report.md as its final step.
        if not (REPO_ROOT / ".agy_clean_gallery_report.md").is_file():
            return False
    return True


def build_home_page(
    entries: list[dict[str, Any]],
    registry: list[dict[str, str]],
    galleries: list[dict[str, str]],
) -> None:
    """Build docs/index.html showing Gates G0-G5 status from lab files."""
    gates_spec = [
        (
            "G0",
            "Reproduce published numbers",
            "trace_num==1 count equals paper row count (395/395 in 12 complete experiments); contact area within 0.5% for 369/419; exact 0.00% error on CA, ADC, ARFS with raw actin.",
            "verified",
            ["step3_reproduction"],
        ),
        (
            "G1",
            "Cell identity (masks to tracked T cell + contacted target)",
            "Tracked T cell and contacted target are each one whole, separate, solid cell away from the FOV edge. Layer-by-layer failure rates (U-Net, extraction, biology) verified on raw membrane images.",
            "unverified",
            ["lab_masks", "qc_flags", "clean_subset_v0.1"],
        ),
        (
            "G2",
            "Death labels",
            "Overlap restricted to contacted target in code and panels; neighbour-overlap false kills removed; gallery/table mismatch resolved; threshold sensitivity (403-410 ADU, 20-50 px) verified.",
            "unverified",
            ["step4_death_labels_v2"],
        ),
        (
            "G3",
            "Contact and cup descriptors",
            "Recomputed on G1-verified clean cells only; gallery of random and extreme cups shows real synaptic dents without 3D hull bridging across fragments; axial Z caveat kept.",
            "unverified",
            ["step5_descriptors"],
        ),
        (
            "G4",
            "Associations (Steps 5b, 7)",
            "Pre-registered Cox, donor, and drug-control analyses rerun on G1-G3 verified datasets, with earlier unverified numbers retained for comparison.",
            "unverified",
            ["step5b_associations", "step7_focused_pushing"],
        ),
        (
            "G5",
            "Mechanical model (Step 6b)",
            "Parked unless G4 shows an effect to explain. Requires rigid-body pinning, reference shape registration, path-independent re-solve reproducibility, and real-data fit gallery review.",
            "rejected",
            ["step6b_fits"],
        ),
    ]

    reg_by_id = {r["dataset_id"]: r for r in registry}
    entries_by_gate: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        entries_by_gate.setdefault(e["gate"], []).append(e)

    gate_rows_html: list[str] = []
    for gid, stage, req, default_st, ds_ids in gates_spec:
        # Derive current status from registry datasets for this gate
        ds_statuses = [
            reg_by_id[d]["status"] for d in ds_ids if d in reg_by_id
        ]
        if ds_statuses and all(s in ("verified", "verified_with_exclusions") for s in ds_statuses):
            gate_status = "verified"
        elif any(s == "rejected" for s in ds_statuses):
            gate_status = "rejected"
        else:
            gate_status = default_st

        ds_links = ", ".join(
            f'<a href="registry.html#ds-{_esc(d)}"><code>{_esc(d)}</code></a>'
            for d in ds_ids
        )
        g_entries = entries_by_gate.get(gid, [])
        nb_links = ", ".join(
            f'<a href="notebook/{_esc(e["id"])}.html">{_esc(e["id"])}</a>'
            for e in g_entries
        ) or "—"

        gate_rows_html.append(
            f"<tr>"
            f"<td><strong>{_esc(gid)}</strong></td>"
            f"<td>{_esc(stage)}</td>"
            f"<td>{_status_pill(gate_status)}</td>"
            f"<td>{_esc(req)}</td>"
            f"<td>{ds_links}</td>"
            f"<td>{nb_links}</td>"
            f"</tr>"
        )

    gal_rows_html: list[str] = []
    for g in galleries:
        gid = g["gallery_id"]
        revs = load_reviews(gid)
        summary = _summarize_reviews_by_panel(revs)
        n_rev = len(summary)
        gal_rows_html.append(
            f"<tr>"
            f'<td><a href="galleries/{_esc(gid)}.html"><strong>{_esc(gid)}</strong></a></td>'
            f"<td>{_esc(g.get('created', ''))}</td>"
            f"<td>{n_rev} reviewed panels</td>"
            f"<td>{_esc(g.get('description', ''))}</td>"
            f"</tr>"
        )

    nb_rows_html: list[str] = []
    for e in reversed(entries):
        nb_rows_html.append(
            f"<tr>"
            f"<td>{_esc(e['date'])}</td>"
            f'<td><a href="notebook/{_esc(e["id"])}.html">{_esc(e["title"])}</a></td>'
            f"<td>{_esc(e['gate'])}</td>"
            f"<td>{_status_pill(e['status'])}</td>"
            f"<td class=\"muted\">{_esc(e['author'])}</td>"
            f"</tr>"
        )

    body = f"""
    <h1>Gates</h1>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Gate</th>
            <th>Stage</th>
            <th>Status</th>
            <th>Requirement</th>
            <th>Datasets</th>
            <th>Notebook entries</th>
          </tr>
        </thead>
        <tbody>
          {"".join(gate_rows_html)}
        </tbody>
      </table>
    </div>

    <h2>Galleries</h2>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Gallery</th>
            <th>Date</th>
            <th>Reviews</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {"".join(gal_rows_html)}
        </tbody>
      </table>
    </div>

    <h2>Notebook</h2>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Date</th>
            <th>Entry</th>
            <th>Gate</th>
            <th>Status</th>
            <th>Author</th>
          </tr>
        </thead>
        <tbody>
          {"".join(nb_rows_html)}
        </tbody>
      </table>
    </div>
    """
    (DOCS_DIR / "index.html").write_text(
        _page_shell("Gates", "gates", body, root_prefix=""), encoding="utf-8"
    )


def build_notebook_pages(entries: list[dict[str, Any]]) -> None:
    """Build docs/notebook/index.html and docs/notebook/<id>.html."""
    nb_out_dir = DOCS_DIR / "notebook"
    nb_out_dir.mkdir(parents=True, exist_ok=True)

    sidebar_links = [
        '<div class="section">Entries</div>',
        '<a href="index.html">All entries</a>',
    ]
    for e in reversed(entries):
        sidebar_links.append(
            f'<a href="{_esc(e["id"])}.html">{_esc(e["id"][:13])} {_esc(e["gate"])}</a>'
        )
    sidebar_html = "\n".join(sidebar_links)

    rows_html: list[str] = []
    for e in reversed(entries):
        ds_str = ", ".join(e["datasets"]) if e["datasets"] else "—"
        rows_html.append(
            f"<tr>"
            f"<td>{_esc(e['date'])}</td>"
            f'<td><a href="{_esc(e["id"])}.html"><strong>{_esc(e["id"])}</strong></a><br>{_esc(e["title"])}</td>'
            f"<td>{_esc(e['gate'])}</td>"
            f"<td>{_status_pill(e['status'])}</td>"
            f"<td>{_esc(e['author'])}</td>"
            f"<td><code>{_esc(e['code_commit'])}</code></td>"
            f"<td>{_esc(ds_str)}</td>"
            f"</tr>"
        )

    idx_body = f"""
    <h1>Notebook</h1>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Date</th>
            <th>Entry</th>
            <th>Gate</th>
            <th>Status</th>
            <th>Author</th>
            <th>Commits</th>
            <th>Datasets</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    (nb_out_dir / "index.html").write_text(
        _page_shell(
            "Notebook",
            "notebook",
            idx_body,
            root_prefix="../",
            sidebar_html=sidebar_html,
        ),
        encoding="utf-8",
    )

    for e in entries:
        ds_links = (
            ", ".join(
                f'<a href="../registry.html#ds-{_esc(d)}"><code>{_esc(d)}</code></a>'
                for d in e["datasets"]
            )
            or "—"
        )
        prot_links = (
            ", ".join(
                f'<a href="../protocols/index.html#{_esc(p.replace("@", "-"))}"><code>{_esc(p)}</code></a>'
                for p in e["protocols"]
            )
            or "—"
        )
        gal_links = (
            ", ".join(
                f'<a href="../galleries/{_esc(g)}.html"><code>{_esc(g)}</code></a>'
                for g in e["galleries"]
            )
            or "—"
        )
        revs_str = ", ".join(e["reviewers"]) if e["reviewers"] else "—"
        signoff_str = str(e["signoff"]) if e["signoff"] else "—"
        sup_str = str(e["supersedes"]) if e["supersedes"] else "—"
        sup_by_str = str(e["superseded_by"]) if e["superseded_by"] else "—"

        rendered_body = _render_md(e["body"])
        entry_html = f"""
        <h1>{_esc(e['title'])}</h1>
        <div class="tblwrap">
          <table class="data">
            <tbody>
              <tr><th>ID</th><td><code>{_esc(e['id'])}</code></td><th>Date</th><td>{_esc(e['date'])}</td></tr>
              <tr><th>Gate</th><td><strong>{_esc(e['gate'])}</strong></td><th>Status</th><td>{_status_pill(e['status'])}</td></tr>
              <tr><th>Author</th><td>{_esc(e['author'])}</td><th>Reviewers</th><td>{_esc(revs_str)}</td></tr>
              <tr><th>Signoff</th><td>{_esc(signoff_str)}</td><th>Commits</th><td><code>{_esc(e['code_commit'])}</code></td></tr>
              <tr><th>Datasets</th><td>{ds_links}</td><th>Protocols</th><td>{prot_links}</td></tr>
              <tr><th>Galleries</th><td>{gal_links}</td><th>Supersedes / By</th><td>{_esc(sup_str)} / {_esc(sup_by_str)}</td></tr>
            </tbody>
          </table>
        </div>
        {rendered_body}
        """
        (nb_out_dir / f"{e['id']}.html").write_text(
            _page_shell(
                e["id"],
                "notebook",
                entry_html,
                root_prefix="../",
                sidebar_html=sidebar_html,
            ),
            encoding="utf-8",
        )


def build_protocols_pages(protocols: list[dict[str, Any]]) -> None:
    """Build docs/protocols/index.html and individual protocol pages."""
    prot_out_dir = DOCS_DIR / "protocols"
    prot_out_dir.mkdir(parents=True, exist_ok=True)

    sidebar_links = ['<div class="section">Protocols</div>']
    for p in protocols:
        anchor = f"{p['id']}-{p['version']}"
        sidebar_links.append(
            f'<a href="#{_esc(anchor)}">{_esc(p["id"])}@{_esc(p["version"])}</a>'
        )
    sidebar_html = "\n".join(sidebar_links)

    table_rows: list[str] = []
    sections_html: list[str] = []
    for p in protocols:
        anchor = f"{p['id']}-{p['version']}"
        table_rows.append(
            f"<tr>"
            f'<td><a href="#{_esc(anchor)}"><strong>{_esc(p["id"])}@{_esc(p["version"])}</strong></a></td>'
            f"<td>{_esc(p['date'])}</td>"
            f"<td>{_status_pill(p['status'])}</td>"
            f'<td><a href="{_esc(p["stem"])}.html">Standalone page</a></td>'
            f"</tr>"
        )
        rendered = _render_md(p["body"])
        sections_html.append(
            f'<section id="{_esc(anchor)}" style="border-top:1px solid var(--ink);padding-top:20px;margin-top:32px">'
            f'<p class="meta">Protocol <code>{_esc(p["id"])}@{_esc(p["version"])}</code> · Date {_esc(p["date"])} · {_status_pill(p["status"])}</p>'
            f"{rendered}"
            f"</section>"
        )

        single_body = (
            f'<p class="meta"><a href="index.html">All protocols</a> · '
            f'<code>{_esc(p["id"])}@{_esc(p["version"])}</code> · Date {_esc(p["date"])} · {_status_pill(p["status"])}</p>'
            f"{rendered}"
        )
        (prot_out_dir / f"{p['stem']}.html").write_text(
            _page_shell(
                f"{p['id']}@{p['version']}",
                "protocols",
                single_body,
                root_prefix="../",
                sidebar_html=sidebar_html,
            ),
            encoding="utf-8",
        )

    idx_body = f"""
    <h1>Protocols</h1>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Protocol</th>
            <th>Date</th>
            <th>Status</th>
            <th>Page</th>
          </tr>
        </thead>
        <tbody>
          {"".join(table_rows)}
        </tbody>
      </table>
    </div>
    {"".join(sections_html)}
    """
    (prot_out_dir / "index.html").write_text(
        _page_shell(
            "Protocols",
            "protocols",
            idx_body,
            root_prefix="../",
            sidebar_html=sidebar_html,
        ),
        encoding="utf-8",
    )


def build_registry_page(registry: list[dict[str, str]]) -> None:
    """Build docs/registry.html from lab/registry.csv."""
    rows_html: list[str] = []
    for r in registry:
        ds_id = r.get("dataset_id", "").strip()
        ver_by = r.get("verified_by", "").strip()
        ver_link = (
            f'<a href="notebook/{_esc(ver_by)}.html"><code>{_esc(ver_by)}</code></a>'
            if ver_by
            else "—"
        )
        rows_html.append(
            f'<tr id="ds-{_esc(ds_id)}">'
            f"<td><strong><code>{_esc(ds_id)}</code></strong><br><span class=\"muted\">{_esc(r.get('layer', ''))}</span></td>"
            f"<td>{_status_pill(r.get('status', ''))}</td>"
            f"<td>{_esc(r.get('n_items', ''))}</td>"
            f"<td>{_esc(r.get('description', ''))}</td>"
            f"<td><code>{_esc(r.get('code_commit', ''))}</code></td>"
            f"<td>{_esc(r.get('inputs', '') or '—')}</td>"
            f"<td>{ver_link}</td>"
            f"<td>{_esc(r.get('notes', ''))}</td>"
            f"</tr>"
        )

    body = f"""
    <h1>Registry</h1>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Status</th>
            <th>Items</th>
            <th>Description</th>
            <th>Commit</th>
            <th>Inputs</th>
            <th>Verified by</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    (DOCS_DIR / "registry.html").write_text(
        _page_shell("Registry", "registry", body, root_prefix=""),
        encoding="utf-8",
    )


def build_handbook_pages() -> None:
    """Build docs/handbook/index.html and individual handbook section pages."""
    hb_out_dir = DOCS_DIR / "handbook"
    hb_out_dir.mkdir(parents=True, exist_ok=True)

    hb_files = [
        ("data", "Dataset layers", HANDBOOK_DIR / "data.md"),
        ("definitions", "Definitions and caveats", HANDBOOK_DIR / "definitions.md"),
        ("known-issues", "Known issues", HANDBOOK_DIR / "known-issues.md"),
        ("open-questions", "Open questions", HANDBOOK_DIR / "open-questions.md"),
    ]

    sidebar_links = ['<div class="section">Handbook</div>']
    for slug, label, _ in hb_files:
        sidebar_links.append(f'<a href="#{_esc(slug)}">{_esc(label)}</a>')
    sidebar_html = "\n".join(sidebar_links)

    sections_html: list[str] = []
    for slug, label, path in hb_files:
        if not path.is_file():
            continue
        _, md_body = parse_markdown_with_front_matter(path)
        rendered = _render_md(md_body)
        sections_html.append(
            f'<section id="{_esc(slug)}" style="border-top:1px solid var(--ink);padding-top:16px;margin-top:28px">'
            f"{rendered}"
            f"</section>"
        )
        (hb_out_dir / f"{slug}.html").write_text(
            _page_shell(
                label,
                "handbook",
                f'<p class="meta"><a href="index.html">Handbook index</a></p>{rendered}',
                root_prefix="../",
                sidebar_html=sidebar_html,
            ),
            encoding="utf-8",
        )

    idx_body = f"""
    <h1>Handbook</h1>
    <ul>
      <li><a href="#data">Dataset layers</a> (<a href="data.html">page</a>)</li>
      <li><a href="#definitions">Definitions and caveats</a> (<a href="definitions.html">page</a>)</li>
      <li><a href="#known-issues">Known issues</a> (<a href="known-issues.html">page</a>)</li>
      <li><a href="#open-questions">Open questions for the Fei lab</a> (<a href="open-questions.html">page</a>)</li>
    </ul>
    {"".join(sections_html)}
    """
    (hb_out_dir / "index.html").write_text(
        _page_shell(
            "Handbook",
            "handbook",
            idx_body,
            root_prefix="../",
            sidebar_html=sidebar_html,
        ),
        encoding="utf-8",
    )


def build_gallery_pages(galleries: list[dict[str, str]]) -> None:
    """Build docs/galleries/index.html and docs/galleries/<id>.html with JPEG thumbnails."""
    gal_out_dir = DOCS_DIR / "galleries"
    gal_out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[str] = []

    for g in galleries:
        gid = g["gallery_id"].strip()
        thumb_dir = gal_out_dir / gid
        thumb_dir.mkdir(parents=True, exist_ok=True)

        if _is_gallery_safe_to_read(g):
            avail, panels = _load_gallery_index(g)
        else:
            avail, panels = False, []

        rev_rows = load_reviews(gid)
        reviews_by_panel = _summarize_reviews_by_panel(rev_rows)

        # Convert PNGs to JPEGs in parallel using sips
        conversion_tasks: list[tuple[Path, Path]] = []
        for p in panels:
            png_rel = p["png_rel"]
            if not png_rel:
                continue
            src_png = (REPO_ROOT / png_rel).resolve()
            dst_jpg = thumb_dir / f"{p['panel_id']}.jpg"
            conversion_tasks.append((src_png, dst_jpg))

        if conversion_tasks:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(
                    pool.map(
                        lambda item: _convert_png_to_jpeg(item[0], item[1]),
                        conversion_tasks,
                    )
                )

        index_rows.append(
            f"<tr>"
            f'<td><a href="{_esc(gid)}.html"><strong>{_esc(gid)}</strong></a></td>'
            f"<td>{_esc(g.get('created', ''))}</td>"
            f"<td>{len(panels) if avail else 'Pending'}</td>"
            f"<td>{len(reviews_by_panel)} reviewed</td>"
            f"<td>{_esc(g.get('description', ''))}</td>"
            f"</tr>"
        )

        if not avail or not panels:
            placeholder_body = f"""
            <h1>{_esc(gid)}</h1>
            <p>{_esc(g.get('description', ''))}</p>
            <p class="muted">Panels for this gallery are currently being generated or not yet on disk. Re-run <code>python3 tools/build_site.py</code> once rendering completes.</p>
            """
            (gal_out_dir / f"{gid}.html").write_text(
                _page_shell(gid, "galleries", placeholder_body, root_prefix="../"),
                encoding="utf-8",
            )
            continue

        # Collect groups for filter dropdown
        group_counts: dict[str, int] = {}
        for p in panels:
            group_counts[p["group"]] = group_counts.get(p["group"], 0) + 1

        group_opts = [f'<option value="__all__">All groups ({len(panels)})</option>']
        for gname, cnt in group_counts.items():
            group_opts.append(
                f'<option value="{_esc(gname)}">{_esc(gname)} ({cnt})</option>'
            )

        panel_cards_html: list[str] = []
        for p in panels:
            pid = p["panel_id"]
            grp = p["group"]
            r_info = reviews_by_panel.get(
                pid, {"all_reviews": [], "latest_by_reviewer": [], "disagreement": False}
            )
            all_revs = r_info["all_reviews"]
            disagree = r_info["disagreement"]

            verdict_tokens = {
                r["verdict"] for r in all_revs if r["verdict"] in VALID_VERDICTS
            }
            if not verdict_tokens:
                verdict_attr = "unreviewed"
            else:
                verdict_attr = " ".join(sorted(verdict_tokens))
            if disagree:
                verdict_attr += " disagreement"

            num_rows = "".join(
                f"<tr><td>{_esc(item['label'])}</td><td>{_esc(item['value'])}</td></tr>"
                for item in p["numbers"]
            )

            if all_revs:
                rev_rows_html = "".join(
                    f"<tr>"
                    f"<td><strong>{_esc(rv['reviewer'])}</strong></td>"
                    f"<td>{_status_pill(rv['verdict'])}</td>"
                    f"<td>{_esc(rv['tags'].replace(';', ', ') if rv['tags'] else '—')}</td>"
                    f"<td>{_esc(rv['note'] or '—')}</td>"
                    f"<td class=\"muted\">{_esc(rv['timestamp'] or '—')}</td>"
                    f"</tr>"
                    for rv in all_revs
                )
            else:
                rev_rows_html = (
                    '<tr><td colspan="5" class="muted">No review verdicts recorded yet.</td></tr>'
                )

            dis_html = (
                '<p class="disagree-note">Disagreement between reviewers.</p>'
                if disagree
                else ""
            )

            jpg_rel = f"{gid}/{pid}.jpg"
            panel_cards_html.append(
                f'<article class="panel-card" id="panel-{_esc(pid)}" '
                f'data-group="{_esc(grp)}" data-verdicts="{_esc(verdict_attr)}">'
                f'<div class="panel-head">'
                f'<h2>{_esc(pid)} <span class="muted">({_esc(grp)})</span></h2>'
                f'<a class="meta" href="#panel-{_esc(pid)}">#{_esc(pid)}</a>'
                f"</div>"
                f'<div class="fig">'
                f'<img src="{_esc(jpg_rel)}" alt="{_esc(pid)}" loading="lazy">'
                f"</div>"
                f'<div class="panel-grid">'
                f"<div>"
                f"<h3>Review verdicts ({len(all_revs)})</h3>"
                f"{dis_html}"
                f'<div class="tblwrap"><table class="data">'
                f"<thead><tr><th>Reviewer</th><th>Verdict</th><th>Tags</th><th>Note</th><th>Timestamp</th></tr></thead>"
                f"<tbody>{rev_rows_html}</tbody>"
                f"</table></div>"
                f"</div>"
                f"<div>"
                f"<h3>Numbers</h3>"
                f'<div class="tblwrap"><table class="data">'
                f"<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
                f"<tbody>{num_rows}</tbody>"
                f"</table></div>"
                f"</div>"
                f"</div>"
                f"</article>"
            )

        gallery_body = f"""
        <h1>{_esc(gid)}</h1>
        <p>{_esc(g.get('description', ''))}</p>
        <div class="filter-bar">
          <div>
            <label for="flt-group">Group</label>
            <select id="flt-group">{"".join(group_opts)}</select>
          </div>
          <div>
            <label for="flt-verdict">Verdict</label>
            <select id="flt-verdict">
              <option value="__all__">All verdicts</option>
              <option value="ok">ok</option>
              <option value="problem">problem</option>
              <option value="unsure">unsure</option>
              <option value="unreviewed">unreviewed</option>
              <option value="disagreement">disagreement</option>
            </select>
          </div>
          <span class="meta" id="flt-count">Showing {len(panels)} of {len(panels)} panels</span>
        </div>
        <div id="gallery-list">
          {"".join(panel_cards_html)}
        </div>
        <script>
        (function() {{
          const selG = document.getElementById("flt-group");
          const selV = document.getElementById("flt-verdict");
          const lblC = document.getElementById("flt-count");
          const cards = Array.from(document.querySelectorAll(".panel-card"));
          function filter() {{
            const g = selG.value;
            const v = selV.value;
            let shown = 0;
            cards.forEach(c => {{
              const cg = c.getAttribute("data-group") || "";
              const cv = (c.getAttribute("data-verdicts") || "").split(/\\s+/);
              const okG = (g === "__all__" || cg === g);
              const okV = (v === "__all__" || cv.includes(v));
              if (okG && okV) {{
                c.style.display = "";
                shown += 1;
              }} else {{
                c.style.display = "none";
              }}
            }});
            lblC.textContent = "Showing " + shown + " of " + cards.length + " panels";
          }}
          selG.addEventListener("change", filter);
          selV.addEventListener("change", filter);
        }})();
        </script>
        """
        (gal_out_dir / f"{gid}.html").write_text(
            _page_shell(gid, "galleries", gallery_body, root_prefix="../"),
            encoding="utf-8",
        )

    gal_idx_body = f"""
    <h1>Galleries</h1>
    <div class="tblwrap">
      <table class="data">
        <thead>
          <tr>
            <th>Gallery</th>
            <th>Created</th>
            <th>Panels</th>
            <th>Reviews</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {"".join(index_rows)}
        </tbody>
      </table>
    </div>
    """
    (gal_out_dir / "index.html").write_text(
        _page_shell("Galleries", "galleries", gal_idx_body, root_prefix="../"),
        encoding="utf-8",
    )


def build_site() -> None:
    """Build the entire static site under docs/."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (DOCS_DIR / "style.css").write_text(SITE_CSS, encoding="utf-8")

    entries = list_notebook_entries(NOTEBOOK_DIR)
    protocols = list_protocols(PROTOCOLS_DIR)
    registry = load_registry(REGISTRY_CSV)
    galleries = load_galleries(GALLERIES_CSV)

    build_home_page(entries, registry, galleries)
    build_notebook_pages(entries)
    build_protocols_pages(protocols)
    build_registry_page(registry)
    build_handbook_pages()
    build_gallery_pages(galleries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_site.py",
        description="Build static HTML/CSS lab site into docs/ for GitHub Pages.",
    )
    parser.parse_args(argv)
    build_site()
    print(f"Built static site in {DOCS_DIR.relative_to(REPO_ROOT)}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
