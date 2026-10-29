#!/usr/bin/env python3
"""Local web application for panel-by-panel gallery review (stdlib http.server only).

Run:
    python3 tools/review_app.py [--port 8766] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import csv
import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from lab import (  # noqa: E402
    FIXED_REVIEW_TAGS,
    GALLERIES_CSV,
    REPO_ROOT,
    REVIEWS_DIR,
    VALID_VERDICTS,
    load_galleries,
    load_reviews,
)

_WRITE_LOCK = threading.Lock()

# Internal file-path columns hidden from the UI display table per House Style
_HIDDEN_INDEX_COLUMNS: frozenset[str] = frozenset(
    {
        "source_data_path",
        "panel_png_path",
        "image_path",
        "file_path",
        "raw_path",
        "mask_h5_path",
        "raw_h5_path",
        "movie",
        "movie_path",
        "video_path",
        "path",
    }
)

_HUMAN_FIELD_LABELS: dict[str, str] = {
    "case": "Group",
    "group": "Group",
    "id": "Panel ID",
    "panel_id": "Panel ID",
    "experiment": "Experiment",
    "roi": "Chamber",
    "visit": "Visit",
    "sweep": "Sweep",
    "condition": "Condition",
    "selection_tag": "Selection",
    "has_local_raw_h5": "Raw volume present",
    "voxel_size_um": "Voxel pitch (um)",
    "slice_zc_vox": "Slice Z center (vox)",
    "slice_yc_vox": "Slice Y center (vox)",
    "cup_depth_hull_p95_um": "Cup depth hull p95 (um)",
    "patch_equiv_radius_um": "Patch equiv radius (um)",
    "cup_depth_over_radius": "Cup depth / radius",
    "contact_area_um2": "Contact area (um^2)",
    "target_volume_um3": "Target volume (um^3)",
    "tcell_volume_um3": "T-cell volume (um^3)",
    "killed_200m": "Killed by 200 min",
    "ref_type": "Reference type",
    "pi_solved": "Solved Pi",
    "solved_depth_delta_um": "Solved depth delta (um)",
    "solved_depth_raw_hull_um": "Solved depth raw hull (um)",
    "max_curve_depth_delta_um": "Max curve depth delta (um)",
    "min_contact_sep_um": "Min contact sep (um)",
    "max_vol_rel_err": "Max volume rel error",
    "n_inverted": "Inverted faces",
    "converged": "Converged",
    "min_sep_before_push_um": "Min sep before push (um)",
    "push_apart_um": "Push apart (um)",
    "min_sep_after_push_um": "Min sep after push (um)",
    "d_raw_1p0_um": "Depth raw 1.0 um (um)",
    "d_raw_0p5_um": "Depth raw 0.5 um (um)",
    "d_delta_1p0_um": "Depth delta 1.0 um (um)",
    "d_delta_0p5_um": "Depth delta 0.5 um (um)",
    "rel_diff_raw_1p0_vs_0p5": "Rel diff 1.0 vs 0.5 um",
    "n_verts_target_1p0": "Target vertices (1.0 um)",
    "n_verts_target_0p5": "Target vertices (0.5 um)",
    "death_subcategory": "Death subcategory",
    "ever_dead_200m": "Sustained death (200 min)",
    "death_visit_200m": "Death visit (200 min)",
    "visits_shown": "Visits shown",
    "center_visit_dead_call": "Center visit dead call",
    "center_overlap_xy_px": "Center XY overlap (px)",
    "center_overlap_xz_px": "Center XZ overlap (px)",
    "center_max_gauss405_xy": "Center max 405 XY (ADU)",
    "center_max_gauss405_xz": "Center max 405 XZ (ADU)",
}


def _human_label(col: str) -> str:
    if col in _HUMAN_FIELD_LABELS:
        return _HUMAN_FIELD_LABELS[col]
    return col.replace("_", " ").capitalize()


def _load_gallery_index(gallery_row: dict[str, str]) -> tuple[bool, list[dict[str, Any]]]:
    """Load panels from a gallery's index_csv if it exists on disk."""
    index_rel = gallery_row.get("index_csv", "").strip()
    if not index_rel:
        return False, []
    index_path = (REPO_ROOT / index_rel).resolve()
    try:
        index_path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False, []
    if not index_path.is_file():
        return False, []

    panels: list[dict[str, Any]] = []
    try:
        with open(index_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                panel_id = (
                    row.get("id")
                    or row.get("panel_id")
                    or f"panel_{idx + 1:03d}"
                ).strip()
                group = (
                    row.get("case")
                    or row.get("group")
                    or "default"
                ).strip()
                png_rel = (
                    row.get("panel_png_path")
                    or row.get("image_path")
                    or ""
                ).strip()
                movie_rel = (
                    row.get("movie")
                    or row.get("movie_path")
                    or row.get("video_path")
                    or (row.get("path") if str(row.get("path", "")).endswith(".mp4") else "")
                    or ""
                ).strip()
                exp_val = (row.get("experiment") or row.get("exp") or "").strip()
                roi_val = (row.get("roi") or "").strip()
                numbers: list[dict[str, str]] = []
                for k, v in row.items():
                    if k is None or k in _HIDDEN_INDEX_COLUMNS:
                        continue
                    val_str = (v or "").strip()
                    if val_str == "":
                        continue
                    numbers.append(
                        {
                            "key": k,
                            "label": _human_label(k),
                            "value": val_str,
                        }
                    )
                panels.append(
                    {
                        "panel_id": panel_id,
                        "group": group,
                        "png_rel": png_rel,
                        "movie_rel": movie_rel,
                        "exp": exp_val,
                        "roi": roi_val,
                        "numbers": numbers,
                    }
                )
    except OSError:
        return False, []
    return True, panels


def _summarize_reviews_by_panel(
    review_rows: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    """Group all review rows by panel_id and compute latest per reviewer + disagreement."""
    by_panel: dict[str, list[dict[str, str]]] = {}
    for r in review_rows:
        pid = (r.get("panel_id") or "").strip()
        if not pid:
            continue
        by_panel.setdefault(pid, []).append(
            {
                "panel_id": pid,
                "reviewer": (r.get("reviewer") or "").strip() or "user",
                "verdict": (r.get("verdict") or "").strip(),
                "tags": (r.get("tags") or "").strip(),
                "note": (r.get("note") or "").strip(),
                "timestamp": (r.get("timestamp") or "").strip(),
            }
        )

    out: dict[str, dict[str, Any]] = {}
    for pid, rows in by_panel.items():
        latest_by_rev: dict[str, dict[str, str]] = {}
        for r in rows:
            latest_by_rev[r["reviewer"]] = r
        verdicts = {
            r["verdict"]
            for r in latest_by_rev.values()
            if r["verdict"] in VALID_VERDICTS
        }
        out[pid] = {
            "all_reviews": rows,
            "latest_by_reviewer": list(latest_by_rev.values()),
            "disagreement": len(verdicts) > 1,
        }
    return out


def append_review_row(
    gallery_id: str,
    panel_id: str,
    reviewer: str,
    verdict: str,
    tags: list[str],
    note: str,
) -> dict[str, str]:
    """Append a validated review row to lab/reviews/<gallery_id>.csv."""
    galleries = {g["gallery_id"]: g for g in load_galleries(GALLERIES_CSV)}
    if gallery_id not in galleries:
        raise ValueError(f"Unknown gallery_id: {gallery_id}")
    if not panel_id or not panel_id.strip():
        raise ValueError("panel_id is required.")
    verdict_clean = verdict.strip().lower()
    if verdict_clean not in VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict '{verdict}'. Must be one of {VALID_VERDICTS}."
        )
    rev_clean = (reviewer or "").strip() or "user"
    valid_tag_set = set(FIXED_REVIEW_TAGS)
    clean_tags = [t.strip() for t in tags if t and t.strip() in valid_tag_set]
    tags_str = ";".join(dict.fromkeys(clean_tags))
    note_clean = (note or "").replace("\r", " ").replace("\n", " ").strip()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    row_dict = {
        "panel_id": panel_id.strip(),
        "reviewer": rev_clean,
        "verdict": verdict_clean,
        "tags": tags_str,
        "note": note_clean,
        "timestamp": ts,
    }

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REVIEWS_DIR / f"{gallery_id}.csv"
    fieldnames = ["panel_id", "reviewer", "verdict", "tags", "note", "timestamp"]

    with _WRITE_LOCK:
        need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()
            writer.writerow(row_dict)

    return row_dict


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panel review</title>
<style>
/* house-style base.css tokens and components */
:root{
  --paper:#FFFFFF; --paper-2:#F6F6F6;
  --ink:#111111; --text:#1F1F1F; --muted:#5C5C5C;
  --rule:#E3E3E3; --line:#C9C9C9;
  --accent:#0B5FD1; --danger:#B3261E;
  --plot-1:#1F4E79; --plot-2:#C0561F; --plot-3:#2E7D5B; --plot-4:#6B4C9A;
  color-scheme:light;
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
body{
  margin:0;background:var(--paper);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:16px;line-height:1.5;
}
h1,h2,h3{color:var(--ink);font-weight:700;letter-spacing:-.01em;line-height:1.2;margin:0}
h1{font-size:22px}
h2{font-size:18px;margin:20px 0 8px}
h3{font-size:16px;margin:16px 0 6px}
p{margin:6px 0}
a{color:var(--accent)}
.meta,.muted{color:var(--muted);font-size:13px}
.label{color:var(--muted);font-size:13px;font-weight:600}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

.top{border-bottom:1px solid var(--rule);background:var(--paper)}
.top .in{
  max-width:1440px;margin:0 auto;padding:12px 20px;
  display:flex;align-items:center;gap:16px;flex-wrap:wrap;
}
.brand{font-size:18px;font-weight:700;color:var(--ink);text-decoration:none;white-space:nowrap}
.toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-left:auto}
.field-inline{display:inline-flex;align-items:center;gap:6px;font-size:14px}
.field-inline label{color:var(--muted);font-size:13px;font-weight:600}

.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border:1px solid var(--ink);border-radius:4px;padding:6px 12px;
  font:inherit;font-size:14px;font-weight:600;line-height:1.2;
  background:transparent;color:var(--ink);cursor:pointer;text-decoration:none;
}
.btn:hover{background:var(--paper-2)}
.btn.primary{background:var(--ink);color:var(--paper)}
.btn.danger-outline{border-color:var(--danger);color:var(--danger)}
.btn.danger-outline.primary{background:var(--danger);color:var(--paper)}
.btn:disabled{opacity:.35;pointer-events:none}

select,input[type=text],textarea{
  font:inherit;font-size:14px;color:var(--text);background:var(--paper);
  border:1px solid var(--line);border-radius:4px;padding:5px 8px;max-width:100%;
}
textarea{width:100%;resize:vertical}

.workspace{
  max-width:1440px;margin:0 auto;padding:16px 20px 60px;
  display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:24px;align-items:start;
}
@media (max-width:1024px){
  .workspace{grid-template-columns:1fr}
}

.nav-bar{
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  padding-bottom:10px;border-bottom:1px solid var(--rule);margin-bottom:12px;
}
.nav-status{font-size:15px;font-weight:600;color:var(--ink)}
.nav-spacer{margin-left:auto;display:flex;align-items:center;gap:8px}

.fig-full{
  border:1px solid var(--rule);border-radius:3px;padding:8px;
  background:var(--paper);overflow-x:auto;
}
.fig-full img{
  display:block;height:auto;max-width:100%;
}
.fig-full.natural-zoom img{
  max-width:none;
}

.review-card{
  border:1px solid var(--rule);border-radius:4px;padding:16px;
  background:var(--paper);position:sticky;top:12px;
}
.verdict-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 16px}
.tags-grid{
  display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px;
  margin:8px 0 16px;font-size:14px;
}
.tags-grid label{
  display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--text);
}
.save-row{display:flex;gap:8px;margin-top:12px}
.save-row .btn{flex:1}

table.data{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
table.data th,table.data td{padding:6px 10px 6px 0;text-align:left;vertical-align:top;border-top:1px solid var(--rule)}
table.data th{font-size:12px;color:var(--muted);font-weight:600;border-top:0;border-bottom:1px solid var(--ink)}
.tblwrap{overflow-x:auto;max-width:100%}

.disagree-alert{
  color:var(--danger);font-weight:700;font-size:14px;
  border-left:3px solid var(--danger);padding-left:8px;margin:8px 0;
}
tr.disagree-row td:first-child{
  border-left:3px solid var(--danger);padding-left:6px;
}
.verdict-text-ok{font-weight:700;color:var(--ink)}
.verdict-text-problem{font-weight:700;color:var(--danger)}
.verdict-text-unsure{font-weight:600;color:var(--muted)}
.status-msg{font-size:13px;color:var(--muted);margin-top:8px;min-height:18px}
</style>
</head>
<body>
<header class="top">
  <div class="in">
    <span class="brand">Panel review</span>
    <div class="toolbar">
      <div class="field-inline">
        <label for="sel-gallery">Gallery</label>
        <select id="sel-gallery" aria-label="Gallery"></select>
      </div>
      <div class="field-inline" id="wrap-group">
        <label for="sel-group">Group</label>
        <select id="sel-group" aria-label="Group"></select>
      </div>
      <label class="field-inline" for="chk-unreviewed">
        <input type="checkbox" id="chk-unreviewed"> Unreviewed
      </label>
      <label class="field-inline" for="chk-disagree">
        <input type="checkbox" id="chk-disagree"> Disagreements
      </label>
      <div class="field-inline">
        <label for="inp-reviewer">Reviewer</label>
        <input type="text" id="inp-reviewer" value="user" size="10" aria-label="Reviewer">
      </div>
    </div>
  </div>
</header>

<div class="workspace" id="main-workspace">
  <section aria-label="Panel display and data">
    <div class="nav-bar">
      <button type="button" class="btn sm" id="btn-prev">Previous</button>
      <button type="button" class="btn sm" id="btn-next">Next</button>
      <span class="nav-status" id="lbl-position">Loading...</span>
      <div class="nav-spacer">
        <label for="sel-panel" class="label">Jump</label>
        <select id="sel-panel" aria-label="Jump to panel"></select>
        <button type="button" class="btn sm" id="btn-zoom">Full pixels</button>
      </div>
    </div>

    <div id="empty-state" style="display:none;padding:24px 0">
      <p id="empty-msg" class="muted"></p>
    </div>

    <div id="panel-view">
      <div class="fig-full" id="fig-box">
        <img id="img-panel" alt="Gallery panel">
      </div>

      <h2>Verdicts</h2>
      <div id="disagree-banner" class="disagree-alert" style="display:none">
        Disagreement between reviewers on this panel.
      </div>
      <div class="tblwrap">
        <table class="data" id="tbl-verdicts">
          <thead>
            <tr>
              <th>Reviewer</th>
              <th>Verdict</th>
              <th>Tags</th>
              <th>Note</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody id="tbody-verdicts"></tbody>
        </table>
      </div>

      <h2>Numbers</h2>
      <div class="tblwrap">
        <table class="data" id="tbl-numbers">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody id="tbody-numbers"></tbody>
        </table>
      </div>
    </div>
  </section>

  <aside class="review-card" id="aside-controls" aria-label="Record verdict">
    <h2 style="margin-top:0">Verdict</h2>
    <p class="meta">Shortcuts: <strong>o</strong> OK, <strong>p</strong> Problem, <strong>u</strong> Unsure, <strong>[ / ]</strong> Prev / Next</p>
    <div class="verdict-row" role="group" aria-label="Verdict buttons">
      <button type="button" class="btn" id="btn-v-ok" data-verdict="ok">OK (o)</button>
      <button type="button" class="btn" id="btn-v-problem" data-verdict="problem">Problem (p)</button>
      <button type="button" class="btn" id="btn-v-unsure" data-verdict="unsure">Unsure (u)</button>
    </div>

    <h3>Tags</h3>
    <div class="tags-grid" id="tags-container"></div>

    <h3>Note</h3>
    <textarea id="txt-note" rows="3" aria-label="Note"></textarea>

    <div class="save-row">
      <button type="button" class="btn primary" id="btn-save">Save verdict</button>
      <button type="button" class="btn" id="btn-save-next">Save &amp; next</button>
    </div>
    <div class="status-msg" id="lbl-save-status" aria-live="polite"></div>
  </aside>
</div>

<script>
(function() {
  const TAGS = __FIXED_TAGS_JSON__;
  let galleries = [];
  let currentGalleryId = "";
  let allPanels = [];
  let filteredPanels = [];
  let reviewsByPanel = {};
  let currentIndex = 0;
  let selectedVerdict = "ok";
  let naturalZoom = false;

  const selGallery = document.getElementById("sel-gallery");
  const selGroup = document.getElementById("sel-group");
  const wrapGroup = document.getElementById("wrap-group");
  const chkUnreviewed = document.getElementById("chk-unreviewed");
  const chkDisagree = document.getElementById("chk-disagree");
  const inpReviewer = document.getElementById("inp-reviewer");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const lblPosition = document.getElementById("lbl-position");
  const selPanel = document.getElementById("sel-panel");
  const btnZoom = document.getElementById("btn-zoom");
  const figBox = document.getElementById("fig-box");
  const imgPanel = document.getElementById("img-panel");
  const emptyState = document.getElementById("empty-state");
  const emptyMsg = document.getElementById("empty-msg");
  const panelView = document.getElementById("panel-view");
  const asideControls = document.getElementById("aside-controls");
  const disagreeBanner = document.getElementById("disagree-banner");
  const tbodyVerdicts = document.getElementById("tbody-verdicts");
  const tbodyNumbers = document.getElementById("tbody-numbers");
  const tagsContainer = document.getElementById("tags-container");
  const txtNote = document.getElementById("txt-note");
  const btnSave = document.getElementById("btn-save");
  const btnSaveNext = document.getElementById("btn-save-next");
  const lblSaveStatus = document.getElementById("lbl-save-status");
  const verdictBtns = [
    document.getElementById("btn-v-ok"),
    document.getElementById("btn-v-problem"),
    document.getElementById("btn-v-unsure")
  ];

  // Build tag checkboxes
  TAGS.forEach(tag => {
    const lbl = document.createElement("label");
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.value = tag;
    chk.className = "tag-chk";
    lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode(" " + tag.replace(/_/g, " ")));
    tagsContainer.appendChild(lbl);
  });

  function setVerdictUI(v) {
    selectedVerdict = v;
    verdictBtns.forEach(btn => {
      if (btn.dataset.verdict === v) {
        btn.classList.add("primary");
      } else {
        btn.classList.remove("primary");
      }
    });
  }
  setVerdictUI("ok");

  function getCheckedTags() {
    return Array.from(document.querySelectorAll(".tag-chk:checked")).map(c => c.value);
  }

  function setCheckedTags(tagList) {
    const set = new Set(tagList || []);
    document.querySelectorAll(".tag-chk").forEach(c => {
      c.checked = set.has(c.value);
    });
  }

  function isPanelReviewedByCurrent(panelId) {
    const rev = (inpReviewer.value || "user").trim() || "user";
    const info = reviewsByPanel[panelId];
    if (!info || !info.latest_by_reviewer) return false;
    return info.latest_by_reviewer.some(r => r.reviewer === rev);
  }

  function hasPanelDisagreement(panelId) {
    const info = reviewsByPanel[panelId];
    return Boolean(info && info.disagreement);
  }

  function applyFilters(preservePanelId) {
    const grp = selGroup.value;
    const onlyUnrev = chkUnreviewed.checked;
    const onlyDis = chkDisagree.checked;

    filteredPanels = allPanels.filter(p => {
      if (grp && grp !== "__all__" && p.group !== grp) return false;
      if (onlyUnrev && isPanelReviewedByCurrent(p.panel_id)) return false;
      if (onlyDis && !hasPanelDisagreement(p.panel_id)) return false;
      return true;
    });

    selPanel.innerHTML = "";
    filteredPanels.forEach((p, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = p.panel_id + " (" + p.group + ")";
      selPanel.appendChild(opt);
    });

    if (preservePanelId) {
      const foundIdx = filteredPanels.findIndex(p => p.panel_id === preservePanelId);
      currentIndex = foundIdx >= 0 ? foundIdx : 0;
    } else {
      currentIndex = 0;
    }
    renderCurrentPanel();
  }

  function renderCurrentPanel() {
    if (filteredPanels.length === 0) {
      panelView.style.display = "none";
      asideControls.style.display = "none";
      emptyState.style.display = "block";
      btnPrev.disabled = true;
      btnNext.disabled = true;
      selPanel.style.display = "none";
      btnZoom.style.display = "none";
      if (allPanels.length === 0) {
        lblPosition.textContent = "0 panels";
        emptyMsg.textContent = "This gallery has no rendered panels on disk yet.";
      } else {
        lblPosition.textContent = "0 matching panels";
        emptyMsg.textContent = "No panels match the active filter.";
      }
      return;
    }

    panelView.style.display = "block";
    asideControls.style.display = "block";
    emptyState.style.display = "none";
    selPanel.style.display = "";
    btnZoom.style.display = "";

    if (currentIndex < 0) currentIndex = 0;
    if (currentIndex >= filteredPanels.length) currentIndex = filteredPanels.length - 1;

    const panel = filteredPanels[currentIndex];
    selPanel.value = String(currentIndex);
    btnPrev.disabled = (currentIndex === 0);
    btnNext.disabled = (currentIndex === filteredPanels.length - 1);

    lblPosition.textContent =
      "Panel " + (currentIndex + 1) + " of " + filteredPanels.length +
      " — " + panel.panel_id + " (" + panel.group + ")";

    imgPanel.src = "/image/" + encodeURIComponent(currentGalleryId) + "/" + encodeURIComponent(panel.panel_id);
    imgPanel.alt = panel.panel_id;

    // Render verdicts table
    const info = reviewsByPanel[panel.panel_id] || {latest_by_reviewer: [], all_reviews: [], disagreement: false};
    disagreeBanner.style.display = info.disagreement ? "block" : "none";
    tbodyVerdicts.innerHTML = "";
    if (info.latest_by_reviewer.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 5;
      td.className = "muted";
      td.textContent = "No reviews recorded yet.";
      tr.appendChild(td);
      tbodyVerdicts.appendChild(tr);
    } else {
      info.latest_by_reviewer.forEach(r => {
        const tr = document.createElement("tr");
        if (info.disagreement) tr.className = "disagree-row";

        const tdRev = document.createElement("td");
        tdRev.textContent = r.reviewer;
        tr.appendChild(tdRev);

        const tdV = document.createElement("td");
        tdV.textContent = r.verdict;
        tdV.className = "verdict-text-" + r.verdict;
        tr.appendChild(tdV);

        const tdTags = document.createElement("td");
        tdTags.textContent = r.tags ? r.tags.split(";").join(", ") : "—";
        tr.appendChild(tdTags);

        const tdNote = document.createElement("td");
        tdNote.textContent = r.note || "—";
        tr.appendChild(tdNote);

        const tdTs = document.createElement("td");
        tdTs.className = "muted";
        tdTs.textContent = r.timestamp || "—";
        tr.appendChild(tdTs);

        tbodyVerdicts.appendChild(tr);
      });
    }

    // Pre-fill controls if current reviewer already reviewed this panel
    const curRev = (inpReviewer.value || "user").trim() || "user";
    const existingMine = info.latest_by_reviewer.find(r => r.reviewer === curRev);
    if (existingMine) {
      setVerdictUI(existingMine.verdict || "ok");
      setCheckedTags(existingMine.tags ? existingMine.tags.split(";") : []);
      txtNote.value = existingMine.note || "";
    } else {
      setVerdictUI("ok");
      setCheckedTags([]);
      txtNote.value = "";
    }

    // Render numbers table
    tbodyNumbers.innerHTML = "";
    panel.numbers.forEach(item => {
      const tr = document.createElement("tr");
      const tdK = document.createElement("td");
      tdK.textContent = item.label;
      const tdV = document.createElement("td");
      tdV.textContent = item.value;
      tr.appendChild(tdK);
      tr.appendChild(tdV);
      tbodyNumbers.appendChild(tr);
    });
  }

  async function loadGallery(galleryId) {
    currentGalleryId = galleryId;
    lblSaveStatus.textContent = "";
    const resp = await fetch("/api/gallery/" + encodeURIComponent(galleryId));
    const data = await resp.json();
    allPanels = data.panels || [];
    reviewsByPanel = data.reviews_by_panel || {};

    const groups = data.groups || [];
    selGroup.innerHTML = "";
    if (groups.length <= 1) {
      wrapGroup.style.display = "none";
    } else {
      wrapGroup.style.display = "";
      const optAll = document.createElement("option");
      optAll.value = "__all__";
      optAll.textContent = "All groups (" + allPanels.length + ")";
      selGroup.appendChild(optAll);
      groups.forEach(g => {
        const opt = document.createElement("option");
        opt.value = g.name;
        opt.textContent = g.name + " (" + g.count + ")";
        selGroup.appendChild(opt);
      });
    }
    applyFilters(null);
  }

  async function saveReview(advanceNext) {
    if (filteredPanels.length === 0) return;
    const panel = filteredPanels[currentIndex];
    const payload = {
      gallery_id: currentGalleryId,
      panel_id: panel.panel_id,
      reviewer: (inpReviewer.value || "user").trim() || "user",
      verdict: selectedVerdict,
      tags: getCheckedTags(),
      note: txtNote.value.trim()
    };
    lblSaveStatus.textContent = "Saving...";
    const resp = await fetch("/api/review", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      lblSaveStatus.textContent = "Error: " + (err.error || resp.statusText);
      return;
    }
    const res = await resp.json();
    reviewsByPanel[panel.panel_id] = res.panel_reviews;
    lblSaveStatus.textContent = "Saved " + payload.verdict.toUpperCase() + " for " + panel.panel_id + ".";

    if (advanceNext && currentIndex + 1 < filteredPanels.length) {
      currentIndex += 1;
      renderCurrentPanel();
    } else {
      applyFilters(panel.panel_id);
    }
  }

  verdictBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      setVerdictUI(btn.dataset.verdict);
    });
  });

  btnSave.addEventListener("click", () => saveReview(false));
  btnSaveNext.addEventListener("click", () => saveReview(true));

  btnPrev.addEventListener("click", () => {
    if (currentIndex > 0) {
      currentIndex -= 1;
      lblSaveStatus.textContent = "";
      renderCurrentPanel();
    }
  });

  btnNext.addEventListener("click", () => {
    if (currentIndex + 1 < filteredPanels.length) {
      currentIndex += 1;
      lblSaveStatus.textContent = "";
      renderCurrentPanel();
    }
  });

  selPanel.addEventListener("change", () => {
    currentIndex = parseInt(selPanel.value, 10) || 0;
    lblSaveStatus.textContent = "";
    renderCurrentPanel();
  });

  btnZoom.addEventListener("click", () => {
    naturalZoom = !naturalZoom;
    figBox.classList.toggle("natural-zoom", naturalZoom);
    btnZoom.textContent = naturalZoom ? "Fit width" : "Full pixels";
  });

  selGallery.addEventListener("change", () => loadGallery(selGallery.value));
  selGroup.addEventListener("change", () => applyFilters(null));
  chkUnreviewed.addEventListener("change", () => applyFilters(null));
  chkDisagree.addEventListener("change", () => applyFilters(null));
  inpReviewer.addEventListener("change", () => renderCurrentPanel());

  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        saveReview(true);
      }
      return;
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key.toLowerCase();
    if (k === "o") {
      ev.preventDefault();
      setVerdictUI("ok");
      saveReview(false);
    } else if (k === "p") {
      ev.preventDefault();
      setVerdictUI("problem");
      saveReview(false);
    } else if (k === "u") {
      ev.preventDefault();
      setVerdictUI("unsure");
      saveReview(false);
    } else if (ev.key === "ArrowRight" || k === "]" || k === "j") {
      ev.preventDefault();
      if (currentIndex + 1 < filteredPanels.length) {
        currentIndex += 1;
        lblSaveStatus.textContent = "";
        renderCurrentPanel();
      }
    } else if (ev.key === "ArrowLeft" || k === "[" || k === "k") {
      ev.preventDefault();
      if (currentIndex > 0) {
        currentIndex -= 1;
        lblSaveStatus.textContent = "";
        renderCurrentPanel();
      }
    }
  });

  async function init() {
    const resp = await fetch("/api/galleries");
    const data = await resp.json();
    galleries = data.galleries || [];
    selGallery.innerHTML = "";
    galleries.forEach(g => {
      const opt = document.createElement("option");
      opt.value = g.gallery_id;
      opt.textContent = g.gallery_id;
      selGallery.appendChild(opt);
    });
    if (galleries.length > 0) {
      await loadGallery(galleries[0].gallery_id);
    }
  }

  init();
})();
</script>
</body>
</html>
""".replace(
    "__FIXED_TAGS_JSON__", json.dumps(list(FIXED_REVIEW_TAGS))
)


class ReviewAppHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local gallery review app."""

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet standard request logging unless needed
        pass

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
            return

        if path == "/api/galleries":
            rows = load_galleries(GALLERIES_CSV)
            out = []
            for r in rows:
                gid = r.get("gallery_id", "").strip()
                out.append(
                    {
                        "gallery_id": gid,
                        "description": r.get("description", ""),
                        "created": r.get("created", ""),
                    }
                )
            self._send_json({"galleries": out})
            return

        if path.startswith("/api/gallery/"):
            gallery_id = unquote(path[len("/api/gallery/") :]).strip("/")
            galleries = {g["gallery_id"]: g for g in load_galleries(GALLERIES_CSV)}
            if gallery_id not in galleries:
                self._send_json(
                    {"error": f"Unknown gallery '{gallery_id}'"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            g_row = galleries[gallery_id]
            avail, panels = _load_gallery_index(g_row)
            rev_rows = load_reviews(gallery_id)
            reviews_by_panel = _summarize_reviews_by_panel(rev_rows)

            group_counts: dict[str, int] = {}
            for p in panels:
                gname = p["group"]
                group_counts[gname] = group_counts.get(gname, 0) + 1
            groups = [{"name": k, "count": v} for k, v in group_counts.items()]

            self._send_json(
                {
                    "gallery": g_row,
                    "available": avail,
                    "panels": panels,
                    "groups": groups,
                    "reviews_by_panel": reviews_by_panel,
                }
            )
            return

        if path.startswith("/image/"):
            rest = path[len("/image/") :]
            parts = rest.split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image path")
                return
            gallery_id = unquote(parts[0]).strip()
            panel_id = unquote(parts[1]).strip()
            galleries = {g["gallery_id"]: g for g in load_galleries(GALLERIES_CSV)}
            if gallery_id not in galleries:
                self.send_error(HTTPStatus.NOT_FOUND, "Gallery not found")
                return
            avail, panels = _load_gallery_index(galleries[gallery_id])
            if not avail:
                self.send_error(HTTPStatus.NOT_FOUND, "Gallery index not available")
                return
            match = next((p for p in panels if p["panel_id"] == panel_id), None)
            if not match or not match["png_rel"]:
                self.send_error(HTTPStatus.NOT_FOUND, "Panel not found")
                return
            img_path = (REPO_ROOT / match["png_rel"]).resolve()
            try:
                img_path.relative_to(REPO_ROOT.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid file path")
                return
            if not img_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Image file missing")
                return
            data = img_path.read_bytes()
            ctype = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=300")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(
                {"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST
            )
            return

        gallery_id = str(payload.get("gallery_id") or "").strip()
        panel_id = str(payload.get("panel_id") or "").strip()
        reviewer = str(payload.get("reviewer") or "user").strip()
        verdict = str(payload.get("verdict") or "").strip()
        tags = payload.get("tags") or []
        if isinstance(tags, str):
            tags = [t for t in tags.split(";") if t.strip()]
        note = str(payload.get("note") or "")

        try:
            saved_row = append_review_row(
                gallery_id=gallery_id,
                panel_id=panel_id,
                reviewer=reviewer,
                verdict=verdict,
                tags=tags,
                note=note,
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        all_revs = load_reviews(gallery_id)
        summary = _summarize_reviews_by_panel(all_revs)
        panel_reviews = summary.get(
            panel_id,
            {"all_reviews": [], "latest_by_reviewer": [], "disagreement": False},
        )
        self._send_json({"saved": saved_row, "panel_reviews": panel_reviews})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="review_app.py",
        description="Local panel-by-panel review web app (port 8766).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Bind port (default: 8766)")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), ReviewAppHandler)
    print(f"Review app listening on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
