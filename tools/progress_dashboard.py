#!/usr/bin/env python3
"""Local live progress dashboard for Cell-Kill-Sim (re-segmentation phase).

Uses Python standard library only (argparse, collections, datetime, html,
http.server, json, os, pathlib, re, subprocess, sys, time, urllib.parse).
Run:
    python3 tools/progress_dashboard.py [--port 8765]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import html as html_mod
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

# Import portable paths from src/paths.py (with defensive fallback)
_REPO_FALLBACK = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_FALLBACK / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from paths import DATA_DIR, REPO_ROOT  # type: ignore
except Exception:
    REPO_ROOT = _REPO_FALLBACK
    DATA_DIR = REPO_ROOT / "data"

PLAN_PATH: Path = REPO_ROOT / "status" / "plan.json"
ABSTRACT_PATH: Path = REPO_ROOT / "lab" / "notebook" / "ABSTRACT.md"
NOTEBOOK_DIR: Path = REPO_ROOT / "lab" / "notebook"
PILOT_SEG_DIR: Path = DATA_DIR / "pilot_seg"
C4_DIR: Path = PILOT_SEG_DIR / "C4"
C5_DIR: Path = PILOT_SEG_DIR / "C5"
P5_TRACKS_DIR: Path = PILOT_SEG_DIR / "p5_tracks"
P5_MOVIES_DIR: Path = REPO_ROOT / "figures" / "pilot" / "p5"
CLI_LOG_PATH: Path = Path.home() / ".gemini" / "antigravity-cli" / "cli.log"

LOG_TAIL_BYTES: int = 256 * 1024  # 256 KB
REPORT_MAX_AGE_S: float = 24.0 * 3600.0  # 24 hours
VERDICT_ORDER: tuple[str, ...] = ("good", "ok", "merge", "track", "poor", "unclear")


def _human_bytes(num_bytes: int | float) -> str:
    val = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(val)} B"
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TB"


def _human_age(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    if sec < 60:
        return f"{sec}s ago"
    mins, secs = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m ago"
    hours, rem_m = divmod(mins, 60)
    return f"{hours}h {rem_m:02d}m ago"


def _inline_markdown(text: str) -> str:
    """Escape HTML and render **bold** and `code` spans."""
    escaped = html_mod.escape(text, quote=True)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def render_mini_markdown(md_text: str) -> str:
    """Render headings, paragraphs, nested bullets, numbered lists, and bold from markdown."""
    lines = md_text.splitlines()
    out: list[str] = []
    para_lines: list[str] = []

    # List state: stack of ("ul" | "ol", current_li_parts)
    list_stack: list[tuple[str, list[str]]] = []

    def flush_para() -> None:
        if para_lines:
            joined = " ".join(para_lines)
            out.append(f"<p>{_inline_markdown(joined)}</p>")
            para_lines.clear()

    def close_top_list() -> None:
        tag, li_parts = list_stack.pop()
        if li_parts:
            out.append(f"<li>{_inline_markdown(' '.join(li_parts))}</li>")
        out.append(f"</{tag}>")

    def close_all_lists() -> None:
        while list_stack:
            close_top_list()

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_para()
            close_all_lists()
            continue

        # Heading (#, ##, ###)
        m_hdr = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_hdr:
            flush_para()
            close_all_lists()
            out.append(f'<h3 class="abs-heading">{_inline_markdown(m_hdr.group(2).strip())}</h3>')
            continue

        # Indented sub-bullet (2+ leading spaces followed by - or *)
        m_sub_ul = re.match(r"^(\s{2,})[-*]\s+(.*)$", raw_line)
        if m_sub_ul:
            flush_para()
            item_text = m_sub_ul.group(2).strip()
            if not list_stack:
                list_stack.append(("ul", []))
                out.append("<ul>")
            if len(list_stack) == 1:
                # Keep parent <li> open while starting nested <ul>
                parent_tag, parent_li = list_stack[0]
                if parent_li:
                    out.append(f"<li>{_inline_markdown(' '.join(parent_li))}")
                    parent_li.clear()
                else:
                    out.append("<li>")
                out.append("<ul>")
                list_stack.append(("ul", [item_text]))
            else:
                # Already in sub-list
                _, sub_li = list_stack[-1]
                if sub_li:
                    out.append(f"<li>{_inline_markdown(' '.join(sub_li))}</li>")
                sub_li[:] = [item_text]
            continue

        # Top-level unordered bullet (- or *)
        m_ul = re.match(r"^[-*]\s+(.*)$", stripped)
        if m_ul:
            flush_para()
            item_text = m_ul.group(1).strip()
            if len(list_stack) > 1:
                close_top_list()
                out.append("</li>")
            if list_stack and list_stack[-1][0] != "ul":
                close_all_lists()
            if not list_stack:
                out.append("<ul>")
                list_stack.append(("ul", [item_text]))
            else:
                _, cur_li = list_stack[-1]
                if cur_li:
                    out.append(f"<li>{_inline_markdown(' '.join(cur_li))}</li>")
                cur_li[:] = [item_text]
            continue

        # Top-level ordered list item (1. , 2. , ...)
        m_ol = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m_ol:
            flush_para()
            item_text = m_ol.group(1).strip()
            if len(list_stack) > 1:
                close_top_list()
                out.append("</li>")
            if list_stack and list_stack[-1][0] != "ol":
                close_all_lists()
            if not list_stack:
                out.append("<ol>")
                list_stack.append(("ol", [item_text]))
            else:
                _, cur_li = list_stack[-1]
                if cur_li:
                    out.append(f"<li>{_inline_markdown(' '.join(cur_li))}</li>")
                cur_li[:] = [item_text]
            continue

        # Indented continuation line inside an active list item
        if list_stack and raw_line[:1].isspace():
            list_stack[-1][1].append(stripped)
            continue

        # Normal paragraph line
        close_all_lists()
        para_lines.append(stripped)

    flush_para()
    while list_stack:
        depth = len(list_stack)
        close_top_list()
        if depth > 1:
            out.append("</li>")

    return "\n".join(out)


def get_plan_section(plan_path: Path = PLAN_PATH) -> dict:
    """Read status/plan.json defensively."""
    if not plan_path.is_file():
        return {
            "state": "not available",
            "updated": "not available",
            "aim": "not available",
            "now": "not available",
            "total_chambers": 502,
            "phases": [],
            "stages_done": 0,
            "stages_total": 0,
            "stages_summary": "not available",
            "chambers": [],
            "verdict_counts": {},
            "verdict_tally": "not available",
            "needs_pi": [],
            "parked_note": "not available",
        }
    try:
        with open(plan_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        phases = data.get("phases", [])
        if not isinstance(phases, list):
            phases = []

        done_count = sum(1 for p in phases if isinstance(p, dict) and p.get("status") == "done")
        total_phases = len(phases)
        stages_summary = f"{done_count} of {total_phases} stages done" if total_phases > 0 else "0 of 0 stages done"

        chambers = data.get("chambers", [])
        if not isinstance(chambers, list):
            chambers = []

        counts: collections.Counter[str] = collections.Counter()
        for ch in chambers:
            if isinstance(ch, dict):
                v = str(ch.get("verdict", "unclear")).strip().lower() or "unclear"
                counts[v] += 1

        tally_parts: list[str] = []
        seen_verdicts: set[str] = set()
        for v in VERDICT_ORDER:
            if counts.get(v, 0) > 0:
                tally_parts.append(f"{counts[v]} {v}")
                seen_verdicts.add(v)
        for v, c in counts.items():
            if v not in seen_verdicts and c > 0:
                tally_parts.append(f"{c} {v}")
        verdict_tally = ", ".join(tally_parts) if tally_parts else "none"

        needs_pi = data.get("needs_pi", [])
        if not isinstance(needs_pi, list):
            needs_pi = []

        total_chambers = data.get("total_chambers", 502)
        if not isinstance(total_chambers, int) or total_chambers <= 0:
            total_chambers = 502

        return {
            "state": "ok",
            "updated": data.get("updated") or "not available",
            "aim": data.get("aim") or "not available",
            "now": data.get("now") or "not available",
            "total_chambers": total_chambers,
            "phases": phases,
            "stages_done": done_count,
            "stages_total": total_phases,
            "stages_summary": stages_summary,
            "chambers": chambers,
            "verdict_counts": dict(counts),
            "verdict_tally": verdict_tally,
            "needs_pi": [str(x) for x in needs_pi],
            "parked_note": data.get("parked_note") or "not available",
        }
    except Exception as exc:
        return {
            "state": "not available",
            "error": str(exc),
            "updated": "not available",
            "aim": "not available",
            "now": "not available",
            "total_chambers": 502,
            "phases": [],
            "stages_done": 0,
            "stages_total": 0,
            "stages_summary": "not available",
            "chambers": [],
            "verdict_counts": {},
            "verdict_tally": "not available",
            "needs_pi": [],
            "parked_note": "not available",
        }


def get_abstract_section(abstract_path: Path = ABSTRACT_PATH) -> dict:
    """Read and render lab/notebook/ABSTRACT.md defensively."""
    if not abstract_path.is_file():
        return {
            "state": "not available",
            "raw": None,
            "html": '<p class="muted">not available</p>',
        }
    try:
        raw = abstract_path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return {
                "state": "not available",
                "raw": "",
                "html": '<p class="muted">not available</p>',
            }
        return {
            "state": "ok",
            "raw": raw,
            "html": render_mini_markdown(raw),
        }
    except Exception as exc:
        return {
            "state": "not available",
            "error": str(exc),
            "raw": None,
            "html": '<p class="muted">not available</p>',
        }


def _parse_frontmatter(md_path: Path) -> dict[str, str]:
    """Parse simple YAML-ish frontmatter (id, date, title, gate, status, author) using stdlib."""
    stem = md_path.stem
    meta: dict[str, str] = {
        "filename": md_path.name,
        "id": stem,
        "date": stem[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", stem) else "not available",
        "title": stem,
        "gate": "none",
        "status": "draft",
        "author": "",
    }
    try:
        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
            if first.strip() != "---":
                return meta
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if s == "---":
                    break
                if ":" not in s or s.startswith("-"):
                    continue
                key, val = s.split(":", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                if key in ("id", "date", "title", "gate", "status", "author") and val:
                    meta[key] = val
    except OSError:
        pass
    return meta


def get_notebook_section(notebook_dir: Path = NOTEBOOK_DIR, limit: int = 6) -> dict:
    """Return the latest 6 lab/notebook/*.md entries by filename."""
    if not notebook_dir.is_dir():
        return {"state": "not available", "entries": []}
    try:
        candidates: list[Path] = []
        with os.scandir(notebook_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.endswith(".md"):
                    continue
                if name.upper() in ("ABSTRACT.MD", "README.MD"):
                    continue
                candidates.append(Path(entry.path))
        candidates.sort(key=lambda p: p.name, reverse=True)
        latest = candidates[:limit]
        entries = [_parse_frontmatter(p) for p in latest]
        return {"state": "ok", "entries": entries}
    except Exception as exc:
        return {"state": "not available", "error": str(exc), "entries": []}


def _scan_seg_chambers(seg_dir: Path) -> set[tuple[str, str]]:
    """Count (experiment, roi) subdirectories under seg_dir that contain at least one *.npz file."""
    found: set[tuple[str, str]] = set()
    if not seg_dir.is_dir():
        return found
    try:
        with os.scandir(seg_dir) as exp_it:
            for exp_entry in exp_it:
                if not exp_entry.is_dir(follow_symlinks=False) or exp_entry.name.startswith("."):
                    continue
                try:
                    with os.scandir(exp_entry.path) as roi_it:
                        for roi_entry in roi_it:
                            if not roi_entry.is_dir(follow_symlinks=False) or roi_entry.name.startswith("."):
                                continue
                            has_npz = False
                            try:
                                with os.scandir(roi_entry.path) as f_it:
                                    for f_entry in f_it:
                                        if (
                                            f_entry.is_file(follow_symlinks=False)
                                            and f_entry.name.endswith(".npz")
                                            and not f_entry.name.startswith(".")
                                            and ".part" not in f_entry.name
                                        ):
                                            has_npz = True
                                            break
                            except OSError:
                                pass
                            if has_npz:
                                found.add((exp_entry.name, roi_entry.name))
                except OSError:
                    continue
    except OSError:
        pass
    return found


def _count_files_with_suffix(dir_path: Path, suffix: str) -> int:
    """Count regular files ending with suffix in dir_path (non-recursive)."""
    if not dir_path.is_dir():
        return 0
    count = 0
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if (
                    entry.is_file(follow_symlinks=False)
                    and entry.name.endswith(suffix)
                    and not entry.name.startswith(".")
                    and ".part" not in entry.name
                ):
                    count += 1
    except OSError:
        return 0
    return count


def get_coverage_section(total_chambers: int = 502, test_chambers_total: int = 20) -> dict:
    """Compute re-segmentation coverage for C4, C5, p5_tracks, and figures/pilot/p5 movies."""
    try:
        c4_set = _scan_seg_chambers(C4_DIR)
        c5_set = _scan_seg_chambers(C5_DIR)
        redrawn_set = c4_set | c5_set

        c4_count = len(c4_set)
        c5_count = len(c5_set)
        redrawn_count = len(redrawn_set)

        tracks_count = _count_files_with_suffix(P5_TRACKS_DIR, ".csv")
        movies_count = _count_files_with_suffix(P5_MOVIES_DIR, ".mp4")

        denom = max(1, total_chambers)
        test_denom = max(1, test_chambers_total)
        redrawn_pct = round(redrawn_count / denom * 100.0, 1)

        return {
            "state": "ok",
            "c4_chambers": c4_count,
            "c5_chambers": c5_count,
            "redrawn_chambers": redrawn_count,
            "total_chambers": total_chambers,
            "redrawn_percent": redrawn_pct,
            "test_chambers_total": test_denom,
            "p5_c5_split": c5_count,
            "p5_tracks": tracks_count,
            "p5_movies": movies_count,
        }
    except Exception as exc:
        return {
            "state": "not available",
            "error": str(exc),
            "c4_chambers": 0,
            "c5_chambers": 0,
            "redrawn_chambers": 0,
            "total_chambers": total_chambers,
            "redrawn_percent": 0.0,
            "test_chambers_total": test_chambers_total,
            "p5_c5_split": 0,
            "p5_tracks": 0,
            "p5_movies": 0,
        }


def _list_ps_rows() -> tuple[bool, list[tuple[int, int, str, str]]]:
    """Return (ok, [(pid, ppid, etime, args), ...]) from ps."""
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "pid=,ppid=,etime=,args="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return False, []
        rows: list[tuple[int, int, str, str]] = []
        for line in proc.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split(None, 3)
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            etime = parts[2].strip()
            args = parts[3].strip()
            rows.append((pid, ppid, etime, args))
        return True, rows
    except Exception:
        return False, []


_PROMPT_FILE_RE = re.compile(r"prompt_([A-Za-z0-9_.-]+)\.txt")
_AGY_JSON_RE = re.compile(r"agy_([A-Za-z0-9_.-]+)\.json")
_REPORT_ARG_RE = re.compile(r"\.agy_([A-Za-z0-9_.-]+?)_report\.md")
_REASON_RE = re.compile(r'reason="((?:[^"\\]|\\.)*)"')
_LOG_PREFIX_RE = re.compile(r"^([IWEF](\d{2})(\d{2})\s+(\d{2}:\d{2}:\d{2})(?:\.\d+)?)\s+\d+\s+([^\]]+)\]\s*(.*)$")


def _tail_cli_log(log_path: Path = CLI_LOG_PATH, max_lines: int = 3) -> tuple[str, list[dict[str, str]]]:
    """Read the last ~3 lines from ~/.gemini/antigravity-cli/cli.log."""
    if not log_path.is_file():
        return "not available", []
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > LOG_TAIL_BYTES:
                f.seek(size - LOG_TAIL_BYTES)
                raw = f.read()
                nl = raw.find(b"\n")
                if nl != -1:
                    raw = raw[nl + 1 :]
            else:
                raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        all_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not all_lines:
            return "ok", []

        # Prefer concise monitor summary lines if present in the tail, else raw tail lines
        monitor_items: list[dict[str, str]] = []
        for line in all_lines:
            if "sir_flash_lite FINISHED" not in line:
                continue
            m_reason = _REASON_RE.search(line)
            if not m_reason:
                continue
            reason = m_reason.group(1).replace('\\"', '"').replace("\\\\", "\\")
            m_ts = _LOG_PREFIX_RE.match(line)
            ts = m_ts.group(4) if m_ts else ""
            monitor_items.append({"timestamp": ts, "text": reason})

        if monitor_items:
            return "ok", monitor_items[-max_lines:]

        fallback_items: list[dict[str, str]] = []
        for line in all_lines[-max_lines:]:
            m_ts = _LOG_PREFIX_RE.match(line)
            if m_ts:
                fallback_items.append({"timestamp": m_ts.group(4), "text": m_ts.group(6)[:200]})
            else:
                fallback_items.append({"timestamp": "", "text": line[:200]})
        return "ok", fallback_items
    except Exception:
        return "not available", []


def _scan_recent_agy_reports(repo_root: Path = REPO_ROOT, now_ts: float | None = None) -> list[dict]:
    """List .agy_*report*.md files modified in the last 24 hours."""
    if now_ts is None:
        now_ts = time.time()
    cutoff = now_ts - REPORT_MAX_AGE_S
    reports: list[tuple[float, dict]] = []
    try:
        for p in repo_root.glob(".agy_*report*.md"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                continue
            dt = datetime.datetime.fromtimestamp(st.st_mtime).astimezone()
            reports.append(
                (
                    st.st_mtime,
                    {
                        "name": p.name,
                        "mtime": dt.strftime("%Y-%m-%d %H:%M"),
                        "age_human": _human_age(now_ts - st.st_mtime),
                        "size_bytes": st.st_size,
                        "size_human": _human_bytes(st.st_size),
                    },
                )
            )
    except OSError:
        return []
    reports.sort(key=lambda item: item[0], reverse=True)
    return [r[1] for r in reports][:5]


def get_active_work_section() -> dict:
    """Collect running agy jobs, raw-image downloader (only if active), cli.log tail, and recent reports."""
    ok_ps, rows = _list_ps_rows()
    my_pid = os.getpid()

    # Index children by ppid to pair agy_run.sh wrappers with child agy processes
    children_by_ppid: dict[int, list[tuple[int, str, str]]] = collections.defaultdict(list)
    for pid, ppid, etime, args in rows:
        children_by_ppid[ppid].append((pid, etime, args))

    jobs: list[dict] = []
    wrapper_pids: set[int] = set()
    paired_agy_pids: set[int] = set()
    raw_dl_procs: list[dict] = []

    for pid, ppid, etime, args in rows:
        if pid == my_pid:
            continue
        tokens = args.split()
        if not tokens:
            continue
        first_base = os.path.basename(tokens[0])

        # Skip grep/ps inspection commands
        if first_base in ("grep", "egrep", "fgrep", "ps", "sed", "cat", "head", "tail"):
            continue

        # 1. Detect agy_run.sh wrapper processes
        is_wrapper = any( os.path.basename(tok) == "agy_run.sh" for tok in tokens[:3] )
        if is_wrapper:
            wrapper_pids.add(pid)
            m_prompt = _PROMPT_FILE_RE.search(args)
            if m_prompt:
                prompt_name = m_prompt.group(1)
                prompt_file = f"prompt_{prompt_name}.txt"
            else:
                m_json = _AGY_JSON_RE.search(args)
                prompt_name = m_json.group(1) if m_json else "agy_run"
                prompt_file = "agy_run.sh"

            child_agy_pid: int | None = None
            child_etime = etime
            for c_pid, c_etime, c_args in children_by_ppid.get(pid, []):
                c_tokens = c_args.split()
                if c_tokens and os.path.basename(c_tokens[0]) == "agy":
                    child_agy_pid = c_pid
                    child_etime = c_etime or etime
                    paired_agy_pids.add(c_pid)
                    break

            jobs.append(
                {
                    "prompt": prompt_name,
                    "prompt_file": prompt_file,
                    "wrapper_pid": pid,
                    "agy_pid": child_agy_pid,
                    "etime": child_etime,
                }
            )
            continue

        # 2. Detect active raw-image download processes (only if actually running now)
        if first_base.startswith("python") and ("download_raw.py" in args or "download_mips.py" in args):
            raw_dl_procs.append(
                {
                    "pid": pid,
                    "etime": etime,
                    "command": args[:160],
                }
            )

    # 3. Also include any standalone `agy -p` jobs not launched via agy_run.sh
    for pid, ppid, etime, args in rows:
        if pid == my_pid or pid in paired_agy_pids or ppid in wrapper_pids:
            continue
        tokens = args.split()
        if not tokens:
            continue
        if os.path.basename(tokens[0]) == "agy" and ("-p" in tokens or "--prompt" in tokens):
            m_rep = _REPORT_ARG_RE.search(args)
            prompt_name = m_rep.group(1) if m_rep else "agy -p"
            jobs.append(
                {
                    "prompt": prompt_name,
                    "prompt_file": "agy -p",
                    "wrapper_pid": None,
                    "agy_pid": pid,
                    "etime": etime,
                }
            )

    jobs.sort(key=lambda j: str(j.get("prompt", "")))
    log_state, log_lines = _tail_cli_log(CLI_LOG_PATH, max_lines=3)
    recent_reports = _scan_recent_agy_reports(REPO_ROOT)

    if not ok_ps:
        agy_status = "not available"
        sec_state = "not available"
    else:
        agy_status = "running" if len(jobs) > 0 else "idle"
        sec_state = "ok"

    return {
        "state": sec_state,
        "agy_status": agy_status,
        "job_count": len(jobs),
        "jobs": jobs,
        "log_state": log_state,
        "log_lines": log_lines,
        "recent_reports": recent_reports,
        "raw_download": {
            "running": len(raw_dl_procs) > 0,
            "process_count": len(raw_dl_procs),
            "processes": raw_dl_procs,
        },
    }


def get_git_section(repo_root: Path = REPO_ROOT) -> dict:
    """Return branch, last commit (hash+subject), uncommitted file count, and commits today."""
    if not (repo_root / ".git").exists():
        return {
            "state": "not available",
            "branch": "not available",
            "last_commit": {"hash": "—", "subject": "not available", "relative_time": ""},
            "uncommitted_count": 0,
            "commits_today": 0,
        }
    try:
        branch_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else "not available"

        status_proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if status_proc.returncode == 0:
            uncommitted_count = sum(1 for ln in status_proc.stdout.splitlines() if ln.strip())
        else:
            uncommitted_count = 0

        last_proc = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-n", "1", "--pretty=format:%h%x1f%s%x1f%cr"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        last_commit = {"hash": "—", "subject": "not available", "relative_time": ""}
        if last_proc.returncode == 0 and last_proc.stdout.strip():
            parts = last_proc.stdout.strip().split("\x1f")
            if len(parts) == 3:
                last_commit = {
                    "hash": parts[0],
                    "subject": parts[1],
                    "relative_time": parts[2],
                }

        today_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-list", "--count", "--since=midnight", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        commits_today = 0
        if today_proc.returncode == 0 and today_proc.stdout.strip().isdigit():
            commits_today = int(today_proc.stdout.strip())

        return {
            "state": "ok",
            "branch": branch,
            "last_commit": last_commit,
            "uncommitted_count": uncommitted_count,
            "commits_today": commits_today,
        }
    except Exception as exc:
        return {
            "state": "not available",
            "error": str(exc),
            "branch": "not available",
            "last_commit": {"hash": "—", "subject": "not available", "relative_time": ""},
            "uncommitted_count": 0,
            "commits_today": 0,
        }


def build_status_payload() -> dict:
    """Assemble full /api/status dictionary defensively."""
    now_dt = datetime.datetime.now().astimezone()
    plan_sec = get_plan_section()
    total_ch = int(plan_sec.get("total_chambers") or 502)
    test_ch = len(plan_sec.get("chambers") or []) or 20

    abstract_sec = get_abstract_section()
    coverage_sec = get_coverage_section(total_chambers=total_ch, test_chambers_total=test_ch)
    active_sec = get_active_work_section()
    notebook_sec = get_notebook_section()
    git_sec = get_git_section()

    return {
        "updated_iso": now_dt.isoformat(timespec="seconds"),
        "updated_human": now_dt.strftime("%H:%M:%S"),
        "plan": plan_sec,
        "abstract": abstract_sec,
        "coverage": coverage_sec,
        "active_work": active_sec,
        "notebook": notebook_sec,
        "git": git_sec,
    }


INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cell-Kill-Sim progress</title>
<style>
:root {
  --paper: #FFFFFF;
  --paper-2: #F6F6F6;
  --ink: #111111;
  --text: #1F1F1F;
  --muted: #5C5C5C;
  --rule: #E3E3E3;
  --line: #C9C9C9;
  --accent: #0B5FD1;
  --danger: #B3261E;

  /* Status & verdict colours (light) */
  --st-done-fg: #1B5E3A;
  --st-done-bg: #EBF5EE;
  --st-done-bd: #9AC8AD;

  --st-ok-fg: #1F5E4B;
  --st-ok-bg: #EEF6F3;
  --st-ok-bd: #A3C9BC;

  --st-prog-fg: #8A4B08;
  --st-prog-bg: #FDF3E7;
  --st-prog-bd: #E0B483;

  --st-track-fg: #1F4E79;
  --st-track-bg: #EDF3F9;
  --st-track-bd: #9BB8D6;

  --st-todo-fg: #525252;
  --st-todo-bg: #F2F2F2;
  --st-todo-bd: #CCCCCC;

  --st-block-fg: #9E1F18;
  --st-block-bg: #FCECEB;
  --st-block-bd: #E3A09C;

  --dot-run: #1B7F46;
  --dot-idle: #8C8C8C;

  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #121212;
    --paper-2: #1C1C1C;
    --ink: #F2F2F2;
    --text: #E6E6E6;
    --muted: #A8A8A8;
    --rule: #303030;
    --line: #4A4A4A;
    --accent: #7FB3FF;
    --danger: #FF8A7A;

    --st-done-fg: #8CE0B1;
    --st-done-bg: #142B1F;
    --st-done-bd: #2C5E43;

    --st-ok-fg: #84D4BC;
    --st-ok-bg: #152923;
    --st-ok-bd: #2C594B;

    --st-prog-fg: #F5B971;
    --st-prog-bg: #2E2010;
    --st-prog-bd: #6B4920;

    --st-track-fg: #8FBCE6;
    --st-track-bg: #152230;
    --st-track-bd: #315375;

    --st-todo-fg: #B8B8B8;
    --st-todo-bg: #222222;
    --st-todo-bd: #444444;

    --st-block-fg: #FF9488;
    --st-block-bg: #301614;
    --st-block-bd: #6E2D28;

    --dot-run: #4CD97B;
    --dot-idle: #707070;

    color-scheme: dark;
  }
}

*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
.num { font-variant-numeric: tabular-nums; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.muted { color: var(--muted); }
a {
  color: var(--accent);
  text-decoration: underline;
}

/* Header */
.top {
  border-bottom: 1px solid var(--rule);
  background: var(--paper);
}
.top-in {
  max-width: 860px;
  margin: 0 auto;
  padding: 14px 16px 12px;
}
.top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
h1 {
  font-size: 20px;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
  letter-spacing: -0.01em;
}
.hdr-meta {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.agy-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  color: var(--ink);
}
.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  display: inline-block;
  background: var(--dot-idle);
}
.status-dot.running {
  background: var(--dot-run);
}
.status-dot.idle {
  background: var(--dot-idle);
}
.stale-banner {
  display: none;
  color: var(--st-block-fg);
  background: var(--st-block-bg);
  border: 1px solid var(--st-block-bd);
  border-radius: 3px;
  padding: 1px 7px;
  font-size: 11px;
  font-weight: 600;
}
.stale-banner.visible { display: inline-block; }

.aim-line {
  margin: 6px 0 0;
  font-size: 14px;
  color: var(--text);
}
.now-banner {
  margin-top: 8px;
  padding: 7px 10px;
  background: var(--paper-2);
  border-left: 3px solid var(--ink);
  font-size: 13px;
  color: var(--ink);
}
.now-banner strong {
  font-weight: 700;
}

/* Main layout */
main {
  max-width: 860px;
  margin: 0 auto;
  padding: 16px 16px 32px;
}
section {
  margin-bottom: 24px;
}
.sec-hdr {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--ink);
  padding-bottom: 4px;
  margin-bottom: 10px;
}
h2 {
  font-size: 15px;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
}
.sec-sub {
  font-size: 12px;
  color: var(--muted);
}

/* Status pills & verdict chips */
.chip {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.25;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid var(--st-todo-bd);
  color: var(--st-todo-fg);
  background: var(--st-todo-bg);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.chip.done, .chip.good, .chip.reviewed, .chip.signed {
  color: var(--st-done-fg);
  background: var(--st-done-bg);
  border-color: var(--st-done-bd);
}
.chip.ok {
  color: var(--st-ok-fg);
  background: var(--st-ok-bg);
  border-color: var(--st-ok-bd);
}
.chip.in_progress, .chip.merge {
  color: var(--st-prog-fg);
  background: var(--st-prog-bg);
  border-color: var(--st-prog-bd);
}
.chip.track {
  color: var(--st-track-fg);
  background: var(--st-track-bg);
  border-color: var(--st-track-bd);
}
.chip.todo, .chip.unclear, .chip.draft {
  color: var(--st-todo-fg);
  background: var(--st-todo-bg);
  border-color: var(--st-todo-bd);
}
.chip.waiting {
  color: var(--muted);
  background: transparent;
  border: 1px solid var(--line);
}
.chip.blocked, .chip.poor {
  color: var(--st-block-fg);
  background: var(--st-block-bg);
  border-color: var(--st-block-bd);
}
.chip.parked {
  color: var(--muted);
  background: var(--paper-2);
  border: 1px dashed var(--line);
}

/* Where we are (Abstract) */
.abstract-box {
  font-size: 14px;
  line-height: 1.5;
}
.abstract-box .abs-heading {
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  margin: 0 0 6px;
}
.abstract-box p {
  margin: 0 0 8px;
}
.abstract-box ul, .abstract-box ol {
  margin: 0 0 8px;
  padding-left: 20px;
}
.abstract-box li {
  margin-bottom: 3px;
}

/* Stage tracker */
.plan-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.plan-item {
  padding: 7px 0;
  border-bottom: 1px solid var(--rule);
  display: grid;
  grid-template-columns: 52px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: start;
}
.plan-item:last-child { border-bottom: none; }
.step-id {
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
}
.step-title {
  font-weight: 600;
  color: var(--ink);
  font-size: 14px;
}
.step-note {
  font-size: 13px;
  color: var(--muted);
  margin-top: 2px;
}

/* Coverage & progress bars */
.cov-box {
  padding: 8px 0 10px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--rule);
}
.cov-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 4px;
  font-size: 13px;
}
.bar-lg {
  height: 8px;
  background: var(--paper-2);
  border: 1px solid var(--line);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
}
.bar-lg > i, .bar-sm > i {
  display: block;
  height: 100%;
  background: var(--ink);
  width: 0%;
}
.p5-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 6px;
}
.p5-item {
  font-size: 12px;
  color: var(--muted);
}
.bar-sm {
  height: 5px;
  background: var(--paper-2);
  border: 1px solid var(--line);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 3px;
}

/* Test chambers table */
.set-group {
  margin-top: 10px;
}
.set-hdr {
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin: 8px 0 4px;
}
table.ch-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.ch-table th {
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  padding: 4px 8px 4px 0;
  border-bottom: 1px solid var(--ink);
}
table.ch-table td {
  padding: 5px 8px 5px 0;
  border-bottom: 1px solid var(--rule);
  vertical-align: top;
}
table.ch-table tr:last-child td {
  border-bottom: none;
}

/* Highlight box (Needs PI) */
.pi-box {
  padding: 10px 12px;
  background: var(--paper-2);
  border: 1px solid var(--line);
  border-left: 3px solid var(--ink);
  border-radius: 3px;
}
.pi-list {
  margin: 0;
  padding-left: 18px;
}
.pi-list li {
  margin-bottom: 4px;
}
.pi-list li:last-child {
  margin-bottom: 0;
}

/* Active work & lists */
.sub-lbl {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
  margin: 10px 0 4px;
}
.kv-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.kv-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  padding: 5px 0;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
  flex-wrap: wrap;
}
.kv-item:last-child { border-bottom: none; }
.log-box {
  background: var(--paper-2);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 6px 10px;
  font-size: 12px;
}
.log-row {
  padding: 3px 0;
  border-bottom: 1px solid var(--rule);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px;
}
.log-row:last-child { border-bottom: none; }

/* Notebook entries */
.nb-item {
  padding: 6px 0;
  border-bottom: 1px solid var(--rule);
  display: grid;
  grid-template-columns: 86px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: baseline;
  font-size: 13px;
}
.nb-item:last-child { border-bottom: none; }
.nb-link {
  color: var(--ink);
  font-weight: 600;
  text-decoration: underline;
  text-decoration-color: var(--line);
}

/* Footer */
footer.foot {
  max-width: 860px;
  margin: 0 auto;
  padding: 14px 16px 40px;
  border-top: 1px solid var(--ink);
  font-size: 12px;
  color: var(--muted);
}
.foot-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}

@media (max-width: 560px) {
  .p5-grid { grid-template-columns: 1fr; gap: 6px; }
  .nb-item { grid-template-columns: 1fr; gap: 2px; }
  .plan-item { grid-template-columns: 44px minmax(0, 1fr) auto; }
}
</style>
</head>
<body>
<header class="top">
  <div class="top-in">
    <div class="top-row">
      <h1>Cell-Kill-Sim progress</h1>
      <div class="hdr-meta num">
        <span id="stale-warning" class="stale-banner">Stale: fetch failed</span>
        <span class="agy-badge">
          <span id="hdr-agy-dot" class="status-dot idle"></span>
          <span id="hdr-agy-text">agy: —</span>
        </span>
        <span id="hdr-plan-updated">plan: —</span>
        <span id="hdr-page-updated">refreshed: —</span>
      </div>
    </div>
    <p id="hdr-aim" class="aim-line">—</p>
    <div class="now-banner"><strong>Now:</strong> <span id="hdr-now">—</span></div>
  </div>
</header>

<main>
  <section id="sec-abstract">
    <div class="sec-hdr">
      <h2>Where we are</h2>
      <span class="sec-sub mono">lab/notebook/ABSTRACT.md</span>
    </div>
    <div id="abstract-body" class="abstract-box">Loading…</div>
  </section>

  <section id="sec-stages">
    <div class="sec-hdr">
      <h2>Stage tracker</h2>
      <span id="stages-meta" class="sec-sub num"></span>
    </div>
    <div id="stages-body">Loading…</div>
  </section>

  <section id="sec-chambers">
    <div class="sec-hdr">
      <h2>Test chambers (20)</h2>
      <span id="chambers-meta" class="sec-sub num"></span>
    </div>
    <div id="coverage-body" class="cov-box"></div>
    <div id="chambers-body">Loading…</div>
  </section>

  <section id="sec-needs-pi">
    <div class="sec-hdr">
      <h2>Needs the PI / Fei lab</h2>
      <span class="sec-sub">open questions</span>
    </div>
    <div id="needs-pi-body" class="pi-box">Loading…</div>
  </section>

  <section id="sec-active">
    <div class="sec-hdr">
      <h2>Active work</h2>
      <span id="active-meta" class="sec-sub num"></span>
    </div>
    <div id="active-body">Loading…</div>
  </section>

  <section id="sec-notebook">
    <div class="sec-hdr">
      <h2>Recent lab notebook entries</h2>
      <span class="sec-sub">latest 6 by filename</span>
    </div>
    <div id="notebook-body">Loading…</div>
  </section>
</main>

<footer class="foot">
  <div id="foot-git" class="foot-row num">Loading git info…</div>
  <div id="foot-parked" class="foot-row"></div>
  <div class="foot-row">
    <span>Links: <a href="http://localhost:8767">Lab site (http://localhost:8767)</a> · <a href="http://localhost:8766">Review app (http://localhost:8766)</a></span>
    <span class="mono">/api/status (5s auto-refresh)</span>
  </div>
</footer>

<script id="initial-status" type="application/json">__INITIAL_STATUS_JSON__</script>
<script>
(function() {
  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  function chipHtml(kind, labelOverride) {
    const raw = String(kind || 'todo').toLowerCase().trim();
    const cls = raw.replace(/[^a-z0-9_]/g, '');
    const label = labelOverride != null ? labelOverride : raw.replace(/_/g, ' ');
    return '<span class="chip ' + cls + '">' + esc(label) + '</span>';
  }

  function renderHeader(data) {
    const plan = data.plan || {};
    const active = data.active_work || {};
    const agyStatus = active.agy_status || 'idle';

    const dotEl = document.getElementById('hdr-agy-dot');
    const txtEl = document.getElementById('hdr-agy-text');
    dotEl.className = 'status-dot ' + (agyStatus === 'running' ? 'running' : 'idle');
    txtEl.textContent = 'agy ' + agyStatus;

    document.getElementById('hdr-plan-updated').textContent = 'plan: ' + (plan.updated || 'not available');
    document.getElementById('hdr-page-updated').textContent = 'refreshed: ' + (data.updated_human || '—');
    document.getElementById('hdr-aim').textContent = plan.aim || 'not available';
    document.getElementById('hdr-now').textContent = plan.now || 'not available';
  }

  function renderAbstract(abs) {
    const el = document.getElementById('abstract-body');
    if (!abs || abs.state !== 'ok' || !abs.html) {
      el.innerHTML = '<p class="muted">not available</p>';
      return;
    }
    el.innerHTML = abs.html;
  }

  function renderStages(plan) {
    const metaEl = document.getElementById('stages-meta');
    const bodyEl = document.getElementById('stages-body');
    if (!plan || plan.state !== 'ok' || !Array.isArray(plan.phases) || plan.phases.length === 0) {
      metaEl.textContent = 'not available';
      bodyEl.innerHTML = '<div class="muted">not available</div>';
      return;
    }
    metaEl.textContent = plan.stages_summary || '';
    let html = '<ul class="plan-list">';
    for (const ph of plan.phases) {
      html += '<li class="plan-item">';
      html += '<div class="step-id mono num">' + esc(ph.id) + '</div>';
      html += '<div><div class="step-title">' + esc(ph.title) + '</div>';
      if (ph.note) {
        html += '<div class="step-note">' + esc(ph.note) + '</div>';
      }
      html += '</div>';
      html += '<div>' + chipHtml(ph.status) + '</div>';
      html += '</li>';
    }
    html += '</ul>';
    bodyEl.innerHTML = html;
  }

  function renderCoverageAndChambers(cov, plan) {
    const covEl = document.getElementById('coverage-body');
    const metaEl = document.getElementById('chambers-meta');
    const bodyEl = document.getElementById('chambers-body');

    if (!cov || cov.state !== 'ok') {
      covEl.innerHTML = '<div class="muted">Coverage not available</div>';
    } else {
      const redrawn = Number(cov.redrawn_chambers || 0);
      const total = Number(cov.total_chambers || 502);
      const pct = Math.max(0, Math.min(100, Number(cov.redrawn_percent || 0)));
      const testTot = Number(cov.test_chambers_total || 20);
      const c5Split = Number(cov.p5_c5_split || 0);
      const tracks = Number(cov.p5_tracks || 0);
      const movies = Number(cov.p5_movies || 0);

      const c5Pct = Math.min(100, Math.round((c5Split / Math.max(1, testTot)) * 100));
      const trPct = Math.min(100, Math.round((tracks / Math.max(1, testTot)) * 100));
      const mvPct = Math.min(100, Math.round((movies / Math.max(1, testTot)) * 100));

      let cHtml = '';
      cHtml += '<div class="cov-row num"><strong>chambers redrawn: ' + redrawn + ' of ' + total + '</strong>';
      cHtml += '<span class="muted">C4: ' + Number(cov.c4_chambers || 0) + ' · C5: ' + c5Split + ' (' + pct.toFixed(1) + '%)</span></div>';
      cHtml += '<div class="bar-lg"><i style="width:' + pct.toFixed(2) + '%"></i></div>';
      cHtml += '<div class="cov-row num"><span>P5 outputs: C5 split ' + c5Split + '/' + testTot + ' chambers, tracks ' + tracks + '/' + testTot + ', movies <code class="mono">figures/pilot/p5/*.mp4</code> ' + movies + '/' + testTot + '</span></div>';
      cHtml += '<div class="p5-grid num">';
      cHtml += '<div class="p5-item">C5 split (' + c5Split + '/' + testTot + ')<div class="bar-sm"><i style="width:' + c5Pct + '%"></i></div></div>';
      cHtml += '<div class="p5-item">tracks (' + tracks + '/' + testTot + ')<div class="bar-sm"><i style="width:' + trPct + '%"></i></div></div>';
      cHtml += '<div class="p5-item">movies (' + movies + '/' + testTot + ')<div class="bar-sm"><i style="width:' + mvPct + '%"></i></div></div>';
      cHtml += '</div>';
      covEl.innerHTML = cHtml;
    }

    if (!plan || plan.state !== 'ok' || !Array.isArray(plan.chambers) || plan.chambers.length === 0) {
      metaEl.textContent = 'not available';
      bodyEl.innerHTML = '<div class="muted">not available</div>';
      return;
    }

    metaEl.textContent = plan.verdict_tally || '';

    const groups = new Map();
    for (const ch of plan.chambers) {
      const s = ch.set || 'other';
      if (!groups.has(s)) groups.set(s, []);
      groups.get(s).push(ch);
    }

    let html = '';
    for (const [setName, list] of groups.entries()) {
      html += '<div class="set-group">';
      html += '<div class="set-hdr num">' + esc(setName) + ' (' + list.length + ')</div>';
      html += '<table class="ch-table"><thead><tr><th style="width:160px">Chamber</th><th style="width:86px">Verdict</th><th>Issue</th></tr></thead><tbody>';
      for (const ch of list) {
        html += '<tr>';
        html += '<td class="mono num">' + esc(ch.id) + '</td>';
        html += '<td>' + chipHtml(ch.verdict) + '</td>';
        html += '<td>' + (ch.issue ? esc(ch.issue) : '<span class="muted">—</span>') + '</td>';
        html += '</tr>';
      }
      html += '</tbody></table></div>';
    }
    bodyEl.innerHTML = html;
  }

  function renderNeedsPi(plan) {
    const el = document.getElementById('needs-pi-body');
    if (!plan || plan.state !== 'ok' || !Array.isArray(plan.needs_pi)) {
      el.innerHTML = '<span class="muted">not available</span>';
      return;
    }
    if (plan.needs_pi.length === 0) {
      el.innerHTML = '<span class="muted">None currently</span>';
      return;
    }
    let html = '<ul class="pi-list">';
    for (const item of plan.needs_pi) {
      html += '<li>' + esc(item) + '</li>';
    }
    html += '</ul>';
    el.innerHTML = html;
  }

  function renderActiveWork(active) {
    const metaEl = document.getElementById('active-meta');
    const bodyEl = document.getElementById('active-body');
    if (!active || active.state === 'not available') {
      metaEl.textContent = 'not available';
      bodyEl.innerHTML = '<div class="muted">not available</div>';
      return;
    }

    const st = active.agy_status || 'idle';
    const jobs = Array.isArray(active.jobs) ? active.jobs : [];
    metaEl.textContent = 'state: ' + st + ' (' + jobs.length + ' job' + (jobs.length === 1 ? '' : 's') + ')';

    let html = '';

    // Show raw-image download ONLY if actually running now
    if (active.raw_download && active.raw_download.running && Array.isArray(active.raw_download.processes)) {
      html += '<div class="sub-lbl">Raw-image download (active)</div>';
      html += '<ul class="kv-list">';
      for (const p of active.raw_download.processes) {
        html += '<li class="kv-item num"><span class="mono">' + esc(p.command) + '</span>';
        html += '<span class="mono muted">PID ' + esc(p.pid) + ' · elapsed ' + esc(p.etime) + '</span></li>';
      }
      html += '</ul>';
    }

    // Running agy jobs
    html += '<div class="sub-lbl" style="margin-top:2px">Running agy jobs</div>';
    if (jobs.length === 0) {
      html += '<div class="muted" style="font-size:13px;padding:2px 0 6px">idle (no active agy_run.sh / agy jobs)</div>';
    } else {
      html += '<ul class="kv-list">';
      for (const j of jobs) {
        const pidInfo = j.agy_pid ? ('wrapper PID ' + j.wrapper_pid + ' → agy PID ' + j.agy_pid) : ('PID ' + (j.wrapper_pid || '—'));
        html += '<li class="kv-item num">';
        html += '<span><strong>' + esc(j.prompt) + '</strong> <span class="mono muted">(' + esc(j.prompt_file) + ')</span></span>';
        html += '<span class="mono">elapsed ' + esc(j.etime) + ' <span class="muted">· ' + esc(pidInfo) + '</span></span>';
        html += '</li>';
      }
      html += '</ul>';
    }

    // Shared agent log (~3 lines of cli.log)
    html += '<div class="sub-lbl">Agent log (~last 3 lines of cli.log)</div>';
    if (active.log_state === 'not available') {
      html += '<div class="log-box muted">not available</div>';
    } else if (!Array.isArray(active.log_lines) || active.log_lines.length === 0) {
      html += '<div class="log-box muted">No recent log lines</div>';
    } else {
      html += '<div class="log-box">';
      for (const ln of active.log_lines) {
        html += '<div class="log-row">';
        html += '<span class="mono num muted">' + esc(ln.timestamp || '—') + '</span>';
        html += '<span>' + esc(ln.text) + '</span>';
        html += '</div>';
      }
      html += '</div>';
    }

    // Recent agy reports (last 24 h)
    const reports = Array.isArray(active.recent_reports) ? active.recent_reports : [];
    html += '<div class="sub-lbl">Recent agy reports (last 24 h: ' + reports.length + ')</div>';
    if (reports.length === 0) {
      html += '<div class="muted" style="font-size:13px">None modified in the last 24 h</div>';
    } else {
      html += '<ul class="kv-list">';
      for (const r of reports) {
        html += '<li class="kv-item num">';
        html += '<span class="mono">' + esc(r.name) + '</span>';
        html += '<span class="muted">' + esc(r.mtime) + ' (' + esc(r.age_human) + ', ' + esc(r.size_human) + ')</span>';
        html += '</li>';
      }
      html += '</ul>';
    }

    bodyEl.innerHTML = html;
  }

  function renderNotebook(nb) {
    const bodyEl = document.getElementById('notebook-body');
    if (!nb || nb.state !== 'ok' || !Array.isArray(nb.entries)) {
      bodyEl.innerHTML = '<div class="muted">not available</div>';
      return;
    }
    if (nb.entries.length === 0) {
      bodyEl.innerHTML = '<div class="muted">No notebook entries found</div>';
      return;
    }
    let html = '<div>';
    for (const e of nb.entries) {
      html += '<div class="nb-item">';
      html += '<span class="mono num muted">' + esc(e.date) + '</span>';
      html += '<div><span class="nb-link">' + esc(e.title) + '</span> <span class="mono muted">(' + esc(e.filename) + ')</span></div>';
      html += '<div class="num">' + chipHtml('waiting', 'gate ' + (e.gate || 'none')) + ' ' + chipHtml(e.status || 'draft') + '</div>';
      html += '</div>';
    }
    html += '</div>';
    bodyEl.innerHTML = html;
  }

  function renderFooter(git, plan) {
    const gitEl = document.getElementById('foot-git');
    const parkedEl = document.getElementById('foot-parked');

    if (!git || git.state !== 'ok') {
      gitEl.textContent = 'Git: not available';
    } else {
      const lc = git.last_commit || {};
      const commitStr = lc.hash ? (lc.hash + ' ' + (lc.subject || '')) : 'not available';
      gitEl.innerHTML =
        '<span>Branch: <strong class="mono">' + esc(git.branch) + '</strong> · Last commit: <span class="mono">' +
        esc(commitStr) + '</span></span>' +
        '<span>Uncommitted files: <strong>' + Number(git.uncommitted_count || 0) +
        '</strong> · Commits today: <strong>' + Number(git.commits_today || 0) + '</strong></span>';
    }

    const parked = (plan && plan.parked_note) ? plan.parked_note : '';
    parkedEl.textContent = parked ? ('Parked: ' + parked) : '';
  }

  function applyStatus(data) {
    if (!data) return;
    renderHeader(data);
    renderAbstract(data.abstract);
    renderStages(data.plan);
    renderCoverageAndChambers(data.coverage, data.plan);
    renderNeedsPi(data.plan);
    renderActiveWork(data.active_work);
    renderNotebook(data.notebook);
    renderFooter(data.git, data.plan);
  }

  try {
    const initEl = document.getElementById('initial-status');
    if (initEl && initEl.textContent) {
      applyStatus(JSON.parse(initEl.textContent));
    }
  } catch (_) {}

  async function refresh() {
    const staleEl = document.getElementById('stale-warning');
    try {
      const resp = await fetch('/api/status', { cache: 'no-store' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      staleEl.classList.remove('visible');
      applyStatus(data);
    } catch (err) {
      staleEl.classList.add('visible');
    }
  }

  setInterval(refresh, 5000);
})();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        # Quiet default stderr logging
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/" or route == "/index.html":
            payload = build_status_payload()
            init_json = json.dumps(payload).replace("</", "<\\/")
            page_html = INDEX_HTML_TEMPLATE.replace("__INITIAL_STATUS_JSON__", init_json)
            body = page_html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if route == "/api/status":
            payload = build_status_payload()
            raw = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return

        msg = b"Not Found"
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell-Kill-Sim local live progress dashboard.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving Cell-Kill-Sim progress dashboard on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
