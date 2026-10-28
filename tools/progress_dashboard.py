#!/usr/bin/env python3
"""Local live progress dashboard for Cell-Kill-Sim.

Uses Python standard library only (http.server, json, os, subprocess, threading).
Run:
    python3 tools/progress_dashboard.py [--port 8765]
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

# Import portable paths from src/paths.py
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import DATA_DIR, REPO_ROOT  # noqa: E402

PLAN_PATH: Path = REPO_ROOT / "status" / "plan.json"
URLS_PATH: Path = DATA_DIR / "scidb_19235_all_urls.txt"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
MANIFEST_PATH: Path = MASKS_DIR / "manifest.csv"
CLI_LOG_PATH: Path = Path.home() / ".gemini" / "antigravity-cli" / "cli.log"

SCAN_CACHE_TTL_S: float = 5.0
RATE_WINDOW_S: float = 120.0
LOG_TAIL_BYTES: int = 2 * 1024 * 1024  # ~2 MB


def _human_bytes(num_bytes: int | float) -> str:
    val = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(val)} B"
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TB"


def _human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    sec = int(round(seconds))
    if sec == 0:
        return "0s"
    hours, rem = divmod(sec, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def compute_expected_masks(urls_path: Path = URLS_PATH) -> tuple[bool, dict[str, int], int]:
    """Compute expected _predictions.h5 counts per experiment from scidb_19235_all_urls.txt."""
    if not urls_path.is_file():
        return False, {}, 0
    counts: dict[str, int] = collections.Counter()
    try:
        with open(urls_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "_predictions.h5" not in line:
                    continue
                idx = line.find("/V1/")
                if idx == -1:
                    continue
                rest = line[idx + 4 :]
                slash = rest.find("/")
                if slash <= 0:
                    continue
                exp = rest[:slash].strip()
                if exp:
                    counts[exp] += 1
    except OSError:
        return False, {}, 0

    ordered = dict(sorted(counts.items(), key=lambda kv: kv[0]))
    total = sum(ordered.values())
    return True, ordered, total


class MaskProgressTracker:
    """Thread-safe 5-second cached scanner for mask files and rolling 2-minute rate tracker."""

    def __init__(self, urls_path: Path = URLS_PATH, masks_dir: Path = MASKS_DIR) -> None:
        self.urls_path = urls_path
        self.masks_dir = masks_dir
        self._lock = threading.Lock()
        self._urls_loaded = False
        self._expected_by_exp: dict[str, int] = {}
        self._expected_total: int = 0

        self._last_scan_ts: float = 0.0
        self._cached_result: dict | None = None
        # Rolling history of (timestamp, present_total) over the last ~2 minutes
        self._history: collections.deque[tuple[float, int]] = collections.deque()
        self._seeded_from_mtime: bool = False

        self._ensure_expected_loaded()

    def _ensure_expected_loaded(self) -> None:
        if not self._urls_loaded:
            ok, by_exp, total = compute_expected_masks(self.urls_path)
            if ok:
                self._urls_loaded = True
                self._expected_by_exp = by_exp
                self._expected_total = total

    @staticmethod
    def _scan_experiment_dir(exp_dir: Path, now_ts: float) -> tuple[int, int, int]:
        """Recursively scan an experiment directory with os.scandir.

        Returns (file_count, total_bytes, recent_120s_count).
        Ignores .part files.
        """
        if not exp_dir.is_dir():
            return 0, 0, 0
        count = 0
        total_bytes = 0
        recent_count = 0
        cutoff = now_ts - RATE_WINDOW_S
        stack: list[Path] = [exp_dir]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        name = entry.name
                        if name.endswith(".part") or ".part." in name:
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False) and name.endswith("_predictions.h5"):
                                st = entry.stat(follow_symlinks=False)
                                count += 1
                                total_bytes += st.st_size
                                if st.st_mtime >= cutoff:
                                    recent_count += 1
                        except OSError:
                            continue
            except OSError:
                continue
        return count, total_bytes, recent_count

    @staticmethod
    def _read_manifest_failures(manifest_path: Path) -> tuple[bool, int, int, int]:
        """Read manifest.csv if present.

        Returns (manifest_exists, failures_count, not_downloaded_count, raw_ok_false_count).
        Note: src/download_masks.py pre-populates unattempted rows with ok=False and
        error='not_downloaded'. Actual download/validation failures have ok=False and
        error != 'not_downloaded'.
        """
        if not manifest_path.is_file():
            return False, 0, 0, 0
        failures = 0
        not_downloaded = 0
        raw_false = 0
        try:
            with open(manifest_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ok_val = str(row.get("ok", "")).strip().lower()
                    if ok_val in ("false", "0", "no"):
                        raw_false += 1
                        err_val = str(row.get("error", "")).strip()
                        if err_val == "not_downloaded":
                            not_downloaded += 1
                        else:
                            failures += 1
        except Exception:
            return False, 0, 0, 0
        return True, failures, not_downloaded, raw_false

    def get_status(self) -> dict:
        with self._lock:
            self._ensure_expected_loaded()
            now = time.time()
            if self._cached_result is not None and (now - self._last_scan_ts) < SCAN_CACHE_TTL_S:
                return self._cached_result

            if not self._urls_loaded:
                res = {
                    "state": "not started",
                    "message": f"Missing URL list at {self.urls_path.name}",
                    "expected_total": 0,
                    "present_total": 0,
                    "percent": 0.0,
                    "bytes_total": 0,
                    "bytes_human": "0 B",
                    "failures": 0,
                    "rate_files_per_min": 0.0,
                    "eta_seconds": None,
                    "eta_human": "—",
                    "experiments": [],
                }
                self._last_scan_ts = now
                self._cached_result = res
                return res

            masks_root_exists = self.masks_dir.is_dir()
            exp_rows: list[dict] = []
            present_total = 0
            bytes_total = 0
            recent_120s_total = 0

            for exp, expected_cnt in self._expected_by_exp.items():
                exp_dir = self.masks_dir / exp
                c, b, r = self._scan_experiment_dir(exp_dir, now) if masks_root_exists else (0, 0, 0)
                present_total += c
                bytes_total += b
                recent_120s_total += r
                pct = (c / expected_cnt * 100.0) if expected_cnt > 0 else 0.0
                exp_rows.append(
                    {
                        "experiment": exp,
                        "expected": expected_cnt,
                        "present": c,
                        "percent": round(pct, 2),
                        "bytes": b,
                        "bytes_human": _human_bytes(b),
                    }
                )

            # Seed history on first scan using file mtime counts over the last 2 minutes
            # so rate/ETA is immediately informative before 2 scan intervals elapse.
            if not self._seeded_from_mtime:
                self._seeded_from_mtime = True
                if recent_120s_total > 0:
                    self._history.append((now - RATE_WINDOW_S, max(0, present_total - recent_120s_total)))

            self._history.append((now, present_total))
            while len(self._history) > 2 and (now - self._history[0][0]) > RATE_WINDOW_S:
                self._history.popleft()

            rate_per_min = 0.0
            if len(self._history) >= 2:
                t0, n0 = self._history[0]
                dt = now - t0
                dn = present_total - n0
                if dt >= 1.0 and dn >= 0:
                    rate_per_min = (dn / dt) * 60.0

            remaining = max(0, self._expected_total - present_total)
            if remaining == 0 and self._expected_total > 0:
                eta_seconds: float | None = 0.0
                eta_human = "0s"
            elif rate_per_min > 0.0:
                eta_seconds = (remaining / rate_per_min) * 60.0
                eta_human = _human_duration(eta_seconds)
            else:
                eta_seconds = None
                eta_human = "—"

            manifest_exists, failures, not_downloaded, raw_false = self._read_manifest_failures(
                MANIFEST_PATH
            )

            overall_pct = (
                (present_total / self._expected_total * 100.0) if self._expected_total > 0 else 0.0
            )
            state = "ok" if (masks_root_exists or present_total > 0 or manifest_exists) else "not started"

            res = {
                "state": state,
                "expected_total": self._expected_total,
                "present_total": present_total,
                "percent": round(overall_pct, 2),
                "bytes_total": bytes_total,
                "bytes_human": _human_bytes(bytes_total),
                "manifest_present": manifest_exists,
                "failures": failures,
                "manifest_not_downloaded": not_downloaded,
                "manifest_ok_false_raw": raw_false,
                "rate_files_per_min": round(rate_per_min, 1),
                "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                "eta_human": eta_human,
                "experiments": exp_rows,
            }
            self._last_scan_ts = now
            self._cached_result = res
            return res


def get_plan_section(plan_path: Path = PLAN_PATH) -> dict:
    """Read status/plan.json safely."""
    if not plan_path.is_file():
        return {
            "state": "not started",
            "updated": None,
            "review_url": None,
            "steps": [],
        }
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        return {
            "state": "ok",
            "updated": data.get("updated"),
            "review_url": data.get("review_url"),
            "steps": steps,
        }
    except Exception as exc:
        return {
            "state": "not started",
            "error": str(exc),
            "updated": None,
            "review_url": None,
            "steps": [],
        }


def _list_processes() -> list[tuple[int, str, str]]:
    """Return list of (pid, comm, args) from ps."""
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return []
        rows: list[tuple[int, str, str]] = []
        for line in proc.stdout.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            comm = parts[1]
            args = parts[2] if len(parts) > 2 else comm
            rows.append((pid, comm, args))
        return rows
    except Exception:
        return []


_REASON_RE = re.compile(r'reason="((?:[^"\\]|\\.)*)"')
_TRIG_RE = re.compile(r"triggered=(true|false)")
_LOG_PREFIX_RE = re.compile(r"^([IWEF](\d{2})(\d{2})\s+(\d{2}:\d{2}:\d{2})(?:\.\d+)?)")


def _tail_cli_log(log_path: Path = CLI_LOG_PATH, max_lines: int = 12) -> tuple[bool, list[dict]]:
    """Read last ~2 MB of ~/.gemini/antigravity-cli/cli.log for 'sir_flash_lite FINISHED'."""
    if not log_path.is_file():
        return False, []
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > LOG_TAIL_BYTES:
                f.seek(size - LOG_TAIL_BYTES)
                raw = f.read()
                # Drop partial first line after seek
                nl = raw.find(b"\n")
                if nl != -1:
                    raw = raw[nl + 1 :]
            else:
                raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        matched: list[dict] = []
        for line in text.splitlines():
            if "sir_flash_lite FINISHED" not in line:
                continue
            reason_m = _REASON_RE.search(line)
            reason = reason_m.group(1).replace('\\"', '"') if reason_m else line.strip()
            trig_m = _TRIG_RE.search(line)
            triggered = (trig_m.group(1) == "true") if trig_m else False
            prefix_m = _LOG_PREFIX_RE.match(line)
            if prefix_m:
                mm, dd, hhmmss = prefix_m.group(2), prefix_m.group(3), prefix_m.group(4)
                timestamp = f"{mm}-{dd} {hhmmss}"
                raw_ts = prefix_m.group(1)
            else:
                timestamp = ""
                raw_ts = ""
            matched.append(
                {
                    "timestamp": timestamp,
                    "raw_timestamp": raw_ts,
                    "reason": reason,
                    "triggered": triggered,
                }
            )
        return True, matched[-max_lines:]
    except Exception:
        return False, []


def get_agent_And_downloader_sections() -> tuple[dict, dict]:
    """Inspect ps for active agy -p and download_masks.py processes, plus cli.log tail."""
    procs = _list_processes()
    my_pid = os.getpid()

    agy_procs: list[dict] = []
    dl_procs: list[dict] = []

    for pid, comm, args in procs:
        if pid == my_pid:
            continue
        base_comm = os.path.basename(comm)
        # Check for processes named agy with -p
        if base_comm == "agy":
            tokens = args.split()
            if "-p" in tokens or "--prompt" in tokens:
                agy_procs.append({"pid": pid, "command": args[:160]})

        # Check for mask or raw downloader processes (ignore shell wrappers / grep / caffeinate)
        if base_comm not in ("zsh", "sh", "bash", "grep", "ps", "agy", "caffeinate"):
            if "download_masks.py" in args or "download_raw.py" in args:
                dl_procs.append({"pid": pid, "command": args})

    log_exists, log_lines = _tail_cli_log(CLI_LOG_PATH, max_lines=12)
    agent_state = "ok" if (log_exists or agy_procs) else "not started"

    agent_sec = {
        "state": agent_state,
        "active": len(agy_procs) > 0,
        "process_count": len(agy_procs),
        "processes": agy_procs,
        "log_present": log_exists,
        "log_lines": log_lines,
    }
    downloader_sec = {
        "state": "ok" if dl_procs else "not started",
        "running": len(dl_procs) > 0,
        "process_count": len(dl_procs),
        "processes": dl_procs,
    }
    return agent_sec, downloader_sec


def get_git_section(repo_root: Path = REPO_ROOT) -> dict:
    """Return current branch, last 6 commits, and count of uncommitted files."""
    if not (repo_root / ".git").exists():
        return {
            "state": "not started",
            "branch": None,
            "uncommitted_count": 0,
            "commits": [],
        }
    try:
        branch_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else "unknown"

        status_proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if status_proc.returncode == 0:
            uncommitted_lines = [ln for ln in status_proc.stdout.splitlines() if ln.strip()]
            uncommitted_count = len(uncommitted_lines)
        else:
            uncommitted_count = 0

        log_proc = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-n", "6", "--pretty=format:%h%x1f%s%x1f%cr"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        commits: list[dict] = []
        if log_proc.returncode == 0:
            for line in log_proc.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.split("\x1f")
                if len(parts) == 3:
                    commits.append(
                        {
                            "hash": parts[0],
                            "subject": parts[1],
                            "relative_time": parts[2],
                        }
                    )
        return {
            "state": "ok",
            "branch": branch,
            "uncommitted_count": uncommitted_count,
            "commits": commits,
        }
    except Exception as exc:
        return {
            "state": "not started",
            "error": str(exc),
            "branch": None,
            "uncommitted_count": 0,
            "commits": [],
        }


def get_disk_section(data_dir: Path = DATA_DIR) -> dict:
    """Return free space on the volume holding data/."""
    target = data_dir if data_dir.exists() else REPO_ROOT
    try:
        usage = shutil.disk_usage(target)
        free_pct = (usage.free / usage.total * 100.0) if usage.total > 0 else 0.0
        return {
            "state": "ok",
            "path": str(target),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "total_human": _human_bytes(usage.total),
            "used_human": _human_bytes(usage.used),
            "free_human": _human_bytes(usage.free),
            "free_percent": round(free_pct, 1),
        }
    except Exception as exc:
        return {
            "state": "not started",
            "error": str(exc),
            "path": str(target),
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "total_human": "—",
            "used_human": "—",
            "free_human": "—",
            "free_percent": 0.0,
        }


INDEX_HTML = """<!DOCTYPE html>
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

  /* Status colors: done green, in_progress amber, todo grey, blocked red */
  --st-done-fg: #1B5E3A;
  --st-done-bg: #EBF5EE;
  --st-done-bd: #9AC8AD;

  --st-prog-fg: #8A4B08;
  --st-prog-bg: #FDF3E7;
  --st-prog-bd: #E0B483;

  --st-todo-fg: #525252;
  --st-todo-bg: #F2F2F2;
  --st-todo-bd: #CCCCCC;

  --st-block-fg: #9E1F18;
  --st-block-bg: #FCECEB;
  --st-block-bd: #E3A09C;

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

    --st-prog-fg: #F5B971;
    --st-prog-bg: #2E2010;
    --st-prog-bd: #6B4920;

    --st-todo-fg: #B8B8B8;
    --st-todo-bg: #222222;
    --st-todo-bd: #444444;

    --st-block-fg: #FF9488;
    --st-block-bg: #301614;
    --st-block-bd: #6E2D28;

    color-scheme: dark;
  }
}

*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.45;
}
.num {
  font-variant-numeric: tabular-nums;
}
a {
  color: var(--accent);
  text-decoration: underline;
}
.top {
  border-bottom: 1px solid var(--rule);
  background: var(--paper);
  position: sticky;
  top: 0;
  z-index: 10;
}
.top-in {
  max-width: 820px;
  margin: 0 auto;
  padding: 12px 16px;
  display: flex;
  align-items: baseline;
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
  font-size: 13px;
  color: var(--muted);
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.stale-banner {
  display: none;
  color: var(--st-block-fg);
  background: var(--st-block-bg);
  border: 1px solid var(--st-block-bd);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;
}
.stale-banner.visible {
  display: inline-block;
}
main {
  max-width: 820px;
  margin: 0 auto;
  padding: 16px 16px 64px;
}
section {
  margin-bottom: 28px;
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
  text-transform: none;
}
.sec-sub {
  font-size: 13px;
  color: var(--muted);
}

/* Status chip */
.chip {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.2;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid var(--st-todo-bd);
  color: var(--st-todo-fg);
  background: var(--st-todo-bg);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.chip.done {
  color: var(--st-done-fg);
  background: var(--st-done-bg);
  border-color: var(--st-done-bd);
}
.chip.in_progress {
  color: var(--st-prog-fg);
  background: var(--st-prog-bg);
  border-color: var(--st-prog-bd);
}
.chip.todo {
  color: var(--st-todo-fg);
  background: var(--st-todo-bg);
  border-color: var(--st-todo-bd);
}
.chip.blocked {
  color: var(--st-block-fg);
  background: var(--st-block-bg);
  border-color: var(--st-block-bd);
}

/* Plan list */
.plan-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.plan-item {
  padding: 8px 0;
  border-bottom: 1px solid var(--rule);
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: start;
}
.plan-item:last-child {
  border-bottom: none;
}
.plan-item.gate {
  margin: 6px 0;
  padding: 9px 10px;
  border: 1px dashed var(--ink);
  border-radius: 4px;
  background: var(--paper-2);
  grid-template-columns: 46px minmax(0, 1fr) auto;
}
.step-id {
  font-size: 13px;
  font-weight: 700;
  color: var(--muted);
}
.plan-item.gate .step-id {
  color: var(--ink);
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

/* Progress bars */
.overall-box {
  padding: 10px 0 12px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 8px;
}
.overall-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 6px;
}
.overall-count {
  font-size: 18px;
  font-weight: 700;
  color: var(--ink);
}
.overall-stats {
  font-size: 13px;
  color: var(--muted);
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.bar-lg {
  height: 10px;
  background: var(--paper-2);
  border: 1px solid var(--line);
  border-radius: 3px;
  overflow: hidden;
}
.bar-lg > i, .bar-sm > i {
  display: block;
  height: 100%;
  background: var(--ink);
  width: 0%;
}
.bar-sm {
  height: 6px;
  background: var(--paper-2);
  border-radius: 2px;
  overflow: hidden;
}
.exp-list {
  display: flex;
  flex-direction: column;
}
.exp-row {
  display: grid;
  grid-template-columns: minmax(145px, 1.3fr) minmax(80px, 1fr) 115px;
  gap: 10px;
  align-items: center;
  padding: 5px 0;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
.exp-row:last-child {
  border-bottom: none;
}
.exp-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.exp-cnt {
  text-align: right;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}

/* Compact key-value rows and tables */
.kv-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px solid var(--rule);
  font-size: 14px;
}
.kv-row:last-child {
  border-bottom: none;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.log-list, .commit-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.log-item, .commit-item {
  padding: 5px 0;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 10px;
  align-items: baseline;
}
.log-item:last-child, .commit-item:last-child {
  border-bottom: none;
}
.ts {
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.commit-meta {
  color: var(--muted);
  font-size: 12px;
  margin-left: 6px;
  white-space: nowrap;
}
.danger-text {
  color: var(--danger);
  font-weight: 600;
}
@media (max-width: 520px) {
  .exp-row {
    grid-template-columns: minmax(120px, 1.2fr) minmax(60px, 0.8fr) 102px;
    gap: 6px;
  }
}
</style>
</head>
<body>
<header class="top">
  <div class="top-in">
    <h1>Cell-Kill-Sim progress</h1>
    <div class="hdr-meta num">
      <span id="stale-warning" class="stale-banner">Stale: fetch failed</span>
      <span id="last-updated">Updated: —</span>
    </div>
  </div>
</header>

<main>
  <section id="sec-plan">
    <div class="sec-hdr">
      <h2>Plan</h2>
      <span id="plan-meta" class="sec-sub"></span>
    </div>
    <div id="plan-body" class="sec-sub">Loading…</div>
  </section>

  <section id="sec-masks">
    <div class="sec-hdr">
      <h2>Masks</h2>
      <span id="masks-meta" class="sec-sub num"></span>
    </div>
    <div id="masks-body" class="sec-sub">Loading…</div>
  </section>

  <section id="sec-processes">
    <div class="sec-hdr">
      <h2>Processes &amp; Agent Log</h2>
      <span id="proc-meta" class="sec-sub"></span>
    </div>
    <div id="proc-body" class="sec-sub">Loading…</div>
  </section>

  <section id="sec-git-disk">
    <div class="sec-hdr">
      <h2>Git &amp; Disk</h2>
      <span id="git-disk-meta" class="sec-sub num"></span>
    </div>
    <div id="git-disk-body" class="sec-sub">Loading…</div>
  </section>
</main>

<script>
(function() {
  const fmtInt = (n) => Number(n || 0).toLocaleString('en-US');
  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  function chipHtml(status, labelOverride) {
    const st = String(status || 'todo').toLowerCase();
    const cls = (st === 'done' || st === 'in_progress' || st === 'blocked') ? st : 'todo';
    const label = labelOverride != null ? labelOverride : st.replace('_', ' ');
    return '<span class="chip ' + cls + '">' + esc(label) + '</span>';
  }

  function renderPlan(plan) {
    const metaEl = document.getElementById('plan-meta');
    const bodyEl = document.getElementById('plan-body');
    if (!plan || plan.state === 'not started' || !Array.isArray(plan.steps) || plan.steps.length === 0) {
      metaEl.innerHTML = '';
      bodyEl.innerHTML = '<div class="sec-sub">not started</div>';
      return;
    }
    let metaParts = [];
    if (plan.updated) metaParts.push('Updated ' + esc(plan.updated));
    if (plan.review_url) {
      metaParts.push('<a href="' + esc(plan.review_url) + '" target="_blank" rel="noopener">Review artifact</a>');
    }
    metaEl.innerHTML = metaParts.join(' · ');

    let html = '<ul class="plan-list">';
    for (const step of plan.steps) {
      const isGate = String(step.id) === 'check';
      html += '<li class="plan-item' + (isGate ? ' gate' : '') + '">';
      html += '<div class="step-id num">' + esc(step.id) + '</div>';
      html += '<div><div class="step-title">' + esc(step.title) + '</div>';
      if (step.note) {
        html += '<div class="step-note">' + esc(step.note) + '</div>';
      }
      html += '</div>';
      html += '<div>' + chipHtml(step.status) + '</div>';
      html += '</li>';
    }
    html += '</ul>';
    bodyEl.innerHTML = html;
  }

  function renderMasks(masks) {
    const metaEl = document.getElementById('masks-meta');
    const bodyEl = document.getElementById('masks-body');
    if (!masks || (masks.state === 'not started' && !masks.expected_total)) {
      metaEl.textContent = 'not started';
      bodyEl.innerHTML = '<div class="sec-sub">not started</div>';
      return;
    }
    const expTotal = masks.expected_total || 63053;
    const presTotal = masks.present_total || 0;
    const pct = Number(masks.percent || 0).toFixed(1);
    const rate = Number(masks.rate_files_per_min || 0).toFixed(1);
    const eta = masks.eta_human || '—';
    const bytesHuman = masks.bytes_human || '0 B';
    const failures = masks.failures || 0;

    metaEl.innerHTML = esc(bytesHuman) + ' on disk · ' +
      (failures > 0 ? '<span class="danger-text">' + fmtInt(failures) + ' failures</span>' : '0 failures');

    let html = '<div class="overall-box num">';
    html += '<div class="overall-top">';
    html += '<div class="overall-count">' + fmtInt(presTotal) + ' / ' + fmtInt(expTotal) + ' <span class="sec-sub">(' + pct + '%)</span></div>';
    html += '<div class="overall-stats">';
    html += '<span>Rate: <strong>' + esc(rate) + '</strong> files/min</span>';
    html += '<span>ETA: <strong>' + esc(eta) + '</strong></span>';
    html += '</div></div>';
    html += '<div class="bar-lg"><i style="width:' + Math.min(100, Math.max(0, Number(masks.percent || 0))) + '%"></i></div>';
    html += '</div>';

    if (Array.isArray(masks.experiments) && masks.experiments.length > 0) {
      html += '<div class="exp-list num">';
      for (const row of masks.experiments) {
        const rPct = Math.min(100, Math.max(0, Number(row.percent || 0)));
        html += '<div class="exp-row">';
        html += '<div class="exp-name" title="' + esc(row.experiment) + '">' + esc(row.experiment) + '</div>';
        html += '<div class="bar-sm"><i style="width:' + rPct + '%"></i></div>';
        html += '<div class="exp-cnt">' + fmtInt(row.present) + ' / ' + fmtInt(row.expected) + '</div>';
        html += '</div>';
      }
      html += '</div>';
    }
    bodyEl.innerHTML = html;
  }

  function renderProcesses(agent, downloader) {
    const metaEl = document.getElementById('proc-meta');
    const bodyEl = document.getElementById('proc-body');

    const agyActive = Boolean(agent && agent.active);
    const dlActive = Boolean(downloader && downloader.running);

    metaEl.innerHTML =
      'agy: ' + chipHtml(agyActive ? 'in_progress' : 'todo', agyActive ? 'active' : 'idle') +
      ' &nbsp; downloader: ' + chipHtml(dlActive ? 'in_progress' : 'todo', dlActive ? 'running' : 'stopped');

    let html = '';
    html += '<div class="kv-row"><span>Downloader (<code>download_masks.py</code> / <code>download_raw.py</code>)</span><span class="mono num">' +
      (dlActive ? esc(downloader.processes.map(p => 'PID ' + p.pid).join(', ')) : 'not running') + '</span></div>';
    html += '<div class="kv-row"><span>Antigravity agent (<code>agy -p</code>)</span><span class="mono num">' +
      (agyActive ? esc(agent.processes.map(p => 'PID ' + p.pid).join(', ')) : 'idle') + '</span></div>';

    const lines = (agent && Array.isArray(agent.log_lines)) ? agent.log_lines : [];
    if (lines.length === 0) {
      html += '<div class="sec-sub" style="margin-top:8px">No recent sir_flash_lite progress lines in cli.log</div>';
    } else {
      html += '<div class="sec-sub" style="margin:10px 0 4px;font-weight:600;color:var(--ink)">Last ' + lines.length + ' agent monitor lines</div>';
      html += '<ul class="log-list">';
      for (let i = lines.length - 1; i >= 0; i--) {
        const item = lines[i];
        html += '<li class="log-item">';
        html += '<span class="ts mono num">' + esc(item.timestamp || '—') + '</span>';
        html += '<span>' + esc(item.reason) + '</span>';
        html += '</li>';
      }
      html += '</ul>';
    }
    bodyEl.innerHTML = html;
  }

  function renderGitAndDisk(git, disk) {
    const metaEl = document.getElementById('git-disk-meta');
    const bodyEl = document.getElementById('git-disk-body');

    const freeHuman = (disk && disk.free_human) ? disk.free_human : '—';
    const freePct = (disk && disk.free_percent != null) ? (' (' + disk.free_percent + '% free)') : '';
    metaEl.textContent = 'Disk free: ' + freeHuman + freePct;

    let html = '';
    if (disk && disk.state === 'ok') {
      html += '<div class="kv-row num"><span>Volume free space (<code>data/</code>)</span><span><strong>' +
        esc(disk.free_human) + '</strong> free of ' + esc(disk.total_human) + '</span></div>';
    } else {
      html += '<div class="kv-row"><span>Volume free space (<code>data/</code>)</span><span>not started</span></div>';
    }

    if (!git || git.state === 'not started') {
      html += '<div class="kv-row"><span>Git status</span><span>not started</span></div>';
    } else {
      html += '<div class="kv-row num"><span>Branch: <strong class="mono">' + esc(git.branch || '—') + '</strong></span>' +
        '<span>Uncommitted files: <strong>' + fmtInt(git.uncommitted_count) + '</strong></span></div>';
      const commits = Array.isArray(git.commits) ? git.commits : [];
      if (commits.length > 0) {
        html += '<ul class="commit-list" style="margin-top:6px">';
        for (const c of commits) {
          html += '<li class="commit-item">';
          html += '<span class="mono num">' + esc(c.hash) + '</span>';
          html += '<span>' + esc(c.subject) + '<span class="commit-meta num">' + esc(c.relative_time) + '</span></span>';
          html += '</li>';
        }
        html += '</ul>';
      }
    }
    bodyEl.innerHTML = html;
  }

  async function refresh() {
    const staleEl = document.getElementById('stale-warning');
    const updatedEl = document.getElementById('last-updated');
    try {
      const resp = await fetch('/api/status', { cache: 'no-store' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      staleEl.classList.remove('visible');
      updatedEl.textContent = 'Updated: ' + (data.updated_human || new Date().toLocaleTimeString());
      renderPlan(data.plan);
      renderMasks(data.masks);
      renderProcesses(data.agent, data.downloader);
      renderGitAndDisk(data.git, data.disk);
    } catch (err) {
      staleEl.classList.add('visible');
    }
  }

  refresh();
  setInterval(refresh, 5000);
})();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    tracker: MaskProgressTracker

    def log_message(self, fmt: str, *args) -> None:
        # Quiet default stderr logging
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/" or route == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if route == "/api/status":
            now_dt = datetime.datetime.now().astimezone()
            try:
                plan_sec = get_plan_section()
            except Exception as exc:
                plan_sec = {"state": "not started", "error": str(exc), "steps": []}

            try:
                masks_sec = self.tracker.get_status()
            except Exception as exc:
                masks_sec = {
                    "state": "not started",
                    "error": str(exc),
                    "expected_total": 0,
                    "present_total": 0,
                    "experiments": [],
                }

            try:
                agent_sec, downloader_sec = get_agent_And_downloader_sections()
            except Exception as exc:
                agent_sec = {"state": "not started", "active": False, "log_lines": [], "error": str(exc)}
                downloader_sec = {"state": "not started", "running": False, "error": str(exc)}

            try:
                git_sec = get_git_section()
            except Exception as exc:
                git_sec = {"state": "not started", "branch": None, "uncommitted_count": 0, "commits": [], "error": str(exc)}

            try:
                disk_sec = get_disk_section()
            except Exception as exc:
                disk_sec = {"state": "not started", "free_bytes": 0, "free_human": "—", "error": str(exc)}

            payload = {
                "updated_iso": now_dt.isoformat(timespec="seconds"),
                "updated_human": now_dt.strftime("%H:%M:%S"),
                "plan": plan_sec,
                "masks": masks_sec,
                "agent": agent_sec,
                "downloader": downloader_sec,
                "git": git_sec,
                "disk": disk_sec,
            }
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

    tracker = MaskProgressTracker()
    DashboardHandler.tracker = tracker

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
