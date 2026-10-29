#!/usr/bin/env python3
"""Lab record system helpers and CLI for Cell-Kill-Sim.

Provides:
  - require_verified(dataset_id): raises DatasetNotVerifiedError (a RuntimeError)
    unless the dataset's status in lab/registry.csv is 'verified' or
    'verified_with_exclusions'.
  - new_entry(title, gate='none', ...): creates the next numbered notebook entry
    lab/notebook/YYYY-MM-DD-NN-slug.md from the standard template.
  - Shared loaders for registry, galleries, reviews, protocols, and notebook entries.
  - CLI for checking datasets, creating notebook entries, and inspecting lab records.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
import re
import sys
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
LAB_DIR: Path = REPO_ROOT / "lab"
NOTEBOOK_DIR: Path = LAB_DIR / "notebook"
PROTOCOLS_DIR: Path = LAB_DIR / "protocols"
REVIEWS_DIR: Path = LAB_DIR / "reviews"
HANDBOOK_DIR: Path = LAB_DIR / "handbook"
REGISTRY_CSV: Path = LAB_DIR / "registry.csv"
GALLERIES_CSV: Path = LAB_DIR / "galleries.csv"

VALID_GATES: tuple[str, ...] = ("G0", "G1", "G2", "G3", "G4", "G5", "none")
VALID_DATASET_STATUSES: tuple[str, ...] = (
    "unverified",
    "verified",
    "verified_with_exclusions",
    "rejected",
)
PASSING_DATASET_STATUSES: frozenset[str] = frozenset(
    {"verified", "verified_with_exclusions"}
)
VALID_VERDICTS: tuple[str, ...] = ("ok", "problem", "unsure")
FIXED_REVIEW_TAGS: tuple[str, ...] = (
    "fragmented",
    "merged",
    "hollow",
    "edge",
    "tiny",
    "misaligned",
    "drift",
    "label_mismatch",
    "neighbour_overlap",
    "real_death",
    "clean",
    "other",
)


class DatasetNotVerifiedError(RuntimeError):
    """Raised by require_verified() when a dataset layer is not verified."""


def read_head_commit_without_git(repo_root: Path = REPO_ROOT) -> str:
    """Read current short commit hash from .git/ files without invoking git."""
    git_dir = repo_root / ".git"
    head_file = git_dir / "HEAD"
    try:
        if not head_file.is_file():
            return "unknown"
        head_val = head_file.read_text(encoding="utf-8").strip()
        if head_val.startswith("ref:"):
            ref_rel = head_val.split(":", 1)[1].strip()
            ref_path = git_dir / ref_rel
            if ref_path.is_file():
                return ref_path.read_text(encoding="utf-8").strip()[:7]
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == ref_rel:
                        return parts[0][:7]
            return "unknown"
        return head_val[:7]
    except OSError:
        return "unknown"


def load_registry(registry_path: Path | str | None = None) -> list[dict[str, str]]:
    """Load all rows from lab/registry.csv."""
    path = Path(registry_path) if registry_path is not None else REGISTRY_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Registry file not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def require_verified(
    dataset_id: str, registry_path: Path | str | None = None
) -> dict[str, str]:
    """Ensure dataset_id has status 'verified' or 'verified_with_exclusions'.

    Raises:
        KeyError: if dataset_id does not exist in lab/registry.csv.
        DatasetNotVerifiedError (RuntimeError): if dataset status is not verified
            or verified_with_exclusions.
    """
    rows = load_registry(registry_path)
    by_id = {r.get("dataset_id", "").strip(): r for r in rows}
    if dataset_id not in by_id:
        raise KeyError(
            f"Dataset '{dataset_id}' is not registered in "
            f"{registry_path or REGISTRY_CSV}."
        )
    row = by_id[dataset_id]
    status = row.get("status", "").strip()
    if status not in PASSING_DATASET_STATUSES:
        notes = row.get("notes", "").strip()
        raise DatasetNotVerifiedError(
            f"Dataset '{dataset_id}' is not verified (status='{status}'). "
            f"Analysis scripts may only consume datasets with status in "
            f"{sorted(PASSING_DATASET_STATUSES)}. Notes: {notes}"
        )
    return row


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "entry"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def _normalize_gate(gate: str | None) -> str:
    if gate is None:
        return "none"
    g = gate.strip()
    if not g or g.lower() == "none":
        return "none"
    g_up = g.upper()
    if g_up in VALID_GATES:
        return g_up
    raise ValueError(
        f"Invalid gate '{gate}'. Expected one of {VALID_GATES}."
    )


def new_entry(
    title: str,
    gate: str = "none",
    *,
    author: str = "agy",
    date: str | None = None,
    slug: str | None = None,
    datasets: list[str] | None = None,
    protocols: list[str] | None = None,
    galleries: list[str] | None = None,
    notebook_dir: Path | str | None = None,
) -> Path:
    """Create the next numbered notebook entry lab/notebook/YYYY-MM-DD-NN-slug.md."""
    if not title or not title.strip():
        raise ValueError("Notebook entry title must be non-empty.")
    norm_gate = _normalize_gate(gate)
    nb_dir = Path(notebook_dir) if notebook_dir is not None else NOTEBOOK_DIR
    nb_dir.mkdir(parents=True, exist_ok=True)

    entry_date = (date or datetime.date.today().isoformat()).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", entry_date):
        raise ValueError(f"Date must be YYYY-MM-DD, got '{entry_date}'.")

    # Find highest NN for entry_date
    pat = re.compile(rf"^{re.escape(entry_date)}-(\d{{2}})-.+\.md$")
    max_nn = 0
    for existing in nb_dir.iterdir():
        if existing.is_file():
            m = pat.match(existing.name)
            if m:
                max_nn = max(max_nn, int(m.group(1)))

    next_nn = max_nn + 1
    entry_slug = _slugify(slug if slug is not None else title)
    entry_id = f"{entry_date}-{next_nn:02d}-{entry_slug}"
    out_path = nb_dir / f"{entry_id}.md"

    commit_sha = read_head_commit_without_git()
    ds_list = list(datasets) if datasets else []
    pr_list = list(protocols) if protocols else []
    ga_list = list(galleries) if galleries else []

    front_matter = {
        "id": entry_id,
        "date": entry_date,
        "title": title.strip(),
        "gate": norm_gate,
        "status": "draft",
        "author": author,
        "reviewers": [],
        "signoff": None,
        "code_commit": commit_sha,
        "datasets": ds_list,
        "protocols": pr_list,
        "galleries": ga_list,
        "supersedes": None,
        "superseded_by": None,
    }
    yaml_str = yaml.safe_dump(
        front_matter, sort_keys=False, allow_unicode=True
    ).strip()

    content = (
        f"---\n{yaml_str}\n---\n\n"
        "## Question\n\n"
        "State the question tested by this entry.\n\n"
        "## What was run\n\n"
        "```bash\n# exact command(s)\n```\n\n"
        "## What was looked at\n\n"
        "- Gallery ID and panel IDs inspected.\n\n"
        "## Observations\n\n"
        "- Quantitative and visual observations by panel ID.\n\n"
        "## Decision\n\n"
        "Pass/fail gate decision or next action.\n\n"
        "## Open issues\n\n"
        "- Remaining items to resolve.\n"
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path


def parse_markdown_with_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    """Parse YAML front matter and markdown body from a file."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end_idx = text.find("\n---\n", 4)
        if end_idx != -1:
            raw_yaml = text[4:end_idx]
            body = text[end_idx + 5 :].lstrip("\n")
            meta = yaml.safe_load(raw_yaml) or {}
            if isinstance(meta, dict):
                return meta, body
    return {}, text


def list_notebook_entries(notebook_dir: Path = NOTEBOOK_DIR) -> list[dict[str, Any]]:
    """Return all dated notebook entries sorted chronologically."""
    if not notebook_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for p in sorted(notebook_dir.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        meta, body = parse_markdown_with_front_matter(p)
        entries.append(
            {
                "path": p,
                "stem": p.stem,
                "id": str(meta.get("id") or p.stem),
                "date": str(meta.get("date") or p.stem[:10]),
                "title": str(meta.get("title") or p.stem),
                "gate": str(meta.get("gate") or "none"),
                "status": str(meta.get("status") or "draft"),
                "author": str(meta.get("author") or ""),
                "reviewers": meta.get("reviewers") or [],
                "signoff": meta.get("signoff"),
                "code_commit": str(meta.get("code_commit") or ""),
                "datasets": meta.get("datasets") or [],
                "protocols": meta.get("protocols") or [],
                "galleries": meta.get("galleries") or [],
                "supersedes": meta.get("supersedes"),
                "superseded_by": meta.get("superseded_by"),
                "meta": meta,
                "body": body,
            }
        )
    return entries


def list_protocols(protocols_dir: Path = PROTOCOLS_DIR) -> list[dict[str, Any]]:
    """Return all versioned protocols sorted by filename."""
    if not protocols_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(protocols_dir.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        meta, body = parse_markdown_with_front_matter(p)
        items.append(
            {
                "path": p,
                "stem": p.stem,
                "id": str(meta.get("id") or p.stem),
                "version": str(meta.get("version") or "v1"),
                "date": str(meta.get("date") or ""),
                "status": str(meta.get("status") or "draft"),
                "meta": meta,
                "body": body,
            }
        )
    return items


def load_galleries(galleries_path: Path = GALLERIES_CSV) -> list[dict[str, str]]:
    """Load all rows from lab/galleries.csv."""
    if not galleries_path.is_file():
        return []
    with open(galleries_path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_reviews(
    gallery_id: str, reviews_dir: Path = REVIEWS_DIR
) -> list[dict[str, str]]:
    """Load review rows from lab/reviews/<gallery_id>.csv."""
    p = reviews_dir / f"{gallery_id}.csv"
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _cmd_check(args: argparse.Namespace) -> int:
    exit_code = 0
    for ds_id in args.dataset_ids:
        try:
            row = require_verified(ds_id)
            print(
                f"OK: {ds_id} is {row['status']} "
                f"(verified_by={row.get('verified_by') or '—'})"
            )
        except (KeyError, DatasetNotVerifiedError) as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


def _cmd_new_entry(args: argparse.Namespace) -> int:
    title = args.title_opt or (args.title_pos if args.title_pos else None)
    if not title:
        print("Error: --title or positional title is required.", file=sys.stderr)
        return 2
    out = new_entry(
        title=title,
        gate=args.gate,
        author=args.author,
        date=args.date,
        slug=args.slug,
    )
    print(out.relative_to(REPO_ROOT))
    return 0


def _cmd_registry(_args: argparse.Namespace) -> int:
    rows = load_registry()
    for r in rows:
        print(
            f"{r['dataset_id']:24s}  {r['status']:24s}  "
            f"n={r['n_items']:>6s}  {r['description']}"
        )
    return 0


def _cmd_entries(_args: argparse.Namespace) -> int:
    for e in list_notebook_entries():
        print(
            f"{e['id']:38s}  gate={e['gate']:4s}  "
            f"status={e['status']:8s}  {e['title']}"
        )
    return 0


def _cmd_protocols(_args: argparse.Namespace) -> int:
    for p in list_protocols():
        print(
            f"{p['id']}@{p['version']:4s}  date={p['date']:10s}  "
            f"status={p['status']:8s}  ({p['path'].name})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab.py",
        description="Cell-Kill-Sim lab record system CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check",
        help="Verify that dataset_id(s) have status verified or verified_with_exclusions.",
    )
    p_check.add_argument("dataset_ids", nargs="+", help="Dataset ID(s) from lab/registry.csv")
    p_check.set_defaults(func=_cmd_check)

    p_new = sub.add_parser(
        "new-entry",
        help="Create the next numbered notebook entry in lab/notebook/.",
    )
    p_new.add_argument("title_pos", nargs="?", help="Entry title")
    p_new.add_argument("--title", dest="title_opt", help="Entry title")
    p_new.add_argument("--gate", default="none", help="Gate (G0..G5 or none)")
    p_new.add_argument("--author", default="agy", help="Author name (default: agy)")
    p_new.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    p_new.add_argument("--slug", default=None, help="Optional filename slug override")
    p_new.set_defaults(func=_cmd_new_entry)

    p_reg = sub.add_parser("registry", help="List datasets in lab/registry.csv.")
    p_reg.set_defaults(func=_cmd_registry)

    p_ent = sub.add_parser("entries", help="List notebook entries in lab/notebook/.")
    p_ent.set_defaults(func=_cmd_entries)

    p_prot = sub.add_parser("protocols", help="List protocols in lab/protocols/.")
    p_prot.set_defaults(func=_cmd_protocols)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
