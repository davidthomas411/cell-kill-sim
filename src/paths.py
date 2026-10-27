"""Repository path constants for portable script execution."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = Path(os.environ.get("CELLKILL_DATA_DIR", REPO_ROOT / "data"))
FIGURES_DIR: Path = REPO_ROOT / "figures"
EXTERNAL_DIR: Path = REPO_ROOT / "external"
VIDEOS_DIR: Path = REPO_ROOT / "videos"
