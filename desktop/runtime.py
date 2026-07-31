"""Runtime path helpers for source and frozen desktop builds."""

from __future__ import annotations

import sys
from pathlib import Path


def get_desktop_root() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        return base / "desktop"
    return Path(__file__).resolve().parent


def get_project_root() -> Path:
    if getattr(sys, "frozen", False):
        return get_desktop_root().parent
    return Path(__file__).resolve().parents[1]
