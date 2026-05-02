"""Monitor bundle loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ui.config.settings import MONITOR_BUNDLE_PATH


def _auto_build_bundle(path: Path) -> None:
    from ui.scripts.build_monitor_bundle import build_monitor_bundle

    build_monitor_bundle(output_path=path)


def load_monitor_bundle(path: Path | None = None, auto_build: bool = True) -> dict[str, Any]:
    """Load monitor bundle with optional auto-build."""
    target = path or MONITOR_BUNDLE_PATH
    if not target.exists():
        if not auto_build:
            raise FileNotFoundError(f"Missing monitor bundle: {target}")
        _auto_build_bundle(target)
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)

