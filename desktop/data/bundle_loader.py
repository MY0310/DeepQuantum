"""Local monitor bundle loader for the desktop client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from desktop.config.settings import MONITOR_BUNDLE_PATH


def load_monitor_bundle(path: Path | None = None, auto_build: bool = False) -> dict[str, Any]:
    target = path or MONITOR_BUNDLE_PATH
    if not target.exists():
        raise FileNotFoundError(f"Missing monitor bundle: {target}")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)
