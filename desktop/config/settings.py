"""Local settings for the desktop client."""

from __future__ import annotations

from desktop.runtime import get_desktop_root

DESKTOP_DIR = get_desktop_root()
RESOURCES_DIR = DESKTOP_DIR / "resources"
ASSETS_DIR = DESKTOP_DIR / "assets"
STORAGE_DIR = DESKTOP_DIR / "storage"

MONITOR_BUNDLE_PATH = RESOURCES_DIR / "ui" / "storage" / "monitor_bundle.v2.json"
DEEPQUANTUM_ROOT = RESOURCES_DIR / "Deepquantum"
DEEPQUANTUM_SRC = DEEPQUANTUM_ROOT / "src"
CHECKPOINT_MODEL = DEEPQUANTUM_ROOT / "checkpoints" / "elliptic_model.pt"
ELLIPTIC_DATA_DIR = DEEPQUANTUM_ROOT / "data" / "elliptic"
ELLIPTIC_CACHE_DIR = ELLIPTIC_DATA_DIR / "processed"

REALTIME_TIMEOUT_SECONDS = 8.0
REALTIME_DEVICE = "cpu"
