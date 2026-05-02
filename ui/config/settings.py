"""Central settings for the monitoring UI."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "智能反洗钱风险监测平台"
APP_VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parents[2]
UI_DIR = ROOT_DIR / "ui"
DEEPQUANTUM_ROOT = ROOT_DIR / "Deepquantum"
ASSETS_DIR = UI_DIR / "assets"
STORAGE_DIR = UI_DIR / "storage"
SCRIPTS_DIR = UI_DIR / "scripts"

MONITOR_BUNDLE_PATH = STORAGE_DIR / "monitor_bundle.v2.json"

REALTIME_TIMEOUT_SECONDS = 8.0
REALTIME_DEVICE = "cpu"

SOURCE_PATHS = {
    "threshold_eval_dir": DEEPQUANTUM_ROOT / "outputs" / "threshold_eval",
    "checkpoint_model": DEEPQUANTUM_ROOT / "checkpoints" / "elliptic_model.pt",
    "elliptic_data_dir": DEEPQUANTUM_ROOT / "data" / "elliptic",
    "elliptic_cache_dir": DEEPQUANTUM_ROOT / "data" / "elliptic" / "processed",
}
