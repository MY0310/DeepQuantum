"""Theme tokens and CSS helpers."""

from __future__ import annotations

from pathlib import Path

from ui.config.settings import ASSETS_DIR

THEME_TOKENS = {
    "bg_0": "#0a1018",
    "bg_1": "#101a27",
    "bg_2": "#132235",
    "panel": "#111c2a",
    "panel_alt": "#162436",
    "text_main": "#e8edf5",
    "text_muted": "#9fb2c7",
    "accent": "#16b8a8",
    "accent_soft": "#1f7cbd",
    "warn": "#ff9f43",
    "danger": "#ff5f6d",
    "ok": "#27d980",
}


def _token_css() -> str:
    return "\n".join(f"  --qgad-{k.replace('_', '-')}: {v};" for k, v in THEME_TOKENS.items())


def load_global_css() -> str:
    """Load stylesheet and inject color tokens."""
    style_path = ASSETS_DIR / "styles.css"
    base_css = style_path.read_text(encoding="utf-8") if style_path.exists() else ""
    return f":root {{\n{_token_css()}\n}}\n{base_css}"

