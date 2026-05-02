"""
Matplotlib setup helpers for stable plotting on Windows/headless environments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _ensure_windows_conda_dll_paths() -> None:
    """
    Ensure conda DLL directories are discoverable on Windows.

    This prevents low-level crashes when scripts are launched via
    `...\\envs\\qgad\\python.exe` without an activated conda shell.
    """
    if os.name != "nt":
        return

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        exe_parent = Path(sys.executable).resolve().parent
        if (exe_parent / "conda-meta").exists():
            prefix = str(exe_parent)
    if not prefix:
        return

    dll_dirs = [
        Path(prefix),
        Path(prefix) / "Library" / "mingw-w64" / "bin",
        Path(prefix) / "Library" / "usr" / "bin",
        Path(prefix) / "Library" / "bin",
        Path(prefix) / "Scripts",
    ]
    existing = [str(p) for p in dll_dirs if p.exists()]
    if not existing:
        return

    current_path = os.environ.get("PATH", "")
    path_parts = [p for p in current_path.split(";") if p]
    for p in reversed(existing):
        if p not in path_parts:
            path_parts.insert(0, p)
    os.environ["PATH"] = ";".join(path_parts)

    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is not None:
        for p in existing:
            try:
                add_dll(p)
            except OSError:
                pass


def setup_matplotlib(project_root: Path, backend: str = "Agg"):
    """
    Configure a writable matplotlib cache and a non-interactive backend.
    """
    mpl_dir = project_root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    _ensure_windows_conda_dll_paths()
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", backend)

    import matplotlib

    try:
        matplotlib.use(backend, force=True)
    except Exception:
        # Backend may already be initialized in embedded contexts.
        pass
    return matplotlib


def get_plot_modules(project_root: Path, backend: str = "Agg"):
    """
    Return pyplot after safe backend/cache initialization.
    """
    setup_matplotlib(project_root, backend=backend)
    import matplotlib.pyplot as plt

    return plt


def apply_default_style(plt) -> None:
    """
    Apply publication-oriented defaults (Times New Roman first).

    Notes:
    - Keeps pure matplotlib style to avoid seaborn hard dependency.
    - Uses a restrained, print-friendly palette and line style defaults.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def journal_palette() -> dict:
    """
    Color palette with restrained contrast for paper figures.
    """
    return {
        "qgad": "#A13D2D",
        "qgad_light": "#D98B73",
        "gnn": "#2F5D8C",
        "gnn_light": "#87A9C7",
        "xgb": "#5B6C37",
        "classical": "#3A6B7A",
        "quantum": "#8B6F47",
        "hybrid": "#A13D2D",
        "full": "#A13D2D",
        "zero": "#5C6672",
        "accent": "#C49A3A",
        "neutral": "#5C6672",
        "grid": "#D9D9D9",
        "text": "#1E1E1E",
    }


def set_axis_style(ax) -> None:
    """
    Apply final axis polishing used across paper figures.
    """
    ax.tick_params(axis="both", direction="out", length=3.5, width=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig,
    output_base: Path,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int | None = None,
) -> None:
    """
    Save a figure as high-resolution PNG and vector PDF.
    """
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"dpi": dpi} if dpi is not None else {}
        fig.savefig(
            output_base.with_suffix(f".{fmt}"),
            facecolor="white",
            edgecolor="white",
            transparent=False,
            **kwargs,
        )
