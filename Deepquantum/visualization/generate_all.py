"""
Generate all paper-ready visualization figures.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from visualization.experiments_dashboard import main as experiments_main
from visualization.model_comparison import main as model_comparison_main


def main() -> None:
    model_comparison_main()
    experiments_main()


if __name__ == "__main__":
    main()
