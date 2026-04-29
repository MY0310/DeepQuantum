"""
Structured readers for paper visualization data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_dir(parent: Path, prefix: str) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No {prefix}* directory found under {parent}")
    # Use mtime instead of lexical order to avoid stale selection when naming
    # conventions change or manual folders are added.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_file(parent: Path, pattern: str) -> Path:
    candidates = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern} under {parent}")
    return candidates[-1]


def load_model_comparison() -> pd.DataFrame:
    """
    Load Table-2 style metrics for Q-GAD and GNN baselines.
    """
    gnn_exp = latest_dir(ROOT / "gnn_baseline" / "outputs", "experiment_")
    gnn_csv = gnn_exp / "table2_metrics.csv"
    gnn = pd.read_csv(gnn_csv)
    gnn = gnn.rename(columns={"model": "method", "total_params": "params"})

    summary = read_json(ROOT / "experiment_summary.json")
    test = summary.get("test_results", {})
    qgad = pd.DataFrame(
        [
            {
                "method": "Q-GAD",
                "params": int(summary.get("model_config", {}).get("total_parameters", 0)),
                "accuracy": float(test.get("accuracy", 0.0)) * 100.0,
                "precision": float(test.get("precision", 0.0)),
                "recall": float(test.get("recall", 0.0)),
                "f1": float(test.get("f1", 0.0)),
                "auc": float(test.get("auc", 0.0)),
                "ap": float(test.get("average_precision", test.get("ap", 0.0))),
            }
        ]
    )

    df = pd.concat([gnn, qgad], ignore_index=True)
    df["method"] = df["method"].replace({"SAGE": "GraphSAGE"})
    order = ["GCN", "GAT", "GraphSAGE", "GIN", "Q-GAD"]
    df["order"] = df["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    for col in ["params", "accuracy", "precision", "recall", "f1", "auc", "ap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def load_threshold_summary() -> Dict:
    return read_json(latest_file(ROOT / "outputs" / "threshold_eval", "threshold_summary_*.json"))


def load_feature_forgery() -> Dict:
    return read_json(ROOT / "experiments" / "feature_forgery_resistance" / "results" / "feature_forgery_unified_results.json")


def load_adversarial() -> Dict:
    return read_json(ROOT / "experiments" / "adversarial_robustness" / "results" / "qgad_robustness_results.json")


def load_feature_sensitivity() -> Dict:
    return read_json(ROOT / "experiments" / "feature_sensitivity" / "results" / "feature_sensitivity_results.json")


def load_quantum_protection() -> Dict:
    return read_json(ROOT / "experiments" / "quantum_protection_effect" / "results" / "quantum_protection_results.json")


def load_temporal_generalization() -> Dict:
    return read_json(ROOT / "experiments" / "temporal_generalization" / "results" / "temporal_generalization_results.json")


def load_topological_invariance() -> Dict:
    return read_json(ROOT / "experiments" / "topological_invariance" / "results" / "topological_invariance_qgad_results.json")


def load_ablation(prefer_current: bool = True) -> pd.DataFrame:
    """
    Load ablation metrics.

    prefer_current=True uses the latest per-model files when present. This is
    useful after parallel runs where each variant is saved independently.
    """
    results_dir = ROOT / "experiments" / "ablation_study" / "results"
    rows: List[Dict] = []

    if prefer_current:
        for tag, label in [("classical", "Classical"), ("quantum", "Quantum"), ("hybrid", "Hybrid")]:
            files = sorted(results_dir.glob(f"ablation_training_{tag}_*.json"), key=lambda p: p.stat().st_mtime)
            if not files:
                continue
            payload = read_json(files[-1])
            metrics = payload.get("results", {}).get(label, {})
            if metrics:
                rows.append(
                    {
                        "model": label,
                        "f1": float(metrics.get("f1", 0.0)),
                        "auc": float(metrics.get("auc", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "precision": float(metrics.get("precision", 0.0)),
                        "threshold": float(metrics.get("threshold", 0.0)),
                        "source": files[-1].name,
                    }
                )

    if len(rows) < 3:
        legacy = read_json(results_dir / "ablation_training_20260423_142602.json")
        rows = []
        for label, metrics in legacy.get("results", {}).items():
            rows.append(
                {
                    "model": label,
                    "f1": float(metrics.get("f1", 0.0)),
                    "auc": float(metrics.get("auc", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "threshold": float(metrics.get("threshold", 0.0)),
                    "source": "ablation_training_20260423_142602.json",
                }
            )

    order = {"Classical": 0, "Quantum": 1, "Hybrid": 2}
    df = pd.DataFrame(rows)
    df["order"] = df["model"].map(order).fillna(99)
    return df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def metric_matrix(df: pd.DataFrame, metrics: Iterable[str]) -> np.ndarray:
    return df[list(metrics)].to_numpy(dtype=float)
