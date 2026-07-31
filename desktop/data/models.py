"""Typed models used by the desktop client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PredictionResult:
    risk_score: float
    risk_level: str
    pred_label: int
    latency_ms: float
    fallback_used: bool
    mode: str
    sample_id: int
    decision_threshold: float
    message: str = ""
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "pred_label": int(self.pred_label),
            "latency_ms": float(self.latency_ms),
            "fallback_used": bool(self.fallback_used),
            "mode": self.mode,
            "sample_id": int(self.sample_id),
            "decision_threshold": float(self.decision_threshold),
            "message": self.message,
            "backend": self.backend,
        }


@dataclass(slots=True)
class TopologySnapshot:
    period: int
    threshold: float
    summary: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    risk_nodes: list[int]
    risk_clusters: list[list[int]]


@dataclass(slots=True)
class NodeDetail:
    node_id: int
    period: int
    risk_score: float
    risk_level: str
    degree: int
    evidence: list[str]
    actions: list[str]
