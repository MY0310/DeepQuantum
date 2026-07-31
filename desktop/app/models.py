"""Typed models for the desktop client."""

from __future__ import annotations

from dataclasses import dataclass, field

from desktop.data.models import PredictionResult


@dataclass(slots=True)
class AlertItem:
    node_id: int
    period: int
    risk_score: float
    risk_level: str
    degree: int
    status: str = "pending"


@dataclass(slots=True)
class HandlingRecord:
    node_id: int
    period: int
    status: str
    note: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class DashboardState:
    period: int
    threshold: float
    selected_node_id: int | None
    summary: dict
    alerts: list[AlertItem] = field(default_factory=list)
    prediction: PredictionResult | None = None
