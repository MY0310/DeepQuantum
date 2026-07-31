"""Controller for the Q-GAD desktop main window."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from desktop.adapters.repositories import DesktopBundleRepository, HandlingStateRepository, status_label
from desktop.app.models import AlertItem, DashboardState
from desktop.services.prediction_service import predict_single


class MainWindowController:
    """Bridge between repositories and Qt widgets."""

    def __init__(self, bundle_repo: DesktopBundleRepository, handling_repo: HandlingStateRepository) -> None:
        self.bundle_repo = bundle_repo
        self.handling_repo = handling_repo
        periods = self.bundle_repo.list_periods()
        meta = self.bundle_repo.get_meta()
        default_threshold = float(meta.get("default_threshold", 0.26))
        first_period = 46 if 46 in periods else int(min(periods))
        self._state = DashboardState(
            period=first_period,
            threshold=default_threshold,
            selected_node_id=None,
            summary={},
            alerts=[],
        )
        self.refresh_state()

    @property
    def state(self) -> DashboardState:
        return self._state

    def list_periods(self) -> list[int]:
        return self.bundle_repo.list_periods()

    def get_meta(self) -> dict[str, Any]:
        return self.bundle_repo.get_meta()

    def get_sources(self) -> dict[str, Any]:
        return self.bundle_repo.get_sources()

    def get_period_stats(self) -> list[dict[str, Any]]:
        stats = []
        for row in self.bundle_repo.get_period_stats():
            active_nodes = max(int(row.get("active_nodes", 0)), 1)
            risk_nodes = int(row.get("risk_nodes", 0))
            risk_clusters = 0
            try:
                snap = self.bundle_repo.load_snapshot(int(row["period"]), self._state.threshold)
                risk_clusters = int(snap.summary.get("risk_clusters", 0))
            except Exception:
                risk_clusters = 0
            stats.append(
                {
                    "period": int(row["period"]),
                    "risk_ratio": risk_nodes / active_nodes,
                    "cluster_ratio": risk_clusters / max(risk_nodes, 1),
                    "risk_nodes": risk_nodes,
                    "active_nodes": active_nodes,
                    "risk_clusters": risk_clusters,
                }
            )
        return stats

    def get_demo_presets(self) -> list[dict[str, Any]]:
        stats = self.get_period_stats()
        if not stats:
            return []
        first = min(stats, key=lambda x: x["period"])
        latest = max(stats, key=lambda x: x["period"])
        highest_risk = max(stats, key=lambda x: (x["risk_nodes"], x["risk_ratio"]))
        highest_ratio = max(stats, key=lambda x: (x["risk_ratio"], x["risk_nodes"]))
        presets = [
            {"key": "first", "label": "起始时段", "period": first["period"], "note": "展示系统初始告警态势"},
            {"key": "peak_risk", "label": "告警峰值", "period": highest_risk["period"], "note": "展示高风险节点最多的时段"},
            {"key": "peak_ratio", "label": "风险占比峰值", "period": highest_ratio["period"], "note": "展示风险占比最高的时段"},
            {"key": "latest", "label": "最新时段", "period": latest["period"], "note": "展示最终累计态势"},
        ]
        dedup = []
        seen: set[int] = set()
        for item in presets:
            if item["period"] in seen:
                continue
            dedup.append(item)
            seen.add(item["period"])
        return dedup

    def set_period(self, period: int) -> DashboardState:
        self._state.period = int(period)
        self.refresh_state()
        return self._state

    def set_threshold(self, threshold: float) -> DashboardState:
        self._state.threshold = float(threshold)
        self.refresh_state()
        return self._state

    def select_node(self, node_id: int | None) -> DashboardState:
        self._state.selected_node_id = int(node_id) if node_id is not None else None
        self._state.prediction = None
        return self._state

    def select_next_alert(self) -> DashboardState:
        if not self._state.alerts:
            self._state.selected_node_id = None
            return self._state
        unresolved = [x for x in self._state.alerts if x.status not in {"resolved", "false_positive"}]
        source = unresolved or self._state.alerts
        current = self._state.selected_node_id
        if current is None:
            self._state.selected_node_id = source[0].node_id
            return self._state
        ids = [x.node_id for x in source]
        if current in ids:
            idx = ids.index(current)
            self._state.selected_node_id = source[(idx + 1) % len(source)].node_id
        else:
            self._state.selected_node_id = source[0].node_id
        return self._state

    def apply_demo_preset(self, period: int) -> DashboardState:
        self._state.period = int(period)
        self.refresh_state()
        self.select_next_alert()
        return self._state

    def refresh_state(self) -> DashboardState:
        snapshot = self.bundle_repo.load_snapshot(self._state.period, self._state.threshold)
        alerts = []
        for node in snapshot.nodes:
            if int(node["id"]) not in snapshot.risk_nodes:
                continue
            record = self.handling_repo.get_record(snapshot.period, int(node["id"]))
            alerts.append(
                AlertItem(
                    node_id=int(node["id"]),
                    period=snapshot.period,
                    risk_score=float(node["risk_score"]),
                    risk_level=str(node["risk_level"]),
                    degree=int(node["degree"]),
                    status=record.status if record else "pending",
                )
            )
        alerts.sort(key=lambda x: x.risk_score, reverse=True)
        self._state.summary = dict(snapshot.summary)
        self._state.alerts = alerts
        self._state.prediction = None
        if self._state.selected_node_id is None or all(x.node_id != self._state.selected_node_id for x in alerts):
            self._state.selected_node_id = alerts[0].node_id if alerts else None
        return self._state

    def get_snapshot(self):
        return self.bundle_repo.load_snapshot(self._state.period, self._state.threshold)

    def get_node_detail(self, node_id: int | None = None):
        target = node_id if node_id is not None else self._state.selected_node_id
        if target is None:
            return None
        return self.bundle_repo.load_node_detail(int(target), self._state.period, self._state.threshold)

    def get_handling_record(self, node_id: int | None = None):
        target = node_id if node_id is not None else self._state.selected_node_id
        if target is None:
            return None
        return self.handling_repo.get_record(self._state.period, int(target))

    def get_focus_summary(self) -> dict[str, Any]:
        detail = self.get_node_detail()
        if detail is None:
            return {
                "headline": "当前时段暂无高风险节点",
                "subline": "可以切换时段或调整阈值查看其他离线快照。",
            }
        record = self.get_handling_record(detail.node_id)
        status = record.status if record else "pending"
        return {
            "headline": f"节点 {detail.node_id}｜{detail.risk_score:.3f}",
            "subline": f"风险等级 {detail.risk_level} · 连接度 {detail.degree} · 处置状态 {status_label(status)}",
        }

    def update_status(self, node_id: int, status: str, note: str = "") -> DashboardState:
        self.handling_repo.save_record(self._state.period, int(node_id), status, note)
        return self.refresh_state()

    def run_realtime_prediction(self) -> Any:
        node_id = self._state.selected_node_id
        if node_id is None:
            return None
        item = self.bundle_repo.get_node_item(int(node_id))
        sample_id = int(item["sample_idx"])
        fallback = {
            "risk_score": float(item["risk_score"]),
            "risk_level": str(item.get("risk_level", "MEDIUM")),
            "pred_label": 1 if float(item["risk_score"]) >= self._state.threshold else 0,
            "latency_ms": 0.0,
        }
        result = predict_single(sample_id=sample_id, threshold=self._state.threshold, fallback=fallback)
        self._state.prediction = result
        return result

    def export_alerts(self, path: str) -> None:
        rows = []
        for alert in self._state.alerts:
            rows.append(
                {
                    "period": alert.period,
                    "node_id": alert.node_id,
                    "risk_score": f"{alert.risk_score:.6f}",
                    "risk_level": alert.risk_level,
                    "degree": alert.degree,
                    "status": alert.status,
                }
            )
        with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["period", "node_id", "risk_score", "risk_level", "degree", "status"])
            writer.writeheader()
            writer.writerows(rows)

    def export_records(self, path: str) -> None:
        self.handling_repo.export_records_csv(Path(path))

    def export_node_summary(self, path: str, node_id: int | None = None) -> None:
        detail = self.get_node_detail(node_id)
        record = self.get_handling_record(node_id)
        if detail is None:
            return
        payload = {
            "detail": asdict(detail),
            "record": asdict(record) if record else None,
            "threshold": self._state.threshold,
            "period": self._state.period,
        }
        target = Path(path)
        if target.suffix.lower() == ".txt":
            lines = [
                f"节点ID: {detail.node_id}",
                f"时段: {detail.period}",
                f"风险评分: {detail.risk_score:.4f}",
                f"风险等级: {detail.risk_level}",
                f"连接度: {detail.degree}",
                f"处置状态: {record.status if record else 'pending'}",
                f"备注: {record.note if record else ''}",
                "风险依据:",
                *[f"- {line}" for line in detail.evidence],
                "建议动作:",
                *[f"- {line}" for line in detail.actions],
            ]
            target.write_text("\n".join(lines), encoding="utf-8")
        else:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
