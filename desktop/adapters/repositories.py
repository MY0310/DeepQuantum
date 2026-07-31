"""Repositories for bundle access and local handling state."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from desktop.app.models import AlertItem, HandlingRecord
from desktop.config.settings import MONITOR_BUNDLE_PATH
from desktop.data.bundle_loader import load_monitor_bundle
from desktop.data.models import NodeDetail, TopologySnapshot

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
STATUS_LABELS = {
    "pending": "待处理",
    "review": "人工复核",
    "watch": "持续观察",
    "false_positive": "误报",
    "resolved": "已处置",
}


def _risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


class DesktopBundleRepository:
    """Read-only access to the offline monitor bundle."""

    def __init__(self, bundle_path: Path | None = None) -> None:
        self.bundle_path = bundle_path or MONITOR_BUNDLE_PATH
        self._bundle: dict[str, Any] | None = None

    def _require_bundle_path(self) -> Path:
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"缺少监控数据包：{self.bundle_path}")
        return self.bundle_path

    def load_bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            self._require_bundle_path()
            self._bundle = load_monitor_bundle(self.bundle_path, auto_build=False)
        return self._bundle

    def reload_bundle(self) -> dict[str, Any]:
        self._bundle = None
        return self.load_bundle()

    def load_snapshot(self, period: int, threshold: float) -> TopologySnapshot:
        bundle = self.load_bundle()
        raw_snapshot = bundle["snapshots"][str(int(period))]
        node_table = bundle["node_index"]
        edges_table = bundle["edges"]
        active_node_ids = raw_snapshot["active_node_ids"]
        active_edge_ids = raw_snapshot["active_edge_ids"]
        active_edges = [edges_table[i] for i in active_edge_ids]

        risk_nodes = []
        for nid in active_node_ids:
            score = float(node_table[str(nid)]["risk_score"])
            if score >= threshold:
                risk_nodes.append(int(nid))

        graph = nx.Graph()
        graph.add_nodes_from(active_node_ids)
        graph.add_edges_from((int(e["u"]), int(e["v"])) for e in active_edges)
        risk_graph = graph.subgraph(risk_nodes).copy()
        clusters = [sorted(list(comp)) for comp in nx.connected_components(risk_graph)] if risk_nodes else []
        clusters.sort(key=len, reverse=True)

        nodes_render = []
        for nid in active_node_ids:
            item = node_table[str(nid)]
            nodes_render.append(
                {
                    "id": int(nid),
                    "sample_idx": int(item["sample_idx"]),
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "risk_score": float(item["risk_score"]),
                    "risk_level": _risk_level(float(item["risk_score"]), threshold),
                    "period": int(item["period"]),
                    "degree": int(item["degree"]),
                }
            )
        nodes_render.sort(key=lambda n: (RISK_ORDER[n["risk_level"]], n["risk_score"]), reverse=True)

        edges_render = []
        for e in active_edges:
            u = int(e["u"])
            v = int(e["v"])
            nu = node_table[str(u)]
            nv = node_table[str(v)]
            edges_render.append(
                {
                    "u": u,
                    "v": v,
                    "x0": float(nu["x"]),
                    "y0": float(nu["y"]),
                    "x1": float(nv["x"]),
                    "y1": float(nv["y"]),
                }
            )

        summary = {
            "active_nodes": len(active_node_ids),
            "active_edges": len(active_edge_ids),
            "risk_nodes": len(risk_nodes),
            "risk_clusters": len(clusters),
        }
        return TopologySnapshot(
            period=int(period),
            threshold=float(threshold),
            summary=summary,
            nodes=nodes_render,
            edges=edges_render,
            risk_nodes=[int(x) for x in risk_nodes],
            risk_clusters=clusters,
        )

    def load_node_detail(self, node_id: int, period: int, threshold: float) -> NodeDetail:
        bundle = self.load_bundle()
        item = bundle["node_index"][str(int(node_id))]
        score = float(item["risk_score"])
        level = _risk_level(score, threshold)
        template = bundle["action_templates"][level]
        evidence = [
            f"风险评分为 {score:.4f}，当前阈值为 {threshold:.2f}。",
            f"节点连接度为 {int(item['degree'])}，位于第 {int(period)} 时段。",
            "与相邻节点形成局部异常结构，建议优先关注关联交易链。"
            if level != "LOW"
            else "当前网络关联相对平稳，建议继续监测。",
        ]
        return NodeDetail(
            node_id=int(node_id),
            period=int(period),
            risk_score=score,
            risk_level=level,
            degree=int(item["degree"]),
            evidence=evidence,
            actions=list(template),
        )

    def list_periods(self) -> list[int]:
        bundle = self.load_bundle()
        return [int(x) for x in bundle["periods"]]

    def get_meta(self) -> dict[str, Any]:
        return dict(self.load_bundle().get("meta", {}))

    def get_sources(self) -> dict[str, Any]:
        return dict(self.load_bundle().get("sources", {}))

    def get_period_stats(self) -> list[dict[str, Any]]:
        return list(self.load_bundle().get("period_stats", []))

    def get_node_item(self, node_id: int) -> dict[str, Any]:
        bundle = self.load_bundle()
        return dict(bundle["node_index"][str(int(node_id))])


class HandlingStateRepository:
    """Persist local handling state in JSON."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> dict[str, dict[str, Any]]:
        if not self.storage_path.exists():
            return {}
        with self.storage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("records", {})

    def save_record(self, period: int, node_id: int, status: str, note: str) -> None:
        payload = {"records": self.load_records()}
        key = f"{int(period)}:{int(node_id)}"
        payload["records"][key] = {
            "period": int(period),
            "node_id": int(node_id),
            "status": str(status),
            "note": str(note),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_record(self, period: int, node_id: int) -> HandlingRecord | None:
        data = self.load_records().get(f"{int(period)}:{int(node_id)}")
        if not data:
            return None
        return HandlingRecord(**data)

    def list_records(self, period: int | None = None) -> list[HandlingRecord]:
        items = []
        for row in self.load_records().values():
            record = HandlingRecord(**row)
            if period is None or int(record.period) == int(period):
                items.append(record)
        items.sort(key=lambda x: (x.period, x.node_id))
        return items

    def export_records_csv(self, path: Path) -> None:
        rows = self.list_records()
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["period", "node_id", "status", "note", "updated_at"])
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)
