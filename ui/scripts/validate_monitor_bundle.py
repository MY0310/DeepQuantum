"""Validate monitor bundle integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.config.settings import MONITOR_BUNDLE_PATH


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_bundle(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return [f"文件不存在: {path}"], warnings

    data = _read_json(path)
    for key in ["meta", "periods", "nodes", "node_index", "edges", "snapshots", "action_templates"]:
        if key not in data:
            errors.append(f"缺少字段: {key}")

    periods = data.get("periods", [])
    snapshots = data.get("snapshots", {})
    node_index = data.get("node_index", {})
    edges = data.get("edges", [])

    if not periods:
        errors.append("periods 为空")
    else:
        for p in periods:
            if str(p) not in snapshots:
                errors.append(f"snapshots 缺少时段 {p}")

    for nid, item in node_index.items():
        score = float(item.get("risk_score", -1))
        if not (0.0 <= score <= 1.0):
            errors.append(f"节点 {nid} 风险分数越界: {score}")

    node_set = set(node_index.keys())
    for i, e in enumerate(edges[:2000]):
        if str(int(e["u"])) not in node_set or str(int(e["v"])) not in node_set:
            errors.append(f"边索引 {i} 引用了不存在节点")
            break

    for p in periods:
        snap = snapshots[str(p)]
        act_nodes = snap.get("active_node_ids", [])
        act_edges = snap.get("active_edge_ids", [])
        if len(act_nodes) == 0:
            warnings.append(f"时段 {p} active_node_ids 为空")
        if len(act_edges) == 0:
            warnings.append(f"时段 {p} active_edge_ids 为空")
        summary = snap.get("summary", {})
        if summary.get("active_nodes", -1) != len(act_nodes):
            errors.append(f"时段 {p} summary.active_nodes 不一致")
        if summary.get("active_edges", -1) != len(act_edges):
            errors.append(f"时段 {p} summary.active_edges 不一致")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 monitor_bundle.v2.json")
    parser.add_argument("--bundle", default=str(MONITOR_BUNDLE_PATH))
    args = parser.parse_args()
    errors, warnings = validate_bundle(Path(args.bundle))
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    if errors:
        raise SystemExit(1)
    print("[OK] monitor bundle 校验通过")
    print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()

