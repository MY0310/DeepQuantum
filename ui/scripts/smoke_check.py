"""Smoke checks for monitoring UI."""

from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.services.topology_timeline_service import (
    get_high_risk_subgraph,
    get_node_detail,
    get_period_snapshot,
    load_monitor_bundle,
)


def main() -> None:
    bundle = load_monitor_bundle()
    periods = bundle["periods"]
    period = int(periods[0])
    threshold = float(bundle["meta"]["default_threshold"])

    snap = get_period_snapshot(period, threshold)
    sub = get_high_risk_subgraph(period, threshold, center_node=None)
    if snap.nodes:
        detail = get_node_detail(snap.nodes[0]["id"], period, threshold)
        print(f"[OK] 详情节点={detail.node_id} 风险={detail.risk_level} 分数={detail.risk_score:.4f}")
    print(f"[OK] 时段={period} 节点={snap.summary['active_nodes']} 边={snap.summary['active_edges']} 高风险={snap.summary['risk_nodes']}")
    print(f"[OK] 子图节点={len(sub.get('nodes', []))} 边={len(sub.get('edges', []))}")
    print("[OK] 监控台 smoke check 通过")


if __name__ == "__main__":
    main()

