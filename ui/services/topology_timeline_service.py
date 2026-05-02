"""Topology timeline service for monitoring UI."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import networkx as nx

from ui.data.bundle_loader import load_monitor_bundle as _load_monitor_bundle_file
from ui.data.models import NodeDetail, TopologySnapshot

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


@lru_cache(maxsize=1)
def load_monitor_bundle() -> dict[str, Any]:
    """Load monitor bundle once per process."""
    return _load_monitor_bundle_file(auto_build=True)


@lru_cache(maxsize=64)
def _build_snapshot(period: int, threshold_key: int) -> TopologySnapshot:
    bundle = load_monitor_bundle()
    threshold = threshold_key / 1000.0
    period_key = str(period)
    raw_snapshot = bundle["snapshots"][period_key]
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

    g = nx.Graph()
    g.add_nodes_from(active_node_ids)
    g.add_edges_from((int(e["u"]), int(e["v"])) for e in active_edges)
    risk_sub = g.subgraph(risk_nodes).copy()
    clusters = [sorted(list(comp)) for comp in nx.connected_components(risk_sub)] if len(risk_nodes) > 0 else []
    clusters.sort(key=len, reverse=True)

    nodes_render = []
    for nid in active_node_ids:
        item = node_table[str(nid)]
        level = _risk_level(float(item["risk_score"]), threshold)
        nodes_render.append(
            {
                "id": int(nid),
                "sample_idx": int(item["sample_idx"]),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "risk_score": float(item["risk_score"]),
                "risk_level": level,
                "period": int(item["period"]),
                "degree": int(item["degree"]),
            }
        )
    nodes_render.sort(key=lambda n: (_RISK_ORDER[n["risk_level"]], n["risk_score"]), reverse=True)

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
        threshold=threshold,
        summary=summary,
        nodes=nodes_render,
        edges=edges_render,
        risk_nodes=[int(x) for x in risk_nodes],
        risk_clusters=clusters,
    )


def get_period_snapshot(period: int, threshold: float) -> TopologySnapshot:
    """Get timeline snapshot for a given period and threshold."""
    key = int(round(float(threshold) * 1000))
    return _build_snapshot(int(period), key)


def get_high_risk_subgraph(period: int, threshold: float, center_node: int | None) -> dict[str, Any]:
    """Get high risk focused subgraph for rendering."""
    snap = get_period_snapshot(period, threshold)
    risk_set = set(snap.risk_nodes)
    if not risk_set:
        return {"period": period, "nodes": [], "edges": [], "center_node": None, "cluster_size": 0}

    chosen_center = None
    if center_node is not None and int(center_node) in risk_set:
        chosen_center = int(center_node)
    else:
        # choose highest risk node
        for n in snap.nodes:
            if n["id"] in risk_set:
                chosen_center = n["id"]
                break
    if chosen_center is None:
        return {"period": period, "nodes": [], "edges": [], "center_node": None, "cluster_size": 0}

    target_cluster: list[int] | None = None
    for cluster in snap.risk_clusters:
        if chosen_center in cluster:
            target_cluster = cluster
            break
    target_cluster = target_cluster or [chosen_center]

    selected = set(target_cluster)
    # add 1-hop neighbors for context
    for e in snap.edges:
        if e["u"] in selected or e["v"] in selected:
            selected.add(e["u"])
            selected.add(e["v"])

    selected_nodes = [n for n in snap.nodes if n["id"] in selected]
    selected_node_ids = {n["id"] for n in selected_nodes}
    selected_edges = [e for e in snap.edges if e["u"] in selected_node_ids and e["v"] in selected_node_ids]

    return {
        "period": period,
        "nodes": selected_nodes,
        "edges": selected_edges,
        "center_node": chosen_center,
        "cluster_size": len(target_cluster),
    }


def get_node_detail(node_id: int, period: int, threshold: float) -> NodeDetail:
    """Get node detail with business evidence and actions."""
    bundle = load_monitor_bundle()
    item = bundle["node_index"][str(int(node_id))]
    score = float(item["risk_score"])
    level = _risk_level(score, threshold)
    template = bundle["action_templates"][level]
    evidence = [
        f"风险评分为 {score:.4f}，当前阈值为 {threshold:.2f}。",
        f"节点连接度为 {int(item['degree'])}，处于第 {int(period)} 时段网络。",
        "与相邻节点形成异常聚集，建议优先关注资金链关联。"
        if level != "LOW"
        else "当前网络关联相对平稳，建议持续监测。",
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


def refresh_monitor_data() -> dict[str, Any]:
    """Clear in-process caches and reload bundle from disk."""
    load_monitor_bundle.cache_clear()
    _build_snapshot.cache_clear()
    return load_monitor_bundle()
