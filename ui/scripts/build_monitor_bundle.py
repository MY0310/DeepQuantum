"""Build timeline monitoring bundle from local project resources."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.config.settings import MONITOR_BUNDLE_PATH, ROOT_DIR, SOURCE_PATHS, STORAGE_DIR


def _risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


def _to_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _load_test_dataset():
    import sys

    src_root = SOURCE_PATHS["elliptic_data_dir"].parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from data.financial_dataset import load_elliptic_dataset

    _, test_ds = load_elliptic_dataset(
        data_dir=str(SOURCE_PATHS["elliptic_data_dir"]),
        max_nodes=20,
        ego_radius=1.5,
        train_periods=(1, 34),
        test_periods=(35, 49),
        cache_dir=str(SOURCE_PATHS["elliptic_cache_dir"]),
    )
    return test_ds


def _component_layout(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    """Build stable coordinates without heavy full-graph spring iterations."""
    rng = np.random.default_rng(42)
    graph_u = graph.to_undirected() if graph.is_directed() else graph
    positions: dict[int, tuple[float, float]] = {}
    comps = sorted(nx.connected_components(graph_u), key=len, reverse=True)
    if not comps:
        return positions

    for idx, comp_nodes in enumerate(comps):
        nodes = list(comp_nodes)
        comp = graph_u.subgraph(nodes).copy()
        # Use a phyllotaxis spiral for component centers to avoid radial spoke artifacts.
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        radius = 0.55 * math.sqrt(idx + 1)
        angle = idx * golden_angle
        cx = math.cos(angle) * radius
        cy = math.sin(angle) * radius

        if len(nodes) <= 120:
            local = nx.spring_layout(comp, seed=42, iterations=50, scale=0.65)
            for n, (x, y) in local.items():
                positions[int(n)] = (float(cx + x), float(cy + y))
        else:
            for n in nodes:
                jitter = rng.normal(0.0, 0.30, size=2)
                positions[int(n)] = (float(cx + jitter[0]), float(cy + jitter[1]))

    return positions


def build_monitor_bundle(output_path: Path | None = None) -> dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_path or MONITOR_BUNDLE_PATH

    test_ds = _load_test_dataset()
    threshold_npz = sorted(SOURCE_PATHS["threshold_eval_dir"].glob("qgad_probs_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not threshold_npz:
        raise FileNotFoundError("未找到 qgad_probs_*.npz")
    npz_path = threshold_npz[0]
    npz = np.load(npz_path)
    y_test = npz["y_test"]
    p_test = npz["p_test"]
    if len(p_test) != len(test_ds):
        raise ValueError(f"概率文件与测试集样本数不一致: {len(p_test)} vs {len(test_ds)}")

    default_threshold = 0.26
    nodes = [int(n) for n in test_ds.nodes]
    periods = test_ds.node_features[:, 0].astype(int)
    labels = test_ds.labels.astype(int)
    degrees = dict(test_ds.graph.degree())
    coords = _component_layout(test_ds.graph)

    node_rows = []
    node_index: dict[str, dict[str, Any]] = {}
    period_to_nodes_exact: dict[int, list[int]] = {}
    period_values = sorted({int(x) for x in periods.tolist()})
    for idx, node_id in enumerate(nodes):
        period = int(periods[idx])
        score = float(p_test[idx])
        level = _risk_level(score, default_threshold)
        x, y = coords.get(node_id, (0.0, 0.0))
        row = {
            "id": int(node_id),
            "sample_idx": int(idx),
            "period": period,
            "risk_score": score,
            "risk_level": level,
            "label": int(labels[idx]),
            "degree": int(degrees.get(node_id, 0)),
            "x": float(x),
            "y": float(y),
        }
        node_rows.append(row)
        node_index[str(node_id)] = row
        period_to_nodes_exact.setdefault(period, []).append(int(node_id))

    edges: list[dict[str, int]] = [{"u": int(u), "v": int(v)} for u, v in test_ds.graph.edges()]

    snapshots: dict[str, dict[str, Any]] = {}
    cumulative_nodes: set[int] = set()
    period_stats: list[dict[str, int]] = []
    for p in period_values:
        for nid in period_to_nodes_exact.get(p, []):
            cumulative_nodes.add(nid)
        active_nodes = sorted(cumulative_nodes)
        active_set = set(active_nodes)
        active_edge_ids = [i for i, e in enumerate(edges) if e["u"] in active_set and e["v"] in active_set]

        risk_nodes_default = [nid for nid in active_nodes if node_index[str(nid)]["risk_score"] >= default_threshold]
        risk_graph = nx.Graph()
        risk_graph.add_nodes_from(risk_nodes_default)
        for i in active_edge_ids:
            e = edges[i]
            if e["u"] in risk_graph and e["v"] in risk_graph:
                risk_graph.add_edge(e["u"], e["v"])
        risk_clusters_default = [sorted(list(c)) for c in nx.connected_components(risk_graph)] if len(risk_nodes_default) else []
        risk_clusters_default.sort(key=len, reverse=True)

        snapshots[str(p)] = {
            "active_node_ids": active_nodes,
            "active_edge_ids": active_edge_ids,
            "risk_nodes_default": risk_nodes_default,
            "risk_clusters_default": risk_clusters_default,
            "summary": {
                "active_nodes": len(active_nodes),
                "active_edges": len(active_edge_ids),
                "risk_nodes": len(risk_nodes_default),
                "risk_clusters": len(risk_clusters_default),
            },
        }
        period_stats.append({"period": int(p), "active_nodes": len(active_nodes), "active_edges": len(active_edge_ids), "risk_nodes": len(risk_nodes_default)})

    bundle = {
        "meta": {
            "bundle_version": "monitor.v2",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "period_range": [int(min(period_values)), int(max(period_values))],
            "default_threshold": default_threshold,
            "description": "时序拓扑风险监控数据包（累计活跃节点诱导子图）",
        },
        "sources": {
            "threshold_npz": _to_rel(npz_path),
            "data_dir": _to_rel(SOURCE_PATHS["elliptic_data_dir"]),
        },
        "periods": [int(x) for x in period_values],
        "nodes": node_rows,
        "node_index": node_index,
        "edges": edges,
        "snapshots": snapshots,
        "period_stats": period_stats,
        "action_templates": {
            "HIGH": [
                "立即发起人工复核并冻结可疑链路。",
                "优先核查关联账户与资金路径。",
                "进入高优先级告警队列持续跟踪。",
            ],
            "MEDIUM": [
                "开启增强监控并提高抽检频率。",
                "结合历史行为做二次判断。",
                "下一交易窗口继续观察变化。",
            ],
            "LOW": [
                "保持常规监控策略。",
                "纳入周期性风险复审。",
                "暂不触发人工干预。",
            ],
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 monitor_bundle.v2.json")
    parser.add_argument("--output", type=str, default=str(MONITOR_BUNDLE_PATH))
    args = parser.parse_args()
    bundle = build_monitor_bundle(Path(args.output))
    print(f"生成完成: {args.output}")
    print(f"时段范围: {bundle['meta']['period_range']}")
    print(f"节点数量: {len(bundle['nodes'])}, 边数量: {len(bundle['edges'])}")


if __name__ == "__main__":
    main()
