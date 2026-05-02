"""Plotly charts for timeline monitoring UI."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

COLOR_BG = "rgba(0,0,0,0)"
COLOR_GRID = "rgba(192,210,228,0.12)"
COLOR_TEXT = "#e8edf5"
COLOR_LOW = "#6f879f"
COLOR_MID = "#ff9f43"
COLOR_HIGH = "#ff5f6d"
COLOR_LINK = "rgba(142,180,214,0.36)"
COLOR_LINK_RISK = "rgba(255,95,109,0.62)"


def _layout_common(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.98, font=dict(size=16, color="#edf4ff")),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=8, r=8, t=46, b=8),
        showlegend=False,
        hoverlabel=dict(bgcolor="rgba(18,28,42,0.95)", bordercolor="rgba(132,164,198,0.7)", font=dict(color="#e8edf5")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    return fig


def _node_color(level: str) -> str:
    if level == "HIGH":
        return COLOR_HIGH
    if level == "MEDIUM":
        return COLOR_MID
    return COLOR_LOW


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _robust_xy_range(nodes: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    if not nodes:
        return [-1.0, 1.0], [-1.0, 1.0]
    xs = [float(n["x"]) for n in nodes]
    ys = [float(n["y"]) for n in nodes]
    x0 = _percentile(xs, 0.01)
    x1 = _percentile(xs, 0.99)
    y0 = _percentile(ys, 0.01)
    y1 = _percentile(ys, 0.99)
    dx = max(x1 - x0, 1.0)
    dy = max(y1 - y0, 1.0)
    pad_x = dx * 0.08
    pad_y = dy * 0.08
    return [x0 - pad_x, x1 + pad_x], [y0 - pad_y, y1 + pad_y]


def _edge_coords(edges: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for e in edges:
        x_vals.extend([float(e["x0"]), float(e["x1"]), None])  # type: ignore[arg-type]
        y_vals.extend([float(e["y0"]), float(e["y1"]), None])  # type: ignore[arg-type]
    return x_vals, y_vals


def build_summary_topology_figure(
    snapshot: dict[str, Any],
    selected_node: int | None = None,
    pulse_step: int = 0,
    blink_node_ids: set[int] | None = None,
) -> go.Figure:
    """Build topology summary graph."""
    fig = go.Figure()
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])
    risk_set = set(snapshot.get("risk_nodes", []))

    # Draw links in two layers to improve readability: base links + risk links.
    risk_edges = [e for e in edges if e["u"] in risk_set and e["v"] in risk_set]
    normal_edges = [e for e in edges if not (e["u"] in risk_set and e["v"] in risk_set)]
    nx, ny = _edge_coords(normal_edges)
    rx, ry = _edge_coords(risk_edges)

    fig.add_trace(
        go.Scattergl(
            x=nx,
            y=ny,
            mode="lines",
            line=dict(color="rgba(154,190,225,0.46)", width=1.05),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=rx,
            y=ry,
            mode="lines",
            line=dict(color=COLOR_LINK_RISK, width=1.6),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers",
            marker=dict(
                size=[12 if selected_node is not None and n["id"] == selected_node else (6.8 if n["risk_level"] == "HIGH" else 3.2) for n in nodes],
                color=[_node_color(n["risk_level"]) for n in nodes],
                line=dict(color="rgba(230,240,255,0.18)", width=0.25),
                opacity=0.78,
            ),
            text=[
                f"节点 {n['id']}<br>风险分数 {n['risk_score']:.4f}<br>等级 {n['risk_level']}<br>度 {n['degree']}"
                for n in nodes
            ],
            customdata=[n["id"] for n in nodes],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    if blink_node_ids:
        blink_nodes = [n for n in nodes if int(n["id"]) in blink_node_ids]
        if blink_nodes:
            alpha = 0.9 if (pulse_step % 2 == 0) else 0.35
            fig.add_trace(
                go.Scattergl(
                    x=[n["x"] for n in blink_nodes],
                    y=[n["y"] for n in blink_nodes],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=f"rgba(255,95,109,{alpha})",
                        line=dict(color="rgba(255,220,224,0.92)", width=0.7),
                    ),
                    hoverinfo="skip",
                )
            )

    if selected_node is not None:
        focus = next((n for n in nodes if int(n["id"]) == int(selected_node)), None)
        if focus is not None:
            fig.add_trace(
                go.Scattergl(
                    x=[focus["x"]],
                    y=[focus["y"]],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color="rgba(255,255,255,0.06)",
                        line=dict(color="rgba(255,255,255,0.92)", width=1.8),
                    ),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=[focus["x"]],
                    y=[focus["y"]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="rgba(255,95,109,0.92)",
                        line=dict(color="rgba(255,255,255,0.95)", width=1.5),
                    ),
                    hoverinfo="skip",
                )
            )

    xr, yr = _robust_xy_range(nodes)
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)
    return _layout_common(fig, f"全网拓扑（时段 {snapshot['period']}）")


def build_risk_subgraph_figure(subgraph: dict[str, Any]) -> go.Figure:
    """Build focused risk cluster graph."""
    fig = go.Figure()
    if not subgraph.get("nodes"):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="当前时段无高风险聚集子图",
            showarrow=False,
            font=dict(color=COLOR_TEXT, size=14),
            xref="paper",
            yref="paper",
        )
        return _layout_common(fig, "高风险子图")

    center_node = subgraph.get("center_node")
    risk_nodes = {n["id"] for n in subgraph["nodes"] if n["risk_level"] == "HIGH"}
    edges = subgraph["edges"]
    risk_edges = [e for e in edges if e["u"] in risk_nodes and e["v"] in risk_nodes]
    normal_edges = [e for e in edges if not (e["u"] in risk_nodes and e["v"] in risk_nodes)]

    nx, ny = _edge_coords(normal_edges)
    rx, ry = _edge_coords(risk_edges)
    fig.add_trace(
        go.Scattergl(
            x=nx,
            y=ny,
            mode="lines",
            line=dict(color="rgba(138,161,186,0.48)", width=1.2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=rx,
            y=ry,
            mode="lines",
            line=dict(color="rgba(255,95,109,0.82)", width=2.0),
            hoverinfo="skip",
        )
    )

    nodes = subgraph["nodes"]
    fig.add_trace(
        go.Scattergl(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers",
            marker=dict(
                size=[16 if n["id"] == center_node else 9 for n in nodes],
                color=[_node_color(n["risk_level"]) for n in nodes],
                line=dict(
                    color=["rgba(255,255,255,0.95)" if n["id"] == center_node else "rgba(255,255,255,0.45)" for n in nodes],
                    width=[2.0 if n["id"] == center_node else 0.9 for n in nodes],
                ),
                opacity=0.98,
            ),
            text=[
                f"节点 {n['id']}<br>风险分数 {n['risk_score']:.4f}<br>等级 {n['risk_level']}<br>度 {n['degree']}"
                for n in nodes
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    xr, yr = _robust_xy_range(nodes)
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)
    cluster_size = int(subgraph.get("cluster_size", 0))
    return _layout_common(fig, f"风险子图（团簇规模 {cluster_size}）")


def build_timeline_metric_figure(period_stats: list[dict[str, Any]]) -> go.Figure:
    """Build compact timeline trend chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[x["period"] for x in period_stats],
            y=[x["risk_nodes"] for x in period_stats],
            mode="lines+markers",
            line=dict(color=COLOR_HIGH, width=2.6),
            marker=dict(size=7),
            name="高风险节点数",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x["period"] for x in period_stats],
            y=[x["active_nodes"] for x in period_stats],
            mode="lines",
            line=dict(color="rgba(90,135,188,0.8)", width=1.8),
            name="活跃节点数",
        )
    )
    fig.update_layout(
        title="时段风险变化",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=16, r=12, t=44, b=22),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.01),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False, title="时段")
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False, title="数量")
    return fig
