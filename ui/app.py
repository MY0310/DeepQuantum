"""单页时序拓扑风险监控台。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.components import build_summary_topology_figure
from ui.config import APP_NAME
from ui.config.theme import load_global_css
from ui.services import get_node_detail, get_period_snapshot, load_monitor_bundle, refresh_monitor_data


def _init_state(default_period: int) -> None:
    if "period" not in st.session_state:
        st.session_state.period = default_period
    if "handled_by_period" not in st.session_state:
        st.session_state.handled_by_period = {}
    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = None
    if "pulse_on" not in st.session_state:
        st.session_state.pulse_on = True


def _build_risk_queue(snapshot) -> list[dict]:
    risk_set = set(snapshot.risk_nodes)
    rows = []
    for n in snapshot.nodes:
        if n["id"] in risk_set:
            rows.append(
                {
                    "node_id": int(n["id"]),
                    "risk_score": float(n["risk_score"]),
                    "risk_level": n["risk_level"],
                    "degree": int(n["degree"]),
                    "x": float(n["x"]),
                    "y": float(n["y"]),
                }
            )
    rows.sort(key=lambda x: x["risk_score"], reverse=True)
    return rows


def _ensure_selected(queue: list[dict], handled_set: set[int]) -> None:
    if not queue:
        st.session_state.selected_node_id = None
        return
    if st.session_state.selected_node_id is not None and any(r["node_id"] == st.session_state.selected_node_id for r in queue):
        return
    for r in queue:
        if r["node_id"] not in handled_set:
            st.session_state.selected_node_id = int(r["node_id"])
            return
    st.session_state.selected_node_id = int(queue[0]["node_id"])


def _build_blink_post_script(blink_trace_index: int, enabled: bool) -> str:
    if not enabled:
        return ""
    return f"""
const gd = document.getElementById('{{plot_id}}');
if (gd && gd.data && gd.data.length > {blink_trace_index}) {{
  Plotly.relayout(gd, {{dragmode: 'zoom'}});
  const idx = {blink_trace_index};
  const n = (gd.data[idx].x || []).length;
  if (n > 0) {{
    if (window.__qgadBlinkTimer) clearInterval(window.__qgadBlinkTimer);
    const pick = (m, k) => {{
      const arr = Array.from({{length: m}}, (_, i) => i);
      for (let i = m - 1; i > 0; i--) {{
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }}
      return arr.slice(0, k);
    }};
    const tick = () => {{
      const k = Math.max(6, Math.min(160, Math.floor(n * 0.08)));
      const ids = pick(n, k);
      const opacity = Array(n).fill(0.06);
      const size = Array(n).fill(7.0);
      for (const id of ids) {{
        opacity[id] = 0.92;
        size[id] = 9.6;
      }}
      Plotly.restyle(gd, {{'marker.opacity': [opacity], 'marker.size': [size]}}, [idx]);
    }};
    tick();
    window.__qgadBlinkTimer = setInterval(tick, 650);
  }}
}}
"""


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=":satellite:", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(f"<style>{load_global_css()}</style>", unsafe_allow_html=True)

    bundle = load_monitor_bundle()
    periods = [int(x) for x in bundle["periods"]]
    period_min, period_max = int(min(periods)), int(max(periods))
    default_threshold = float(bundle["meta"]["default_threshold"])
    _init_state(period_min)

    st.markdown(
        f"""
        <div class="qgad-hero qgad-hero-simple">
          <h1>{APP_NAME}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns([1.75, 1.05, 0.7, 0.7, 0.65])
    with c1:
        st.session_state.period = st.slider("时段", period_min, period_max, int(st.session_state.period), 1)
    with c2:
        threshold = st.slider("风险阈值", 0.01, 0.99, float(default_threshold), 0.01)
    with c3:
        if st.button("重置状态", use_container_width=True):
            st.session_state.handled_by_period[str(st.session_state.period)] = []
    with c4:
        if st.button("刷新数据", use_container_width=True):
            refresh_monitor_data()
            st.rerun()
    with c5:
        st.session_state.pulse_on = st.toggle("闪烁", value=bool(st.session_state.pulse_on))

    snapshot = get_period_snapshot(st.session_state.period, threshold)
    risk_queue = _build_risk_queue(snapshot)
    handled_set = set(st.session_state.handled_by_period.get(str(st.session_state.period), []))
    pending = [x for x in risk_queue if x["node_id"] not in handled_set]
    _ensure_selected(risk_queue, handled_set)

    top_left, top_right = st.columns([1.35, 1.65], gap="large")
    with top_left:
        st.markdown("### 可疑处置")
        st.markdown(
            f"""
            <div class="qgad-side-stat">
              <span>时段 {st.session_state.period}</span>
              <span>阈值 {threshold:.2f}</span>
              <span>待处理 {len(pending)}</span>
              <span>已处理 {len(handled_set)}</span>
              <span>总边数 {snapshot.summary["active_edges"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source = pending if pending else risk_queue
        queue_options = [f"{i+1:03d} | 节点 {r['node_id']} | 分数 {r['risk_score']:.4f}" for i, r in enumerate(source)]
        if queue_options:
            selected_index = 0
            if st.session_state.selected_node_id is not None:
                for i, r in enumerate(source):
                    if r["node_id"] == st.session_state.selected_node_id:
                        selected_index = i
                        break
            chosen = st.selectbox("选择可疑节点", options=queue_options, index=selected_index, label_visibility="collapsed")
            st.session_state.selected_node_id = int(source[queue_options.index(chosen)]["node_id"])

        b1, b2 = st.columns(2)
        with b1:
            if st.button("标记已处理", use_container_width=True) and st.session_state.selected_node_id is not None:
                period_key = str(st.session_state.period)
                cur = set(st.session_state.handled_by_period.get(period_key, []))
                cur.add(int(st.session_state.selected_node_id))
                st.session_state.handled_by_period[period_key] = sorted(cur)
                st.rerun()
        with b2:
            if st.button("下一个", use_container_width=True):
                for r in source:
                    if r["node_id"] != st.session_state.selected_node_id:
                        st.session_state.selected_node_id = int(r["node_id"])
                        break

    with top_right:
        st.markdown("### 节点详情")
        if st.session_state.selected_node_id is not None:
            detail = get_node_detail(int(st.session_state.selected_node_id), st.session_state.period, threshold)
            row = next((x for x in risk_queue if x["node_id"] == st.session_state.selected_node_id), None)
            if row:
                st.markdown(
                    f"""
                    <div class="qgad-node-brief">
                      <div><b>节点ID</b> {detail.node_id}</div>
                      <div><b>风险分数</b> {detail.risk_score:.4f}</div>
                      <div><b>风险等级</b> {detail.risk_level}</div>
                      <div><b>连接度</b> {detail.degree}</div>
                      <div><b>坐标</b> ({row['x']:.3f}, {row['y']:.3f})</div>
                      <div><b>建议</b> {detail.actions[0] if detail.actions else "持续监测"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("当前时段没有可疑节点。")

    snapshot_view = {
        "period": snapshot.period,
        "threshold": snapshot.threshold,
        "summary": snapshot.summary,
        "nodes": snapshot.nodes,
        "edges": snapshot.edges,
        "risk_nodes": snapshot.risk_nodes,
        "risk_clusters": snapshot.risk_clusters,
    }
    fig: go.Figure = build_summary_topology_figure(
        snapshot_view,
        selected_node=st.session_state.selected_node_id,
    )
    risk_nodes_for_blink = [n for n in snapshot.nodes if n["risk_level"] == "HIGH"]
    fig.add_trace(
        go.Scattergl(
            x=[n["x"] for n in risk_nodes_for_blink],
            y=[n["y"] for n in risk_nodes_for_blink],
            mode="markers",
            marker=dict(
                size=7.0,
                color="rgba(255,95,109,0.92)",
                opacity=0.06,
                line=dict(color="rgba(255,220,224,0.92)", width=0.7),
            ),
            hoverinfo="skip",
            showlegend=False,
            name="blink-overlay",
        )
    )
    fig.update_layout(height=860)
    html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "doubleClick": "reset+autosize",
            "responsive": True,
        },
        post_script=_build_blink_post_script(blink_trace_index=len(fig.data) - 1, enabled=bool(st.session_state.pulse_on)),
    )
    components.html(html, height=885, scrolling=False)


if __name__ == "__main__":
    main()
