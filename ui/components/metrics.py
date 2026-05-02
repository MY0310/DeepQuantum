"""Risk summary card render helpers."""

from __future__ import annotations

import streamlit as st


def render_risk_summary_card(*, period: int, threshold: float, active_nodes: int, active_edges: int, risk_nodes: int, risk_clusters: int) -> None:
    """Render compact summary card."""
    st.markdown(
        f"""
        <div class="qgad-pred-card">
          <div class="qgad-pred-top">
            <span class="qgad-chip-neutral">时段：{period}</span>
            <span class="qgad-chip-neutral">阈值：{threshold:.2f}</span>
          </div>
          <div class="qgad-pred-meta">
            <span>活跃节点：{active_nodes}</span>
            <span>活跃边：{active_edges}</span>
          </div>
          <div class="qgad-pred-meta">
            <span class="qgad-risk-high">高风险节点：{risk_nodes}</span>
            <span>高风险团簇：{risk_clusters}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

