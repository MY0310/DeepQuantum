"""Business panels for monitoring UI."""

from __future__ import annotations

import streamlit as st

from ui.data.models import NodeDetail


def render_node_detail_panel(detail: NodeDetail) -> None:
    """Render node risk detail."""
    level_text = {"HIGH": "高风险", "MEDIUM": "中风险", "LOW": "低风险"}.get(detail.risk_level, "低风险")
    chip_class = {"HIGH": "qgad-risk-high", "MEDIUM": "qgad-risk-medium", "LOW": "qgad-risk-low"}.get(detail.risk_level, "qgad-risk-low")
    st.markdown("#### 研判详情")
    st.markdown(
        f"""
        <div class="qgad-note">
          节点 <b>{detail.node_id}</b> · 时段 <b>{detail.period}</b><br/>
          风险分数 <b>{detail.risk_score:.4f}</b> ·
          <span class="{chip_class}">{level_text}</span> ·
          连接度 <b>{detail.degree}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 风险依据")
    for line in detail.evidence:
        st.markdown(f"- {line}")

    st.markdown("#### 处置建议")
    for action in detail.actions:
        st.markdown(f"- {action}")

