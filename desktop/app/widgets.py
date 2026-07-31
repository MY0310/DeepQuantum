"""Qt widgets used by the desktop client."""

from __future__ import annotations

import random
from typing import Any

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QLinearGradient, QPainter, QPen, QRadialGradient
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.adapters.repositories import STATUS_LABELS, status_label
from desktop.app.models import AlertItem, HandlingRecord
from desktop.data.models import PredictionResult

RISK_COLORS = {
    "LOW": QColor("#6f879f"),
    "MEDIUM": QColor("#ff9f43"),
    "HIGH": QColor("#ff5f6d"),
}
STATUS_COLORS = {
    "pending": "#7fa0c3",
    "review": "#ffb054",
    "watch": "#18bfaa",
    "false_positive": "#8f9bae",
    "resolved": "#2fd27e",
}


class TopologyHeader(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("topologyHeader")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(6)
        self.title_label = QLabel("全局态势")
        self.title_label.setObjectName("topologyTitle")
        self.focus_label = QLabel("正在载入当前聚焦节点。")
        self.focus_label.setObjectName("topologyFocus")
        layout.addWidget(self.title_label, 0)
        layout.addWidget(self.focus_label, 1)

    def set_content(self, headline: str) -> None:
        self.focus_label.setText(headline)


class TopologyLegend(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("topologyLegend")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(8)
        items = [
            ("低", "#6f879f"),
            ("中", "#ff9f43"),
            ("高", "#ff5f6d"),
        ]
        for text, color in items:
            chip = QLabel(f"● {text}")
            chip.setObjectName("legendChip")
            chip.setStyleSheet(f"color: {color};")
            layout.addWidget(chip)


class TopologyCanvas(QWidget):
    def __init__(self, topology_view: "TopologyView", header: TopologyHeader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.topology_view = topology_view
        self.header = header
        self.legend = TopologyLegend(self)
        self._ambient_phase = 0
        self._ambient_timer = QTimer(self)
        self._ambient_timer.timeout.connect(self._tick_ambient)
        self._ambient_timer.start(70)
        self.top_overlay: QWidget | None = None
        self.left_overlay: QWidget | None = None
        self.right_overlay: QWidget | None = None
        self.header.setParent(self)
        self.legend.setParent(self)
        self.header.raise_()
        self.legend.raise_()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.topology_view)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        phase = self._ambient_phase / 100.0

        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#08111a"))
        gradient.setColorAt(0.55, QColor("#0b1622"))
        gradient.setColorAt(1.0, QColor("#09131e"))
        painter.fillRect(rect, gradient)

        glow_center = QPointF(
            rect.center().x() + rect.width() * 0.04 * phase,
            rect.center().y() - rect.height() * 0.03 * phase,
        )
        glow_alpha = 34 + int(8 * abs(phase))
        glow = QRadialGradient(glow_center, max(rect.width(), rect.height()) * 0.44)
        glow.setColorAt(0.0, QColor(31, 85, 117, glow_alpha))
        glow.setColorAt(0.55, QColor(23, 54, 78, 18))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(rect, glow)

        vignette = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.72)
        vignette.setColorAt(0.7, QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 42))
        painter.fillRect(rect, vignette)

        sweep_x = rect.left() + rect.width() * ((phase + 1.0) / 2.0)
        sweep = QLinearGradient(QPointF(sweep_x - 140, rect.top()), QPointF(sweep_x + 140, rect.top()))
        sweep.setColorAt(0.0, QColor(0, 0, 0, 0))
        sweep.setColorAt(0.5, QColor(102, 196, 232, 14))
        sweep.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(rect, sweep)

        scan_pen = QPen(QColor(118, 170, 208, 6), 1.0)
        painter.setPen(scan_pen)
        scan_step = 22
        offset = int((self._ambient_phase * 0.6) % scan_step)
        for y in range(rect.top() + offset, rect.bottom(), scan_step):
            painter.drawLine(rect.left(), y, rect.right(), y)
        super().paintEvent(event)

    def _tick_ambient(self) -> None:
        self._ambient_phase = (self._ambient_phase + 2) % 200
        self.update()

    def attach_overlays(self, top_overlay: QWidget, left_overlay: QWidget, right_overlay: QWidget) -> None:
        self.top_overlay = top_overlay
        self.left_overlay = left_overlay
        self.right_overlay = right_overlay
        for widget in [top_overlay, left_overlay, right_overlay]:
            widget.setParent(self)
            widget.raise_()
        self._layout_overlay_widgets()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_overlay_widgets()

    def _layout_overlay_widgets(self) -> None:
        margin = 18
        header_width = min(max(int(self.width() * 0.16), 220), 300)
        self.header.setGeometry(margin, margin, header_width, 40)
        self.legend.setGeometry(margin, self.height() - 38, 102, 22)
        self.header.raise_()
        self.legend.raise_()
        if self.top_overlay is not None:
            hint_width = self.top_overlay.sizeHint().width() + 28
            top_width = min(max(hint_width, 560), 860)
            self.top_overlay.setGeometry(self.width() - top_width - margin, margin, top_width, 46)
            self.top_overlay.raise_()
        if self.left_overlay is not None:
            self.left_overlay.setGeometry(8, 94, 320, max(360, self.height() - 148))
            self.left_overlay.raise_()
        if self.right_overlay is not None:
            panel_width = 432
            self.right_overlay.setGeometry(self.width() - panel_width - 8, 94, panel_width, max(420, self.height() - 148))
            self.right_overlay.raise_()


class AlertListWidget(QListWidget):
    node_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSpacing(6)
        self.itemSelectionChanged.connect(self._emit_selected)

    def set_alerts(self, alerts: list[AlertItem], selected_node_id: int | None) -> None:
        self.blockSignals(True)
        self.clear()
        selected_item = None
        for alert in alerts:
            text = (
                f"{alert.node_id}   {alert.risk_level}   {alert.risk_score:.4f}\n"
                f"度 {alert.degree}   {status_label(alert.status)}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, alert.node_id)
            item.setForeground(QColor("#e8eef9"))
            item.setBackground(QColor("#162130"))
            border_color = STATUS_COLORS.get(alert.status, "#7fa0c3")
            item.setData(Qt.UserRole + 1, border_color)
            item.setSizeHint(QSize(item.sizeHint().width(), item.sizeHint().height() + 18))
            self.addItem(item)
            if selected_node_id is not None and alert.node_id == selected_node_id:
                selected_item = item
        if selected_item is not None:
            self.setCurrentItem(selected_item)
            selected_item.setSelected(True)
        elif self.count() > 0:
            self.setCurrentRow(0)
        self.blockSignals(False)

    def _emit_selected(self) -> None:
        item = self.currentItem()
        if item is None:
            return
        self.node_selected.emit(int(item.data(Qt.UserRole)))


class NodeItem(QGraphicsEllipseItem):
    def __init__(self, node_data: dict[str, Any], radius: float, parent: QGraphicsItem | None = None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.node_id = int(node_data["id"])
        self.node_data = node_data
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(RISK_COLORS.get(str(node_data["risk_level"]), QColor("#6f879f"))))
        self.setPen(QPen(QColor(255, 255, 255, 40), 0.8))
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setToolTip(
            f"节点 {self.node_id}\n"
            f"风险分数 {float(node_data['risk_score']):.4f}\n"
            f"等级 {node_data['risk_level']}\n"
            f"连接度 {node_data['degree']}"
        )


class HaloItem(QGraphicsEllipseItem):
    def __init__(self, radius: float, parent: QGraphicsItem | None = None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setPen(QPen(QColor(255, 95, 109, 0), 0))
        self.setBrush(QBrush(QColor(255, 95, 109, 32)))
        self.setZValue(-1)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)


class ClusterGlowItem(QGraphicsEllipseItem):
    def __init__(self, radius: float, color: QColor, parent: QGraphicsItem | None = None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setPen(QPen(QColor(0, 0, 0, 0), 0))
        self.setBrush(QBrush(color))
        self.setZValue(-3)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)


class FocusRingItem(QGraphicsEllipseItem):
    def __init__(self, radius: float, parent: QGraphicsItem | None = None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        pen = QPen(QColor(255, 255, 255, 235), 1.4)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(255, 255, 255, 0)))
        self.setZValue(5)
        self.setVisible(False)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)


class FocusRippleItem(QGraphicsEllipseItem):
    def __init__(self, radius: float, parent: QGraphicsItem | None = None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        pen = QPen(QColor(92, 214, 255, 140), 1.1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(255, 255, 255, 0)))
        self.setZValue(4)
        self.setVisible(False)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)


class TopologyView(QGraphicsView):
    node_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setBackgroundBrush(QBrush(Qt.NoBrush))
        self._nodes: dict[int, NodeItem] = {}
        self._high_risk_nodes: list[NodeItem] = []
        self._high_risk_halos: dict[int, HaloItem] = {}
        self._cluster_glows: list[ClusterGlowItem] = []
        self._edge_items: list[tuple[int, int, QGraphicsLineItem]] = []
        self._selected_node_id: int | None = None
        self._has_snapshot = False
        self._pulse_phase = 0
        self._flicker_rng = random.Random(42)
        self._flicker_state: dict[int, tuple[float, float]] = {}
        self._node_visual_base: dict[int, tuple[float, float]] = {}
        self._hotspot_bucket: dict[int, int] = {}
        self._min_zoom = 0.01
        self._max_zoom = 12.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self._pulse_timer.start(520)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.viewport().setStyleSheet("background: transparent;")
        self._focus_ring = FocusRingItem(9.5)
        self._focus_ripple = FocusRippleItem(15.0)
        self._scene.addItem(self._focus_ring)
        self._scene.addItem(self._focus_ripple)

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 0.85
        current = self.transform().m11()
        target = current * factor
        if target < self._min_zoom or target > self._max_zoom:
            return
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._has_snapshot:
            rect = self._scene.itemsBoundingRect()
            if not rect.isNull():
                self._scene.setSceneRect(self._expanded_scene_rect(rect))

    def drawBackground(self, painter: QPainter, rect) -> None:
        del painter, rect

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        if isinstance(item, NodeItem):
            self.node_selected.emit(item.node_id)
        super().mousePressEvent(event)

    def set_snapshot(self, snapshot, selected_node_id: int | None = None) -> None:
        self._scene.clear()
        self._nodes.clear()
        self._high_risk_nodes.clear()
        self._high_risk_halos.clear()
        self._cluster_glows.clear()
        self._edge_items.clear()
        self._flicker_state.clear()
        self._node_visual_base.clear()
        self._hotspot_bucket.clear()
        self._focus_ring = FocusRingItem(9.5)
        self._focus_ripple = FocusRippleItem(15.0)
        self._scene.addItem(self._focus_ring)
        self._scene.addItem(self._focus_ripple)
        self._selected_node_id = selected_node_id
        self._has_snapshot = True
        for edge in snapshot.edges:
            color = QColor(255, 95, 109, 182) if edge["u"] in snapshot.risk_nodes and edge["v"] in snapshot.risk_nodes else QColor(168, 202, 232, 154)
            line = QGraphicsLineItem(edge["x0"] * 140, -edge["y0"] * 140, edge["x1"] * 140, -edge["y1"] * 140)
            pen = QPen(color, 1.0)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setZValue(-2)
            self._scene.addItem(line)
            self._edge_items.append((int(edge["u"]), int(edge["v"]), line))
        node_lookup = {int(node["id"]): node for node in snapshot.nodes}
        for cluster in snapshot.risk_clusters[:10]:
            if len(cluster) < 2:
                continue
            members = [node_lookup[nid] for nid in cluster if nid in node_lookup]
            if not members:
                continue
            for idx, nid in enumerate(cluster):
                self._hotspot_bucket[int(nid)] = idx % 3
            cx = sum(float(node["x"]) for node in members) / len(members)
            cy = sum(float(node["y"]) for node in members) / len(members)
            radius = max(30.0, min(76.0, 18.0 + len(members) * 9.0))
            alpha = max(12, min(34, 10 + len(members) * 4))
            glow = ClusterGlowItem(radius, QColor(255, 95, 109, alpha))
            glow.setPos(QPointF(cx * 140, -cy * 140))
            self._scene.addItem(glow)
            self._cluster_glows.append(glow)
        for node in snapshot.nodes:
            if node["risk_level"] == "HIGH":
                halo = HaloItem(8.0)
                halo.setPos(QPointF(float(node["x"]) * 140, -float(node["y"]) * 140))
                self._scene.addItem(halo)
                self._high_risk_halos[int(node["id"])] = halo
            radius = 4.2 if node["risk_level"] == "HIGH" else 2.4
            item = NodeItem(node, radius)
            item.setPos(QPointF(float(node["x"]) * 140, -float(node["y"]) * 140))
            item.setZValue(2 if node["risk_level"] == "HIGH" else 1)
            self._scene.addItem(item)
            self._nodes[item.node_id] = item
            if node["risk_level"] == "HIGH":
                self._high_risk_nodes.append(item)
                self._flicker_state[item.node_id] = (1.0, 1.0)
        self.highlight_node(selected_node_id)
        self.reset_view()

    def highlight_node(self, node_id: int | None) -> None:
        self._selected_node_id = node_id
        neighbor_ids: set[int] = set()
        if node_id is not None:
            for u, v, _line in self._edge_items:
                if u == node_id:
                    neighbor_ids.add(v)
                elif v == node_id:
                    neighbor_ids.add(u)
        for current_id, item in self._nodes.items():
            is_focus = current_id == node_id
            is_neighbor = current_id in neighbor_ids
            pen = QPen(QColor(255, 255, 255, 225), 2.0) if is_focus else QPen(QColor(255, 255, 255, 35), 0.65)
            item.setPen(pen)
            base_scale = 1.45 if is_focus else (1.08 if is_neighbor else 1.0)
            base_opacity = 1.0 if is_focus else (0.92 if is_neighbor else 0.42 if node_id is not None else 0.86)
            self._node_visual_base[current_id] = (base_scale, base_opacity)
            item.setScale(base_scale)
            item.setOpacity(base_opacity)
            item.setZValue(4 if is_focus else (3 if is_neighbor else (2 if item.node_data["risk_level"] == "HIGH" else 1)))
        for u, v, line in self._edge_items:
            if node_id is None:
                line.setOpacity(0.9)
                pen = line.pen()
                base_color = QColor(255, 95, 109, 182) if u in self._high_risk_halos and v in self._high_risk_halos else QColor(168, 202, 232, 154)
                pen.setColor(base_color)
                pen.setWidthF(1.0)
                line.setPen(pen)
                line.setZValue(-2)
                continue
            if u == node_id or v == node_id:
                line.setOpacity(1.0)
                pen = line.pen()
                pen.setColor(QColor(114, 224, 255, 224))
                pen.setWidthF(1.35)
                line.setPen(pen)
                line.setZValue(3)
            elif u in neighbor_ids and v in neighbor_ids:
                line.setOpacity(0.86)
                pen = line.pen()
                pen.setColor(QColor(255, 95, 109, 156))
                pen.setWidthF(1.15)
                line.setPen(pen)
                line.setZValue(1)
            else:
                line.setOpacity(0.5)
                pen = line.pen()
                pen.setColor(QColor(148, 182, 210, 108))
                pen.setWidthF(0.95)
                line.setPen(pen)
                line.setZValue(-3)
        for nid, halo in self._high_risk_halos.items():
            halo.setOpacity(1.0 if node_id is None or nid == node_id or nid in neighbor_ids else 0.18)
        for glow in self._cluster_glows:
            glow.setOpacity(1.0 if node_id is None else 0.38)
        if node_id is not None and node_id in self._nodes:
            self.centerOn(self._nodes[node_id])
            self._focus_ring.setVisible(True)
            self._focus_ripple.setVisible(True)
            self._focus_ring.setPos(self._nodes[node_id].pos())
            self._focus_ripple.setPos(self._nodes[node_id].pos())
        else:
            self._focus_ring.setVisible(False)
            self._focus_ripple.setVisible(False)
        self._tick_pulse()

    def reset_view(self) -> None:
        if not self._has_snapshot:
            return
        self.resetTransform()
        rect = self._scene.itemsBoundingRect()
        if not rect.isNull():
            scene_rect = self._expanded_scene_rect(rect)
            self._scene.setSceneRect(scene_rect)
            self.fitInView(scene_rect, Qt.KeepAspectRatio)

    def _expanded_scene_rect(self, content_rect: QRectF) -> QRectF:
        rect = content_rect.adjusted(-180, -160, 180, 160)
        viewport = self.viewport().rect()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return rect
        target_ratio = float(viewport.width()) / float(max(viewport.height(), 1))
        current_ratio = rect.width() / max(rect.height(), 1.0)
        if current_ratio < target_ratio:
            target_width = rect.height() * target_ratio
            extra = (target_width - rect.width()) / 2.0
            rect.adjust(-extra, 0.0, extra, 0.0)
        else:
            target_height = rect.width() / max(target_ratio, 0.01)
            extra = (target_height - rect.height()) / 2.0
            rect.adjust(0.0, -extra, 0.0, extra)
        return rect

    def _tick_pulse(self) -> None:
        if not self._high_risk_halos:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 8
        pulse_scale = 1.22 if self._pulse_phase in {0, 1, 2, 3} else 1.0
        pulse_alpha = 52 if self._pulse_phase in {0, 1, 2, 3} else 24
        active_bucket = (self._pulse_phase // 2) % 3
        surge_on = self._pulse_phase in {0, 1, 4, 5}
        for u, v, line in self._edge_items:
            edge_focus = self._selected_node_id is not None and (u == self._selected_node_id or v == self._selected_node_id)
            hotspot_u = self._hotspot_bucket.get(u, -1) == active_bucket
            hotspot_v = self._hotspot_bucket.get(v, -1) == active_bucket
            pen = line.pen()
            if edge_focus:
                pen.setColor(QColor(126, 232, 255, 240 if surge_on else 208))
                pen.setWidthF(1.6 if surge_on else 1.28)
                line.setPen(pen)
                line.setOpacity(1.0)
                line.setZValue(4)
            elif hotspot_u and hotspot_v:
                pen.setColor(QColor(255, 118, 132, 214 if surge_on else 172))
                pen.setWidthF(1.45 if surge_on else 1.16)
                line.setPen(pen)
                line.setOpacity(0.94 if surge_on else 0.78)
                line.setZValue(2)
            elif hotspot_u or hotspot_v:
                pen.setColor(QColor(102, 210, 245, 176 if surge_on else 136))
                pen.setWidthF(1.18 if surge_on else 1.0)
                line.setPen(pen)
                line.setOpacity(max(line.opacity(), 0.66 if surge_on else 0.56))
                line.setZValue(max(line.zValue(), 0))
        for node_id, halo in self._high_risk_halos.items():
            flicker_scale, flicker_opacity = self._flicker_state.get(node_id, (1.0, 1.0))
            hotspot_on = self._hotspot_bucket.get(node_id, -1) == active_bucket
            hotspot_scale = 1.28 if hotspot_on else 0.96
            hotspot_alpha = 1.4 if hotspot_on else 0.75
            if node_id == self._selected_node_id:
                halo.setScale(1.38 * flicker_scale * hotspot_scale)
                halo.setBrush(QBrush(QColor(255, 255, 255, 28)))
                continue
            halo.setScale(pulse_scale * flicker_scale * hotspot_scale)
            color = QColor(255, 95, 109)
            color.setAlpha(int(pulse_alpha * flicker_opacity * hotspot_alpha))
            halo.setBrush(QBrush(color))
        for item in self._high_risk_nodes:
            is_focus = item.node_id == self._selected_node_id
            base_scale, base_opacity = self._node_visual_base.get(item.node_id, (1.0, 0.86))
            hotspot_on = self._hotspot_bucket.get(item.node_id, -1) == active_bucket
            if hotspot_on:
                flicker_scale = 1.10 + self._flicker_rng.uniform(0.08, 0.22)
                flicker_opacity = 1.02 + self._flicker_rng.uniform(0.06, 0.16)
            elif self._flicker_rng.random() < 0.18:
                flicker_scale = 1.02 + self._flicker_rng.uniform(0.02, 0.08)
                flicker_opacity = 0.92 + self._flicker_rng.uniform(0.03, 0.08)
            else:
                flicker_scale = 1.0
                flicker_opacity = 0.94
            self._flicker_state[item.node_id] = (flicker_scale, flicker_opacity)
            item.setScale(base_scale * flicker_scale)
            if not is_focus:
                item.setOpacity(min(1.0, base_opacity * flicker_opacity))
        cluster_alpha = 34 if self._pulse_phase in {0, 1, 2, 3} else 20
        for glow in self._cluster_glows:
            color = QColor(255, 95, 109, cluster_alpha)
            glow.setBrush(QBrush(color))
        if self._focus_ring.isVisible():
            focus_scale = 1.14 if self._pulse_phase in {0, 1, 2, 3} else 1.0
            self._focus_ring.setScale(focus_scale)
        if self._focus_ripple.isVisible():
            ripple_scale = 1.28 if self._pulse_phase in {0, 1, 2, 3} else 1.08
            ripple_alpha = 108 if self._pulse_phase in {0, 1, 2, 3} else 54
            pen = QPen(QColor(92, 214, 255, ripple_alpha), 1.1)
            pen.setCosmetic(True)
            self._focus_ripple.setPen(pen)
            self._focus_ripple.setScale(ripple_scale)


class DetailPanel(QFrame):
    status_changed = pyqtSignal(str)
    predict_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("detailPanel")
        self._status_buttons: dict[str, QPushButton] = {}
        self.node_id: int | None = None
        self._realtime_hint = "点击下方按钮，使用本地真实模型对当前节点重新推理。"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.title_label = QLabel("节点研判")
        self.title_label.setObjectName("panelTitle")
        layout.addWidget(self.title_label)

        self.summary_label = QLabel("请选择左侧告警节点。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("detailSummary")
        layout.addWidget(self.summary_label)

        self.meta_card = QLabel("节点ID --\n风险评分 --\n风险等级 --\n处置状态 --")
        self.meta_card.setWordWrap(True)
        self.meta_card.setObjectName("multilineCard")
        layout.addWidget(self.meta_card)

        self.evidence_label = QLabel("依据")
        self.evidence_label.setObjectName("sectionLabel")
        layout.addWidget(self.evidence_label)
        self.evidence_text = QLabel("")
        self.evidence_text.setWordWrap(True)
        self.evidence_text.setObjectName("multilineCard")
        layout.addWidget(self.evidence_text)

        self.actions_label = QLabel("动作")
        self.actions_label.setObjectName("sectionLabel")
        layout.addWidget(self.actions_label)
        self.actions_text = QLabel("")
        self.actions_text.setWordWrap(True)
        self.actions_text.setObjectName("multilineCard")
        layout.addWidget(self.actions_text)

        self.predict_label = QLabel("实时推理")
        self.predict_label.setObjectName("sectionLabel")
        layout.addWidget(self.predict_label)
        self.predict_card = QLabel("点击下方按钮，使用本地真实模型对当前节点重新推理。")
        self.predict_card.setWordWrap(True)
        self.predict_card.setObjectName("multilineCard")
        layout.addWidget(self.predict_card)
        self.predict_button = QPushButton("实时推理")
        self.predict_button.clicked.connect(self.predict_requested.emit)
        layout.addWidget(self.predict_button)

        self.note_label = QLabel("备注")
        self.note_label.setObjectName("sectionLabel")
        layout.addWidget(self.note_label)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("记录人工判断、关注点或后续动作。")
        self.note_edit.setMaximumBlockCount(12)
        self.note_edit.setFixedHeight(84)
        layout.addWidget(self.note_edit)

        status_layout = QGridLayout()
        status_layout.setHorizontalSpacing(10)
        status_layout.setVerticalSpacing(8)
        for idx, key in enumerate(STATUS_LABELS):
            button = QPushButton(status_label(key))
            button.clicked.connect(lambda _, s=key: self.status_changed.emit(s))
            self._status_buttons[key] = button
            button.setMinimumHeight(34)
            status_layout.addWidget(button, idx // 2, idx % 2)
        layout.addLayout(status_layout)
        layout.addStretch(1)

    def set_realtime_capability(self, available: bool, hint: str) -> None:
        self._realtime_hint = str(hint)

    def note_text(self) -> str:
        return self.note_edit.toPlainText().strip()

    def set_detail(self, detail, record: HandlingRecord | None, prediction: PredictionResult | None = None) -> None:
        if detail is None:
            self.summary_label.setText("当前时段没有符合条件的高风险节点。")
            self.meta_card.setText("节点ID --\n风险评分 --\n风险等级 --\n处置状态 --")
            self.evidence_text.setText("")
            self.actions_text.setText("")
            self.predict_card.setText("当前没有可推理的节点。")
            self.predict_button.setEnabled(False)
            self.note_edit.setPlainText("")
            for button in self._status_buttons.values():
                button.setProperty("activeStatus", False)
                button.style().unpolish(button)
                button.style().polish(button)
            return
        self.node_id = detail.node_id
        status = record.status if record else "pending"
        note = record.note if record else ""
        self.summary_label.setText(f"节点 {detail.node_id}｜风险评分 {detail.risk_score:.3f}｜{detail.risk_level}")
        self.meta_card.setText(
            "\n".join(
                [
                    f"节点ID  {detail.node_id}",
                    f"风险评分  {detail.risk_score:.4f}",
                    f"风险等级  {detail.risk_level}",
                    f"处置状态  {status_label(status)}",
                ]
            )
        )
        compact_evidence = detail.evidence[:2]
        compact_actions = detail.actions[:2]
        self.evidence_text.setText("\n".join(f"• {line}" for line in compact_evidence))
        self.actions_text.setText("\n".join(f"• {line}" for line in compact_actions))
        self.predict_button.setEnabled(True)
        if prediction is None:
            self.predict_card.setText(self._realtime_hint)
        else:
            self.predict_card.setText(
                "\n".join(
                    [
                        f"• 实时评分 {prediction.risk_score:.4f}｜等级 {prediction.risk_level}",
                        f"• 耗时 {prediction.latency_ms:.1f} ms｜后端 {prediction.backend or '本地模型'}",
                        f"• 样本索引 {prediction.sample_id}｜阈值 {prediction.decision_threshold:.2f}",
                        f"• {prediction.message}",
                    ]
                )
            )
        self.note_edit.setPlainText(note)
        for key, button in self._status_buttons.items():
            button.setProperty("activeStatus", key == status)
            button.style().unpolish(button)
            button.style().polish(button)


class AnalysisDialog(QDialog):
    def __init__(self, period_stats: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("时序分析")
        self.resize(620, 420)
        layout = QVBoxLayout(self)
        intro = QLabel("下表仅展示比值型指标，避免累计统计被误读为风险持续恶化。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        table = QTableWidget(len(period_stats), 5)
        table.setHorizontalHeaderLabels(["时段", "风险占比", "团簇比值", "风险节点", "活跃节点"])
        table.verticalHeader().setVisible(False)
        for row, item in enumerate(period_stats):
            values = [
                str(item["period"]),
                f"{item['risk_ratio']:.2%}",
                f"{item['cluster_ratio']:.2%}",
                str(item["risk_nodes"]),
                str(item["active_nodes"]),
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        layout.addWidget(table)
