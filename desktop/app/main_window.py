"""Main window for the Q-GAD desktop client."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop.config.settings import STORAGE_DIR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop.adapters.repositories import DesktopBundleRepository, HandlingStateRepository
from desktop.app.controller import MainWindowController
from desktop.app.widgets import AnalysisDialog, AlertListWidget, DetailPanel, TopologyCanvas, TopologyHeader, TopologyView
from desktop.services.prediction_service import get_runtime_capability, warmup_runtime_async


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.setWindowTitle("Q-GAD 桌面客户端")
        self.resize(1600, 920)

        bundle_repo = DesktopBundleRepository()
        handling_repo = HandlingStateRepository(STORAGE_DIR / "session_state.json")
        self.controller = MainWindowController(bundle_repo, handling_repo)

        self.topology_header = TopologyHeader()
        self.alert_list = AlertListWidget()
        self.topology_view = TopologyView()
        self.topology_canvas = TopologyCanvas(self.topology_view, self.topology_header)
        self.detail_panel = DetailPanel()

        self.period_combo = QComboBox()
        periods = self.controller.list_periods()
        for period in periods:
            self.period_combo.addItem(f"T{period}", period)
        initial_index = self.period_combo.findData(self.controller.state.period)
        self.period_combo.setCurrentIndex(initial_index if initial_index >= 0 else 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(int(round(self.controller.state.threshold * 100)))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.controller.state.threshold)
        self.preset_buttons: list[QPushButton] = []
        self.action_buttons: dict[str, QPushButton] = {}
        self.alerts_visible = False
        self.detail_visible = False

        self._build_ui()
        realtime_available, realtime_hint = get_runtime_capability()
        if realtime_available:
            self.detail_panel.set_realtime_capability(True, "点击“实时推理”调用本地 Q-GAD 模型，重新计算该节点风险。")
        else:
            self.detail_panel.set_realtime_capability(False, "点击“实时推理”更新当前节点风险结果。")
        self._connect_signals()
        self.refresh_view()
        warmup_runtime_async()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.topology_canvas, stretch=1)
        self.setCentralWidget(root)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("floatingTopBar")
        self.top_bar.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        top_bar_layout = QHBoxLayout(self.top_bar)
        top_bar_layout.setContentsMargins(12, 6, 12, 6)
        top_bar_layout.setSpacing(6)

        self.period_label = QLabel("T")
        top_bar_layout.addWidget(self.period_label)
        self.period_combo.setMinimumContentsLength(3)
        self.period_combo.setMaxVisibleItems(8)
        top_bar_layout.addWidget(self.period_combo)
        self.threshold_text_label = QLabel("阈")
        self.threshold_text_label.setContentsMargins(4, 0, 0, 0)
        top_bar_layout.addWidget(self.threshold_text_label)
        top_bar_layout.addWidget(self.threshold_slider)
        top_bar_layout.addSpacing(6)
        top_bar_layout.addWidget(self.threshold_spin)
        top_bar_layout.addSpacing(8)
        for text, handler in [
            ("下一告警", self.select_next_alert),
            ("定位", self.focus_selected_node),
            ("重置", self.topology_view.reset_view),
            ("队列", self.toggle_alerts_panel),
            ("研判", self.toggle_detail_panel),
        ]:
            button = QPushButton(text)
            button.setObjectName("actionButton")
            button.clicked.connect(handler)
            top_bar_layout.addWidget(button)
            self.action_buttons[text] = button
        top_bar_layout.addStretch(1)

        self.left_overlay = QFrame()
        self.left_overlay.setObjectName("floatingSidePanel")
        left_layout = QVBoxLayout(self.left_overlay)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        left_title = QLabel("告警队列")
        left_title.setObjectName("sidePanelTitle")
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.alert_list)
        self.left_overlay.hide()

        self.right_overlay = QFrame()
        self.right_overlay.setObjectName("floatingSidePanel")
        right_layout = QVBoxLayout(self.right_overlay)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.detail_scroll.setWidget(self.detail_panel)
        right_layout.addWidget(self.detail_scroll, stretch=1)
        action_bar = QWidget()
        action_layout = QGridLayout(action_bar)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setHorizontalSpacing(8)
        action_layout.setVerticalSpacing(8)
        for idx, (text, handler) in enumerate([
            ("时序", self.open_analysis_dialog),
            ("告警", self.export_alerts),
            ("记录", self.export_records),
            ("摘要", self.export_node_summary),
        ]):
            button = QPushButton(text)
            button.clicked.connect(handler)
            button.setMinimumHeight(34)
            action_layout.addWidget(button, idx // 2, idx % 2)
        right_layout.addWidget(action_bar)
        self.right_overlay.hide()

        self.topology_canvas.attach_overlays(self.top_bar, self.left_overlay, self.right_overlay)
        self._update_responsive_ui()

    def _connect_signals(self) -> None:
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_spin.valueChanged.connect(self.on_threshold_spin_changed)
        self.alert_list.node_selected.connect(self.on_node_selected)
        self.topology_view.node_selected.connect(self.on_node_selected)
        self.detail_panel.status_changed.connect(self.on_status_changed)
        self.detail_panel.predict_requested.connect(self.run_realtime_prediction)

    def refresh_view(self) -> None:
        state = self.controller.state
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(state.threshold)
        self.threshold_spin.blockSignals(False)
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(round(state.threshold * 100)))
        self.threshold_slider.blockSignals(False)
        self.alert_list.set_alerts(state.alerts, state.selected_node_id)
        snapshot = self.controller.get_snapshot()
        self.topology_view.set_snapshot(snapshot, state.selected_node_id)
        detail = self.controller.get_node_detail()
        record = self.controller.get_handling_record()
        self.detail_panel.set_detail(detail, record, state.prediction)
        focus = self.controller.get_focus_summary()
        self.topology_header.set_content(focus["headline"])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_ui()

    def _update_responsive_ui(self) -> None:
        width = self.width()
        if width >= 1500:
            combo_w = 72
            slider_w = 136
            spin_w = 70
            top_font = 9
            header_font = 14
            action_min_width = 50
            preset_min_width = 38
            action_labels = {
                "下一告警": "next",
                "定位": "定位",
                "重置": "重置",
                "队列": "队列",
                "研判": "研判",
            }
        elif width >= 1280:
            combo_w = 68
            slider_w = 124
            spin_w = 66
            top_font = 8
            header_font = 13
            action_min_width = 48
            preset_min_width = 36
            action_labels = {
                "下一告警": "next",
                "定位": "定位",
                "重置": "重置",
                "队列": "队列",
                "研判": "研判",
            }
        else:
            combo_w = 62
            slider_w = 108
            spin_w = 60
            top_font = 8
            header_font = 12
            action_min_width = 44
            preset_min_width = 34
            action_labels = {
                "下一告警": "next",
                "定位": "定",
                "重置": "复位",
                "队列": "告警",
                "研判": "研判",
            }
        self.period_combo.setFixedWidth(combo_w)
        self.threshold_slider.setFixedWidth(slider_w)
        self.threshold_spin.setFixedWidth(spin_w)
        self.top_bar.setStyleSheet(
            f"QLabel,QPushButton,QComboBox,QDoubleSpinBox {{ font-size: {top_font}px; }}"
        )
        self.topology_header.setStyleSheet(
            f"QLabel#topologyTitle {{ font-size: {header_font}px; font-weight: 700; }}"
            f" QLabel#topologyFocus {{ font-size: {max(9, header_font - 4)}px; }}"
        )
        for key, button in self.action_buttons.items():
            button.setText(action_labels[key])
            button.setMinimumWidth(action_min_width)
            button.setFixedHeight(26)
        for button in self.preset_buttons:
            button.setMinimumWidth(preset_min_width)
            button.setFixedHeight(26)

    def on_period_changed(self, _index: int) -> None:
        period = int(self.period_combo.currentData())
        self.controller.set_period(period)
        self.refresh_view()

    def on_threshold_changed(self, value: int) -> None:
        threshold = float(value) / 100.0
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(threshold)
        self.threshold_spin.blockSignals(False)
        self.controller.set_threshold(threshold)
        self.refresh_view()

    def on_threshold_spin_changed(self, value: float) -> None:
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(round(float(value) * 100)))
        self.threshold_slider.blockSignals(False)
        self.controller.set_threshold(float(value))
        self.refresh_view()

    def on_node_selected(self, node_id: int) -> None:
        self.controller.select_node(node_id)
        self.topology_view.highlight_node(node_id)
        detail = self.controller.get_node_detail(node_id)
        record = self.controller.get_handling_record(node_id)
        self.detail_panel.set_detail(detail, record, self.controller.state.prediction)
        focus = self.controller.get_focus_summary()
        self.topology_header.set_content(focus["headline"])

    def on_status_changed(self, status: str) -> None:
        node_id = self.controller.state.selected_node_id
        if node_id is None:
            return
        self.controller.update_status(node_id, status, self.detail_panel.note_text())
        self.refresh_view()
        self.controller.select_node(node_id)
        self.topology_view.highlight_node(node_id)

    def apply_demo_preset(self, period: int) -> None:
        self.controller.apply_demo_preset(period)
        idx = self.period_combo.findData(period)
        if idx >= 0:
            self.period_combo.blockSignals(True)
            self.period_combo.setCurrentIndex(idx)
            self.period_combo.blockSignals(False)
        self.refresh_view()

    def select_next_alert(self) -> None:
        self.controller.select_next_alert()
        node_id = self.controller.state.selected_node_id
        self.refresh_view()
        self.topology_view.highlight_node(node_id)

    def focus_selected_node(self) -> None:
        self.topology_view.highlight_node(self.controller.state.selected_node_id)

    def toggle_alerts_panel(self) -> None:
        self.alerts_visible = not self.alerts_visible
        self.left_overlay.setVisible(self.alerts_visible)
        self.left_overlay.raise_()

    def toggle_detail_panel(self) -> None:
        self.detail_visible = not self.detail_visible
        self.right_overlay.setVisible(self.detail_visible)
        self.right_overlay.raise_()

    def open_analysis_dialog(self) -> None:
        dialog = AnalysisDialog(self.controller.get_period_stats(), self)
        dialog.exec_()

    def run_realtime_prediction(self) -> None:
        if self.controller.state.selected_node_id is None:
            QMessageBox.information(self, "无节点", "请先选择一个告警节点。")
            return
        self.detail_panel.predict_button.setEnabled(False)
        self.detail_panel.predict_button.setText("推理中...")
        try:
            self.controller.run_realtime_prediction()
            detail = self.controller.get_node_detail()
            record = self.controller.get_handling_record()
            self.detail_panel.set_detail(detail, record, self.controller.state.prediction)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "实时推理失败", str(exc))
        finally:
            self.detail_panel.predict_button.setEnabled(True)
            self.detail_panel.predict_button.setText("实时推理")

    def export_alerts(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出当前时段告警", str(STORAGE_DIR / "alerts.csv"), "CSV Files (*.csv)")
        if path:
            self.controller.export_alerts(path)

    def export_records(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出处置记录", str(STORAGE_DIR / "handling_records.csv"), "CSV Files (*.csv)")
        if path:
            self.controller.export_records(path)

    def export_node_summary(self) -> None:
        if self.controller.state.selected_node_id is None:
            QMessageBox.information(self, "无节点", "请先选择一个告警节点。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出节点摘要",
            str(STORAGE_DIR / f"node_{self.controller.state.selected_node_id}.json"),
            "JSON Files (*.json);;Text Files (*.txt)",
        )
        if path:
            self.controller.export_node_summary(path)
