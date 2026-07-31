"""PyQt5 desktop entrypoint for the Q-GAD offline demo."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMessageBox

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from desktop.runtime import get_desktop_root, get_project_root

ROOT = get_project_root()
DESKTOP_ROOT = get_desktop_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop.app.main_window import MainWindow


def _load_stylesheet() -> str:
    style_path = DESKTOP_ROOT / "assets" / "style.qss"
    return style_path.read_text(encoding="utf-8") if style_path.exists() else ""


def _install_exception_hook(app: QApplication) -> None:
    def handle_exception(exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Q-GAD 桌面端错误")
        box.setText("客户端发生未处理异常。")
        box.setDetailedText(text)
        box.exec_()

    sys.excepthook = handle_exception


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Q-GAD Desktop")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    icon_path = DESKTOP_ROOT / "assets" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyleSheet(_load_stylesheet())
    _install_exception_hook(app)

    try:
        window = MainWindow(project_root=ROOT)
    except Exception as exc:  # noqa: BLE001
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Q-GAD 桌面端启动失败")
        box.setText("无法启动桌面客户端。")
        box.setInformativeText(str(exc))
        box.exec_()
        return 1
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
