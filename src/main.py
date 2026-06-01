import sys
import os
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont
from PyQt6.QtCore import Qt
from .gui.main_window import MainWindow
from .database.db import init_db


def _make_splash_pixmap() -> QPixmap:
    pix = QPixmap(480, 110)
    pix.fill(QColor("#1e1e1e"))
    p = QPainter(pix)
    p.setPen(QColor("#cccccc"))
    p.setFont(QFont("Arial", 14))
    p.drawText(
        pix.rect(),
        Qt.AlignmentFlag.AlignCenter,
        "Metabolic Lesion Quantification\nLoading…",
    )
    p.end()
    return pix


def main():
    # Initialize Database
    init_db()

    # Initialize Application
    app = QApplication(sys.argv)

    # Show splash + spinning cursor while napari viewers initialise (~5 s)
    splash = QSplashScreen(_make_splash_pixmap(), Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.setOverrideCursor(Qt.CursorShape.WaitCursor)
    app.processEvents()

    window = MainWindow()
    window.show()

    app.restoreOverrideCursor()
    splash.finish(window)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
