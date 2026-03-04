"""Eraser tab: Connected-component eraser toggle, undo, save."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal


class EraserTab(QWidget):
    """Eraser tool toggle + undo + save controls."""

    # Signals
    sig_eraser_mode_toggled = pyqtSignal(bool)  # True=enable, False=disable
    sig_eraser_undo_clicked = pyqtSignal()
    sig_eraser_save_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Info
        info = QLabel(
            "Click on a false-positive region to remove\n"
            "the entire connected component (contour)."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addSpacing(10)

        # 2. Enable / Disable toggle
        self.btn_eraser_toggle = QPushButton("Enable Eraser")
        self.btn_eraser_toggle.setCheckable(True)
        self.btn_eraser_toggle.setChecked(False)
        self.btn_eraser_toggle.toggled.connect(self._on_eraser_toggled)
        layout.addWidget(self.btn_eraser_toggle)

        # 3. Undo Last
        self.btn_eraser_undo = QPushButton("Undo Last Erase")
        self.btn_eraser_undo.clicked.connect(self.sig_eraser_undo_clicked.emit)
        layout.addWidget(self.btn_eraser_undo)

        layout.addSpacing(10)

        # 4. Save (Overwrite)
        self.btn_eraser_save = QPushButton("Save Erased Mask (Overwrite)")
        self.btn_eraser_save.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold;"
        )
        self.btn_eraser_save.clicked.connect(self.sig_eraser_save_clicked.emit)
        layout.addWidget(self.btn_eraser_save)

        layout.addStretch()

    def _on_eraser_toggled(self, checked: bool):
        self.btn_eraser_toggle.setText("Disable Eraser" if checked else "Enable Eraser")
        self.sig_eraser_mode_toggled.emit(checked)
