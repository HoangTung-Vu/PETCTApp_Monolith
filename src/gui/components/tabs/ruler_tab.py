"""Ruler tab: distance-measurement toggle + live distance readout."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal


class RulerTab(QWidget):
    """Distance-measurement tool toggle + clear + distance readout."""

    # Signals
    sig_ruler_toggled = pyqtSignal(bool)  # True=enable, False=disable
    sig_ruler_clear = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Info
        info = QLabel(
            "Measure the distance between two points.\n\n"
            "• Click to place the start point, move, then click to place\n"
            "  the end point — the 3-D distance (mm) is shown.\n"
            "• You may scroll to another slice between the two clicks;\n"
            "  the distance accounts for the through-plane offset.\n"
            "• Click empty space to start a new measurement.\n"
            "• Double-click to clear the current measurement.\n\n"
            "Enabling the ruler turns off the crosshair."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addSpacing(10)

        # 2. Enable / Disable toggle
        self.btn_ruler_toggle = QPushButton("Enable Ruler")
        self.btn_ruler_toggle.setCheckable(True)
        self.btn_ruler_toggle.setChecked(False)
        self.btn_ruler_toggle.toggled.connect(self._on_ruler_toggled)
        layout.addWidget(self.btn_ruler_toggle)

        # 3. Distance readout
        self.lbl_distance = QLabel("Distance: — mm")
        self.lbl_distance.setStyleSheet(
            "QLabel { color: #ffff88; font-size: 16px; font-weight: bold;"
            " padding: 8px 4px; }"
        )
        layout.addWidget(self.lbl_distance)

        # 4. Clear
        self.btn_ruler_clear = QPushButton("Clear Measurement")
        self.btn_ruler_clear.clicked.connect(self.sig_ruler_clear.emit)
        layout.addWidget(self.btn_ruler_clear)

        layout.addStretch()

    def _on_ruler_toggled(self, checked: bool):
        self.btn_ruler_toggle.setText("Disable Ruler" if checked else "Enable Ruler")
        self.sig_ruler_toggled.emit(checked)

    def set_distance(self, mm):
        """Update the distance readout. ``mm`` is a float, or None to clear."""
        if mm is None:
            self.lbl_distance.setText("Distance: — mm")
        else:
            self.lbl_distance.setText(f"Distance: {float(mm):.1f} mm")
