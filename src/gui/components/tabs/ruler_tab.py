"""Ruler tab: distance-measurement toggle + multi-measurement readout."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget
)
from PyQt6.QtCore import pyqtSignal


class RulerTab(QWidget):
    """Distance-measurement tool toggle + clear + per-measurement readout."""

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
            "• Each completed measurement stays on screen; keep clicking\n"
            "  to add more.\n"
            "• Double-click (or Clear All) to remove every measurement.\n\n"
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

        # 3. Live readout (current in-progress segment)
        self.lbl_live = QLabel("")
        self.lbl_live.setStyleSheet(
            "QLabel { color: #ffff88; font-size: 16px; font-weight: bold;"
            " padding: 8px 4px; }"
        )
        layout.addWidget(self.lbl_live)

        # 4. Completed-measurement list (#1, #2, …)
        layout.addWidget(QLabel("Measurements:"))
        self.list_measurements = QListWidget()
        layout.addWidget(self.list_measurements)

        # 5. Clear all
        self.btn_ruler_clear = QPushButton("Clear All")
        self.btn_ruler_clear.clicked.connect(self.sig_ruler_clear.emit)
        layout.addWidget(self.btn_ruler_clear)

    def _on_ruler_toggled(self, checked: bool):
        self.btn_ruler_toggle.setText("Disable Ruler" if checked else "Enable Ruler")
        self.sig_ruler_toggled.emit(checked)

    def set_measurements(self, completed, active):
        """Refresh the readout.

        ``completed`` is a list of distances (mm) for finished measurements;
        ``active`` is the distance of the in-progress segment, or None.
        """
        self.list_measurements.clear()
        for i, d in enumerate(completed, 1):
            self.list_measurements.addItem(f"#{i}   {float(d):.1f} mm")
        if active is None:
            self.lbl_live.setText("")
        else:
            self.lbl_live.setText(f"Measuring: {float(active):.1f} mm")
