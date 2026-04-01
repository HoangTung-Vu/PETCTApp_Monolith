"""Popup dialog for per-component threshold adjustment with live preview."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QDoubleSpinBox, QPushButton, QGroupBox, QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


class ThresholdPreviewDialog(QDialog):
    """Modal dialog that shows computed threshold for one ROI component.

    The user can adjust the threshold with a slider and see live preview
    on the current slice. Emitting threshold_changed on slider move.
    """

    threshold_changed = pyqtSignal(float)  # emitted on slider/spin change (debounced)

    def __init__(
        self,
        component_info: dict,
        component_index: int,
        total_components: int,
        method_info: str,
        parent=None,
    ):
        super().__init__(parent)
        self.component_info = component_info
        self.final_threshold = component_info["threshold"]

        self.setWindowTitle(
            f"Threshold — Region {component_index + 1}/{total_components}"
        )
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)
        self._debounce.timeout.connect(self._emit_threshold)

        self._init_ui(component_info, component_index, total_components, method_info)

    def _init_ui(self, info, idx, total, method_info):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"<b>{method_info}</b><br>"
            f"Region {idx + 1} / {total} — {info['n_voxels']:,} voxels"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Stats
        grp = QGroupBox("Computed Statistics")
        stats_layout = QFormLayout()

        if "i_max" in info:
            stats_layout.addRow("I_max (SUV):", QLabel(f"{info['i_max']:.4f}"))
        if "i_mean" in info:
            stats_layout.addRow("I_mean (SUV):", QLabel(f"{info['i_mean']:.4f}"))
        if "i_background" in info:
            stats_layout.addRow("I_background (SUV):", QLabel(f"{info['i_background']:.4f}"))
        if "i_source" in info:
            stats_layout.addRow("I_source (SUV):", QLabel(f"{info['i_source']:.4f}"))
        if "iterations" in info:
            stats_layout.addRow("Iterations:", QLabel(str(info["iterations"])))

        stats_layout.addRow(
            "<b>Computed threshold:</b>",
            QLabel(f"<b>{info['threshold']:.4f} SUV</b>"),
        )
        grp.setLayout(stats_layout)
        layout.addWidget(grp)

        # Threshold slider + spinbox
        thresh_layout = QHBoxLayout()

        self.lbl_thresh = QLabel("Threshold (SUV):")
        thresh_layout.addWidget(self.lbl_thresh)

        # Determine slider range: 0 to component's I_max (or I_source)
        i_peak = info.get("i_max", info.get("i_source", 20.0))
        self._slider_max = max(i_peak * 1.2, info["threshold"] * 2.0, 1.0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)  # map to 0..slider_max via factor
        self._set_slider_from_value(info["threshold"])
        thresh_layout.addWidget(self.slider, stretch=1)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, self._slider_max)
        self.spin.setDecimals(4)
        self.spin.setSingleStep(0.01)
        self.spin.setValue(info["threshold"])
        self.spin.setFixedWidth(100)
        thresh_layout.addWidget(self.spin)

        layout.addLayout(thresh_layout)

        # Sync slider ↔ spin
        self.slider.valueChanged.connect(self._on_slider_moved)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self._syncing = False

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_apply = QPushButton("Apply && Next" if idx < total - 1 else "Apply && Finish")
        self.btn_apply.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 8px;")
        self.btn_apply.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.btn_apply)

        self.btn_reset = QPushButton("Reset to Computed")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_reset)

        layout.addLayout(btn_layout)

    # ── Slider / Spin sync ──

    def _set_slider_from_value(self, value):
        pos = int((value / self._slider_max) * 1000) if self._slider_max > 0 else 0
        self.slider.setValue(max(0, min(1000, pos)))

    def _value_from_slider(self):
        return (self.slider.value() / 1000.0) * self._slider_max

    def _on_slider_moved(self, _pos):
        if self._syncing:
            return
        self._syncing = True
        val = self._value_from_slider()
        self.spin.setValue(val)
        self.final_threshold = val
        self._syncing = False
        self._debounce.start()

    def _on_spin_changed(self, val):
        if self._syncing:
            return
        self._syncing = True
        self._set_slider_from_value(val)
        self.final_threshold = val
        self._syncing = False
        self._debounce.start()

    def _emit_threshold(self):
        self.threshold_changed.emit(self.final_threshold)

    # ── Buttons ──

    def _on_apply(self):
        self.accept()

    def _on_reset(self):
        original = self.component_info["threshold"]
        self.spin.setValue(original)
        self.final_threshold = original
