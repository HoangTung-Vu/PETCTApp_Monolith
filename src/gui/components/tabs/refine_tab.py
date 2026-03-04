"""Refine tab: Manual tools, brush size, target layer, SUV refinement."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QSlider, QHBoxLayout, QLabel, QGridLayout, QComboBox,
    QDoubleSpinBox, QProgressBar, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal


class RefineTab(QWidget):
    """Manual drawing tools + SUV refinement controls."""

    # Signals
    sig_set_tool = pyqtSignal(str)           # 'pan_zoom', 'paint', 'erase'
    sig_brush_size_changed = pyqtSignal(int)
    sig_target_layer_changed = pyqtSignal(str)  # 'tumor', 'organ'
    sig_refine_suv_clicked = pyqtSignal(float)  # threshold
    sig_save_refine_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Tool Selection
        grp_tools = QGroupBox("Manual Tools")
        tools_layout = QGridLayout()

        self.btn_pan = QPushButton("Pan/Zoom")
        self.btn_paint = QPushButton("Paint")

        self.btn_pan.setCheckable(True)
        self.btn_paint.setCheckable(True)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_paint)

        self.btn_pan.setChecked(True)

        self.btn_pan.clicked.connect(lambda: self.sig_set_tool.emit("pan_zoom"))
        self.btn_paint.clicked.connect(lambda: self.sig_set_tool.emit("paint"))

        tools_layout.addWidget(self.btn_pan, 0, 0)
        tools_layout.addWidget(self.btn_paint, 0, 1)

        grp_tools.setLayout(tools_layout)
        layout.addWidget(grp_tools)

        # 2. Brush Size
        grp_brush = QGroupBox("Brush Size")
        brush_layout = QHBoxLayout()
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 50)
        self.slider_brush.setValue(10)

        self.lbl_brush_size = QLabel("10")
        self.lbl_brush_size.setFixedWidth(30)

        self.slider_brush.valueChanged.connect(self.sig_brush_size_changed.emit)
        self.slider_brush.valueChanged.connect(lambda v: self.lbl_brush_size.setText(str(v)))

        brush_layout.addWidget(self.slider_brush)
        brush_layout.addWidget(self.lbl_brush_size)
        grp_brush.setLayout(brush_layout)
        layout.addWidget(grp_brush)

        # 3. Target Layer
        grp_layer = QGroupBox("Target Layer")
        layer_layout = QHBoxLayout()
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(["Tumor Mask", "Organ Mask"])
        self.combo_layer.currentTextChanged.connect(
            lambda t: self.sig_target_layer_changed.emit("tumor" if "Tumor" in t else "organ")
        )
        layer_layout.addWidget(self.combo_layer)
        grp_layer.setLayout(layer_layout)
        layout.addWidget(grp_layer)

        layout.addSpacing(10)

        layout.addSpacing(10)

        # 5. SUV Refinement
        grp_suv = QGroupBox("SUV Refinement")
        suv_layout = QFormLayout()

        self.spin_suv = QDoubleSpinBox()
        self.spin_suv.setRange(0.0, 50.0)
        self.spin_suv.setValue(2.5)
        self.spin_suv.setSingleStep(0.1)

        self.btn_refine = QPushButton("Refine ROI by SUV")
        self.btn_refine.clicked.connect(self._emit_refine_suv)

        suv_layout.addRow("Min SUV:", self.spin_suv)
        suv_layout.addRow(self.btn_refine)

        grp_suv.setLayout(suv_layout)
        layout.addWidget(grp_suv)

        # 6. Save Refinement
        self.btn_save_refine = QPushButton("Save Refinement (Overwrite)")
        self.btn_save_refine.clicked.connect(self.sig_save_refine_clicked.emit)
        self.btn_save_refine.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        layout.addWidget(self.btn_save_refine)

        # 7. Refinement Progress Bar
        self.refine_progress = QProgressBar()
        self.refine_progress.setRange(0, 0)  # Indeterminate
        self.refine_progress.setVisible(False)
        layout.addWidget(self.refine_progress)

        layout.addStretch()

    def _emit_refine_suv(self):
        val = self.spin_suv.value()
        self.sig_refine_suv_clicked.emit(val)

    def show_refine_progress(self):
        self.refine_progress.setVisible(True)

    def hide_refine_progress(self):
        self.refine_progress.setVisible(False)
