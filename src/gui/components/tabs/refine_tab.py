"""Refine tab: Manual tools, brush size, target layer, SUV refinement, Adaptive Thresholding."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QSlider, QHBoxLayout, QLabel, QGridLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QProgressBar, QButtonGroup,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal


class RefineTab(QWidget):
    """Manual drawing tools + SUV refinement + Adaptive Thresholding controls."""

    # Signals
    sig_set_tool = pyqtSignal(str)           # 'pan_zoom', 'paint', 'sphere', 'square'
    sig_brush_size_changed = pyqtSignal(int)
    sig_refine_suv_clicked = pyqtSignal(float)  # threshold
    sig_refine_adaptive_clicked = pyqtSignal(float, str, int)  # isocontour_fraction, bg_mode, border_thickness
    sig_refine_iterative_clicked = pyqtSignal(float, float, float, float, int)  # m, c1, c0, convergence_tol, max_iterations
    sig_confirm_roi_clicked = pyqtSignal()
    sig_save_refine_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Create a container widget for the scroll area
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. Tool Selection — 4 buttons: Pan/Zoom, Paint, Sphere, Square
        grp_tools = QGroupBox("Manual Tools")
        tools_layout = QGridLayout()

        self.btn_pan = QPushButton("Pan/Zoom")
        self.btn_paint = QPushButton("Paint")
        self.btn_sphere = QPushButton("Sphere")
        self.btn_square = QPushButton("Square")

        self.btn_pan.setCheckable(True)
        self.btn_paint.setCheckable(True)
        self.btn_sphere.setCheckable(True)
        self.btn_square.setCheckable(True)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_paint)
        self.tool_group.addButton(self.btn_sphere)
        self.tool_group.addButton(self.btn_square)

        self.btn_pan.setChecked(True)

        self.btn_confirm_roi = QPushButton("Confirm ROI")
        self.btn_confirm_roi.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.btn_confirm_roi.clicked.connect(self.sig_confirm_roi_clicked.emit)
        self.btn_confirm_roi.setEnabled(False)

        self.btn_pan.clicked.connect(lambda: self._set_tool_and_enable_confirm("pan_zoom", False))
        self.btn_paint.clicked.connect(lambda: self._set_tool_and_enable_confirm("paint", False))
        self.btn_sphere.clicked.connect(lambda: self._set_tool_and_enable_confirm("sphere", True))
        self.btn_square.clicked.connect(lambda: self._set_tool_and_enable_confirm("square", True))

        tools_layout.addWidget(self.btn_pan, 0, 0)
        tools_layout.addWidget(self.btn_paint, 0, 1)
        tools_layout.addWidget(self.btn_sphere, 1, 0)
        tools_layout.addWidget(self.btn_square, 1, 1)
        tools_layout.addWidget(self.btn_confirm_roi, 2, 0, 1, 2)

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

        layout.addSpacing(5)

        # 4. SUV Refinement
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

        layout.addSpacing(8)

        # 5. Adaptive Thresholding Refinement
        grp_adaptive = QGroupBox("Adaptive Thresholding")
        adaptive_layout = QFormLayout()

        self.spin_isocontour = QDoubleSpinBox()
        self.spin_isocontour.setRange(0.01, 0.99)
        self.spin_isocontour.setValue(0.70)
        self.spin_isocontour.setSingleStep(0.05)
        self.spin_isocontour.setDecimals(2)

        self.combo_bg_mode = QComboBox()
        self.combo_bg_mode.addItems(["outside_isocontour", "border_pixels"])

        self.spin_border_thickness = QSpinBox()
        self.spin_border_thickness.setRange(1, 10)
        self.spin_border_thickness.setValue(3)

        self.btn_refine_adaptive = QPushButton("Refine ROI by Adaptive Threshold")
        self.btn_refine_adaptive.clicked.connect(self._emit_refine_adaptive)

        adaptive_layout.addRow("Isocontour Fraction:", self.spin_isocontour)
        adaptive_layout.addRow("Background Mode:", self.combo_bg_mode)
        adaptive_layout.addRow("Border Thickness:", self.spin_border_thickness)
        adaptive_layout.addRow(self.btn_refine_adaptive)

        grp_adaptive.setLayout(adaptive_layout)
        layout.addWidget(grp_adaptive)

        layout.addSpacing(8)

        # 6. Iterative Thresholding Refinement
        grp_iterative = QGroupBox("Iterative Thresholding")
        iterative_layout = QFormLayout()

        self.spin_m = QDoubleSpinBox()
        self.spin_m.setRange(0.1, 100.0)
        self.spin_m.setValue(7.8)
        self.spin_m.setSingleStep(0.1)

        self.spin_c1 = QDoubleSpinBox()
        self.spin_c1.setRange(0.0, 100.0)
        self.spin_c1.setValue(61.7)
        self.spin_c1.setSingleStep(1.0)

        self.spin_c0 = QDoubleSpinBox()
        self.spin_c0.setRange(0.0, 100.0)
        self.spin_c0.setValue(31.6)
        self.spin_c0.setSingleStep(1.0)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.01, 1.0)
        self.spin_tol.setValue(0.03)
        self.spin_tol.setSingleStep(0.01)
        self.spin_tol.setDecimals(3)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(1, 50)
        self.spin_iters.setValue(10)

        self.btn_refine_iterative = QPushButton("Refine ROI by Iterative Threshold")
        self.btn_refine_iterative.clicked.connect(self._emit_refine_iterative)

        iterative_layout.addRow("m (Slope):", self.spin_m)
        iterative_layout.addRow("c1 (B/S Coeff):", self.spin_c1)
        iterative_layout.addRow("c0 (Intercept):", self.spin_c0)
        iterative_layout.addRow("Tolerance:", self.spin_tol)
        iterative_layout.addRow("Max Iterations:", self.spin_iters)
        iterative_layout.addRow(self.btn_refine_iterative)

        grp_iterative.setLayout(iterative_layout)
        layout.addWidget(grp_iterative)

        layout.addSpacing(8)

        # 7. Save Refinement
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

        # Set the container as the scroll area's widget
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _emit_refine_suv(self):
        val = self.spin_suv.value()
        self.sig_refine_suv_clicked.emit(val)

    def _emit_refine_adaptive(self):
        iso_frac = self.spin_isocontour.value()
        bg_mode = self.combo_bg_mode.currentText()
        border = self.spin_border_thickness.value()
        self.sig_refine_adaptive_clicked.emit(iso_frac, bg_mode, border)

    def _emit_refine_iterative(self):
        m = self.spin_m.value()
        c1 = self.spin_c1.value()
        c0 = self.spin_c0.value()
        tol = self.spin_tol.value()
        iters = self.spin_iters.value()
        self.sig_refine_iterative_clicked.emit(m, c1, c0, tol, iters)

    def _set_tool_and_enable_confirm(self, tool, enabled):
        self.btn_confirm_roi.setEnabled(enabled)
        self.sig_set_tool.emit(tool)

    def reset_tools(self):
        """Reset UI to reflect Pan/Zoom tool state."""
        self.btn_pan.setChecked(True)
        self.btn_confirm_roi.setEnabled(False)
        # BUG-1 FIX: Emit signal so shape drag is disabled and camera is re-enabled
        self.sig_set_tool.emit("pan_zoom")

    def show_refine_progress(self):
        self.refine_progress.setVisible(True)

    def hide_refine_progress(self):
        self.refine_progress.setVisible(False)
