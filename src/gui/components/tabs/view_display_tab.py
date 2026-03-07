"""View & Display tab: view mode switching and display settings."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSlider, QHBoxLayout, QLabel, QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ViewDisplayTab(QWidget):
    """View mode buttons + display settings (W/L, opacity, zoom, mask visibility)."""

    # Signals
    sig_layout_changed = pyqtSignal(str)
    sig_toggle_3d_pet = pyqtSignal(bool)
    sig_pet_opacity_changed = pyqtSignal(float)
    sig_tumor_opacity_changed = pyqtSignal(float)
    sig_ct_window_level_changed = pyqtSignal(float, float)
    sig_pet_window_level_changed = pyqtSignal(float, float)
    sig_zoom_changed = pyqtSignal(int)
    sig_zoom_to_fit = pyqtSignal()
    sig_toggle_mask = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        # Use a scroll area so the combined content fits in the sidebar
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # ── View Mode ──
        grp_view = QGroupBox("View Mode")
        view_layout = QVBoxLayout()

        # Mono Section
        lbl_mono = QLabel("Mono-orthogonal:")
        view_layout.addWidget(lbl_mono)

        self.btn_mono_axial = QPushButton("Axial")
        self.btn_mono_axial.clicked.connect(lambda: self.sig_layout_changed.emit("mono_axial"))
        view_layout.addWidget(self.btn_mono_axial)

        self.btn_mono_sag = QPushButton("Sagittal")
        self.btn_mono_sag.clicked.connect(lambda: self.sig_layout_changed.emit("mono_sagittal"))
        view_layout.addWidget(self.btn_mono_sag)

        self.btn_mono_cor = QPushButton("Coronal")
        self.btn_mono_cor.clicked.connect(lambda: self.sig_layout_changed.emit("mono_coronal"))
        view_layout.addWidget(self.btn_mono_cor)

        # Grid Button
        self.btn_grid = QPushButton("Grid View (6-Cell)")
        self.btn_grid.clicked.connect(lambda: self.sig_layout_changed.emit("grid"))
        view_layout.addWidget(self.btn_grid)

        # Overlay Section
        lbl_overlay = QLabel("Overlay Mode:")
        view_layout.addWidget(lbl_overlay)

        self.btn_overlay = QPushButton("Axial")
        self.btn_overlay.clicked.connect(lambda: self.sig_layout_changed.emit("overlay_axial"))
        view_layout.addWidget(self.btn_overlay)

        self.btn_overlay_sag = QPushButton("Sagittal")
        self.btn_overlay_sag.clicked.connect(lambda: self.sig_layout_changed.emit("overlay_sagittal"))
        view_layout.addWidget(self.btn_overlay_sag)

        self.btn_overlay_cor = QPushButton("Coronal")
        self.btn_overlay_cor.clicked.connect(lambda: self.sig_layout_changed.emit("overlay_coronal"))
        view_layout.addWidget(self.btn_overlay_cor)

        # 3D Section
        lbl_3d = QLabel("3D Mode:")
        view_layout.addWidget(lbl_3d)

        self.btn_3d = QPushButton("3D View")
        self.btn_3d.clicked.connect(lambda: self.sig_layout_changed.emit("3d"))
        view_layout.addWidget(self.btn_3d)

        self.chk_3d_pet = QPushButton("Toggle 3D PET")
        self.chk_3d_pet.setCheckable(True)
        self.chk_3d_pet.setChecked(True)
        self.chk_3d_pet.clicked.connect(lambda checked: self.sig_toggle_3d_pet.emit(checked))
        view_layout.addWidget(self.chk_3d_pet)

        grp_view.setLayout(view_layout)
        layout.addWidget(grp_view)

        # ── Display Settings ──
        grp_display = QGroupBox("Display Settings")
        display_layout = QFormLayout()

        # CT Window/Level
        self.spin_ct_window = QDoubleSpinBox()
        self.spin_ct_window.setRange(1, 4000)
        self.spin_ct_window.setValue(350)
        self.spin_ct_window.valueChanged.connect(self._emit_ct_wl)

        self.spin_ct_level = QDoubleSpinBox()
        self.spin_ct_level.setRange(-2000, 2000)
        self.spin_ct_level.setValue(35)
        self.spin_ct_level.valueChanged.connect(self._emit_ct_wl)

        display_layout.addRow("CT Window:", self.spin_ct_window)
        display_layout.addRow("CT Level:", self.spin_ct_level)

        # PET Window/Level
        self.spin_pet_window = QDoubleSpinBox()
        self.spin_pet_window.setRange(0.1, 10000)
        self.spin_pet_window.setValue(10)
        self.spin_pet_window.valueChanged.connect(self._emit_pet_wl)

        self.spin_pet_level = QDoubleSpinBox()
        self.spin_pet_level.setRange(0, 10000)
        self.spin_pet_level.setValue(5)
        self.spin_pet_level.valueChanged.connect(self._emit_pet_wl)

        display_layout.addRow("PET Window:", self.spin_pet_window)
        display_layout.addRow("PET Level:", self.spin_pet_level)

        # Zoom
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(0, 100)
        self.slider_zoom.setValue(20)
        self.slider_zoom.valueChanged.connect(self.sig_zoom_changed.emit)

        self.btn_zoom_fit = QPushButton("Zoom to Fit")
        self.btn_zoom_fit.clicked.connect(self.sig_zoom_to_fit.emit)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.slider_zoom)
        zoom_layout.addWidget(self.btn_zoom_fit)

        display_layout.addRow("Zoom:", zoom_layout)

        # PET Opacity
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        self.slider_opacity.valueChanged.connect(
            lambda v: self.sig_pet_opacity_changed.emit(v / 100.0)
        )
        display_layout.addRow("PET Overlay Opacity:", self.slider_opacity)

        # Tumor Mask Opacity
        self.slider_tumor_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_tumor_opacity.setRange(0, 100)
        self.slider_tumor_opacity.setValue(70)
        self.slider_tumor_opacity.valueChanged.connect(
            lambda v: self.sig_tumor_opacity_changed.emit(v / 100.0)
        )
        display_layout.addRow("Tumor Mask Opacity:", self.slider_tumor_opacity)

        grp_display.setLayout(display_layout)
        layout.addWidget(grp_display)

        # ── Segmentation Visibility ──
        grp_seg_disp = QGroupBox("Segmentation Visibility")
        seg_disp_layout = QVBoxLayout()

        self.chk_tumor = QCheckBox("Show Tumor Mask")
        self.chk_tumor.setChecked(True)
        self.chk_tumor.toggled.connect(lambda c: self.sig_toggle_mask.emit("tumor", c))

        seg_disp_layout.addWidget(self.chk_tumor)
        grp_seg_disp.setLayout(seg_disp_layout)
        layout.addWidget(grp_seg_disp)

        layout.addStretch()

        scroll.setWidget(inner)

        # Wrap scroll in the outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Window/Level emitters ──

    def _emit_ct_wl(self):
        self.sig_ct_window_level_changed.emit(
            float(self.spin_ct_window.value()),
            float(self.spin_ct_level.value())
        )

    def _emit_pet_wl(self):
        self.sig_pet_window_level_changed.emit(
            float(self.spin_pet_window.value()),
            float(self.spin_pet_level.value())
        )
