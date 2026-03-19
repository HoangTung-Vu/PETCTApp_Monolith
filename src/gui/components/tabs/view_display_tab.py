"""View & Display tab: view mode switching and display settings."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSlider, QHBoxLayout, QLabel, QScrollArea, QCheckBox,
    QComboBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal


# ── Preset W/L tables ──────────────────────────────────────────────────────
CT_PRESETS = {
    "Custom":        None,
    "Soft Tissue":   (400,   40),
    "Brain":         (80,    40),
    "Liver":         (160,   60),
    "Abdomen":       (400,   40),
    "Lung":          (1600, -600),
    "Bone":          (1500,  300),
    "Mediastinum":   (350,   25),
}
CT_PRESET_TIPS = {
    "Soft Tissue":  "W 400 / L 40",
    "Brain":        "W 80 / L 40",
    "Liver":        "W 160 / L 60",
    "Abdomen":      "W 400 / L 40",
    "Lung":         "W 1600 / L -600",
    "Bone":         "W 1500 / L 300",
    "Mediastinum":  "W 350 / L 25",
}

PET_PRESETS = {
    "Custom":    None,
    "Standard":  (10,   5.0),
    "High":      (20,  10.0),
    "Low":       (5,    2.5),
}
PET_PRESET_TIPS = {
    "Standard":  "W 10 / L 5",
    "High":      "W 20 / L 10",
    "Low":       "W 5 / L 2.5",
}

CT_COLORMAPS  = ["gray", "green", "cyan", "blue", "twilight"]
PET_COLORMAPS = ["jet", "gray", "hot", "inferno", "magma", "plasma", "viridis"]


def _make_collapsible(title: str, content_widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 2)
    layout.setSpacing(0)

    toggle = QPushButton(f"▼  {title}")
    toggle.setCheckable(True)
    toggle.setChecked(True)
    toggle.setStyleSheet(
        "QPushButton { text-align: left; font-weight: bold; "
        "background: #2d2d2d; border: none; padding: 4px 6px; }"
        "QPushButton:checked { background: #3a3a3a; }"
    )
    toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _on_toggle(checked):
        content_widget.setVisible(checked)
        toggle.setText(f"{'▼' if checked else '▶'}  {title}")

    toggle.clicked.connect(_on_toggle)
    layout.addWidget(toggle)
    layout.addWidget(content_widget)
    return container


class ViewDisplayTab(QWidget):
    """View mode buttons + display settings (W/L, opacity, zoom, mask visibility)."""

    # ── Signals ────────────────────────────────────────────────────────────
    sig_layout_changed           = pyqtSignal(str)
    sig_toggle_3d_pet            = pyqtSignal(bool)
    sig_pet_opacity_changed      = pyqtSignal(float)
    sig_tumor_opacity_changed    = pyqtSignal(float)
    sig_ct_window_level_changed  = pyqtSignal(float, float)
    sig_pet_window_level_changed = pyqtSignal(float, float)
    sig_zoom_changed             = pyqtSignal(int)
    sig_zoom_to_fit              = pyqtSignal()
    sig_toggle_mask              = pyqtSignal(str, bool)

    sig_ct_colormap_changed    = pyqtSignal(str)
    sig_pet_colormap_changed   = pyqtSignal(str)

    # True = pan mode ON (crosshair OFF); False = crosshair ON (default)
    sig_pan_mode_toggled       = pyqtSignal(bool)
    sig_interpolation_toggled  = pyqtSignal(bool)

    # True = crosshair overlay ON; False = crosshair overlay OFF (small cross cursor)
    sig_crosshair_toggled      = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(4)

        # ── View Mode ──────────────────────────────────────────────────────
        view_content = QWidget()
        vc_lay = QVBoxLayout(view_content)
        vc_lay.setContentsMargins(4, 4, 4, 4)
        vc_lay.setSpacing(3)

        vc_lay.addWidget(QLabel("Orthogonal (CT + PET):"))
        for label, mode in [("Axial", "mono_axial"), ("Sagittal", "mono_sagittal"), ("Coronal", "mono_coronal")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=mode: self.sig_layout_changed.emit(m))
            vc_lay.addWidget(btn)

        self.btn_grid = QPushButton("Grid (6-Cell)")
        self.btn_grid.clicked.connect(lambda: self.sig_layout_changed.emit("grid"))
        vc_lay.addWidget(self.btn_grid)

        vc_lay.addWidget(QLabel("Mono-modality (3 views):"))
        btn_mono_ct = QPushButton("CT Only")
        btn_mono_ct.clicked.connect(lambda: self.sig_layout_changed.emit("mono_single_ct"))
        vc_lay.addWidget(btn_mono_ct)
        btn_mono_pet = QPushButton("PET Only")
        btn_mono_pet.clicked.connect(lambda: self.sig_layout_changed.emit("mono_single_pet"))
        vc_lay.addWidget(btn_mono_pet)

        vc_lay.addWidget(QLabel("Overlay (fused):"))
        for label, mode in [("Axial", "overlay_axial"), ("Sagittal", "overlay_sagittal"), ("Coronal", "overlay_coronal")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=mode: self.sig_layout_changed.emit(m))
            vc_lay.addWidget(btn)

        self.btn_3d = QPushButton("3D View")
        self.btn_3d.clicked.connect(lambda: self.sig_layout_changed.emit("3d"))
        vc_lay.addWidget(self.btn_3d)

        self.chk_3d_pet = QPushButton("3D: CT View")
        self.chk_3d_pet.setCheckable(True)
        self.chk_3d_pet.setChecked(False)

        def _on_3d_toggle(checked):
            self.chk_3d_pet.setText("3D: PET View" if checked else "3D: CT View")
            self.sig_toggle_3d_pet.emit(checked)

        self.chk_3d_pet.clicked.connect(_on_3d_toggle)
        vc_lay.addWidget(self.chk_3d_pet)

        layout.addWidget(_make_collapsible("View Mode", view_content))

        # ── Cursor & Interaction ──────────────────────────────────────────
        cursor_content = QWidget()
        cc_lay = QVBoxLayout(cursor_content)
        cc_lay.setContentsMargins(4, 4, 4, 4)
        cc_lay.setSpacing(4)

        # Crosshair toggle (main toggle: overlay ON/OFF)
        self.btn_crosshair = QPushButton("Crosshair: ON")
        self.btn_crosshair.setCheckable(True)
        self.btn_crosshair.setChecked(True)
        self.btn_crosshair.setStyleSheet(
            "QPushButton { background: #1a4a1a; }"
            "QPushButton:checked { background: #1a4a1a; }"
            "QPushButton:!checked { background: #4a1a1a; }"
        )
        self.btn_crosshair.clicked.connect(self._on_crosshair_toggled)
        cc_lay.addWidget(self.btn_crosshair)

        # Pan mode toggle (separate from crosshair toggle)
        self.btn_pan_mode = QPushButton("Switch to Pan Mode")
        self.btn_pan_mode.setCheckable(True)
        self.btn_pan_mode.setChecked(False)
        self.btn_pan_mode.clicked.connect(self._on_pan_mode_toggled)
        cc_lay.addWidget(self.btn_pan_mode)

        self.btn_interpolation = QPushButton("Smooth Interpolation: OFF")
        self.btn_interpolation.setCheckable(True)
        self.btn_interpolation.setChecked(False)
        self.btn_interpolation.clicked.connect(self._on_interpolation_toggled)
        cc_lay.addWidget(self.btn_interpolation)

        layout.addWidget(_make_collapsible("Cursor & Interaction", cursor_content))

        # ── CT Display ──────────────────────────────────────────────────────
        ct_content = QWidget()
        ct_lay = QFormLayout(ct_content)
        ct_lay.setContentsMargins(4, 4, 4, 4)
        ct_lay.setSpacing(3)

        self.combo_ct_colormap = QComboBox()
        self.combo_ct_colormap.addItems(CT_COLORMAPS)
        self.combo_ct_colormap.setCurrentText("gray")
        self.combo_ct_colormap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_ct_colormap.currentTextChanged.connect(self.sig_ct_colormap_changed.emit)
        ct_lay.addRow("Colormap:", self.combo_ct_colormap)

        self.combo_ct_preset = QComboBox()
        for name, val in CT_PRESETS.items():
            self.combo_ct_preset.addItem(name)
            tip = CT_PRESET_TIPS.get(name, "")
            if tip:
                self.combo_ct_preset.setItemData(
                    self.combo_ct_preset.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                )
        self.combo_ct_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_ct_preset.textActivated.connect(self._on_ct_preset_changed)
        ct_lay.addRow("Preset:", self.combo_ct_preset)

        self.spin_ct_window = QDoubleSpinBox()
        self.spin_ct_window.setRange(1, 4000)
        self.spin_ct_window.setValue(350)
        self.spin_ct_window.valueChanged.connect(self._emit_ct_wl)
        ct_lay.addRow("Window:", self.spin_ct_window)

        self.spin_ct_level = QDoubleSpinBox()
        self.spin_ct_level.setRange(-2000, 2000)
        self.spin_ct_level.setValue(35)
        self.spin_ct_level.valueChanged.connect(self._emit_ct_wl)
        ct_lay.addRow("Level:", self.spin_ct_level)

        layout.addWidget(_make_collapsible("CT Display", ct_content))

        # ── PET Display ──────────────────────────────────────────────────────
        pet_content = QWidget()
        pet_lay = QFormLayout(pet_content)
        pet_lay.setContentsMargins(4, 4, 4, 4)
        pet_lay.setSpacing(3)

        self.combo_pet_colormap = QComboBox()
        self.combo_pet_colormap.addItems(PET_COLORMAPS)
        self.combo_pet_colormap.setCurrentText("jet")
        self.combo_pet_colormap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pet_colormap.currentTextChanged.connect(self.sig_pet_colormap_changed.emit)
        pet_lay.addRow("Colormap:", self.combo_pet_colormap)

        self.combo_pet_preset = QComboBox()
        for name, val in PET_PRESETS.items():
            self.combo_pet_preset.addItem(name)
            tip = PET_PRESET_TIPS.get(name, "")
            if tip:
                self.combo_pet_preset.setItemData(
                    self.combo_pet_preset.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                )
        self.combo_pet_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pet_preset.textActivated.connect(self._on_pet_preset_changed)
        pet_lay.addRow("Preset:", self.combo_pet_preset)

        self.spin_pet_window = QDoubleSpinBox()
        self.spin_pet_window.setRange(0.1, 10000)
        self.spin_pet_window.setValue(10)
        self.spin_pet_window.valueChanged.connect(self._emit_pet_wl)
        pet_lay.addRow("Window:", self.spin_pet_window)

        self.spin_pet_level = QDoubleSpinBox()
        self.spin_pet_level.setRange(0, 10000)
        self.spin_pet_level.setValue(5)
        self.spin_pet_level.valueChanged.connect(self._emit_pet_wl)
        pet_lay.addRow("Level:", self.spin_pet_level)

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        self.slider_opacity.valueChanged.connect(
            lambda v: self.sig_pet_opacity_changed.emit(v / 100.0)
        )
        pet_lay.addRow("Opacity:", self.slider_opacity)

        layout.addWidget(_make_collapsible("PET Display", pet_content))

        # ── Zoom & Mask ──────────────────────────────────────────────────────
        zm_content = QWidget()
        zm_lay = QFormLayout(zm_content)
        zm_lay.setContentsMargins(4, 4, 4, 4)
        zm_lay.setSpacing(3)

        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(0, 100)
        self.slider_zoom.setValue(20)
        self.slider_zoom.valueChanged.connect(self.sig_zoom_changed.emit)
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setFixedWidth(40)
        self.btn_zoom_fit.clicked.connect(self.sig_zoom_to_fit.emit)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.slider_zoom)
        zoom_row.addWidget(self.btn_zoom_fit)
        zm_lay.addRow("Zoom:", zoom_row)

        self.slider_tumor_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_tumor_opacity.setRange(0, 100)
        self.slider_tumor_opacity.setValue(70)
        self.slider_tumor_opacity.valueChanged.connect(
            lambda v: self.sig_tumor_opacity_changed.emit(v / 100.0)
        )
        zm_lay.addRow("Mask opacity:", self.slider_tumor_opacity)

        self.chk_tumor = QCheckBox("Show Tumor Mask")
        self.chk_tumor.setChecked(True)
        self.chk_tumor.toggled.connect(lambda c: self.sig_toggle_mask.emit("tumor", c))
        zm_lay.addRow(self.chk_tumor)

        layout.addWidget(_make_collapsible("Zoom & Mask", zm_content))

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _emit_ct_wl(self):
        self.combo_ct_preset.blockSignals(True)
        self.combo_ct_preset.setCurrentText("Custom")
        self.combo_ct_preset.blockSignals(False)
        self.sig_ct_window_level_changed.emit(
            float(self.spin_ct_window.value()),
            float(self.spin_ct_level.value())
        )

    def _emit_pet_wl(self):
        self.combo_pet_preset.blockSignals(True)
        self.combo_pet_preset.setCurrentText("Custom")
        self.combo_pet_preset.blockSignals(False)
        self.sig_pet_window_level_changed.emit(
            float(self.spin_pet_window.value()),
            float(self.spin_pet_level.value())
        )

    def _on_ct_preset_changed(self, name: str):
        preset = CT_PRESETS.get(name)
        if preset is None:
            return
        window, level = preset
        self.spin_ct_window.blockSignals(True)
        self.spin_ct_level.blockSignals(True)
        self.spin_ct_window.setValue(window)
        self.spin_ct_level.setValue(level)
        self.spin_ct_window.blockSignals(False)
        self.spin_ct_level.blockSignals(False)
        self.sig_ct_window_level_changed.emit(float(window), float(level))

    def _on_pet_preset_changed(self, name: str):
        preset = PET_PRESETS.get(name)
        if preset is None:
            return
        window, level = preset
        self.spin_pet_window.blockSignals(True)
        self.spin_pet_level.blockSignals(True)
        self.spin_pet_window.setValue(window)
        self.spin_pet_level.setValue(level)
        self.spin_pet_window.blockSignals(False)
        self.spin_pet_level.blockSignals(False)
        self.sig_pet_window_level_changed.emit(float(window), float(level))

    def _on_crosshair_toggled(self, checked: bool):
        self.btn_crosshair.setText("Crosshair: ON" if checked else "Crosshair: OFF")
        # If enabling crosshair, ensure pan mode is off
        if checked and self.btn_pan_mode.isChecked():
            self.btn_pan_mode.setChecked(False)
            self.btn_pan_mode.setText("Switch to Pan Mode")
            self.sig_pan_mode_toggled.emit(False)
        self.sig_crosshair_toggled.emit(checked)

    def _on_pan_mode_toggled(self, checked: bool):
        if checked:
            self.btn_pan_mode.setText("Switch to Crosshair Mode")
            # Pan mode disables crosshair overlay
            if self.btn_crosshair.isChecked():
                self.btn_crosshair.setChecked(False)
                self.btn_crosshair.setText("Crosshair: OFF")
                self.sig_crosshair_toggled.emit(False)
        else:
            self.btn_pan_mode.setText("Switch to Pan Mode")
            # Re-enable crosshair when exiting pan mode
            if not self.btn_crosshair.isChecked():
                self.btn_crosshair.setChecked(True)
                self.btn_crosshair.setText("Crosshair: ON")
                self.sig_crosshair_toggled.emit(True)
        self.sig_pan_mode_toggled.emit(checked)

    def _on_interpolation_toggled(self, checked: bool):
        self.btn_interpolation.setText(
            "Smooth Interpolation: ON" if checked else "Smooth Interpolation: OFF"
        )
        self.sig_interpolation_toggled.emit(checked)

