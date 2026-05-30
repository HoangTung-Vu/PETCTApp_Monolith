"""View & Display tab: 9-view toggles and display settings."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSlider, QHBoxLayout, QLabel, QScrollArea, QCheckBox,
    QComboBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


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

CT_COLORMAPS  = ["gray", "gray_r", "green", "cyan", "blue", "twilight"]
PET_COLORMAPS = ["jet", "gray", "gray_r", "hot", "inferno", "magma", "plasma", "viridis"]

# View definitions: (view_id, display_label, checked_by_default)
VIEW_DEFS = [
    ("axial_ct",        "Axial — CT",         True),
    ("axial_pet",       "Axial — PET",        True),
    ("axial_overlay",   "Axial — Overlay",    False),
    ("coronal_ct",      "Coronal — CT",       False),
    ("coronal_pet",     "Coronal — PET",      False),
    ("coronal_overlay", "Coronal — Overlay",  False),
    ("sagittal_ct",     "Sagittal — CT",      False),
    ("sagittal_pet",    "Sagittal — PET",     False),
    ("sagittal_overlay","Sagittal — Overlay", False),
]


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
    """9-view toggles + display settings (W/L, opacity, zoom, mask visibility)."""

    # ── Signals ────────────────────────────────────────────────────────────
    # Emits list of currently active view_ids whenever toggles change
    sig_active_views_changed     = pyqtSignal(list)

    # Kept only for 3D view button
    sig_layout_changed           = pyqtSignal(str)
    sig_toggle_3d_pet            = pyqtSignal(bool)

    sig_overlay_pet_opacity_changed = pyqtSignal(float)
    sig_tumor_opacity_changed    = pyqtSignal(float)
    sig_roi_opacity_changed      = pyqtSignal(float)
    sig_ct_window_level_changed  = pyqtSignal(float, float)
    sig_pet_window_level_changed = pyqtSignal(float, float)
    sig_zoom_changed             = pyqtSignal(int)
    sig_zoom_to_fit              = pyqtSignal()
    sig_toggle_mask              = pyqtSignal(str, bool)

    sig_ct_colormap_changed    = pyqtSignal(str)
    sig_pet_colormap_changed   = pyqtSignal(str)
    sig_overlay_pet_colormap_changed = pyqtSignal(str)

    # Emits the effective napari interpolation2d mode string ("nearest" when OFF).
    sig_interpolation_changed  = pyqtSignal(str)

    # napari 0.6.6 Interpolation enum minus "custom" (needs a kernel) and
    # "nearest" (that is the OFF state of the toggle).
    INTERPOLATION_MODES = [
        "linear", "bessel", "blackman", "catrom", "cubic", "gaussian",
        "hamming", "hanning", "hermite", "kaiser", "lanczos", "mitchell",
        "spline16", "spline36",
    ]

    # True = crosshair overlay ON; False = crosshair overlay OFF
    sig_crosshair_toggled      = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view_checkboxes: dict[str, QCheckBox] = {}
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(4)

        # ── View Toggles ───────────────────────────────────────────────────
        view_content = QWidget()
        vc_lay = QVBoxLayout(view_content)
        vc_lay.setContentsMargins(4, 4, 4, 4)
        vc_lay.setSpacing(3)

        for view_id, label, default in VIEW_DEFS:
            chk = QCheckBox(label)
            chk.toggled.connect(self._on_view_toggle_changed)
            chk.setChecked(default)      # fires toggled → initial emit
            vc_lay.addWidget(chk)
            self._view_checkboxes[view_id] = chk

        vc_lay.addWidget(QLabel(""))    # spacer

        self.btn_3d = QPushButton("3D View")
        self.btn_3d.setCheckable(True)
        self.btn_3d.toggled.connect(self._on_btn_3d_toggled)
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

        # ── Cursor & Interaction ───────────────────────────────────────────
        cursor_content = QWidget()
        cc_lay = QVBoxLayout(cursor_content)
        cc_lay.setContentsMargins(4, 4, 4, 4)
        cc_lay.setSpacing(4)

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

        self.btn_interpolation = QPushButton("Interpolation: ON")
        self.btn_interpolation.setCheckable(True)
        self.btn_interpolation.setChecked(True)
        self.btn_interpolation.clicked.connect(self._on_interpolation_toggled)
        cc_lay.addWidget(self.btn_interpolation)

        # Kernel chooser — applied only when interpolation is ON; OFF forces nearest.
        self.combo_interpolation = QComboBox()
        self.combo_interpolation.addItems(self.INTERPOLATION_MODES)
        self.combo_interpolation.setCurrentText("linear")
        self.combo_interpolation.currentTextChanged.connect(self._on_interpolation_toggled)
        cc_lay.addWidget(self.combo_interpolation)

        layout.addWidget(_make_collapsible("Cursor & Interaction", cursor_content))

        # ── CT Display ─────────────────────────────────────────────────────
        ct_content = QWidget()
        ct_lay = QFormLayout(ct_content)
        ct_lay.setContentsMargins(4, 4, 4, 4)
        ct_lay.setSpacing(3)

        self.combo_ct_colormap = QComboBox()
        self.combo_ct_colormap.addItems(CT_COLORMAPS)
        self.combo_ct_colormap.setCurrentText("gray")
        self.combo_ct_colormap.setMaxVisibleItems(15)
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
        self.combo_ct_preset.setMaxVisibleItems(15)
        self.combo_ct_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_ct_preset.currentTextChanged.connect(self._on_ct_preset_changed)
        ct_lay.addRow("Preset:", self.combo_ct_preset)

        self.spin_ct_window = QDoubleSpinBox()
        self.spin_ct_window.setRange(1, 4000)
        self.spin_ct_window.setValue(350)
        ct_lay.addRow("Window:", self.spin_ct_window)

        self.spin_ct_level = QDoubleSpinBox()
        self.spin_ct_level.setRange(-2000, 2000)
        self.spin_ct_level.setValue(35)
        ct_lay.addRow("Level:", self.spin_ct_level)

        self._ct_wl_timer = QTimer(self)
        self._ct_wl_timer.setSingleShot(True)
        self._ct_wl_timer.setInterval(150)
        self._ct_wl_timer.timeout.connect(self._emit_ct_wl)
        self.spin_ct_window.valueChanged.connect(lambda _: self._ct_wl_timer.start())
        self.spin_ct_level.valueChanged.connect(lambda _: self._ct_wl_timer.start())

        layout.addWidget(_make_collapsible("CT Display", ct_content))

        # ── PET Display ────────────────────────────────────────────────────
        pet_content = QWidget()
        pet_lay = QFormLayout(pet_content)
        pet_lay.setContentsMargins(4, 4, 4, 4)
        pet_lay.setSpacing(3)

        self.combo_pet_colormap = QComboBox()
        self.combo_pet_colormap.addItems(PET_COLORMAPS)
        self.combo_pet_colormap.setCurrentText("jet")
        self.combo_pet_colormap.setMaxVisibleItems(15)
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
        self.combo_pet_preset.setMaxVisibleItems(15)
        self.combo_pet_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pet_preset.currentTextChanged.connect(self._on_pet_preset_changed)
        pet_lay.addRow("Preset:", self.combo_pet_preset)

        self.spin_pet_window = QDoubleSpinBox()
        self.spin_pet_window.setRange(0.1, 10000)
        self.spin_pet_window.setValue(10)
        pet_lay.addRow("Window:", self.spin_pet_window)

        self.spin_pet_level = QDoubleSpinBox()
        self.spin_pet_level.setRange(0, 10000)
        self.spin_pet_level.setValue(5)
        pet_lay.addRow("Level:", self.spin_pet_level)

        self._pet_wl_timer = QTimer(self)
        self._pet_wl_timer.setSingleShot(True)
        self._pet_wl_timer.setInterval(150)
        self._pet_wl_timer.timeout.connect(self._emit_pet_wl)
        self.spin_pet_window.valueChanged.connect(lambda _: self._pet_wl_timer.start())
        self.spin_pet_level.valueChanged.connect(lambda _: self._pet_wl_timer.start())

        layout.addWidget(_make_collapsible("PET Display", pet_content))

        # ── Overlay Display ────────────────────────────────────────────────────
        overlay_content = QWidget()
        overlay_lay = QFormLayout(overlay_content)
        overlay_lay.setContentsMargins(4, 4, 4, 4)
        overlay_lay.setSpacing(3)

        self.combo_overlay_colormap = QComboBox()
        self.combo_overlay_colormap.addItems(PET_COLORMAPS)
        self.combo_overlay_colormap.setCurrentText("jet")
        self.combo_overlay_colormap.setMaxVisibleItems(15)
        self.combo_overlay_colormap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_overlay_colormap.currentTextChanged.connect(self.sig_overlay_pet_colormap_changed.emit)
        overlay_lay.addRow("PET Colormap:", self.combo_overlay_colormap)

        self.combo_overlay_preset = QComboBox()
        self.combo_overlay_preset.addItems([
            "Custom",
            "35% - Jet",
            "50% - Jet",
            "65% - Jet",
            "35% - Hot Iron",
            "50% - Hot Iron",
            "65% - Hot Iron"
        ])
        self.combo_overlay_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_overlay_preset.currentTextChanged.connect(self._on_overlay_preset_changed)
        overlay_lay.addRow("Preset:", self.combo_overlay_preset)

        self.slider_overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlay_opacity.setRange(0, 100)
        self.slider_overlay_opacity.setValue(50)
        self.slider_overlay_opacity.valueChanged.connect(
            lambda v: self.sig_overlay_pet_opacity_changed.emit(v / 100.0)
        )
        overlay_lay.addRow("PET Opacity:", self.slider_overlay_opacity)

        layout.addWidget(_make_collapsible("Overlay (Fusion) Display", overlay_content))

        # ── Zoom & Mask ────────────────────────────────────────────────────
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

        self.chk_roi = QCheckBox("Show ROI Mask")
        self.chk_roi.setChecked(True)
        self.chk_roi.toggled.connect(lambda c: self.sig_toggle_mask.emit("roi", c))
        zm_lay.addRow(self.chk_roi)

        self.slider_roi_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_roi_opacity.setRange(0, 100)
        self.slider_roi_opacity.setValue(90)
        self.slider_roi_opacity.valueChanged.connect(
            lambda v: self.sig_roi_opacity_changed.emit(v / 100.0)
        )
        zm_lay.addRow("ROI opacity:", self.slider_roi_opacity)

        layout.addWidget(_make_collapsible("Zoom & Mask", zm_content))

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_view_toggle_changed(self):
        # Toggling any 2D view exits 3D mode visually — uncheck the 3D button
        # without re-emitting (the active views emit below already returns to 2D).
        if hasattr(self, "btn_3d") and self.btn_3d.isChecked():
            self.btn_3d.blockSignals(True)
            self.btn_3d.setChecked(False)
            self.btn_3d.blockSignals(False)
        if not getattr(self, '_pending_view_update', False):
            self._pending_view_update = True
            QTimer.singleShot(0, self._flush_view_toggle)

    def _flush_view_toggle(self):
        self._pending_view_update = False
        active = [vid for vid, chk in self._view_checkboxes.items() if chk.isChecked()]
        self.sig_active_views_changed.emit(active)

    def _on_btn_3d_toggled(self, checked: bool):
        if checked:
            self.sig_layout_changed.emit("3d")
        else:
            # Returning to 2D: re-emit current active views
            active = [vid for vid, chk in self._view_checkboxes.items() if chk.isChecked()]
            self.sig_active_views_changed.emit(active)

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
        self.sig_crosshair_toggled.emit(checked)

    def _on_interpolation_toggled(self, *_):
        # Shared by the on/off button and the kernel combo box; read live state.
        enabled = self.btn_interpolation.isChecked()
        self.btn_interpolation.setText(
            "Interpolation: ON" if enabled else "Interpolation: OFF"
        )
        self.combo_interpolation.setEnabled(enabled)
        mode = self.combo_interpolation.currentText() if enabled else "nearest"
        self.sig_interpolation_changed.emit(mode)

    def _on_overlay_preset_changed(self, text: str):
        if text == "Custom":
            return
        
        self.slider_overlay_opacity.blockSignals(True)
        self.combo_overlay_colormap.blockSignals(True)

        if "35%" in text:
            opacity = 35
        elif "50%" in text:
            opacity = 50
        elif "65%" in text:
            opacity = 65
        else:
            opacity = 50

        if "Jet" in text:
            cmap = "jet"
        elif "Hot Iron" in text:
            cmap = "hot"
        else:
            cmap = "jet"

        self.slider_overlay_opacity.setValue(opacity)
        self.combo_overlay_colormap.setCurrentText(cmap)

        self.slider_overlay_opacity.blockSignals(False)
        self.combo_overlay_colormap.blockSignals(False)

        self.sig_overlay_pet_opacity_changed.emit(opacity / 100.0)
        self.sig_overlay_pet_colormap_changed.emit(cmap)

