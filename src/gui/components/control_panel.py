from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTabWidget, 
    QLabel, QSlider, QFormLayout, QGroupBox, QComboBox, QDoubleSpinBox,
    QProgressBar, QHBoxLayout, QGridLayout, QListWidget
)
from PyQt6.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    """
    Control Panel with Workflow and Display tabs.
    """
    # Signals
    sig_load_ct_clicked = pyqtSignal()
    sig_load_pet_clicked = pyqtSignal()
    sig_segment_clicked = pyqtSignal()

    sig_layout_changed = pyqtSignal(str)
    sig_toggle_3d_pet = pyqtSignal(bool)
    
    # Display Settings Signals
    sig_pet_opacity_changed = pyqtSignal(float)
    sig_ct_window_level_changed = pyqtSignal(float, float) # window, level
    sig_pet_window_level_changed = pyqtSignal(float, float) # window, level
    sig_zoom_changed = pyqtSignal(int)
    sig_toggle_mask = pyqtSignal(str, bool)
    sig_zoom_to_fit = pyqtSignal()
    
    # Session Signals
    sig_new_session_clicked = pyqtSignal(str, str) # doctor, patient
    sig_load_session_clicked = pyqtSignal(int)     # session_id

    # Refinement Signals
    sig_set_tool = pyqtSignal(str) # 'pan_zoom', 'paint', 'erase'
    sig_brush_size_changed = pyqtSignal(int)
    sig_refine_suv_clicked = pyqtSignal(float) # threshold
    sig_sync_masks_clicked = pyqtSignal()
    sig_save_refine_clicked = pyqtSignal()
    sig_target_layer_changed = pyqtSignal(str) # 'tumor', 'organ'
    
    # AutoPET Interactive Signals
    sig_autopet_click_mode_changed = pyqtSignal(str)  # 'tumor' or 'background'
    sig_autopet_run_clicked = pyqtSignal()
    sig_autopet_save_clicked = pyqtSignal()
    sig_autopet_sync_clicked = pyqtSignal()
    sig_autopet_clear_clicks = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.tabs = QTabWidget()
        
        self._init_workflow_tab()

        self._init_view_tab()
        self._init_refine_tab()
        self._init_autopet_tab()
        self._init_display_tab()
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        
    def _init_workflow_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Session Management
        grp_session = QGroupBox("Session")
        session_layout = QFormLayout()
        
        from PyQt6.QtWidgets import QLineEdit
        self.input_doctor = QLineEdit()
        self.input_patient = QLineEdit()
        session_layout.addRow("Doctor:", self.input_doctor)
        session_layout.addRow("Patient:", self.input_patient)
        
        self.btn_new_session = QPushButton("New Session")
        self.btn_new_session.clicked.connect(self._emit_new_session)
        session_layout.addRow(self.btn_new_session)
        
        self.combo_sessions = QComboBox()
        # self.combo_sessions.addItem("Select Session...") # Populate later
        
        self.btn_load_this_session = QPushButton("Load Selected")
        self.btn_load_this_session.clicked.connect(self._emit_load_session)
        
        session_layout.addRow("Previous:", self.combo_sessions)
        session_layout.addRow(self.btn_load_this_session)
        
        grp_session.setLayout(session_layout)
        layout.addWidget(grp_session)
        
        # Action Buttons
        grp_actions = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        self.btn_load_ct = QPushButton("Load CT")
        self.btn_load_ct.clicked.connect(self.sig_load_ct_clicked.emit)
        
        self.btn_load_pet = QPushButton("Load PET")
        self.btn_load_pet.clicked.connect(self.sig_load_pet_clicked.emit)
        
        self.btn_segment = QPushButton("Run Segmentation")
        self.btn_segment.clicked.connect(self.sig_segment_clicked.emit)
        

        
        action_layout.addWidget(self.btn_load_ct)
        action_layout.addWidget(self.btn_load_pet)
        action_layout.addWidget(self.btn_segment)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        
        grp_actions.setLayout(action_layout)
        
        layout.addWidget(grp_actions)
        
        # Segmentation Toggles
        from PyQt6.QtWidgets import QCheckBox
        grp_seg_disp = QGroupBox("Segmentation Visibility")
        seg_disp_layout = QVBoxLayout()
        
        self.chk_tumor = QCheckBox("Show Tumor Mask")
        self.chk_tumor.setChecked(True)
        self.chk_tumor.toggled.connect(lambda c: self.sig_toggle_mask.emit("tumor", c))
        
        self.chk_body = QCheckBox("Show Body Mask")
        self.chk_body.setChecked(True)
        self.chk_body.toggled.connect(lambda c: self.sig_toggle_mask.emit("body", c))
        
        seg_disp_layout.addWidget(self.chk_tumor)
        seg_disp_layout.addWidget(self.chk_body)
        grp_seg_disp.setLayout(seg_disp_layout)
        layout.addWidget(grp_seg_disp)

        layout.addStretch()
        
        self.tabs.addTab(tab, "Workflow")

    def _init_view_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # View Selection
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
        self.chk_3d_pet.clicked.connect(self._emit_3d_pet_toggle)
        view_layout.addWidget(self.chk_3d_pet)
        
        grp_view.setLayout(view_layout)
        layout.addWidget(grp_view)
        layout.addStretch()
        
        self.tabs.addTab(tab, "View")

    def _init_display_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # CT Window/Level
        # Window (Width) - typically 1 to 3000
        # Level (Center) - typically -1000 to 1000
        
        self.spin_ct_window = QDoubleSpinBox()
        self.spin_ct_window.setRange(1, 4000)
        self.spin_ct_window.setValue(400) # Default Lung/Soft Tissue
        self.spin_ct_window.valueChanged.connect(self._emit_ct_wl)
        
        self.spin_ct_level = QDoubleSpinBox()
        self.spin_ct_level.setRange(-2000, 2000)
        self.spin_ct_level.setValue(40)
        self.spin_ct_level.valueChanged.connect(self._emit_ct_wl)
        
        layout.addRow("CT Window (Width):", self.spin_ct_window)
        layout.addRow("CT Level (Center):", self.spin_ct_level)

        # PET Window/Level
        # PET is usually 0 to MAX. Window/Level might be weird but user asked for it.
        # Window = Range, Level = Middle.
        
        self.spin_pet_window = QDoubleSpinBox()
        self.spin_pet_window.setRange(0.1, 10000) # Arbitrary scale, large range
        self.spin_pet_window.setValue(20)
        self.spin_pet_window.valueChanged.connect(self._emit_pet_wl)
        
        self.spin_pet_level = QDoubleSpinBox()
        self.spin_pet_level.setRange(0, 10000)
        self.spin_pet_level.setValue(10)
        self.spin_pet_level.valueChanged.connect(self._emit_pet_wl)

        layout.addRow("PET Window:", self.spin_pet_window)
        layout.addRow("PET Level:", self.spin_pet_level)
        
        # Zoom
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(0, 100)
        self.slider_zoom.setValue(20) # 1.0ish
        self.slider_zoom.valueChanged.connect(self.sig_zoom_changed.emit)
        
        self.btn_zoom_fit = QPushButton("Zoom to Fit")
        self.btn_zoom_fit.clicked.connect(self.sig_zoom_to_fit.emit)
        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.slider_zoom)
        zoom_layout.addWidget(self.btn_zoom_fit)
        
        layout.addRow("Zoom:", zoom_layout)
        
        # PET Opacity
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        self.slider_opacity.valueChanged.connect(
            lambda v: self.sig_pet_opacity_changed.emit(v / 100.0)
        )
        
        layout.addRow("PET Overlay Opacity:", self.slider_opacity)
        
        self.tabs.addTab(tab, "Display")
        
    def _emit_ct_wl(self):
        self.sig_ct_window_level_changed.emit(
            float(self.spin_ct_window.value()),
            float(self.spin_ct_level.value())
        )
        
    def _emit_pet_wl(self):
        # We might need to map these slider values to actual intensity units if we want valid rendering
        # For now, pass raw slider values and let LayoutManager (or Main) scale them?
        # Or Just pass as is.
        self.sig_pet_window_level_changed.emit(
            float(self.spin_pet_window.value()),
            float(self.spin_pet_level.value())
        )

    def _emit_new_session(self):
        doc = self.input_doctor.text()
        pat = self.input_patient.text()
        self.sig_new_session_clicked.emit(doc, pat)
        
    def _emit_load_session(self):
        data = self.combo_sessions.currentData()
        if data is not None:
             self.sig_load_session_clicked.emit(int(data))
        
    def _emit_3d_pet_toggle(self, checked):
        self.sig_toggle_3d_pet.emit(checked)

    def _init_refine_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1. Tool Selection
        grp_tools = QGroupBox("Manual Tools")
        tools_layout = QGridLayout()
        
        self.btn_pan = QPushButton("Pan/Zoom")
        self.btn_paint = QPushButton("Paint")
        self.btn_erase = QPushButton("Eraser")
        
        self.btn_pan.setCheckable(True)
        self.btn_paint.setCheckable(True)
        self.btn_erase.setCheckable(True)
        
        # Exclusive check
        from PyQt6.QtWidgets import QButtonGroup
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_paint)
        self.tool_group.addButton(self.btn_erase)
        
        self.btn_pan.setChecked(True)
        
        self.btn_pan.clicked.connect(lambda: self.sig_set_tool.emit("pan_zoom"))
        self.btn_paint.clicked.connect(lambda: self.sig_set_tool.emit("paint"))
        self.btn_erase.clicked.connect(lambda: self.sig_set_tool.emit("erase"))
        
        tools_layout.addWidget(self.btn_pan, 0, 0)
        tools_layout.addWidget(self.btn_paint, 0, 1)
        tools_layout.addWidget(self.btn_erase, 0, 2)
        
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
        
        # 4. Sync/Save Manual
        self.btn_sync = QPushButton("Sync Modifications")
        self.btn_sync.setToolTip("Push manual edits to all viewers and save to session.")
        self.btn_sync.clicked.connect(self.sig_sync_masks_clicked.emit)
        layout.addWidget(self.btn_sync)
        
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
        

        
        # 8. Refinement Progress Bar
        self.refine_progress = QProgressBar()
        self.refine_progress.setRange(0, 0) # Indeterminate
        self.refine_progress.setVisible(False)
        layout.addWidget(self.refine_progress)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Refine")

    def _emit_refine_suv(self):
        val = self.spin_suv.value()
        self.sig_refine_suv_clicked.emit(val)

    def show_progress(self):
        self.progress_bar.setVisible(True)
        
    def hide_progress(self):
        self.progress_bar.setVisible(False)

    def show_refine_progress(self):
        self.refine_progress.setVisible(True)

    def hide_refine_progress(self):
        self.refine_progress.setVisible(False)

    # ──── AutoPET Interactive Tab ────
    
    def _init_autopet_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1. Click Mode Selection
        grp_click = QGroupBox("Click Mode")
        click_layout = QGridLayout()
        
        self.btn_tumor_click = QPushButton("Tumor")
        self.btn_bg_click = QPushButton("Background")
        
        self.btn_tumor_click.setCheckable(True)
        self.btn_bg_click.setCheckable(True)
        self.btn_tumor_click.setChecked(True)
        
        from PyQt6.QtWidgets import QButtonGroup
        self.autopet_click_group = QButtonGroup(self)
        self.autopet_click_group.addButton(self.btn_tumor_click)
        self.autopet_click_group.addButton(self.btn_bg_click)
        
        self.btn_tumor_click.clicked.connect(
            lambda: self.sig_autopet_click_mode_changed.emit("tumor")
        )
        self.btn_bg_click.clicked.connect(
            lambda: self.sig_autopet_click_mode_changed.emit("background")
        )
        
        click_layout.addWidget(self.btn_tumor_click, 0, 0)
        click_layout.addWidget(self.btn_bg_click, 0, 1)
        
        grp_click.setLayout(click_layout)
        layout.addWidget(grp_click)
        
        # 2. Click List
        grp_list = QGroupBox("Added Clicks")
        list_layout = QVBoxLayout()
        
        self.autopet_click_list = QListWidget()
        self.autopet_click_list.setMaximumHeight(120)
        list_layout.addWidget(self.autopet_click_list)
        
        self.btn_clear_clicks = QPushButton("Clear All Clicks")
        self.btn_clear_clicks.clicked.connect(self._on_clear_autopet_clicks)
        list_layout.addWidget(self.btn_clear_clicks)
        
        grp_list.setLayout(list_layout)
        layout.addWidget(grp_list)
        
        # 3. Info label
        info_label = QLabel("Double-click on viewer to place points.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        # 4. Sync Modifications
        self.btn_autopet_sync = QPushButton("Sync Modifications")
        self.btn_autopet_sync.setToolTip("Push manual edits to all viewers and save to session.")
        self.btn_autopet_sync.clicked.connect(self.sig_autopet_sync_clicked.emit)
        layout.addWidget(self.btn_autopet_sync)
        
        # 5. Run Button
        self.btn_autopet_run = QPushButton("Run AutoPET Interactive")
        self.btn_autopet_run.clicked.connect(self.sig_autopet_run_clicked.emit)
        layout.addWidget(self.btn_autopet_run)
        
        # 6. Save (Overwrite)
        self.btn_autopet_save = QPushButton("Save Result (Overwrite)")
        self.btn_autopet_save.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_autopet_save.clicked.connect(self.sig_autopet_save_clicked.emit)
        layout.addWidget(self.btn_autopet_save)
        
        # 7. Progress Bar
        self.autopet_progress = QProgressBar()
        self.autopet_progress.setRange(0, 0)  # Indeterminate
        self.autopet_progress.setVisible(False)
        layout.addWidget(self.autopet_progress)
        
        layout.addStretch()
        self.tabs.addTab(tab, "AutoPET")
    
    def _on_clear_autopet_clicks(self):
        self.autopet_click_list.clear()
        self.sig_autopet_clear_clicks.emit()
    
    def add_autopet_click_item(self, coord_zyx, label):
        """Add an entry to the click list widget."""
        text = f"[{label}] Z={coord_zyx[0]}, Y={coord_zyx[1]}, X={coord_zyx[2]}"
        self.autopet_click_list.addItem(text)
    
    def show_autopet_progress(self):
        self.autopet_progress.setVisible(True)
        self.btn_autopet_run.setEnabled(False)
    
    def hide_autopet_progress(self):
        self.autopet_progress.setVisible(False)
        self.btn_autopet_run.setEnabled(True)
