"""Main application window.
All handler logic lives in ``handlers/`` as mixins.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSplitter, QPushButton,
    QMessageBox
)
from PyQt6.QtGui import QShortcut, QKeySequence
from pathlib import Path

from ..core.session_manager import SessionManager
from .components.control_panel import ControlPanel
from .components.layout import LayoutManager

from .handlers.segmentation_handler import SegmentationHandlerMixin
from .handlers.refinement_handler import RefinementHandlerMixin
from .handlers.eraser_handler import EraserHandlerMixin
from .handlers.report_handler import ReportHandlerMixin
from .handlers.dicom_import_handler import DicomImportHandlerMixin


class MainWindow(
    SegmentationHandlerMixin,
    RefinementHandlerMixin,
    EraserHandlerMixin,
    ReportHandlerMixin,
    DicomImportHandlerMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metabolic Lesion Quantification on PET/CT")
        self.setGeometry(100, 100, 1600, 900)

        # Core Logic
        self.session_manager = SessionManager()

        # GUI Components
        self.control_panel = ControlPanel()
        self.layout_manager = LayoutManager()

        # Setup UI
        self._init_ui()
        self._connect_signals()

        # Load initial data
        self._refresh_session_list()

        # Refinement State
        self.current_tool = "pan_zoom"
        self.brush_size = 10
        self._last_tab_index = 0
        self._init_refinement_state()

        # Eraser State
        self._eraser_undo_stack = []

        # DICOM Import State
        self._init_dicom_import_handler()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar Restore Button (appears when sidebar is collapsed)
        self.sidebar_stub = QWidget()
        self.sidebar_stub.setFixedWidth(20)
        self.sidebar_stub.setVisible(False)
        self.sidebar_stub.setStyleSheet("background-color: #2c2c2c; border-right: 1px solid #444;")
        
        stub_layout = QVBoxLayout(self.sidebar_stub)
        stub_layout.setContentsMargins(0, 2, 0, 0)
        stub_layout.setSpacing(0)

        self.btn_restore_sidebar = QPushButton(">>>")
        self.btn_restore_sidebar.setFixedSize(20, 24)
        self.btn_restore_sidebar.setToolTip("Show Sidebar")
        self.btn_restore_sidebar.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888;
                border: none;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #00aaff;
            }
        """)
        self.btn_restore_sidebar.clicked.connect(self._restore_sidebar)
        stub_layout.addWidget(self.btn_restore_sidebar)
        stub_layout.addStretch()
        
        main_layout.addWidget(self.sidebar_stub)

        # Draggable splitter between sidebar and viewer area
        from PyQt6.QtCore import Qt
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(5)
        self._splitter.setOpaqueResize(False)  # Reduces lag during drag by showing a preview line
        self._splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #444, stop:0.5 #666, stop:1 #444);
                width: 5px;
            }
            QSplitter::handle:horizontal:hover {
                background: #888;
            }
        """)

        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(4, 0, 0, 4)
        sidebar_layout.setSpacing(0)

        # Collapse button at the top of sidebar
        collapse_header = QWidget()
        collapse_header_layout = QHBoxLayout(collapse_header)
        collapse_header_layout.setContentsMargins(0, 2, 4, 2)
        collapse_header_layout.addStretch()
        self.btn_collapse_sidebar = QPushButton("<<<")
        self.btn_collapse_sidebar.setFixedSize(24, 24)
        self.btn_collapse_sidebar.setToolTip("Collapse Sidebar")
        self.btn_collapse_sidebar.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #666;
                border: none;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                color: #00aaff;
            }
        """)
        self.btn_collapse_sidebar.clicked.connect(self._collapse_sidebar)
        collapse_header_layout.addWidget(self.btn_collapse_sidebar)
        sidebar_layout.addWidget(collapse_header)

        sidebar_layout.addWidget(self.control_panel)
        sidebar_container.setMinimumWidth(200)
        sidebar_container.setMaximumWidth(500)

        self._splitter.addWidget(sidebar_container)
        self._splitter.addWidget(self.layout_manager)
        # Set initial sizes: sidebar 280, viewer gets the rest
        self._splitter.setSizes([280, 1320])
        self._splitter.setStretchFactor(0, 0)  # sidebar doesn't auto-stretch
        self._splitter.setStretchFactor(1, 1)  # viewer area stretches

        main_layout.addWidget(self._splitter)

        # Re-enforce camera settings after splitter drag (napari resets them
        # on resize, which breaks crosshair / pan mouse behaviour)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

    def _connect_signals(self):
        cp = self.control_panel
        lm = self.layout_manager

        # Workflow
        cp.sig_load_ct_clicked.connect(self.load_ct_dialog)
        cp.sig_load_pet_clicked.connect(self.load_pet_dialog)
        cp.sig_load_seg_clicked.connect(self.load_seg_dialog)
        cp.sig_segment_clicked.connect(self.run_segmentation_dialog)
        cp.sig_active_views_changed.connect(lm.set_active_views)
        cp.sig_layout_changed.connect(lm.set_view_mode)      # "3d" only
        cp.sig_toggle_3d_pet.connect(lm.toggle_3d_pet)

        # Display
        cp.sig_overlay_pet_opacity_changed.connect(lm.set_overlay_pet_opacity)
        cp.sig_tumor_opacity_changed.connect(lm.set_tumor_opacity)
        cp.sig_roi_opacity_changed.connect(lm.set_roi_opacity)
        cp.sig_ct_window_level_changed.connect(lm.set_ct_window_level)
        cp.sig_pet_window_level_changed.connect(lm.set_pet_window_level)
        cp.sig_zoom_changed.connect(lm.set_zoom)
        cp.sig_zoom_to_fit.connect(lm.reset_zoom)
        cp.sig_toggle_mask.connect(lm.toggle_mask)
        cp.sig_ct_colormap_changed.connect(lm.set_ct_colormap)
        cp.sig_pet_colormap_changed.connect(lm.set_pet_colormap)
        cp.sig_overlay_pet_colormap_changed.connect(lm.set_overlay_pet_colormap)
        cp.sig_interpolation_toggled.connect(lm.set_interpolation)
        cp.sig_crosshair_toggled.connect(self._on_crosshair_toggled)

        # Session
        cp.sig_new_session_clicked.connect(self.create_new_session)
        cp.sig_load_session_clicked.connect(self.load_existing_session)

        # Refinement — Manual Edit (tumor mask)
        cp.sig_manual_edit_tool.connect(self._on_manual_edit_tool)
        cp.sig_manual_edit_brush_changed.connect(self._on_manual_edit_brush_changed)

        # Refinement — ROI tools
        cp.sig_set_tool.connect(self._on_set_tool)
        cp.sig_brush_size_changed.connect(self._on_brush_size_changed)
        cp.sig_refine_suv_clicked.connect(self._on_refine_suv)
        cp.sig_refine_adaptive_clicked.connect(self._on_refine_adaptive)
        cp.sig_refine_iterative_clicked.connect(self._on_refine_iterative)
        cp.sig_confirm_roi_clicked.connect(self._on_confirm_roi)
        cp.sig_save_refine_clicked.connect(self._on_confirm_and_save)

        # Eraser
        cp.sig_eraser_mode_toggled.connect(self._on_eraser_mode_toggled)
        cp.sig_eraser_undo_clicked.connect(self._on_eraser_undo)
        cp.sig_eraser_save_clicked.connect(self.save_session)
        lm.sig_eraser_region_removed.connect(self._on_eraser_region_removed)
        lm.sig_eraser_background_click.connect(self._on_eraser_background_click)

        # Report
        cp.sig_report_clicked.connect(self._on_report_clicked)
        cp.sig_toggle_lesion_ids.connect(self._on_toggle_lesion_ids)

        # DICOM Import
        cp.sig_load_from_dicom.connect(self._on_load_from_dicom)

        # Tabs
        cp.sig_tab_changed.connect(self._on_tab_changed)

        # Auto-sync: debounced paint → session manager
        lm.sig_mask_painted.connect(self._on_auto_sync)

        # Global Shortcuts
        self.shortcut_toggle_mask = QShortcut(QKeySequence("s"), self)
        self.shortcut_toggle_mask.activated.connect(self._on_shortcut_toggle_tumor_mask)

    def _on_splitter_moved(self, pos, index):
        """Re-enforce napari camera settings after sidebar splitter drag.

        Napari resets ``camera.mouse_pan`` and ``camera.mouse_zoom`` during
        widget resize — reuse the existing crosshair toggle to restore state.
        """
        xhair_on = self.control_panel.view_display_tab.btn_crosshair.isChecked()
        if not getattr(self, '_crosshair_suppressed_by_tab', False):
            self._on_crosshair_toggled(xhair_on)

        # Show restore button if sidebar is collapsed
        sidebar_width = self._splitter.sizes()[0]
        self.sidebar_stub.setVisible(sidebar_width == 0)

    def _restore_sidebar(self):
        """Restore sidebar to default width."""
        self._splitter.setSizes([280, self._splitter.width() - 280])
        self.sidebar_stub.setVisible(False)
        # Re-enforce state after resize
        self._on_splitter_moved(280, 1)

    def _collapse_sidebar(self):
        """Collapse the sidebar completely."""
        self._splitter.setSizes([0, self._splitter.width()])
        self.sidebar_stub.setVisible(True)
        # Re-enforce state after resize
        self._on_splitter_moved(0, 1)

    def _on_shortcut_toggle_tumor_mask(self):
        """Toggle tumor mask visibility via 's' hotkey."""
        chk = self.control_panel.view_display_tab.chk_tumor
        chk.setChecked(not chk.isChecked())
        print(f"[MainWindow] Shortcut 's' toggled tumor mask visibility to {chk.isChecked()}")

    def _on_eraser_background_click(self):
        """Notify the user they clicked on background (no tumor voxel there)."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "Eraser — Background Clicked",
            "No tumor voxel found at the clicked position.\n\n"
            "Double-click directly on a highlighted lesion region to erase it.",
        )

    def _on_crosshair_toggled(self, enabled: bool):
        """Toggle the full crosshair overlay."""
        if enabled:
            self.layout_manager.enable_crosshair_mode()
        else:
            self.layout_manager.disable_crosshair_mode()

    # ──── Shared helpers ────

    def _clear_all_report_ui(self):
        """Clear report data and all related UI elements."""
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

    def _show_worker_error(self, msg: str, title: str = "Operation Failed"):
        """Unified worker error handler: reset busy state and show dialog."""
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        QMessageBox.critical(self, title, msg)

    def _spawn_worker(self, worker, on_finished, on_error=None, progress_type="general"):
        """Connect signals, show progress, mark UI busy, and start worker."""
        worker.finished.connect(on_finished)
        if on_error is not None:
            worker.error.connect(on_error)
        if progress_type == "refine":
            self.control_panel.show_refine_progress()
        elif progress_type == "report":
            self.control_panel.show_report_progress()
        else:
            self.control_panel.show_progress()
        self._set_ui_busy(True)
        worker.start()

    # ──── Session Management ────

    def _reset_all_state(self):
        """Clear all viewing and application state before session change."""
        # Clear viewers
        self.layout_manager.clear_all_viewers()

        # Clear application state
        self._eraser_undo_stack.clear()
        self.session_manager.roi_mask = None

        # Clear report display
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()

        # Uncheck lesion IDs checkbox
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        # Reset session label
        self.control_panel.set_current_session_label("None")

        print("[MainWindow] All state reset.")

    def create_new_session(self, doctor: str, patient: str):
        self._reset_all_state()
        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, action="create",
            new_doctor=doctor, new_patient=patient
        )
        self._spawn_worker(self.loader_worker, self._on_data_loaded, self._on_data_error)

    def load_existing_session(self, session_id: int):
        self._reset_all_state()
        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, current_session_id=session_id, action="load"
        )
        self._spawn_worker(self.loader_worker, self._on_data_loaded, self._on_data_error)

    def load_ct_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CT Image", "", "NIfTI files (*.nii.gz *.nii)"
        )
        if file_path:
            self._update_session_files(ct_path=Path(file_path))

    def load_pet_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load PET Image", "", "NIfTI files (*.nii.gz *.nii)"
        )
        if file_path:
            self._update_session_files(pet_path=Path(file_path))

    def load_seg_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segmentation Image", "", "NIfTI files (*.nii.gz *.nii)"
        )
        if file_path:
            self._update_session_files(tumor_seg_path=Path(file_path))

    def _update_session_files(self, ct_path=None, pet_path=None, tumor_seg_path=None):
        """Spawns worker to load CT/PET/Seg to prevent UI block."""
        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, ct_path=ct_path, pet_path=pet_path, tumor_seg_path=tumor_seg_path, action="update"
        )
        self._spawn_worker(self.loader_worker, self._on_data_loaded, self._on_data_error)

    def _on_data_loaded(self, success):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        if not success:
            return
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._do_refresh_after_load)

    def _do_refresh_after_load(self):
        self._refresh_viewers()
        self._refresh_session_list()
        self.control_panel._emit_ct_wl()
        self.control_panel._emit_pet_wl()

        # Update session label
        session_name = f"ID: {self.session_manager.current_session_id} - {self.session_manager.patient_name}"
        self.control_panel.set_current_session_label(session_name)

        # Crosshair is on by default — enable after viewers are populated
        # Respect the toggle button state
        xhair_btn = self.control_panel.view_display_tab.btn_crosshair
        if xhair_btn.isChecked():
            self.layout_manager.enable_crosshair_mode()
        self._crosshair_suppressed_by_tab = False

        print(f"Async data loading completed for session {self.session_manager.current_session_id}.")

    def _on_data_error(self, error_msg):
        print(f"Data Loading Error: {error_msg}")
        self._show_worker_error(error_msg, "Loading Failed")

    def _refresh_session_list(self):
        sessions = self.session_manager.get_all_sessions()
        self.control_panel.combo_sessions.clear()
        for s in sessions:
            label = f"{s.id}: {s.patient_name} ({s.created_at.strftime('%Y-%m-%d %H:%M')})"
            self.control_panel.combo_sessions.addItem(label, userData=s.id)

    def _refresh_viewers(self):
        ct_data = self.session_manager.get_ct_data()
        pet_data = self.session_manager.get_pet_data()

        ct_affine = self.session_manager.ct_image.affine if self.session_manager.ct_image else None
        pet_affine = self.session_manager.pet_image.affine if self.session_manager.pet_image else None

        ct_filename = ""
        pet_filename = ""
        session_id = self.session_manager.current_session_id
        if session_id is not None:
            session = self.session_manager.repository.get_by_id(session_id)
            if session:
                if session.ct_path:
                    ct_filename = Path(session.ct_path).name
                if session.pet_path:
                    pet_filename = Path(session.pet_path).name

        if ct_affine is not None or pet_affine is not None:
            tumor_mask = self.session_manager.get_tumor_mask_data()
            self.layout_manager.load_data(
                ct_data=ct_data, 
                pet_data=pet_data, 
                ct_affine=ct_affine, 
                pet_affine=pet_affine, 
                tumor_mask=tumor_mask, 
                ct_filename=ct_filename, 
                pet_filename=pet_filename
            )
        else:
            print("No affine available (no images loaded?)")

    def save_session(self):
        from .workers import SaveWorker
        self.save_worker = SaveWorker(self.session_manager)
        self._spawn_worker(self.save_worker, self._on_save_finished, self._on_save_error)

    def _on_save_finished(self):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        print("[MainWindow] Session saved asynchronously.")

    def _on_save_error(self, error_msg):
        print(f"Save Error: {error_msg}")
        self._show_worker_error(error_msg, "Save Failed")

    # Tab indices
    _TAB_WORKFLOW = 0
    _TAB_VIEW     = 1
    _TAB_REFINE   = 2
    _TAB_ERASER   = 3

    def _on_tab_changed(self, index: int):
        """Delegate tab change events to specialized handlers."""
        self._on_refinement_tab_changed(index)

        # Disable crosshair overlay in paint/click tabs (crosshair interferes)
        _PAINT_TABS = (self._TAB_REFINE, self._TAB_ERASER)
        xhair_btn = self.control_panel.view_display_tab.btn_crosshair
        crosshair_was_on = xhair_btn.isChecked()

        if index in _PAINT_TABS:
            # Temporarily hide crosshair overlay (don't change button state)
            if self.layout_manager._crosshair_enabled:
                self.layout_manager.disable_crosshair_mode()
                self._crosshair_suppressed_by_tab = True
        else:
            # Restore crosshair if it was suppressed and the toggle is ON
            if getattr(self, '_crosshair_suppressed_by_tab', False) and crosshair_was_on:
                self.layout_manager.enable_crosshair_mode()
            self._crosshair_suppressed_by_tab = False
            # Deselect Labels layers so their built-in drag doesn't conflict
            # with crosshair / pan mouse callbacks
            self.layout_manager.deactivate_labels()

    def _set_ui_busy(self, busy: bool):
        """Enable/Disable interactive controls during background tasks."""
        self.control_panel.tabs.setEnabled(not busy)
        # Also disable important action buttons specifically if needed, 
        # but disabling the whole tab widget is safer.
        print(f"[MainWindow] UI Busy: {busy}")

    def closeEvent(self, event):
        # Hide immediately to give feedback to user
        self.hide()
        
        # Close all viewers properly
        all_viewers = list(self.layout_manager._viewer_pool) + [self.layout_manager.viewer_3d]
        for v in all_viewers:
            if v is not None:
                try:
                    v.close()
                except Exception:
                    pass
                    
        # SAFELY STOP ANY RUNNING WORKERS SO PYTHON PROCESS EXITS
        # By quitting the threads safely, we prevent silent QThread destructor segfaults 
        workers = [
            getattr(self, 'save_worker', None),
            getattr(self, 'loader_worker', None),
            getattr(self, 'refine_worker', None),
            getattr(self, '_threshold_worker', None),
            getattr(self, 'report_worker', None),
            getattr(self, 'worker', None),       # segmentation worker
            getattr(self, 'dicom_worker', None),
            getattr(self, '_merge_save_worker', None),
        ]
        
        for worker in workers:
            if worker is not None and worker.isRunning():
                print(f"[MainWindow] Forcing background worker {worker.__class__.__name__} to quit...")
                worker.quit()
                worker.wait(2000) # Give it 2s to clean up gracefully
                
        super().closeEvent(event)