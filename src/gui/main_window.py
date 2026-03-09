"""Main application window.

Slim core (~150 lines): UI setup, signal wiring, session management.
All handler logic lives in ``handlers/`` as mixins.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt6.QtGui import QShortcut, QKeySequence
from pathlib import Path

from ..core.session_manager import SessionManager
from .components.control_panel import ControlPanel
from .components.layout import LayoutManager

from .handlers.segmentation_handler import SegmentationHandlerMixin
from .handlers.refinement_handler import RefinementHandlerMixin
from .handlers.autopet_handler import AutoPETHandlerMixin
from .handlers.eraser_handler import EraserHandlerMixin
from .handlers.report_handler import ReportHandlerMixin


class MainWindow(
    SegmentationHandlerMixin,
    RefinementHandlerMixin,
    AutoPETHandlerMixin,
    EraserHandlerMixin,
    ReportHandlerMixin,
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
        self._last_tab_index = 0 # Track for snapshot/revert logic

        # AutoPET Interactive State
        self.autopet_clicks = []

        # Eraser State
        self._eraser_undo_stack = []

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.addWidget(self.control_panel)
        sidebar_container.setFixedWidth(350)

        main_layout.addWidget(sidebar_container)
        main_layout.addWidget(self.layout_manager)

    def _connect_signals(self):
        cp = self.control_panel
        lm = self.layout_manager

        # Workflow
        cp.sig_load_ct_clicked.connect(self.load_ct_dialog)
        cp.sig_load_pet_clicked.connect(self.load_pet_dialog)
        cp.sig_segment_clicked.connect(self.run_segmentation_dialog)
        cp.sig_layout_changed.connect(lm.set_view_mode)
        cp.sig_toggle_3d_pet.connect(lm.toggle_3d_pet)

        # Display
        cp.sig_pet_opacity_changed.connect(lm.set_pet_opacity)
        cp.sig_tumor_opacity_changed.connect(lm.set_tumor_opacity)
        cp.sig_ct_window_level_changed.connect(lm.set_ct_window_level)
        cp.sig_pet_window_level_changed.connect(lm.set_pet_window_level)
        cp.sig_zoom_changed.connect(lm.set_zoom)
        cp.sig_zoom_to_fit.connect(lm.reset_zoom)
        cp.sig_toggle_mask.connect(lm.toggle_mask)

        # Session
        cp.sig_new_session_clicked.connect(self.create_new_session)
        cp.sig_load_session_clicked.connect(self.load_existing_session)

        # Refinement
        cp.sig_set_tool.connect(self._on_set_tool)
        cp.sig_brush_size_changed.connect(self._on_brush_size_changed)
        cp.sig_refine_suv_clicked.connect(self._on_refine_suv)
        cp.sig_refine_adaptive_clicked.connect(self._on_refine_adaptive)
        cp.sig_refine_iterative_clicked.connect(self._on_refine_iterative)
        cp.sig_confirm_roi_clicked.connect(self._on_confirm_roi)
        cp.sig_save_refine_clicked.connect(self.save_session)

        # AutoPET
        cp.sig_autopet_click_mode_changed.connect(lm.enable_autopet_click_mode)
        cp.sig_autopet_run_clicked.connect(self._on_autopet_run)
        cp.sig_autopet_save_clicked.connect(self.save_session)
        cp.sig_autopet_clear_clicks.connect(self._on_autopet_clear_clicks)
        lm.sig_autopet_click_added.connect(self._on_autopet_click_added)

        # Eraser
        cp.sig_eraser_mode_toggled.connect(self._on_eraser_mode_toggled)
        cp.sig_eraser_undo_clicked.connect(self._on_eraser_undo)
        cp.sig_eraser_save_clicked.connect(self.save_session)
        lm.sig_eraser_region_removed.connect(self._on_eraser_region_removed)

        # Report
        cp.sig_report_clicked.connect(self._on_report_clicked)
        cp.sig_toggle_lesion_ids.connect(self._on_toggle_lesion_ids)

        # Tabs
        cp.sig_tab_changed.connect(self._on_tab_changed)

        # Auto-sync: debounced paint → session manager
        lm.sig_mask_painted.connect(self._on_auto_sync)

        # Global Shortcuts
        self.shortcut_toggle_mask = QShortcut(QKeySequence("s"), self)
        self.shortcut_toggle_mask.activated.connect(self._on_shortcut_toggle_tumor_mask)

    def _on_shortcut_toggle_tumor_mask(self):
        """Toggle tumor mask visibility via 's' hotkey."""
        chk = self.control_panel.view_display_tab.chk_tumor
        chk.setChecked(not chk.isChecked())
        print(f"[MainWindow] Shortcut 's' toggled tumor mask visibility to {chk.isChecked()}")

    # ──── Session Management ────

    def _reset_all_state(self):
        """Clear all viewing and application state before session change."""
        # Clear viewers
        self.layout_manager.clear_all_viewers()

        # Clear application state
        self.autopet_clicks.clear()
        self._eraser_undo_stack.clear()

        # Clear report display
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()

        # Uncheck lesion IDs checkbox
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        # Reset session label
        self.control_panel.set_current_session_label("None")

        print("[MainWindow] All state reset.")

    def create_new_session(self, doctor: str, patient: str):
        if not doctor or not patient:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Info", "Please enter both Doctor and Patient names.")
            return

        self._reset_all_state()

        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, action="create",
            new_doctor=doctor, new_patient=patient
        )
        self.loader_worker.finished.connect(self._on_data_loaded)
        self.loader_worker.error.connect(self._on_data_error)
        self.control_panel.show_progress()
        self._set_ui_busy(True)
        self.loader_worker.start()

    def load_existing_session(self, session_id: int):
        self._reset_all_state()

        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, current_session_id=session_id, action="load"
        )
        self.loader_worker.finished.connect(self._on_data_loaded)
        self.loader_worker.error.connect(self._on_data_error)
        self.control_panel.show_progress()
        self._set_ui_busy(True)
        self.loader_worker.start()

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

    def _update_session_files(self, ct_path=None, pet_path=None):
        """Spawns worker to load CT/PET to prevent UI block."""
        from .workers import DataLoaderWorker
        self.loader_worker = DataLoaderWorker(
            self.session_manager, ct_path=ct_path, pet_path=pet_path, action="update"
        )
        self.loader_worker.finished.connect(self._on_data_loaded)
        self.loader_worker.error.connect(self._on_data_error)
        self.control_panel.show_progress()
        self._set_ui_busy(True)
        self.loader_worker.start()

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
        
        print(f"Async data loading completed for session {self.session_manager.current_session_id}.")

    def _on_data_error(self, error_msg):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        print(f"Data Loading Error: {error_msg}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Loading Failed", error_msg)

    def _refresh_session_list(self):
        sessions = self.session_manager.get_all_sessions()
        self.control_panel.combo_sessions.clear()
        for s in sessions:
            label = f"{s.id}: {s.patient_name} ({s.created_at.strftime('%Y-%m-%d %H:%M')})"
            self.control_panel.combo_sessions.addItem(label, userData=s.id)

    def _refresh_viewers(self):
        ct_data = self.session_manager.get_ct_data()
        pet_data = self.session_manager.get_pet_data()

        affine = None
        if self.session_manager.ct_image:
            affine = self.session_manager.ct_image.affine
        elif self.session_manager.pet_image:
            affine = self.session_manager.pet_image.affine

        if affine is not None:
            tumor_mask = self.session_manager.get_tumor_mask_data()
            self.layout_manager.load_data(ct_data, pet_data, affine, tumor_mask)
        else:
            print("No affine available (no images loaded?)")

    def save_session(self):
        self.session_manager.save_session()

    def _on_tab_changed(self, index: int):
        """Delegate tab change events to specialized handlers."""
        # Mixin-based tab handlers (implemented in refinement_handler.py, etc.)
        self._on_refinement_tab_changed(index)

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
        all_viewers = (
            list(self.layout_manager.grid_viewers.values())
            + [self.layout_manager.overlay_viewer]
            + list(self.layout_manager.mono_viewers.values())
            + [self.layout_manager.viewer_3d]
        )
        for v in all_viewers:
            if v is not None:
                try:
                    v.close()
                except Exception:
                    pass
                    
        # SAFELY STOP ANY RUNNING WORKERS SO PYTHON PROCESS EXITS
        # By quitting the threads safely, we prevent silent QThread destructor segfaults 
        workers = [
            getattr(self, 'loader_worker', None),
            getattr(self, 'snapshot_worker', None),
            getattr(self, 'refine_worker', None),
            getattr(self, 'autopet_worker', None),
            getattr(self, 'report_worker', None),
            getattr(self, 'worker', None), # segmentation worker
        ]
        
        for worker in workers:
            if worker is not None and worker.isRunning():
                print(f"[MainWindow] Forcing background worker {worker.__class__.__name__} to quit...")
                worker.quit()
                worker.wait(2000) # Give it 2s to clean up gracefully
                
        super().closeEvent(event)