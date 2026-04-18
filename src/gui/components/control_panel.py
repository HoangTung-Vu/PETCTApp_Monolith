"""Control Panel — signal hub that assembles tab sub-modules.

Each tab is defined in ``tabs/`` as its own QWidget with its own signals.
ControlPanel simply wires them together and exposes a flat signal surface
so that MainWindow does not need to know about the inner tabs.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtCore import pyqtSignal

from .tabs.workflow_tab import WorkflowTab
from .tabs.view_display_tab import ViewDisplayTab
from .tabs.refine_tab import RefineTab
from .tabs.eraser_tab import EraserTab


class ControlPanel(QWidget):
    """Control Panel with Workflow, View & Display, Refine, and Eraser tabs."""

    # ── Re-exported Signals (flat surface for MainWindow) ──

    # Workflow
    sig_load_ct_clicked = pyqtSignal()
    sig_load_pet_clicked = pyqtSignal()
    sig_segment_clicked = pyqtSignal()
    sig_new_session_clicked = pyqtSignal(str, str)
    sig_load_session_clicked = pyqtSignal(int)
    sig_report_clicked = pyqtSignal()
    sig_toggle_lesion_ids = pyqtSignal(bool)
    sig_tab_changed = pyqtSignal(int) # index

    # View & Display
    sig_layout_changed = pyqtSignal(str)
    sig_toggle_3d_pet = pyqtSignal(bool)
    sig_pet_opacity_changed = pyqtSignal(float)
    sig_tumor_opacity_changed = pyqtSignal(float)
    sig_roi_opacity_changed = pyqtSignal(float)
    sig_ct_window_level_changed = pyqtSignal(float, float)
    sig_pet_window_level_changed = pyqtSignal(float, float)
    sig_zoom_changed = pyqtSignal(int)
    sig_zoom_to_fit = pyqtSignal()
    sig_toggle_mask = pyqtSignal(str, bool)
    sig_ct_colormap_changed = pyqtSignal(str)
    sig_pet_colormap_changed = pyqtSignal(str)
    sig_interpolation_toggled = pyqtSignal(bool)
    sig_crosshair_toggled = pyqtSignal(bool)

    # Refine — Manual Edit (tumor mask)
    sig_manual_edit_tool = pyqtSignal(str)
    sig_manual_edit_brush_changed = pyqtSignal(int)

    # Refine — ROI tools
    sig_set_tool = pyqtSignal(str)
    sig_brush_size_changed = pyqtSignal(int)
    sig_refine_suv_clicked = pyqtSignal(float)
    sig_refine_adaptive_clicked = pyqtSignal(float, str, int)
    sig_refine_iterative_clicked = pyqtSignal(float, float, float, float, int)
    sig_confirm_roi_clicked = pyqtSignal()
    sig_save_refine_clicked = pyqtSignal()

    # Eraser
    sig_eraser_mode_toggled = pyqtSignal(bool)
    sig_eraser_undo_clicked = pyqtSignal()
    sig_eraser_save_clicked = pyqtSignal()

    sig_load_from_dicom = pyqtSignal(str, str, str)   # dcm_folder, doctor, patient

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QTabWidget()

        # Instantiate tabs
        self.workflow_tab = WorkflowTab()
        self.view_display_tab = ViewDisplayTab()
        self.refine_tab = RefineTab()
        self.eraser_tab = EraserTab()

        self.tabs.addTab(self.workflow_tab, "Workflow")
        self.tabs.addTab(self.view_display_tab, "View & Display")
        self.tabs.addTab(self.refine_tab, "Refine")
        self.tabs.addTab(self.eraser_tab, "Eraser")

        # Tab indices for the tab-change handler
        self._view_display_tab_index = 1
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Wire tab signals → ControlPanel signals
        self._connect_tab_signals()

    def _connect_tab_signals(self):
        """Forward all tab signals through the ControlPanel signal hub."""

        # Workflow
        w = self.workflow_tab
        w.sig_load_ct_clicked.connect(self.sig_load_ct_clicked)
        w.sig_load_pet_clicked.connect(self.sig_load_pet_clicked)
        w.sig_segment_clicked.connect(self.sig_segment_clicked)
        w.sig_new_session_clicked.connect(self.sig_new_session_clicked)
        w.sig_load_session_clicked.connect(self.sig_load_session_clicked)
        w.sig_report_clicked.connect(self.sig_report_clicked)
        w.sig_load_from_dicom.connect(self.sig_load_from_dicom)
        w.sig_toggle_lesion_ids.connect(self.sig_toggle_lesion_ids)

        # View & Display
        vd = self.view_display_tab
        vd.sig_layout_changed.connect(self.sig_layout_changed)
        vd.sig_toggle_3d_pet.connect(self.sig_toggle_3d_pet)
        vd.sig_pet_opacity_changed.connect(self.sig_pet_opacity_changed)
        vd.sig_tumor_opacity_changed.connect(self.sig_tumor_opacity_changed)
        vd.sig_roi_opacity_changed.connect(self.sig_roi_opacity_changed)
        vd.sig_ct_window_level_changed.connect(self.sig_ct_window_level_changed)
        vd.sig_pet_window_level_changed.connect(self.sig_pet_window_level_changed)
        vd.sig_zoom_changed.connect(self.sig_zoom_changed)
        vd.sig_zoom_to_fit.connect(self.sig_zoom_to_fit)
        vd.sig_toggle_mask.connect(self.sig_toggle_mask)
        vd.sig_ct_colormap_changed.connect(self.sig_ct_colormap_changed)
        vd.sig_pet_colormap_changed.connect(self.sig_pet_colormap_changed)
        vd.sig_interpolation_toggled.connect(self.sig_interpolation_toggled)
        vd.sig_crosshair_toggled.connect(self.sig_crosshair_toggled)

        # Refine — Manual Edit
        r = self.refine_tab
        r.sig_manual_edit_tool.connect(self.sig_manual_edit_tool)
        r.sig_manual_edit_brush_changed.connect(self.sig_manual_edit_brush_changed)

        # Refine — ROI tools
        r.sig_set_tool.connect(self.sig_set_tool)
        r.sig_brush_size_changed.connect(self.sig_brush_size_changed)
        r.sig_refine_suv_clicked.connect(self.sig_refine_suv_clicked)
        r.sig_refine_adaptive_clicked.connect(self.sig_refine_adaptive_clicked)
        r.sig_refine_iterative_clicked.connect(self.sig_refine_iterative_clicked)
        r.sig_confirm_roi_clicked.connect(self.sig_confirm_roi_clicked)
        r.sig_save_refine_clicked.connect(self.sig_save_refine_clicked)

        # Eraser
        e = self.eraser_tab
        e.sig_eraser_mode_toggled.connect(self.sig_eraser_mode_toggled)
        e.sig_eraser_undo_clicked.connect(self.sig_eraser_undo_clicked)
        e.sig_eraser_save_clicked.connect(self.sig_eraser_save_clicked)


    # ── Proxy accessors (kept for backwards compat with MainWindow) ──

    @property
    def combo_sessions(self):
        return self.workflow_tab.combo_sessions

    @property
    def chk_show_lesion_ids(self):
        return self.workflow_tab.chk_show_lesion_ids

    @property
    def btn_pan(self):
        return self.refine_tab.btn_pan

    @property
    def btn_eraser_toggle(self):
        return self.eraser_tab.btn_eraser_toggle

    # ── Progress / report helpers (delegate to tabs) ──

    def show_progress(self):
        self.workflow_tab.show_progress()

    def hide_progress(self):
        self.workflow_tab.hide_progress()

    def show_refine_progress(self):
        self.refine_tab.show_refine_progress()

    def hide_refine_progress(self):
        self.refine_tab.hide_refine_progress()

    def show_report_progress(self):
        self.workflow_tab.show_report_progress()

    def hide_report_progress(self):
        self.workflow_tab.hide_report_progress()

    def show_report_results(self, metrics: dict):
        self.workflow_tab.show_report_results(metrics)

    def clear_report_results(self):
        self.workflow_tab.clear_report_results()

    def set_current_session_label(self, text: str):
        self.workflow_tab.set_current_session_label(text)

    def _emit_ct_wl(self):
        self.view_display_tab._emit_ct_wl()

    def _emit_pet_wl(self):
        self.view_display_tab._emit_pet_wl()

    # ── Tab change handler ──

    def _on_tab_changed(self, index: int):
        """Reset mouse tool to pan/zoom on EVERY tab switch (except View & Display).
        Also disable eraser mode when leaving their tabs."""
        # Always reset tool to pan/zoom (except View & Display tab)
        if index != self._view_display_tab_index:
            self.refine_tab.btn_pan.setChecked(True)
            self.sig_set_tool.emit("pan_zoom")
            # Also reset manual edit tools
            self.refine_tab.reset_manual_edit()

        # Always disable eraser if it was enabled
        if self.eraser_tab.btn_eraser_toggle.isChecked():
            self.eraser_tab.btn_eraser_toggle.setChecked(False)

        # Emit general tab changed signal for MainWindow/Handlers
        self.sig_tab_changed.emit(index)
