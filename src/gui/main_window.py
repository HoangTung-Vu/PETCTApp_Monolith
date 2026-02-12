import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QListWidget, QDockWidget, QToolBar, QFileDialog
)
from PyQt6.QtCore import Qt
from pathlib import Path

from ..core.session_manager import SessionManager
from .components.control_panel import ControlPanel
from .components.layout_manager import LayoutManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PET/CT Segmentation App")
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
        self.target_layer = "tumor" # Default
        self.target_layer = "tumor" # Default

        
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Control Panel (Sidebar)
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.addWidget(self.control_panel)
        sidebar_container.setFixedWidth(350)
        
        # Center: Layout Manager (Viewers)
        main_layout.addWidget(sidebar_container)
        main_layout.addWidget(self.layout_manager)
        
    def _connect_signals(self):
        # Control Panel -> Main Window Actions
        self.control_panel.sig_load_ct_clicked.connect(self.load_ct_dialog)
        self.control_panel.sig_load_pet_clicked.connect(self.load_pet_dialog)
        self.control_panel.sig_segment_clicked.connect(self.run_segmentation_dialog) # Changed to dialog or direct
        # self.control_panel.sig_save_clicked.connect(self.save_session) # Removed
        self.control_panel.sig_layout_changed.connect(self.layout_manager.set_view_mode)
        self.control_panel.sig_toggle_3d_pet.connect(self.layout_manager.toggle_3d_pet)
        
        # Display settings
        self.control_panel.sig_pet_opacity_changed.connect(self.layout_manager.set_pet_opacity)
        self.control_panel.sig_ct_window_level_changed.connect(self.layout_manager.set_ct_window_level)
        self.control_panel.sig_pet_window_level_changed.connect(self.layout_manager.set_pet_window_level)
        self.control_panel.sig_zoom_changed.connect(self.layout_manager.set_zoom)
        self.control_panel.sig_zoom_to_fit.connect(self.layout_manager.reset_zoom)
        self.control_panel.sig_toggle_mask.connect(self.layout_manager.toggle_mask)
        
        # Session Management
        self.control_panel.sig_new_session_clicked.connect(self.create_new_session)
        self.control_panel.sig_load_session_clicked.connect(self.load_existing_session)
        
        # Refinement
        self.control_panel.sig_set_tool.connect(self._on_set_tool)
        self.control_panel.sig_brush_size_changed.connect(self._on_brush_size_changed)
        self.control_panel.sig_target_layer_changed.connect(self._on_target_layer_changed)
        self.control_panel.sig_sync_masks_clicked.connect(self._on_sync_masks)
        
        # Layout Manager Signals
        # self.layout_manager.sig_mask_edited.connect(self._on_mask_edited)

        
    def _on_refine_suv(self, threshold):
        """
        Refine the current mask using SUV threshold logic (async).
        """
        # 1. Sync manual edits first to ensure SessionManager has latest data
        self._on_sync_masks()
        
        if self.session_manager.pet_image is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Data", "PET image required for SUV refinement.")
            return
            
        mask_img = None
        if self.target_layer == "tumor":
            mask_img = self.session_manager.tumor_mask
        elif self.target_layer == "organ":
            mask_img = self.session_manager.organ_mask
            
        if mask_img is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Data", f"No {self.target_layer} mask available to refine.")
            return

        from .workers import RefinementWorker
        self.refine_worker = RefinementWorker(
            self.session_manager.pet_image,
            mask_img,
            threshold
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)
        
        self.control_panel.show_refine_progress()
        self.refine_worker.start()

    def _on_refinement_finished(self, refined_img):
        # Update SessionManager
        data = refined_img.get_fdata()
        if self.target_layer == "tumor":
            self.session_manager.tumor_mask = refined_img
        elif self.target_layer == "organ":
            self.session_manager.organ_mask = refined_img
            
        # Update Viewers
        self._push_mask_to_all(self.target_layer, data)
        print(f"Refined {self.target_layer} finished.")
        
        self.control_panel.hide_refine_progress()

    def _on_refinement_error(self, error_msg):
        self.control_panel.hide_refine_progress()
        print(f"Refinement Error: {error_msg}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Refinement Failed", error_msg)

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

    def create_new_session(self, doctor: str, patient: str):
        if not doctor or not patient:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Info", "Please enter both Doctor and Patient names.")
            return
            
        # Create session in DB
        try:
            self.session_manager.create_session(doctor, patient)
            self._refresh_session_list()
            self._refresh_viewers() # Clear viewers
            
            # Trigger file loading? Or wait for user?
            # User workflow: New Session -> Load CT -> Load PET
            # So just clear viewers is enough.
        except Exception as e:
            print(f"Error creating session: {e}")

    def load_existing_session(self, session_id: int):
        try:
            self.session_manager.load_session(session_id)
            self._refresh_viewers()
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")

    def _refresh_session_list(self):
        sessions = self.session_manager.get_all_sessions()
        self.control_panel.combo_sessions.clear()
        for s in sessions:
            label = f"{s.id}: {s.patient_name} ({s.created_at.strftime('%Y-%m-%d %H:%M')})"
            self.control_panel.combo_sessions.addItem(label, userData=s.id)

    def _update_session_files(self, ct_path: Path = None, pet_path: Path = None):
        try:
            if self.session_manager.current_session_id is None:
                # If no session, prompt to create one? Or create default?
                # For now, let's create a "Quick Session"
                self.session_manager.create_session("System", "Anonymous", ct_path=ct_path, pet_path=pet_path)
            else:
                # Update existing
                self.session_manager.update_current_session(ct_path=ct_path, pet_path=pet_path)
            
            # Push to Viewer
            self._refresh_viewers()
            self._refresh_session_list() # Update list to show file changes if any (not in label though)
            
        except Exception as e:
            print(f"Error updating session files: {e}")
            import traceback
            traceback.print_exc()

    def _refresh_viewers(self):
        ct_data = self.session_manager.get_ct_data()
        pet_data = self.session_manager.get_pet_data()
        
        # Get affine from whatever is available
        affine = None
        if self.session_manager.ct_image:
             affine = self.session_manager.ct_image.affine
        elif self.session_manager.pet_image:
             affine = self.session_manager.pet_image.affine
             
        if affine is not None:
            tumor_mask = self.session_manager.get_tumor_mask_data()
            organ_mask = self.session_manager.get_organ_mask_data()
            self.layout_manager.load_data(ct_data, pet_data, affine, tumor_mask, organ_mask)
        else:
            print("No affine available (no images loaded?)")
            
    def run_segmentation_dialog(self):
        """
        Ask user which segmentation to run (Tumor or Organ).
        For now, let's just run Tumor for demo, or add a dialog.
        """
        from PyQt6.QtWidgets import QInputDialog
        items = ["Tumor Segmentation (nnUNet)", "Organ Segmentation (TotalSegmentator)"]
        item, ok = QInputDialog.getItem(self, "Select Segmentation", 
                                        "Choose model:", items, 0, False)
        
        if ok and item:
            if "Tumor" in item:
                self._run_segmentation("tumor")
            else:
                self._run_segmentation("organ")
                
    def _run_segmentation(self, seg_type: str):
        ct_img = self.session_manager.ct_image
        pet_img = self.session_manager.pet_image
        
        from ..gui.workers import SegmentationWorker
        from PyQt6.QtWidgets import QMessageBox

        if seg_type == "tumor":
             # Tumor requires BOTH CT and PET
             if not ct_img or not pet_img:
                 QMessageBox.warning(self, "Missing Data", "Tumor segmentation requires both CT and PET images.")
                 return
             
             # Pass list [ct, pet] as expected by NNUNetEngine
             input_data = [ct_img, pet_img]
             
        elif seg_type == "organ":
             # Organ requires CT
             if not ct_img:
                 QMessageBox.warning(self, "Missing Data", "Organ segmentation requires a CT image.")
                 return
                 
             # Pass single CT image
             input_data = ct_img
        
        else:
            return

        self.worker = SegmentationWorker(seg_type, input_data)
        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)
        
        self.control_panel.show_progress()
        self.worker.start()
        
    def _on_segmentation_finished(self, result_tuple):
        mask_img, seg_type = result_tuple
        affine = mask_img.affine
        data = mask_img.get_fdata()
        
        if seg_type == "tumor":
            self.session_manager.set_tumor_mask(data)
            # Update Viewers
            # Iterate all viewers and add mask
            self._push_mask_to_all("tumor", data)
            
        elif seg_type == "organ":
            self.session_manager.set_organ_mask(data)
            self._push_mask_to_all("organ", data)
            
        # Auto-save immediately
        self.session_manager.save_session()
        print(f"Segmentation {seg_type} finished and saved.")
        
        self.control_panel.hide_progress()
        
    def _push_mask_to_all(self, layer_type, data):
        self.layout_manager.update_mask(data, layer_type)

    def _on_segmentation_error(self, error_msg):
        self.control_panel.hide_progress()
        print(f"Segmentation Error: {error_msg}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Segmentation Failed", error_msg)

    def save_session(self):
        self.session_manager.save_session()
        


    def closeEvent(self, event):
        # Close all viewers properly if needed
        # self.layout_manager.close_all()
        super().closeEvent(event)

    # Refinement Slots
    def _on_set_tool(self, tool):
        self.current_tool = tool
        self._update_all_tools()
        
    def _on_brush_size_changed(self, size):
        self.brush_size = size
        self._update_all_tools()
        
    def _on_target_layer_changed(self, layer):
        self.target_layer = layer
        self._update_all_tools()
        
    def _update_all_tools(self):
        # Push tool settings to Layout Manager
        self.layout_manager.set_drawing_tool(
            self.current_tool, 
            self.brush_size, 
            self.target_layer
        )

    def _on_sync_masks(self):
        """
        Pull mask from active viewer and sync to all others + session.
        """
        # 1. Get current mask data from active viewer
        mask_data = self.layout_manager.get_active_mask_data(self.target_layer)
        if mask_data is None:
            print("No mask data found to sync.")
            return

        # 2. Update Session Manager
        if self.target_layer == "tumor":
            self.session_manager.set_tumor_mask(mask_data)
        elif self.target_layer == "organ":
            self.session_manager.set_organ_mask(mask_data)
        
        # 3. Push back to all viewers (sync)
        self._push_mask_to_all(self.target_layer, mask_data)
        print(f"Synced {self.target_layer} mask from active viewer to all.")



