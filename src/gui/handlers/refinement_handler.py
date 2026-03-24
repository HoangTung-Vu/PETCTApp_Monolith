"""Refinement handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox, QDialog, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt
import numpy as np

class RefinementHandlerMixin:
    """Handles SUV refinement, sync masks, and drawing tool slots."""

    def _on_refine_suv(self, threshold):
        """Refine the current mask using SUV threshold logic (async)."""
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for SUV refinement.")
            return

        self._sync_roi_from_viewer()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        roi_mask = self.session_manager.get_roi_mask_data()

        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before running SUV refinement."
            )
            return

        from ..workers import RefinementWorker
        self.refine_worker = RefinementWorker(
            self.session_manager.pet_image, mask_img, threshold, roi_mask
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self._set_ui_busy(True)
        self.refine_worker.start()

    def _on_refine_adaptive(self, isocontour_fraction, background_mode, border_thickness):
        """Refine the current mask using Adaptive Thresholding (preview, async)."""
        self._last_refine_info = (
            "adaptive",
            f"Isocontour: {isocontour_fraction:.2f}\n"
            f"Background mode: {background_mode}\n"
            f"Border thickness: {border_thickness} voxels"
        )
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for Adaptive Thresholding refinement.")
            return

        self._sync_roi_from_viewer()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        roi_mask = self.session_manager.get_roi_mask_data()

        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before running adaptive thresholding."
            )
            return

        from ..workers import AdaptiveThresholdingWorker
        self.refine_worker = AdaptiveThresholdingWorker(
            self.session_manager.pet_image,
            mask_img,
            roi_mask,
            isocontour_fraction=isocontour_fraction,
            background_mode=background_mode,
            border_thickness=border_thickness,
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self._set_ui_busy(True)
        self.refine_worker.start()

    def _on_refine_iterative(self, m, c1, c0, convergence_tol, max_iterations):
        """Refine the current mask using Iterative Thresholding (preview, async)."""
        self._last_refine_info = (
            "iterative",
            f"m (slope): {m}\n"
            f"c1 (B/S coeff): {c1}\n"
            f"c0 (intercept): {c0}\n"
            f"Tolerance: {convergence_tol}\n"
            f"Max iterations: {max_iterations}"
        )
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for Iterative Thresholding refinement.")
            return

        self._sync_roi_from_viewer()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        roi_mask = self.session_manager.get_roi_mask_data()

        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before running iterative thresholding."
            )
            return

        from ..workers import IterativeThresholdingWorker
        self.refine_worker = IterativeThresholdingWorker(
            self.session_manager.pet_image,
            mask_img,
            roi_mask,
            m=m,
            c1=c1,
            c0=c0,
            convergence_tol=convergence_tol,
            max_iterations=max_iterations,
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self._set_ui_busy(True)
        self.refine_worker.start()

    def _on_refinement_finished(self, refined_img, computed_threshold: float = 0.0):
        refined_data = refined_img.get_fdata(dtype=np.float32)

        # Put the refined result into roi_mask (only within the painted ROI region)
        roi_data = self.session_manager.get_roi_mask_data()
        if roi_data is not None:
            roi_region = roi_data > 0
            refined_within_roi = np.zeros(refined_data.shape, dtype=np.uint8)
            refined_within_roi[roi_region] = (refined_data[roi_region] > 0).astype(np.uint8)
            self.session_manager.set_roi_mask(refined_within_roi)
            self._push_mask_to_all("roi", refined_within_roi)

        print("Refined ROI preview ready. Click 'Save Refinement' to confirm and persist.")
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        self.control_panel.refine_tab.reset_tools()

        # Show threshold notification for adaptive/iterative methods
        info = getattr(self, "_last_refine_info", None)
        if info and info[0] in ("adaptive", "iterative"):
            method_name = "Adaptive Thresholding" if info[0] == "adaptive" else "Iterative Thresholding"
            threshold_line = (
                f"\n─────────────────────────────\n"
                f"Computed threshold:  {computed_threshold:.4f} SUV\n"
            ) if computed_threshold > 0 else ""
            QMessageBox.information(
                self,
                f"Preview — {method_name}",
                f"Parameters used:\n\n{info[1]}"
                f"{threshold_line}\n"
                f"Click 'Save Refinement to Disk' to confirm and persist.",
            )
        self._last_refine_info = None

    def _on_refinement_error(self, error_msg):
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        print(f"Refinement Error: {error_msg}")
        QMessageBox.critical(self, "Refinement Failed", error_msg)

    def _on_refinement_tab_changed(self, index: int):
        """Handle roi_mask setup/teardown for Refine (2) and AutoPET (3) tabs."""
        REFINE_TAB_INDEX = 2
        AUTOPET_TAB_INDEX = 3

        is_refine_mode = index in [REFINE_TAB_INDEX, AUTOPET_TAB_INDEX]
        was_refine_mode = self._last_tab_index in [REFINE_TAB_INDEX, AUTOPET_TAB_INDEX]

        # LEAVING Refine/AutoPET Mode — sync ROI to session, then reset tools
        if was_refine_mode and not is_refine_mode:
            print("[RefineHandler] Leaving Refine/AutoPET. Syncing & keeping ROI mask.")
            self._sync_roi_from_viewer()
            self.control_panel.refine_tab.reset_tools()
            self.layout_manager.disable_shape_drag()

        # ENTERING Refine/AutoPET Mode
        if is_refine_mode and not was_refine_mode:
            print("[RefineHandler] Entering Refine/AutoPET. Ensuring ROI mask exists...")
            
            if self.session_manager.roi_mask is not None:
                # ROI already exists from a previous visit — re-push to viewers
                roi_data = self.session_manager.get_roi_mask_data()
                tumor_data = self.session_manager.get_tumor_mask_data()
                if roi_data is not None:
                    self._push_mask_to_all("roi", roi_data)
                if tumor_data is not None:
                    self._push_mask_to_all("tumor", tumor_data)
            else:
                # First time entering refine — create mask via worker
                from ..workers.roi_worker import EnsureROIWorker
                self.ensure_roi_worker = EnsureROIWorker(self.session_manager)
                self.ensure_roi_worker.finished.connect(self._on_roi_ready)
                self.control_panel.show_refine_progress()
                self._set_ui_busy(True)
                self.ensure_roi_worker.start()

            # Clear report UI when entering interactive mode
            self.session_manager.clear_lesion_data()
            self.control_panel.clear_report_results()
            self.control_panel.chk_show_lesion_ids.setChecked(False)
            self.layout_manager.hide_lesion_ids()

        self._last_tab_index = index

    def _on_roi_ready(self, roi_data, tumor_data, roi_zyx, tumor_zyx):
        """Callback when ROI worker finishes."""
        # Push current roi_mask to viewers so the layer exists for painting
        if roi_data is not None:
            self._push_mask_to_all("roi", roi_data, data_zyx=roi_zyx)
        # Also push tumor mask to ensure it's loaded
        if tumor_data is not None:
            self._push_mask_to_all("tumor", tumor_data, data_zyx=tumor_zyx)
            
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()

    # ── Drawing tool slots ──

    def _on_set_tool(self, tool):
        self.current_tool = tool
        self._update_all_tools()

    def _on_brush_size_changed(self, size):
        self.brush_size = size
        self._update_all_tools()

    def _on_confirm_roi(self):
        """Commit interactive shapes to the ROI mask layer."""
        self.layout_manager.commit_shape("roi")
        print(f"[RefineHandler] ROI shapes committed to roi mask")

    def _update_all_tools(self):
        self.layout_manager.set_drawing_tool(
            self.current_tool, self.brush_size, "roi"
        )

    def _sync_roi_from_viewer(self):
        """Pull ROI mask from active viewer and sync to session manager."""
        mask_data = self.layout_manager.get_active_mask_data("roi")
        if mask_data is None:
            return
        self.session_manager.set_roi_mask(mask_data)
        self.layout_manager.sync_mask_cache(mask_data, "roi")

    def _on_sync_masks(self):
        """Pull mask from active viewer and sync to all others + session."""
        # Sync ROI mask (used during Refine/AutoPET tabs)
        self._sync_roi_from_viewer()
        print("Synced ROI mask from active viewer.")

    def _on_auto_sync(self, layer_type: str):
        """Auto-sync after debounced paint/erase (300ms after last stroke)."""
        mask_data = self.layout_manager.get_active_mask_data(layer_type)
        if mask_data is None:
            return

        if layer_type == "roi":
            self.session_manager.set_roi_mask(mask_data)
        elif layer_type == "tumor":
            self.session_manager.set_tumor_mask(mask_data)

        # BUG-10 FIX: Use lightweight cache sync instead of full update_mask.
        self.layout_manager.sync_mask_cache(mask_data, layer_type)

        # Dismiss stale lesion data if painting on tumor mask
        if layer_type == "tumor":
            self.session_manager.clear_lesion_data()
            self.layout_manager.hide_lesion_ids()
            self.control_panel.chk_show_lesion_ids.setChecked(False)

        print(f"[AutoSync] Synced {layer_type} mask after painting.")

    def _on_confirm_and_save(self):
        """Merge ROI into tumor, push to viewers, then save to disk."""
        merged = self.session_manager.merge_roi_into_tumor()
        if merged is not None:
            self._push_mask_to_all("tumor", merged)

        # Push cleared roi to viewers
        roi_data = self.session_manager.get_roi_mask_data()
        if roi_data is not None:
            self._push_mask_to_all("roi", roi_data)

        # Clear stale report data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        # Persist to disk
        self.save_session()
        print("[RefineHandler] ROI merged into tumor and saved to disk.")
