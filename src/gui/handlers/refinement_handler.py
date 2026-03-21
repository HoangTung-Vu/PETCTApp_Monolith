"""Refinement handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox, QDialog, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt
import numpy as np

class RefinementHandlerMixin:
    """Handles SUV refinement, sync masks, and drawing tool slots."""

    def _on_refine_suv(self, threshold):
        """Refine the current mask using SUV threshold logic (async)."""
        # BUG-04 FIX: Validate BEFORE syncing
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for SUV refinement.")
            return

        self._on_sync_masks()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        # Compute ROI from snapshot
        roi_mask = self.session_manager.get_paint_roi("tumor")

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

    def _on_refine_adaptive(self, isocontour_fraction, background_mode, border_thickness):  # noqa: E501
        """Refine the current mask using Adaptive Thresholding (preview, async)."""
        # Store params so we can display them after preview
        self._last_refine_info = (
            "adaptive",
            f"Isocontour: {isocontour_fraction:.2f}\n"
            f"Background mode: {background_mode}\n"
            f"Border thickness: {border_thickness} voxels"
        )
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for Adaptive Thresholding refinement.")
            return

        self._on_sync_masks()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        # Compute ROI from snapshot
        roi_mask = self.session_manager.get_paint_roi("tumor")

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
        # Store params so we can display them after preview
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

        self._on_sync_masks()

        mask_img = self.session_manager.tumor_mask
        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available to refine."
            )
            return

        # Compute ROI from snapshot
        roi_mask = self.session_manager.get_paint_roi("tumor")

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
        # Use float32 to prevent float64 memory bloat since it will be cast to uint8 later anyway
        data = refined_img.get_fdata(dtype=np.float32)
        # Use set_*_mask() to properly trigger report invalidation
        self.session_manager.set_tumor_mask(data)

        self._push_mask_to_all("tumor", data)
        # Clear stale report UI and cached data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        # PREVIEW MODE: Do NOT auto-save. The user must click "Save Refinement" explicitly.
        # We do NOT re-snapshot here. The ROI diff should continue pointing to the
        # originally painted blob until the user confirms the preview by saving.

        print("Refined tumor preview ready. Click 'Save Refinement' to persist to disk.")
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
                f"Click 'Save Refinement to Disk' to persist.",
            )
        self._last_refine_info = None

    def _on_refinement_error(self, error_msg):
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        print(f"Refinement Error: {error_msg}")
        QMessageBox.critical(self, "Refinement Failed", error_msg)

    def _on_refinement_tab_changed(self, index: int):
        """Handle snapshot/revert logic for Refine (2) and AutoPET (3) tabs."""
        REFINE_TAB_INDEX = 2
        AUTOPET_TAB_INDEX = 3
        
        is_refine_mode = index in [REFINE_TAB_INDEX, AUTOPET_TAB_INDEX]
        was_refine_mode = self._last_tab_index in [REFINE_TAB_INDEX, AUTOPET_TAB_INDEX]

        # LEAVING Refine/AutoPET Mode
        if was_refine_mode and not is_refine_mode:
            print("[RefineHandler] Leaving Refine/AutoPET. Preserving painted ROI and discarding unconfirmed shapes...")
            self.control_panel.refine_tab.reset_tools()
            self.layout_manager.disable_shape_drag()

        # ENTERING Refine/AutoPET Mode
        if is_refine_mode and not was_refine_mode:
            # Only snapshot if there isn't one already (e.g. from previous paint session)
            # If a snapshot exists, it means the doctor started painting and switched tabs,
            # so we want to keep the old snapshot so the ROI diff is still valid against the current painted mask.
            if self.session_manager._tumor_mask_snapshot is None:
                print("[RefineHandler] Entering Refine/AutoPET. Launching async tumor snapshot...")
                from ..workers import SnapshotWorker
                self._set_ui_busy(True)
                
                # We do NOT want to copy memory on the main thread because getting 
                # .get_fdata().copy() will freeze the UI for ~1 second.
                # Tell SnapshotWorker to do it asynchronously.
                self.snapshot_worker = SnapshotWorker(self.session_manager, "tumor")
                self.snapshot_worker.finished.connect(self._on_snapshot_finished)
                self.snapshot_worker.start()
                
                # Clear report UI and hide lesion IDs when snapshot is created
                self.session_manager.clear_lesion_data()
                self.control_panel.clear_report_results()
                self.control_panel.chk_show_lesion_ids.setChecked(False)
                self.layout_manager.hide_lesion_ids()
            else:
                print("[RefineHandler] Entering Refine/AutoPET. Existing snapshot found, skipping to preserve ROI.")

        self._last_tab_index = index

    def _on_snapshot_finished(self):
        self._set_ui_busy(False)
        print("[RefineHandler] Async snapshot finished. Mask is ready.")
        # If the snapshot worker created a new zeroed mask, we need to push it
        # to ensure Napari viewers have the layer.
        data = self.session_manager.get_tumor_mask_data()
        if data is not None:
             self._push_mask_to_all("tumor", data)

    # ── Drawing tool slots ──

    def _on_set_tool(self, tool):
        self.current_tool = tool
        self._update_all_tools()

    def _on_brush_size_changed(self, size):
        self.brush_size = size
        self._update_all_tools()

    def _on_confirm_roi(self):
        """Commit interactive shapes to the actual mask layer."""
        self.layout_manager.commit_shape("tumor")
        # The painting itself triggers _on_auto_sync via sig_mask_painted,
        # which updates SessionManager.
        print(f"[RefineHandler] ROI shapes committed to tumor")

    def _update_all_tools(self):
        self.layout_manager.set_drawing_tool(
            self.current_tool, self.brush_size, "tumor"
        )

    def _on_sync_masks(self):
        """Pull mask from active viewer and sync to all others + session."""
        mask_data = self.layout_manager.get_active_mask_data("tumor")
        if mask_data is None:
            print("No mask data found to sync.")
            return

        self.session_manager.set_tumor_mask(mask_data)

        self._push_mask_to_all("tumor", mask_data)
        print("Synced tumor mask from active viewer to all.")

    def _on_auto_sync(self, layer_type: str):
        """Auto-sync after debounced paint/erase (300ms after last stroke)."""
        mask_data = self.layout_manager.get_active_mask_data(layer_type)
        if mask_data is None:
            return

        if layer_type == "tumor":
            self.session_manager.set_tumor_mask(mask_data)

        # BUG-10 FIX: Use lightweight cache sync instead of full update_mask.
        # Visible viewers already share the painted data via _on_mask_data_changed.
        # This only updates caches and invalidates non-visible layouts.
        self.layout_manager.sync_mask_cache(mask_data, layer_type)
        
        # Dismiss stale lesion data if painting on tumor mask
        if layer_type == "tumor":
            self.session_manager.clear_lesion_data()
            self.layout_manager.hide_lesion_ids()
            self.control_panel.chk_show_lesion_ids.setChecked(False)

        print(f"[AutoSync] Synced {layer_type} mask after painting.")
