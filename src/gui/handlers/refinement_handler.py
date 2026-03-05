"""Refinement handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox
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

    def _on_refine_adaptive(self, isocontour_fraction, background_mode, border_thickness):
        """Refine the current mask using Adaptive Thresholding logic (async)."""
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

    def _on_refinement_finished(self, refined_img):
        # BUG-6 FIX: Use float32 to prevent float64 memory bloat since it will be cast to uint8 later anyway
        data = refined_img.get_fdata(dtype=np.float32)
        # BUG-05 FIX: Use set_*_mask() to properly trigger report invalidation
        self.session_manager.set_tumor_mask(data)

        self._push_mask_to_all("tumor", data)
        # BUG-05 FIX: Clear stale report UI and cached data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        
        # COMMIT: Save to disk
        self.session_manager.save_session()
        
        # RE-SNAPSHOT: Very important! After refinement is committed, we need a 
        # NEW baseline for the NEXT ROI drawing, otherwise diff logic fails.
        self.session_manager.snapshot_current_mask("tumor")
        
        print("Refined tumor finished, saved, and re-snapshotted.")
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        self.control_panel.refine_tab.reset_tools()

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
            # BUG-FIX: Only snapshot if there isn't one already (e.g. from previous paint session)
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
