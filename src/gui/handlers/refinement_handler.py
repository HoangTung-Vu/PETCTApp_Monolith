"""Refinement handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
import numpy as np


class RefinementHandlerMixin:
    """Handles SUV refinement, threshold computation, sync masks, and drawing tool slots.

    Key states:
        _roi_dirty (bool):  True after ROI is painted but before threshold is applied.
                            Refine/compute buttons are enabled only when True.
        _painted_roi (ndarray | None):  Snapshot of ROI mask at the time of compute,
                            used as the base for live preview and final apply.
        _preview_active (bool):  True while a threshold preview dialog is open.
                            Suppresses auto-sync so preview data doesn't leak
                            back into session_manager.
        _roi_labels (ndarray | None):  Connected-component labels from compute step.
        _components_info (list | None):  Per-component threshold info from compute.
        _method_info (str):  Description of the method used for compute.
    """

    def _init_refinement_state(self):
        """Call from MainWindow.__init__ to set up refinement state."""
        self._painted_roi = None
        self._preview_active = False
        self._roi_labels = None
        self._components_info = None
        self._method_info = ""
        self._current_component_idx = 0
        self._cached_pet_f32 = None

        # Manual Edit state
        self._manual_edit_tool = "pan_zoom"
        self._manual_edit_brush = 10

    # ── is_dirty management ──

    def _is_roi_dirty(self) -> bool:
        """Dynamically check if ROI mask has painted content."""
        roi_data = self.session_manager.get_roi_mask_data()
        return bool(roi_data is not None and roi_data.any())

    def _update_refine_button_states(self):
        self.control_panel.refine_tab.set_refine_buttons_enabled(self._is_roi_dirty())

    # ── Merge thresholded result into tumor ──

    # ── Apply thresholded result to ROI ──

    def _apply_result_to_roi(self, result, base_roi):
        """Applies thresholded result strictly to the ROI mask (deferring tumor merge to Save)."""
        try:
            roi_data = self.session_manager.get_roi_mask_data()
            if roi_data is None:
                roi_data = np.zeros_like(result)
            else:
                roi_data = roi_data.copy()

            # Using assignment replaces existing ROI correctly within the painted box
            roi_data[base_roi > 0] = result[base_roi > 0]

            self.session_manager.set_roi_mask(roi_data)
            self._push_mask_to_all("roi", roi_data)
        except Exception as e:
            print(f"Apply ROI Failed: {e}")

    def _clear_refinement_state(self):
        """Reset all transient refinement state after apply or cancel."""
        self._preview_active = False
        self._painted_roi = None
        self._roi_labels = None
        self._components_info = None
        self._current_component_idx = 0
        self._cached_pet_f32 = None
        self._current_preview_dialog = None
        self._base_preview = None
        self._active_comp_mask = None
        self._active_pet_vals = None
        self._update_refine_button_states()
        self.control_panel.refine_tab.reset_tools()

    # ── SUV Threshold Apply (synchronous, the "base" operation) ──

    def _on_refine_suv(self, threshold):
        """Apply SUV threshold to painted ROI → update ROI (synchronous)."""
        if not self._is_roi_dirty():
            QMessageBox.information(
                self, "No ROI Change",
                "Paint or draw an ROI region first.\n"
                "The ROI must be modified before running refinement again."
            )
            return

        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for SUV refinement.")
            return

        self._sync_roi_from_viewer()

        roi_mask = self.session_manager.get_roi_mask_data()
        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before running SUV refinement."
            )
            return

        base_roi = self._painted_roi if self._painted_roi is not None else roi_mask
        pet_data = self.session_manager.pet_image.get_fdata(dtype=np.float32)
        roi = base_roi > 0

        result = np.zeros(base_roi.shape, dtype=np.uint8)
        result[roi & (pet_data >= threshold)] = 1

        # Apply to ROI
        self._apply_result_to_roi(result, base_roi)
        self._clear_refinement_state()

        print(f"[RefineHandler] SUV threshold {threshold:.4f} applied to ROI.")

    # ── Adaptive / Iterative — compute threshold only ──

    def _on_refine_adaptive(self, isocontour_fraction, background_mode, border_thickness):
        """Compute adaptive threshold per ROI component (async, no mask modification)."""
        if not self._is_roi_dirty():
            QMessageBox.information(
                self, "No ROI Change",
                "Paint or draw an ROI region first.\n"
                "The ROI must be modified before computing a new threshold."
            )
            return

        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for Adaptive Thresholding.")
            return

        self._sync_roi_from_viewer()

        roi_mask = self.session_manager.get_roi_mask_data()
        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before computing adaptive threshold."
            )
            return

        self._painted_roi = roi_mask.copy()

        from ..workers import ThresholdComputeWorker
        self._threshold_worker = ThresholdComputeWorker(
            self.session_manager.pet_image,
            self.session_manager.tumor_mask,
            roi_mask,
            method="adaptive",
            isocontour_fraction=isocontour_fraction,
            background_mode=background_mode,
            border_thickness=border_thickness,
        )
        self._threshold_worker.threshold_computed.connect(self._on_threshold_computed)
        self._threshold_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self._set_ui_busy(True)
        self._threshold_worker.start()

    def _on_refine_iterative(self, m, c1, c0, convergence_tol, max_iterations):
        """Compute iterative threshold per ROI component (async, no mask modification)."""
        if not self._is_roi_dirty():
            QMessageBox.information(
                self, "No ROI Change",
                "Paint or draw an ROI region first.\n"
                "The ROI must be modified before computing a new threshold."
            )
            return

        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for Iterative Thresholding.")
            return

        if self.session_manager.tumor_mask is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No tumor mask available. Iterative thresholding needs an existing mask\n"
                "for background estimation. Run segmentation first or use Adaptive Thresholding."
            )
            return

        self._sync_roi_from_viewer()

        roi_mask = self.session_manager.get_roi_mask_data()
        if roi_mask is None or not roi_mask.any():
            QMessageBox.warning(
                self, "No ROI",
                "Please paint/draw an ROI region first before computing iterative threshold."
            )
            return

        self._painted_roi = roi_mask.copy()

        from ..workers import ThresholdComputeWorker
        self._threshold_worker = ThresholdComputeWorker(
            self.session_manager.pet_image,
            self.session_manager.tumor_mask,
            roi_mask,
            method="iterative",
            m=m, c1=c1, c0=c0,
            convergence_tol=convergence_tol,
            max_iterations=max_iterations,
        )
        self._threshold_worker.threshold_computed.connect(self._on_threshold_computed)
        self._threshold_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self._set_ui_busy(True)
        self._threshold_worker.start()

    # ── Threshold computation result → sequential popup dialogs ──

    def _on_threshold_computed(self, components_info, labels, method_info):
        """Called when ThresholdComputeWorker finishes. Shows per-component dialog."""
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()

        if not components_info:
            QMessageBox.warning(
                self, "No Result",
                "No valid ROI components found. Check your ROI painting and PET data."
            )
            return

        self._components_info = components_info
        self._roi_labels = labels
        self._method_info = method_info
        self._preview_active = True
        self._cached_pet_f32 = self.session_manager.pet_image.get_fdata(dtype=np.float32)

        print(
            f"[RefineHandler] {method_info}: {len(components_info)} component(s) found. "
            f"Showing threshold dialog(s)..."
        )

        # Force tool to pan_zoom so user can safely navigate viewers without painting
        self.control_panel.refine_tab.reset_tools()
        # Disable other workflow tabs so user cannot start other actions while previewing,
        # but keep View & Display tab accessible for crosshair/opacity adjustments.
        for tab in [
            self.control_panel.refine_tab,
            self.control_panel.workflow_tab,
            self.control_panel.eraser_tab
        ]:
            tab.setEnabled(False)

        # Process components sequentially via modeless dialogs
        self._current_component_idx = 0
        self._show_next_component_dialog()

    def _show_next_component_dialog(self):
        """Show the threshold dialog for the current component."""
        if self._current_component_idx >= len(self._components_info):
            self._apply_all_component_thresholds()
            return

        from ..components.threshold_preview_dialog import ThresholdPreviewDialog

        comp = self._components_info[self._current_component_idx]
        total = len(self._components_info)
        idx = self._current_component_idx

        # PRECOMPUTE: Static background components (i < current and i > current)
        self._base_preview = np.zeros(self._painted_roi.shape, dtype=np.uint8)
        pet_data = self._cached_pet_f32
        for i, c in enumerate(self._components_info):
            c_mask = self._roi_labels == c["label"]
            if i < self._current_component_idx:
                self._base_preview[c_mask & (pet_data >= c["threshold"])] = 1
            elif i > self._current_component_idx:
                self._base_preview[c_mask & (self._painted_roi > 0)] = 1

        # PRECOMPUTE: Extract 1D array of PET values and boolean 3D mask for CURRENT component
        self._active_comp_mask = (self._roi_labels == comp["label"])
        self._active_pet_vals = pet_data[self._active_comp_mask]

        # Jump viewer to centroid of this component
        self._jump_to_component(comp["label"])

        dialog = ThresholdPreviewDialog(
            component_info=comp,
            component_index=idx,
            total_components=total,
            method_info=self._method_info,
            parent=self,
        )

        # Modeless dialog allows interacting with main window (viewers)
        dialog.setModal(False)
        
        # Initial preview before user adjusts slider
        self._update_component_preview(comp["threshold"])

        # Live preview as user adjusts slider - lightning fast using precomputed 1D arrays
        dialog.threshold_changed.connect(
            lambda thresh: self._update_component_preview(thresh)
        )

        dialog.accepted.connect(lambda: self._on_component_accepted(dialog, comp, idx))
        dialog.rejected.connect(self._on_component_rejected)

        self._current_preview_dialog = dialog
        dialog.show()

    def _on_component_accepted(self, dialog, comp, idx):
        """Handle user applying the threshold for the current component."""
        self._components_info[idx]["threshold"] = dialog.final_threshold
        print(
            f"[RefineHandler] Component {comp['label']}: "
            f"final threshold = {dialog.final_threshold:.4f} SUV"
        )
        self._current_component_idx += 1
        self._show_next_component_dialog()

    def _on_component_rejected(self):
        """Handle user cancelling the threshold adjustment."""
        print("[RefineHandler] Threshold adjustment cancelled.")
        self._preview_active = False
        if self._painted_roi is not None:
            self.session_manager.set_roi_mask(self._painted_roi)
            self._push_mask_to_all("roi", self._painted_roi)
        # Keep dirty state true natively as mask is restored
        self._painted_roi = None
        self._roi_labels = None
        self._components_info = None
        self._current_preview_dialog = None
        
        # Re-enable tabs
        for tab in [
            self.control_panel.refine_tab,
            self.control_panel.workflow_tab,
            self.control_panel.eraser_tab
        ]:
            tab.setEnabled(True)

    def _jump_to_component(self, label_id):
        """Jump the viewer to the centroid of the given connected component."""
        if self._roi_labels is None:
            return

        from scipy.ndimage import center_of_mass

        comp_mask = self._roi_labels == label_id
        # center_of_mass returns indices in array dimension order.
        # roi_mask/labels are stored in XYZ (NIfTI convention),
        # so result is (x_idx, y_idx, z_idx).
        centroid_xyz = center_of_mass(comp_mask)
        if centroid_xyz is None or any(np.isnan(centroid_xyz)):
            return

        cx, cy, cz = centroid_xyz  # x, y, z in NIfTI XYZ space
        shape_xyz = self._roi_labels.shape

        # to_napari does: transpose(XYZ→ZYX) then flip(Z, Y).
        # So Napari coords require the same flip:
        z_napari = (shape_xyz[2] - 1) - cz
        y_napari = (shape_xyz[1] - 1) - cy
        x_napari = cx  # X is not flipped

        self.layout_manager.jump_to_position(z_napari, y_napari, x_napari)
        print(f"[RefineHandler] Jumped to component {label_id} centroid: z={z_napari:.0f}, y={y_napari:.0f}, x={x_napari:.0f}")

    def _update_component_preview(self, threshold):
        """Update the ROI mask in the viewer during threshold adjustment."""
        # 1. Start with the precomputed static preview parts
        preview = self._base_preview.copy()

        # 2. Fast 1D boolean masking for the active component only
        valid_indices = self._active_pet_vals >= threshold
        
        # 3. Assign 1D validation back into the 3D footprint
        preview[self._active_comp_mask] = valid_indices

        self._push_mask_to_all("roi", preview)

    def _apply_all_component_thresholds(self):
        """Apply the final per-component thresholds, merge into tumor_mask, clear ROI."""
        # Re-enable tabs since preview is finished
        for tab in [
            self.control_panel.refine_tab,
            self.control_panel.workflow_tab,
            self.control_panel.eraser_tab
        ]:
            tab.setEnabled(True)
        self._current_preview_dialog = None

        if self._painted_roi is None or self._roi_labels is None:
            return

        pet_data = self.session_manager.pet_image.get_fdata()
        result = np.zeros(self._painted_roi.shape, dtype=np.uint8)

        for comp in self._components_info:
            comp_mask = self._roi_labels == comp["label"]
            result[comp_mask & (pet_data >= comp["threshold"])] = 1

        # Update SUV spinbox with average threshold (for reference)
        if self._components_info:
            avg = np.mean([c["threshold"] for c in self._components_info])
            self.control_panel.refine_tab.spin_suv.setValue(round(avg, 2))

        # Apply threshold to ROI mask
        self._apply_result_to_roi(result, self._painted_roi)
        self._clear_refinement_state()

        print("[RefineHandler] All component thresholds applied to ROI mask.")

    # ── Error handling ──

    def _on_refinement_error(self, error_msg):
        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        self._clear_refinement_state()
        print(f"Refinement Error: {error_msg}")
        QMessageBox.critical(self, "Refinement Failed", error_msg)

    # ── Tab entry/exit ──

    def _on_refinement_tab_changed(self, index: int):
        """Handle roi_mask setup/teardown for Refine (2) tab."""
        REFINE_TAB_INDEX = 2

        is_refine_mode = index == REFINE_TAB_INDEX
        was_refine_mode = self._last_tab_index == REFINE_TAB_INDEX

        # LEAVING Refine Mode — sync ROI to session, then reset tools
        if was_refine_mode and not is_refine_mode:
            print("[RefineHandler] Leaving Refine. Syncing & keeping ROI mask.")
            self._sync_roi_from_viewer()
            self.control_panel.refine_tab.reset_tools()
            self.layout_manager.disable_shape_drag()

        # ENTERING Refine Mode
        if is_refine_mode and not was_refine_mode:
            print("[RefineHandler] Entering Refine. Ensuring ROI mask exists...")

            if getattr(self, '_preview_active', False) and getattr(self, '_current_preview_dialog', None):
                print("[RefineHandler] Preview active, restoring active threshold preview...")
                thresh = self._current_preview_dialog.slider.value() / 100.0
                self._update_component_preview(thresh)
                
                # Still need to push tumor mask
                tumor_data = self.session_manager.get_tumor_mask_data()
                if tumor_data is not None:
                    self._push_mask_to_all("tumor", tumor_data)
            elif self.session_manager.roi_mask is not None:
                # ROI already exists from a previous visit — re-push to viewers
                roi_data = self.session_manager.get_roi_mask_data()
                tumor_data = self.session_manager.get_tumor_mask_data()
                if roi_data is not None:
                    self._push_mask_to_all("roi", roi_data)
                    # Button states will be updated below
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

            self._update_refine_button_states()

            # Clear report UI when entering interactive mode
            self.session_manager.clear_lesion_data()
            self.control_panel.clear_report_results()
            self.control_panel.chk_show_lesion_ids.setChecked(False)
            self.layout_manager.hide_lesion_ids()

        self._last_tab_index = index

    def _on_roi_ready(self, roi_data, tumor_data, roi_zyx, tumor_zyx):
        """Callback when ROI worker finishes."""
        if roi_data is not None:
            self._push_mask_to_all("roi", roi_data, data_zyx=roi_zyx)
        if tumor_data is not None:
            self._push_mask_to_all("tumor", tumor_data, data_zyx=tumor_zyx)

        self._set_ui_busy(False)
        self.control_panel.hide_refine_progress()
        self._update_refine_button_states()

    # ── Manual Edit (tumor mask) slots ──

    def _on_manual_edit_tool(self, tool: str):
        """Switch manual edit tool: 'pan_zoom', 'paint', 'erase' on tumor layer."""
        self._manual_edit_tool = tool
        # When activating manual edit, reset ROI tools to pan_zoom
        if tool != "pan_zoom":
            self.control_panel.refine_tab.reset_tools()
        self._apply_manual_edit_tool()

    def _on_manual_edit_brush_changed(self, size: int):
        self._manual_edit_brush = size
        if self._manual_edit_tool != "pan_zoom":
            self._apply_manual_edit_tool()

    def _apply_manual_edit_tool(self):
        """Push manual edit tool state to viewers, targeting tumor layer."""
        self.layout_manager.set_drawing_tool(
            self._manual_edit_tool, self._manual_edit_brush, "tumor"
        )

    # ── ROI Drawing tool slots ──

    def _on_set_tool(self, tool):
        self.current_tool = tool
        # When activating ROI tool, reset manual edit to pan_zoom
        if tool != "pan_zoom":
            self.control_panel.refine_tab.reset_manual_edit()
        self._update_all_tools()

    def _on_brush_size_changed(self, size):
        self.brush_size = size
        self._update_all_tools()

    def _on_confirm_roi(self):
        """Commit shapes (if any) and sync ROI from viewer, then mark dirty."""
        self.layout_manager.commit_shape("roi")
        self._sync_roi_from_viewer()
        self._update_refine_button_states()
        print(f"[RefineHandler] ROI confirmed, dirty={self._is_roi_dirty()}")

    def _update_all_tools(self):
        self.layout_manager.set_drawing_tool(
            self.current_tool, self.brush_size, "roi"
        )

    def _sync_roi_from_viewer(self):
        """Pull ROI mask from active viewer and sync to session manager."""
        if getattr(self, '_preview_active', False):
            # Don't sync the temporary preview mask back to the session natively
            return

        mask_data = self.layout_manager.get_active_mask_data("roi")
        if mask_data is None:
            return
        self.session_manager.set_roi_mask(mask_data)
        self.layout_manager.sync_mask_cache(mask_data, "roi")

    def _on_sync_masks(self):
        """Pull mask from active viewer and sync to all others + session."""
        self._sync_roi_from_viewer()
        print("Synced ROI mask from active viewer.")

    def _on_auto_sync(self, layer_type: str):
        """Auto-sync after debounced paint/erase (300ms after last stroke)."""
        # Suppress auto-sync for ROI while preview dialog is active
        if layer_type == "roi" and self._preview_active:
            return

        mask_data = self.layout_manager.get_active_mask_data(layer_type)
        if mask_data is None:
            return

        if layer_type == "roi":
            self.session_manager.set_roi_mask(mask_data)
            self._update_refine_button_states()
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
        """Merge ROI into tumor (if ROI exists), then persist tumor mask to disk."""
        roi_data_check = self.session_manager.get_roi_mask_data()
        has_roi = roi_data_check is not None and roi_data_check.any()

        if has_roi:
            self._sync_roi_from_viewer()
            merged = self.session_manager.merge_roi_into_tumor()
            if merged is not None:
                self._push_mask_to_all("tumor", merged)

            # Push cleared roi to viewers
            roi_data = self.session_manager.get_roi_mask_data()
            if roi_data is not None:
                self._push_mask_to_all("roi", roi_data)

            self._update_refine_button_states()

        # Clear stale report data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        # Persist to disk
        self.save_session()
        print("[RefineHandler] Tumor mask saved to disk." +
              (" (ROI merged)" if has_roi else " (manual edit)"))
