"""Refinement handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox


class RefinementHandlerMixin:
    """Handles SUV refinement, sync masks, and drawing tool slots."""

    def _on_refine_suv(self, threshold):
        """Refine the current mask using SUV threshold logic (async)."""
        # BUG-04 FIX: Validate BEFORE syncing
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image required for SUV refinement.")
            return

        self._on_sync_masks()

        mask_img = None
        if self.target_layer == "tumor":
            mask_img = self.session_manager.tumor_mask
        elif self.target_layer == "organ":
            mask_img = self.session_manager.organ_mask

        if mask_img is None:
            QMessageBox.warning(
                self, "Missing Data",
                f"No {self.target_layer} mask available to refine."
            )
            return

        from ..workers import RefinementWorker
        self.refine_worker = RefinementWorker(
            self.session_manager.pet_image, mask_img, threshold
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)

        self.control_panel.show_refine_progress()
        self.refine_worker.start()

    def _on_refinement_finished(self, refined_img):
        data = refined_img.get_fdata()
        # BUG-05 FIX: Use set_*_mask() to properly trigger report invalidation
        if self.target_layer == "tumor":
            self.session_manager.set_tumor_mask(data)
        elif self.target_layer == "organ":
            self.session_manager.set_organ_mask(data)

        self._push_mask_to_all(self.target_layer, data)
        # BUG-05 FIX: Clear stale report UI and cached data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        
        print(f"Refined {self.target_layer} finished.")
        self.control_panel.hide_refine_progress()

    def _on_refinement_error(self, error_msg):
        self.control_panel.hide_refine_progress()
        print(f"Refinement Error: {error_msg}")
        QMessageBox.critical(self, "Refinement Failed", error_msg)

    # ── Drawing tool slots ──

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
        self.layout_manager.set_drawing_tool(
            self.current_tool, self.brush_size, self.target_layer
        )

    def _on_sync_masks(self):
        """Pull mask from active viewer and sync to all others + session."""
        mask_data = self.layout_manager.get_active_mask_data(self.target_layer)
        if mask_data is None:
            print("No mask data found to sync.")
            return

        if self.target_layer == "tumor":
            self.session_manager.set_tumor_mask(mask_data)
        elif self.target_layer == "organ":
            self.session_manager.set_organ_mask(mask_data)

        self._push_mask_to_all(self.target_layer, mask_data)
        print(f"Synced {self.target_layer} mask from active viewer to all.")

    def _on_auto_sync(self, layer_type: str):
        """Auto-sync after debounced paint/erase (300ms after last stroke)."""
        mask_data = self.layout_manager.get_active_mask_data(layer_type)
        if mask_data is None:
            return

        if layer_type == "tumor":
            self.session_manager.set_tumor_mask(mask_data)
        elif layer_type == "organ":
            self.session_manager.set_organ_mask(mask_data)

        # BUG-10 FIX: Use lightweight cache sync instead of full update_mask.
        # Visible viewers already share the painted data via _on_mask_data_changed.
        # This only updates caches and invalidates non-visible layouts.
        self.layout_manager.sync_mask_cache(mask_data, layer_type)
        
        # Dismiss stale lesion IDs if painting on tumor mask
        if layer_type == "tumor":
            self.layout_manager.hide_lesion_ids()
            self.control_panel.chk_show_lesion_ids.setChecked(False)

        print(f"[AutoSync] Synced {layer_type} mask after painting.")
