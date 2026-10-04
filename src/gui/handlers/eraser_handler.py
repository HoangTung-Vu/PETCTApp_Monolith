"""Eraser tool handler mixin for MainWindow."""


class EraserHandlerMixin:
    """Handles eraser mode toggle, region removal preview, and undo."""

    def _on_eraser_mode_toggled(self, enabled: bool):
        """Enable or disable eraser click mode on viewers."""
        if enabled:
            self.layout_manager.enable_eraser_click_mode()

            # Clear report UI and hide lesion IDs when eraser is enabled
            self._clear_all_report_ui()

            print("[Eraser] Mode enabled.")
        else:
            self.layout_manager.disable_eraser_click_mode()
            print("[Eraser] Mode disabled.")

    def _on_eraser_region_removed(self, erased_indices_zyx, mask_zyx):
        """Called after eraser removes a connected component. Preview only (no save).

        The ZYX viewer array is already zeroed; mirror the change into the XYZ
        session mask in place (only the component's voxels are touched).
        """
        z_idx, y_idx, x_idx = erased_indices_zyx
        shape_z, shape_y, _ = mask_zyx.shape
        # to_napari: z_nap = Z-1-z, y_nap = Y-1-y, x_nap = x
        erased_indices_xyz = (x_idx, shape_y - 1 - y_idx, shape_z - 1 - z_idx)

        # Limit undo stack depth to 5
        if len(self._eraser_undo_stack) >= 5:
            self._eraser_undo_stack.pop(0)
        self._eraser_undo_stack.append({"xyz": erased_indices_xyz, "zyx": erased_indices_zyx})

        self._set_tumor_voxels(erased_indices_xyz, erased_indices_zyx, 0)

        # Clear report UI and hide lesion IDs
        self._clear_all_report_ui()
        print(f"[Eraser] Preview updated. Undo stack depth: {len(self._eraser_undo_stack)}")

    def _on_eraser_undo(self):
        """Restore the mask by replaying the diff in reverse."""
        if not self._eraser_undo_stack:
            print("[Eraser] Nothing to undo.")
            return
        if self.session_manager.tumor_mask is None:
            print("[Eraser] No current mask to restore into.")
            return

        backup = self._eraser_undo_stack.pop()
        self._set_tumor_voxels(backup["xyz"], backup["zyx"], 1)
        self._clear_all_report_ui()
        print(f"[Eraser] Undo successful. Undo stack depth: {len(self._eraser_undo_stack)}")

    def _set_tumor_voxels(self, indices_xyz, indices_zyx, value: int):
        """Write ``value`` at the given voxels of the session and viewer masks, in place."""
        sm = self.session_manager
        current_mask = sm.get_tumor_mask_data()
        if current_mask is not None:
            current_mask[indices_xyz] = value
            sm.tumor_dirty = True
            sm.clear_lesion_data()
        mask_zyx = self.layout_manager._cached_data_zyx.get("tumor")
        if mask_zyx is not None:
            mask_zyx[indices_zyx] = value
        self.layout_manager.refresh_mask("tumor")
