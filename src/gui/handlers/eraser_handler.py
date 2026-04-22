"""Eraser tool handler mixin for MainWindow."""

import numpy as np


class EraserHandlerMixin:
    """Handles eraser mode toggle, region removal preview, and undo."""

    def _on_eraser_mode_toggled(self, enabled: bool):
        """Enable or disable eraser click mode on viewers."""
        if enabled:
            self.layout_manager.enable_eraser_click_mode()

            # Clear report UI and hide lesion IDs when eraser is enabled
            self.session_manager.clear_lesion_data()
            self.control_panel.clear_report_results()
            self.control_panel.chk_show_lesion_ids.setChecked(False)
            self.layout_manager.hide_lesion_ids()

            print("[Eraser] Mode enabled.")
        else:
            self.layout_manager.disable_eraser_click_mode()
            print("[Eraser] Mode disabled.")

    def _on_eraser_region_removed(self, erased_indices_zyx, component_mask_zyx, mask_zyx=None):
        """Called after eraser removes a connected component. Preview only (no save)."""
        z_idx, y_idx, x_idx = erased_indices_zyx
        shape_z, shape_y, shape_x = mask_zyx.shape
        
        # Convert ZYX back to XYZ to store in the undo stack
        z_new = shape_z - 1 - z_idx
        y_new = shape_y - 1 - y_idx
        x_new = x_idx
        erased_indices_xyz = (x_new, y_new, z_new)

        backup = {
            "indices": erased_indices_xyz,
            "shape": mask_zyx.shape,
        }

        # Limit undo stack depth to 5
        if len(self._eraser_undo_stack) >= 5:
            self._eraser_undo_stack.pop(0)
        self._eraser_undo_stack.append(backup)

        # Update session manager with erased mask (in-memory only)
        current_mask = self.session_manager.get_tumor_mask_data()
        if current_mask is not None:
             current_mask[erased_indices_xyz] = 0
             self.session_manager.set_tumor_mask(current_mask)

        # Synchronize efficiently: the ZYX memory is already zeroed in eraser_manager,
        # so sync_mask_cache will just re-trigger layer.refresh() on visible viewers!
        if current_mask is not None:
             self.layout_manager.sync_mask_cache(current_mask, "tumor")

        # Clear report UI and hide lesion IDs
        self.control_panel.clear_report_results()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        self.layout_manager.hide_lesion_ids()
        print(f"[Eraser] Preview updated. Undo stack depth: {len(self._eraser_undo_stack)}")

    def _on_eraser_undo(self):
        """Restore the mask by replaying the diff in reverse."""
        if not self._eraser_undo_stack:
            print("[Eraser] Nothing to undo.")
            return

        backup = self._eraser_undo_stack.pop()

        # Reconstruct mask by re-setting the erased voxels to 1
        current_mask = self.session_manager.get_tumor_mask_data()
        if current_mask is None:
            print("[Eraser] No current mask to restore into.")
            return

        restored_mask = current_mask.copy()
        restored_mask[backup["indices"]] = 1

        self.session_manager.set_tumor_mask(restored_mask)
        self._push_mask_to_all("tumor", restored_mask)

        print(f"[Eraser] Undo successful. Undo stack depth: {len(self._eraser_undo_stack)}")
