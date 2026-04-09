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

    def _on_eraser_region_removed(self, old_mask_xyz, new_mask_xyz, mask_zyx=None):
        """Called after eraser removes a connected component. Preview only (no save)."""
        # Store only the DIFF (erased voxel indices).
        erased = (old_mask_xyz > 0) & (new_mask_xyz == 0)
        erased_indices = np.nonzero(erased)

        backup = {
            "indices": erased_indices,
            "shape": old_mask_xyz.shape,
        }

        # Limit undo stack depth to 5
        if len(self._eraser_undo_stack) >= 5:
            self._eraser_undo_stack.pop(0)
        self._eraser_undo_stack.append(backup)

        # Update session manager with erased mask (in-memory only)
        self.session_manager.set_tumor_mask(new_mask_xyz)
        self._push_mask_to_all("tumor", new_mask_xyz, data_zyx=mask_zyx)

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
