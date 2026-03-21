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

    def _on_eraser_region_removed(self, old_mask_xyz, new_mask_xyz):
        """Called after eraser removes a connected component. Preview only (no save)."""
        old_prob_data = self.session_manager.get_tumor_prob()

        # Store only the DIFF (erased voxel indices + their old prob values).
        # This is O(erased voxels) vs O(full volume) — ~100x smaller for typical lesion removals.
        erased = (old_mask_xyz > 0) & (new_mask_xyz == 0)
        erased_indices = np.nonzero(erased)  # tuple of index arrays

        snapshot = self.session_manager._tumor_mask_snapshot
        backup = {
            "indices": erased_indices,
            "shape": old_mask_xyz.shape,
            "prob_values": old_prob_data[erased].copy() if old_prob_data is not None else None,
            "snapshot_values": snapshot[erased_indices].copy() if snapshot is not None else None,
        }

        # Limit undo stack depth to 5
        if len(self._eraser_undo_stack) >= 5:
            self._eraser_undo_stack.pop(0)
        self._eraser_undo_stack.append(backup)

        # Patch snapshot so erased voxels no longer block ROI diff on Refine tab
        if snapshot is not None:
            snapshot[erased_indices] = 0

        # Zero out prob for erased voxels
        if old_prob_data is not None:
            new_prob = old_prob_data.copy()
            new_prob[erased] = 0.0
            self.session_manager.set_tumor_prob(new_prob)
            print(f"[Eraser] Zeroed {int(np.sum(erased))} prob voxels.")

        # Update session manager with erased mask (in-memory only)
        self.session_manager.set_tumor_mask(new_mask_xyz)
        self._push_mask_to_all("tumor", new_mask_xyz)

        # Clear report UI and hide lesion IDs
        self.control_panel.clear_report_results()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        self.layout_manager.hide_lesion_ids()
        print(f"[Eraser] Preview updated. Undo stack depth: {len(self._eraser_undo_stack)}")

    def _on_eraser_undo(self):
        """Restore the mask and prob by replaying the diff in reverse."""
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

        # Restore snapshot at the formerly-erased voxels
        if backup["snapshot_values"] is not None:
            snapshot = self.session_manager._tumor_mask_snapshot
            if snapshot is not None:
                snapshot[backup["indices"]] = backup["snapshot_values"]

        # Restore prob values at the formerly-erased voxels
        if backup["prob_values"] is not None:
            current_prob = self.session_manager.get_tumor_prob()
            if current_prob is not None:
                restored_prob = current_prob.copy()
            else:
                restored_prob = np.zeros(backup["shape"], dtype=np.float32)
            restored_prob[backup["indices"]] = backup["prob_values"]
            self.session_manager.set_tumor_prob(restored_prob)

        print(f"[Eraser] Undo successful. Undo stack depth: {len(self._eraser_undo_stack)}")
