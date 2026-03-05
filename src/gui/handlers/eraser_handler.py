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
        # BUG-16 FIX: Copy prob BEFORE mutation to avoid in-place corruption
        old_prob_data = self.session_manager.get_tumor_prob()
        prob_backup = old_prob_data.copy() if old_prob_data is not None else None

        backup = {
            "mask": old_mask_xyz,
            "prob": prob_backup,
        }
        # Limit undo stack depth to 5 to prevent memory leak
        if len(self._eraser_undo_stack) >= 5:
            self._eraser_undo_stack.pop(0)
        self._eraser_undo_stack.append(backup)

        # Find erased voxels (were 1, now 0) and zero out prob
        # BUG-16 FIX: Mutate a copy, then set it on the session manager
        if old_mask_xyz is not None and old_prob_data is not None:
            erased = (old_mask_xyz > 0) & (new_mask_xyz == 0)
            new_prob = old_prob_data.copy()
            new_prob[erased] = 0.0
            self.session_manager.set_tumor_prob(new_prob)
            print(f"[Eraser] Zeroed {int(np.sum(erased))} prob voxels.")

        # Update session manager with erased mask (in-memory only)
        # Note: Session manager also clears lesion ids internally
        self.session_manager.set_tumor_mask(new_mask_xyz)
        self._push_mask_to_all("tumor", new_mask_xyz)

        # BUG-03 FIX: Clear report UI and hide lesion IDs
        self.control_panel.clear_report_results()
        self.control_panel.chk_show_lesion_ids.setChecked(False)
        self.layout_manager.hide_lesion_ids()
        print(f"[Eraser] Preview updated. Undo stack depth: {len(self._eraser_undo_stack)}")

    def _on_eraser_undo(self):
        """Restore the mask and prob from before the last erase."""
        if not self._eraser_undo_stack:
            print("[Eraser] Nothing to undo.")
            return

        backup = self._eraser_undo_stack.pop()

        if backup["mask"] is not None:
            self.session_manager.set_tumor_mask(backup["mask"])
            self._push_mask_to_all("tumor", backup["mask"])

        if backup["prob"] is not None:
            self.session_manager.set_tumor_prob(backup["prob"])

        print(f"[Eraser] Undo successful. Undo stack depth: {len(self._eraser_undo_stack)}")
