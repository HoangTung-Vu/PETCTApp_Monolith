"""AutoPET Interactive handler mixin for MainWindow."""

import numpy as np
from PyQt6.QtWidgets import QMessageBox


class AutoPETHandlerMixin:
    """Handles AutoPET Interactive click tracking, inference, and result merging."""

    def _on_autopet_click_added(self, coord_zyx, label):
        """Track a new click and update the UI list."""
        click = {"point": coord_zyx, "name": label}
        self.autopet_clicks.append(click)
        self.control_panel.add_autopet_click_item(coord_zyx, label)
        
        # Dismiss stale lesion IDs when adding clicks
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

        print(f"[AutoPET] Added click #{len(self.autopet_clicks)}: {label} at {coord_zyx}")

    def _on_autopet_clear_clicks(self):
        """Clear all tracked clicks and viewer markers."""
        self.autopet_clicks.clear()
        self.layout_manager.clear_autopet_clicks()
        print("[AutoPET] Cleared all clicks.")

    def _on_autopet_run(self):
        """Run AutoPET Interactive inference with collected clicks."""

        ct_img = self.session_manager.ct_image
        pet_img = self.session_manager.pet_image

        if not ct_img or not pet_img:
            QMessageBox.warning(
                self, "Missing Data",
                "AutoPET Interactive requires both CT and PET images."
            )
            return

        from ..workers import AutoPETWorker
        self.autopet_worker = AutoPETWorker(ct_img, pet_img, list(self.autopet_clicks))
        self.autopet_worker.finished.connect(self._on_autopet_finished)
        self.autopet_worker.error.connect(self._on_autopet_error)

        self.control_panel.show_autopet_progress()
        self._set_ui_busy(True)
        self.autopet_worker.start()

    def _on_autopet_finished(self, refinement_prob):
        """Combine AutoPET prob with existing nnUNet prob."""
        print(f"[AutoPET] Inference finished!")
        print(f"[AutoPET] Refinement prob shape: {refinement_prob.shape}, dtype: {refinement_prob.dtype}")

        old_prob = self.session_manager.get_tumor_prob()
        if old_prob is not None:
            combined_prob = (old_prob + refinement_prob) / 2.0
        else:
            combined_prob = refinement_prob

        new_mask = (combined_prob >= 0.5).astype(np.uint8)
        print(f"[AutoPET] New mask nonzero voxels: {np.count_nonzero(new_mask)}")

        self.session_manager.set_tumor_mask(new_mask)
        self.session_manager.set_tumor_prob(combined_prob)
        self._push_mask_to_all("tumor", new_mask)

        # BUG-05 FIX: Clear stale report UI and lesion data
        self.session_manager.clear_lesion_data()
        self.control_panel.clear_report_results()

        self.session_manager.save_session()
        # BUG-J FIX: Re-snapshot after commit so tab-switch revert uses the new baseline
        self.session_manager.snapshot_current_mask("tumor")
        print("[AutoPET] Session saved and snapshot updated.")

        self.layout_manager.clear_autopet_clicks()
        self.autopet_clicks.clear()
        self.control_panel.clear_autopet_click_list()

        self.control_panel.hide_autopet_progress()
        self._set_ui_busy(False)

    def _on_autopet_error(self, error_msg):
        self._set_ui_busy(False)
        self.control_panel.hide_autopet_progress()
        print(f"[AutoPET] Error: {error_msg}")
        QMessageBox.critical(self, "AutoPET Failed", error_msg)
