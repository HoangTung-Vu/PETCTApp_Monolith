"""Lesion label handler mixin for MainWindow — the Segmentation tab's
"Lesion Label" section, "Give lesion ids", and the "same lesion?" prompt.

The selected id rules the whole Segmentation tab: the brush paints it, the
erase brush removes only it, ROI thresholding merges into it and the
iterative threshold treats the other ids as background.
"""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

from ...core import lesion_labels


class LesionLabelHandlerMixin:

    def _init_lesion_label_state(self):
        self._tumor_label = 1

    def _active_tumor_label(self) -> int:
        """Lesion id the tools write: the selected one; always 1 on a binary mask."""
        return self._tumor_label if self.session_manager.tumor_labeled else 1

    # ── Selector ──

    def _refresh_label_selector(self):
        """Rebuild the toggles from the session's ids (after load / save / merge)."""
        sm = self.session_manager
        self.layout_manager.set_tumor_labeled(sm.tumor_labeled)
        self.control_panel.refine_tab.set_label_choices(
            sm.tumor_label_ids, sm.tumor_labeled, self._tumor_label,
            has_mask=sm.tumor_mask is not None,
        )

    def _on_tumor_label_selected(self, label_id: int):
        self._tumor_label = int(label_id)
        self.layout_manager.set_tumor_label(self._active_tumor_label())

    # ── "Same lesion?" ──

    def _ask_same_lesion(self, label_id: int, covered: list, what: str) -> bool:
        """Yes = the covered lesions become ``label_id``; No = they keep their ids."""
        names = ", ".join(str(i) for i in covered)
        plural = len(covered) > 1
        reply = QMessageBox.question(
            self, "Same lesion?",
            f"{what} of lesion {label_id} covers lesion{'s' if plural else ''} {names}.\n\n"
            f"Do you really want to override {'them' if plural else 'it'} and treat "
            f"{'them' if plural else 'it'} as the same lesion as {label_id}?\n\n"
            f"Yes — lesion {names} becomes lesion {label_id} (all of its voxels).\n"
            f"No — lesion {names} is kept; only the background is painted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_tumor_overwrite(self, atoms, label_id: int, covered):
        # Not inside napari's mouse handler: a modal dialog there re-enters vispy.
        QTimer.singleShot(0, lambda: self._resolve_tumor_overwrite(atoms, label_id, list(covered)))

    def _resolve_tumor_overwrite(self, atoms, label_id: int, covered: list):
        data = self.layout_manager.get_active_mask_data_zyx("tumor")
        if data is None:
            return
        merge = self._ask_same_lesion(label_id, covered, "The brush")
        lesion_labels.resolve_overwrite(data, atoms, label_id, merge)
        self.layout_manager.refresh_mask("tumor")
        sm = self.session_manager
        sm.tumor_dirty = True
        self._mark_lesion_stats_stale()
        if merge:
            sm.tumor_label_ids = [i for i in sm.tumor_label_ids if i not in covered]
            self._refresh_label_selector()
        print(f"[LesionLabel] Stroke of lesion {label_id} over {covered}: "
              f"{'merged' if merge else 'kept'}")

    def _roi_merge_ids(self, roi_xyz) -> tuple:
        """Lesions under the ROI that the user wants merged into the selected one."""
        sm = self.session_manager
        if roi_xyz is None or not sm.tumor_labeled or sm.tumor_mask is None:
            return ()
        label_id = self._active_tumor_label()
        covered = lesion_labels.roi_conflicts(sm.get_tumor_mask_data(), roi_xyz, label_id)
        if covered and self._ask_same_lesion(label_id, covered, "The ROI"):
            return tuple(covered)
        return ()

    # ── Give lesion ids ──

    def _on_give_lesion_ids(self):
        sm = self.session_manager
        if sm.tumor_mask is None:
            QMessageBox.information(self, "Give lesion ids", "No segmentation loaded.")
            return
        self._sync_tumor_from_viewer()
        if sm.tumor_labeled:
            reply = QMessageBox.question(
                self, "Give lesion ids",
                "Lesion ids are given again from scratch: every lesion voxel counts, each "
                "connected region becomes one lesion and the ids run 1..N from head to feet.\n\n"
                "Lesions you merged are split again. The segmentation file is overwritten.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        from ..workers import GiveIdsWorker
        self._give_ids_worker = GiveIdsWorker(sm)
        self._give_ids_worker.mask_ready.connect(
            lambda xyz, zyx: self._push_mask_to_all("tumor", xyz, data_zyx=zyx))
        self._spawn_worker(self._give_ids_worker, self._on_give_ids_finished, self._on_give_ids_error)

    def _on_give_ids_finished(self):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        self._refresh_lesion_ui()      # values, table and the new toggles
        n = len(self.session_manager.tumor_label_ids)
        print(f"[LesionLabel] Lesion ids given: {n} lesion(s).")

    def _on_give_ids_error(self, msg: str):
        self._show_worker_error(msg, "Give Lesion Ids Failed")
