"""Segmentation handler mixin for MainWindow."""

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..components.segmentation_progress_dialog import SegmentationProgressDialog


class SegmentationHandlerMixin:
    """Handles segmentation launch, progress dialog, and result processing."""

    def run_segmentation_dialog(self):
        """Entry point wired to the Segment button — runs the custom nnUNet model."""
        self._run_segmentation()

    def _run_segmentation(self):
        ct_img = self.session_manager.ct_image
        pet_img = self.session_manager.pet_image

        from ..workers import SegmentationWorker

        if not ct_img or not pet_img:
            QMessageBox.warning(
                self, "Missing Data",
                "Tumor segmentation requires both CT and PET images."
            )
            return

        input_data = [ct_img, pet_img]

        self.worker = SegmentationWorker(input_data)

        # Non-modal status dialog: upload → inference % → done. The user can keep
        # viewing/panning/zooming while it runs.
        self._seg_progress_dialog = SegmentationProgressDialog(self)
        self.worker.uploaded.connect(self._seg_progress_dialog.set_uploaded)
        self.worker.progress.connect(self._seg_progress_dialog.set_progress)
        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)

        # Lock only the Segmentation (Refine) + Eraser tabs; all viewing stays open.
        self._lock_seg_eraser_tabs(True)
        self.control_panel.workflow_tab.btn_segment.setEnabled(False)
        self._seg_progress_dialog.show()
        self.worker.start()

    def _on_segmentation_finished(self, result_tuple):
        mask_img, _prob_array, seg_type = result_tuple
        data = np.asarray(mask_img.dataobj, dtype=np.uint8)

        self.session_manager.set_tumor_mask(data)
        self._push_mask_to_all("tumor", data)

        # Clear stale report UI and lesion
        self._clear_all_report_ui()

        # Save immediately
        self.save_session()
        print(f"Segmentation {seg_type} finished and saved.")

        self._close_seg_progress_dialog()
        self._lock_seg_eraser_tabs(False)
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)

    def _push_mask_to_all(self, layer_type, data, data_zyx=None):
        self.layout_manager.update_mask(data, layer_type, data_zyx)
        # Hide lesion IDs since mask changed
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

    def _on_segmentation_error(self, error_msg):
        self._close_seg_progress_dialog()
        self._lock_seg_eraser_tabs(False)
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)
        print(f"Segmentation Error: {error_msg}")
        QMessageBox.critical(self, "Segmentation Failed", error_msg)

    def _close_seg_progress_dialog(self):
        dlg = getattr(self, "_seg_progress_dialog", None)
        if dlg is not None:
            dlg.complete()
            self._seg_progress_dialog = None
