"""Segmentation handler mixin for MainWindow."""

from PyQt6.QtWidgets import QInputDialog, QMessageBox


class SegmentationHandlerMixin:
    """Handles segmentation dialog, worker launch, and result processing."""

    def run_segmentation_dialog(self):
        """Ask user which segmentation to run."""
        items = ["Tumor Segmentation (nnUNet)", "Organ Segmentation (TotalSegmentator)"]
        item, ok = QInputDialog.getItem(
            self, "Select Segmentation", "Choose model:", items, 0, False
        )
        if ok and item:
            if "Tumor" in item:
                self._run_segmentation("tumor")
            else:
                self._run_segmentation("organ")

    def _run_segmentation(self, seg_type: str):
        ct_img = self.session_manager.ct_image
        pet_img = self.session_manager.pet_image

        from ..workers import SegmentationWorker

        if seg_type == "tumor":
            if not ct_img or not pet_img:
                QMessageBox.warning(
                    self, "Missing Data",
                    "Tumor segmentation requires both CT and PET images."
                )
                return
            input_data = [ct_img, pet_img]

        elif seg_type == "organ":
            if not ct_img:
                QMessageBox.warning(
                    self, "Missing Data",
                    "Organ segmentation requires a CT image."
                )
                return
            input_data = ct_img
        else:
            return

        self.worker = SegmentationWorker(seg_type, input_data)
        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)

        # BUG-13 FIX: Disable segment button during inference
        self.control_panel.workflow_tab.btn_segment.setEnabled(False)
        self.control_panel.show_progress()
        self.worker.start()

    def _on_segmentation_finished(self, result_tuple):
        mask_img, prob_array, seg_type = result_tuple
        data = mask_img.get_fdata()

        if seg_type == "tumor":
            self.session_manager.set_tumor_mask(data)
            if prob_array is not None:
                self.session_manager.set_tumor_prob(prob_array)
            self._push_mask_to_all("tumor", data)
        elif seg_type == "organ":
            self.session_manager.set_organ_mask(data)
            self._push_mask_to_all("organ", data)

        # BUG-05 FIX: Clear stale report UI since mask changed
        self.control_panel.clear_report_results()

        self.session_manager.save_session()
        print(f"Segmentation {seg_type} finished and saved.")

        self.control_panel.hide_progress()
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)

    def _push_mask_to_all(self, layer_type, data):
        self.layout_manager.update_mask(data, layer_type)
        # Hide lesion IDs since mask changed
        self.layout_manager.hide_lesion_ids()
        self.control_panel.chk_show_lesion_ids.setChecked(False)

    def _on_segmentation_error(self, error_msg):
        self.control_panel.hide_progress()
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)
        print(f"Segmentation Error: {error_msg}")
        QMessageBox.critical(self, "Segmentation Failed", error_msg)
