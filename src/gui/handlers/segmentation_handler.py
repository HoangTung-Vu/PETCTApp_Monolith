"""Segmentation handler mixin for MainWindow."""

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ...utils.nifti_utils import MASK_DTYPE
from ..components.segmentation_progress_dialog import SegmentationProgressDialog


class SegmentationHandlerMixin:
    """Handles segmentation launch, progress dialog, and result processing."""

    def run_segmentation_dialog(self):
        """Entry point wired to the Segment button — runs the model chosen under AI Engine."""
        self._run_segmentation()

    def _run_segmentation(self):
        from ..workers import SegmentationWorker
        from ..workers.core import get_engine_model, is_totalseg_model

        sm = self.session_manager
        model = get_engine_model()
        organ = is_totalseg_model(model)

        # LAS volumes (the training orientation); unchanged LAS files are streamed.
        if organ:
            # TotalSegmentator segments the CT alone; the organ mask comes back on its grid.
            if sm.ct_image is None:
                QMessageBox.warning(self, "Missing Data", "TotalSegmentator requires a CT image.")
                return
            images, paths = [sm.ct_image], [sm.ct_stream_path]
        else:
            if not sm.ct_image or not sm.pet_image:
                QMessageBox.warning(
                    self, "Missing Data",
                    "Tumor segmentation requires both CT and PET images."
                )
                return
            images, paths = [sm.ct_image, sm.pet_image], [sm.ct_stream_path, sm.pet_stream_path]
        self.worker = SegmentationWorker(images, paths, model=model,
                                         kind="organ" if organ else "tumor")
        # The Workflow tab stays usable during inference: a result that arrives
        # after a session switch belongs to the old session and is dropped.
        self._seg_session_id = sm.current_session_id

        # Non-modal status dialog: upload → inference % → done. The user can keep
        # viewing/panning/zooming while it runs.
        self._seg_progress_dialog = SegmentationProgressDialog(
            self, title="Organ Segmentation (TotalSegmentator)" if organ else "Tumor Segmentation")
        self.worker.uploaded.connect(self._seg_progress_dialog.set_uploaded)
        self.worker.progress.connect(self._seg_progress_dialog.set_progress)
        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)

        # Lock only the Segmentation tab; all viewing stays open.
        self._lock_seg_tab(True)
        self.control_panel.workflow_tab.btn_segment.setEnabled(False)
        self._seg_progress_dialog.show()
        self.worker.start()

    def _on_segmentation_finished(self, result_tuple):
        mask_img, model, kind = result_tuple
        # Header shape only — the organ volume is decoded in OrganSaveWorker,
        # not on the GUI thread.
        shape = tuple(mask_img.shape[:3])

        if self.session_manager.current_session_id != getattr(self, "_seg_session_id", None):
            self._on_segmentation_error(
                "The session changed while the segmentation was running; "
                "its result was not applied. Run it again on this session."
            )
            return
        ct_img = self.session_manager.ct_image
        if ct_img is None or shape != tuple(ct_img.shape):
            self._on_segmentation_error(
                f"The engine returned a mask of shape {shape}, "
                f"but the image grid is {None if ct_img is None else tuple(ct_img.shape)}."
            )
            return

        if kind == "organ":
            # Organ labels are not lesions: written to <CT>_organ.nii.gz only,
            # the tumor mask and the lesion values stay as they are.
            from ..workers import OrganSaveWorker
            self._organ_save_worker = OrganSaveWorker(self.session_manager, mask_img)
            self._spawn_worker(self._organ_save_worker, self._on_organ_saved, self._on_save_error)
        else:
            # The model returns a binary mask; it replaces a labelled one (the
            # user gives the lesion ids again).
            data = np.asarray(mask_img.dataobj, dtype=MASK_DTYPE)
            self.session_manager.set_tumor_mask(data, labeled=False)
            self._push_mask_to_all("tumor", data)
            # Save immediately to this model's own file, <CT>_Segmentation_<tag>.nii.gz
            # with a short tag (nnUNet), not the long model name in the URL
            # (the save worker also computes the lesion values).
            from ..workers.core import model_file_tag
            self.save_session(
                seg_path=self.session_manager.model_segmentation_path(model_file_tag(model)))
        print(f"Segmentation {kind} ({model}) finished.")

        self._close_seg_progress_dialog()
        self._lock_seg_tab(False)
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)

    def _on_organ_saved(self, path: str):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        # Free the engine's decoded copy (the thread object itself is kept: its
        # run() may still be returning).
        self._organ_save_worker.organ_img = None
        # Builds the organ layers only if "Show Organ Mask" is on (off by default).
        self.layout_manager.set_organ_mask(self.session_manager.get_organ_mask_data())
        shown = self.control_panel.view_display_tab.chk_organ.isChecked()
        QMessageBox.information(
            self, "Organ Segmentation",
            f"TotalSegmentator organ mask saved to:\n{path}"
            + ("" if shown else "\n\nTurn on \"Show Organ Mask\" (View & Display → Zoom & Mask) to see it."))

    def _push_mask_to_all(self, layer_type, data, data_zyx=None):
        self.layout_manager.update_mask(data, layer_type, data_zyx)

    def _on_segmentation_error(self, error_msg):
        self._close_seg_progress_dialog()
        self._lock_seg_tab(False)
        self.control_panel.workflow_tab.btn_segment.setEnabled(True)
        print(f"Segmentation Error: {error_msg}")
        QMessageBox.critical(self, "Segmentation Failed", error_msg)

    def _close_seg_progress_dialog(self):
        dlg = getattr(self, "_seg_progress_dialog", None)
        if dlg is not None:
            dlg.complete()
            self._seg_progress_dialog = None
