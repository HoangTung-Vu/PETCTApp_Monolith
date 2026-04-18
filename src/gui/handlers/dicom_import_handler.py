"""
DicomImportHandlerMixin — DICOM conversion triggered from the Workflow tab.

Flow:
  1. User clicks "Load from DICOM Folder…" in the Workflow tab.
  2. _on_load_from_dicom starts DicomConversionWorker (background thread).
  3. On success, _on_dicom_auto_finished immediately creates a new session
     with the converted NIfTI files — no extra click required.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox


class DicomImportHandlerMixin:

    def _init_dicom_import_handler(self):
        """Call from MainWindow.__init__ after UI is set up."""
        self.dicom_worker = None
        self._dicom_auto_doctor  = "Doctor"
        self._dicom_auto_patient = "Patient"

    # ------------------------------------------------------------------
    # Entry point — Workflow tab "Load from DICOM Folder…"
    # ------------------------------------------------------------------

    def _on_load_from_dicom(self, dcm_folder: str, doctor: str, patient: str, resample_mode: str = "ct"):
        from ..workers.dicom_conversion_worker import DicomConversionWorker

        out_dir = os.path.join(dcm_folder, "nifti_output")
        pid_str = os.path.basename(dcm_folder) or "PATIENT"

        self._dicom_auto_doctor  = doctor  or "Doctor"
        self._dicom_auto_patient = patient or "Patient"

        self.dicom_worker = DicomConversionWorker(
            dcm_root=dcm_folder,
            out_dir=out_dir,
            pid_str=pid_str,
            do_suv=True,
            do_resample=True,
            resample_mode=resample_mode,
        )
        self.dicom_worker.sig_log.connect(self._on_dicom_log)
        self.dicom_worker.sig_finished.connect(self._on_dicom_auto_finished)
        self.dicom_worker.sig_error.connect(self._on_dicom_error)

        self.control_panel.show_progress()
        self._set_ui_busy(True)
        self.dicom_worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_dicom_log(self, msg: str):
        print(f"[DICOM] {msg}")

    def _on_dicom_auto_finished(self, ct_path: str, pet_path: str):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        self._on_dicom_load_into_session(
            ct_path, pet_path,
            self._dicom_auto_doctor,
            self._dicom_auto_patient,
        )

    def _on_dicom_error(self, error_msg: str):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        QMessageBox.critical(self, "DICOM Conversion Failed", error_msg)

    # ------------------------------------------------------------------
    # Load converted NIfTI into a new session
    # ------------------------------------------------------------------

    def _on_dicom_load_into_session(
        self, ct_path: str, pet_path: str, doctor: str, patient: str
    ):
        if not ct_path and not pet_path:
            QMessageBox.warning(
                self,
                "Nothing to Load",
                "No converted files available. Run conversion first.",
            )
            return

        self._reset_all_state()

        from ..workers.data_loader_worker import DataLoaderWorker

        self.loader_worker = DataLoaderWorker(
            self.session_manager,
            action="create",
            ct_path=Path(ct_path)  if ct_path  else None,
            pet_path=Path(pet_path) if pet_path else None,
            new_doctor=doctor  or "Doctor",
            new_patient=patient or "Patient",
        )
        self.loader_worker.finished.connect(self._on_data_loaded)
        self.loader_worker.error.connect(self._on_data_error)
        self.control_panel.show_progress()
        self._set_ui_busy(True)
        self.loader_worker.start()

        # Switch to Workflow tab so user sees the progress bar
        self.control_panel.tabs.setCurrentIndex(
            self.control_panel.tabs.indexOf(self.control_panel.workflow_tab)
        )
