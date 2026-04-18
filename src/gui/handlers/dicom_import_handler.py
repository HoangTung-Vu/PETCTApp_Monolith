"""
DicomImportHandlerMixin — wires DICOM conversion worker to MainWindow.

Handles:
  - Starting DicomConversionWorker when user clicks "Run Conversion"
  - Forwarding log messages to the tab's log area
  - On success: enabling "Load into Session" in the tab
  - On "Load into Session" click: creating a new session + loading NIfTI files
"""

from pathlib import Path
from PyQt6.QtWidgets import QMessageBox


class DicomImportHandlerMixin:

    def _init_dicom_import_handler(self):
        """Call from MainWindow.__init__ after UI is set up."""
        self.dicom_worker = None

    # ------------------------------------------------------------------
    # Entry point — called by ControlPanel signal
    # ------------------------------------------------------------------

    def _on_dicom_run_conversion(
        self,
        dcm_root: str,
        out_dir: str,
        pid_str: str,
        do_suv: bool,
        do_resample: bool,
    ):
        from ..workers.dicom_conversion_worker import DicomConversionWorker

        self.dicom_worker = DicomConversionWorker(
            dcm_root=dcm_root,
            out_dir=out_dir,
            pid_str=pid_str,
            do_suv=do_suv,
            do_resample=do_resample,
        )
        self.dicom_worker.sig_log.connect(self._on_dicom_log)
        self.dicom_worker.sig_finished.connect(self._on_dicom_finished)
        self.dicom_worker.sig_error.connect(self._on_dicom_error)

        self.control_panel.dicom_import_tab.show_progress()
        self._set_ui_busy(True)
        self.dicom_worker.start()

    def _on_dicom_log(self, msg: str):
        self.control_panel.dicom_import_tab.append_log(msg)

    def _on_dicom_finished(self, ct_path: str, pet_path: str):
        self._set_ui_busy(False)
        self.control_panel.dicom_import_tab.hide_progress()
        self.control_panel.dicom_import_tab.on_conversion_finished(ct_path, pet_path)

    def _on_dicom_error(self, error_msg: str):
        self._set_ui_busy(False)
        self.control_panel.dicom_import_tab.hide_progress()
        self.control_panel.dicom_import_tab.append_log(f"ERROR: {error_msg}")
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

        # Reset viewers / state, then create a new session with the NIfTI paths
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

        # Switch to Workflow tab so user sees the progress bar there
        self.control_panel.tabs.setCurrentIndex(
            self.control_panel.tabs.indexOf(self.control_panel.workflow_tab)
        )
