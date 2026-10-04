"""
DicomImportHandlerMixin — DICOM import triggered from the Workflow tab.

Flow:
  1. User clicks "Load from DICOM Folder…" in the Workflow tab.
  2. DicomScanWorker reads the headers and lists the CT / PET series.
  3. DicomSeriesDialog lets the user pick the series (auto-picks preselected)
     and the common grid; patient/physician names come from the DICOM tags.
  4. DataLoaderWorker(action="create_dicom") reads the series into memory and
     creates the session — no NIfTI files are written.
"""

from PyQt6.QtWidgets import QDialog, QMessageBox


class DicomImportHandlerMixin:

    def _init_dicom_import_handler(self):
        """Call from MainWindow.__init__ after UI is set up."""
        self.dicom_worker = None
        self._dicom_folder = ""
        self._dicom_manual_names = ("", "")

    # ------------------------------------------------------------------
    # Entry point — Workflow tab "Load from DICOM Folder…"
    # ------------------------------------------------------------------

    def _on_load_from_dicom(self, dcm_folder: str, doctor: str = "", patient: str = ""):
        from ..workers.dicom_scan_worker import DicomScanWorker

        self._dicom_folder = dcm_folder
        self._dicom_manual_names = (doctor, patient)
        self.dicom_worker = DicomScanWorker(dcm_folder)
        self._spawn_worker(self.dicom_worker, self._on_dicom_scanned, self._on_dicom_error)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_dicom_scanned(self, series: list):
        self._set_ui_busy(False)
        self.control_panel.hide_progress()
        if not series:
            QMessageBox.warning(
                self,
                "No DICOM Series",
                "No usable CT or PET series were found in the selected folder.\n\n"
                "Scouts/localizers, multi-frame and dynamic series are skipped "
                "(see the Logs tab for details).",
            )
            return

        from ..components.dicom_series_dialog import DicomSeriesDialog
        dialog = DicomSeriesDialog(series, self._dicom_folder, *self._dicom_manual_names, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        doctor, patient = dialog.names()
        self._load_dicom_session(dialog.selected_ct, dialog.selected_pet, dialog.resample_mode, doctor, patient)

    def _on_dicom_error(self, error_msg: str):
        self._show_worker_error(error_msg, "DICOM Import Failed")

    # ------------------------------------------------------------------
    # Load the chosen series into a new session
    # ------------------------------------------------------------------

    def _load_dicom_session(self, ct_series, pet_series, resample_mode: str, doctor: str, patient: str):
        # DICOM import creates a new session — prompt to save unsaved tumor first.
        if not self._prompt_unsaved_segmentation("switch"):
            return

        self._reset_all_state()

        from ..workers.data_loader_worker import DataLoaderWorker

        self.loader_worker = DataLoaderWorker(
            self.session_manager,
            action="create_dicom",
            dicom_ct=ct_series,
            dicom_pet=pet_series,
            resample_mode=resample_mode,
            new_doctor=doctor,
            new_patient=patient,
        )
        self._spawn_worker(self.loader_worker, self._on_data_loaded, self._on_data_error)

        # Switch to Workflow tab so user sees the progress bar
        self.control_panel.tabs.setCurrentIndex(
            self.control_panel.tabs.indexOf(self.control_panel.workflow_tab)
        )
