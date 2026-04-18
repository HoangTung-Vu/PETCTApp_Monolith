"""DICOM Import tab — convert a DICOM folder to NIfTI and load into session."""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFormLayout,
    QGroupBox, QLineEdit, QProgressBar, QLabel, QCheckBox,
    QPlainTextEdit, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class DicomImportTab(QWidget):
    """
    Provides a UI for:
      1. Selecting a DICOM source folder (auto-detects CT + PET series)
      2. Choosing an output NIfTI folder
      3. Setting conversion options (SUV, resample)
      4. Naming the new session (doctor / patient)
      5. Running conversion in background and loading the result
    """

    # Emitted when user clicks "Run Conversion"
    sig_run_conversion = pyqtSignal(
        str,   # dcm_root
        str,   # out_dir
        str,   # pid_str  (used as filename prefix)
        bool,  # do_suv
        bool,  # do_resample
    )

    # Emitted when user clicks "Load into Session"
    # Carries (ct_path, pet_path, doctor, patient)
    sig_load_into_session = pyqtSignal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ct_result  = ""
        self._pet_result = ""
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Source ──────────────────────────────────────────────────────
        grp_src = QGroupBox("DICOM Source")
        form_src = QFormLayout()

        self.input_dcm_folder = QLineEdit()
        self.input_dcm_folder.setPlaceholderText("Folder containing .dcm files or series sub-folders")
        btn_browse_dcm = QPushButton("Browse…")
        btn_browse_dcm.setFixedWidth(80)
        btn_browse_dcm.clicked.connect(self._browse_dcm)
        row_dcm = QHBoxLayout()
        row_dcm.addWidget(self.input_dcm_folder)
        row_dcm.addWidget(btn_browse_dcm)
        form_src.addRow("Input folder:", row_dcm)

        grp_src.setLayout(form_src)
        layout.addWidget(grp_src)

        # ── Output ──────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        form_out = QFormLayout()

        self.input_out_folder = QLineEdit()
        self.input_out_folder.setPlaceholderText("Where to write .nii.gz files")
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setFixedWidth(80)
        btn_browse_out.clicked.connect(self._browse_out)
        row_out = QHBoxLayout()
        row_out.addWidget(self.input_out_folder)
        row_out.addWidget(btn_browse_out)
        form_out.addRow("Output folder:", row_out)

        self.input_pid = QLineEdit()
        self.input_pid.setPlaceholderText("e.g. 0001  (used as filename prefix)")
        form_out.addRow("Patient prefix:", self.input_pid)

        grp_out.setLayout(form_out)
        layout.addWidget(grp_out)

        # ── Options ─────────────────────────────────────────────────────
        grp_opt = QGroupBox("Conversion Options")
        opt_layout = QVBoxLayout()

        self.chk_suv = QCheckBox("Compute SUV (requires radiopharmaceutical DICOM tags)")
        self.chk_suv.setChecked(True)
        self.chk_resample = QCheckBox("Upsample PET/SUV to CT grid after conversion")
        self.chk_resample.setChecked(True)

        opt_layout.addWidget(self.chk_suv)
        opt_layout.addWidget(self.chk_resample)
        grp_opt.setLayout(opt_layout)
        layout.addWidget(grp_opt)

        # ── Session ──────────────────────────────────────────────────────
        grp_session = QGroupBox("New Session Info")
        form_session = QFormLayout()

        self.input_doctor  = QLineEdit()
        self.input_patient = QLineEdit()
        form_session.addRow("Doctor:",  self.input_doctor)
        form_session.addRow("Patient:", self.input_patient)

        grp_session.setLayout(form_session)
        layout.addWidget(grp_session)

        # ── Actions ──────────────────────────────────────────────────────
        self.btn_run = QPushButton("Run Conversion")
        self.btn_run.setFixedHeight(36)
        self.btn_run.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Log ──────────────────────────────────────────────────────────
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(130)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.log_area.setFont(mono)
        layout.addWidget(self.log_area)

        # ── Load result ──────────────────────────────────────────────────
        self.btn_load = QPushButton("Load into Session")
        self.btn_load.setFixedHeight(36)
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.btn_load)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet("color: #5cb85c; font-size: 11px;")
        layout.addWidget(self.lbl_result)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Slots / helpers
    # ------------------------------------------------------------------

    def _browse_dcm(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.input_dcm_folder.setText(folder)
            # Auto-fill output folder alongside input if empty
            if not self.input_out_folder.text():
                self.input_out_folder.setText(os.path.join(folder, "nifti_output"))
            # Auto-fill patient prefix from folder name if empty
            if not self.input_pid.text():
                self.input_pid.setText(os.path.basename(folder))

    def _browse_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.input_out_folder.setText(folder)

    def _on_run_clicked(self):
        dcm_root = self.input_dcm_folder.text().strip()
        out_dir  = self.input_out_folder.text().strip()
        pid_str  = self.input_pid.text().strip() or "PATIENT"

        if not dcm_root or not os.path.isdir(dcm_root):
            self._append_log("ERROR: Please select a valid DICOM input folder.")
            return
        if not out_dir:
            self._append_log("ERROR: Please specify an output folder.")
            return

        self._ct_result  = ""
        self._pet_result = ""
        self.btn_load.setEnabled(False)
        self.lbl_result.setText("")
        self.log_area.clear()

        self.sig_run_conversion.emit(
            dcm_root,
            out_dir,
            pid_str,
            self.chk_suv.isChecked(),
            self.chk_resample.isChecked(),
        )

    def _on_load_clicked(self):
        doctor  = self.input_doctor.text().strip()
        patient = self.input_patient.text().strip()
        self.sig_load_into_session.emit(
            self._ct_result, self._pet_result, doctor, patient
        )

    # ------------------------------------------------------------------
    # Called by handler to update state
    # ------------------------------------------------------------------

    def show_progress(self):
        self.progress_bar.setVisible(True)
        self.btn_run.setEnabled(False)

    def hide_progress(self):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)

    def on_conversion_finished(self, ct_path: str, pet_path: str):
        self._ct_result  = ct_path
        self._pet_result = pet_path
        self.btn_load.setEnabled(True)

        parts = []
        if ct_path:
            parts.append(f"CT: {os.path.basename(ct_path)}")
        if pet_path:
            parts.append(f"PET/SUV: {os.path.basename(pet_path)}")
        self.lbl_result.setText("Ready: " + "  |  ".join(parts))

    def append_log(self, msg: str):
        self._append_log(msg)

    def _append_log(self, msg: str):
        self.log_area.appendPlainText(msg)
        # Auto-scroll to bottom
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())
