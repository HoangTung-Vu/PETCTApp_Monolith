"""Workflow tab: Session management, actions, and report."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QComboBox, QProgressBar, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QLabel
)
from PyQt6.QtCore import pyqtSignal


class WorkflowTab(QWidget):
    """Session management + segmentation actions + report display."""

    # Signals
    sig_load_ct_clicked = pyqtSignal()
    sig_load_pet_clicked = pyqtSignal()
    sig_segment_clicked = pyqtSignal()
    sig_new_session_clicked = pyqtSignal(str, str)   # doctor, patient
    sig_load_session_clicked = pyqtSignal(int)        # session_id
    sig_report_clicked = pyqtSignal()
    sig_toggle_lesion_ids = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # ── Current Session Info ──
        self.lbl_current_session = QLabel("Current Session: None")
        self.lbl_current_session.setStyleSheet("font-weight: bold; color: #337ab7; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(self.lbl_current_session)

        # ── Session Management ──
        grp_session = QGroupBox("Session")
        session_layout = QFormLayout()

        self.input_doctor = QLineEdit()
        self.input_patient = QLineEdit()
        session_layout.addRow("Doctor:", self.input_doctor)
        session_layout.addRow("Patient:", self.input_patient)

        self.btn_new_session = QPushButton("New Session")
        self.btn_new_session.clicked.connect(self._emit_new_session)
        session_layout.addRow(self.btn_new_session)

        self.combo_sessions = QComboBox()
        self.btn_load_this_session = QPushButton("Load Selected")
        self.btn_load_this_session.clicked.connect(self._emit_load_session)

        session_layout.addRow("Previous:", self.combo_sessions)
        session_layout.addRow(self.btn_load_this_session)

        grp_session.setLayout(session_layout)
        layout.addWidget(grp_session)

        # ── Action Buttons ──
        grp_actions = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        self.btn_load_ct = QPushButton("Load CT")
        self.btn_load_ct.clicked.connect(self.sig_load_ct_clicked.emit)

        self.btn_load_pet = QPushButton("Load PET")
        self.btn_load_pet.clicked.connect(self.sig_load_pet_clicked.emit)

        self.btn_segment = QPushButton("Run Segmentation")
        self.btn_segment.clicked.connect(self.sig_segment_clicked.emit)

        action_layout.addWidget(self.btn_load_ct)
        action_layout.addWidget(self.btn_load_pet)
        action_layout.addWidget(self.btn_segment)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        grp_actions.setLayout(action_layout)
        layout.addWidget(grp_actions)

        # ── Report Section ──
        grp_report = QGroupBox("Report")
        report_layout = QVBoxLayout()

        self.btn_report = QPushButton("Generate Report")
        self.btn_report.clicked.connect(self.sig_report_clicked.emit)
        report_layout.addWidget(self.btn_report)

        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 0)
        self.report_progress.setVisible(False)
        report_layout.addWidget(self.report_progress)

        # Global gTLG
        gtlg_form = QFormLayout()
        self.lbl_gtlg = QLabel("—")
        gtlg_form.addRow("gTLG:", self.lbl_gtlg)
        report_layout.addLayout(gtlg_form)

        # Per-lesion table
        self.tbl_lesions = QTableWidget()
        self.tbl_lesions.setColumnCount(4)
        self.tbl_lesions.setHorizontalHeaderLabels(["ID", "SUVmax", "SUVmean", "MTV (mL)"])
        self.tbl_lesions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_lesions.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_lesions.setMaximumHeight(180)
        report_layout.addWidget(self.tbl_lesions)

        # Toggle lesion bounding boxes
        self.chk_show_lesion_ids = QCheckBox("Show Lesion IDs")
        self.chk_show_lesion_ids.setChecked(False)
        self.chk_show_lesion_ids.toggled.connect(self.sig_toggle_lesion_ids.emit)
        report_layout.addWidget(self.chk_show_lesion_ids)

        grp_report.setLayout(report_layout)
        layout.addWidget(grp_report)

        layout.addStretch()

    # ── Helpers ──

    def _emit_new_session(self):
        doc = self.input_doctor.text()
        pat = self.input_patient.text()
        self.sig_new_session_clicked.emit(doc, pat)

    def _emit_load_session(self):
        data = self.combo_sessions.currentData()
        if data is not None:
            self.sig_load_session_clicked.emit(int(data))

    def show_progress(self):
        self.progress_bar.setVisible(True)

    def hide_progress(self):
        self.progress_bar.setVisible(False)

    def show_report_progress(self):
        self.report_progress.setVisible(True)
        self.btn_report.setEnabled(False)

    def hide_report_progress(self):
        self.report_progress.setVisible(False)
        self.btn_report.setEnabled(True)

    def show_report_results(self, metrics: dict):
        """Populate the report labels and table with computed metrics."""
        self.lbl_gtlg.setText(str(metrics.get("gTLG", "—")))
        lesions = metrics.get("lesions", [])
        self.tbl_lesions.setRowCount(len(lesions))
        for row, lesion in enumerate(lesions):
            self.tbl_lesions.setItem(row, 0, QTableWidgetItem(str(lesion.get("id", ""))))
            self.tbl_lesions.setItem(row, 1, QTableWidgetItem(str(lesion.get("SUVmax", ""))))
            self.tbl_lesions.setItem(row, 2, QTableWidgetItem(str(lesion.get("SUVmean", ""))))
            self.tbl_lesions.setItem(row, 3, QTableWidgetItem(str(lesion.get("MTV", ""))))

    def clear_report_results(self):
        """Reset report labels and table to placeholder."""
        self.lbl_gtlg.setText("—")
        self.tbl_lesions.setRowCount(0)

    def set_current_session_label(self, text: str):
        """Update the session name at the top of the tab."""
        self.lbl_current_session.setText(f"Current Session: {text}")
