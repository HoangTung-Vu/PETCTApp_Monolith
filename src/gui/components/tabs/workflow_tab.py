"""Workflow tab: Session management, actions, and report."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QComboBox, QProgressBar, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QLabel, QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush

from ..viewers.label_colors import label_qcolor, text_color_for


class WorkflowTab(QWidget):
    """Session management + segmentation actions + report display."""

    # Signals
    sig_load_ct_clicked = pyqtSignal()
    sig_load_pet_clicked = pyqtSignal()
    sig_load_seg_clicked = pyqtSignal()
    sig_segment_clicked = pyqtSignal()
    sig_new_session_clicked = pyqtSignal(str, str)   # doctor, patient
    sig_load_session_clicked = pyqtSignal(int)        # session_id
    sig_delete_session_clicked = pyqtSignal(int)      # session_id
    sig_report_clicked = pyqtSignal()
    sig_toggle_lesion_ids = pyqtSignal(bool)
    sig_lesion_selected = pyqtSignal(int)             # lesion id (table row clicked)
    sig_load_from_dicom = pyqtSignal(str, str, str)   # dcm_folder, fallback doctor, fallback patient

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)

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

        self.btn_load_dicom = QPushButton("Load from DICOM Folder…")
        self.btn_load_dicom.setToolTip(
            "Import CT/PET series directly from DICOM. Patient and physician names "
            "come from the DICOM tags; the fields above are only used when the tags are empty."
        )
        self.btn_load_dicom.clicked.connect(self._emit_load_from_dicom)
        session_layout.addRow(self.btn_load_dicom)

        self.combo_sessions = QComboBox()
        # Long session names must not widen the page past the sidebar (the
        # scroll area has no horizontal scrollbar); they are elided instead.
        self.combo_sessions.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_sessions.setMinimumContentsLength(8)
        self.btn_load_this_session = QPushButton("Load Selected")
        self.btn_load_this_session.clicked.connect(self._emit_load_session)
        self.btn_delete_session = QPushButton("Delete Selected")
        self.btn_delete_session.setToolTip(
            "Remove the session from the app. Image files and the saved segmentation stay on disk."
        )
        self.btn_delete_session.clicked.connect(self._emit_delete_session)

        session_layout.addRow("Previous:", self.combo_sessions)
        session_layout.addRow(self.btn_load_this_session)
        session_layout.addRow(self.btn_delete_session)

        grp_session.setLayout(session_layout)
        layout.addWidget(grp_session)

        # ── Action Buttons ──
        grp_actions = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        self.btn_load_ct = QPushButton("Load CT")
        self.btn_load_ct.clicked.connect(self.sig_load_ct_clicked.emit)

        self.btn_load_pet = QPushButton("Load PET")
        self.btn_load_pet.clicked.connect(self.sig_load_pet_clicked.emit)

        self.btn_load_seg = QPushButton("Load Segmentation")
        self.btn_load_seg.clicked.connect(self.sig_load_seg_clicked.emit)

        self.btn_segment = QPushButton("Run Segmentation")
        self.btn_segment.clicked.connect(self.sig_segment_clicked.emit)

        action_layout.addWidget(self.btn_load_ct)
        action_layout.addWidget(self.btn_load_pet)
        action_layout.addWidget(self.btn_load_seg)
        action_layout.addWidget(self.btn_segment)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        grp_actions.setLayout(action_layout)
        layout.addWidget(grp_actions)

        # ── AI Engine ──
        grp_engine = QGroupBox("AI Engine")
        engine_layout = QFormLayout()

        from ...workers.core import (
            MODEL_PRESETS, get_engine_ip, get_engine_model, get_engine_port,
            set_engine_endpoint, set_engine_model,
        )
        self._set_engine_endpoint = set_engine_endpoint
        self._set_engine_model = set_engine_model
        self._engine_layout = engine_layout

        self.input_engine_host = QLineEdit(get_engine_ip())
        self.input_engine_host.setPlaceholderText("localhost or 202.122.49.242")
        self.input_engine_host.returnPressed.connect(self._apply_engine_host)
        engine_layout.addRow("IP:", self.input_engine_host)

        self.input_engine_port = QLineEdit(get_engine_port())
        self.input_engine_port.setPlaceholderText("8104")
        self.input_engine_port.returnPressed.connect(self._apply_engine_host)
        engine_layout.addRow("Port:", self.input_engine_port)

        # Model = last part of the engine URL. Picking a preset also fills in that
        # engine's default port; "Other…" takes any model name typed below.
        self.combo_engine_model = QComboBox()
        self.combo_engine_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_engine_model.setMinimumContentsLength(8)
        for name, label, port in MODEL_PRESETS:
            self.combo_engine_model.addItem(label, userData=(name, port))
            self.combo_engine_model.setItemData(self.combo_engine_model.count() - 1,
                                                f"{name} (port {port})", Qt.ItemDataRole.ToolTipRole)
        self.combo_engine_model.addItem("Other…", userData=None)
        engine_layout.addRow("Model:", self.combo_engine_model)

        self.input_engine_model = QLineEdit()
        self.input_engine_model.setPlaceholderText("Model name")
        self.input_engine_model.returnPressed.connect(self._apply_engine_host)
        engine_layout.addRow("Name:", self.input_engine_model)

        self._select_engine_model(get_engine_model())
        self.combo_engine_model.currentIndexChanged.connect(self._on_engine_model_changed)

        self.btn_apply_engine = QPushButton("Apply")
        self.btn_apply_engine.clicked.connect(self._apply_engine_host)
        engine_layout.addRow(self.btn_apply_engine)

        grp_engine.setLayout(engine_layout)
        layout.addWidget(grp_engine)

        # ── Data Folder (where petct.db lives; fixed, see config.default_storage_dir) ──
        grp_storage = QGroupBox("Data Folder")
        storage_layout = QFormLayout()

        from ....core.config import settings

        self.input_data_folder = QLineEdit(str(settings.STORAGE_DIR))
        self.input_data_folder.setReadOnly(True)
        self.input_data_folder.setToolTip(
            "The session database (petct.db) is stored here. Updating or "
            "reinstalling the app keeps it.")
        storage_layout.addRow("Folder:", self.input_data_folder)

        grp_storage.setLayout(storage_layout)
        layout.addWidget(grp_storage)

        # ── Report Section ──
        # Lesion values are computed when a segmentation is loaded or saved;
        # the export button only writes them (CSV + images) to a folder.
        grp_report = QGroupBox("Report")
        report_layout = QVBoxLayout()

        self.lbl_lesion_status = QLabel("No segmentation loaded")
        self.lbl_lesion_status.setWordWrap(True)
        report_layout.addWidget(self.lbl_lesion_status)

        # Total TLG
        ttlg_form = QFormLayout()
        self.lbl_ttlg = QLabel("—")
        ttlg_form.addRow("tTLG:", self.lbl_ttlg)
        report_layout.addLayout(ttlg_form)

        # Per-lesion table
        self.tbl_lesions = QTableWidget()
        self.tbl_lesions.setColumnCount(len(self._LESION_COLUMNS))
        self.tbl_lesions.setHorizontalHeaderLabels([title for title, _ in self._LESION_COLUMNS])
        self.tbl_lesions.verticalHeader().setVisible(False)
        self.tbl_lesions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_lesions.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_lesions.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_lesions.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.tbl_lesions.setMaximumHeight(220)
        self.tbl_lesions.cellClicked.connect(self._emit_lesion_selected)
        report_layout.addWidget(self.tbl_lesions)

        hint = QLabel("Click a lesion in the table, or double-click it in a 2D view, "
                      "to show its values there.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-style: italic;")
        report_layout.addWidget(hint)

        # Toggle lesion bounding boxes
        self.chk_show_lesion_ids = QCheckBox("Show Lesion IDs")
        self.chk_show_lesion_ids.setChecked(False)
        self.chk_show_lesion_ids.toggled.connect(self.sig_toggle_lesion_ids.emit)
        report_layout.addWidget(self.chk_show_lesion_ids)

        self.btn_report = QPushButton("Export Report…")
        self.btn_report.setToolTip("Write the lesion values (CSV) and per-lesion images to a folder.")
        self.btn_report.clicked.connect(self.sig_report_clicked.emit)
        report_layout.addWidget(self.btn_report)

        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 0)
        self.report_progress.setVisible(False)
        report_layout.addWidget(self.report_progress)

        grp_report.setLayout(report_layout)
        layout.addWidget(grp_report)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # ── Helpers ──

    def _emit_new_session(self):
        doc = self.input_doctor.text()
        pat = self.input_patient.text()
        self.sig_new_session_clicked.emit(doc, pat)

    def _emit_load_from_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
        doc = self.input_doctor.text().strip()
        pat = self.input_patient.text().strip()
        self.sig_load_from_dicom.emit(folder, doc, pat)

    def _apply_engine_host(self):
        # Persist + clean via the config setters, then reflect the cleaned
        # IP/port/model back into the fields.
        ip, port = self._set_engine_endpoint(
            self.input_engine_host.text(), self.input_engine_port.text()
        )
        self.input_engine_host.setText(ip)
        self.input_engine_port.setText(port)
        preset = self.combo_engine_model.currentData()
        model = self._set_engine_model(preset[0] if preset else self.input_engine_model.text())
        self._select_engine_model(model)

    def _select_engine_model(self, model: str):
        """Show ``model`` in the dropdown: its preset, else "Other…" with the name typed in."""
        combo = self.combo_engine_model
        index = next((i for i in range(combo.count())
                      if combo.itemData(i) and combo.itemData(i)[0] == model), combo.count() - 1)
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        other = combo.itemData(index) is None
        self.input_engine_model.setText(model if other else "")
        self._engine_layout.setRowVisible(self.input_engine_model, other)

    def _on_engine_model_changed(self, index: int):
        preset = self.combo_engine_model.itemData(index)
        if preset is None:
            # "Other…": applied once a name is typed (Enter / Apply).
            self._engine_layout.setRowVisible(self.input_engine_model, True)
            self.input_engine_model.setFocus()
            return
        self.input_engine_port.setText(preset[1])     # that engine's default port
        self._apply_engine_host()


    def _emit_load_session(self):
        data = self.combo_sessions.currentData()
        if data is not None:
            self.sig_load_session_clicked.emit(int(data))

    def _emit_delete_session(self):
        data = self.combo_sessions.currentData()
        if data is not None:
            self.sig_delete_session_clicked.emit(int(data))

    def _emit_lesion_selected(self, row: int, _col: int):
        item = self.tbl_lesions.item(row, 0)     # the ID column
        if item is not None:
            self.sig_lesion_selected.emit(int(item.text()))

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

    # (header, lesion key) — values shown with 2 decimals, the ID as is
    _LESION_COLUMNS = [
        ("ID", "id"),
        ("MTV (mL)", "MTV"),
        ("SUVmax", "SUVmax"),
        ("SUVmean", "SUVmean"),
        ("SUVpeak", "SUVpeak"),
        ("TLG", "TLG"),
    ]

    def show_lesion_stats(self, stats, stale: bool = False, message: str = ""):
        """Fill the tTLG label and lesion table from ``SessionManager.lesion_stats``.

        ``stats`` None clears them and shows ``message`` as the status. ``stale``
        flags values computed before the latest (unsaved) mask edit.
        """
        if stats is None:
            self.clear_report_results(message)
            return
        lesions = stats["lesions"]
        self.lbl_ttlg.setText(f"{stats['tTLG']:.2f}")
        # A recompute (after a save) may renumber the lesions: drop the selection.
        # The current cell must go too, or removing its row selects a neighbour.
        self.tbl_lesions.setCurrentCell(-1, -1)
        self.tbl_lesions.clearSelection()
        self.tbl_lesions.setRowCount(len(lesions))
        for row, lesion in enumerate(lesions):
            for col, (_, key) in enumerate(self._LESION_COLUMNS):
                value = lesion[key]
                text = str(value) if key == "id" else f"{value:.2f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if key == "id" and stats.get("labeled"):
                    # The id is the mask's value: show it in the lesion's colour.
                    item.setBackground(QBrush(label_qcolor(value)))
                    item.setForeground(QBrush(Qt.GlobalColor.black if text_color_for(value) == "black"
                                              else Qt.GlobalColor.white))
                self.tbl_lesions.setItem(row, col, item)

        if stale:
            self.lbl_lesion_status.setText("Segmentation edited — values update when it is saved.")
            self.lbl_lesion_status.setStyleSheet("color: #f0ad4e;")
        else:
            n = len(lesions)
            self.lbl_lesion_status.setText(
                "No lesions in the segmentation." if n == 0 else f"{n} lesion{'s' if n != 1 else ''}"
            )
            self.lbl_lesion_status.setStyleSheet("")

    def clear_report_results(self, message: str = "No segmentation loaded"):
        """Reset report labels and table to placeholder."""
        self.lbl_ttlg.setText("—")
        self.tbl_lesions.setRowCount(0)
        self.lbl_lesion_status.setText(message)
        self.lbl_lesion_status.setStyleSheet("")

    def set_current_session_label(self, text: str):
        """Update the session name at the top of the tab."""
        self.lbl_current_session.setText(f"Current Session: {text}")
