"""Dialog to choose the CT and PET series of a scanned DICOM folder."""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLabel, QDialogButtonBox, QGroupBox,
)

from ...core.dicom_loader import auto_pick, names_from_header, format_dicom_date


class DicomSeriesDialog(QDialog):
    """Lists the CT and PET series found in a folder, with the auto-picks preselected.

    Patient and physician names are read from the DICOM tags of the selected
    series; ``manual_doctor`` / ``manual_patient`` are only fallbacks for empty tags.
    """

    def __init__(self, series: list, folder: str, manual_doctor: str = "",
                 manual_patient: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import DICOM")
        self.setMinimumWidth(620)
        self._folder = folder
        self._manual = (manual_doctor, manual_patient)
        self._cts = [s for s in series if s.modality == "CT"]
        self._pets = [s for s in series if s.modality == "PT"]

        layout = QVBoxLayout(self)

        grp_patient = QGroupBox("Patient (from DICOM)")
        form_patient = QFormLayout(grp_patient)
        self.lbl_patient = QLabel()
        self.lbl_physician = QLabel()
        self.lbl_date = QLabel()
        form_patient.addRow("Patient:", self.lbl_patient)
        form_patient.addRow("Physician:", self.lbl_physician)
        form_patient.addRow("Study date:", self.lbl_date)
        layout.addWidget(grp_patient)

        grp_series = QGroupBox(f"Series in {Path(folder).name}")
        form_series = QFormLayout(grp_series)
        self.combo_ct = self._make_combo(self._cts)
        self.combo_pet = self._make_combo(self._pets)
        self.combo_resample = QComboBox()
        self.combo_resample.addItem("Resample PET to the CT grid", "ct")
        self.combo_resample.addItem("Resample CT to the PET grid", "pet")
        form_series.addRow("CT:", self.combo_ct)
        form_series.addRow("PET:", self.combo_pet)
        form_series.addRow("Grid:", self.combo_resample)
        layout.addWidget(grp_series)

        self.lbl_warnings = QLabel()
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("color: #e0a800;")
        layout.addWidget(self.lbl_warnings)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        ct, pet = auto_pick(series)
        if ct is not None:
            self.combo_ct.setCurrentIndex(self._cts.index(ct) + 1)
        if pet is not None:
            self.combo_pet.setCurrentIndex(self._pets.index(pet) + 1)
        self.combo_ct.currentIndexChanged.connect(self._update)
        self.combo_pet.currentIndexChanged.connect(self._update)
        self._update()

    @staticmethod
    def _make_combo(items: list) -> QComboBox:
        combo = QComboBox()
        combo.addItem("None", None)
        for s in items:
            combo.addItem(s.label(), s.uid)
            combo.setItemData(combo.count() - 1, s.directory, Qt.ItemDataRole.ToolTipRole)
        return combo

    @property
    def selected_ct(self):
        i = self.combo_ct.currentIndex()
        return self._cts[i - 1] if i > 0 else None

    @property
    def selected_pet(self):
        i = self.combo_pet.currentIndex()
        return self._pets[i - 1] if i > 0 else None

    @property
    def resample_mode(self) -> str:
        return self.combo_resample.currentData()

    def names(self) -> tuple:
        """``(doctor, patient)`` for the selected series."""
        ref = self.selected_ct or self.selected_pet
        if ref is None:
            return self._manual
        return names_from_header(ref.header, *self._manual, fallback=Path(self._folder).name)

    def _update(self):
        ct, pet = self.selected_ct, self.selected_pet
        ref = ct or pet
        doctor, patient = self.names()
        self.lbl_patient.setText(patient or "—")
        self.lbl_physician.setText(doctor or "—")
        self.lbl_date.setText(format_dicom_date(ref.date) if ref else "—")
        self.combo_resample.setEnabled(ct is not None and pet is not None)

        warnings = list((ct.warnings if ct else []) + (pet.warnings if pet else []))
        if ct is not None and pet is not None and ct.frame_of_reference_uid != pet.frame_of_reference_uid:
            warnings.append("CT and PET have different FrameOfReferenceUID — check the alignment.")
        if pet is not None and not pet.is_attenuation_corrected:
            warnings.append("The selected PET series is not marked attenuation corrected (AC).")
        self.lbl_warnings.setText("\n".join(f"⚠ {w}" for w in warnings))
        self.lbl_warnings.setVisible(bool(warnings))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ref is not None)
