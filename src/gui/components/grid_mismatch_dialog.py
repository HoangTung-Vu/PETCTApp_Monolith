"""Ask how a CT/PET voxel-grid mismatch is resolved when a NIfTI image is loaded."""

from typing import Optional

from PyQt6.QtWidgets import QMessageBox


def ask_resample_mode(parent, mismatch) -> Optional[str]:
    """Show the two grids and return "ct", "pet", "none" or None (cancelled).

    ``mismatch`` is a ``SessionManager.GridMismatch``. "ct" resamples PET onto
    the CT grid, "pet" resamples CT onto the PET grid, "none" (offered only when
    both have the same size) loads without resampling.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("CT / PET Grid Mismatch")
    box.setText("The CT and PET images are not on the same voxel grid.")

    lines = [
        f"CT:   {mismatch.ct.describe()}",
        f"PET:  {mismatch.pet.describe()}",
        "",
        "Resample one image onto the other's grid:",
        "  • PET → CT grid keeps the CT resolution (recommended, same as the DICOM import default).",
        "  • CT → PET grid is lighter; the segmentation is then saved next to the PET file.",
    ]
    closing = [m for m in ("ct", "pet", "none") if mismatch.closes_mask.get(m)]
    if closing:
        names = {"ct": "PET → CT grid", "pet": "CT → PET grid", "none": "loading without resampling"}
        lines += ["", "The current segmentation is on the other grid. "
                  f"{' / '.join(names[m] for m in closing)} closes it (the file on disk is kept)."]
    box.setInformativeText("\n".join(lines))
    # Wide enough for one grid per line.
    box.setStyleSheet("QLabel#qt_msgbox_informativelabel { min-width: 600px; }")

    choices = {
        box.addButton("Resample PET → CT grid", QMessageBox.ButtonRole.AcceptRole): "ct",
        box.addButton("Resample CT → PET grid", QMessageBox.ButtonRole.AcceptRole): "pet",
    }
    if mismatch.same_shape:
        choices[box.addButton("Load without resampling", QMessageBox.ButtonRole.AcceptRole)] = "none"
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(next(iter(choices)))

    box.exec()
    return choices.get(box.clickedButton())
