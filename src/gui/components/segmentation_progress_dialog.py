"""Non-modal progress dialog for tumor segmentation (upload → inference → done)."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar


class SegmentationProgressDialog(QDialog):
    """Floating, non-modal status window shown while a segmentation job runs.

    Non-modal so the user can keep panning/zooming/scrolling the viewers while
    inference runs on the backend. There is no cancel: the user cannot close it
    (no close button, Esc ignored); it is dismissed programmatically via
    ``complete()`` when the job finishes or errors.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tumor Segmentation")
        self.setModal(False)
        # Title bar only (no close/help/min buttons), kept above the main window.
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(360)
        self._allow_close = False

        self._label = QLabel("Preparing data…")
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate until the first real percentage
        self._bar.setTextVisible(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)

    # ── State updates (driven by SegmentationWorker signals) ──

    def set_uploaded(self):
        """Upload finished; waiting for the model to start inferring."""
        self._label.setText("Upload complete. Preparing model…")

    def set_progress(self, percent: int, stage: str):
        """Update the dialog from an upload or server progress update."""
        if stage == "preparing":
            self._bar.setRange(0, 0)  # indeterminate — serializing CT/PET to bytes
            self._label.setText("Preparing CT/PET data…")
        elif stage == "upload":
            self._bar.setRange(0, 100)
            self._bar.setValue(percent)
            self._label.setText(f"Uploading to AI engine… {percent}%")
        elif stage == "inference":
            self._bar.setRange(0, 100)
            self._bar.setValue(percent)
            self._label.setText(f"Running inference… {percent}%")
        elif stage in ("postprocessing", "done"):
            # Inference finished; hold the bar at 100% (no backward flicker) while
            # the server builds/serializes the mask.
            self._bar.setRange(0, 100)
            self._bar.setValue(100)
            self._label.setText("Finalizing…")
        else:  # queued / preprocessing
            self._bar.setRange(0, 0)
            self._label.setText("Upload complete. Preparing model…")

    def complete(self):
        """Programmatically close the dialog once the job is done/errored."""
        self._allow_close = True
        self.close()

    # ── Block user-initiated close (no cancel) ──

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
        else:
            event.ignore()

    def reject(self):
        # Esc triggers reject(); only honor it once we initiated the close.
        if self._allow_close:
            super().reject()
