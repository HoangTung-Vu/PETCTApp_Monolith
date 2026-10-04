"""Worker that scans a folder for CT/PET DICOM series (headers only)."""

from PyQt6.QtCore import QThread, pyqtSignal


class DicomScanWorker(QThread):
    """Runs ``dicom_loader.scan_folder`` off the UI thread.

    Emits ``finished(list[SeriesInfo])`` or ``error(str)``.
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            from ...core.dicom_loader import scan_folder
            series = scan_folder(self.folder, log=lambda msg: print(f"[DICOM] {msg}"))
            self.finished.emit(series)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
