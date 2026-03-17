from PyQt6.QtCore import QThread, pyqtSignal

class ReportWorker(QThread):
    """Computes the clinical PET report in a background thread."""

    finished = pyqtSignal(dict)   # metrics dict
    error = pyqtSignal(str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    def run(self):
        try:
            metrics = self.session_manager.generate_report()
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))
