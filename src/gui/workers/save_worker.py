from PyQt6.QtCore import QThread, pyqtSignal

class SaveWorker(QThread):
    """
    Saves the session (NIfTI files, DB records) in a background thread
    to prevent UI lag when finalizing segmentations/refinements.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    def run(self):
        try:
            print(f"[SaveWorker] Starting async save for session {self.session_manager.current_session_id}...")
            self.session_manager.save_session()
            # A saved (edited) mask gets fresh lesion statistics.
            self.session_manager.update_lesion_stats()
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
