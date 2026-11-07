from PyQt6.QtCore import QThread, pyqtSignal

class SaveWorker(QThread):
    """
    Saves the session (NIfTI files, DB records) in a background thread
    to prevent UI lag when finalizing segmentations/refinements.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager, seg_path=None):
        """``seg_path``: file to write (AI run); None = the session's current file."""
        super().__init__()
        self.session_manager = session_manager
        self.seg_path = seg_path

    def run(self):
        try:
            print(f"[SaveWorker] Starting async save for session {self.session_manager.current_session_id}...")
            self.session_manager.save_session(self.seg_path)
            # A saved (edited) mask gets fresh lesion statistics.
            self.session_manager.update_lesion_stats()
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class OrganSaveWorker(QThread):
    """Writes a TotalSegmentator organ mask next to the CT in a background thread."""
    finished = pyqtSignal(str)   # path written
    error = pyqtSignal(str)

    def __init__(self, session_manager, organ_img):
        super().__init__()
        self.session_manager = session_manager
        self.organ_img = organ_img

    def run(self):
        try:
            path = self.session_manager.save_organ_mask(self.organ_img)
            self.finished.emit(str(path))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
