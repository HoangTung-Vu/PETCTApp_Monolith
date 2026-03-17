import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class SnapshotWorker(QThread):
    """
    Moves the heavy mask.get_fdata().copy() to a background thread
    to prevent UI lag when entering Refine/AutoPET tabs.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager, mask_type: str, mask_data_snapshot: np.ndarray = None):
        super().__init__()
        self.session_manager = session_manager
        self.mask_type = mask_type
        self.mask_data_snapshot = mask_data_snapshot

    def run(self):
        try:
            print(f"[SnapshotWorker] Taking async snapshot for {self.mask_type}...")
            # We fetch the mask data here inside the QThread so .get_fdata().copy()
            # doesn't freeze the main GUI thread.
            snapshot_data = None
            if self.mask_type == "tumor":
                if self.session_manager.tumor_mask:
                    # Explicit fast copy
                    snapshot_data = np.array(self.session_manager.tumor_mask.dataobj).copy()
                    
            self.session_manager.snapshot_current_mask(self.mask_type, snapshot_data)
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
