from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from ...utils.nifti_utils import to_napari

class EnsureROIWorker(QThread):
    finished = pyqtSignal(object, object, object, object)  # roi_data, tumor_data, roi_zyx, tumor_zyx
    
    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager
        
    def run(self):
        # Create zeroed numpy arrays if missing
        self.session_manager.ensure_roi_mask()
        
        tumor_data = self.session_manager.get_tumor_mask_data()
        roi_data = self.session_manager.get_roi_mask_data()
        
        # Slow part: to_napari transformation
        roi_zyx = None
        if roi_data is not None:
            roi_zyx = to_napari(roi_data.astype(np.uint8))
            
        tumor_zyx = None
        if tumor_data is not None:
            tumor_zyx = to_napari(tumor_data.astype(np.uint8))
            
        self.finished.emit(roi_data, tumor_data, roi_zyx, tumor_zyx)
