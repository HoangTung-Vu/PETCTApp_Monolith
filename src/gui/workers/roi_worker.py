from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from ...utils.nifti_utils import to_napari

class EnsureROIWorker(QThread):
    # roi_data, tumor_data, roi_zyx, tumor_zyx, created_new_tumor
    finished = pyqtSignal(object, object, object, object, bool)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    def run(self):
        # Track whether tumor mask existed before ensure
        had_tumor = self.session_manager.tumor_mask is not None

        # Create zeroed in-memory masks if missing
        self.session_manager.ensure_roi_mask()

        created_new_tumor = not had_tumor and self.session_manager.tumor_mask is not None

        # Persist the new empty mask to disk next to the CT file so it survives
        # app restarts and the DB path is populated for future loads.
        if created_new_tumor:
            try:
                self.session_manager.save_session()
                print("[EnsureROIWorker] Empty segmentation saved next to CT.")
            except Exception as e:
                print(f"[EnsureROIWorker] Warning: could not save empty mask: {e}")

        tumor_data = self.session_manager.get_tumor_mask_data()
        roi_data = self.session_manager.get_roi_mask_data()

        print("[EnsureROIWorker] Starting to_napari for ROI...")
        roi_zyx = None
        if roi_data is not None:
            roi_zyx = to_napari(roi_data.astype(np.uint8, copy=False))

        # Only convert tumor when it was just created (zeros mask).
        # If it already existed, _cached_data_zyx["tumor"] in LayoutManager is still valid.
        tumor_zyx = None
        if created_new_tumor and tumor_data is not None:
            print("[EnsureROIWorker] Starting to_napari for new Tumor...")
            tumor_zyx = to_napari(tumor_data.astype(np.uint8, copy=False))

        print(f"[EnsureROIWorker] Emitting finished signal... (created_new_tumor={created_new_tumor})")
        self.finished.emit(roi_data, tumor_data, roi_zyx, tumor_zyx, created_new_tumor)

