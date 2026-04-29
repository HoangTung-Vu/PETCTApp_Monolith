"""Worker that merges ROI into tumor mask and prepares ZYX data off-thread."""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np


class MergeSaveWorker(QThread):
    """
    Performs the heavy numpy work off the main thread:
    1. merge ROI into tumor (np.maximum)
    2. to_napari for both merged tumor and cleared ROI
    3. nib.save to disk

    Emits merge_ready with pre-computed ZYX arrays so main thread only does
    the lightweight load_mask_zyx (in-place copyto + refresh).
    """
    # (merged_tumor_xyz, merged_tumor_zyx, cleared_roi_xyz, cleared_roi_zyx)
    merge_ready = pyqtSignal(object, object, object, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager, roi_data_xyz=None):
        super().__init__()
        self.session_manager = session_manager
        # Snapshot ROI data before thread starts (caller already synced)
        self.roi_data_xyz = roi_data_xyz

    def run(self):
        try:
            from ...utils.nifti_utils import to_napari

            sm = self.session_manager

            # 1. Merge ROI into tumor
            if self.roi_data_xyz is not None and sm.tumor_mask is not None:
                tumor_data = sm.get_tumor_mask_data()
                merged = np.maximum(tumor_data, self.roi_data_xyz).astype(np.uint8)
                sm.set_tumor_mask(merged)

                # Clear ROI
                sm.clear_roi_mask()
                roi_cleared = sm.get_roi_mask_data()  # zeros array

                # 2. Pre-compute ZYX for both masks
                merged_zyx = to_napari(merged)
                roi_zyx = to_napari(roi_cleared) if roi_cleared is not None else None
            else:
                merged = sm.get_tumor_mask_data()
                merged_zyx = to_napari(merged) if merged is not None else None
                roi_cleared = sm.get_roi_mask_data()
                roi_zyx = to_napari(roi_cleared) if roi_cleared is not None else None

            # 3. Emit pre-computed data for UI update (will be received on main thread)
            self.merge_ready.emit(merged, merged_zyx, roi_cleared, roi_zyx)

            # 4. Save to disk (already off-thread)
            sm.save_session()

            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
