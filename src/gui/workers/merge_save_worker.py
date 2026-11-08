"""Worker that merges ROI into tumor mask and prepares ZYX data off-thread."""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np


class MergeSaveWorker(QThread):
    """
    Performs the heavy numpy work off the main thread:
    1. merge ROI into tumor as lesion ``label_id`` (``lesion_labels.merge_roi``)
    2. to_napari for both merged tumor and cleared ROI
    3. nib.save to disk

    Emits merge_ready with pre-computed ZYX arrays so main thread only does
    the lightweight load_mask_zyx (in-place copyto + refresh).
    """
    # (merged_tumor_xyz, merged_tumor_zyx, cleared_roi_xyz, cleared_roi_zyx)
    merge_ready = pyqtSignal(object, object, object, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager, roi_data_xyz=None, label_id: int = 1, merge_ids=()):
        """``label_id``: the lesion the ROI becomes; ``merge_ids``: lesions under
        the ROI the user chose to treat as that same lesion."""
        super().__init__()
        self.session_manager = session_manager
        # Snapshot ROI data before thread starts (caller already synced)
        self.roi_data_xyz = roi_data_xyz
        self.label_id = int(label_id)
        self.merge_ids = tuple(merge_ids)

    def run(self):
        try:
            from ...core.lesion_labels import merge_roi
            from ...utils.nifti_utils import to_napari

            sm = self.session_manager

            # 1. Merge ROI into tumor: background voxels become the selected
            #    lesion; other lesions keep their ids unless merged.
            if self.roi_data_xyz is not None and sm.tumor_mask is not None:
                # In place: the session array was just synced from the viewer.
                merged = merge_roi(sm.get_tumor_mask_data(), self.roi_data_xyz,
                                   self.label_id, self.merge_ids)
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

            # 5. Recompute lesion statistics for the saved mask
            sm.update_lesion_stats()
            sm.refresh_label_ids()

            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class GiveIdsWorker(QThread):
    """"Give lesion ids": relabel the tumor mask from scratch, save it and
    recompute the lesion values (``SessionManager.give_lesion_ids``).

    Emits ``mask_ready(xyz, zyx)`` once the new mask exists, so the viewers can
    show it while the file is written.
    """
    mask_ready = pyqtSignal(object, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    def run(self):
        try:
            from ...core import lesion_labels
            from ...utils.nifti_utils import to_napari

            sm = self.session_manager
            data = sm.get_tumor_mask_data()
            if data is None:
                raise ValueError("No tumor mask to label.")
            labels, n = lesion_labels.give_lesion_ids(data)
            sm.set_tumor_mask(labels, labeled=True)
            self.mask_ready.emit(labels, to_napari(labels))
            sm.save_session()
            sm.update_lesion_stats()
            sm.refresh_label_ids()
            print(f"[GiveIdsWorker] {n} lesion(s) labelled and saved.")
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
