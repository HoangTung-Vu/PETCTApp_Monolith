import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from PyQt6.QtWidgets import QApplication
import numpy as np
import nibabel as nib
from src.core.session_manager import SessionManager
from src.gui.workers.roi_worker import EnsureROIWorker

app = QApplication([])

session = SessionManager()
# Mock CT image
session.ct_image = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4))

worker = EnsureROIWorker(session)
def on_finish(roi_data, tumor_data, roi_zyx, tumor_zyx):
    print("Worker finished successfully.")
    print("ROI ZYX shape:", roi_zyx.shape if roi_zyx is not None else "None")
    print("Tumor ZYX shape:", tumor_zyx.shape if tumor_zyx is not None else "None")
    app.quit()

worker.finished.connect(on_finish)
worker.start()
app.exec()
