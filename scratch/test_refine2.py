import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from PyQt6.QtWidgets import QApplication
import numpy as np
import nibabel as nib
from src.core.session_manager import SessionManager
from src.gui.main_window import MainWindow

app = QApplication([])
window = MainWindow()

# Simulate a full start and session load
window.session_manager.ct_image = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4))
window.layout_manager.set_active_views(["axial_ct", "axial_pet"])
window.control_panel.tabs.setCurrentIndex(2) # Refine Tab
app.processEvents() # Trigger the worker start

# Simulate changing active views while RefineTab is running or after
window.layout_manager.set_active_views(["axial_ct"]) # Hides axial_pet, clears its layers!

def check_after_worker():
    print("Worker should be finished.")
    # Check if we can change views again
    window.layout_manager.set_active_views(["axial_ct", "axial_pet"])
    print("Changed views successfully")
    app.quit()

from PyQt6.QtCore import QTimer
QTimer.singleShot(1000, check_after_worker)

app.exec()
