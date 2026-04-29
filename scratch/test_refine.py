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
window.session_manager.ct_image = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4))
window.layout_manager._active_views = ["axial_ct"]
window._push_mask_to_all = lambda *args, **kwargs: print("Push mask called")

window.control_panel.tabs.setCurrentIndex(2) # Refine Tab
app.processEvents()

print("Finished without crashing")
app.quit()
