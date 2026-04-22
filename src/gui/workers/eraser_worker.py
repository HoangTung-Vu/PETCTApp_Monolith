from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np


class EraserFloodWorker(QThread):
    """
    Worker to compute connected components (flood fill) asynchronously.
    Finds the 3D component containing the given click coordinate and signals
    the indices to be removed.
    """
    
    # Emits exactly the boolean mask of the component to erase:
    # (component_mask_zyx)
    component_found = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, mask_zyx: np.ndarray, click_zyx: tuple):
        super().__init__()
        # Take a reference (since it's read-only for the flood fill mostly, 
        # but to be safe, we don't modify it here)
        self.mask_zyx = mask_zyx
        self.click_zyx = click_zyx

    def run(self):
        try:
            z, y, x = self.click_zyx
            
            # Check bounds just in case
            if not (0 <= z < self.mask_zyx.shape[0] and
                    0 <= y < self.mask_zyx.shape[1] and
                    0 <= x < self.mask_zyx.shape[2]):
                self.error.emit("Click out of bounds.")
                self.finished.emit()
                return

            if self.mask_zyx[z, y, x] == 0:
                self.error.emit("Clicked on background.")
                self.finished.emit()
                return

            try:
                from skimage.morphology import flood
                component_mask = flood(self.mask_zyx, (z, y, x))
            except ImportError:
                from scipy.ndimage import label as nd_label
                labeled, num_features = nd_label(self.mask_zyx)
                component_id = labeled[z, y, x]
                component_mask = (labeled == component_id)

            self.component_found.emit(component_mask)
        except Exception as e:
            self.error.emit(f"Eraser Flood Failed: {str(e)}")
        finally:
            self.finished.emit()
