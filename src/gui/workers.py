from PyQt6.QtCore import QThread, pyqtSignal
import nibabel as nib
import numpy as np

# We'll need to import engines. 
# To avoid heavy imports at startup, imports inside run() might be better, 
# or import at module level if we accept startup time.
# Given "Engine" classes are lightweight wrappers (mostly), module level is fine.
# But torch/totalsegmentator imports are heavy.
from ..core.engine.totalseg_engine import TotalSegEngine
from ..core.engine.nnunet_engine import NNUNetEngine

class SegmentationWorker(QThread):
    """
    Runs segmentation in a background thread.
    """
    finished = pyqtSignal(object) # Emits (mask_nib_image, type)
    error = pyqtSignal(str)
    
    def __init__(self, engine_type: str, images):
        super().__init__()
        self.engine_type = engine_type
        self.images = images
        
    def run(self):
        try:
            print(f"[Worker] Starting {self.engine_type} segmentation...")
            if self.engine_type == "tumor":
                # Use nnUNet for tumor
                # Expects list of images [CT, PET]
                engine = NNUNetEngine(dataset_id=42, device="auto") # Configurable?
                result = engine.run_nib(self.images)
                self.finished.emit((result, "tumor"))
                
            elif self.engine_type == "organ":
                # Use TotalSegmentator for organs
                # Expects single image (CT)
                engine = TotalSegEngine(task="total", fast=True, device="auto")
                result = engine.run_nib(self.images)
                self.finished.emit((result, "organ"))
                
            else:
                self.error.emit(f"Unknown engine type: {self.engine_type}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
