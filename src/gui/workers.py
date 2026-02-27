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
                # Use nnUNet for tumor — get probability volume
                engine = NNUNetEngine(dataset_id=42, device="auto")
                prob = engine.run_nib_prob(self.images, single_channel=True)  # shape (X,Y,Z)
                # Threshold to create binary mask
                mask_array = (prob >= 0.5).astype(np.uint8)
                # Wrap as NIfTI using first image's affine
                ref_img = self.images[0] if isinstance(self.images, list) else self.images
                mask_nib = nib.Nifti1Image(mask_array, ref_img.affine, ref_img.header)
                self.finished.emit((mask_nib, prob, "tumor"))
                
            elif self.engine_type == "organ":
                # Use TotalSegmentator for organs
                # Expects single image (CT)
                engine = TotalSegEngine(task="total", fast=True, device="auto")
                result = engine.run_nib(self.images)
                self.finished.emit((result, None, "organ"))
                
            else:
                self.error.emit(f"Unknown engine type: {self.engine_type}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class RefinementWorker(QThread):
    """
    Runs SUV refinement in a background thread.
    """
    finished = pyqtSignal(object) # Emits nib.Nifti1Image
    error = pyqtSignal(str)
    
    def __init__(self, pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.threshold = threshold
        
    def run(self):
        try:
            # Import here to avoid circular dependencies if any, though engine is core
            from ..core.refinement_engine import RefinementEngine
            
            print(f"[Worker] Starting SUV Refinement (Threshold {self.threshold})...")
            refined_image = RefinementEngine.refine_suv(self.pet_image, self.mask_image, self.threshold)
            
            self.finished.emit(refined_image)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class AutoPETWorker(QThread):
    """
    Runs AutoPET Interactive inference in a background thread.
    Emits the probability array (single-channel, shape X,Y,Z).
    """
    finished = pyqtSignal(object)  # Emits np.ndarray (prob)
    error = pyqtSignal(str)
    
    def __init__(self, ct_image: nib.Nifti1Image, pet_image: nib.Nifti1Image, clicks: list):
        super().__init__()
        self.ct_image = ct_image
        self.pet_image = pet_image
        self.clicks = clicks  # [{"point": [z,y,x], "name": "tumor"/"background"}, ...]
        
    def run(self):
        try:
            from ..core.engine.autopet_interactive_engine import AutoPETInteractiveEngine
            
            print(f"[Worker] Starting AutoPET Interactive ({len(self.clicks)} clicks)...")
            engine = AutoPETInteractiveEngine(device="auto")
            prob = engine.run_nib_prob(
                [self.ct_image, self.pet_image],
                clicks=self.clicks,
                single_channel=True
            )
            
            self.finished.emit(prob)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
