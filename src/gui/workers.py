"""Background workers for segmentation tasks.

Each worker sends HTTP requests to Docker-based engine backends
instead of importing engine libraries directly.

Inference Lock: Only 1 inference at a time to prevent race conditions.
"""

import json
import threading

from PyQt6.QtCore import QThread, pyqtSignal
import nibabel as nib
import numpy as np
import httpx

from ..utils.nifti_utils import nifti_to_bytes, bytes_to_nifti, bytes_to_npz, make_nifti_upload

# ──── Configuration ────

ENGINE_NNUNET_URL = "http://localhost:8101"
ENGINE_AUTOPET_URL = "http://localhost:8102"
ENGINE_TOTALSEG_URL = "http://localhost:8103"

# Global inference lock — only 1 engine can infer at a time
_inference_lock = threading.Lock()

# Timeout for HTTP requests (seconds) — inference can be slow
HTTP_TIMEOUT = 600.0


# ──── Workers ────

class SegmentationWorker(QThread):
    """
    Runs segmentation in a background thread via HTTP to engine backends.
    """
    finished = pyqtSignal(object)  # Emits (mask_nib_image, prob_array_or_None, type)
    error = pyqtSignal(str)

    def __init__(self, engine_type: str, images):
        super().__init__()
        self.engine_type = engine_type
        self.images = images

    def run(self):
        try:
            with _inference_lock:
                print(f"[Worker] Starting {self.engine_type} segmentation (HTTP)...")

                if self.engine_type == "tumor":
                    images = self.images if isinstance(self.images, list) else [self.images]
                    files = [make_nifti_upload(img, f"image_{i}.nii.gz") for i, img in enumerate(images)]

                    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                        response = client.post(
                            f"{ENGINE_NNUNET_URL}/run_nib_prob",
                            files=files,
                            data={"single_channel": "true"},
                        )
                        response.raise_for_status()

                    npz = bytes_to_npz(response.content)
                    prob = npz["prob"]
                    affine = npz["affine"]

                    mask_array = (prob >= 0.5).astype(np.uint8)
                    ref_img = images[0]
                    mask_nib = nib.Nifti1Image(mask_array, ref_img.affine, ref_img.header)
                    self.finished.emit((mask_nib, prob, "tumor"))

                elif self.engine_type == "organ":
                    image = self.images if isinstance(self.images, nib.Nifti1Image) else self.images[0]
                    files = [make_nifti_upload(image, "ct.nii.gz")]

                    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                        response = client.post(
                            f"{ENGINE_TOTALSEG_URL}/run_nib",
                            files=files,
                        )
                        response.raise_for_status()

                    result = bytes_to_nifti(response.content)
                    self.finished.emit((result, None, "organ"))

                else:
                    self.error.emit(f"Unknown engine type: {self.engine_type}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class RefinementWorker(QThread):
    """
    Runs SUV refinement in a background thread (local, no HTTP needed).
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.threshold = threshold

    def run(self):
        try:
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
    Runs AutoPET Interactive inference via HTTP to the autopet engine backend.
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
            with _inference_lock:
                print(f"[Worker] Starting AutoPET Interactive ({len(self.clicks)} clicks) (HTTP)...")

                files = [
                    make_nifti_upload(self.ct_image, "ct.nii.gz"),
                    make_nifti_upload(self.pet_image, "pet.nii.gz"),
                ]

                with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                    response = client.post(
                        f"{ENGINE_AUTOPET_URL}/run_nib_prob",
                        files=files,
                        data={
                            "clicks": json.dumps(self.clicks),
                            "single_channel": "true",
                        },
                    )
                    response.raise_for_status()

                npz = bytes_to_npz(response.content)
                prob = npz["prob"]

                self.finished.emit(prob)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
