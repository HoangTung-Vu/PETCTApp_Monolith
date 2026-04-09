import nibabel as nib
import numpy as np
import httpx
from PyQt6.QtCore import QThread, pyqtSignal

from ...utils.nifti_utils import bytes_to_nifti, bytes_to_npz, make_nifti_upload
from .core import (
    ENGINE_NNUNET_URL,
    ENGINE_NNUNET_URL_PRETRAINED,
    ENGINE_TOTALSEG_URL,
    HTTP_TIMEOUT,
    _inference_lock
)

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
                    mask_nib = nib.Nifti1Image(mask_array, ref_img.affine)
                    self.finished.emit((mask_nib, prob, "tumor"))

                elif self.engine_type == "tumor_pretrained":
                    images = self.images if isinstance(self.images, list) else [self.images]
                    files = [make_nifti_upload(img, f"image_{i}.nii.gz") for i, img in enumerate(images)]

                    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                        response = client.post(
                            f"{ENGINE_NNUNET_URL_PRETRAINED}/run_nib",
                            files=files,
                        )
                        response.raise_for_status()

                    mask_nib = bytes_to_nifti(response.content)
                    self.finished.emit((mask_nib, None, "tumor_pretrained"))

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
