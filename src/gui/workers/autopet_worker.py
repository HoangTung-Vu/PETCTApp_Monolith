import json
import nibabel as nib
import httpx
from PyQt6.QtCore import QThread, pyqtSignal

from ...utils.nifti_utils import bytes_to_npz, make_nifti_upload
from .core import ENGINE_AUTOPET_URL, HTTP_TIMEOUT, _inference_lock

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
