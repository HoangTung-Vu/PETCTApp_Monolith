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

ENGINE_NNUNET_URL_PRETRAINED = "http://localhost:8101"
ENGINE_NNUNET_URL = "http://localhost:8104/nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"
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


class RefinementWorker(QThread):
    """
    Runs SUV refinement in a background thread (local, no HTTP needed).
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float, roi_mask: np.ndarray = None):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.threshold = threshold
        self.roi_mask = roi_mask

    def run(self):
        try:
            from ..core.engine.refinement_engine import RefinementEngine

            print(f"[Worker] Starting SUV Refinement (Threshold {self.threshold}, ROI {'Yes' if self.roi_mask is not None else 'No'})...")
            refined_image = RefinementEngine.refine_suv(
                self.pet_image, self.mask_image, self.threshold, self.roi_mask
            )

            self.finished.emit(refined_image)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class AdaptiveThresholdingWorker(QThread):
    """
    Runs Adaptive Thresholding refinement in a background thread (local, no HTTP needed).
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
        isocontour_fraction: float = 0.70,
        background_mode: str = "outside_isocontour",
        border_thickness: int = 3,
    ):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.roi_mask = roi_mask
        self.isocontour_fraction = isocontour_fraction
        self.background_mode = background_mode
        self.border_thickness = border_thickness

    def run(self):
        try:
            from ..core.engine.adaptive_thresholding_refinement_engine import (
                AdaptiveThresholdingRefinementEngine,
                BackgroundMode,
            )

            bg_mode = BackgroundMode(self.background_mode)

            print(
                f"[Worker] Starting Adaptive Thresholding Refinement "
                f"(iso={self.isocontour_fraction}, bg={self.background_mode}, "
                f"border={self.border_thickness}, ROI={'Yes' if self.roi_mask is not None else 'No'})..."
            )

            engine = AdaptiveThresholdingRefinementEngine(
                isocontour_fraction=self.isocontour_fraction,
                background_mode=bg_mode,
                border_thickness=self.border_thickness,
            )

            refined_image = engine.refine(self.pet_image, self.mask_image, self.roi_mask)
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

class DataLoaderWorker(QThread):
    """
    Loads NIfTI files and updates the SessionManager in a background thread 
    to prevent UI freezing.
    """
    finished = pyqtSignal(bool)  # Emits True when done
    error = pyqtSignal(str)

    def __init__(self, session_manager, current_session_id=None, ct_path=None, pet_path=None, action="update", new_doctor=None, new_patient=None):
        """
        action: "update" (existing session), "create" (new session), "load" (load existing by ID)
        """
        super().__init__()
        self.session_manager = session_manager
        self.current_session_id = current_session_id
        self.ct_path = ct_path
        self.pet_path = pet_path
        self.action = action
        self.new_doctor = new_doctor
        self.new_patient = new_patient

    def run(self):
        try:
            print(f"[DataLoaderWorker] Starting async data loading (Action: {self.action})...")
            
            if self.action == "create":
                self.session_manager.create_session(
                    self.new_doctor, 
                    self.new_patient, 
                    ct_path=self.ct_path, 
                    pet_path=self.pet_path
                )
            elif self.action == "update":
                # For update, we might need a temporary session if none exists
                if self.session_manager.current_session_id is None:
                    self.session_manager.create_session(
                        "System", 
                        "Anonymous", 
                        ct_path=self.ct_path, 
                        pet_path=self.pet_path
                    )
                else:
                    self.session_manager.update_current_session(
                        ct_path=self.ct_path, 
                        pet_path=self.pet_path
                    )
            elif self.action == "load":
                if self.current_session_id is not None:
                     self.session_manager.load_session(self.current_session_id)
                else:
                     raise ValueError("Session ID required for loading.")
                     
            # Pre-compute get_fdata() here in the background thread so it's cached in memory
            # This is the operation that usually blocks the main thread
            if self.session_manager.ct_image is not None:
                _ = self.session_manager.ct_image.get_fdata()
            if self.session_manager.pet_image is not None:
                _ = self.session_manager.pet_image.get_fdata()

            self.finished.emit(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ReportWorker(QThread):
    """Computes the clinical PET report in a background thread."""

    finished = pyqtSignal(dict)   # metrics dict
    error = pyqtSignal(str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager

    def run(self):
        try:
            metrics = self.session_manager.generate_report()
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))


class SnapshotWorker(QThread):
    """
    Moves the heavy mask.get_fdata().copy() to a background thread
    to prevent UI lag when entering Refine/AutoPET tabs.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, session_manager, mask_type: str, mask_data_snapshot: np.ndarray = None):
        super().__init__()
        self.session_manager = session_manager
        self.mask_type = mask_type
        self.mask_data_snapshot = mask_data_snapshot

    def run(self):
        try:
            print(f"[SnapshotWorker] Taking async snapshot for {self.mask_type}...")
            # We fetch the mask data here inside the QThread so .get_fdata().copy()
            # doesn't freeze the main GUI thread.
            snapshot_data = None
            if self.mask_type == "tumor":
                if self.session_manager.tumor_mask:
                    # Explicit fast copy
                    snapshot_data = np.array(self.session_manager.tumor_mask.dataobj).copy()
                    
            self.session_manager.snapshot_current_mask(self.mask_type, snapshot_data)
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

