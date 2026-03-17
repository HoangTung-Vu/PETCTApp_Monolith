import nibabel as nib
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class RefinementWorker(QThread):
    """
    Runs SUV refinement in a background thread (local, no HTTP needed).
    finished emits (refined_image, computed_threshold) — threshold is the user-supplied SUV value.
    """
    finished = pyqtSignal(object, float)
    error = pyqtSignal(str)

    def __init__(self, pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float, roi_mask: np.ndarray = None):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.threshold = threshold
        self.roi_mask = roi_mask

    def run(self):
        try:
            from ...core.engine.refinement_engine import RefinementEngine

            print(f"[Worker] Starting SUV Refinement (Threshold {self.threshold}, ROI {'Yes' if self.roi_mask is not None else 'No'})...")
            refined_image = RefinementEngine.refine_suv(
                self.pet_image, self.mask_image, self.threshold, self.roi_mask
            )

            self.finished.emit(refined_image, float(self.threshold))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class AdaptiveThresholdingWorker(QThread):
    """
    Runs Adaptive Thresholding refinement in a background thread (local, no HTTP needed).
    finished emits (refined_image, computed_threshold_suv).
    """
    finished = pyqtSignal(object, float)
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
            from ...core.engine.adaptive_thresholding_refinement_engine import (
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
            self.finished.emit(refined_image, float(getattr(engine, "last_threshold", 0.0)))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class IterativeThresholdingWorker(QThread):
    """
    Runs Iterative Thresholding refinement in a background thread (local, no HTTP needed).
    finished emits (refined_image, computed_threshold_abs_suv).
    """
    finished = pyqtSignal(object, float)
    error = pyqtSignal(str)

    def __init__(
        self,
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
        m: float = 7.8,
        c1: float = 61.7,
        c0: float = 31.6,
        convergence_tol: float = 0.03,
        max_iterations: int = 10,
    ):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.roi_mask = roi_mask
        self.m = m
        self.c1 = c1
        self.c0 = c0
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations

    def run(self):
        try:
            from ...core.engine.iterative_thresholding_refinement import IterativeThresholdingEngine

            print(
                f"[Worker] Starting Iterative Thresholding Refinement "
                f"(m={self.m}, c1={self.c1}, c0={self.c0}, tol={self.convergence_tol}, "
                f"iters={self.max_iterations}, ROI={'Yes' if self.roi_mask is not None else 'No'})..."
            )

            engine = IterativeThresholdingEngine(
                m=self.m,
                c1=self.c1,
                c0=self.c0,
                convergence_tol=self.convergence_tol,
                max_iterations=self.max_iterations,
            )

            refined_image = engine.refine(self.pet_image, self.mask_image, self.roi_mask)
            self.finished.emit(refined_image, float(getattr(engine, "last_threshold_abs", 0.0)))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
