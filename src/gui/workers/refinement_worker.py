import nibabel as nib
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class ThresholdComputeWorker(QThread):
    """Compute threshold only (no mask modification) using adaptive or iterative method.

    Uses connected-component labelling to compute per-region thresholds.
    Emits threshold_computed(components_info, labels_array, method_info_str).
    """
    threshold_computed = pyqtSignal(object, object, str)  # components_info, labels, method_info
    error = pyqtSignal(str)

    def __init__(
        self,
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
        method: str,
        **kwargs,
    ):
        super().__init__()
        self.pet_image = pet_image
        self.mask_image = mask_image
        self.roi_mask = roi_mask
        self.method = method
        self.kwargs = kwargs

    def run(self):
        try:
            if self.method == "adaptive":
                from ...core.engine.adaptive_thresholding_refinement_engine import (
                    AdaptiveThresholdingRefinementEngine,
                    BackgroundMode,
                )
                bg_mode = BackgroundMode(self.kwargs.get("background_mode", "outside_isocontour"))
                engine = AdaptiveThresholdingRefinementEngine(
                    isocontour_fraction=self.kwargs.get("isocontour_fraction", 0.70),
                    background_mode=bg_mode,
                    border_thickness=self.kwargs.get("border_thickness", 3),
                )
                print(
                    f"[ThresholdCompute] Adaptive — iso={self.kwargs.get('isocontour_fraction', 0.70)}, "
                    f"bg={self.kwargs.get('background_mode', 'outside_isocontour')}, "
                    f"border={self.kwargs.get('border_thickness', 3)}"
                )
                components_info, labels = engine.compute_threshold(self.pet_image, self.roi_mask)
                method_info = "Adaptive Thresholding (Nestle et al.)"

            elif self.method == "iterative":
                from ...core.engine.iterative_thresholding_refinement import IterativeThresholdingEngine
                engine = IterativeThresholdingEngine(
                    m=self.kwargs.get("m", 7.8),
                    c1=self.kwargs.get("c1", 61.7),
                    c0=self.kwargs.get("c0", 31.6),
                    convergence_tol=self.kwargs.get("convergence_tol", 0.03),
                    max_iterations=self.kwargs.get("max_iterations", 10),
                )
                print(
                    f"[ThresholdCompute] Iterative — m={self.kwargs.get('m', 7.8)}, "
                    f"c1={self.kwargs.get('c1', 61.7)}, c0={self.kwargs.get('c0', 31.6)}"
                )
                components_info, labels = engine.compute_threshold(
                    self.pet_image, self.mask_image, self.roi_mask
                )
                method_info = "Iterative Thresholding (Jentzen et al.)"
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Log per-component results
            for comp in components_info:
                print(
                    f"[ThresholdCompute] Component {comp['label']}: "
                    f"threshold={comp['threshold']:.4f} SUV, "
                    f"n_voxels={comp['n_voxels']}"
                )

            self.threshold_computed.emit(components_info, labels, method_info)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
