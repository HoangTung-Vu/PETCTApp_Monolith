import numpy as np
import nibabel as nib
from ...utils.dimension_utils import get_voxel_volume_ml_from_affine


class IterativeThresholdingEngine:
    """PET volume segmentation via iterative thresholding — Jentzen et al., J Nucl Med 2007; 48:108-114.

    Algorithm:
        Given the measured source-to-background (S/B) ratio of a lesion, the
        optimum threshold T is estimated from a calibrated S/B-threshold-volume
        curve parameterised as:

            T(V, B/S) = m / (V / mL) + c1 * (B/S) + c0          [Eq. 1, paper]

        where B/S = 1 / S/B.

        Starting from a fixed-threshold first estimate (large-volume plateau T1),
        the procedure iterates:
            1. Apply threshold Tn to PET inside the ROI → compute volume Vn.
            2. Evaluate Tn+1 = T(Vn, B/S).
            3. Stop when |Tn+1 - Tn| <= convergence_tol (as a fraction of T1).

    Default calibration coefficients are those derived for the Siemens Biograph
    Emotion Duo / ECAT EXACT HR1 with FORE + AW-OSEM reconstruction (Eq. 1):
        m   = 7.8  % mL
        c1  = 61.7 %           (coefficient of B/S)
        c0  = 31.6 %           (intercept / fixed-threshold offset)

    These should be re-derived for other scanners / reconstruction algorithms.

    Volume computation:
        Volumes are computed by counting suprathreshold voxels within the ROI
        and multiplying by the voxel volume (derived from the NIfTI affine). The
        paper used an ellipsoid model; voxel counting is more accurate for
        irregular shapes and is the recommended approach for modern software
        (see paper Discussion, last paragraph of limitations section).

    Usage::

        engine = IterativeThresholdingEngine()
        refined_nifti = engine.refine(pet_image, mask_image, roi_mask)
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        m: float = 7.8,
        c1: float = 61.7,
        c0: float = 31.6,
        convergence_tol: float = 0.03,
        max_iterations: int = 10,
    ):
        """
        Args:
            m: Slope of the 1/V term in Eq. 1 (units: % · mL). Default 7.8.
            c1: Coefficient of the B/S term in Eq. 1 (units: %). Default 61.7.
            c0: Intercept / fixed-threshold offset (units: %). Default 31.6.
            convergence_tol: Stop iterating when |ΔT| / T1 <= this value.
                Jentzen et al. recommend 3–5 % (0.03–0.05). Default 0.03.
            max_iterations: Hard cap on iterations (safety guard). Default 10.
        """
        if m <= 0:
            raise ValueError("m must be positive.")
        if not (0.0 < convergence_tol < 1.0):
            raise ValueError("convergence_tol must be in (0, 1).")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")

        self.m = m
        self.c1 = c1
        self.c0 = c0
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def refine(
        self,
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
    ) -> nib.Nifti1Image:
        """Apply iterative threshold refinement within the ROI.

        Steps:
            1. Measure source activity (I_max within ROI) and background
               activity (mean PET outside current mask inside ROI bounding box,
               or a caller-supplied background estimate via `i_background`).
            2. Compute measured S/B ratio → B/S ratio.
            3. Evaluate fixed-threshold T1 (large-volume plateau, V → ∞).
            4. Iterate: apply Tn, measure Vn, compute Tn+1, check convergence.
            5. Apply final threshold to produce refined binary mask.

        Background estimation:
            Background is estimated as the mean PET signal of all voxels inside
            the ROI that lie *outside* the current binary mask. If the current
            mask covers the entire ROI, the global ROI minimum is used as a
            conservative fallback.

        Args:
            pet_image:  PET NIfTI (SUV or raw intensity).
            mask_image: Current binary mask NIfTI (0/1).
            roi_mask:   Binary ndarray, same shape as mask_image. Defines the
                        region in which thresholding is applied (e.g. the union
                        of painted voxels from the GUI).

        Returns:
            Refined binary mask NIfTI with the same affine/header as mask_image.
        """
        pet_data = pet_image.get_fdata()
        mask_data = mask_image.get_fdata()
        roi = roi_mask > 0

        self._validate_shapes(pet_data, mask_data, roi)

        # Voxel volume in mL (1 mm³ = 0.001 mL)
        voxel_volume_ml = get_voxel_volume_ml_from_affine(mask_image.affine)

        # Source activity: maximum PET inside ROI
        i_source = float(pet_data[roi].max())
        if i_source <= 0:
            # No PET signal — return mask unchanged
            return nib.Nifti1Image(
                (mask_data > 0).astype(np.uint8),
                mask_image.affine,
                mask_image.header,
            )

        # Background activity: mean PET in ROI voxels outside current mask
        i_background = self._estimate_background(pet_data, roi, mask_data)

        # S/B and B/S ratios
        sb_ratio = i_source / i_background if i_background > 0 else float("inf")
        bs_ratio = 1.0 / sb_ratio if sb_ratio != float("inf") else 0.0

        # Fixed threshold T1 (large-volume plateau, m/V → 0 as V → ∞)
        t1_pct = self.c1 * bs_ratio + self.c0          # percentage of I_source
        t1_abs = (t1_pct / 100.0) * i_source

        # Iterative refinement
        t_current = t1_abs
        refined_mask = self._threshold_mask(pet_data, roi, t_current)

        for _iteration in range(self.max_iterations):
            volume_ml = float(refined_mask[roi].sum()) * voxel_volume_ml

            if volume_ml <= 0:
                # Threshold too high — nothing survives; revert to T1 result
                refined_mask = self._threshold_mask(pet_data, roi, t1_abs)
                break

            t_next_pct = (self.m / volume_ml) + self.c1 * bs_ratio + self.c0
            t_next_abs = (t_next_pct / 100.0) * i_source

            # Convergence check: relative change w.r.t. fixed threshold T1
            delta = abs(t_next_abs - t_current)
            if t1_abs > 0 and (delta / t1_abs) <= self.convergence_tol:
                break

            t_current = t_next_abs
            refined_mask = self._threshold_mask(pet_data, roi, t_current)

        output = (mask_data > 0).astype(np.uint8)
        output[roi] = refined_mask[roi].astype(np.uint8)

        return nib.Nifti1Image(output, mask_image.affine, mask_image.header)

    @staticmethod
    def _validate_shapes(pet_data, mask_data, roi):
        if pet_data.shape != mask_data.shape:
            raise ValueError(
                f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}"
            )
        if pet_data.shape != roi.shape:
            raise ValueError(
                f"Shape mismatch: PET {pet_data.shape} vs ROI {roi.shape}"
            )
        if not roi.any():
            raise ValueError("roi_mask is empty.")

    @staticmethod
    def _estimate_background(
        pet_data: np.ndarray, roi: np.ndarray, mask_data: np.ndarray
    ) -> float:
        """Mean PET in ROI voxels outside the current mask (background region)."""
        bg_mask = roi & (mask_data <= 0)
        if bg_mask.any():
            return float(pet_data[bg_mask].mean())
        return float(pet_data[roi].min())

    @staticmethod
    def _threshold_mask(
        pet_data: np.ndarray, roi: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return boolean mask of voxels inside ROI with PET >= threshold."""
        result = np.zeros(pet_data.shape, dtype=bool)
        result[roi] = pet_data[roi] >= threshold
        return result
