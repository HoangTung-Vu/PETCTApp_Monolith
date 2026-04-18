import numpy as np
import nibabel as nib
from enum import Enum


class BackgroundMode(Enum):
    """How to sample background within the ROI.

    BORDER_PIXELS: N-voxel erosion shell of the ROI (outside isocontour).
    OUTSIDE_ISOCONTOUR: all ROI voxels outside the isocontour.
    """
    BORDER_PIXELS = "border_pixels"
    OUTSIDE_ISOCONTOUR = "outside_isocontour"


class AdaptiveThresholdingRefinementEngine:
    """GTVbg adaptive thresholding — Nestle et al., J Nucl Med 2005; 46:1342-1348.

    Formula:  I_threshold = (0.15 * I_mean) + I_background

        I_mean       : mean PET inside the isocontour_fraction * I_max isocontour (ROI).
        I_background : mean PET of the background region inside the ROI.

    Only voxels in roi_mask are modified.
    """

    def __init__(
        self,
        isocontour_fraction: float = 0.70,
        background_mode: BackgroundMode = BackgroundMode.OUTSIDE_ISOCONTOUR,
        border_thickness: int = 3,
    ):
        """
        Args:
            isocontour_fraction: fraction of I_max for isocontour boundary. Default 0.70.
            background_mode: how to define background inside the ROI.
            border_thickness: erosion depth (voxels) for BORDER_PIXELS mode.
        """
        if not (0.0 < isocontour_fraction < 1.0):
            raise ValueError("isocontour_fraction must be in (0, 1).")
        if border_thickness < 1:
            raise ValueError("border_thickness must be >= 1.")

        self.isocontour_fraction = isocontour_fraction
        self.background_mode = background_mode
        self.border_thickness = border_thickness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_threshold(
        self,
        pet_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
    ):
        """Compute adaptive threshold per connected component in the ROI.

        Uses scipy.ndimage.label to split ROI into connected components,
        then computes the adaptive threshold for each independently.

        Returns:
            Tuple of (components_info, labels_array) where:
            - components_info: list of dicts with keys
              {label, threshold, n_voxels, i_max, i_mean, i_background}
            - labels_array: ndarray from scipy.ndimage.label (same shape as roi_mask)
        """
        from scipy.ndimage import label as cc_label, find_objects

        pet_data = pet_image.get_fdata(dtype=np.float32)
        roi = roi_mask > 0

        if not roi.any():
            raise ValueError("roi_mask is empty.")

        labels, num_components = cc_label(roi)
        slices = find_objects(labels)
        components_info = []

        for comp_id, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            comp = labels[slc] == comp_id
            pet_slc = pet_data[slc]

            n_voxels = int(comp.sum())
            if n_voxels < 2:
                continue

            i_max = float(pet_slc[comp].max())
            if i_max <= 0:
                print(f"[Adaptive] Component {comp_id}: I_max <= 0, skipping.")
                continue

            isocontour = comp & (pet_slc >= self.isocontour_fraction * i_max)
            i_mean = self._compute_i_mean(pet_slc, isocontour, comp)
            i_bg = self._compute_i_background(pet_slc, comp, isocontour)
            thresh = 0.15 * i_mean + i_bg

            # Fallback: if threshold >= i_max, the entire component would vanish.
            if thresh >= i_max:
                print(
                    f"[Adaptive] Component {comp_id}: threshold {thresh:.4f} >= I_max {i_max:.4f}, "
                    f"falling back to 0.5 * I_max = {0.5 * i_max:.4f}"
                )
                thresh = 0.5 * i_max

            components_info.append({
                "label": comp_id,
                "threshold": float(thresh),
                "n_voxels": n_voxels,
                "i_max": float(i_max),
                "i_mean": float(i_mean),
                "i_background": float(i_bg),
            })

        if not components_info:
            print("[Adaptive] No valid components found.")

        # Expose for backward compat
        if components_info:
            weights = [c["n_voxels"] for c in components_info]
            thresholds = [c["threshold"] for c in components_info]
            self.last_threshold = float(np.average(thresholds, weights=weights))
        else:
            self.last_threshold = 0.0

        return components_info, labels

    def refine(
        self,
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
        roi_mask: np.ndarray,
    ) -> nib.Nifti1Image:
        """Apply adaptive threshold refinement within the ROI.

        Steps:
            1. I_max  = max PET in ROI.
            2. isocontour = ROI voxels with PET >= isocontour_fraction * I_max.
            3. I_mean = mean PET inside isocontour.
            4. I_background = mean PET of background (per BackgroundMode).
            5. I_threshold = 0.15 * I_mean + I_background.
            6. Within ROI: remove voxels below threshold, add voxels above.

        Args:
            pet_image:  PET NIfTI (SUV or raw intensity).
            mask_image: current binary mask NIfTI (0/1).
            roi_mask:   binary ndarray, same shape as mask_image.
                        Typically: current_mask - snapshot_mask from GUI paint tool.

        Returns:
            Refined binary mask NIfTI, same affine/header as mask_image.
        """
        pet_data = pet_image.get_fdata(dtype=np.float32)
        mask_data = np.asarray(mask_image.dataobj, dtype=np.uint8)
        roi = (roi_mask > 0)

        self._validate_shapes(pet_data, mask_data, roi)

        i_max = float(pet_data[roi].max())

        if i_max <= 0:
            # No PET signal in ROI — return mask unchanged
            return nib.Nifti1Image(
                (mask_data > 0).astype(np.uint8),
                mask_image.affine,
                mask_image.header,
            )

        isocontour_mask = roi & (pet_data >= self.isocontour_fraction * i_max)
        i_mean = self._compute_i_mean(pet_data, isocontour_mask, roi)
        i_background = self._compute_i_background(pet_data, roi, isocontour_mask)
        i_threshold = 0.15 * i_mean + i_background

        # Expose computed values for the GUI notification
        self.last_threshold = float(i_threshold)
        self.last_i_max = float(i_max)
        self.last_i_mean = float(i_mean)
        self.last_i_background = float(i_background)

        refined = self._apply_threshold(mask_data, pet_data, roi, i_threshold)
        return nib.Nifti1Image(refined, mask_image.affine, mask_image.header)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_shapes(self, pet_data, mask_data, roi):
        if pet_data.shape != mask_data.shape:
            raise ValueError(f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}")
        if pet_data.shape != roi.shape:
            raise ValueError(f"Shape mismatch: PET {pet_data.shape} vs ROI {roi.shape}")
        if not roi.any():
            raise ValueError("roi_mask is empty.")

    def _compute_i_mean(self, pet_data: np.ndarray, isocontour_mask: np.ndarray, roi: np.ndarray) -> float:
        """Mean PET inside isocontour. Falls back to ROI max if isocontour is empty."""
        if not isocontour_mask.any():
            return float(pet_data[roi].max())
        return float(pet_data[isocontour_mask].mean())

    def _compute_i_background(
        self, pet_data: np.ndarray, roi: np.ndarray, isocontour_mask: np.ndarray
    ) -> float:
        """Mean PET of background region inside ROI.

        OUTSIDE_ISOCONTOUR: all ROI voxels outside the isocontour.
        BORDER_PIXELS: erosion shell of ROI, excluding isocontour.
        Falls back to outside_iso mean, then 0.0 if no background voxels exist.
        """
        outside_iso = roi & ~isocontour_mask

        if self.background_mode == BackgroundMode.OUTSIDE_ISOCONTOUR:
            bg_mask = outside_iso
        else:
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(roi, iterations=self.border_thickness, border_value=False)
            bg_mask = (roi & ~eroded) & ~isocontour_mask  # border shell minus isocontour

        if not bg_mask.any():
            return float(pet_data[outside_iso].mean()) if outside_iso.any() else 0.0

        return float(pet_data[bg_mask].mean())

    def _apply_threshold(
        self, mask_data: np.ndarray, pet_data: np.ndarray, roi: np.ndarray, i_threshold: float
    ) -> np.ndarray:
        """Within ROI: keep/add voxels >= threshold, remove voxels below. Outside ROI: unchanged."""
        refined = mask_data.copy()
        refined[roi & (pet_data <  i_threshold)] = 0
        refined[roi & (pet_data >= i_threshold)] = 1
        return (refined > 0).astype(np.uint8)