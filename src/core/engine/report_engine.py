"""Engine for computing PET/CT clinical report metrics."""
import numpy as np
import nibabel as nib
from scipy.ndimage import uniform_filter


class ReportEngine:
    """Computes clinical PET metrics from a segmentation mask and PET image.

    Metrics:
        SUVmax  – Maximum PET intensity within the mask.
        SUVmean – Mean PET intensity within the mask.
        SUVpeak – Mean PET intensity in a 1 cm³ sphere centred on the hottest voxel.
        MTV     – Metabolic Tumour Volume (mL).
        TLG     – Total Lesion Glycolysis = SUVmean × MTV.
    """

    @staticmethod
    def _voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
        """Derive per-axis voxel sizes (mm) from the affine matrix.

        This is the correct approach – it handles shearing, oblique
        orientations, and non-standard headers that ``header.get_zooms()``
        may misrepresent.

        Returns:
            1-D array of shape (3,) with voxel edge lengths in mm.
        """
        return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    @staticmethod
    def compute_report(
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
    ) -> dict:
        """Compute the full report dictionary.

        Args:
            pet_image:  PET NIfTI image (intensity values = SUV or raw counts).
            mask_image: Binary segmentation mask NIfTI image (0/1).

        Returns:
            dict with keys: SUVmax, SUVmean, SUVpeak, MTV, TLG.

        Raises:
            ValueError: If shapes mismatch or mask is empty.
        """
        pet_data = pet_image.get_fdata().astype(np.float64)
        mask_data = mask_image.get_fdata().astype(np.uint8)

        if pet_data.shape != mask_data.shape:
            raise ValueError(
                f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}"
            )

        roi = mask_data > 0
        if not np.any(roi):
            raise ValueError("Mask is empty – no voxels to compute report on.")

        pet_roi = pet_data[roi]

        # Voxel dimensions derived from the affine (source of truth)
        voxel_dims = ReportEngine._voxel_sizes_from_affine(pet_image.affine)

        # ── Basic metrics ──
        suv_max = float(np.max(pet_roi))
        suv_mean = float(np.mean(pet_roi))

        # ── SUVpeak (1 cm³ sphere around hottest voxel) ──
        suv_peak = ReportEngine._compute_suv_peak(pet_data, roi, voxel_dims)

        # ── MTV (mL) ──
        voxel_vol_mm3 = float(np.prod(voxel_dims))
        voxel_vol_ml = voxel_vol_mm3 / 1000.0
        n_voxels = int(np.count_nonzero(roi))
        mtv = n_voxels * voxel_vol_ml

        # ── TLG ──
        tlg = suv_mean * mtv

        return {
            "SUVmax": round(suv_max, 4),
            "SUVmean": round(suv_mean, 4),
            "SUVpeak": round(suv_peak, 4),
            "MTV": round(mtv, 4),
            "TLG": round(tlg, 4),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_suv_peak(
        pet_data: np.ndarray,
        roi_mask: np.ndarray,
        voxel_dims: np.ndarray,
    ) -> float:
        """SUVpeak = mean intensity inside a 1 cm³ sphere centred on the
        hottest voxel within the ROI.

        The sphere radius is derived so that 4/3·π·r³ = 1000 mm³  →  r ≈ 6.2 mm.
        The kernel size in each dimension is ``2·r / voxel_size`` (rounded to
        the nearest odd integer, minimum 1).

        Args:
            pet_data:   3-D PET array.
            roi_mask:   Boolean mask of the ROI.
            voxel_dims: Per-axis voxel sizes in mm (from affine).
        """
        # Sphere radius for 1 cm³ (1000 mm³)
        sphere_radius_mm = (3.0 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # ≈6.2 mm

        # Kernel sizes per axis (diameter in voxels, at least 1)
        kernel_voxels = np.maximum(
            np.round(2.0 * sphere_radius_mm / voxel_dims).astype(int), 1
        )

        # Use uniform_filter as a fast box-mean approximation
        smoothed = uniform_filter(pet_data, size=kernel_voxels, mode="constant", cval=0.0)

        # Restrict to ROI and pick the maximum mean
        smoothed_roi = smoothed[roi_mask]
        return float(np.max(smoothed_roi))
