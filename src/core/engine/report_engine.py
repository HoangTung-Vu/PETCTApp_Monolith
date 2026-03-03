"""Engine for computing PET/CT clinical report metrics (per-lesion)."""
import numpy as np
import nibabel as nib
from scipy.ndimage import label


class ReportEngine:
    """Computes clinical PET metrics on a per-lesion basis using connected
    component analysis.

    Per-lesion metrics:
        SUVmax  Maximum PET intensity within the lesion.
        SUVmean Mean PET intensity within the lesion.
        MTV     Metabolic Tumour Volume (mL) of the lesion.

    Global metrics:
        gTLG    Global Total Lesion Glycolysis = Σ(SUVmean_i × MTV_i).
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
        """Compute the per-lesion report dictionary.

        Uses connected component labeling to identify individual lesions in
        the binary mask.  Each lesion gets its own SUVmax, SUVmean, MTV, as
        well as its voxel coordinates and 3-D bounding box.

        Args:
            pet_image:  PET NIfTI image (intensity values = SUV or raw counts).
            mask_image: Binary segmentation mask NIfTI image (0/1).

        Returns:
            dict with keys:
                ``gTLG``    float, global Total Lesion Glycolysis.
                ``lesions`` list of dicts, each containing:
                    ``id``      int, 1-based lesion ID.
                    ``SUVmax``  float
                    ``SUVmean`` float
                    ``MTV``     float (mL)
                    ``voxels``  list[tuple[int,int,int]] (z, y, x)
                    ``bbox``    tuple (z_min, y_min, x_min, z_max, y_max, x_max)

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
            raise ValueError("Mask is empty - no voxels to compute report on.")

        # Voxel dimensions derived from the affine (source of truth)
        voxel_dims = ReportEngine._voxel_sizes_from_affine(pet_image.affine)
        voxel_vol_mm3 = float(np.prod(voxel_dims))
        voxel_vol_ml = voxel_vol_mm3 / 1000.0

        # ── Connected component labeling ──
        labeled_array, num_features = label(roi)

        lesions = []
        g_tlg = 0.0

        for lesion_id in range(1, num_features + 1):
            component_mask = labeled_array == lesion_id
            pet_vals = pet_data[component_mask]

            suv_max = float(np.max(pet_vals))
            suv_mean = float(np.mean(pet_vals))
            n_voxels = int(np.count_nonzero(component_mask))
            mtv = n_voxels * voxel_vol_ml

            # Accumulate global TLG
            g_tlg += suv_mean * mtv

            # Voxel positions as list of (z, y, x) tuples
            coords = np.argwhere(component_mask)  # shape (N, 3)
            voxels = [tuple(int(c) for c in row) for row in coords]

            # 3-D bounding box: (z_min, y_min, x_min, z_max, y_max, x_max)
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            bbox = (int(z_min), int(y_min), int(x_min),
                    int(z_max), int(y_max), int(x_max))

            lesions.append({
                "id": lesion_id,
                "SUVmax": round(suv_max, 2),
                "SUVmean": round(suv_mean, 2),
                "MTV": round(mtv, 2),
                "voxels": voxels,
                "bbox": bbox,
            })

        # Sort lesions by Z position (descending) so IDs go from head to feet.
        # In nibabel (X, Y, Z) order: bbox indices 2 and 5 are z_min and z_max.
        lesions.sort(key=lambda l: (l["bbox"][2] + l["bbox"][5]) / 2.0,
                     reverse=True)

        # Re-assign sequential IDs after sorting
        for idx, lesion in enumerate(lesions, start=1):
            lesion["id"] = idx

        return {
            "gTLG": round(g_tlg, 2),
            "lesions": lesions,
        }
