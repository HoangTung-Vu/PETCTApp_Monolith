"""Engine for computing PET/CT clinical report metrics (per-lesion)."""
import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
from ...utils.dimension_utils import get_voxel_volume_from_affine


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
    def compute_report(
        pet_image: nib.Nifti1Image,
        mask_image: nib.Nifti1Image,
    ) -> dict:
        """Compute the per-lesion report dictionary.

        Uses connected component labeling to identify individual lesions in
        the binary mask.  Each lesion gets its own SUVmax, SUVmean, MTV, and
        3-D bounding box.

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
                    ``bbox``    tuple (x_min, y_min, z_min, x_max, y_max, z_max)

        Raises:
            ValueError: If shapes mismatch or mask is empty.
        """
        pet_data = pet_image.get_fdata(dtype=np.float32)
        mask_data = mask_image.get_fdata().astype(np.uint8)

        if pet_data.shape != mask_data.shape:
            raise ValueError(
                f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}"
            )

        roi = mask_data > 0
        if not np.any(roi):
            raise ValueError("Mask is empty - no voxels to compute report on.")

        # Voxel volume derived from the affine determinant (source of truth)
        voxel_vol_mm3 = get_voxel_volume_from_affine(pet_image.affine)
        voxel_vol_ml = voxel_vol_mm3 / 1000.0

        # ── Connected component labeling ──
        labeled_array, num_features = label(roi)

        # find_objects returns tight bounding slices per component (O(N) total,
        # not O(N×L)) — avoids scanning the full array once per lesion.
        component_slices = find_objects(labeled_array)

        lesions = []
        g_tlg = 0.0

        for lesion_id, sl in enumerate(component_slices, start=1):
            if sl is None:
                continue

            # Work on the bounding-box subarray only
            sub_labeled = labeled_array[sl]
            sub_pet = pet_data[sl]
            component_mask = sub_labeled == lesion_id
            pet_vals = sub_pet[component_mask]

            suv_max = float(np.max(pet_vals))
            suv_mean = float(np.mean(pet_vals))
            n_voxels = len(pet_vals)
            mtv = n_voxels * voxel_vol_ml

            g_tlg += suv_mean * mtv

            # Bounding box from tight slices (find_objects guarantees tight fit)
            bbox = (
                sl[0].start, sl[1].start, sl[2].start,
                sl[0].stop - 1, sl[1].stop - 1, sl[2].stop - 1,
            )

            lesions.append({
                "id": lesion_id,
                "SUVmax": round(suv_max, 2),
                "SUVmean": round(suv_mean, 2),
                "MTV": round(mtv, 2),
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
