import numpy as np
import nibabel as nib

class RefinementEngine:
    """
    Engine for refining segmentation masks based on SUV thresholds or other criteria.
    Operates on numpy arrays (X, Y, Z) directly.
    """
    
    @staticmethod
    def refine_suv(pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float, roi_mask: np.ndarray = None) -> nib.Nifti1Image:
        """
        Refines the mask by keeping only voxels where PET intensity >= threshold.
        If roi_mask is provided, refinement only happens within the ROI.
        
        Args:
            pet_image: NIfTI image of PET data.
            mask_image: NIfTI image of the mask to refine (0 or 1).
            threshold: SUV threshold value.
            roi_mask: Optional boolean/binary array of the ROI to refine within.
            
        Returns:
            Refined mask NIfTI image (same shape and affine as mask_image).
        """
        pet_data = pet_image.get_fdata(dtype=np.float32)
        mask_data = np.asarray(mask_image.dataobj, dtype=np.uint8)

        if pet_data.shape != mask_data.shape:
            raise ValueError(f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}")
            
        if roi_mask is not None:
            # ROI-scoped refinement: only touch voxels inside ROI
            refined_mask = mask_data.copy()
            
            # Voxels to remove: (inside ROI) AND (originally in mask) AND (PET < threshold)
            to_remove = (roi_mask > 0) & (mask_data > 0) & (pet_data < threshold)
            refined_mask[to_remove] = 0
            
            # Ensure binary
            refined_mask = (refined_mask > 0).astype(np.uint8)
        else:
            # Global refinement (original behavior)
            threshold_mask = pet_data >= threshold
            refined_mask = np.logical_and(mask_data > 0, threshold_mask).astype(np.uint8)
        
        # Wrap back into NIfTI using the original mask's affine
        return nib.Nifti1Image(refined_mask, mask_image.affine, mask_image.header)
