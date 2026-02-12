import numpy as np
import nibabel as nib

class RefinementEngine:
    """
    Engine for refining segmentation masks based on SUV thresholds or other criteria.
    Operates on numpy arrays (X, Y, Z) directly.
    """
    
    @staticmethod
    def refine_suv(pet_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, threshold: float) -> nib.Nifti1Image:
        """
        Refines the mask by keeping only voxels where PET intensity >= threshold.
        
        Args:
            pet_image: NIfTI image of PET data.
            mask_image: NIfTI image of the mask to refine (0 or 1).
            threshold: SUV threshold value.
            
        Returns:
            Refined mask NIfTI image (same shape and affine as mask_image).
        """
        pet_data = pet_image.get_fdata()
        mask_data = mask_image.get_fdata()

        if pet_data.shape != mask_data.shape:
            # Try to handle shape mismatch if it's just a channel dim issue
            # But strictly they should match.
            raise ValueError(f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}")
            
        # Create a boolean mask where PET >= threshold
        threshold_mask = pet_data >= threshold
        
        # Intersect with existing mask (keep only parts of the mask that meet the threshold)
        refined_mask = np.logical_and(mask_data > 0, threshold_mask)
        
        # Wrap back into NIfTI using the original mask's affine
        return nib.Nifti1Image(refined_mask.astype(np.uint8), mask_image.affine, mask_image.header)
