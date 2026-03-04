
import sys
import os
import numpy as np
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.engine.refinement_engine import RefinementEngine
from src.utils.nifti_utils import to_napari, from_napari

class TestRefinement(unittest.TestCase):
    def test_suv_refinement(self):
        # 1. Create Synthetic Data (5x5x5)
        # Background = 1.0, Hot Spot = 5.0
        pet_data = np.ones((5, 5, 5))
        pet_data[2, 2, 2] = 5.0 # Hot center
        
        # Mask covers center and some background
        mask_data = np.zeros((5, 5, 5))
        mask_data[2, 2, 2] = 1 # Center
        mask_data[2, 2, 1] = 1 # Neighbor (cold)
        
        # 2. Refine with Threshold 2.5
        import nibabel as nib
        pet_img = nib.Nifti1Image(pet_data, np.eye(4))
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        
        refined_img = RefinementEngine.refine_suv(pet_img, mask_img, 2.5)
        refined = refined_img.get_fdata()
        
        # 3. Assertions
        # Center should be kept
        self.assertEqual(refined[2, 2, 2], 1)
        # Neighbor should be removed
        self.assertEqual(refined[2, 2, 1], 0)
        
    def test_roi_scoped_refinement(self):
        # 1. Create Synthetic Data (5x5x5)
        # Background = 1.0
        # Hot spot 1 = 5.0 at [2,2,2]
        # Hot spot 2 = 3.0 at [4,4,4] (this one should be PROTECTED if outside ROI)
        pet_data = np.ones((5, 5, 5))
        pet_data[2, 2, 2] = 5.0
        pet_data[4, 4, 4] = 3.0
        
        # Mask covers both hot spots and one cold spot
        mask_data = np.zeros((5, 5, 5))
        mask_data[2, 2, 2] = 1 # Hot 1
        mask_data[2, 2, 1] = 1 # Cold neighbor of Hot 1
        mask_data[4, 4, 4] = 1 # Hot 2
        
        # ROI only covers [2,2,*] region
        roi_mask = np.zeros((5, 5, 5))
        roi_mask[2, 2, :] = 1
        
        # 2. Refine with Threshold 4.0
        import nibabel as nib
        pet_img = nib.Nifti1Image(pet_data, np.eye(4))
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        
        # With ROI: 
        # [2,2,2] (5.0 >= 4.0, in ROI) -> KEEP
        # [2,2,1] (1.0 < 4.0, in ROI) -> REMOVE
        # [4,4,4] (3.0 < 4.0, but OUTSIDE ROI) -> PROTECT/KEEP
        refined_img = RefinementEngine.refine_suv(pet_img, mask_img, 4.0, roi_mask)
        refined = refined_img.get_fdata()
        
        # 3. Assertions
        self.assertEqual(refined[2, 2, 2], 1, "Hot 1 should be kept (above threshold)")
        self.assertEqual(refined[2, 2, 1], 0, "Cold neighbor should be removed (in ROI, below threshold)")
        self.assertEqual(refined[4, 4, 4], 1, "Hot 2 should be PROTECTED (outside ROI, even if below threshold)")
        
        # Global refinement check (for comparison)
        refined_global_img = RefinementEngine.refine_suv(pet_img, mask_img, 4.0, None)
        refined_global = refined_global_img.get_fdata()
        self.assertEqual(refined_global[4, 4, 4], 0, "Global refinement should have removed Hot 2")
        
    def test_coordinate_transforms(self):
        # Create unique data
        # shape (X, Y, Z) = (2, 3, 4)
        data = np.arange(24).reshape((2, 3, 4))
        
        # 1. Transform to Napari
        data_napari = to_napari(data)
        
        # Expected shape: (Z, Y, X) = (4, 3, 2)
        self.assertEqual(data_napari.shape, (4, 3, 2))
        
        # 2. Transform back
        data_back = from_napari(data_napari)
        
        # 3. Assert equality
        np.testing.assert_array_equal(data_back, data)
        print("Coordinate transformation cycle successful.")

if __name__ == '__main__':
    unittest.main()
