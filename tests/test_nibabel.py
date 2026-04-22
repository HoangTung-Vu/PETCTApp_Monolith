import nibabel as nib
import numpy as np
import time

img = nib.Nifti1Image(np.zeros((500, 512, 512), dtype=np.uint8), np.eye(4))
nib.save(img, 'test_cache.nii.gz')

img_loaded = nib.load('test_cache.nii.gz')

t0 = time.time()
_ = img_loaded.get_fdata(dtype=np.float32)
t1 = time.time()
print(f"First get_fdata: {t1-t0:.4f}s")

t2 = time.time()
_ = img_loaded.get_fdata(dtype=np.float32)
t3 = time.time()
print(f"Second get_fdata: {t3-t2:.4f}s")

