import numpy as np
import time
from scipy.ndimage import label, find_objects

def simulate_compute():
    roi_mask = np.zeros((500, 512, 512), dtype=np.uint8)
    # create a big box in the middle
    roi_mask[100:400, 100:400, 100:400] = 1
    
    # simulate from_napari view
    roi_mask = np.flip(np.transpose(roi_mask, (2, 1, 0)), axis=(0, 1))

    t0 = time.time()
    roi_mask = np.ascontiguousarray(roi_mask)
    t1 = time.time()
    print(f"ascontiguousarray: {t1-t0:.4f}s")
    
    roi = roi_mask > 0
    t2 = time.time()
    print(f"roi > 0: {t2-t1:.4f}s")
    
    z_nz, y_nz, x_nz = np.nonzero(roi)
    t3 = time.time()
    print(f"nonzero: {t3-t2:.4f}s")
    
    z_min, z_max = z_nz.min(), z_nz.max() + 1
    y_min, y_max = y_nz.min(), y_nz.max() + 1
    x_min, x_max = x_nz.min(), x_nz.max() + 1
    roi_cropped = roi[z_min:z_max, y_min:y_max, x_min:x_max]
    
    labels_cropped, n = label(roi_cropped)
    t4 = time.time()
    print(f"label cropped ({roi_cropped.shape}): {t4-t3:.4f}s")
    
    labels = np.zeros(roi.shape, dtype=np.int32)
    labels[z_min:z_max, y_min:y_max, x_min:x_max] = labels_cropped
    slices = find_objects(labels)
    t5 = time.time()
    print(f"find_objects: {t5-t4:.4f}s")
    
simulate_compute()
