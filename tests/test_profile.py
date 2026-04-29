import time
import numpy as np
from scipy.ndimage import label as cc_label

roi_mask = np.random.randint(0, 2, (512, 512, 274), dtype=np.uint8)
print("Running cc_label on 71M noisy voxels...")
start = time.time()
labels, n_comp = cc_label(roi_mask)
print(f"cc_label found {n_comp} components in {time.time()-start:.4f}s")
