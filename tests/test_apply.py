import numpy as np
import time

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
base_roi = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))
base_roi.copy() # still non-contiguous

result = np.zeros((512, 512, 500), dtype=np.uint8)
roi_data = base_roi.copy()

print(f"base_roi contiguous? {base_roi.flags.c_contiguous}")

t0 = time.time()
roi_data[base_roi > 0] = result[base_roi > 0]
t1 = time.time()
print(f"ROI assignment took: {t1-t0:.4f} seconds")

