import numpy as np
import time

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
zyx_data[100:400, 100:400, 100:400] = 1
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

t0 = time.time()
roi_bool = roi_data > 0
t1 = time.time()
print(f"roi > 0: {t1-t0:.4f}s")  # generates a non-contiguous bool array

t2 = time.time()
roi_bool_contig = np.ascontiguousarray(roi_bool)
t3 = time.time()
print(f"ascontiguousarray(bool): {t3-t2:.4f}s")

