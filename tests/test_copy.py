import numpy as np
import time

roi_mask = np.zeros((500, 512, 512), dtype=np.uint8)
roi_mask[100:400, 100:400, 100:400] = 1
roi_data = np.flip(np.transpose(roi_mask, (2, 1, 0)), axis=(0, 1))

t0 = time.time()
roi_contig3 = roi_data.copy(order='C')
t1 = time.time()
print(f"copy(order='C'): {t1-t0:.4f}")

