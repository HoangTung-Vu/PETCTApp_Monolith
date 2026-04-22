import numpy as np
import time

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

t0 = time.time()
roi_contig1 = np.ascontiguousarray(roi_data)
t1 = time.time()
print(f"ascontiguousarray: {t1-t0:.4f}")

t2 = time.time()
roi_contig2 = np.zeros(roi_data.shape, dtype=roi_data.dtype)
np.copyto(roi_contig2, roi_data)
t3 = time.time()
print(f"zeros -> copyto: {t3-t2:.4f}")

t4 = time.time()
roi_contig3 = roi_data.copy(order='C')
t5 = time.time()
print(f"copy(order='C'): {t5-t4:.4f}")

