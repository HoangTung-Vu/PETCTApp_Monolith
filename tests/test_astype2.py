import numpy as np
import time

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

t0 = time.time()
res = roi_data.astype(np.uint8)
t1 = time.time()
print(f"astype: {t1-t0:.4f} seconds")

# Also let's check roi & pet_data
pet_data = np.random.rand(512, 512, 500).astype(np.float32)

t2 = time.time()
thresh_test = roi_data > 0
res2 = thresh_test & (pet_data >= 0.5)
t3 = time.time()
print(f"logical AND with pet_data: {t3-t2:.4f} seconds")

t4 = time.time()
res3 = np.copyto(np.zeros_like(roi_data), pet_data >= 0.5, where=thresh_test)
t5 = time.time()
print(f"copyto with where: {t5-t4:.4f} seconds")

