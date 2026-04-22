import numpy as np
import time

print("Creating array...")
a = np.random.randint(0, 2, (512, 512, 300), dtype=np.uint8)

print("Testing to_napari...")
t0 = time.time()
data_zyx = np.transpose(a, (2, 1, 0))
data_zyx = np.flip(data_zyx, axis=(0, 1))
res = np.ascontiguousarray(data_zyx)
t1 = time.time()
print(f"to_napari took {t1-t0:.4f} seconds")

print("Testing boolean thresholding...")
pet_data = np.random.rand(512, 512, 300).astype(np.float32)
base_roi = a
threshold = 0.5
t2 = time.time()
roi = base_roi > 0
result = np.zeros(base_roi.shape, dtype=np.uint8)
result[roi & (pet_data >= threshold)] = 1
t3 = time.time()
print(f"Thresholding took {t3-t2:.4f} seconds")

