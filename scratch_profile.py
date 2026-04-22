import numpy as np
import time

def from_napari(data_zyx: np.ndarray) -> np.ndarray:
    return np.flip(data_zyx, axis=(0, 1)).transpose(2, 1, 0)

# Create 512x512x800 array
# A typical huge PETCT image might be 512x512 with 400 slices.
data_zyx = np.zeros((400, 512, 512), dtype=np.uint8)

t0 = time.time()
mask_array = from_napari(data_zyx)
t1 = time.time()
print(f"from_napari: {t1-t0:.4f}s")

t0 = time.time()
new_array = mask_array.astype(np.uint8)
t1 = time.time()
print(f"astype(np.uint8): {t1-t0:.4f}s")

t0 = time.time()
new_array_copy = mask_array.copy()
t1 = time.time()
print(f"copy(): {t1-t0:.4f}s")

t0 = time.time()
for _ in range(3):
    other = np.zeros_like(data_zyx)
    np.copyto(other, data_zyx)
t1 = time.time()
print(f"np.copyto (3 times): {t1-t0:.4f}s")
