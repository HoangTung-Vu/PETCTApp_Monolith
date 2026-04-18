import time
import numpy as np
def to_napari(data_xyz):
    return np.ascontiguousarray(np.flip(data_xyz.transpose(2, 1, 0), axis=(1, 2)))

print("Testing to_napari...")
mask_data = np.zeros((512, 512, 274), dtype=np.uint8)
mask_data[200:300, 200:300, 100:150] = 1

start = time.time()
data_zyx = to_napari(mask_data)
print(f"to_napari took: {time.time() - start:.4f}s")

print("Testing find_objects...")
from scipy.ndimage import label, find_objects
labels, num = label(mask_data)
start = time.time()
slices = find_objects(labels)
print(f"find_objects took: {time.time() - start:.4f}s")

