import numpy as np
import time

# Simulate 512x512x500 image
zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
# Add a non-zero element at the very end
zyx_data[100, 100, 100] = 1

# View with flip and transpose
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

start = time.time()
res = roi_data.any()
end = time.time()

print(f"Time for any() on view: {end - start:.4f} seconds")

start = time.time()
res2 = np.copyto(np.zeros_like(roi_data), roi_data)
end = time.time()

print(f"Time for copyto: {end - start:.4f} seconds")

# Also let's check ndimage.label on the view
from scipy.ndimage import label
start = time.time()
labels, num = label(roi_data)
end = time.time()

print(f"Time for ndimage.label on view: {end - start:.4f} seconds")

start = time.time()
roi_contig = np.ascontiguousarray(roi_data)
end = time.time()

print(f"Time for ascontiguousarray: {end - start:.4f} seconds")

start = time.time()
labels, num = label(roi_contig)
end = time.time()

print(f"Time for ndimage.label on contiguous: {end - start:.4f} seconds")
