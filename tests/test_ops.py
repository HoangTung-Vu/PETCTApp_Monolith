import numpy as np
import time
from scipy.ndimage import label, find_objects

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
pet_data = np.random.rand(512, 512, 500).astype(np.float32)

zyx_data[200:300, 200:300, 200:300] = 1
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

start_total = time.time()

t0 = time.time()
roi = roi_data > 0
t1 = time.time()
print(f"roi = roi_data > 0: {t1-t0:.4f}")

z_nz, y_nz, x_nz = np.nonzero(roi)
t2 = time.time()
print(f"nonzero(roi): {t2-t1:.4f}")

z_min, z_max = z_nz.min(), z_nz.max() + 1
y_min, y_max = y_nz.min(), y_nz.max() + 1
x_min, x_max = x_nz.min(), x_nz.max() + 1

roi_cropped = roi[z_min:z_max, y_min:y_max, x_min:x_max]
labels_cropped, num_components = label(roi_cropped)
t3 = time.time()
print(f"label(roi_cropped): {t3-t2:.4f}")

labels = np.zeros(roi.shape, dtype=np.int32)
labels[z_min:z_max, y_min:y_max, x_min:x_max] = labels_cropped
slices = find_objects(labels)
t4 = time.time()
print(f"find_objects: {t4-t3:.4f}")

comp_id = 1
slc = slices[0]
comp = labels[slc] == comp_id
pet_slc = pet_data[slc]

i_max = float(pet_slc[comp].max())
isocontour = comp & (pet_slc >= 0.7 * i_max)
t5 = time.time()
print(f"isocontour: {t5-t4:.4f}")

outside_iso = comp & ~isocontour
t6 = time.time()
print(f"outside_iso: {t6-t5:.4f}")

print(f"Total time: {t6-start_total:.4f}")
