import numpy as np
import time

def fast_to_napari(data_xyz):
    X, Y, Z = data_xyz.shape
    res = np.zeros((Z, Y, X), dtype=data_xyz.dtype)
    for z in range(Z):
        res[Z - 1 - z, :, :] = data_xyz[:, ::-1, z].T
    return res

roi_xyz = np.zeros((512, 512, 500), dtype=np.uint8)
roi_xyz[100:400, 100:400, 100:400] = 1

t0 = time.time()
res = fast_to_napari(roi_xyz)
t1 = time.time()
print(f"fast to_napari: {t1-t0:.4f}")

t2 = time.time()
old_res = np.ascontiguousarray(np.flip(np.transpose(roi_xyz, (2, 1, 0)), axis=(0, 1)))
t3 = time.time()
print(f"old to_napari: {t3-t2:.4f}")

