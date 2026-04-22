import numpy as np
import time

def fast_from_napari(data_zyx):
    # Preallocate perfectly contiguous block
    res = np.zeros((data_zyx.shape[2], data_zyx.shape[1], data_zyx.shape[0]), dtype=data_zyx.dtype)
    
    # Do we get faster performance by allocating first?
    # data_zyx flipped on Z, Y (axes 0, 1): data_zyx[::-1, ::-1, :]
    # assigned transposed into res (XYZ):
    # res(x, y, z) = data(z[::-1], y[::-1], x)
    
    # Let's try direct numpy slicing
    t0 = time.time()
    
    # Actually, iterate through Z axis avoids massive stride penalites on the last dimension?
    # data_zyx shape (Z, Y, X)
    Z, Y, X = data_zyx.shape
    for z in range(Z):
        res[:, :, Z - 1 - z] = data_zyx[z, ::-1, :].T
        
    t1 = time.time()
    print(f"slice assign per Z: {t1-t0:.4f}")
    return res

roi_mask = np.zeros((500, 512, 512), dtype=np.uint8)
roi_mask[100:400, 100:400, 100:400] = 1

t0 = time.time()
fast_from_napari(roi_mask)
print(f"done: {time.time()-t0:.4f}")

