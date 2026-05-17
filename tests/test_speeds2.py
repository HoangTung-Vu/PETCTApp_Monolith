import numpy as np
import time
from src.utils.nifti_utils import to_napari

data = np.zeros((512, 512, 500), dtype=np.uint8)

t0 = time.time()
res1 = to_napari(data)
t1 = time.time()
print(f"ThreadPoolExecutor (to_napari): {t1-t0:.4f}")

t2 = time.time()
res2 = np.ascontiguousarray(np.flip(data.transpose(2, 1, 0), axis=(0, 1)))
t3 = time.time()
print(f"Numpy ascontiguousarray: {t3-t2:.4f}")

t4 = time.time()
res3 = data.transpose(2, 1, 0)[::-1, ::-1, :].copy(order='C')
t5 = time.time()
print(f"Numpy copy(order='C'): {t5-t4:.4f}")
