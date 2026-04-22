import numpy as np
import time

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
preview_zyx = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))
existing_zyx = np.zeros((500, 512, 512), dtype=np.uint8)

t0 = time.time()
np.copyto(existing_zyx, preview_zyx, casting='unsafe')
t1 = time.time()
print(f"copyto from view to contiguous: {t1-t0:.4f}")
