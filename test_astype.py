import numpy as np

a = np.zeros((2,2,2), dtype=np.uint8)
b = a.astype(np.uint8, copy=False)
b[0,0,0] = 1
print("shares memory:", a[0,0,0] == 1)

a2 = np.zeros((2,2,2), dtype=int)
b2 = a2.astype(np.uint8, copy=False)
b2[0,0,0] = 1
print("shares memory (diff type):", a2[0,0,0] == 1)
