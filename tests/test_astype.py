import numpy as np

arr = np.zeros((10, 10), dtype=np.uint8)
view = arr.T
res = view.astype(np.uint8)

print(f"Is view? {np.may_share_memory(view, res)}")
print(f"Is contiguous? {res.flags.c_contiguous}")
