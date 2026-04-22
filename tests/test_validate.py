import numpy as np

np.random.seed(42)
data_zyx = np.random.randint(0, 255, (10, 20, 30), dtype=np.uint8)

# OLD
old_res = np.transpose(np.flip(data_zyx, axis=(0, 1)), (2, 1, 0))

# FAST
Z, Y, X = data_zyx.shape
res = np.zeros((X, Y, Z), dtype=data_zyx.dtype)
for z in range(Z):
    res[:, :, Z - 1 - z] = data_zyx[z, ::-1, :].T

print("Arrays equal?", np.array_equal(old_res, res))
