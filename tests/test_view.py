import numpy as np

def to_napari_old(data):
    X, Y, Z = data.shape
    res = np.zeros((Z, Y, X), dtype=data.dtype)
    for z in range(Z):
        res[Z - 1 - z, ::-1, :] = data[:, :, z].T
    return res

def to_napari_new(data):
    return np.transpose(data, (2, 1, 0))[::-1, ::-1, :]

def from_napari_new(data_zyx):
    return np.transpose(data_zyx[::-1, ::-1, :], (2, 1, 0))

data = np.random.rand(3, 4, 5)
old = to_napari_old(data)
new = to_napari_new(data)
print("Are outputs same?", np.allclose(old, new))

# test if editing new modifies data
new[0, 0, 0] = 999.0
print("Did it modify original data?", data[0, 3, 4] == 999.0) # check mapping
