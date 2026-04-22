import napari
import numpy as np
data = np.zeros((10, 10, 10), dtype=np.uint8)
viewer = napari.Viewer(show=False)
layer = viewer.add_labels(data)
print("Data is original data:", layer.data is data)
