import numpy as np
import napari
import time

print("Creating mask array...")
data = np.zeros((275, 512, 512), dtype=np.uint8)

print("Creating viewer 1...")
v1 = napari.Viewer(show=False)
v1.add_labels(data, name="tumor", multiscale=False)

print("Creating viewer 2...")
v2 = napari.Viewer(show=False)
v2.add_labels(data, name="tumor", multiscale=False)

print("Done. No crash.")
