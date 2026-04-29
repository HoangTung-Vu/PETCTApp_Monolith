import napari
import numpy as np

viewer = napari.Viewer(show=False)
layer = viewer.add_labels(np.zeros((10, 10), dtype=np.uint8), name="test")

def callback():
    pass

layer.events.data.connect(callback)

viewer.layers.clear()

try:
    layer.events.data.disconnect(callback)
    print("Disconnected successfully after clear")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

