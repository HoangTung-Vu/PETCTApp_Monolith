import numpy as np
import napari
import time

print("Creating 10 viewers with multiscale=False")
start = time.time()
data = np.zeros((275, 512, 512), dtype=np.uint8)

viewers = []
for i in range(10):
    t0 = time.time()
    viewer = napari.Viewer(show=False)
    viewer.add_labels(data, name="tumor", multiscale=False)
    viewers.append(viewer)
    print(f"Viewer {i} took {time.time()-t0:.2f}s")
print(f"Total time: {time.time()-start:.2f}s")

print("Creating 10 viewers with multiscale=True/None")
start = time.time()
viewers2 = []
for i in range(10):
    t0 = time.time()
    viewer = napari.Viewer(show=False)
    viewer.add_labels(data, name="tumor")
    viewers2.append(viewer)
    print(f"Viewer {i} took {time.time()-t0:.2f}s")
print(f"Total time: {time.time()-start:.2f}s")
