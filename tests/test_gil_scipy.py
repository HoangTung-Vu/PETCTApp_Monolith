import numpy as np
import time
from threading import Thread
import scipy.ndimage as ndimage

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

def background_work():
    t0 = time.time()
    roi = np.ascontiguousarray(roi_data)
    
    t1 = time.time()
    # Emulate compute ops
    labeled, n = ndimage.label(roi)
    t2 = time.time()
    print(f"Background: label took {t2-t1:.4f}s")
    
    slices = ndimage.find_objects(labeled)
    t3 = time.time()
    print(f"Background: find_objects took {t3-t2:.4f}s")

t_bg = Thread(target=background_work)
t_bg.start()

start = time.time()
while t_bg.is_alive():
    time.sleep(0.1)
    print(f"Main thread ticking... {time.time()-start:.2f}")

t_bg.join()
