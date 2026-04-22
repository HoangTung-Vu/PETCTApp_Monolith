import numpy as np
import time
from threading import Thread

# Create a huge non-contiguous array
zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

def background_work():
    t0 = time.time()
    # Emulate ThresholdComputeWorker first step
    roi = np.ascontiguousarray(roi_data)
    t1 = time.time()
    print(f"Background: ascontiguousarray took {t1-t0:.4f}s")
    
    # Emulate compute ops
    pet = np.zeros((512, 512, 500), dtype=np.float32)
    b = roi > 0
    np.nonzero(b)
    t2 = time.time()
    print(f"Background: ops took {t2-t1:.4f}s")

t_bg = Thread(target=background_work)
t_bg.start()

start = time.time()
while t_bg.is_alive():
    time.sleep(0.1)
    print(f"Main thread ticking... {time.time()-start:.2f}")

t_bg.join()
