import numpy as np

zyx_data = np.zeros((500, 512, 512), dtype=np.uint8)
roi_data = np.flip(np.transpose(zyx_data, (2, 1, 0)), axis=(0, 1))

roi = roi_data > 0
print(f"roi is contiguous? {roi.flags.c_contiguous}")
