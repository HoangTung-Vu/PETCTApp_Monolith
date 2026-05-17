"""
REVIEW TEST — B4 fix correctness
================================

Verify rằng phiên bản tối ưu (bbox-restricted preview) cho ra cùng kết quả
với phiên bản cũ (full-volume copy), trên dữ liệu giả lập.

Cũng đo thời gian per-tick để xác nhận speedup.
"""

import time
import numpy as np
from scipy.ndimage import find_objects, label as cc_label


def old_preview_tick(base_preview, active_comp_mask, active_pet_vals, threshold):
    """Phiên bản cũ — full-volume copy + fancy index assign."""
    preview = base_preview.copy()
    valid_indices = active_pet_vals >= threshold
    preview[active_comp_mask] = valid_indices
    return preview


def new_preview_tick(preview_buffer, base_preview, active_comp_mask,
                     active_pet_vals, active_bbox, threshold):
    """Phiên bản mới — chỉ ghi trong bbox."""
    buf = preview_buffer
    slc = active_bbox
    # Step 1: restore bbox region from base
    buf[slc] = base_preview[slc]
    # Step 2: apply threshold inside bbox
    valid_indices = active_pet_vals >= threshold
    buf_crop = buf[slc]
    mask_crop = active_comp_mask[slc]
    buf_crop[mask_crop] = valid_indices
    return buf


def main():
    shape = (300, 512, 512)
    print(f"=== B4 fix correctness + speed ===  volume={shape}")
    rng = np.random.default_rng(7)

    # Simulate 3 connected components in ROI labels
    roi_labels = np.zeros(shape, dtype=np.int32)
    roi_labels[100:160, 100:160, 100:160] = 1   # current (active)
    roi_labels[200:230, 200:230, 200:230] = 2   # other
    roi_labels[50:80, 300:330, 350:400] = 3     # other

    roi_slices = find_objects(roi_labels)

    # base_preview: components 2 and 3 already thresholded/painted
    base_preview = np.zeros(shape, dtype=np.uint8)
    base_preview[roi_labels == 2] = 1   # threshholded
    base_preview[roi_labels == 3] = 1   # painted

    # active component (current)
    active_comp_mask = (roi_labels == 1)
    pet_data = rng.random(shape, dtype=np.float32) * 20
    active_pet_vals = pet_data[active_comp_mask]
    active_bbox = roi_slices[0]   # bbox of label=1

    # Allocate persistent buffer (like new code)
    preview_buffer = base_preview.copy()

    # Correctness test across thresholds
    thresholds = [0.5, 5.0, 10.0, 15.0]
    print("\nCorrectness:")
    for t in thresholds:
        old = old_preview_tick(base_preview, active_comp_mask, active_pet_vals, t)
        new = new_preview_tick(preview_buffer, base_preview, active_comp_mask,
                               active_pet_vals, active_bbox, t)
        same = np.array_equal(old, new)
        diff = int((old != new).sum())
        print(f"  threshold={t:5.1f}  identical={same}  voxels differing={diff}")

    # Speed
    N = 5
    print(f"\nSpeed (median of {N} ticks):")
    t_old = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = old_preview_tick(base_preview, active_comp_mask, active_pet_vals, 5.0)
        t_old.append(time.perf_counter() - t0)
    print(f"  OLD (full-volume copy):  {np.median(t_old)*1000:7.2f} ms")

    t_new = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = new_preview_tick(preview_buffer, base_preview, active_comp_mask,
                             active_pet_vals, active_bbox, 5.0)
        t_new.append(time.perf_counter() - t0)
    print(f"  NEW (bbox-restricted):   {np.median(t_new)*1000:7.2f} ms")
    print(f"  Speedup:                 {np.median(t_old)/max(np.median(t_new),1e-9):.0f}×")


if __name__ == "__main__":
    main()
