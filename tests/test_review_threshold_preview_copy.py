"""
REVIEW TEST — `_update_component_preview` allocates full volume copy per slider tick
======================================================================================

Hàm `RefinementHandlerMixin._update_component_preview(threshold)` (gọi mỗi 60ms khi
user kéo slider trong dialog) làm:

    preview = self._base_preview.copy()                    # full volume copy uint8
    valid_indices = self._active_pet_vals >= threshold      # 1D boolean
    preview[self._active_comp_mask] = valid_indices         # boolean fancy index
    preview_zyx = np.flip(np.transpose(preview, (2, 1, 0)), axis=(0, 1))  # VIEW (cheap)
    self._push_mask_to_all("roi", preview, data_zyx=preview_zyx)

Mỗi tick:
  - 1 full-volume allocation + copy (`.copy()` ~ N voxels)
  - 1 fancy-indexed assignment (cost ~ active component size)
  - downstream `np.copyto(layer.data, preview_zyx)` trong load_mask_zyx
    (full volume copy lần 2)
  - layer.refresh() trên mỗi visible viewer

Test này đo riêng phần CPU phía Python (không tính UI). Slider step ~60ms;
nếu mỗi tick > 60ms → dồn ứ → lag.
"""

import time
import numpy as np


def simulate_preview_tick(base_preview: np.ndarray,
                          active_comp_mask: np.ndarray,
                          active_pet_vals: np.ndarray,
                          threshold: float):
    t0 = time.perf_counter()
    preview = base_preview.copy()                            # alloc + copy
    valid_indices = active_pet_vals >= threshold              # 1D boolean
    preview[active_comp_mask] = valid_indices                 # fancy index assign
    # downstream view
    preview_zyx = np.flip(np.transpose(preview, (2, 1, 0)), axis=(0, 1))
    # simulate np.copyto into napari layer storage
    dest = np.empty(preview_zyx.shape, dtype=preview_zyx.dtype)
    np.copyto(dest, preview_zyx)
    return time.perf_counter() - t0


def simulate_preview_tick_optimized(base_preview: np.ndarray,
                                    active_comp_mask: np.ndarray,
                                    active_pet_vals: np.ndarray,
                                    bbox: tuple,
                                    threshold: float):
    """Idea: chỉ copy + push BBOX của active component, không phải full volume."""
    t0 = time.perf_counter()
    z0, z1, y0, y1, x0, x1 = bbox
    # Reuse base_preview as-is (assume push delta only) — chỉ copy crop
    crop_base = base_preview[z0:z1, y0:y1, x0:x1].copy()
    crop_mask = active_comp_mask[z0:z1, y0:y1, x0:x1]
    valid_indices = active_pet_vals >= threshold
    crop_base[crop_mask] = valid_indices
    # Caller would push only the crop (with bbox info) to napari layer
    return time.perf_counter() - t0


def main():
    shape = (512, 512, 300)
    print(f"=== Threshold preview cost per slider tick ===  volume={shape}")

    base_preview = np.zeros(shape, dtype=np.uint8)
    base_preview[100:200, 100:200, 50:100] = 1   # other components

    # Active component is small (typical lesion bbox ~50³)
    active_comp_mask = np.zeros(shape, dtype=bool)
    active_comp_mask[250:300, 250:300, 150:200] = True
    n_active = int(active_comp_mask.sum())
    active_pet_vals = np.random.rand(n_active).astype(np.float32) * 20

    bbox = (250, 300, 250, 300, 150, 200)

    times = [simulate_preview_tick(base_preview, active_comp_mask, active_pet_vals, 5.0)
             for _ in range(5)]
    print(f"Current implementation (full-vol copy + push):")
    for i, t in enumerate(times):
        print(f"  tick {i}: {t*1000:6.1f} ms")
    print(f"  median: {np.median(times)*1000:6.1f} ms")

    times2 = [simulate_preview_tick_optimized(base_preview, active_comp_mask,
                                              active_pet_vals, bbox, 5.0)
              for _ in range(5)]
    print(f"\nOptimized (bbox crop only):")
    for i, t in enumerate(times2):
        print(f"  tick {i}: {t*1000:6.3f} ms")
    print(f"  median: {np.median(times2)*1000:6.3f} ms")
    print(f"\nSpeedup if cropped: {np.median(times)/max(np.median(times2),1e-9):.0f}×")

    print("\nSlider debounce interval: 60ms. Nếu median > 60ms → tick bị backlog → lag.")
    print("Active comp size ratio: {:.5f}% of volume → 99.9% công sức là copy zeros."
          .format(active_comp_mask.mean() * 100))


if __name__ == "__main__":
    main()
