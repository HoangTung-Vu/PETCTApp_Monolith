"""
REVIEW TEST — Eraser flood-fill cost (no bounding-box crop)
===========================================================

Trong `EraserMixin._make_eraser_callback` (src/gui/components/layout/eraser_manager.py)
và `EraserFloodWorker.run` (src/gui/workers/eraser_worker.py), code dùng
`skimage.morphology.flood(mask_zyx, seed)` trên TOÀN BỘ volume.

Nếu khối u nhỏ nhưng volume to (512×512×300), flood vẫn quét cả mảng → chi phí
cao và memory cao. Đặc biệt fallback `scipy.ndimage.label(mask_zyx)` thì label
TOÀN BỘ ROI → còn tệ hơn nếu có nhiều khối u.

Test này so sánh:
  - flood() full-volume (current)
  - flood() trên bounding-box crop (alternative)
  - scipy.ndimage.label() full-volume (fallback path)
  - scipy.ndimage.label() trên bounding-box crop
"""

import time
import numpy as np

try:
    from skimage.morphology import flood
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from scipy.ndimage import label as nd_label, find_objects


def make_dataset(shape, num_lesions: int = 10, lesion_size: int = 30):
    """Tạo mask 3D với num_lesions khối u rải rác."""
    rng = np.random.default_rng(42)
    arr = np.zeros(shape, dtype=np.uint8)
    centers = []
    for _ in range(num_lesions):
        cz = rng.integers(lesion_size, shape[0] - lesion_size)
        cy = rng.integers(lesion_size, shape[1] - lesion_size)
        cx = rng.integers(lesion_size, shape[2] - lesion_size)
        r = rng.integers(lesion_size // 4, lesion_size // 2)
        zz, yy, xx = np.ogrid[
            -r:r + 1, -r:r + 1, -r:r + 1
        ]
        sphere = (zz**2 + yy**2 + xx**2) <= r**2
        arr[cz - r:cz + r + 1, cy - r:cy + r + 1, cx - r:cx + r + 1][sphere] = 1
        centers.append((int(cz), int(cy), int(cx)))
    return arr, centers


def bench_flood_full(mask: np.ndarray, seed: tuple) -> float:
    t0 = time.perf_counter()
    _ = flood(mask, seed)
    return time.perf_counter() - t0


def bench_flood_cropped(mask: np.ndarray, seed: tuple, margin: int = 5) -> float:
    """Crop bbox của connected component sơ bộ trước khi flood."""
    t0 = time.perf_counter()
    # Crude bbox: nhân với label trước, lấy bbox của cùng label
    # Trong thực tế ta có thể không biết bbox → giả định ta dùng flood trên crop
    # quanh seed với kích thước dự đoán
    z, y, x = seed
    shape = mask.shape
    crop = mask[
        max(0, z - 80):min(shape[0], z + 80),
        max(0, y - 80):min(shape[1], y + 80),
        max(0, x - 80):min(shape[2], x + 80),
    ]
    new_seed = (min(80, z), min(80, y), min(80, x))
    _ = flood(crop, new_seed)
    return time.perf_counter() - t0


def bench_label_full(mask: np.ndarray, seed: tuple) -> float:
    t0 = time.perf_counter()
    labeled, _ = nd_label(mask)
    cid = labeled[seed]
    _ = labeled == cid
    return time.perf_counter() - t0


def bench_label_cropped(mask: np.ndarray, seed: tuple) -> float:
    """Cropped label, dùng find_objects để xác định bbox."""
    t0 = time.perf_counter()
    # bbox của ROI nói chung
    nz = np.nonzero(mask)
    z0, z1 = nz[0].min(), nz[0].max() + 1
    y0, y1 = nz[1].min(), nz[1].max() + 1
    x0, x1 = nz[2].min(), nz[2].max() + 1
    sub = mask[z0:z1, y0:y1, x0:x1]
    labeled, _ = nd_label(sub)
    new_seed = (seed[0] - z0, seed[1] - y0, seed[2] - x0)
    cid = labeled[new_seed]
    _ = labeled == cid
    return time.perf_counter() - t0


def main():
    shape = (300, 512, 512)
    print(f"=== Eraser flood-fill cost ===  volume={shape}")

    mask, centers = make_dataset(shape, num_lesions=10, lesion_size=30)
    print(f"Filled {mask.sum()/1e6:.2f} M voxels across {len(centers)} lesions.")
    print(f"Density: {mask.mean()*100:.2f}% of volume.\n")

    seed = centers[0]
    N = 3

    if HAS_SKIMAGE:
        flood_full = np.median([bench_flood_full(mask, seed) for _ in range(N)])
        flood_crop = np.median([bench_flood_cropped(mask, seed) for _ in range(N)])
        print(f"skimage.flood (full volume)    : {flood_full*1000:7.1f} ms")
        print(f"skimage.flood (160³ crop)      : {flood_crop*1000:7.1f} ms"
              f"  →  {flood_full/max(flood_crop,1e-9):.1f}× speed-up if cropped")
    else:
        print("(skimage không có — bỏ qua flood)")

    label_full = np.median([bench_label_full(mask, seed) for _ in range(N)])
    label_crop = np.median([bench_label_cropped(mask, seed) for _ in range(N)])
    print(f"scipy.ndimage.label (full)     : {label_full*1000:7.1f} ms")
    print(f"scipy.ndimage.label (bbox crop): {label_crop*1000:7.1f} ms"
          f"  →  {label_full/max(label_crop,1e-9):.1f}× speed-up if cropped")

    print(
        "\nNote: Eraser path dùng skimage.flood (nhanh hơn label) — nhưng nếu skimage"
        "\nkhông có, fallback sang scipy.ndimage.label TOÀN VOLUME. Bug tiềm tàng:"
        "\nuser khong cài skimage → mỗi click eraser tốn ~5-10s thay vì <1s."
    )


if __name__ == "__main__":
    main()
