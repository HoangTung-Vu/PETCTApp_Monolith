"""
REVIEW TEST — `_has_unsaved_segmentation()` cost
================================================

Đo chi phí của hàm `_has_unsaved_segmentation()` trong main_window.py.
Hàm này được gọi mỗi khi user đóng app / chuyển session.

Các bước nó làm:
  1. `_sync_tumor_from_viewer()`  — gọi `from_napari` (ZYX → XYZ, full volume copy)
  2. `set_tumor_mask(mask_array)` — `astype(np.uint8)` (copy nếu khác dtype) +
                                      tạo Nifti1Image (lưu reference)
  3. `nib.load(disk_path).dataobj` + `np.asarray(..., dtype=np.uint8)`
                                   — đọc file .nii.gz từ DISK + decompress + copy
  4. `np.asarray(sm.tumor_mask.dataobj, dtype=np.uint8)`
                                   — copy lần nữa cho in-memory
  5. `np.array_equal(disk_data, mem_data)` — so sánh full-volume

Ta mô phỏng chi phí bước (3)+(4)+(5) trên 1 file thực tế giả lập.
"""

import time
import tempfile
import os
from pathlib import Path
import numpy as np
import nibabel as nib


def simulate_unsaved_check(disk_path: Path, mem_mask: np.ndarray) -> float:
    """Mô phỏng đúng các bước (3) + (4) + (5)."""
    t0 = time.perf_counter()
    disk_data = np.asarray(nib.load(disk_path).dataobj, dtype=np.uint8)
    mem_data = mem_mask.astype(np.uint8, copy=False)
    _ = np.array_equal(disk_data, mem_data)
    return time.perf_counter() - t0


def simulate_optimized_check(disk_path: Path, mem_mask: np.ndarray,
                             cached_disk_mtime: float | None,
                             cached_disk_hash: int | None) -> tuple[float, float, int]:
    """Alternative: chỉ kiểm tra mtime — gần như free."""
    t0 = time.perf_counter()
    cur_mtime = os.path.getmtime(disk_path)
    if cached_disk_mtime is not None and cur_mtime == cached_disk_mtime:
        # Disk file chưa đổi → có thể so sánh in-memory hash với cache
        ...
    cur_hash = int(mem_mask.sum())  # placeholder cheap "hash"
    return time.perf_counter() - t0, cur_mtime, cur_hash


def main():
    shape = (512, 512, 300)
    print(f"=== Unsaved-segmentation check cost ===")
    print(f"Volume shape: {shape}  (= {np.prod(shape)/1e6:.1f} M voxels)")

    mask = np.zeros(shape, dtype=np.uint8)
    mask[200:400, 200:400, 100:200] = 1

    with tempfile.TemporaryDirectory() as tmp:
        disk_path = Path(tmp) / "seg.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.eye(4)), disk_path)
        file_mb = disk_path.stat().st_size / 1e6
        print(f"On-disk .nii.gz size: {file_mb:.2f} MB")

        # Run several times
        N = 5
        times = [simulate_unsaved_check(disk_path, mask) for _ in range(N)]
        print(f"\nCurrent implementation (load disk + array_equal):")
        for i, t in enumerate(times):
            print(f"  run {i}: {t*1000:.1f} ms")
        print(f"  median: {np.median(times)*1000:.1f} ms")

        ot, *_ = simulate_optimized_check(disk_path, mask, None, None)
        print(f"\nIf we just stat mtime + cheap hash:")
        print(f"  {ot*1000:.3f} ms  ({np.median(times)/max(ot,1e-9):.0f}× faster)")

        # PLUS: in real code, BEFORE the steps above, _sync_tumor_from_viewer()
        # runs and does a full from_napari (XYZ→ZYX, full-volume copy) and a
        # full astype/copy back into a new Nifti1Image. That cost is ON TOP.
        from_napari_cost = 0.0
        zyx = np.flip(np.transpose(mask, (2, 1, 0)), axis=(0, 1)).copy()
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.utils.nifti_utils import from_napari
        N = 3
        tts = []
        for _ in range(N):
            t0 = time.perf_counter()
            _ = from_napari(zyx)
            tts.append(time.perf_counter() - t0)
        from_napari_cost = float(np.median(tts))
        print(f"\nPlus `_sync_tumor_from_viewer()` → from_napari median: {from_napari_cost*1000:.1f} ms")
        total = float(np.median(times)) + from_napari_cost
        print(f"\nGRAND TOTAL per check (sync + disk reload + equality): {total*1000:.1f} ms")
        print("→ Gặp khi user đóng app, mở session khác hoặc bất cứ thao tác nào")
        print("  gọi _prompt_unsaved_segmentation('switch' | 'close').")


if __name__ == "__main__":
    main()
