"""
Benchmark: to_napari / from_napari — so sánh các chiến lược tối ưu.

Strategies:
  1. single     – vòng for đơn luồng (baseline)
  2. multi      – ThreadPoolExecutor (code gốc)
  3. numpy_view – np.transpose + flip, trả về VIEW (không copy)
  4. copyto     – loop slice nhưng dùng np.copyto thay vì assignment
  5. flip_copy  – np.flip toàn array rồi transpose, 1 lần ascontiguousarray
  6. chunk_np   – chia chunk, mỗi chunk dùng numpy vectorized (tránh stride penalty lớn)
"""

import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# 1. Single-threaded (baseline)
# ─────────────────────────────────────────────
def to_napari_single(data: np.ndarray) -> np.ndarray:
    X, Y, Z = data.shape
    res = np.zeros((Z, Y, X), dtype=data.dtype)
    for z in range(Z):
        res[Z - 1 - z, ::-1, :] = data[:, :, z].T
    return res


def from_napari_single(data_zyx: np.ndarray) -> np.ndarray:
    Z, Y, X = data_zyx.shape
    res = np.zeros((X, Y, Z), dtype=data_zyx.dtype)
    for z in range(Z):
        res[:, :, Z - 1 - z] = data_zyx[z, ::-1, :].T
    return res


# ─────────────────────────────────────────────
# 2. Multi-threaded (code gốc)
# ─────────────────────────────────────────────
def to_napari_multi(data: np.ndarray, num_threads: Optional[int] = None) -> np.ndarray:
    X, Y, Z = data.shape
    res = np.zeros((Z, Y, X), dtype=data.dtype)
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    def process_chunk(start_z, end_z):
        for z in range(start_z, end_z):
            res[Z - 1 - z, ::-1, :] = data[:, :, z].T

    chunk_size = max(1, Z // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            end = Z if i == num_threads - 1 else (i + 1) * chunk_size
            if start < Z:
                futures.append(executor.submit(process_chunk, start, end))
        for f in futures:
            f.result()
    return res


def from_napari_multi(data_zyx: np.ndarray, num_threads: Optional[int] = None) -> np.ndarray:
    Z, Y, X = data_zyx.shape
    res = np.zeros((X, Y, Z), dtype=data_zyx.dtype)
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    def process_chunk(start_z, end_z):
        for z in range(start_z, end_z):
            res[:, :, Z - 1 - z] = data_zyx[z, ::-1, :].T

    chunk_size = max(1, Z // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            end = Z if i == num_threads - 1 else (i + 1) * chunk_size
            if start < Z:
                futures.append(executor.submit(process_chunk, start, end))
        for f in futures:
            f.result()
    return res


# ─────────────────────────────────────────────
# 3. NumPy view (zero-copy, trả về strided view)
#    Nhanh nhất nếu downstream chấp nhận non-contiguous array.
#    Không dùng ascontiguousarray → tránh stride penalty.
# ─────────────────────────────────────────────
def to_napari_view(data: np.ndarray) -> np.ndarray:
    # (X,Y,Z) -transpose-> (Z,Y,X) -flip Z,Y-> view
    return np.ascontiguousarray(np.flip(np.transpose(data, (2, 1, 0)), axis=(0, 1)))


def from_napari_view(data_zyx: np.ndarray) -> np.ndarray:
    # (Z,Y,X) -flip Z,Y-> -transpose-> (X,Y,Z)
    return np.ascontiguousarray(np.flip(data_zyx, axis=(0, 1)).transpose(2, 1, 0)
)

# ─────────────────────────────────────────────
# 4. np.copyto per slice
#    Dùng np.copyto thay assignment — SIMD tốt hơn trên một số platform.
#    Kết hợp np.ascontiguousarray trên slice nhỏ (tránh penalty lớn).
# ─────────────────────────────────────────────
def to_napari_copyto(data: np.ndarray) -> np.ndarray:
    X, Y, Z = data.shape
    res = np.empty((Z, Y, X), dtype=data.dtype)
    for z in range(Z):
        # data[:, :, z] non-contiguous (column in Z dim) → make contiguous trước
        src = np.ascontiguousarray(data[:, :, z].T[::-1, :])   # shape (Y, X), C-contiguous
        np.copyto(res[Z - 1 - z], src)
    return res


def from_napari_copyto(data_zyx: np.ndarray) -> np.ndarray:
    Z, Y, X = data_zyx.shape
    res = np.empty((X, Y, Z), dtype=data_zyx.dtype)
    for z in range(Z):
        src = np.ascontiguousarray(data_zyx[z, ::-1, :].T)  # shape (X, Y), C-contiguous
        np.copyto(res[:, :, Z - 1 - z], src)
    return res


# ─────────────────────────────────────────────
# 5. flip_copy: flip toàn array 1 lần rồi transpose + 1 copy duy nhất
#    np.flip trả về view, transpose trả về view → chỉ 1 lần copy lớn
#    nhưng copy đó có stride tốt hơn vì flip được fold vào stride
# ─────────────────────────────────────────────
def to_napari_flip_copy(data: np.ndarray) -> np.ndarray:
    # flip axis Z (axis=2) và axis Y (axis=1) rồi transpose
    flipped = np.flip(data, axis=(1, 2))          # view, (X, Y, Z)
    transposed = flipped.transpose(2, 1, 0)        # view, (Z, Y, X)
    return np.ascontiguousarray(transposed)


def from_napari_flip_copy(data_zyx: np.ndarray) -> np.ndarray:
    flipped = np.flip(data_zyx, axis=(0, 1))       # view, (Z, Y, X)
    transposed = flipped.transpose(2, 1, 0)         # view, (X, Y, Z)
    return np.ascontiguousarray(transposed)


# ─────────────────────────────────────────────
# 6. chunk_np: chia nhỏ theo Z-chunk, mỗi chunk dùng numpy vectorized
#    Tránh stride penalty của array lớn, tận dụng cache locality.
# ─────────────────────────────────────────────
def to_napari_chunk_np(data: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    X, Y, Z = data.shape
    res = np.empty((Z, Y, X), dtype=data.dtype)
    for z_start in range(0, Z, chunk_size):
        z_end = min(z_start + chunk_size, Z)
        chunk = data[:, :, z_start:z_end]                 # (X, Y, chunk)
        out_chunk = np.ascontiguousarray(
            np.flip(chunk.transpose(2, 1, 0), axis=(0, 1))
        )                                                   # (chunk, Y, X)
        # Gán vào vị trí đảo ngược trong res
        res_start = Z - z_end
        res_end   = Z - z_start
        res[res_start:res_end] = out_chunk
    return res


def from_napari_chunk_np(data_zyx: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    Z, Y, X = data_zyx.shape
    res = np.empty((X, Y, Z), dtype=data_zyx.dtype)
    for z_start in range(0, Z, chunk_size):
        z_end = min(z_start + chunk_size, Z)
        chunk = data_zyx[z_start:z_end]                    # (chunk, Y, X)
        out_chunk = np.ascontiguousarray(
            np.flip(chunk, axis=(0, 1)).transpose(2, 1, 0)
        )                                                   # (X, Y, chunk)
        res_start = Z - z_end
        res_end   = Z - z_start
        res[:, :, res_start:res_end] = out_chunk
    return res


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────
def bench(label: str, fn, *args, n_runs: int = 3) -> tuple[float, np.ndarray]:
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    median = float(np.median(times))
    print(f"  {label:<22s}: {median:.4f}s  (median of {n_runs})")
    return median, result


def run_tests():
    shape = (512, 512, 300)
    print(f"Generating dummy data {shape} float32...")
    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)
    n_runs = 3

    print(f"\n{'='*55}")
    print(f"  to_napari  — input {data.shape}")
    print(f"{'='*55}")
    _, ref = bench("1. single",        to_napari_single,    data, n_runs=n_runs)
    _, r2  = bench("2. multi",         to_napari_multi,     data, n_runs=n_runs)
    _, r3  = bench("3. view (no copy)",to_napari_view,      data, n_runs=n_runs)
    _, r4  = bench("4. copyto",        to_napari_copyto,    data, n_runs=n_runs)
    _, r5  = bench("5. flip_copy",     to_napari_flip_copy, data, n_runs=n_runs)
    _, r6  = bench("6. chunk_np",      to_napari_chunk_np,  data, n_runs=n_runs)

    ref_c = np.ascontiguousarray(ref)   # view → contiguous để so sánh
    print(f"\n  Correctness (vs single):")
    print(f"    multi      : {np.allclose(ref_c, np.ascontiguousarray(r2))}")
    print(f"    view       : {np.allclose(ref_c, np.ascontiguousarray(r3))}")
    print(f"    copyto     : {np.allclose(ref_c, np.ascontiguousarray(r4))}")
    print(f"    flip_copy  : {np.allclose(ref_c, np.ascontiguousarray(r5))}")
    print(f"    chunk_np   : {np.allclose(ref_c, np.ascontiguousarray(r6))}")

    data_zyx = ref_c  # shape (300, 512, 512)
    print(f"\n{'='*55}")
    print(f"  from_napari — input {data_zyx.shape}")
    print(f"{'='*55}")
    _, bref = bench("1. single",        from_napari_single,    data_zyx, n_runs=n_runs)
    _, b2   = bench("2. multi",         from_napari_multi,     data_zyx, n_runs=n_runs)
    _, b3   = bench("3. view (no copy)",from_napari_view,      data_zyx, n_runs=n_runs)
    _, b4   = bench("4. copyto",        from_napari_copyto,    data_zyx, n_runs=n_runs)
    _, b5   = bench("5. flip_copy",     from_napari_flip_copy, data_zyx, n_runs=n_runs)
    _, b6   = bench("6. chunk_np",      from_napari_chunk_np,  data_zyx, n_runs=n_runs)

    bref_c = np.ascontiguousarray(bref)
    print(f"\n  Correctness (vs single):")
    print(f"    multi      : {np.allclose(bref_c, np.ascontiguousarray(b2))}")
    print(f"    view       : {np.allclose(bref_c, np.ascontiguousarray(b3))}")
    print(f"    copyto     : {np.allclose(bref_c, np.ascontiguousarray(b4))}")
    print(f"    flip_copy  : {np.allclose(bref_c, np.ascontiguousarray(b5))}")
    print(f"    chunk_np   : {np.allclose(bref_c, np.ascontiguousarray(b6))}")

    print(f"\n  Round-trip (data → to_napari → from_napari_single == data):")
    print(f"    OK: {np.allclose(data, bref_c)}")


if __name__ == "__main__":
    run_tests()