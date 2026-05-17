"""
REVIEW TEST — to_napari ThreadPoolExecutor có scale theo CPU cores không?
=========================================================================

`to_napari` dùng ThreadPoolExecutor chia chunk theo trục Z; mỗi worker chạy
một loop Python `for z in range(start, end):` với assignment qua slicing.
Việc gán slice qua transpose+flip view release GIL, nhưng overhead Python loop
+ small-slice ops có thể giữ GIL → multi-thread không tăng tốc thực.

Test: chạy `to_napari` với num_threads = 1, 2, 4, 8 và so với plain numpy
`np.ascontiguousarray(np.flip(transpose(...), axis=(0,1)))` (no Python loop).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def to_napari_threaded(data: np.ndarray, num_threads: int) -> np.ndarray:
    X, Y, Z = data.shape
    res = np.zeros((Z, Y, X), dtype=data.dtype)

    def process_chunk(start_z, end_z):
        for z in range(start_z, end_z):
            res[Z - 1 - z, ::-1, :] = data[:, :, z].T

    chunk_size = max(1, Z // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futs = []
        for i in range(num_threads):
            start = i * chunk_size
            end = Z if i == num_threads - 1 else (i + 1) * chunk_size
            if start < Z:
                futs.append(ex.submit(process_chunk, start, end))
        for f in futs:
            f.result()
    return res


def to_napari_vectorized(data: np.ndarray) -> np.ndarray:
    """Pure numpy, no Python loop. flip+transpose lower into stride trick + 1 copy."""
    return np.ascontiguousarray(np.flip(np.transpose(data, (2, 1, 0)), axis=(0, 1)))


def bench(fn, *args, N: int = 3):
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    shape = (512, 512, 300)
    print(f"=== to_napari thread scaling ===  data shape (X,Y,Z) = {shape}")
    print(f"CPU count: {os.cpu_count()}")
    data = np.random.rand(*shape).astype(np.float32)

    for nt in (1, 2, 4, 8, os.cpu_count() or 4):
        t = bench(to_napari_threaded, data, nt, N=3)
        print(f"  threads={nt:>2d}: {t*1000:7.1f} ms")

    t_vec = bench(to_napari_vectorized, data, N=3)
    print(f"\n  pure numpy (flip+transpose+copy, no loop): {t_vec*1000:7.1f} ms")

    print("\nKỳ vọng (nếu GIL không cản): threads=N → tăng tốc ~N×.")
    print("Thực tế thường: tăng từ 1→2 thấy chút, nhưng 2→4→8 chững vì numpy ops")
    print("trên slice nhỏ (X×Y×1) không đủ vốn để giải phóng GIL hết. Và Python")
    print("loop overhead lớn trên mỗi z.")
    print("\nNếu pure numpy nhanh tương đương hoặc nhanh hơn → bỏ ThreadPool đi.")


if __name__ == "__main__":
    main()
