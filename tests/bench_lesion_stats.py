"""Benchmark of the per-lesion quantification (ReportEngine.compute_lesion_stats).

Headless — no display needed. Times every stage on synthetic whole-body
volumes laid out like loaded NIfTI data (Fortran order, LAS), plus the GUI-side
conversion used for the double-click boxes, and measures how long the main
thread is stalled while the computation runs in a worker thread (what the user
feels: the compute runs inside the load/save QThreads).

Usage:
    .venv/bin/python tests/bench_lesion_stats.py [--shape X Y Z] [--repeat N]
        [--scenario NAME ...] [--pet PET.nii.gz --seg SEG.nii.gz]
        [--profile] [--out results.json]
"""

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.engine.report_engine import (  # noqa: E402
    ReportEngine, _CONNECTIVITY_26, _label_components, _sphere_kernel, _sphere_mean,
    SUVPEAK_RADIUS_MM,
)
from src.gui.components.layout.lesion_info_manager import LesionInfoMixin  # noqa: E402
from src.utils.dimension_utils import get_spacing_from_affine  # noqa: E402
from src.utils.nifti_utils import mask_bbox  # noqa: E402

AFFINE = np.diag([-0.98, 0.98, 2.0, 1.0])     # LAS, PET resampled onto the CT grid


# ── Synthetic data ───────────────────────────────────────────────────────────

def make_pet(shape, rng):
    X, Y, _ = shape
    xx = np.linspace(-1, 1, X, dtype=np.float32)[:, None]
    yy = np.linspace(-1, 1, Y, dtype=np.float32)[None, :]
    body = (xx / 0.8) ** 2 + (yy / 0.6) ** 2 < 1.0           # (X, Y)
    pet = np.zeros(shape, dtype=np.float32, order="F")
    pet[body] = 1.0
    pet += np.abs(rng.standard_normal(size=shape, dtype=np.float32)) * 0.1
    return pet, body


def add_ball(pet, mask, centre, r, suv):
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r * r
    sl = tuple(slice(c - r, c + r + 1) for c in centre)
    mask[sl][ball] = 1
    pet[sl][ball] = suv + np.linspace(0, 2, int(ball.sum()), dtype=np.float32)


def scenario_data(name, shape, pet, body, rng):
    """Mask for ``name`` (pet is modified in place for the lesion uptake)."""
    X, Y, Z = shape
    mask = np.zeros(shape, dtype=np.uint8, order="F")
    if name == "typical":                     # bench_interaction.py layout
        for f in [(0.35, 0.45, 0.5), (0.65, 0.55, 0.45), (0.5, 0.4, 0.7), (0.45, 0.6, 0.25)]:
            add_ball(pet, mask, tuple(int(a * s) for a, s in zip(f, shape)), 8, 8.0)
    elif name == "many_small":                # spread head → feet: crop ≈ whole volume
        xs, ys = np.nonzero(body)
        for _ in range(100):
            i = rng.integers(len(xs))
            r = int(rng.integers(3, 7))
            c = (int(np.clip(xs[i], r, X - r - 1)), int(np.clip(ys[i], r, Y - r - 1)),
                 int(rng.integers(r, Z - r)))
            add_ball(pet, mask, c, r, float(rng.uniform(3, 15)))
    elif name == "bulky":
        add_ball(pet, mask, (X // 2, Y // 2, Z // 2), 60, 6.0)
        for f in [(0.3, 0.4, 0.2), (0.7, 0.6, 0.8), (0.4, 0.55, 0.9)]:
            add_ball(pet, mask, tuple(int(a * s) for a, s in zip(f, shape)), 6, 9.0)
    elif name != "empty":
        raise ValueError(name)
    return mask


# ── Timing helpers ───────────────────────────────────────────────────────────

def timed(fn, repeat):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, {"median_ms": statistics.median(times), "min_ms": min(times)}


class _NoViewers(LesionInfoMixin):
    """LesionInfoMixin without a LayoutManager (no viewers to push to)."""

    def __init__(self):
        self._init_lesion_info()

    def _get_all_2d_viewers(self):
        return []

    def jump_to_position(self, *pos):
        pass


def main_thread_stall(fn, tick_ms=10.0):
    """Run ``fn`` in a thread; return (longest main-thread stall ms, thread ms)."""
    done = threading.Event()
    elapsed = {}

    def work():
        t0 = time.perf_counter()
        fn()
        elapsed["ms"] = (time.perf_counter() - t0) * 1000.0
        done.set()

    th = threading.Thread(target=work)
    worst = 0.0
    th.start()
    while not done.is_set():
        t0 = time.perf_counter()
        time.sleep(tick_ms / 1000.0)
        worst = max(worst, (time.perf_counter() - t0) * 1000.0 - tick_ms)
    th.join()
    return worst, elapsed["ms"]


# ── Benchmark ────────────────────────────────────────────────────────────────

def bench(name, pet, mask, affine, repeat, profile=False):
    res = {"scenario": name}

    bb, res["mask_bbox"] = timed(lambda: mask_bbox(mask), repeat)
    if bb is not None:
        res["crop_voxels"] = int(np.prod([s.stop - s.start for s in bb]))
        (_, n, _), res["label_26"] = timed(
            lambda: _label_components(mask[bb] > 0, _CONNECTIVITY_26), repeat)
    stats, res["compute_total"] = timed(
        lambda: ReportEngine.compute_lesion_stats(pet, mask, affine), repeat)
    res["n_lesions"] = len(stats["lesions"])

    spacing = [abs(float(s)) for s in get_spacing_from_affine(affine)]
    kernel, half = _sphere_kernel(spacing, SUVPEAK_RADIUS_MM)
    res["suvpeak_kernel_voxels"] = int(kernel.sum())
    _, res["suvpeak_all"] = timed(
        lambda: [_sphere_mean(pet, l["suvmax_voxel"], kernel, half) for l in stats["lesions"]],
        repeat)

    gui = _NoViewers()
    _, res["gui_set_lesions"] = timed(lambda: gui.set_lesions(stats, mask.shape), repeat)
    if stats["lesions"]:
        les = stats["lesions"][-1]
        vx, vy, vz = les["suvmax_voxel"]
        X, Y, Z = mask.shape
        pos = (Z - 1 - vz, Y - 1 - vy, vx)
        hit, res["gui_lesion_at"] = timed(lambda: gui.lesion_at(pos), max(repeat, 20))
        assert hit is not None and hit["id"] == les["id"]

    res["main_thread_stall_ms"], res["threaded_compute_ms"] = main_thread_stall(
        lambda: ReportEngine.compute_lesion_stats(pet, mask, affine))

    if profile:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()
        ReportEngine.compute_lesion_stats(pet, mask, affine)
        prof.disable()
        print(f"---- profile: {name} ----")
        pstats.Stats(prof).sort_stats("cumulative").print_stats(15)
    return res


def fmt(res):
    def ms(key):
        v = res.get(key)
        return f"{v['median_ms']:9.1f}" if v else "        -"
    return (f"{res['scenario']:<12} {res['n_lesions']:>4}  {ms('mask_bbox')} {ms('label_26')} "
            f"{ms('suvpeak_all')} {ms('compute_total')} {ms('gui_set_lesions')} "
            f"{res['main_thread_stall_ms']:9.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=3, default=[512, 512, 600])
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--scenario", nargs="*", default=["typical", "many_small", "bulky", "empty"])
    p.add_argument("--pet", help="real PET NIfTI (SUV)")
    p.add_argument("--seg", help="real segmentation NIfTI on the PET grid")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()

    results = []
    if args.pet and args.seg:
        from src.utils.nifti_utils import load_mask_volume, load_nifti_volume
        pet_img, _, _ = load_nifti_volume(args.pet)
        seg_img = load_mask_volume(args.seg, pet_img.shape)
        cases = [("real", np.asanyarray(pet_img.dataobj), np.asarray(seg_img.dataobj),
                  pet_img.affine)]
        print(f"[bench] real study {pet_img.shape}, PET dtype {cases[0][1].dtype}")
    else:
        shape = tuple(args.shape)
        rng = np.random.default_rng(0)
        print(f"[bench] synthetic {shape}, spacing {np.diag(AFFINE)[:3]} mm ...")
        base_pet, body = make_pet(shape, rng)
        cases = []
        for name in args.scenario:
            pet = np.copy(base_pet, order="F")
            mask = scenario_data(name, shape, pet, body, rng)
            cases.append((name, pet, mask, AFFINE))
        del base_pet

    print(f"{'scenario':<12} {'n':>4}  {'bbox ms':>9} {'label ms':>9} {'peak ms':>9} "
          f"{'TOTAL ms':>9} {'gui ms':>9} {'stall ms':>9}")
    for name, pet, mask, affine in cases:
        res = bench(name, pet, mask, affine, args.repeat, args.profile)
        results.append(res)
        print(fmt(res), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
