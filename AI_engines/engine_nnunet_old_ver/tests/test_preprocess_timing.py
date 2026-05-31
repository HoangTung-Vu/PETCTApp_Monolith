"""Profile each nnU-Net preprocessing step on a full-resolution mock volume.

Why this exists
---------------
The GUI's "Preparing model…" wait is **CPU preprocessing**, not GPU inference.
This script times every step the library actually runs in
``DefaultPreprocessor.run_case_npy`` so we can see where the seconds go and how
much a GPU resample would save.

It uses the **real DicewBCE model config** (``plans.json``) — target spacing,
normalization schemes and resampling kwargs are read from the trained model so
the timings match production, not a guessed config.

Mock input: a 2-channel (CT + PET) volume of nibabel shape ``(512, 512, 300)``
(X, Y, Z) — i.e. Z=300 slices at 512×512 in-plane, the size of a raw whole-body
CT *before* it is resampled onto the model grid. It is transposed to
``(C, Z, Y, X)`` exactly like :meth:`NNUNetEngine.run` feeds nnU-Net.

Run directly for the full table::

    ./.venv/bin/python tests/test_preprocess_timing.py

Note: peak RAM is a few GB (the float64 cast inside resampling of a
2×300×512×512 array). A CUDA device is optional — the GPU comparison is skipped
if none is present.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Make ``from src import ...`` work whether run via pytest (rootdir=engine) or as
# a script (``python tests/...``), then set the nnUNet env so library imports
# don't emit the "nnUNet_results is not defined" warnings.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import WEIGHTS_DIR, setup_nnunet_env  # noqa: E402

setup_nnunet_env()

import torch  # noqa: E402
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero  # noqa: E402
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402
from nnunetv2.preprocessing.resampling.default_resampling import (  # noqa: E402
    compute_new_shape,
    get_do_separate_z,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

MODEL = "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"
CONFIG = "3d_fullres"

# Mock geometry. SHAPE is nibabel order (X, Y, Z); SPACING is mm in the same order.
SHAPE_XYZ = (512, 512, 300)
# (1.0, 1.0, 2.5) -> reversed (2.5, 1.0, 1.0), aniso = 2.5 < 3  => single 3D resize.
SPACING_XYZ = (1.0, 1.0, 2.5)
# An anisotropic spacing that crosses ANISO_THRESHOLD=3 to exercise the slower
# "separate-z" path (per-slice 2D resize + map_coordinates along z).
SPACING_XYZ_ANISO = (0.85, 0.85, 3.0)


class _T:
    """Tiny timing context manager that stores elapsed seconds under ``label``."""

    def __init__(self, label, store):
        self.label, self.store = label, store

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.store[self.label] = time.perf_counter() - self._t


def _load_config():
    base = WEIGHTS_DIR / "nnUNet_results" / MODEL
    plans = json.loads((base / "plans.json").read_text())
    dataset = json.loads((base / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration(CONFIG)
    fg = plans["foreground_intensity_properties_per_channel"]
    return plans, dataset, pm, cm, fg


def _make_mock(shape_xyz):
    """Return a (2, Z, Y, X) float32 volume: a nonzero body inside a zero air border.

    The zero border is what makes ``crop_to_nonzero`` do realistic work. Values are
    random in CT-/PET-like ranges; only the *shape* drives the timings, not values.
    """
    X, Y, Z = shape_xyz
    rng = np.random.default_rng(0)
    # Body occupies the central ~85% in-plane and central ~85% of slices.
    bx, by, bz = slice(40, X - 40), slice(40, Y - 40), slice(20, Z - 20)

    def channel(lo, hi):
        v = np.zeros((X, Y, Z), np.float32)
        sub = v[bx, by, bz]
        v[bx, by, bz] = rng.uniform(lo, hi, sub.shape).astype(np.float32)
        return v.transpose(2, 1, 0)  # X,Y,Z -> Z,Y,X (engine convention)

    ct = channel(-500.0, 600.0)
    pet = channel(0.0, 15.0)
    return np.stack([ct, pet], axis=0)


def _fmt(shape):
    return "×".join(str(int(s)) for s in shape)


def profile(shape_xyz=SHAPE_XYZ, spacing_xyz=SPACING_XYZ, verbose=True):
    """Time each preprocessing step on the mock and return (timings, info) dicts."""
    plans, dataset, pm, cm, fg = _load_config()
    target_spacing = cm.spacing
    fwd = pm.transpose_forward

    data0 = _make_mock(shape_xyz)
    spacing_zyx = list(spacing_xyz[::-1])  # nnU-Net props['spacing'] is Z,Y,X
    prep = DefaultPreprocessor(verbose=False)

    t = {}
    # ── replicate DefaultPreprocessor.run_case_npy, step by step ──
    with _T("copy", t):
        data = np.copy(data0)
    with _T("transpose_forward", t):
        data = data.transpose([0, *[i + 1 for i in fwd]])
    original_spacing = [spacing_zyx[i] for i in fwd]
    shape_before_crop = data.shape[1:]
    with _T("crop_to_nonzero", t):
        data, seg, _bbox = crop_to_nonzero(data)
    shape_after_crop = data.shape[1:]
    with _T("compute_new_shape", t):
        new_shape = compute_new_shape(shape_after_crop, original_spacing, target_spacing)
    with _T("normalize", t):
        data = prep._normalize(data, seg, cm, fg)
    # data is now cropped+normalized (the resample input). Keep a copy for the GPU
    # comparison because resampling_fn_data does not mutate its input.
    norm_data = data
    with _T("resample_order3", t):
        data_rs = cm.resampling_fn_data(data, new_shape, original_spacing, target_spacing)

    # Which resample path did the config take for this spacing?
    sep = (get_do_separate_z(original_spacing) or get_do_separate_z(list(target_spacing)))
    aniso_in = max(original_spacing) / min(original_spacing)

    info = {
        "shape_before_crop": shape_before_crop,
        "shape_after_crop": shape_after_crop,
        "new_shape": tuple(int(s) for s in new_shape),
        "original_spacing_zyx": [round(s, 3) for s in original_spacing],
        "target_spacing_zyx": [round(float(s), 3) for s in target_spacing],
        "separate_z": bool(sep),
        "aniso_in": round(aniso_in, 3),
        "norm_data": norm_data,
        "data_rs": data_rs,
    }

    if verbose:
        total = sum(t.values())
        print(f"\n  input (X,Y,Z)     : {_fmt(shape_xyz)}  spacing(X,Y,Z)={spacing_xyz}")
        print(f"  after crop (Z,Y,X): {_fmt(shape_after_crop)}")
        print(f"  resampled (Z,Y,X) : {_fmt(info['new_shape'])}  "
              f"target_spacing(Z,Y,X)={info['target_spacing_zyx']}")
        print(f"  resample path     : {'SEPARATE-Z (2D/slice + map_coordinates)' if sep else 'single 3D resize'}"
              f"  (input aniso={info['aniso_in']}, threshold=3)")
        print(f"  {'step':<20}{'seconds':>10}{'% total':>10}")
        print(f"  {'-' * 40}")
        for k, v in t.items():
            print(f"  {k:<20}{v:>10.3f}{100 * v / total:>9.1f}%")
        print(f"  {'-' * 40}")
        print(f"  {'TOTAL':<20}{total:>10.3f}{100:>9.1f}%")

    return t, info


def cross_check_end_to_end(shape_xyz=SHAPE_XYZ, spacing_xyz=SPACING_XYZ):
    """Time the library's own run_case_npy end-to-end as a sanity check vs the sum."""
    plans, dataset, pm, cm, fg = _load_config()
    prep = DefaultPreprocessor(verbose=False)
    data0 = _make_mock(shape_xyz)
    props = {"spacing": list(spacing_xyz[::-1])}
    t0 = time.perf_counter()
    prep.run_case_npy(np.copy(data0), None, props, pm, cm, dataset)
    return time.perf_counter() - t0


def compare_gpu_resample(info):
    """Time F.interpolate(trilinear) vs the CPU spline and report speed + numeric diff.

    trilinear ≈ order-1, the CPU path is order-3 (+ optional separate-z), so a
    nonzero diff is expected — it quantifies the accuracy trade-off of a GPU port.
    """
    if not torch.cuda.is_available():
        print("\n  [GPU] no CUDA device -> skipping GPU resample comparison")
        return None
    import torch.nn.functional as F

    dev = torch.device("cuda")
    x = torch.from_numpy(np.ascontiguousarray(info["norm_data"])).unsqueeze(0).to(dev)
    size = tuple(int(s) for s in info["new_shape"])
    torch.cuda.synchronize(dev)
    # Warm up (first interpolate compiles kernels).
    F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    torch.cuda.synchronize(dev)
    gpu_t = time.perf_counter() - t0

    y_np = y.squeeze(0).float().cpu().numpy()
    cpu = info["data_rs"].astype(np.float32)
    abs_diff = np.abs(cpu - y_np)
    rng = float(cpu.max() - cpu.min()) or 1.0
    print(f"\n  [GPU] trilinear resample: {gpu_t:.3f}s  (output {_fmt(size)})")
    print(f"  [GPU] vs CPU spline diff : max={abs_diff.max():.4g}  mean={abs_diff.mean():.4g}"
          f"  (max/range={abs_diff.max() / rng:.3%})")
    return gpu_t


def main():
    print("=" * 64)
    print(f"nnU-Net preprocess timing | model={MODEL}")
    print("=" * 64)

    print("\n[1] Per-step breakdown (default spacing -> single 3D resize)")
    t_default, info = profile(SHAPE_XYZ, SPACING_XYZ)

    print("\n[2] End-to-end cross-check (library run_case_npy)")
    e2e = cross_check_end_to_end(SHAPE_XYZ, SPACING_XYZ)
    print(f"  run_case_npy total : {e2e:.3f}s   (sum of steps above: {sum(t_default.values()):.3f}s)")

    print("\n[3] Resample path comparison (same body, anisotropic spacing)")
    t_aniso, info_aniso = profile(SHAPE_XYZ, SPACING_XYZ_ANISO, verbose=False)
    print(f"  single 3D resize  (aniso={info['aniso_in']}) : "
          f"{t_default['resample_order3']:.3f}s -> {_fmt(info['new_shape'])}")
    print(f"  separate-z resize (aniso={info_aniso['aniso_in']}) : "
          f"{t_aniso['resample_order3']:.3f}s -> {_fmt(info_aniso['new_shape'])}  "
          f"[separate_z={info_aniso['separate_z']}]")

    print("\n[4] GPU resample comparison")
    compare_gpu_resample(info)

    print("\n" + "=" * 64)
    rs = t_default["resample_order3"]
    tot = sum(t_default.values())
    print(f"Verdict: resample = {rs:.2f}s = {100 * rs / tot:.0f}% of preprocess "
          f"({tot:.2f}s). Everything else is cheap CPU.")
    print("=" * 64)


# ── pytest entry: lenient asserts so it doubles as a regression guard ──

def test_resample_is_dominant_step():
    t, info = profile(SHAPE_XYZ, SPACING_XYZ, verbose=False)
    # resample must be the single most expensive preprocessing step.
    assert t["resample_order3"] == max(t.values()), t
    # ...and it should clearly dominate (cheap steps are sub-second).
    assert t["resample_order3"] > 0.5 * sum(t.values()), t
    assert info["new_shape"][1] < info["shape_after_crop"][1]  # in-plane downsampled


def test_steps_match_run_case_npy():
    t, _ = profile(SHAPE_XYZ, SPACING_XYZ, verbose=False)
    e2e = cross_check_end_to_end(SHAPE_XYZ, SPACING_XYZ)
    # Summed per-step timing should be within 60% of the end-to-end call (loose:
    # absorbs allocator/cache variance between the two runs).
    assert 0.4 * e2e < sum(t.values()) < 1.6 * e2e, (sum(t.values()), e2e)


if __name__ == "__main__":
    main()
