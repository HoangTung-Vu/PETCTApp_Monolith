"""ReportEngine.compute_lesion_stats + crop_to_napari on synthetic LAS volumes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.engine.report_engine import ReportEngine, SUVPEAK_RADIUS_MM  # noqa: E402
from src.utils.nifti_utils import crop_to_napari, to_napari  # noqa: E402

SHAPE = (40, 36, 30)                                   # X, Y, Z
AFFINE = np.diag([-2.0, 2.0, 3.0, 1.0])                # LAS, anisotropic
VOXEL_ML = 2.0 * 2.0 * 3.0 / 1000.0


def volumes(order="F"):
    pet = np.full(SHAPE, 1.0, dtype=np.float32, order=order)
    mask = np.zeros(SHAPE, dtype=np.uint8, order=order)
    return pet, mask


def brute_sphere_mean(pet, center, spacing, radius):
    idx = np.indices(pet.shape)
    d2 = sum(((idx[a] - center[a]) * spacing[a]) ** 2 for a in range(3))
    return float(pet[d2 <= radius ** 2].mean())


def test_two_lesions_exact_values_and_head_to_feet_order():
    pet, mask = volumes()
    # Lesion A low (z 2..5), lesion B high (z 20..23) → B is lesion 1 (head first).
    mask[5:9, 5:9, 2:6] = 1
    pet[5:9, 5:9, 2:6] = 4.0
    pet[6, 6, 3] = 10.0
    mask[20:23, 20:23, 20:24] = 1
    pet[20:23, 20:23, 20:24] = 2.0

    stats = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    a, b = stats["lesions"][1], stats["lesions"][0]
    assert [l["id"] for l in stats["lesions"]] == [1, 2]
    assert b["bbox"] == (20, 20, 20, 22, 22, 23)

    n_a = 4 * 4 * 4
    assert a["n_voxels"] == n_a
    assert a["MTV"] == pytest.approx(n_a * VOXEL_ML)
    assert a["SUVmax"] == pytest.approx(10.0)
    assert a["SUVmean"] == pytest.approx((4.0 * (n_a - 1) + 10.0) / n_a)
    assert a["TLG"] == pytest.approx(a["SUVmean"] * a["MTV"])
    assert a["suvmax_voxel"] == (6, 6, 3)

    n_b = 3 * 3 * 4
    assert b["MTV"] == pytest.approx(n_b * VOXEL_ML)
    assert b["SUVmean"] == pytest.approx(2.0)
    assert stats["tTLG"] == pytest.approx(a["TLG"] + b["TLG"])


def test_corner_touching_blobs_are_one_lesion():
    pet, mask = volumes()
    mask[10:12, 10:12, 10:12] = 1
    mask[12:14, 12:14, 12:14] = 1          # touches only at a corner
    stats = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    assert len(stats["lesions"]) == 1
    assert stats["lesions"][0]["n_voxels"] == 16


def test_suvpeak_sphere_extends_outside_small_lesion():
    pet, mask = volumes()
    rng = np.random.default_rng(1)
    pet[:] = rng.uniform(0.5, 1.5, SHAPE)
    mask[15, 15, 15] = 1                   # 1-voxel lesion, far smaller than 1 cm³
    pet[15, 15, 15] = 20.0
    stats = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    les = stats["lesions"][0]
    expected = brute_sphere_mean(pet, (15, 15, 15), (2.0, 2.0, 3.0), SUVPEAK_RADIUS_MM)
    assert les["SUVpeak"] == pytest.approx(expected)
    assert les["SUVpeak"] < les["SUVmax"]  # background inside the sphere counts


def test_suvpeak_clipped_at_volume_edge():
    pet, mask = volumes()
    pet[:] = np.arange(pet.size, dtype=np.float32).reshape(SHAPE) / pet.size
    mask[0, 0, 0] = 1
    pet[0, 0, 0] = 50.0
    les = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)["lesions"][0]
    expected = brute_sphere_mean(pet, (0, 0, 0), (2.0, 2.0, 3.0), SUVPEAK_RADIUS_MM)
    assert les["SUVpeak"] == pytest.approx(expected)


@pytest.mark.parametrize("order", ["F", "C"])
def test_memory_order_and_int_pet(order):
    pet, mask = volumes(order)
    pet = (pet * 3).astype(np.int16, order=order)
    mask[3:7, 4:9, 5:8] = 1
    les = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)["lesions"][0]
    assert les["n_voxels"] == 4 * 5 * 3
    assert les["SUVmean"] == pytest.approx(3.0)


def test_empty_mask():
    pet, mask = volumes()
    stats = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    assert stats == {"tTLG": 0.0, "lesions": []}


def test_crop_to_napari_matches_full_conversion():
    _, mask = volumes()
    mask[5:9, 5:9, 2:6] = 1
    mask[6, 8, 2] = 0
    mask[20:23, 21:23, 20:24] = 1
    full_zyx = to_napari(mask)
    stats = ReportEngine.compute_lesion_stats(np.ones(SHAPE, np.float32), mask, AFFINE)
    for les in stats["lesions"]:
        (z0, y0, x0), crop = crop_to_napari(les["slices"], les["mask"], SHAPE)
        dz, dy, dx = crop.shape
        np.testing.assert_array_equal(
            full_zyx[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx].astype(bool), crop
        )
