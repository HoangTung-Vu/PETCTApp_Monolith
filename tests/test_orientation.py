"""Orientation handling: every source orientation must display identically.

Run: .venv/bin/pytest tests/test_orientation.py -q
"""

import itertools

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import axcodes2ornt, ornt_transform

from src.utils.nifti_utils import (
    from_app_orientation, load_mask_volume, load_nifti_volume, to_app_orientation, to_napari,
)

SHAPE_LAS = (20, 16, 12)      # X (→L), Y (→A), Z (→S)
SPACING = (0.8, 0.9, 2.5)


def las_phantom():
    """LAS volume with a distinct marker at the patient-Left, Anterior and Superior ends."""
    data = np.zeros(SHAPE_LAS, dtype=np.float32)
    data[-3:, 6:9, 5:7] = 100.0     # far Left
    data[8:11, -3:, 5:7] = 200.0    # far Anterior
    data[8:11, 6:9, -2:] = 300.0    # far Superior
    affine = np.diag([-SPACING[0], SPACING[1], SPACING[2], 1.0])
    affine[:3, 3] = (40.0, -20.0, 100.0)
    return nib.Nifti1Image(data, affine)


ORIENTATIONS = ["LAS", "LPS", "RAS", "RPI", "LPI", "ASL", "SRP", "PIR"]


def reoriented_copy(img, codes):
    """Physically identical image stored with voxel axes ``codes``."""
    xform = ornt_transform(axcodes2ornt(nib.aff2axcodes(img.affine)), axcodes2ornt(tuple(codes)))
    out = img.as_reoriented(xform)
    assert nib.aff2axcodes(out.affine) == tuple(codes)
    return nib.Nifti1Image(np.ascontiguousarray(np.asarray(out.dataobj)), out.affine)


@pytest.mark.parametrize("codes", ORIENTATIONS)
def test_every_orientation_displays_like_las(codes):
    base = las_phantom()
    ref_zyx = to_napari(np.asarray(base.dataobj))
    src = reoriented_copy(base, codes)

    las, back = to_app_orientation(src)
    assert nib.aff2axcodes(las.affine) == ("L", "A", "S")
    np.testing.assert_allclose(las.affine, base.affine, atol=1e-6)
    np.testing.assert_array_equal(to_napari(np.asarray(las.dataobj)), ref_zyx)

    # Mask round trip: saving restores the source voxel order and affine.
    restored = from_app_orientation(las, back)
    np.testing.assert_allclose(restored.affine, src.affine, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(restored.dataobj), np.asarray(src.dataobj))


def test_display_convention_las():
    """Napari ZYX: row 0 = superior, y-row 0 = anterior, x-col 0 = patient right."""
    zyx = to_napari(np.asarray(las_phantom().dataobj))
    Z, Y, X = zyx.shape
    sup = np.argwhere(zyx == 300.0)
    ant = np.argwhere(zyx == 200.0)
    left = np.argwhere(zyx == 100.0)
    assert sup[:, 0].max() < 2                      # superior at the top of coronal/sagittal
    assert ant[:, 1].max() < 3                      # anterior at the top of axial
    assert left[:, 2].min() >= X - 3                # patient left on the screen's right


def test_load_nifti_volume_keeps_dtype_and_streams_only_las(tmp_path):
    base = las_phantom()
    for codes in ("LAS", "LPS"):
        img = reoriented_copy(base, codes)
        img.set_data_dtype(np.float32)
        path = tmp_path / f"ct_{codes}.nii.gz"
        nib.save(img, path)
        las, back, stream = load_nifti_volume(path)
        assert las.get_data_dtype() == np.float32
        assert np.asarray(las.dataobj).dtype == np.float32
        assert (stream is not None) == (codes == "LAS")
        np.testing.assert_array_equal(to_napari(np.asarray(las.dataobj)),
                                      to_napari(np.asarray(base.dataobj)))

    # int16 stays int16 (no cast) …
    i16 = nib.Nifti1Image(np.asarray(base.dataobj).astype(np.int16), base.affine)
    nib.save(i16, tmp_path / "i16.nii.gz")
    las, _, _ = load_nifti_volume(tmp_path / "i16.nii.gz")
    assert np.asarray(las.dataobj).dtype == np.int16
    # … unless scl_slope scaling has to be applied → float32.
    scaled = nib.Nifti1Image(np.asarray(base.dataobj).astype(np.int16), base.affine)
    scaled.header.set_slope_inter(0.5, -10.0)
    nib.save(scaled, tmp_path / "scaled.nii.gz")
    las, _, _ = load_nifti_volume(tmp_path / "scaled.nii.gz")
    arr = np.asarray(las.dataobj)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, np.asarray(base.dataobj) * 0.5 - 10.0, atol=1e-4)


def test_mask_load_checks_shape(tmp_path):
    base = las_phantom()
    mask = nib.Nifti1Image((np.asarray(base.dataobj) > 0).astype(np.uint8), base.affine)
    lps = reoriented_copy(mask, "LPS")
    nib.save(lps, tmp_path / "seg.nii.gz")
    m = load_mask_volume(tmp_path / "seg.nii.gz", ref_shape=SHAPE_LAS)
    np.testing.assert_array_equal(np.asarray(m.dataobj), np.asarray(mask.dataobj))
    with pytest.raises(ValueError):
        load_mask_volume(tmp_path / "seg.nii.gz", ref_shape=(1, 2, 3))


# ── 3D camera orientation readout ────────────────────────────────────────────

from src.utils.view_orientation import describe_camera, direction_letters  # noqa: E402

# Napari scene axes (z, y, x) = (inferior, posterior, patient left).
SUP, POST, LEFT = (-1, 0, 0), (0, 1, 0), (0, 0, 1)


def neg(v):
    return tuple(-c for c in v)


@pytest.mark.parametrize("view, up, expect", [
    # camera looks posteriorly from the front, head up → anterior coronal MIP
    (POST, SUP, dict(view_from="Anterior", azimuth=0, elevation=0, roll=0,
                     up="S", down="I", left="R", right="L")),
    # camera at the patient's left, looking right → left lateral
    (neg(LEFT), SUP, dict(view_from="Left lateral", azimuth=90, elevation=0, roll=0,
                          up="S", left="A", right="P")),
    # camera behind the patient → posterior view, patient left on screen left
    (neg(POST), SUP, dict(view_from="Posterior", azimuth=180, elevation=0, roll=0,
                          left="L", right="R")),
    # camera above the head looking toward the feet, anterior up
    (neg(SUP), neg(POST), dict(view_from="Superior", elevation=90, roll=0, up="A", down="P")),
    # anterior view, patient rotated 90° clockwise: head points to the screen's right
    (POST, neg(LEFT), dict(view_from="Anterior", roll=90, right="S", up="R", down="L")),
    # … and 90° counter-clockwise: head to the screen's left
    (POST, LEFT, dict(view_from="Anterior", roll=-90, left="S", right="I", up="L")),
])
def test_describe_camera_standard_views(view, up, expect):
    o = describe_camera(view, up)
    for key, value in expect.items():
        got = getattr(o, key)
        if isinstance(value, str):
            assert got == value, (key, got)
        else:
            assert abs(((got - value + 180) % 360) - 180) < 1e-6, (key, got)


def test_oblique_view_letters_and_angles():
    # 30° from anterior toward the left, 20° above the axial plane
    az, el = np.radians(30), np.radians(20)
    pos_ras = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    look_ras = -pos_ras
    view_scene = (-look_ras[2], -look_ras[1], -look_ras[0])
    o = describe_camera(view_scene, SUP)
    assert o.view_from == "Anterior"
    assert abs(o.azimuth - 30) < 1e-6 and abs(o.elevation - 20) < 1e-6 and abs(o.roll) < 1e-6
    assert abs(o.off_axis - np.degrees(np.arccos(np.cos(az) * np.cos(el)))) < 1e-6
    assert direction_letters([-0.8, 0.6, 0]) == "LA"
