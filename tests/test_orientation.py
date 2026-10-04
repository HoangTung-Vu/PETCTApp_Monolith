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
