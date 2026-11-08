"""Lesion ids: binary vs labelled masks, Give lesion ids, merge / overwrite,
int16 storage and the organ-layer ordering fix.

Run: .venv/bin/pytest tests/test_lesion_ids.py -q
"""

import nibabel as nib
import numpy as np
import pytest

from src.core import lesion_labels as LL
from src.core.engine.report_engine import ReportEngine
from tests.test_session_manager import sm  # noqa: F401  (pytest fixture)

SHAPE = (40, 36, 30)                                   # X, Y, Z (LAS: larger z = head)
AFFINE = np.diag([-2.0, 2.0, 3.0, 1.0])


def three_blobs(dtype=np.uint8, order="F"):
    """Binary mask with three lesions; the one at the largest z is the head-most."""
    m = np.zeros(SHAPE, dtype=dtype, order=order)
    m[5:9, 5:9, 2:5] = 1          # feet
    m[20:25, 10:14, 24:28] = 1    # head
    m[30:33, 20:24, 12:15] = 1    # middle
    return m


def pet_for(mask):
    rng = np.random.default_rng(1)
    return np.asfortranarray(rng.random(mask.shape, dtype=np.float32) * 5 + 1)


# ── Give lesion ids ⇔ report ids ─────────────────────────────────────────────

@pytest.mark.parametrize("order", ["F", "C"])
def test_give_ids_equal_binary_report_ids(order):
    mask = three_blobs(order=order)
    pet = pet_for(mask)
    ids, n = LL.give_lesion_ids(mask)
    assert n == 3 and ids.dtype == np.int16 and ids.flags.f_contiguous
    binary = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    labelled = ReportEngine.compute_lesion_stats(pet, ids, AFFINE, labeled=True)
    assert [l["id"] for l in binary["lesions"]] == [1, 2, 3]
    for b, l in zip(binary["lesions"], labelled["lesions"]):
        for key in ("id", "bbox", "n_voxels", "SUVmax", "SUVmean", "suvmax_voxel"):
            assert b[key] == l[key], key
    # head → feet: id 1 is the blob at z 24..27, id 3 the one at z 2..4
    assert ids[22, 12, 25] == 1 and ids[31, 21, 13] == 2 and ids[6, 6, 3] == 3
    assert LL.present_labels(ids) == [1, 2, 3]


def test_give_ids_starts_from_scratch():
    """A labelled mask (with a merge and odd ids) is binarised and renumbered."""
    m = three_blobs(dtype=np.int16)
    m[m > 0] = 7                                   # all three merged into id 7
    m[30:33, 20:24, 12:15] = 300                   # one with a large id
    ids, n = LL.give_lesion_ids(m)
    assert n == 3 and LL.present_labels(ids) == [1, 2, 3]
    assert ids[22, 12, 25] == 1 and ids[31, 21, 13] == 2


def test_empty_mask_gives_no_ids():
    ids, n = LL.give_lesion_ids(np.zeros(SHAPE, np.uint8, order="F"))
    assert n == 0 and not ids.any()


# ── Labelled stats ───────────────────────────────────────────────────────────

def test_labelled_stats_use_mask_values_and_merged_components():
    m = three_blobs(dtype=np.int16)
    m[20:25, 10:14, 24:28] = 4          # head blob → id 4
    m[5:9, 5:9, 2:5] = 4                # feet blob merged into lesion 4
    m[30:33, 20:24, 12:15] = 9          # gap in the ids
    stats = ReportEngine.compute_lesion_stats(pet_for(m), m, AFFINE, labeled=True)
    assert stats["labeled"] is True
    assert [l["id"] for l in stats["lesions"]] == [4, 9]
    les4 = stats["lesions"][0]
    assert les4["n_voxels"] == 5 * 4 * 4 + 4 * 4 * 3
    assert les4["mask"].sum() == les4["n_voxels"]
    # The same mask read as binary splits lesion 4 again.
    binary = ReportEngine.compute_lesion_stats(pet_for(m), m, AFFINE)
    assert len(binary["lesions"]) == 3


# ── Editing helpers ──────────────────────────────────────────────────────────

def test_merge_roi_binary_equals_maximum():
    tumor = three_blobs(dtype=np.int16)
    roi = np.zeros(SHAPE, np.uint8, order="F")
    roi[18:27, 8:16, 20:29] = 1
    expected = np.maximum(tumor, roi)
    out = LL.merge_roi(np.copy(tumor), roi, 1)
    np.testing.assert_array_equal(out, expected)


def test_merge_roi_keeps_or_merges_other_lesions():
    tumor, _ = LL.give_lesion_ids(three_blobs())
    roi = np.zeros(SHAPE, np.uint8, order="F")
    roi[19:34, 9:25, 11:29] = 1          # covers lesion 1 and lesion 2
    assert LL.roi_conflicts(tumor, roi, 5) == [1, 2]

    kept = LL.merge_roi(np.copy(tumor), roi, 5)
    assert kept[22, 12, 25] == 1 and kept[31, 21, 13] == 2      # lesions keep their ids
    assert kept[19, 9, 11] == 5                                   # background → 5
    assert kept[6, 6, 3] == 3

    merged = LL.merge_roi(np.copy(tumor), roi, 5, merge_ids=(1,))
    assert merged[22, 12, 25] == 5 and merged[31, 21, 13] == 2
    assert not (merged == 1).any()


def test_resolve_overwrite_yes_and_no():
    data = np.zeros((10, 10, 10), np.int16)
    data[0:3, 0:3, 0:3] = 2                       # lesion 2
    data[6:9, 6:9, 6:9] = 2                       # a second part of lesion 2
    painted = (np.array([1, 1, 4]), np.array([1, 2, 4]), np.array([1, 2, 4]))
    old = data[painted].copy()                    # [2, 2, 0]
    atoms = [(painted, old, 5)]
    assert LL.overwritten_labels(atoms, 5) == [2]

    no = data.copy()
    no[painted] = 5
    LL.resolve_overwrite(no, atoms, 5, merge=False)
    assert no[1, 1, 1] == 2 and no[1, 2, 2] == 2 and no[4, 4, 4] == 5   # background painted
    assert (no == 2).sum() == 54

    yes = data.copy()
    yes[painted] = 5
    LL.resolve_overwrite(yes, atoms, 5, merge=True)
    assert not (yes == 2).any() and (yes == 5).sum() == 55              # both parts merged


def test_overwrite_ignores_own_label_and_background():
    painted = (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    assert LL.overwritten_labels([(painted, np.array([0, 3]), 3)], 3) == []


# ── Storage ──────────────────────────────────────────────────────────────────

def _session_with_mask(sm, tmp_path, seg_data, seg_dtype):
    ct = nib.Nifti1Image(np.zeros(SHAPE, np.float32), AFFINE)
    ct_path = tmp_path / "pt_ct.nii.gz"
    nib.save(ct, ct_path)
    seg = nib.Nifti1Image(np.asarray(seg_data).astype(seg_dtype), AFFINE)
    seg.set_data_dtype(seg_dtype)
    seg_path = tmp_path / "seg_in.nii.gz"
    nib.save(seg, seg_path)
    sm.create_session("d", "p", ct_path=ct_path, tumor_seg_path=seg_path)
    return seg_path


def test_ids_above_255_survive_load_and_save(sm, tmp_path):
    m = three_blobs(dtype=np.int16)
    m[20:25, 10:14, 24:28] = 300
    m[30:33, 20:24, 12:15] = 256
    seg_path = _session_with_mask(sm, tmp_path, m, np.int16)
    assert sm.tumor_labeled
    assert sm.get_tumor_mask_data().dtype == np.int16
    sm.refresh_label_ids()
    assert sm.tumor_label_ids == [1, 256, 300]
    sm.tumor_dirty = True
    sm.save_session()
    saved = nib.load(seg_path)
    assert saved.get_data_dtype() == np.int16
    assert LL.has_lesion_ids_marker(saved.header)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), m)


def test_binary_mask_stays_binary_uint8(sm, tmp_path):
    seg_path = _session_with_mask(sm, tmp_path, three_blobs(), np.uint8)
    assert not sm.tumor_labeled
    sm.save_session()
    saved = nib.load(seg_path)
    assert saved.get_data_dtype() == np.uint8
    assert not LL.has_lesion_ids_marker(saved.header)


def test_int16_zero_one_file_is_binary(sm, tmp_path):
    """Masks from other tools are often int16 0/1 — still binary."""
    _session_with_mask(sm, tmp_path, three_blobs(), np.int16)
    assert not sm.tumor_labeled


def test_marker_keeps_single_id_mask_labelled(sm, tmp_path):
    """All lesions merged into id 1: values say binary, the marker says labelled."""
    seg_path = _session_with_mask(sm, tmp_path, three_blobs(), np.uint8)
    sm.set_tumor_mask(sm.get_tumor_mask_data(), labeled=True)
    sm.save_session()
    saved = nib.load(seg_path)
    assert saved.get_data_dtype() == np.int16 and LL.has_lesion_ids_marker(saved.header)
    sid = sm.current_session_id
    sm.load_session(sid)
    assert sm.tumor_labeled


def test_negative_or_too_large_values_are_refused(tmp_path):
    from src.utils.nifti_utils import load_mask_volume
    bad = nib.Nifti1Image(np.full(SHAPE, 40000, np.int32), AFFINE)
    path = tmp_path / "bad.nii.gz"
    nib.save(bad, path)
    with pytest.raises(ValueError, match="do not fit"):
        load_mask_volume(path)


# ── Layer order (organ-mask "Not Responding" regression) ────────────────────

def test_reorder_layers_terminates_and_orders():
    from napari.components import ViewerModel
    from src.gui.components.viewers.viewer_widget import reorder_layers

    vm = ViewerModel()
    vm.add_image(np.zeros((2, 2, 2)), name="CT Image")
    vm.add_labels(np.zeros((2, 2, 2), np.int16), name="Tumor Mask")
    vm.add_labels(np.zeros((2, 2, 2), np.uint8), name="ROI Mask")
    vm.add_labels(np.zeros((2, 2, 2), np.uint8), name="Organ Mask")   # added last, on top
    vm.add_points(name="Lesion IDs")
    reorder_layers(vm.layers, ["CT Image", "PET Image", "Organ Mask", "Tumor Mask",
                               "ROI Mask", "_shape_preview", "Lesion IDs"])
    assert [l.name for l in vm.layers] == [
        "CT Image", "Organ Mask", "Tumor Mask", "ROI Mask", "Lesion IDs"]


def test_insert_index_places_new_mask_without_moves():
    from napari.components import ViewerModel
    from src.gui.components.viewers.viewer_widget import insert_index

    order = ["CT Image", "PET Image", "Organ Mask", "Tumor Mask", "ROI Mask",
             "_shape_preview", "Lesion IDs"]
    vm = ViewerModel()
    vm.add_image(np.zeros((2, 2, 2)), name="CT Image")
    vm.add_labels(np.zeros((2, 2, 2), np.int16), name="Tumor Mask")
    vm.add_points(name="Lesion IDs")
    assert insert_index(vm.layers, "Organ Mask", order) == 1
    assert insert_index(vm.layers, "ROI Mask", order) == 2
    assert insert_index(vm.layers, "something else", order) == 3
