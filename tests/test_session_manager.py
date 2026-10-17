"""SessionManager: orientation round trip, DICOM sessions and delete (throw-away DB).

Run: .venv/bin/pytest tests/test_session_manager.py -q
"""

import nibabel as nib
import numpy as np
import pytest

from tests.test_dicom_loader import study  # noqa: F401  (pytest fixture)
from tests.test_orientation import las_phantom, reoriented_copy


@pytest.fixture
def sm(tmp_path, monkeypatch):
    import src.database.db as dbm
    from src.core.config import settings

    dbm.init_db(tmp_path / "test.db")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    from src.core.session_manager import SessionManager
    return SessionManager()


def test_lps_nifti_session_saves_mask_in_source_orientation(sm, tmp_path):
    base = las_phantom()
    ct_lps = reoriented_copy(base, "LPS")
    ct_path = tmp_path / "pt_ct.nii.gz"
    pet_path = tmp_path / "pt_pet.nii.gz"
    nib.save(ct_lps, ct_path)
    nib.save(ct_lps, pet_path)

    sm.create_session("", "", ct_path=ct_path, pet_path=pet_path)
    assert nib.aff2axcodes(sm.ct_image.affine) == ("L", "A", "S")
    assert sm.get_ct_data().dtype == np.float32
    assert sm.ct_stream_path is None                     # reoriented → not streamable
    np.testing.assert_array_equal(sm.get_ct_data(), np.asarray(base.dataobj))

    sm.ensure_roi_mask()
    mask = sm.get_tumor_mask_data()
    mask[2:5, 3:6, 4:7] = 1                              # edits are in place
    sm.save_session()

    seg_path = tmp_path / "pt_ct_Segmentation.nii.gz"
    saved = nib.load(seg_path)
    assert saved.get_data_dtype() == np.uint8
    np.testing.assert_allclose(saved.affine, ct_lps.affine, atol=1e-6)
    expected = reoriented_copy(nib.Nifti1Image(mask.copy(), sm.ct_image.affine), "LPS")
    np.testing.assert_array_equal(np.asarray(saved.dataobj), np.asarray(expected.dataobj))

    # Reload: the mask comes back in the app orientation, identical to the edit.
    sid = sm.current_session_id
    sm.close_session()
    sm.load_session(sid)
    np.testing.assert_array_equal(sm.get_tumor_mask_data(), mask)


def test_grid_mismatch_is_rejected(sm, tmp_path):
    base = las_phantom()
    nib.save(base, tmp_path / "ct.nii.gz")
    small = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), base.affine)
    nib.save(small, tmp_path / "pet.nii.gz")
    with pytest.raises(ValueError, match="same voxel grid"):
        sm.create_session("d", "p", ct_path=tmp_path / "ct.nii.gz", pet_path=tmp_path / "pet.nii.gz")
    assert sm.get_all_sessions() == []                   # no half-created DB row


def test_delete_keeps_files(sm, tmp_path):
    from src.core.file_manager import FileManager

    base = las_phantom()
    ct_path = tmp_path / "ct.nii.gz"
    nib.save(base, ct_path)
    keep = sm.create_session("d", "keep", ct_path=ct_path)
    sid = sm.create_session("d", "gone", ct_path=ct_path)
    sm.ensure_roi_mask()
    sm.save_session()
    seg = tmp_path / "ct_Segmentation.nii.gz"
    legacy_dir = FileManager.get_session_dir(sid)        # app-owned folder (e.g. reports)
    (legacy_dir / "report").mkdir()

    assert sm.delete_session(sid)
    assert sm.current_session_id is None and sm.ct_image is None
    assert [s.id for s in sm.get_all_sessions()] == [keep]
    assert not legacy_dir.exists()
    assert seg.exists() and ct_path.exists()             # source + segmentation kept
    assert not FileManager.get_session_dir(keep, create=False).exists()  # no side-effect mkdir


def test_dicom_session_roundtrip(sm, study):  # noqa: F811
    from src.core.dicom_loader import auto_pick, scan_folder

    ct, pet = auto_pick(scan_folder(str(study)))
    sid = sm.create_session_from_dicom(ct, pet, "ct", "SMITH ANNA", "DOE JOHN")
    ct_arr = sm.get_ct_data().copy()
    pet_arr = sm.get_pet_data().copy()
    assert ct_arr.dtype == np.float32 and pet_arr.dtype == np.float32
    assert ct_arr.shape == pet_arr.shape
    assert sm.ct_stream_path is None and not sm.suv_info.estimated

    sm.ensure_roi_mask()
    sm.get_tumor_mask_data()[3:6, 2:5, 1:4] = 1
    mask = sm.get_tumor_mask_data().copy()
    sm.save_session()
    seg = sm.repository.get_by_id(sid).tumor_seg_path
    assert seg.endswith("ct_Segmentation.nii.gz")
    assert nib.aff2axcodes(nib.load(seg).affine) == ("L", "P", "S")   # native DICOM order

    sm.close_session()
    sm.load_session(sid)
    assert sm.patient_name == "DOE JOHN" and sm.doctor_name == "SMITH ANNA"
    np.testing.assert_array_equal(sm.get_ct_data(), ct_arr)
    np.testing.assert_array_equal(sm.get_pet_data(), pet_arr)
    np.testing.assert_array_equal(sm.get_tumor_mask_data(), mask)

    # Re-importing the same series reuses the saved segmentation.
    sm.create_session_from_dicom(ct, pet, "ct", "SMITH ANNA", "DOE JOHN")
    np.testing.assert_array_equal(sm.get_tumor_mask_data(), mask)


def test_lesion_stats_follow_load_edit_save(sm, tmp_path):
    base = las_phantom()
    ct_path = tmp_path / "les_ct.nii.gz"
    pet_path = tmp_path / "les_pet.nii.gz"
    nib.save(base, ct_path)
    nib.save(base, pet_path)
    seg = np.zeros(base.shape, dtype=np.uint8)
    seg[2:5, 3:6, 4:7] = 1
    seg_path = tmp_path / "les_seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, base.affine), seg_path)

    # Load (what DataLoaderWorker does): values for the loaded mask.
    sm.create_session("", "", ct_path=ct_path, pet_path=pet_path, tumor_seg_path=seg_path)
    sm.update_lesion_stats()
    assert len(sm.lesion_stats["lesions"]) == 1 and not sm.lesion_stats_stale

    # Edit: old values kept but flagged; save + recompute (SaveWorker) refreshes them.
    mask = sm.get_tumor_mask_data()
    mask[10:12, 10:12, 1:3] = 1
    sm.invalidate_lesion_stats()
    assert sm.lesion_stats_stale and len(sm.lesion_stats["lesions"]) == 1
    sm.save_session()
    sm.update_lesion_stats()
    assert len(sm.lesion_stats["lesions"]) == 2 and not sm.lesion_stats_stale

    # A newly loaded segmentation replaces the values (seg_path now holds the
    # saved 2-lesion edit, so load a fresh 1-lesion file).
    seg2_path = tmp_path / "les_seg2.nii.gz"
    nib.save(nib.Nifti1Image(seg, base.affine), seg2_path)
    sm.update_current_session(tumor_seg_path=seg2_path)
    assert sm.lesion_stats is None
    sm.update_lesion_stats()
    assert len(sm.lesion_stats["lesions"]) == 1


def test_new_empty_mask_is_not_dirty_and_not_saved(sm, tmp_path):
    base = las_phantom()
    ct_path = tmp_path / "e_ct.nii.gz"
    nib.save(base, ct_path)
    sm.create_session("", "", ct_path=ct_path)
    sm.ensure_roi_mask()
    assert sm.tumor_mask is not None and not sm.tumor_dirty
    assert not (tmp_path / "e_ct_Segmentation.nii.gz").exists()


# ── CT/PET on different grids (NIfTI resampling) ──────────────────────────────

def _world_ramp(img):
    """Fill ``img``'s grid with a smooth function of the world position (mm)."""
    ijk = np.indices(img.shape, dtype=float).reshape(3, -1)
    xyz = img.affine[:3, :3] @ ijk + img.affine[:3, 3:4]
    vals = 50.0 + 0.5 * xyz[0] + 1.0 * xyz[1] + 1.5 * xyz[2]
    return nib.Nifti1Image(vals.reshape(img.shape).astype(np.float32), img.affine)


def _ct_pet_pair(tmp_path, pet_codes="LPS"):
    """Fine CT (1 mm, LAS) and coarse PET (2 mm, stored ``pet_codes``) over one body."""
    ct_aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    ct_aff[:3, 3] = (20.0, -15.0, 10.0)
    ct = _world_ramp(nib.Nifti1Image(np.zeros((40, 32, 24), np.float32), ct_aff))
    pet_aff = np.diag([-2.0, 2.0, 2.0, 1.0])
    pet_aff[:3, 3] = (20.5, -14.5, 10.5)            # PET voxel centres between CT ones
    pet = _world_ramp(nib.Nifti1Image(np.zeros((20, 16, 12), np.float32), pet_aff))
    ct_path, pet_path = tmp_path / "pair_ct.nii.gz", tmp_path / "pair_pet.nii.gz"
    nib.save(ct, ct_path)
    nib.save(reoriented_copy(pet, pet_codes), pet_path)
    return ct, pet, ct_path, pet_path


def _interior(a, margin=4):
    return a[margin:-margin, margin:-margin, margin:-margin]


def test_las_grid_is_header_only_and_matches_load(tmp_path):
    from src.utils.nifti_utils import las_grid, load_nifti_volume

    base = las_phantom()
    for codes in ("LAS", "LPS", "RPI", "ASL"):
        path = tmp_path / f"g_{codes}.nii.gz"
        nib.save(reoriented_copy(base, codes), path)
        shape, affine = las_grid(path)
        img, _, _ = load_nifti_volume(path)
        assert shape == img.shape
        np.testing.assert_allclose(affine, img.affine, atol=1e-6)


def test_grids_match_tolerance():
    from src.utils.nifti_utils import grids_match

    aff = np.diag([-2.0, 2.0, 3.0, 1.0])
    near, far = aff.copy(), aff.copy()
    near[:3, 3] += 0.1                                  # 0.05 voxel
    far[:3, 3] += 1.0                                   # 0.5 voxel
    assert grids_match((10, 10, 10), aff, (10, 10, 10), near)
    assert not grids_match((10, 10, 10), aff, (10, 10, 10), far)
    assert not grids_match((10, 10, 10), aff, (10, 10, 11), aff)


def test_pet_resampled_onto_ct_grid_and_restored_on_reload(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    sm.create_session("d", "p", ct_path=ct_path)

    mismatch = sm.check_grid_mismatch(pet_path=pet_path)
    assert mismatch.ct.shape == (40, 32, 24) and mismatch.pet.shape == (20, 16, 12)
    assert not mismatch.same_shape and mismatch.closes_mask == {}

    sm.update_current_session(pet_path=pet_path, resample_mode="ct")
    pet_arr = sm.get_pet_data()
    assert pet_arr.shape == ct.shape and pet_arr.dtype == np.float32
    assert pet_arr.flags.f_contiguous and pet_arr.min() >= 0.0
    np.testing.assert_array_equal(sm.pet_image.affine, sm.ct_image.affine)
    # BSpline boundary effects fade within a few PET voxels of the border.
    np.testing.assert_allclose(_interior(pet_arr, 8), _interior(np.asarray(ct.dataobj), 8), atol=0.05)
    assert sm.pet_stream_path is None and sm.resample_mode == "ct"
    assert sm.check_grid_mismatch(pet_path=pet_path) is not None      # the PET file still differs

    sid = sm.current_session_id
    assert sm.repository.get_by_id(sid).resample_mode == "ct"
    expected = pet_arr.copy()
    sm.close_session()
    sm.load_session(sid)
    np.testing.assert_array_equal(sm.get_pet_data(), expected)


def test_ct_resampled_onto_pet_grid_saves_mask_next_to_pet(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path, pet_codes="LPS")
    sm.create_session("d", "p", ct_path=ct_path, pet_path=pet_path, resample_mode="pet")
    ct_arr = sm.get_ct_data()
    assert ct_arr.shape == pet.shape and ct_arr.flags.f_contiguous
    np.testing.assert_allclose(_interior(ct_arr, 2), _interior(np.asarray(pet.dataobj), 2), atol=0.05)

    sm.ensure_roi_mask()
    sm.get_tumor_mask_data()[5:8, 4:7, 3:6] = 1
    mask = sm.get_tumor_mask_data().copy()
    sm.save_session()
    seg_path = tmp_path / "pair_pet_Segmentation.nii.gz"
    saved = nib.load(seg_path)
    assert nib.aff2axcodes(saved.affine) == ("L", "P", "S")          # PET's native order
    np.testing.assert_allclose(saved.affine, nib.load(pet_path).affine, atol=1e-6)

    sid = sm.current_session_id
    sm.close_session()
    sm.load_session(sid)
    assert sm.resample_mode == "pet"
    np.testing.assert_array_equal(sm.get_tumor_mask_data(), mask)


def test_resampled_image_is_reread_from_its_source(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    sm.create_session("d", "p", ct_path=ct_path, pet_path=pet_path, resample_mode="pet")
    assert sm.get_ct_data().shape == pet.shape                         # CT on the PET grid

    # A new PET already on the CT grid: compared with the CT *file*, not the
    # resampled copy, so there is nothing to resolve and CT comes back unchanged.
    pet_on_ct = tmp_path / "pet_on_ct.nii.gz"
    nib.save(ct, pet_on_ct)
    assert sm.check_grid_mismatch(pet_path=pet_on_ct) is None
    sm.update_current_session(pet_path=pet_on_ct)
    np.testing.assert_array_equal(sm.get_ct_data(), np.asarray(ct.dataobj))
    assert sm.resample_mode is None and sm.ct_stream_path is not None
    assert sm.repository.get_by_id(sm.current_session_id).resample_mode is None


def test_grid_change_closes_mask_and_keeps_its_file(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    sm.create_session("d", "p", ct_path=ct_path)
    sm.ensure_roi_mask()
    sm.get_tumor_mask_data()[2:4, 2:4, 2:4] = 1
    sm.save_session()
    ct_seg = tmp_path / "pair_ct_Segmentation.nii.gz"
    assert ct_seg.exists()

    mismatch = sm.check_grid_mismatch(pet_path=pet_path)
    assert mismatch.closes_mask == {"ct": False, "pet": True, "none": False}
    sm.update_current_session(pet_path=pet_path, resample_mode="pet")
    assert sm.tumor_mask is None and not sm.tumor_dirty
    assert not sm.repository.get_by_id(sm.current_session_id).tumor_seg_path
    assert ct_seg.exists()

    sm.ensure_roi_mask()
    sm.save_session()
    assert (tmp_path / "pair_pet_Segmentation.nii.gz").exists()
    assert nib.load(ct_seg).shape == ct.shape                          # old file untouched


def test_same_shape_offset_grid_loads_with_warning(sm, tmp_path):
    base = las_phantom()
    shifted = base.affine.copy()
    shifted[:3, 3] += 5.0
    nib.save(base, tmp_path / "s_ct.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(base.dataobj), shifted), tmp_path / "s_pet.nii.gz")
    sm.create_session("d", "p", ct_path=tmp_path / "s_ct.nii.gz")
    mismatch = sm.check_grid_mismatch(pet_path=tmp_path / "s_pet.nii.gz")
    assert mismatch is not None and mismatch.same_shape
    sm.update_current_session(pet_path=tmp_path / "s_pet.nii.gz")
    assert sm.resample_mode is None and sm.load_warnings


def test_segmentation_on_wrong_grid_does_not_block_reload(sm, tmp_path):
    base = las_phantom()
    ct_path = tmp_path / "w_ct.nii.gz"
    nib.save(base, ct_path)
    wrong = tmp_path / "w_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), base.affine), wrong)
    sid = sm.create_session("d", "p", ct_path=ct_path)
    sm.repository.update(sid, tumor_seg_path=str(wrong))
    sm.close_session()
    sm.load_session(sid)
    assert sm.ct_image is not None and sm.tumor_mask is None
    assert any("Segmentation not loaded" in w for w in sm.load_warnings)


# ── Segmentation on another grid (nearest-neighbour onto the CT grid) ─────────

def _pet_grid_seg(tmp_path, pet, name="seg_on_pet.nii.gz", codes="LPS", centre_mm=None):
    """A box lesion drawn on the PET grid, stored with voxel axes ``codes``."""
    data = np.zeros(pet.shape, np.uint8)
    data[8:12, 6:10, 4:8] = 1                     # PET voxels → world box
    img = nib.Nifti1Image(data, pet.affine)
    path = tmp_path / name
    nib.save(reoriented_copy(img, codes), path)
    return img, path


def _world_of(img, ijk):
    return img.affine[:3, :3] @ np.asarray(ijk, float) + img.affine[:3, 3]


def test_segmentation_on_pet_grid_is_resampled_and_saved_next_to_ct(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    seg_img, seg_path = _pet_grid_seg(tmp_path, pet)
    source_bytes = seg_path.read_bytes()
    sm.create_session("d", "p", ct_path=ct_path)
    sm.update_current_session(tumor_seg_path=seg_path)

    mask = sm.get_tumor_mask_data()
    assert mask.shape == ct.shape and mask.dtype == np.uint8 and mask.flags.f_contiguous
    assert not sm.tumor_dirty
    # The lesion lands at the same place in the world.
    inv = np.linalg.inv(ct.affine)
    inside = np.rint(inv @ np.append(_world_of(pet, (9.5, 7.5, 5.5)), 1))[:3].astype(int)
    outside = np.rint(inv @ np.append(_world_of(pet, (3, 3, 2)), 1))[:3].astype(int)
    assert mask[tuple(inside)] == 1 and mask[tuple(outside)] == 0
    np.testing.assert_allclose(mask.sum() * 1.0, seg_img.get_fdata().sum() * 8.0, rtol=0.15)  # 2 mm → 1 mm voxels

    saved = tmp_path / "pair_ct_Segmentation.nii.gz"
    assert sm.repository.get_by_id(sm.current_session_id).tumor_seg_path == str(saved)
    on_disk = nib.load(saved)
    assert on_disk.shape == ct.shape
    np.testing.assert_allclose(on_disk.affine, nib.load(ct_path).affine, atol=1e-6)
    assert seg_path.read_bytes() == source_bytes                       # source untouched

    sid = sm.current_session_id
    sm.close_session()
    sm.load_session(sid)
    np.testing.assert_array_equal(sm.get_tumor_mask_data(), mask)


def test_resampled_segmentation_overwrites_ct_segmentation_but_not_its_source(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    target = tmp_path / "pair_ct_Segmentation.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(ct.shape, np.uint8), ct.affine), target)   # earlier AI result
    _, seg_path = _pet_grid_seg(tmp_path, pet)
    sm.create_session("d", "p", ct_path=ct_path)
    sm.update_current_session(tumor_seg_path=seg_path)
    assert np.asarray(nib.load(target).dataobj).any()                              # overwritten

    # Loading a wrong-grid file that sits at the target path keeps the original.
    _, same_name = _pet_grid_seg(tmp_path, pet, name="pair_ct_Segmentation.nii.gz")
    original = same_name.read_bytes()
    sm.update_current_session(tumor_seg_path=same_name)
    written = tmp_path / "pair_ct_Segmentation_resampled.nii.gz"
    assert same_name.read_bytes() == original
    assert sm.repository.get_by_id(sm.current_session_id).tumor_seg_path == str(written)
    assert nib.load(written).shape == ct.shape


def test_segmentation_outside_the_image_is_rejected(sm, tmp_path):
    ct, pet, ct_path, pet_path = _ct_pet_pair(tmp_path)
    far = pet.affine.copy()
    far[:3, 3] += 5000.0
    data = np.zeros(pet.shape, np.uint8)
    data[8:12, 6:10, 4:8] = 1
    nib.save(nib.Nifti1Image(data, far), tmp_path / "far.nii.gz")
    sm.create_session("d", "p", ct_path=ct_path)
    with pytest.raises(ValueError, match="does not overlap"):
        sm.update_current_session(tumor_seg_path=tmp_path / "far.nii.gz")
    assert sm.tumor_mask is None
    assert not sm.repository.get_by_id(sm.current_session_id).tumor_seg_path
