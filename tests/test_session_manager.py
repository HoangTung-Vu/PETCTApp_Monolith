"""SessionManager: orientation round trip, DICOM sessions and delete (throw-away DB).

Run: .venv/bin/pytest tests/test_session_manager.py -q
"""

import nibabel as nib
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.test_dicom_loader import study  # noqa: F401  (pytest fixture)
from tests.test_orientation import las_phantom, reoriented_copy


@pytest.fixture
def sm(tmp_path, monkeypatch):
    import src.database.db as dbm
    import src.database.session_repository as repo
    from src.core.config import settings

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(dbm, "engine", engine)
    dbm.init_db()
    monkeypatch.setattr(repo, "SessionLocal", sessionmaker(bind=engine))
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
