"""DICOM import on a synthetic study written by pydicom (no real patient data).

Run: .venv/bin/pytest tests/test_dicom_loader.py -q
"""

import random
from datetime import datetime

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.core.dicom_loader import (
    auto_pick, compute_suv_factor, find_series, load_study, names_from_header, scan_folder,
)

CT_SOP = "1.2.840.10008.5.1.4.1.1.2"
PET_SOP = "1.2.840.10008.5.1.4.1.1.128"

# CT grid (LPS): 20 cols × 16 rows × 12 slices
CT_COLS, CT_ROWS, CT_SLICES = 20, 16, 12
CT_PS = (1.5, 1.2)                  # (row spacing, column spacing)
CT_DZ = 3.0
CT_ORIGIN = np.array([-12.0, -10.0, -20.0])

# PET grid: 10 × 8 × 6 at twice the spacing, same origin
PET_COLS, PET_ROWS, PET_SLICES = 10, 8, 6
PET_PS = (3.0, 2.4)
PET_DZ = 6.0

WEIGHT_KG, DOSE_BQ, HALF_LIFE = 70.0, 370e6, 6586.2


def ct_hu(i, j, k):
    return 10.0 * i + 3.0 * j + 100.0 * k - 500.0


def pet_slope(k):
    return 0.5 + 0.25 * k


def _base(sop_class, study_uid, series_uid, for_uid, modality, desc, number, **names):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = sop_class
    ds.Modality = modality
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = for_uid
    ds.SeriesDescription = desc
    ds.SeriesNumber = number
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.PatientName = names.get("patient", "DOE^JOHN")
    ds.PatientID = "PID123"
    if names.get("referring") is not None:
        ds.ReferringPhysicianName = names["referring"]
    if names.get("performing") is not None:
        ds.PerformingPhysicianName = names["performing"]
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20240501"
    ds.SeriesTime = ds.AcquisitionTime = "110000.00"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    return ds


def write_series(folder, sop, modality, desc, number, cols, rows, slices, ps, dz,
                 pixel_fn, study_uid, for_uid, extra=None, image_type=None, names=None):
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    order = list(range(slices))
    random.Random(number).shuffle(order)                    # file order ≠ slice order
    for n, k in enumerate(order):
        ds = _base(sop, study_uid, series_uid, for_uid, modality, desc, number, **(names or {}))
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.InstanceNumber = slices - k                       # reversed vs position
        ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = list(ps)
        ds.SliceThickness = dz
        ds.ImagePositionPatient = list(CT_ORIGIN + [0, 0, k * dz])
        stored, slope, intercept, signed = pixel_fn(k)
        ds.PixelRepresentation = 1 if signed else 0
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.PixelData = stored.tobytes()
        for key, val in (extra or {}).items():
            setattr(ds, key, val)
        ds.save_as(folder / f"IMG{n:04d}", enforce_file_format=True)   # no .dcm extension
    return series_uid


def radiopharm(start_time="100000.00"):
    item = Dataset()
    item.RadionuclideTotalDose = DOSE_BQ
    item.RadionuclideHalfLife = HALF_LIFE
    item.RadiopharmaceuticalStartTime = start_time
    return [item]


def ct_pixels(k):
    j, i = np.mgrid[0:CT_ROWS, 0:CT_COLS]
    hu = ct_hu(i, j, k)
    return (hu + 1024).astype(np.int16), 1.0, -1024.0, True


def pet_pixels(k):
    raw = np.full((PET_ROWS, PET_COLS), 1000, dtype=np.uint16)
    raw[3:5, 4:6] = 4000
    return raw, pet_slope(k), 0.0, False


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    study_uid, for_uid = generate_uid(), generate_uid()
    names = {"referring": "SMITH^ANNA"}
    write_series(root / "a" / "ct", CT_SOP, "CT", "CT WB 3mm", 2, CT_COLS, CT_ROWS, CT_SLICES,
                 CT_PS, CT_DZ, ct_pixels, study_uid, for_uid, names=names)
    write_series(root / "a" / "scout", CT_SOP, "CT", "Topogram", 1, CT_COLS, CT_ROWS, 2,
                 CT_PS, CT_DZ, ct_pixels, study_uid, for_uid, names=names,
                 image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"])
    pet_extra = {"Units": "BQML", "DecayCorrection": "START", "PatientWeight": WEIGHT_KG,
                 "RadiopharmaceuticalInformationSequence": radiopharm()}
    write_series(root / "b" / "pet_ac", PET_SOP, "PT", "PET WB AC", 3, PET_COLS, PET_ROWS,
                 PET_SLICES, PET_PS, PET_DZ, pet_pixels, study_uid, for_uid, names=names,
                 extra={**pet_extra, "CorrectedImage": ["ATTN", "DECY"]})
    write_series(root / "b" / "pet_nac", PET_SOP, "PT", "PET WB NAC", 4, PET_COLS, PET_ROWS,
                 PET_SLICES, PET_PS, PET_DZ, pet_pixels, study_uid, for_uid, names=names,
                 extra={**pet_extra, "CorrectedImage": ["DECY"]})
    (root / "notes.txt").write_text("not dicom")
    return root


def expected_factor():
    decay = 2.0 ** (-3600.0 / HALF_LIFE)
    return WEIGHT_KG * 1000.0 / (DOSE_BQ * decay)


def lps_to_ras(p):
    return np.array([-p[0], -p[1], p[2]])


def test_scan_and_auto_pick(study):
    series = scan_folder(str(study))
    labels = sorted((s.modality, s.description) for s in series)
    assert labels == [("CT", "CT WB 3mm"), ("PT", "PET WB AC"), ("PT", "PET WB NAC")]
    ct, pet = auto_pick(series)
    assert ct.description == "CT WB 3mm" and pet.description == "PET WB AC"
    assert ct.n_slices == CT_SLICES and abs(ct.slice_spacing - CT_DZ) < 1e-6
    assert find_series(ct.directory, ct.uid).uid == ct.uid


def test_ct_values_and_geometry(study):
    ct, _ = auto_pick(scan_folder(str(study)))
    res = load_study(ct, None)
    img = res.ct
    data = np.asarray(img.dataobj)
    assert data.dtype == np.float32
    assert nib.aff2axcodes(img.affine) == ("L", "A", "S")
    assert data.shape == (CT_COLS, CT_ROWS, CT_SLICES)
    inv = np.linalg.inv(img.affine)
    for i, j, k in [(0, 0, 0), (7, 3, 5), (CT_COLS - 1, CT_ROWS - 1, CT_SLICES - 1)]:
        lps = CT_ORIGIN + [i * CT_PS[1], j * CT_PS[0], k * CT_DZ]
        vox = np.rint(inv @ np.append(lps_to_ras(lps), 1.0))[:3].astype(int)
        assert data[tuple(vox)] == pytest.approx(ct_hu(i, j, k))
    # Masks are saved back in the native (LPS) voxel order.
    native = img.as_reoriented(res.back_ornt)
    assert nib.aff2axcodes(native.affine) == ("L", "P", "S")


def test_pet_per_slice_rescale_and_suv(study):
    _, pet = auto_pick(scan_folder(str(study)))
    res = load_study(None, pet)
    assert not res.suv.estimated
    assert res.suv.factor == pytest.approx(expected_factor(), rel=1e-9)
    data = np.asarray(res.pet.dataobj)
    inv = np.linalg.inv(res.pet.affine)
    for k in range(PET_SLICES):
        lps = CT_ORIGIN + [0, 0, k * PET_DZ]
        vox = np.rint(inv @ np.append(lps_to_ras(lps), 1.0))[:3].astype(int)
        assert data[tuple(vox)] == pytest.approx(1000 * pet_slope(k) * expected_factor(), rel=1e-5)


def test_resample_onto_ct_and_pet_grids(study):
    ct, pet = auto_pick(scan_folder(str(study)))
    on_ct = load_study(ct, pet, "ct")
    assert on_ct.pet.shape == on_ct.ct.shape
    np.testing.assert_allclose(on_ct.pet.affine, on_ct.ct.affine)
    assert np.asarray(on_ct.pet.dataobj).min() >= 0.0
    on_pet = load_study(ct, pet, "pet")
    assert on_pet.ct.shape == (PET_COLS, PET_ROWS, PET_SLICES)
    np.testing.assert_allclose(on_pet.ct.affine, on_pet.pet.affine)


def test_names_fallback_chain():
    ds = Dataset()
    ds.PatientName = "DOE^JOHN"
    ds.ReferringPhysicianName = "SMITH^ANNA"
    ds.PerformingPhysicianName = "LEE^MAI"
    assert names_from_header(ds) == ("SMITH ANNA", "DOE JOHN")
    del ds.ReferringPhysicianName
    assert names_from_header(ds)[0] == "LEE MAI"
    del ds.PerformingPhysicianName
    ds.PatientName = ""
    ds.PatientID = "PID9"
    assert names_from_header(ds, "Dr Manual", "Manual") == ("Dr Manual", "PID9")
    del ds.PatientID
    assert names_from_header(ds, "", "", "folder") == ("", "folder")


def test_suv_estimated_without_weight_and_midnight(study):
    _, pet = auto_pick(scan_folder(str(study)))
    ds = pet.header
    weight = ds.PatientWeight
    try:
        del ds.PatientWeight
        assert compute_suv_factor(pet).estimated
    finally:
        ds.PatientWeight = weight
    # Injection 23:30, series 00:30 the next day, no StartDateTime → 1 h uptake.
    item = ds.RadiopharmaceuticalInformationSequence[0]
    old = (item.RadiopharmaceuticalStartTime, ds.SeriesTime, pet.earliest_acquisition)
    try:
        item.RadiopharmaceuticalStartTime = "233000"
        ds.SeriesTime = "003000"
        pet.earliest_acquisition = datetime(2024, 5, 1, 0, 30)
        info = compute_suv_factor(pet)
        assert not info.estimated
        assert info.factor == pytest.approx(expected_factor(), rel=1e-9)
    finally:
        item.RadiopharmaceuticalStartTime, ds.SeriesTime, pet.earliest_acquisition = old
