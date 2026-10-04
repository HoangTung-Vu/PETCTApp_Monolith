"""Read PET/CT DICOM series straight into memory — no NIfTI conversion.

Headers are parsed with pydicom (``stop_before_pixels``). Pixel data and
resampling go through SimpleITK/GDCM, which also decodes compressed transfer
syntaxes. The slice order and geometry come from the headers, so the result
does not depend on file names. Volumes come out as in-memory nibabel images in
the app's LAS orientation (see ``nifti_utils.to_app_orientation``): CT in
float32 HU, PET in float32 SUV (body weight).
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import pydicom
from pydicom.valuerep import PersonName

from ..utils.nifti_utils import to_app_orientation

# Tags read from every file while scanning (pixel data is never read here).
_PHILIPS_SUV_SCALE = 0x70531000          # Philips private: SUV scale factor
_PHILIPS_ACTIVITY_SCALE = 0x70531009     # Philips private: counts → Bq/ml
_HEADER_TAGS = [
    "SpecificCharacterSet", "SOPClassUID", "Modality", "ImageType",
    "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID",
    "SeriesNumber", "SeriesDescription", "InstanceNumber", "NumberOfFrames",
    "StudyDate", "SeriesDate", "SeriesTime", "AcquisitionDate", "AcquisitionTime",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns",
    "PatientName", "PatientID", "PatientWeight",
    "ReferringPhysicianName", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "Units", "DecayCorrection", "CorrectedImage", "RadiopharmaceuticalInformationSequence",
    _PHILIPS_SUV_SCALE, _PHILIPS_ACTIVITY_SCALE,
]

# Fallbacks when SUV metadata is missing (same values as the old converter).
_DEFAULT_WEIGHT_KG = 75.0
_DEFAULT_DOSE_BQ = 420e6
_DEFAULT_HALF_LIFE_S = 6588.0            # F-18
_DEFAULT_UPTAKE_S = 2 * 3600.0

_NAC_RE = re.compile(r"\b(NAC|NOAC|NON[- ]?AC|NO[- ]AC|UNCORR\w*)\b")


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SeriesInfo:
    """One usable CT or PET series found while scanning a folder."""
    uid: str
    modality: str                         # "CT" or "PT"
    description: str
    number: Optional[int]
    files: list                           # sorted along the slice normal
    directory: str                        # common parent of ``files``
    study_uid: str
    frame_of_reference_uid: str
    rows: int
    cols: int
    pixel_spacing: tuple                  # (row spacing, column spacing) mm
    slice_spacing: float
    origin: np.ndarray                    # ImagePositionPatient of the first slice (LPS)
    direction: np.ndarray                 # 3x3 LPS columns: row dir, column dir, slice normal
    date: str
    header: pydicom.Dataset               # first slice — names and SUV tags
    earliest_acquisition: Optional[datetime] = None
    warnings: list = field(default_factory=list)

    @property
    def n_slices(self) -> int:
        return len(self.files)

    @property
    def is_attenuation_corrected(self) -> bool:
        corrected = self.header.get("CorrectedImage", "")
        values = list(corrected) if isinstance(corrected, (list, tuple, pydicom.multival.MultiValue)) else [corrected]
        return any("ATTN" in str(v).upper() for v in values) and not _NAC_RE.search(self.description.upper())

    def label(self) -> str:
        mod = "PET" if self.modality == "PT" else self.modality
        desc = self.description or "(no description)"
        sp = f"{self.pixel_spacing[1]:.2f}×{self.pixel_spacing[0]:.2f}×{self.slice_spacing:.2f} mm"
        date = format_dicom_date(self.date)
        return f"{mod} · {desc} · {self.n_slices} slices · {self.cols}×{self.rows} · {sp}" + (f" · {date}" if date else "")


@dataclass
class SuvInfo:
    factor: float
    estimated: bool
    notes: list


@dataclass
class DicomStudy:
    """Loaded volumes on a shared grid, in the app's LAS orientation."""
    ct: Optional[nib.Nifti1Image]
    pet: Optional[nib.Nifti1Image]
    back_ornt: np.ndarray                 # LAS → native voxel order of the reference grid
    suv: Optional[SuvInfo]
    warnings: list


# ── Small parsers ────────────────────────────────────────────────────────────

def format_dicom_date(da: str) -> str:
    da = (da or "").strip()
    return f"{da[:4]}-{da[4:6]}-{da[6:8]}" if len(da) >= 8 and da[:8].isdigit() else ""


def _parse_da(da) -> Optional[datetime]:
    da = str(da or "").strip()
    if len(da) >= 8 and da[:8].isdigit():
        return datetime(int(da[:4]), int(da[4:6]), int(da[6:8]))
    return None


def _parse_tm(tm) -> Optional[timedelta]:
    """DICOM TM ("HHMMSS.FFFFFF", any trailing part optional; legacy "HH:MM:SS")."""
    tm = str(tm or "").strip().replace(":", "")
    if not tm:
        return None
    frac = 0.0
    if "." in tm:
        tm, f = tm.split(".", 1)
        frac = float("0." + f) if f.isdigit() else 0.0
    if not tm.isdigit() or len(tm) < 2:
        return None
    hh = int(tm[0:2])
    mm = int(tm[2:4]) if len(tm) >= 4 else 0
    ss = int(tm[4:6]) if len(tm) >= 6 else 0
    return timedelta(hours=hh, minutes=mm, seconds=ss + frac)


def _parse_dt(dt) -> Optional[datetime]:
    """DICOM DT ("YYYYMMDDHHMMSS.FFFFFF&ZZXX"); the UTC offset is ignored."""
    dt = re.split(r"[+-]", str(dt or "").strip(), maxsplit=1)[0]
    day = _parse_da(dt[:8])
    if day is None:
        return None
    tod = _parse_tm(dt[8:]) if len(dt) > 8 else timedelta(0)
    return day + (tod or timedelta(0))


def _combine(da, tm) -> Optional[datetime]:
    day, tod = _parse_da(da), _parse_tm(tm)
    return day + tod if day is not None and tod is not None else None


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii", "ignore").strip("\x00 ")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def format_person_name(value) -> str:
    """PN → "FAMILY GIVEN MIDDLE" (first value if multi-valued)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        for v in value:
            name = format_person_name(v)
            if name:
                return name
        return ""
    pn = value if isinstance(value, PersonName) else PersonName(str(value))
    parts = [pn.family_name, pn.given_name, pn.middle_name]
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return text or str(pn).replace("^", " ").strip()


def names_from_header(ds: pydicom.Dataset, manual_doctor: str = "",
                      manual_patient: str = "", fallback: str = "") -> tuple:
    """Return ``(doctor, patient)`` from DICOM tags with manual fallbacks.

    Physician: ReferringPhysicianName (0008,0090) → PerformingPhysicianName
    (0008,1050) → NameOfPhysiciansReadingStudy (0008,1060) → manual field.
    Patient: PatientName (0010,0010) → PatientID (0010,0020) → manual field
    → ``fallback`` (the folder name).
    """
    doctor = ""
    for kw in ("ReferringPhysicianName", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy"):
        doctor = format_person_name(ds.get(kw))
        if doctor:
            break
    patient = format_person_name(ds.get("PatientName")) or str(ds.get("PatientID", "") or "").strip()
    return (doctor or manual_doctor.strip(), patient or manual_patient.strip() or fallback)


# ── Scanning ─────────────────────────────────────────────────────────────────

def _read_header(path: str) -> Optional[pydicom.Dataset]:
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_HEADER_TAGS)
    except Exception:
        return None


def _iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.upper() == "DICOMDIR":
                continue
            yield os.path.join(dirpath, name)


def scan_folder(root: str, log: Optional[Callable[[str], None]] = None,
                only_uid: Optional[str] = None) -> list:
    """Find the usable CT and PET series under ``root`` (recursive).

    Every readable DICOM file counts, whatever its extension. Localizers,
    multi-frame and dynamic series are skipped with a log line.
    """
    log = log or (lambda msg: None)
    paths = list(_iter_files(root))
    log(f"Reading headers of {len(paths)} file(s)…")
    with ThreadPoolExecutor(max_workers=8) as pool:
        headers = list(pool.map(_read_header, paths))

    groups: dict = {}
    for path, ds in zip(paths, headers):
        if ds is None or "SeriesInstanceUID" not in ds:
            continue
        modality = str(ds.get("Modality", "")).upper()
        if modality not in ("CT", "PT"):
            continue
        uid = str(ds.SeriesInstanceUID)
        if only_uid is not None and uid != only_uid:
            continue
        groups.setdefault(uid, []).append((path, ds))

    series = []
    for uid, items in groups.items():
        info = _build_series(uid, items, log)
        if info is not None:
            series.append(info)
    series.sort(key=lambda s: (s.modality, s.number if s.number is not None else 1 << 30, s.description))
    return series


def _build_series(uid: str, items: list, log) -> Optional[SeriesInfo]:
    ds0 = items[0][1]
    desc = str(ds0.get("SeriesDescription", "") or "").strip()
    name = f"{ds0.Modality} series '{desc or uid}'"

    image_type = [str(v).upper() for v in (ds0.get("ImageType") or [])]
    if "LOCALIZER" in image_type:
        log(f"  Skipping {name}: localizer/scout.")
        return None
    if int(ds0.get("NumberOfFrames", 1) or 1) > 1:
        log(f"  Skipping {name}: multi-frame images are not supported.")
        return None

    try:
        iop = np.array([float(v) for v in ds0.ImageOrientationPatient], dtype=float)
        ps = tuple(float(v) for v in ds0.PixelSpacing)
        rows, cols = int(ds0.Rows), int(ds0.Columns)
    except Exception:
        log(f"  Skipping {name}: missing geometry tags.")
        return None

    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    positions = []
    for path, ds in items:
        try:
            if not np.allclose([float(v) for v in ds.ImageOrientationPatient], iop, atol=1e-3):
                log(f"  Skipping {name}: slices have different orientations.")
                return None
            if int(ds.Rows) != rows or int(ds.Columns) != cols:
                log(f"  Skipping {name}: slices have different sizes.")
                return None
            ipp = np.array([float(v) for v in ds.ImagePositionPatient], dtype=float)
        except Exception:
            log(f"  Skipping {name}: a slice has no position.")
            return None
        positions.append((float(ipp @ normal), path, ipp, ds))

    if len(positions) < 2:
        log(f"  Skipping {name}: fewer than 2 slices.")
        return None
    positions.sort(key=lambda t: t[0])
    dists = np.diff([p[0] for p in positions])
    if np.any(dists < 1e-3):
        log(f"  Skipping {name}: several images share a slice position (dynamic/multi-phase series).")
        return None

    warnings = []
    spacing = float(np.median(dists))
    if np.max(np.abs(dists - spacing)) > max(0.01 * spacing, 1e-3):
        warnings.append(
            f"{name}: non-uniform slice spacing ({dists.min():.2f}–{dists.max():.2f} mm); "
            "missing slices would shift the volume."
        )

    acq = [_combine(ds.get("AcquisitionDate") or ds.get("SeriesDate"), ds.get("AcquisitionTime"))
           for _, _, _, ds in positions]
    acq = [a for a in acq if a is not None]

    files = [p[1] for p in positions]
    number = ds0.get("SeriesNumber")
    return SeriesInfo(
        uid=uid,
        modality=str(ds0.Modality).upper(),
        description=desc,
        number=int(number) if number not in (None, "") else None,
        files=files,
        directory=os.path.commonpath([os.path.dirname(f) for f in files]),
        study_uid=str(ds0.get("StudyInstanceUID", "")),
        frame_of_reference_uid=str(ds0.get("FrameOfReferenceUID", "")),
        rows=rows,
        cols=cols,
        pixel_spacing=ps,
        slice_spacing=spacing,
        origin=positions[0][2],
        direction=np.column_stack([row_dir, col_dir, normal]),
        date=str(ds0.get("SeriesDate") or ds0.get("StudyDate") or ""),
        header=positions[0][3],
        earliest_acquisition=min(acq) if acq else None,
        warnings=warnings,
    )


def auto_pick(series: list) -> tuple:
    """Preselect ``(ct, pet)``: AC PET with most slices, then the CT on its frame of reference."""
    pets = [s for s in series if s.modality == "PT"]
    cts = [s for s in series if s.modality == "CT"]
    pet = min(pets, key=lambda s: (not s.is_attenuation_corrected, -s.n_slices), default=None)
    if pet is not None:
        ct = min(cts, key=lambda s: (s.frame_of_reference_uid != pet.frame_of_reference_uid,
                                     s.study_uid != pet.study_uid, -s.n_slices), default=None)
    else:
        ct = min(cts, key=lambda s: -s.n_slices, default=None)
    return ct, pet


# ── SUV ──────────────────────────────────────────────────────────────────────

def compute_suv_factor(pet: SeriesInfo) -> SuvInfo:
    """Factor turning PET pixel values (after RescaleSlope/Intercept) into SUVbw.

    Follows the QIBA vendor-neutral SUV recipe:
      Units (0054,1001) BQML → weight[g] / decayed dose[Bq];  GML → already SUV;
      CNTS → Philips private SUV scale factor (7053,1000).
      Dose (0018,1074), half-life (0018,1075) and injection time (0018,1078 or
      0018,1072) come from RadiopharmaceuticalInformationSequence (0054,0016).
      Decay reference for DecayCorrection (0054,1102) START = SeriesDate/Time,
      or the earliest AcquisitionDate/Time when the series time is later
      (post-processed series); ADMIN = no decay.
    Missing values fall back to defaults and set ``estimated``.
    """
    ds = pet.header
    notes = []
    estimated = False

    units = str(ds.get("Units", "") or "").upper()
    if units == "GML":
        return SuvInfo(1.0, False, ["Units GML: pixel values are already SUV."])
    if units == "CNTS":
        scale = _to_float(ds.get(_PHILIPS_SUV_SCALE).value if _PHILIPS_SUV_SCALE in ds else None)
        if scale:
            return SuvInfo(scale, False, ["Units CNTS: Philips SUV scale factor (7053,1000) used."])
        activity = _to_float(ds.get(_PHILIPS_ACTIVITY_SCALE).value if _PHILIPS_ACTIVITY_SCALE in ds else None)
        if activity:
            notes.append("Units CNTS: converted with Philips activity scale factor (7053,1009).")
        else:
            activity = 1.0
            estimated = True
            notes.append("Units CNTS without a Philips scale factor: SUV is not reliable.")
    else:
        activity = 1.0
        if units != "BQML":
            notes.append(f"Units '{units or 'missing'}': assumed BQML.")

    weight = _to_float(ds.get("PatientWeight"))
    if not weight or weight <= 0:
        weight = _DEFAULT_WEIGHT_KG
        estimated = True
        notes.append(f"PatientWeight missing: assumed {_DEFAULT_WEIGHT_KG:g} kg.")

    seq = ds.get("RadiopharmaceuticalInformationSequence")
    item = seq[0] if seq else pydicom.Dataset()
    dose = _to_float(item.get("RadionuclideTotalDose"))
    if not dose or dose <= 0:
        dose = _DEFAULT_DOSE_BQ
        estimated = True
        notes.append(f"RadionuclideTotalDose missing: assumed {_DEFAULT_DOSE_BQ / 1e6:g} MBq.")
    half_life = _to_float(item.get("RadionuclideHalfLife"))
    if not half_life or half_life <= 0:
        half_life = _DEFAULT_HALF_LIFE_S
        estimated = True
        notes.append("RadionuclideHalfLife missing: assumed F-18 (6588 s).")

    decay_corr = str(ds.get("DecayCorrection", "START") or "START").upper()
    if decay_corr == "ADMIN":
        decay = 1.0
    else:
        if decay_corr == "NONE":
            estimated = True
            notes.append("DecayCorrection NONE: images are not decay corrected; SUV is approximate.")
        series_dt = _combine(ds.get("SeriesDate") or ds.get("StudyDate"), ds.get("SeriesTime"))
        scan = series_dt
        if scan is None or (pet.earliest_acquisition is not None and scan > pet.earliest_acquisition):
            scan = pet.earliest_acquisition
        injection = _parse_dt(item.get("RadiopharmaceuticalStartDateTime"))
        if injection is None and scan is not None:
            tod = _parse_tm(item.get("RadiopharmaceuticalStartTime"))
            if tod is not None:
                injection = datetime(scan.year, scan.month, scan.day) + tod
                if injection > scan:                       # injected the day before (midnight)
                    injection -= timedelta(days=1)
        uptake = (scan - injection).total_seconds() if scan and injection else None
        if uptake is None or not (0 <= uptake <= 24 * 3600):
            uptake = _DEFAULT_UPTAKE_S
            estimated = True
            notes.append("Injection or scan time missing/invalid: assumed 2 h uptake.")
        decay = 2.0 ** (-uptake / half_life)

    factor = activity * weight * 1000.0 / (dose * decay)
    return SuvInfo(float(factor), estimated, notes)


# ── Loading ──────────────────────────────────────────────────────────────────

def _read_volume(series: SeriesInfo):
    """Read a series as a float32 SimpleITK image with header-derived geometry.

    SimpleITK applies each slice's RescaleSlope/Intercept (PET slices differ).
    """
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.files)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    img = reader.Execute()
    img.SetOrigin(tuple(float(v) for v in series.origin))
    img.SetSpacing((series.pixel_spacing[1], series.pixel_spacing[0], series.slice_spacing))
    img.SetDirection(tuple(float(v) for v in series.direction.flatten()))
    return img


def _to_nibabel(img) -> tuple:
    """SimpleITK (LPS) image → in-memory nibabel image in LAS + back orientation."""
    import SimpleITK as sitk

    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(img.GetSpacing(), dtype=float)
    affine[:3, 3] = img.GetOrigin()
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine              # LPS → RAS
    data = sitk.GetArrayFromImage(img).transpose(2, 1, 0)           # (z,y,x) → (x,y,z) view
    return to_app_orientation(nib.Nifti1Image(data, affine))


def load_study(ct: Optional[SeriesInfo], pet: Optional[SeriesInfo], resample_mode: str = "ct",
               log: Optional[Callable[[str], None]] = None) -> DicomStudy:
    """Load the chosen series, put them on one grid and convert PET to SUV.

    ``resample_mode`` "ct" resamples PET onto the CT grid (BSpline, negatives
    clamped to 0); "pet" resamples CT onto the PET grid (BSpline, −1024 HU
    outside). No clipping or integer casting is applied.
    """
    import SimpleITK as sitk

    log = log or (lambda msg: None)
    if ct is None and pet is None:
        raise ValueError("Select at least one CT or PET series.")
    warnings = list((ct.warnings if ct else []) + (pet.warnings if pet else []))

    ct_img = pet_img = None
    if ct is not None:
        log(f"Reading CT ({ct.n_slices} slices)…")
        ct_img = _read_volume(ct)
    if pet is not None:
        log(f"Reading PET ({pet.n_slices} slices)…")
        pet_img = _read_volume(pet)

    resampled_pet = False
    if ct_img is not None and pet_img is not None:
        if ct.frame_of_reference_uid and ct.frame_of_reference_uid != pet.frame_of_reference_uid:
            warnings.append("CT and PET have different FrameOfReferenceUID — check the alignment.")
        if resample_mode == "pet":
            log("Resampling CT onto the PET grid…")
            ct_img = sitk.Resample(ct_img, pet_img, sitk.Transform(), sitk.sitkBSpline,
                                   -1024.0, sitk.sitkFloat32)
        else:
            log("Resampling PET onto the CT grid…")
            pet_img = sitk.Resample(pet_img, ct_img, sitk.Transform(), sitk.sitkBSpline,
                                    0.0, sitk.sitkFloat32)
            resampled_pet = True

    suv = None
    ct_nib = pet_nib = None
    back_ornt = None
    if ct_img is not None:
        ct_nib, back_ornt = _to_nibabel(ct_img)
    if pet_img is not None:
        suv = compute_suv_factor(pet)
        pet_nib, pet_back = _to_nibabel(pet_img)
        arr = np.asarray(pet_nib.dataobj)
        arr *= np.float32(suv.factor)
        if resampled_pet:
            np.maximum(arr, 0.0, out=arr)                  # BSpline undershoot
        if back_ornt is None:
            back_ornt = pet_back
        log(f"SUV factor {suv.factor:.6g}" + (" (estimated)" if suv.estimated else ""))

    return DicomStudy(ct=ct_nib, pet=pet_nib, back_ornt=back_ornt, suv=suv, warnings=warnings)


def find_series(directory: str, uid: Optional[str]) -> Optional[SeriesInfo]:
    """Re-locate a stored series by its SeriesInstanceUID (session reload)."""
    if not directory or not uid or not os.path.isdir(directory):
        return None
    found = scan_folder(directory, only_uid=uid)
    return found[0] if found else None

