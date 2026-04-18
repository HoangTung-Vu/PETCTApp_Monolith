"""
SUV (Standardized Uptake Value) reconstruction utilities.

SUV = (raw_counts × weight_grams) / injected_dose_decayed
"""

import datetime
import traceback
from typing import Tuple

import numpy as np
import nibabel as nib
import pydicom

# Fallback constants used when DICOM tags are missing
_DEFAULT_WEIGHT_G = 75_000          # 75 kg
_DEFAULT_DOSE_BQ  = 420_000_000     # 420 MBq
_DEFAULT_HALF_LIFE_S = 6_588.0      # F-18 half-life in seconds
_DEFAULT_WAIT_S   = (1.75 + 0.25) * 3600  # 90 min wait + 15 min prep


def compute_suv(
    pet_raw: np.ndarray, first_dcm_path: str
) -> Tuple[np.ndarray, bool]:
    """
    Convert raw PET pixel values to SUV.

    Args:
        pet_raw: 3-D numpy array of raw PET counts (after RescaleSlope applied).
        first_dcm_path: Path to any .dcm file from the same PET series
                        (used to read radiopharmaceutical metadata).

    Returns:
        (suv_array, estimated) where estimated=True means fallback values were used.
    """
    ds = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
    estimated = False

    # Patient weight
    try:
        weight_g = float(ds.PatientWeight) * 1000
    except Exception:
        weight_g = _DEFAULT_WEIGHT_G
        estimated = True

    # Dose and decay
    try:
        scan_time = datetime.datetime.strptime(
            ds.AcquisitionTime, "%H%M%S.%f"
        )
        radio_seq = ds.RadiopharmaceuticalInformationSequence[0]
        inj_time = datetime.datetime.strptime(
            radio_seq.RadiopharmaceuticalStartTime, "%H%M%S.%f"
        )
        half_life = float(radio_seq.RadionuclideHalfLife)
        total_dose = float(radio_seq.RadionuclideTotalDose)
        elapsed_s = (scan_time - inj_time).seconds
        decay = np.exp(-np.log(2) * elapsed_s / half_life)
        injected_dose_decay = total_dose * decay
    except Exception:
        traceback.print_exc()
        decay = np.exp(-np.log(2) * _DEFAULT_WAIT_S / _DEFAULT_HALF_LIFE_S)
        injected_dose_decay = _DEFAULT_DOSE_BQ * decay
        estimated = True

    suv = pet_raw * weight_g / injected_dose_decay
    return suv, estimated


def reconstruct_suv_nifti(
    pet_nii_path: str,
    first_dcm_path: str,
    out_path: str,
) -> Tuple[str, bool]:
    """
    Load an existing PET NIfTI, compute SUV, and save the result.

    Args:
        pet_nii_path: Path to the raw PET .nii.gz file.
        first_dcm_path: Path to any .dcm from the matching PET series.
        out_path: Output SUV .nii.gz path.

    Returns:
        (out_path, estimated) — estimated=True if fallback metadata was used.
    """
    import os

    pet_nii = nib.load(pet_nii_path)
    pet_raw = pet_nii.get_fdata(dtype=np.float32)

    suv, estimated = compute_suv(pet_raw, first_dcm_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(suv, pet_nii.affine, pet_nii.header), out_path)
    return out_path, estimated
