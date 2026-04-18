"""
DICOM series → NIfTI conversion utilities.

Uses dicom2nifti for standard series and a manual nibabel path
for PET (to preserve raw pixel values before SUV reconstruction).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import pydicom

from .dicom_utils import sort_by_instance_number, get_dcm_files


def dicom_series_to_nifti(dcm_dir: str, out_path: str) -> str:
    """
    Convert a CT (or any non-PET) DICOM series to a NIfTI file.

    Uses dicom2nifti under the hood which handles slice ordering,
    orientation, and spacing correctly.

    Args:
        dcm_dir: Directory containing .dcm files for ONE series.
        out_path: Output .nii.gz file path (parent dirs created automatically).

    Returns:
        Absolute path to the written NIfTI file.
    """
    try:
        import dicom2nifti
    except ImportError as e:
        raise ImportError(
            "dicom2nifti is required for CT conversion. "
            "Install with: pip install dicom2nifti"
        ) from e

    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # dicom2nifti writes into a directory; we then rename to the desired path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dicom2nifti.convert_directory(dcm_dir, tmp, compression=True, reorient=True)
        tmp_files = list(Path(tmp).glob("*.nii.gz"))
        if not tmp_files:
            raise RuntimeError(f"dicom2nifti produced no output for {dcm_dir}")
        tmp_files[0].rename(out_path)

    return out_path


def dicom_pet_series_to_nifti(dcm_dir: str, out_path: str) -> str:
    """
    Convert a PET DICOM series to NIfTI preserving raw pixel values
    (RescaleSlope / RescaleIntercept applied, but no SUV normalization).

    Slices are sorted by InstanceNumber and stacked along the Z axis.
    Affine is built from ImagePositionPatient / ImageOrientationPatient / PixelSpacing.

    Args:
        dcm_dir: Directory containing .dcm files for ONE PET series.
        out_path: Output .nii.gz file path.

    Returns:
        Absolute path to the written NIfTI file.
    """
    dcm_files = get_dcm_files(dcm_dir)
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {dcm_dir}")

    dcm_files = sort_by_instance_number(dcm_files)
    slices = [pydicom.dcmread(f) for f in dcm_files]

    # Build 3-D array with physical scaling applied
    def _apply_scale(ds: pydicom.Dataset) -> np.ndarray:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return ds.pixel_array.astype(np.float32) * slope + intercept

    volume = np.stack([_apply_scale(s) for s in slices], axis=-1)  # (H, W, Z)

    # Build affine from DICOM geometry tags
    affine = _build_affine(slices)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_affine(slices: List[pydicom.Dataset]) -> np.ndarray:
    """
    Construct a 4×4 RAS affine from the DICOM geometry tags of a sorted slice list.

    Follows the standard DICOM → NIfTI affine construction described in
    https://nipy.org/nibabel/dicom/dicom_nifti_1.html
    """
    ds0 = slices[0]
    ds_last = slices[-1]

    iop = np.array(ds0.ImageOrientationPatient, dtype=float)  # 6 values
    row_dir = iop[:3]
    col_dir = iop[3:]

    # Pixel spacing: [row spacing, col spacing]
    ps = ds0.PixelSpacing
    dr, dc = float(ps[0]), float(ps[1])

    # Origin
    ipp0 = np.array(ds0.ImagePositionPatient, dtype=float)

    if len(slices) > 1:
        ipp_last = np.array(ds_last.ImagePositionPatient, dtype=float)
        slice_dir = (ipp_last - ipp0) / (len(slices) - 1)
    else:
        slice_dir = np.cross(row_dir, col_dir)
        dz = float(getattr(ds0, "SliceThickness", 1.0))
        slice_dir = slice_dir * dz

    affine = np.eye(4)
    affine[:3, 0] = row_dir * dc
    affine[:3, 1] = col_dir * dr
    affine[:3, 2] = slice_dir
    affine[:3, 3] = ipp0

    return affine
