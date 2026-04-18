"""
Image resampling and intensity clipping utilities.

All resampling is done via SimpleITK for correct physical-space handling.
Clipping is done via nibabel to avoid a round-trip through SimpleITK.
"""

import os
from typing import Optional

import numpy as np
import nibabel as nib
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_image_to_reference(
    input_path: str,
    reference_path: str,
    out_path: Optional[str] = None,
    interpolator=sitk.sitkBSpline,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample input image to match the grid (origin, spacing, size, direction)
    of reference image.

    Suitable for continuous-valued images (CT, PET/SUV).

    Args:
        input_path: Path to image to resample.
        reference_path: Path to reference image whose grid is adopted.
        out_path: If provided, write the result to this path.
        interpolator: SimpleITK interpolator constant (default: BSpline).
        default_value: Fill value for voxels outside the input FOV.

    Returns:
        Resampled SimpleITK image.
    """
    input_img = sitk.ReadImage(input_path)
    ref_img   = sitk.ReadImage(reference_path)

    filt = sitk.ResampleImageFilter()
    filt.SetOutputOrigin(ref_img.GetOrigin())
    filt.SetOutputSpacing(ref_img.GetSpacing())
    filt.SetSize(ref_img.GetSize())
    filt.SetOutputDirection(ref_img.GetDirection())
    filt.SetInterpolator(interpolator)
    filt.SetDefaultPixelValue(default_value)

    resampled = filt.Execute(input_img)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        # SimpleITK uses LPS internally and flips axes when writing NIfTI (LPS→RAS),
        # which causes a data mirror vs nibabel-written NIfTI files. Instead, extract
        # the pixel array and write with nibabel using the reference image's affine so
        # axis ordering stays consistent with the rest of the pipeline.
        arr = sitk.GetArrayFromImage(resampled)   # SimpleITK returns (Z, Y, X)
        arr_xyz = arr.transpose(2, 1, 0)          # → (X, Y, Z) nibabel convention
        ref_nib = nib.load(reference_path)
        nib.save(nib.Nifti1Image(arr_xyz, ref_nib.affine), out_path)

    return resampled


def downsample_ct_to_pet(
    ct_path: str,
    pet_path: str,
    out_path: str,
    default_value: float = -1024.0,
) -> sitk.Image:
    """
    Downsample CT to match PET resolution.
    Uses BSpline interpolation; fills outside FOV with -1024 HU.
    """
    return resample_image_to_reference(
        input_path=ct_path,
        reference_path=pet_path,
        out_path=out_path,
        interpolator=sitk.sitkBSpline,
        default_value=default_value,
    )


def upsample_pet_to_ct(
    pet_path: str,
    ct_path: str,
    out_path: str,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Upsample PET/SUV to match CT resolution.
    Uses BSpline interpolation; fills outside FOV with 0.
    """
    return resample_image_to_reference(
        input_path=pet_path,
        reference_path=ct_path,
        out_path=out_path,
        interpolator=sitk.sitkBSpline,
        default_value=default_value,
    )


def resample_seg_to_reference(
    seg_path: str,
    reference_path: str,
    out_path: Optional[str] = None,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample a segmentation mask to match a reference image grid.
    Uses NearestNeighbor interpolation to preserve integer labels.

    Args:
        seg_path: Path to segmentation .nii.gz file.
        reference_path: Reference image whose grid is adopted.
        out_path: If provided, write the result to this path.
        default_value: Fill value outside FOV (typically 0).

    Returns:
        Resampled SimpleITK image.
    """
    return resample_image_to_reference(
        input_path=seg_path,
        reference_path=reference_path,
        out_path=out_path,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=default_value,
    )


# ---------------------------------------------------------------------------
# Intensity clipping (in-place on file)
# ---------------------------------------------------------------------------

def clip_ct(nii_path: str, low: float = -1024.0, high: float = 2048.0) -> None:
    """
    Clip CT HU values and cast to int16, then overwrite the file.

    Args:
        nii_path: Path to a CT .nii.gz file.
        low: Minimum HU value (below → clamped to low).
        high: Maximum HU value (above → clamped to high).
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    data = np.clip(data, low, high).astype(np.int16)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), nii_path)


def clip_suv(nii_path: str, low: float = 0.0, high: float = 100.0) -> None:
    """
    Clip SUV values to a physiologically meaningful range, then overwrite.

    Args:
        nii_path: Path to a SUV .nii.gz file.
        low: Minimum SUV (negative values are unphysical).
        high: Maximum SUV (outlier suppression; 100 is very generous).
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    data = np.clip(data, low, high).astype(np.float32)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), nii_path)
