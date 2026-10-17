"""Put CT and PET on one voxel grid (SimpleITK) — used by both the DICOM import
(``dicom_loader.load_study``) and NIfTI loading (``SessionManager``).

One policy for both paths:

* **PET onto the CT grid** (upsampling): interpolating cubic BSpline. The
  caller clamps the BSpline undershoot of PET to 0.
* **CT onto the PET grid** (downsampling): Gaussian anti-aliasing matched to
  the spacing change, then linear interpolation. An interpolating BSpline here
  would first need a B-spline decomposition of the whole CT, which ITK runs on
  one thread (~34 s for 512×512×600 against ~1 s), and it aliases when the
  grid shrinks ~4×.
* **Segmentation masks** onto the session grid: nearest neighbour, labels
  kept as they are.
"""
import nibabel as nib
import numpy as np

_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def grid_of(img) -> tuple:
    """``(size, origin, spacing, direction)`` of a SimpleITK image."""
    return img.GetSize(), img.GetOrigin(), img.GetSpacing(), img.GetDirection()


def resample_to_grid(moving, grid: tuple, fill: float, method: str = "bspline"):
    """``moving`` (SimpleITK) resampled onto ``grid``.

    ``method``:
      * "bspline"   — interpolating cubic BSpline, float32 output.
      * "antialias" — smooth ``moving`` along the axes whose spacing grows (the
        Gaussian that turns the input voxel size into the output one), then
        linear interpolation; float32 output.
      * "nearest"   — nearest neighbour in ``moving``'s pixel type (label masks).
    Voxels outside ``moving``'s field of view get ``fill``.
    """
    import SimpleITK as sitk

    size, origin, spacing, direction = grid
    pixel_type = sitk.sitkFloat32
    if method == "nearest":
        interpolator = sitk.sitkNearestNeighbor
        pixel_type = moving.GetPixelID()
    elif method == "antialias":
        variance = [max(out * out - src * src, 0.0) / _FWHM_PER_SIGMA ** 2
                    for out, src in zip(spacing, moving.GetSpacing())]
        if any(v > 0 for v in variance):
            moving = sitk.DiscreteGaussian(moving, variance, 64, 0.01, True)
        interpolator = sitk.sitkLinear
    elif method == "bspline":
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError(f"Unknown resampling method {method!r}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(n) for n in size])
    resampler.SetOutputOrigin(tuple(origin))
    resampler.SetOutputSpacing(tuple(spacing))
    resampler.SetOutputDirection(tuple(direction))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(fill))
    resampler.SetOutputPixelType(pixel_type)
    return resampler.Execute(moving)


# ── nibabel (RAS affine, XYZ voxels) ──────────────────────────────────────────

def _nib_grid(affine: np.ndarray, shape) -> tuple:
    """SimpleITK grid (LPS) of a nibabel image grid (RAS affine)."""
    affine = np.asarray(affine, dtype=float)
    linear = _RAS_TO_LPS @ affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing
    origin = _RAS_TO_LPS @ affine[:3, 3]
    return (tuple(int(n) for n in shape[:3]), tuple(origin), tuple(spacing),
            tuple(direction.flatten()))


def resample_onto(img: nib.Nifti1Image, ref: nib.Nifti1Image, fill: float,
                  clamp_negative: bool = False, method: str = "bspline") -> nib.Nifti1Image:
    """``img`` resampled onto the voxel grid of ``ref`` (see ``resample_to_grid``).

    The result is Fortran-ordered XYZ (float32, or ``img``'s dtype for
    "nearest") and carries ``ref.affine`` unchanged, so it lines up
    voxel-for-voxel with ``ref``.
    """
    import SimpleITK as sitk

    # XYZ (Fortran) → ZYX (C-contiguous) view; SimpleITK copies it once.
    moving = sitk.GetImageFromArray(np.asanyarray(img.dataobj).T)
    _, origin, spacing, direction = _nib_grid(img.affine, img.shape)
    moving.SetOrigin(origin)
    moving.SetSpacing(spacing)
    moving.SetDirection(direction)

    out = resample_to_grid(moving, _nib_grid(ref.affine, ref.shape), fill, method)
    data = sitk.GetArrayFromImage(out).transpose(2, 1, 0)
    if clamp_negative:
        np.maximum(data, 0.0, out=data)
    return nib.Nifti1Image(data, ref.affine)
