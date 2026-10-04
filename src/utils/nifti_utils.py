"""Utility functions for NIfTI image handling."""
import gzip
import io
from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


# ── App orientation ───────────────────────────────────────────────────────────
#
# to_napari()/from_napari() assume nibabel voxel axes pointing to patient
# Left, Anterior, Superior (LAS). That layout is what renders in radiological
# convention: axial = anterior up and patient right on screen left, coronal =
# superior up, sagittal = anterior left. Every volume is therefore reoriented
# to LAS when it enters the app, and masks are reoriented back on save.

APP_AXCODES = ("L", "A", "S")
_APP_ORNT = axcodes2ornt(APP_AXCODES)

# ── Memory layout ─────────────────────────────────────────────────────────────
#
# XYZ volumes are Fortran-ordered (x fastest), the layout nibabel and the DICOM
# loader hand out for free. Masks the app creates must use the same layout:
# numpy ops that mix F and C arrays of a whole-body volume, ``.copy()`` (C by
# default), ``np.nonzero``, boolean-mask indexing with a full-volume mask and
# scipy.ndimage label/find_objects on F input all fall off a cache cliff
# (10–30 s instead of milliseconds). Use ``np.copy``/``zeros(order="F")``,
# ``mask_bbox`` and ``np.copyto(..., where=...)`` instead.


def mask_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Tight bounding box of a 3D mask as a tuple of slices (None if empty).

    Uses per-axis ``any`` projections — fast in any memory layout, unlike
    ``np.nonzero`` on a Fortran-ordered volume.
    """
    out = []
    for axis in range(mask.ndim):
        others = tuple(i for i in range(mask.ndim) if i != axis)
        idx = np.flatnonzero(mask.any(axis=others))
        if idx.size == 0:
            return None
        out.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(out)


def to_app_orientation(img: nib.Nifti1Image) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Reorient ``img`` to the app's LAS voxel order.

    Only numpy views are created (flip/transpose), and an image already in LAS
    is returned unchanged. Returns ``(img_las, back_ornt)``; ``back_ornt`` undoes
    the reorientation (see ``from_app_orientation``).
    """
    src = io_orientation(img.affine)
    img_las = img.as_reoriented(ornt_transform(src, _APP_ORNT))
    return img_las, ornt_transform(_APP_ORNT, src)


def from_app_orientation(img_las: nib.Nifti1Image, back_ornt: Optional[np.ndarray]) -> nib.Nifti1Image:
    """Inverse of ``to_app_orientation`` — back to the source voxel order."""
    if back_ornt is None:
        return img_las
    return img_las.as_reoriented(back_ornt)


def _squeeze_3d(data: np.ndarray, path) -> np.ndarray:
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected a 3D volume, got shape {data.shape}")
    return data


def read_volume_data(img: nib.Nifti1Image) -> np.ndarray:
    """Voxel data in its stored dtype — no cast.

    The one exception is a file with scl_slope/scl_inter scaling: the scaling
    must be applied, so the result is float32 (nibabel would default to float64).
    """
    proxy = img.dataobj
    slope = float(getattr(proxy, "slope", 1.0))
    inter = float(getattr(proxy, "inter", 0.0))
    if slope != 1.0 or inter != 0.0:
        return np.asarray(proxy, dtype=np.float32)
    return np.asanyarray(proxy)


def load_nifti_volume(path) -> tuple[nib.Nifti1Image, np.ndarray, Optional[Path]]:
    """Load an intensity volume (CT/PET) into memory, reoriented to LAS.

    Returns ``(img_las, back_ornt, stream_path)``. ``img_las`` is an in-memory
    image, so later ``dataobj`` reads are free. ``stream_path`` is the source
    file when its bytes match the in-memory image exactly (already LAS, nothing
    squeezed) — the segmentation upload can then stream it unchanged.
    """
    img = nib.load(path)
    raw = read_volume_data(img)
    data = _squeeze_3d(raw, path)
    mem = img.__class__(data, img.affine, img.header)
    img_las, back_ornt = to_app_orientation(mem)
    unchanged = img_las is mem and data is raw
    return img_las, back_ornt, (Path(path) if unchanged else None)


def load_mask_volume(path, ref_shape: Optional[tuple] = None) -> nib.Nifti1Image:
    """Load a label mask as uint8, reoriented to LAS, checked against ``ref_shape``."""
    img = nib.load(path)
    data = _squeeze_3d(np.asarray(img.dataobj, dtype=np.uint8), path)
    img_las, _ = to_app_orientation(nib.Nifti1Image(data, img.affine))
    data_las = np.asarray(img_las.dataobj)
    if not data_las.flags.f_contiguous:
        # Masks are edited in place — keep them F-contiguous (see "Memory layout").
        img_las = nib.Nifti1Image(np.asfortranarray(data_las), img_las.affine)
    if ref_shape is not None and tuple(img_las.shape) != tuple(ref_shape):
        raise ValueError(
            f"Segmentation {Path(path).name} has shape {tuple(img_las.shape)} after "
            f"reorientation, but the image grid is {tuple(ref_shape)}."
        )
    return img_las


def numpy_to_nifti(
    array: np.ndarray, 
    reference_image: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Convert numpy array to NIfTI image using reference affine/header.
    
    Args:
        array: Numpy array to convert
        reference_image: Reference NIfTI image for affine and header
        
    Returns:
        NIfTI image with same spatial metadata as reference
    """
    return nib.Nifti1Image(array, reference_image.affine, reference_image.header)


def nifti_to_numpy(image: nib.Nifti1Image) -> np.ndarray:
    """Extract numpy array from NIfTI image.
    
    Args:
        image: NIfTI image
        
    Returns:
        Numpy array of image data
    """
    return np.asanyarray(image.dataobj)


def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load NIfTI file from disk.
    
    Args:
        path: Path to NIfTI file
        
    Returns:
        Loaded NIfTI image
    """
    return nib.load(path)


def save_nifti(image: nib.Nifti1Image, path: Path) -> None:
    """Save NIfTI image to disk.
    
    Args:
        image: NIfTI image to save
        path: Destination path
    """
    nib.save(image, path)


def get_slices_for_plane(data: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Get 2D slice for given plane and index.
    
    Args:
        data: 3D numpy array (X, Y, Z)
        plane: One of 'axial', 'sagittal', 'coronal'
        index: Slice index
        
    Returns:
        2D slice array
    """
    if plane == "axial":
        return data[:, :, index]
    elif plane == "sagittal":
        return data[index, :, :]
    elif plane == "coronal":
        return data[:, index, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")


def get_shape_for_plane(shape: tuple, plane: str) -> int:
    """Get number of slices for given plane.
    
    Args:
        shape: 3D shape tuple (X, Y, Z)
        plane: One of 'axial', 'sagittal', 'coronal'
        
    Returns:
        Number of slices in that plane
    """
    if plane == "axial":
        return shape[2]
    elif plane == "sagittal":
        return shape[0]
    elif plane == "coronal":
        return shape[1]
    else:
        raise ValueError(f"Unknown plane: {plane}")


def to_napari(data: np.ndarray, num_threads: Optional[int] = None,
              out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Converts Nibabel (X, Y, Z) to Napari (Z, Y, X). OPTIMIZED with multithreading.

    ``out`` (shape (Z, Y, X)) is filled in place instead of allocating — every
    voxel is overwritten, so it does not need to be zeroed.
    """
    X, Y, Z = data.shape
    res = out if out is not None else np.zeros((Z, Y, X), dtype=data.dtype)
    
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    def process_chunk(start_z, end_z):
        for z in range(start_z, end_z):
            res[Z - 1 - z, ::-1, :] = data[:, :, z].T
            
    chunk_size = max(1, Z // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            end = Z if i == num_threads - 1 else (i + 1) * chunk_size
            if start < Z:
                futures.append(executor.submit(process_chunk, start, end))
        for f in futures:
            f.result()
            
    return res


def from_napari(data_zyx: np.ndarray, num_threads: Optional[int] = None) -> np.ndarray:
    """
    Converts Napari (Z, Y, X) back to Nibabel (X, Y, Z). OPTIMIZED with multithreading.

    The result is Fortran-ordered like every XYZ volume in the app, so each
    ``res[:, :, z]`` slab written below is contiguous (≈30× faster than C order).
    """
    Z, Y, X = data_zyx.shape
    res = np.zeros((X, Y, Z), dtype=data_zyx.dtype, order="F")

    if num_threads is None:
        num_threads = os.cpu_count() or 4

    def process_chunk(start_z, end_z):
        for z in range(start_z, end_z):
            res[:, :, Z - 1 - z] = data_zyx[z, ::-1, :].T

    chunk_size = max(1, Z // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            end = Z if i == num_threads - 1 else (i + 1) * chunk_size
            if start < Z:
                futures.append(executor.submit(process_chunk, start, end))
        for f in futures:
            f.result()
            
    return res


def nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to uncompressed .nii bytes in memory."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def nifti_to_gzip_bytes(img: nib.Nifti1Image, level: int = 1,
                        num_threads: Optional[int] = None) -> bytes:
    """Serialize to .nii.gz bytes, compressing chunks in parallel.

    The chunks are concatenated as a multi-member gzip stream, which any gzip
    reader (``gzip.decompress``, nibabel) decodes as one file. zlib releases the
    GIL, so the threads compress concurrently.
    """
    raw = memoryview(nifti_to_bytes(img))
    n = num_threads or os.cpu_count() or 4
    chunk = max(1 << 22, -(-len(raw) // n))
    starts = range(0, len(raw), chunk)
    with ThreadPoolExecutor(max_workers=n) as executor:
        parts = executor.map(
            lambda i: gzip.compress(raw[i:i + chunk], compresslevel=level, mtime=0), starts
        )
        return b"".join(parts)


def bytes_to_nifti(data: bytes) -> nib.Nifti1Image:
    """Deserialize NIfTI bytes (gzipped or raw) to a nibabel Nifti1Image.

    The engine gzip-compresses the mask body (no Content-Encoding header, so httpx
    does not auto-decompress). Reading from a BytesIO via from_file_map skips
    nibabel's filename-based gzip path, so we detect the gzip magic (1f 8b) and
    decompress here — staying backward-compatible with raw NIfTI bytes.

    On failure, re-raise a ValueError carrying the payload size + first bytes so a
    truncated / empty / non-NIfTI response is diagnosable, instead of surfacing as
    a bare gzip or nibabel struct error with no context.
    """
    gzipped = data[:2] == b"\x1f\x8b"
    try:
        if gzipped:
            data = gzip.decompress(data)
        fh = nib.FileHolder(fileobj=io.BytesIO(data))
        return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    except Exception as e:
        head = data[:8].hex(" ") or "(empty)"
        raise ValueError(
            f"Cannot parse NIfTI mask from {len(data)} bytes "
            f"(gzip={gzipped}, head={head}): {type(e).__name__}: {e}"
        ) from e


def bytes_to_npz(data: bytes) -> dict:
    """Deserialize .npz bytes to dict of numpy arrays."""
    bio = io.BytesIO(data)
    return dict(np.load(bio))


def make_nifti_upload(img: nib.Nifti1Image, filename: str = "image.nii.gz") -> tuple:
    """Create a (field_name, (filename, bytes, content_type)) tuple for multipart upload."""
    return ("files", (filename, nifti_to_bytes(img), "application/octet-stream"))

def point_from_napari(point_zyx: tuple, napari_shape_zyx: tuple) -> list:
    """
    Converts a single click coordinate from Napari (Z, Y, X) back to Nibabel (X, Y, Z).
    
    Given that `to_napari` does:
    1. Transpose: (X, Y, Z) -> (Z', Y', X')
    2. Flip axis (0, 1): Z' -> Z, Y' -> Y
       Z = shape_z - 1 - Z'
       Y = shape_y - 1 - Y'
       
    This function reverses these operations on a single coordinate.
    """
    z_nap, y_nap, x_nap = point_zyx
    shape_z, shape_y, shape_x = napari_shape_zyx
    
    # 1. Undo Flip
    z_prime = shape_z - 1 - z_nap
    y_prime = shape_y - 1 - y_nap
    x_prime = x_nap
    
    # 2. Undo Transpose -> (X, Y, Z) == (x_prime, y_prime, z_prime)
    return [x_prime, y_prime, z_prime]
