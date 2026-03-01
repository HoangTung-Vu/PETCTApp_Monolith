"""Utility functions for NIfTI image handling."""
import io
from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np


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


def to_napari(data: np.ndarray) -> np.ndarray:
    """
    Converts Nibabel (X, Y, Z) to Napari (Z, Y, X) with vertical flip.
    """
    # 1. Transpose to (Z, Y, X)
    data_zyx = np.transpose(data, (2, 1, 0))
    # 2. Flip Z (axis 0) and Y (axis 1)
    data_zyx = np.flip(data_zyx, axis=(0, 1))
    return data_zyx


def from_napari(data_zyx: np.ndarray) -> np.ndarray:
    """
    Converts Napari (Z, Y, X) back to Nibabel (X, Y, Z).
    """
    # 1. Undo Flip
    data_zyx = np.flip(data_zyx, axis=(0, 1))
    # 2. Undo Transpose (transpose is its own inverse here)
    data_xyz = np.transpose(data_zyx, (2, 1, 0))
    return data_xyz


def nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to .nii.gz bytes in memory."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def bytes_to_nifti(data: bytes) -> nib.Nifti1Image:
    """Deserialize .nii.gz bytes to a nibabel Nifti1Image."""
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def bytes_to_npz(data: bytes) -> dict:
    """Deserialize .npz bytes to dict of numpy arrays."""
    bio = io.BytesIO(data)
    return dict(np.load(bio))


def make_nifti_upload(img: nib.Nifti1Image, filename: str = "image.nii.gz") -> tuple:
    """Create a (field_name, (filename, bytes, content_type)) tuple for multipart upload."""
    return ("files", (filename, nifti_to_bytes(img), "application/octet-stream"))
