import numpy as np

def get_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """
    Calculate the spacing (dimension XYZ) of a voxel from the affine matrix.
    Returns (spacing_x, spacing_y, spacing_z) in mm.
    """
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

def get_voxel_volume_from_affine(affine: np.ndarray) -> float:
    """
    Compute voxel volume (mm^3) using the determinant of the affine.
    This correctly handles shearing, rotation, and scaling.
    |det(A)| represents the volume of the unit voxel transformed by A.
    """
    return float(np.abs(np.linalg.det(affine[:3, :3])))

def get_voxel_volume_ml_from_affine(affine: np.ndarray) -> float:
    """
    Compute voxel volume in mL (cm^3) from the affine matrix.
    """
    return get_voxel_volume_from_affine(affine) * 1e-3
