"""File manager for session storage operations."""
import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from .config import settings


FileType = Literal["ct", "pet", "tumor_seg", "organ_seg"]
NumpyFileType = Literal["tumor_prob"]


class FileManager:
    """Handles file I/O operations for session storage."""
    
    @staticmethod
    def get_session_dir(session_id: int) -> Path:
        """Get directory path for a session."""
        session_dir = settings.DATA_DIR / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    @staticmethod
    def get_file_path(session_id: int, file_type: FileType) -> Path:
        """Get the standard path for a file type in a session.
        
        Args:
            session_id: Session ID
            file_type: One of 'ct', 'pet', 'tumor_seg', 'organ_seg'
            
        Returns:
            Path to the file
        """
        session_dir = FileManager.get_session_dir(session_id)
        return session_dir / f"{file_type}.nii.gz"
    
    @staticmethod
    def copy_to_storage(src_path: Path, session_id: int, file_type: FileType) -> Path:
        """Copy a file to session storage with standard naming.
        
        Args:
            src_path: Source file path
            session_id: Session ID
            file_type: Type of file ('ct', 'pet', etc.)
            
        Returns:
            Path to copied file in storage
        """
        dest_path = FileManager.get_file_path(session_id, file_type)
        shutil.copy2(src_path, dest_path)
        print(f"[FileManager] Copied {src_path} -> {dest_path}")
        return dest_path
    
    @staticmethod
    def file_exists(session_id: int, file_type: FileType) -> bool:
        """Check if a file exists in session storage."""
        return FileManager.get_file_path(session_id, file_type).exists()
    
    @staticmethod
    def save_nifti(image, session_id: int, file_type: FileType) -> Path:
        """Save a Nifti1Image to session storage.
        
        Args:
            image: nib.Nifti1Image object
            session_id: Session ID
            file_type: Type of file
            
        Returns:
            Path to saved file
        """
        import nibabel as nib
        dest_path = FileManager.get_file_path(session_id, file_type)
        nib.save(image, dest_path)
        print(f"[FileManager] Saved Nifti to {dest_path}")
        return dest_path

    @staticmethod
    def load_nifti(session_id: int, file_type: FileType):
        """Load a Nifti1Image from session storage.
        
        Returns:
            nib.Nifti1Image object
        """
        import nibabel as nib
        file_path = FileManager.get_file_path(session_id, file_type)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return nib.load(file_path)

    @staticmethod
    def delete_session_files(session_id: int) -> None:
        """Delete all files for a session."""
        session_dir = FileManager.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"[FileManager] Deleted session dir: {session_dir}")

    # ──── Numpy Volume Storage ────

    @staticmethod
    def get_numpy_path(session_id: int, file_type: NumpyFileType) -> Path:
        """Get the standard path for a numpy file in a session."""
        session_dir = FileManager.get_session_dir(session_id)
        return session_dir / f"{file_type}.npy"

    @staticmethod
    def save_numpy(array: np.ndarray, session_id: int, file_type: NumpyFileType) -> Path:
        """Save a numpy array to session storage."""
        dest_path = FileManager.get_numpy_path(session_id, file_type)
        np.save(dest_path, array)
        print(f"[FileManager] Saved numpy to {dest_path} (shape={array.shape}, dtype={array.dtype})")
        return dest_path

    @staticmethod
    def load_numpy(session_id: int, file_type: NumpyFileType) -> np.ndarray:
        """Load a numpy array from session storage."""
        file_path = FileManager.get_numpy_path(session_id, file_type)
        if not file_path.exists():
            raise FileNotFoundError(f"Numpy file not found: {file_path}")
        arr = np.load(file_path)
        print(f"[FileManager] Loaded numpy from {file_path} (shape={arr.shape}, dtype={arr.dtype})")
        return arr

    @staticmethod
    def numpy_exists(session_id: int, file_type: NumpyFileType) -> bool:
        """Check if a numpy file exists in session storage."""
        return FileManager.get_numpy_path(session_id, file_type).exists()
