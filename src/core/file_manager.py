"""File manager for session storage operations."""
import shutil
from pathlib import Path
from typing import Literal

from .config import settings


FileType = Literal["ct", "pet", "tumor_seg"]


class FileManager:
    """Handles file I/O operations for session storage."""

    # ── New-style: path utilities ─────────────────────────────────────────

    @staticmethod
    def get_segmentation_path(ct_path: Path) -> Path:
        """Return the segmentation output path next to the CT file.

        E.g. /data/patient.nii.gz  →  /data/patient_Segmentation.nii.gz
        """
        p = Path(ct_path)
        # Strip .nii.gz or .nii to get the base stem
        name = p.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break
        else:
            stem = p.stem
        return p.parent / f"{stem}_Segmentation.nii.gz"

    # ── Legacy helpers (kept for backward-compat with old storage/ sessions) ──

    @staticmethod
    def get_session_dir(session_id: int) -> Path:
        session_dir = settings.DATA_DIR / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def get_file_path(session_id: int, file_type: FileType) -> Path:
        session_dir = FileManager.get_session_dir(session_id)
        return session_dir / f"{file_type}.nii.gz"

    @staticmethod
    def file_exists(session_id: int, file_type: FileType) -> bool:
        return FileManager.get_file_path(session_id, file_type).exists()

    @staticmethod
    def load_nifti(session_id: int, file_type: FileType):
        import nibabel as nib
        file_path = FileManager.get_file_path(session_id, file_type)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return nib.load(file_path)

    @staticmethod
    def save_nifti(image, session_id: int, file_type: FileType) -> Path:
        import nibabel as nib
        dest_path = FileManager.get_file_path(session_id, file_type)
        nib.save(image, dest_path)
        print(f"[FileManager] Saved Nifti to {dest_path}")
        return dest_path

    @staticmethod
    def delete_session_files(session_id: int) -> None:
        session_dir = FileManager.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"[FileManager] Deleted session dir: {session_dir}")
