"""File manager for session storage operations."""
import re
import shutil
from pathlib import Path
from typing import Literal

from .config import settings


FileType = Literal["ct", "pet", "tumor_seg", "organ_seg"]


def safe_filename_part(name: str) -> str:
    """``name`` usable inside a file name: anything but letters, digits, ``._-`` → ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"


class FileManager:
    """Handles file I/O operations for session storage."""

    # ── New-style: path utilities ─────────────────────────────────────────

    @staticmethod
    def get_segmentation_path(ct_path: Path, suffix: str = "Segmentation") -> Path:
        """Return the segmentation output path next to the CT file.

        E.g. /data/patient.nii.gz  →  /data/patient_Segmentation.nii.gz
        For a DICOM series folder the file goes next to the folder:
             /data/study/CT_3mm/    →  /data/study/CT_3mm_Segmentation.nii.gz
        ``suffix`` replaces "Segmentation", e.g. "organ" → patient_organ.nii.gz.
        """
        p = Path(ct_path)
        if p.is_dir():
            return p.parent / f"{p.name}_{suffix}.nii.gz"
        # Strip .nii.gz or .nii to get the base stem
        name = p.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break
        else:
            stem = p.stem
        return p.parent / f"{stem}_{suffix}.nii.gz"

    # ── Legacy helpers (kept for backward-compat with old storage/ sessions) ──

    @staticmethod
    def get_session_dir(session_id: int, create: bool = True) -> Path:
        session_dir = settings.DATA_DIR / str(session_id)
        if create:
            session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def get_file_path(session_id: int, file_type: FileType, create: bool = True) -> Path:
        session_dir = FileManager.get_session_dir(session_id, create=create)
        return session_dir / f"{file_type}.nii.gz"

    @staticmethod
    def file_exists(session_id: int, file_type: FileType) -> bool:
        return FileManager.get_file_path(session_id, file_type, create=False).exists()

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
        session_dir = FileManager.get_session_dir(session_id, create=False)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"[FileManager] Deleted session dir: {session_dir}")
