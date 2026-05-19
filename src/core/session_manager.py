from typing import Optional
import nibabel as nib
import numpy as np
from pathlib import Path

from .file_manager import FileManager
from ..database.session_repository import SessionRepository
from .engine.report_engine import ReportEngine


def _stem_from_path(path: Path) -> str:
    """Return filename without NIfTI extensions (.nii.gz or .nii)."""
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


class SessionManager:
    """Manages the active session state, including in-memory images and database records."""

    def __init__(self):
        self.current_session_id: Optional[int] = None
        self.ct_image: Optional[nib.Nifti1Image] = None
        self.pet_image: Optional[nib.Nifti1Image] = None
        self.tumor_mask: Optional[nib.Nifti1Image] = None

        # ROI mask for interactive refinement (raw uint8 XYZ array, never saved to disk)
        self.roi_mask: Optional[np.ndarray] = None

        # True when the in-memory tumor mask differs from what's on disk.
        # Cleared on fresh load_session/create_session/update_current_session
        # (tumor file path), and on successful save_session.
        # Set by set_tumor_mask, ensure_roi_mask (zeros init), and by viewer
        # paint events via MainWindow's sig_mask_modified handler.
        self.tumor_dirty: bool = False

        self.patient_name: str = ""
        self.doctor_name: str = ""

        self.lesion_bboxes: list = []
        self.lesion_ids: list = []

        self.repository = SessionRepository()

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(
        self,
        doctor_name: str,
        patient_name: str,
        ct_path: Optional[Path] = None,
        pet_path: Optional[Path] = None,
        tumor_seg_path: Optional[Path] = None,
    ) -> int:
        """Create a new session, storing absolute paths (no file copy)."""
        # Auto-populate names from CT filename when both are blank
        if (not doctor_name or not patient_name) and ct_path:
            stem = _stem_from_path(Path(ct_path))
            doctor_name = doctor_name or stem
            patient_name = patient_name or stem

        abs_ct = str(Path(ct_path).absolute()) if ct_path else None
        abs_pet = str(Path(pet_path).absolute()) if pet_path else None
        abs_tumor_seg = str(Path(tumor_seg_path).absolute()) if tumor_seg_path else None

        session = self.repository.create(
            patient_name=patient_name,
            doctor_name=doctor_name,
            ct_path=abs_ct,
            pet_path=abs_pet,
            tumor_seg_path=abs_tumor_seg,
        )
        self.current_session_id = session.id
        self.patient_name = patient_name or ""
        self.doctor_name = doctor_name or ""

        # Load directly from original paths — no copy
        self.ct_image = nib.load(abs_ct) if abs_ct else None
        self.pet_image = nib.load(abs_pet) if abs_pet else None

        self.tumor_mask = nib.load(abs_tumor_seg) if abs_tumor_seg else None
        self.roi_mask = None
        self.tumor_dirty = False
        self.clear_lesion_data()

        print(f"[SessionManager] Created session {self.current_session_id}")
        return self.current_session_id

    def update_current_session(
        self,
        ct_path: Optional[Path] = None,
        pet_path: Optional[Path] = None,
        tumor_seg_path: Optional[Path] = None,
    ):
        """Load new files into the current session without copying."""
        if self.current_session_id is None:
            raise ValueError("No active session to update.")

        update_kwargs = {}
        if ct_path:
            abs_ct = str(Path(ct_path).absolute())
            self.ct_image = nib.load(abs_ct)
            update_kwargs["ct_path"] = abs_ct
            
            session = self.repository.get_by_id(self.current_session_id)
            if session:
                stem = _stem_from_path(Path(ct_path))
                if not session.doctor_name or session.doctor_name == "System":
                    self.doctor_name = stem
                    update_kwargs["doctor_name"] = stem
                if not session.patient_name or session.patient_name == "Anonymous":
                    self.patient_name = stem
                    update_kwargs["patient_name"] = stem
        if pet_path:
            abs_pet = str(Path(pet_path).absolute())
            self.pet_image = nib.load(abs_pet)
            update_kwargs["pet_path"] = abs_pet

        if tumor_seg_path:
            abs_tumor_seg = str(Path(tumor_seg_path).absolute())
            self.tumor_mask = nib.load(abs_tumor_seg)
            update_kwargs["tumor_seg_path"] = abs_tumor_seg
            # Fresh load from disk → in-memory matches disk
            self.tumor_dirty = False

        if update_kwargs:
            self.repository.update(self.current_session_id, **update_kwargs)
        print(f"[SessionManager] Updated session {self.current_session_id}")

    def load_session(self, session_id: int):
        """Load an existing session from DB paths into RAM."""
        self.current_session_id = session_id
        self.clear_lesion_data()

        session = self.repository.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self.patient_name = session.patient_name or ""
        self.doctor_name = session.doctor_name or ""

        # CT
        self.ct_image = self._load_nifti_from_db_path(session.ct_path, session_id, "ct")

        # PET
        self.pet_image = self._load_nifti_from_db_path(session.pet_path, session_id, "pet")

        # Tumor segmentation
        if session.tumor_seg_path:
            tseg = Path(session.tumor_seg_path)
            if tseg.exists():
                self.tumor_mask = nib.load(tseg)
            else:
                legacy_seg = FileManager.get_file_path(session_id, "tumor_seg")
                if legacy_seg.exists():
                    self.tumor_mask = nib.load(legacy_seg)
                else:
                    self.tumor_mask = None
        else:
            legacy_seg = FileManager.get_file_path(session_id, "tumor_seg")
            if legacy_seg.exists():
                self.tumor_mask = nib.load(legacy_seg)
            else:
                self.tumor_mask = None

        self.roi_mask = None
        self.tumor_dirty = False
        print(f"[SessionManager] Loaded session {session_id}")

    def _load_nifti_from_db_path(self, db_path: Optional[str], session_id: int, file_type: str) -> Optional[nib.Nifti1Image]:
        """Load a NIfTI from a DB-stored path. Fallback to legacy storage if missing."""
        if db_path:
            p = Path(db_path)
            if p.exists():
                return nib.load(p)
                
        legacy_path = FileManager.get_file_path(session_id, file_type)
        if legacy_path.exists():
            return nib.load(legacy_path)
            
        print(f"[SessionManager] File not found: {db_path} or {legacy_path}")
        return None

    def save_session(self):
        """Persist the in-memory tumor mask to disk next to the CT file."""
        if self.current_session_id is None:
            print("[SessionManager] No active session to save.")
            return
        if self.tumor_mask is None:
            print("[SessionManager] No tumor mask to save.")
            return

        session = self.repository.get_by_id(self.current_session_id)
        
        # Use existing path only if the file actually exists on disk.
        # If DB has a stale path (file was deleted/moved), regenerate next to CT.
        if session.tumor_seg_path and Path(session.tumor_seg_path).parent.exists():
            seg_path = Path(session.tumor_seg_path)
        else:
            ct_path = Path(session.ct_path) if session.ct_path else None
            if ct_path and ct_path.exists():
                seg_path = FileManager.get_segmentation_path(ct_path)
            else:
                # Fallback: write to the old session storage dir
                seg_path = FileManager.get_file_path(self.current_session_id, "tumor_seg")

        nib.save(self.tumor_mask, seg_path)
        self.repository.update(self.current_session_id, tumor_seg_path=str(seg_path))
        self.tumor_dirty = False
        print(f"[SessionManager] Saved session {self.current_session_id} → {seg_path}")

    # ── Data accessors ────────────────────────────────────────────────────

    def get_ct_data(self) -> Optional[np.ndarray]:
        if self.ct_image:
            return self.ct_image.get_fdata(dtype=np.float32)
        return None

    def get_pet_data(self) -> Optional[np.ndarray]:
        if self.pet_image:
            return self.pet_image.get_fdata(dtype=np.float32)
        return None

    def get_tumor_mask_data(self) -> Optional[np.ndarray]:
        if self.tumor_mask:
            return np.asarray(self.tumor_mask.dataobj, dtype=np.uint8)
        return None

    def get_all_sessions(self):
        return self.repository.get_all()

    def clear_lesion_data(self):
        self.lesion_bboxes = []
        self.lesion_ids = []

    # ── Mask helpers ──────────────────────────────────────────────────────

    def set_tumor_mask(self, mask_array: np.ndarray):
        if self.ct_image is None:
            raise ValueError("CT Image must be loaded to set mask (need affine).")
        self.tumor_mask = nib.Nifti1Image(mask_array.astype(np.uint8), self.ct_image.affine)
        self.tumor_dirty = True
        self.clear_lesion_data()

    def set_roi_mask(self, mask_array: np.ndarray):
        self.roi_mask = mask_array.astype(np.uint8)

    def get_roi_mask_data(self) -> Optional[np.ndarray]:
        return self.roi_mask

    def clear_roi_mask(self):
        if self.roi_mask is not None:
            self.roi_mask[:] = 0

    def ensure_roi_mask(self):
        if self.ct_image is None:
            return
        if self.tumor_mask is None:
            print(f"[SessionManager] Creating new zeroed Tumor Mask ({self.ct_image.shape})")
            self.tumor_mask = nib.Nifti1Image(
                np.zeros(self.ct_image.shape, dtype=np.uint8),
                self.ct_image.affine,
                self.ct_image.header,
            )
            self.tumor_dirty = True
        if self.roi_mask is None:
            self.roi_mask = np.zeros(self.ct_image.shape, dtype=np.uint8)

    def merge_roi_into_tumor(self) -> Optional[np.ndarray]:
        if self.roi_mask is None or self.tumor_mask is None:
            return self.get_tumor_mask_data()
        tumor_data = self.get_tumor_mask_data()
        merged = np.maximum(tumor_data, self.roi_mask).astype(np.uint8)
        self.set_tumor_mask(merged)
        self.clear_roi_mask()
        return merged

    # ── Report generation ─────────────────────────────────────────────────

    def generate_report(self) -> dict:
        """Generate a clinical report from the last-saved tumor segmentation."""
        if self.current_session_id is None:
            raise ValueError("No active session.")
        if self.pet_image is None:
            raise ValueError("PET image must be loaded to generate a report.")

        session = self.repository.get_by_id(self.current_session_id)
        tumor_path = None
        if session.tumor_seg_path:
            p = Path(session.tumor_seg_path)
            if p.exists():
                tumor_path = p

        if tumor_path is None:
            raise ValueError(
                "No saved tumor mask found for this session. "
                "Run segmentation and save first."
            )

        disk_mask = nib.load(tumor_path)
        result = ReportEngine.compute_report(self.pet_image, disk_mask)

        self.lesion_bboxes = [lesion["bbox"] for lesion in result["lesions"]]
        self.lesion_ids = [lesion["id"] for lesion in result["lesions"]]
        return result
