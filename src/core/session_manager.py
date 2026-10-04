from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import nibabel as nib
import numpy as np
from pathlib import Path

from .file_manager import FileManager
from ..database.session_repository import SessionRepository
from .engine.report_engine import ReportEngine
from ..utils.nifti_utils import from_app_orientation, load_mask_volume, load_nifti_volume


def _stem_from_path(path: Path) -> str:
    """Return filename without NIfTI extensions (.nii.gz or .nii)."""
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _load_files(ct_path=None, pet_path=None, seg_path=None) -> dict:
    """Read the given NIfTI files concurrently (zlib releases the GIL while
    decompressing, so the .nii.gz files inflate in parallel)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {}
        if ct_path:
            jobs["ct"] = pool.submit(load_nifti_volume, ct_path)
        if pet_path:
            jobs["pet"] = pool.submit(load_nifti_volume, pet_path)
        if seg_path:
            jobs["seg"] = pool.submit(load_mask_volume, seg_path)
        return {key: job.result() for key, job in jobs.items()}


class SessionManager:
    """Manages the active session state, including in-memory images and database records.

    All images are held in memory in the app's LAS voxel order
    (``nifti_utils.to_app_orientation``); intensities keep their stored dtype.
    The tumor mask is written back in the native voxel order of the session
    grid, so it lines up voxel-for-voxel with the source CT.
    """

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

        # LAS → native voxel order of each loaded image (used when saving the mask).
        self._ct_back_ornt: Optional[np.ndarray] = None
        self._pet_back_ornt: Optional[np.ndarray] = None
        # Source files whose bytes equal the in-memory image — the segmentation
        # upload streams these unchanged. None for DICOM / reoriented NIfTI.
        self.ct_stream_path: Optional[Path] = None
        self.pet_stream_path: Optional[Path] = None
        # DICOM sessions: SUV conversion details and import warnings.
        self.suv_info = None
        self.load_warnings: list = []

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

        # Load (and validate) before creating the DB row, so a bad file leaves no row.
        loaded = _load_files(abs_ct, abs_pet, abs_tumor_seg)
        self._reset_images()
        self._apply_loaded(loaded)

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

        print(f"[SessionManager] Created session {self.current_session_id}")
        return self.current_session_id

    def create_session_from_dicom(self, ct_series, pet_series, resample_mode: str,
                                  doctor_name: str, patient_name: str, log=None) -> int:
        """Create a session from DICOM series read straight into memory."""
        from .dicom_loader import load_study

        study = load_study(ct_series, pet_series, resample_mode, log)
        self._reset_images()
        self._apply_dicom_study(study)

        ref = ct_series or pet_series
        # A segmentation saved by an earlier import of the same series is reused
        # instead of being silently overwritten by a fresh empty mask.
        seg_path = FileManager.get_segmentation_path(Path(ref.directory))
        abs_seg = None
        if seg_path.exists():
            try:
                self.tumor_mask = load_mask_volume(seg_path, self._ref_image().shape)
                abs_seg = str(seg_path)
                print(f"[SessionManager] Reusing existing segmentation {seg_path}")
            except Exception as e:
                print(f"[SessionManager] Ignoring existing segmentation {seg_path}: {e}")

        session = self.repository.create(
            patient_name=patient_name,
            doctor_name=doctor_name,
            ct_path=ct_series.directory if ct_series else None,
            pet_path=pet_series.directory if pet_series else None,
            tumor_seg_path=abs_seg,
            ct_series_uid=ct_series.uid if ct_series else None,
            pet_series_uid=pet_series.uid if pet_series else None,
            resample_mode=resample_mode,
        )
        self.current_session_id = session.id
        self.patient_name = patient_name or ""
        self.doctor_name = doctor_name or ""
        print(f"[SessionManager] Created DICOM session {self.current_session_id}")
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

        abs_ct = str(Path(ct_path).absolute()) if ct_path else None
        abs_pet = str(Path(pet_path).absolute()) if pet_path else None
        abs_tumor_seg = str(Path(tumor_seg_path).absolute()) if tumor_seg_path else None
        loaded = _load_files(abs_ct, abs_pet, abs_tumor_seg)
        self._apply_loaded(loaded)

        update_kwargs = {}
        if abs_ct:
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
        if abs_pet:
            update_kwargs["pet_path"] = abs_pet
        if abs_tumor_seg:
            update_kwargs["tumor_seg_path"] = abs_tumor_seg

        if update_kwargs:
            self.repository.update(self.current_session_id, **update_kwargs)
        print(f"[SessionManager] Updated session {self.current_session_id}")

    def load_session(self, session_id: int):
        """Load an existing session from DB paths into RAM."""
        session = self.repository.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self._reset_images()
        self.current_session_id = session_id
        self.patient_name = session.patient_name or ""
        self.doctor_name = session.doctor_name or ""

        if session.ct_series_uid or session.pet_series_uid:
            self._load_dicom_session(session)
            seg = self._existing_path(session.tumor_seg_path)
        else:
            ct = self._existing_path(session.ct_path) or self._legacy_path(session_id, "ct")
            pet = self._existing_path(session.pet_path) or self._legacy_path(session_id, "pet")
            if not ct and session.ct_path:
                print(f"[SessionManager] File not found: {session.ct_path}")
            if not pet and session.pet_path:
                print(f"[SessionManager] File not found: {session.pet_path}")
            seg = self._existing_path(session.tumor_seg_path)
            self._apply_loaded(_load_files(ct, pet))

        seg = seg or self._legacy_path(session_id, "tumor_seg")
        if seg and self._ref_image() is not None:
            self.tumor_mask = load_mask_volume(seg, self._ref_image().shape)
        self.tumor_dirty = False
        print(f"[SessionManager] Loaded session {session_id}")

    def _load_dicom_session(self, session):
        """Re-read the stored DICOM series (located again by SeriesInstanceUID)."""
        from .dicom_loader import find_series, load_study, scan_folder

        if session.ct_path and session.ct_path == session.pet_path:
            found = {s.uid: s for s in scan_folder(session.ct_path)}
            ct = found.get(session.ct_series_uid)
            pet = found.get(session.pet_series_uid)
        else:
            ct = find_series(session.ct_path, session.ct_series_uid)
            pet = find_series(session.pet_path, session.pet_series_uid)
        for uid, folder, series in ((session.ct_series_uid, session.ct_path, ct),
                                    (session.pet_series_uid, session.pet_path, pet)):
            if uid and series is None:
                raise FileNotFoundError(f"DICOM series {uid} not found in {folder}")
        study = load_study(ct, pet, session.resample_mode or "ct")
        self._apply_dicom_study(study)

    @staticmethod
    def _existing_path(db_path: Optional[str]) -> Optional[str]:
        return db_path if db_path and Path(db_path).exists() else None

    @staticmethod
    def _legacy_path(session_id: int, file_type: str) -> Optional[str]:
        legacy = FileManager.get_file_path(session_id, file_type, create=False)
        return str(legacy) if legacy.exists() else None

    # ── Image state ───────────────────────────────────────────────────────

    def _reset_images(self):
        self.ct_image = self.pet_image = self.tumor_mask = None
        self.roi_mask = None
        self._ct_back_ornt = self._pet_back_ornt = None
        self.ct_stream_path = self.pet_stream_path = None
        self.suv_info = None
        self.load_warnings = []
        self.tumor_dirty = False
        self.clear_lesion_data()

    def _apply_loaded(self, loaded: dict):
        """Commit freshly loaded NIfTI volumes after checking they share a grid."""
        ct = loaded.get("ct", (self.ct_image, self._ct_back_ornt, self.ct_stream_path))
        pet = loaded.get("pet", (self.pet_image, self._pet_back_ornt, self.pet_stream_path))
        self._check_grid(ct[0], pet[0])
        ref = ct[0] if ct[0] is not None else pet[0]
        seg = loaded.get("seg")
        if seg is not None:
            if ref is None:
                raise ValueError("Load a CT or PET image before the segmentation.")
            if tuple(seg.shape) != tuple(ref.shape):
                raise ValueError(
                    f"Segmentation shape {tuple(seg.shape)} does not match the image grid {tuple(ref.shape)}."
                )

        self.ct_image, self._ct_back_ornt, self.ct_stream_path = ct
        self.pet_image, self._pet_back_ornt, self.pet_stream_path = pet
        if seg is not None:
            self.tumor_mask = seg
            # Fresh load from disk → in-memory matches disk
            self.tumor_dirty = False
        if "ct" in loaded or "pet" in loaded:
            self.clear_lesion_data()
            # A new image on a different grid invalidates the old masks.
            if self.tumor_mask is not None and tuple(self.tumor_mask.shape) != tuple(ref.shape):
                self.tumor_mask = None
            if self.roi_mask is not None and tuple(self.roi_mask.shape) != tuple(ref.shape):
                self.roi_mask = None

    def _apply_dicom_study(self, study):
        self.ct_image, self.pet_image = study.ct, study.pet
        self._ct_back_ornt = study.back_ornt if study.ct is not None else None
        self._pet_back_ornt = study.back_ornt if study.pet is not None else None
        self.suv_info = study.suv
        self.load_warnings = list(study.warnings)

    @staticmethod
    def _check_grid(ct, pet):
        if ct is not None and pet is not None and tuple(ct.shape) != tuple(pet.shape):
            raise ValueError(
                f"CT {tuple(ct.shape)} and PET {tuple(pet.shape)} are not on the same voxel grid. "
                "Resample one onto the other first (DICOM import does this automatically)."
            )

    def _ref_image(self) -> Optional[nib.Nifti1Image]:
        """The image that defines the session grid (CT, else PET)."""
        return self.ct_image if self.ct_image is not None else self.pet_image

    @property
    def ref_back_ornt(self) -> Optional[np.ndarray]:
        return self._ct_back_ornt if self.ct_image is not None else self._pet_back_ornt

    def close_session(self):
        """Forget the current session (used when it is deleted)."""
        self._reset_images()
        self.current_session_id = None
        self.patient_name = self.doctor_name = ""

    def delete_session(self, session_id: int) -> bool:
        """Delete the DB record and the app-owned storage folder.

        Source images and the segmentation file next to the CT are kept.
        """
        if session_id == self.current_session_id:
            self.close_session()
        deleted = self.repository.delete(session_id)
        FileManager.delete_session_files(session_id)
        print(f"[SessionManager] Deleted session {session_id}")
        return deleted

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
        is_dicom = bool(session.ct_series_uid or session.pet_series_uid)
        ref_path = session.ct_path or (session.pet_path if is_dicom else None)
        if session.tumor_seg_path and Path(session.tumor_seg_path).parent.exists():
            seg_path = Path(session.tumor_seg_path)
        elif ref_path and Path(ref_path).exists():
            seg_path = FileManager.get_segmentation_path(Path(ref_path))
        else:
            # Fallback: write to the old session storage dir
            seg_path = FileManager.get_file_path(self.current_session_id, "tumor_seg")

        # Back to the native voxel order of the source grid, stored as uint8.
        mask = nib.Nifti1Image(self.get_tumor_mask_data(), self.tumor_mask.affine)
        mask = from_app_orientation(mask, self.ref_back_ornt)
        mask.set_data_dtype(np.uint8)
        try:
            nib.save(mask, seg_path)
        except OSError as e:
            # Read-only source folder (DICOM CD / network share) → app storage.
            fallback = FileManager.get_file_path(self.current_session_id, "tumor_seg")
            print(f"[SessionManager] Cannot write {seg_path} ({e}); saving to {fallback}")
            seg_path = fallback
            nib.save(mask, seg_path)
        self.repository.update(self.current_session_id, tumor_seg_path=str(seg_path))
        self.tumor_dirty = False
        print(f"[SessionManager] Saved session {self.current_session_id} → {seg_path}")

    # ── Data accessors ────────────────────────────────────────────────────

    def get_ct_data(self) -> Optional[np.ndarray]:
        """CT voxels (XYZ, LAS) in their stored dtype — in memory, no copy."""
        if self.ct_image is not None:
            return np.asanyarray(self.ct_image.dataobj)
        return None

    def get_pet_data(self) -> Optional[np.ndarray]:
        """PET voxels (XYZ, LAS) in their stored dtype — in memory, no copy."""
        if self.pet_image is not None:
            return np.asanyarray(self.pet_image.dataobj)
        return None

    def get_tumor_mask_data(self) -> Optional[np.ndarray]:
        """The in-memory uint8 mask array itself (edits through it are in place)."""
        if self.tumor_mask is not None:
            return np.asarray(self.tumor_mask.dataobj, dtype=np.uint8)
        return None

    def get_all_sessions(self):
        return self.repository.get_all()

    def clear_lesion_data(self):
        self.lesion_bboxes = []
        self.lesion_ids = []

    # ── Mask helpers ──────────────────────────────────────────────────────

    def set_tumor_mask(self, mask_array: np.ndarray):
        ref = self._ref_image()
        if ref is None:
            raise ValueError("A CT or PET image must be loaded to set the mask (need affine).")
        self.tumor_mask = nib.Nifti1Image(np.asarray(mask_array, dtype=np.uint8), ref.affine)
        self.tumor_dirty = True
        self.clear_lesion_data()

    def set_roi_mask(self, mask_array: np.ndarray):
        self.roi_mask = np.asarray(mask_array, dtype=np.uint8)

    def get_roi_mask_data(self) -> Optional[np.ndarray]:
        return self.roi_mask

    def clear_roi_mask(self):
        if self.roi_mask is not None:
            self.roi_mask[:] = 0

    def ensure_roi_mask(self):
        ref = self._ref_image()
        if ref is None:
            return
        if self.tumor_mask is None:
            print(f"[SessionManager] Creating new zeroed Tumor Mask ({ref.shape})")
            self.tumor_mask = nib.Nifti1Image(np.zeros(ref.shape, dtype=np.uint8, order="F"), ref.affine)
            self.tumor_dirty = True
        if self.roi_mask is None:
            self.roi_mask = np.zeros(ref.shape, dtype=np.uint8, order="F")

    def merge_roi_into_tumor(self) -> Optional[np.ndarray]:
        if self.roi_mask is None or self.tumor_mask is None:
            return self.get_tumor_mask_data()
        merged = np.maximum(self.get_tumor_mask_data(), self.roi_mask)
        self.set_tumor_mask(merged)
        self.clear_roi_mask()
        return merged

    # ── Report generation ─────────────────────────────────────────────────

    def generate_report(self) -> dict:
        """Generate a clinical report from the tumor segmentation.

        The report handler saves the session right before this runs, so the
        in-memory mask is exactly the saved one.
        """
        if self.current_session_id is None:
            raise ValueError("No active session.")
        if self.pet_image is None:
            raise ValueError("PET image must be loaded to generate a report.")
        if self.tumor_mask is None:
            raise ValueError(
                "No tumor mask found for this session. "
                "Run segmentation and save first."
            )

        result = ReportEngine.compute_report(self.pet_image, self.tumor_mask)

        self.lesion_bboxes = [lesion["bbox"] for lesion in result["lesions"]]
        self.lesion_ids = [lesion["id"] for lesion in result["lesions"]]
        return result
