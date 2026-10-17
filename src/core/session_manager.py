from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import nibabel as nib
import numpy as np
from pathlib import Path

from .file_manager import FileManager
from ..database.session_repository import SessionRepository
from .engine.report_engine import ReportEngine
from ..utils.nifti_utils import (
    from_app_orientation, grids_match, las_grid, load_mask_volume, load_nifti_volume,
)


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


class GridMismatchError(ValueError):
    """CT and PET are on different voxel grids and no resample mode was chosen."""


@dataclass
class GridInfo:
    """A voxel grid in the app's LAS order."""
    shape: tuple
    affine: np.ndarray

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(np.asarray(self.affine, dtype=float)[:3, :3], axis=0)

    def matches(self, other: "GridInfo") -> bool:
        return grids_match(self.shape, self.affine, other.shape, other.affine)

    def describe(self) -> str:
        sp = self.spacing
        fov = sp * np.asarray(self.shape)
        return (f"{'×'.join(str(n) for n in self.shape)} voxels, "
                f"{'×'.join(f'{v:.2f}' for v in sp)} mm, "
                f"FOV {'×'.join(f'{v:.0f}' for v in fov)} mm")


@dataclass
class GridMismatch:
    """What the GUI needs to ask how a CT/PET grid mismatch should be resolved."""
    ct: GridInfo
    pet: GridInfo
    # Per resample mode ("ct" / "pet" / "none"): True when that choice moves the
    # session to another grid while a tumor mask is loaded (the mask is closed).
    closes_mask: dict

    @property
    def same_shape(self) -> bool:
        return tuple(self.ct.shape) == tuple(self.pet.shape)


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
        # Set by set_tumor_mask, the eraser, and by viewer paint events via
        # MainWindow's sig_mask_modified handler.
        self.tumor_dirty: bool = False

        self.patient_name: str = ""
        self.doctor_name: str = ""

        # Per-lesion quantification of the saved tumor mask
        # (ReportEngine.compute_lesion_stats), recomputed when a segmentation is
        # loaded or saved. None = not available (no PET / no mask). ``stale`` is
        # set when the mask is edited after the computation; the old values are
        # kept until the next save recomputes them.
        self.lesion_stats: Optional[dict] = None
        self.lesion_stats_stale: bool = False

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
        # NIfTI sessions whose CT and PET were stored on different grids:
        # "ct" = PET was resampled onto the CT grid, "pet" = CT onto the PET grid.
        self.resample_mode: Optional[str] = None

        self.repository = SessionRepository()

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(
        self,
        doctor_name: str,
        patient_name: str,
        ct_path: Optional[Path] = None,
        pet_path: Optional[Path] = None,
        tumor_seg_path: Optional[Path] = None,
        resample_mode: Optional[str] = None,
    ) -> int:
        """Create a new session, storing absolute paths (no file copy).

        ``resample_mode`` ("ct" / "pet") puts CT and PET on one grid when their
        grids differ; without it a shape mismatch raises ``GridMismatchError``.
        A segmentation on another grid is resampled onto the image grid and
        saved next to the CT (see ``_store_resampled_segmentation``).
        """
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
        _, seg_resampled = self._apply_loaded(loaded, resample_mode)

        session = self.repository.create(
            patient_name=patient_name,
            doctor_name=doctor_name,
            ct_path=abs_ct,
            pet_path=abs_pet,
            tumor_seg_path=abs_tumor_seg,
            resample_mode=self.resample_mode,
        )
        self.current_session_id = session.id
        self.patient_name = patient_name or ""
        self.doctor_name = doctor_name or ""
        if seg_resampled:
            self._store_resampled_segmentation(abs_tumor_seg)

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
        resample_mode: Optional[str] = None,
    ):
        """Load new files into the current session without copying.

        ``resample_mode`` resolves a CT/PET grid mismatch (see ``create_session``).
        """
        if self.current_session_id is None:
            raise ValueError("No active session to update.")

        abs_ct = str(Path(ct_path).absolute()) if ct_path else None
        abs_pet = str(Path(pet_path).absolute()) if pet_path else None
        abs_tumor_seg = str(Path(tumor_seg_path).absolute()) if tumor_seg_path else None

        # The image kept in memory may itself be a resampled copy; re-read its
        # source so it is never resampled twice or onto a grid that is going away.
        reload_ct = reload_pet = None
        if abs_pet and not abs_ct and self._resampled_kind() == "ct":
            reload_ct = self._nifti_source("ct")
        if abs_ct and not abs_pet and self._resampled_kind() == "pet":
            reload_pet = self._nifti_source("pet")

        self.load_warnings = []
        loaded = _load_files(abs_ct or reload_ct, abs_pet or reload_pet, abs_tumor_seg)
        mask_closed, seg_resampled = self._apply_loaded(loaded, resample_mode)

        session = self.repository.get_by_id(self.current_session_id)
        is_dicom = bool(session and (session.ct_series_uid or session.pet_series_uid))
        update_kwargs = {}
        if (abs_ct or abs_pet) and not is_dicom:
            update_kwargs["resample_mode"] = self.resample_mode or ""
        if mask_closed and session and session.tumor_seg_path and not abs_tumor_seg:
            # The saved segmentation belongs to the old grid: keep the file,
            # forget it in the session so the next save does not overwrite it.
            update_kwargs["tumor_seg_path"] = ""
        if abs_ct:
            update_kwargs["ct_path"] = abs_ct
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
        if seg_resampled:
            self._store_resampled_segmentation(abs_tumor_seg)
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
            self._apply_loaded(_load_files(ct, pet), session.resample_mode)

        seg = seg or self._legacy_path(session_id, "tumor_seg")
        if seg and self._ref_image() is not None:
            try:
                self.tumor_mask = load_mask_volume(seg, self._ref_image().shape)
            except ValueError as e:
                # Wrong grid (e.g. saved before the session changed grids):
                # open the images anyway and say why the mask is missing.
                self.load_warnings.append(f"Segmentation not loaded: {e}")
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
        self.resample_mode = None
        self.tumor_dirty = False
        self.clear_lesion_data()

    def _apply_loaded(self, loaded: dict, resample_mode: Optional[str] = None) -> tuple:
        """Commit freshly loaded NIfTI volumes after putting them on one grid.

        Returns ``(mask_closed, seg_resampled)``: a loaded tumor mask was closed
        because the session moved to another grid / the loaded segmentation was
        resampled onto the image grid (the caller writes it to disk).
        """
        ct = loaded.get("ct", (self.ct_image, self._ct_back_ornt, self.ct_stream_path))
        pet = loaded.get("pet", (self.pet_image, self._pet_back_ornt, self.pet_stream_path))
        new_image = "ct" in loaded or "pet" in loaded
        mode, warnings = self.resample_mode, []
        if new_image:
            ct, pet, mode, warnings = self._align_grids(ct, pet, resample_mode)
        ref = ct[0] if ct[0] is not None else pet[0]
        seg = loaded.get("seg")
        seg_resampled = False
        if seg is not None:
            if ref is None:
                raise ValueError("Load a CT or PET image before the segmentation.")
            if tuple(seg.shape) != tuple(ref.shape):
                seg = self._resample_segmentation(seg, ref)
                seg_resampled = True
            elif not grids_match(seg.shape, seg.affine, ref.shape, ref.affine):
                # Same size: kept voxel-for-voxel as before (masks written by
                # scripts often carry a placeholder affine).
                print("[SessionManager] Segmentation header position differs from the image; "
                      "loaded voxel-for-voxel.")

        old_ref = self._ref_image()
        self.ct_image, self._ct_back_ornt, self.ct_stream_path = ct
        self.pet_image, self._pet_back_ornt, self.pet_stream_path = pet
        self.resample_mode = mode
        self.load_warnings.extend(warnings)
        mask_closed = False
        if seg is not None:
            self.tumor_mask = seg
            # Fresh load from disk → in-memory matches disk
            self.tumor_dirty = False
            self.clear_lesion_data()
        if new_image:
            self.clear_lesion_data()
            # A new image on a different grid invalidates the old masks.
            moved = old_ref is not None and not grids_match(
                old_ref.shape, old_ref.affine, ref.shape, ref.affine)
            if seg is None and self.tumor_mask is not None and (
                    moved or tuple(self.tumor_mask.shape) != tuple(ref.shape)):
                self.tumor_mask = None
                self.tumor_dirty = False
                mask_closed = True
            if self.roi_mask is not None and (moved or tuple(self.roi_mask.shape) != tuple(ref.shape)):
                self.roi_mask = None
        return mask_closed, seg_resampled

    @staticmethod
    def _resample_segmentation(seg: nib.Nifti1Image, ref: nib.Nifti1Image) -> nib.Nifti1Image:
        """A segmentation stored on another grid, put on the image grid (nearest neighbour)."""
        from .resample import resample_onto

        out = resample_onto(seg, ref, 0.0, method="nearest")
        data = np.asarray(out.dataobj)
        if not data.any() and np.asarray(seg.dataobj).any():
            raise ValueError(
                f"The segmentation ({tuple(seg.shape)} voxels) does not overlap the image "
                f"({tuple(ref.shape)} voxels) — check that it belongs to this study."
            )
        print(f"[SessionManager] Segmentation {tuple(seg.shape)} resampled (nearest neighbour) "
              f"onto the image grid {tuple(ref.shape)}")
        return nib.Nifti1Image(data, ref.affine)

    @staticmethod
    def _align_grids(ct, pet, resample_mode: Optional[str]):
        """Put the ``(image, back_ornt, stream_path)`` pair on one grid.

        Returns ``(ct, pet, applied_mode, warnings)``. Same resampling as the
        DICOM import (``resample.resample_to_grid``): CT is −1024 HU and PET 0
        outside the source field of view, the PET undershoot is clamped to 0.
        A resampled image takes the reference's native orientation and cannot
        be streamed from its source file.
        """
        if ct[0] is None or pet[0] is None or grids_match(
                ct[0].shape, ct[0].affine, pet[0].shape, pet[0].affine):
            return ct, pet, None, []
        from .resample import resample_onto

        if resample_mode == "ct":
            print(f"[SessionManager] Resampling PET {tuple(pet[0].shape)} onto the CT grid {tuple(ct[0].shape)}")
            pet = (resample_onto(pet[0], ct[0], 0.0, clamp_negative=True), ct[1], None)
            return ct, pet, "ct", []
        if resample_mode == "pet":
            print(f"[SessionManager] Resampling CT {tuple(ct[0].shape)} onto the PET grid {tuple(pet[0].shape)}")
            ct = (resample_onto(ct[0], pet[0], -1024.0, method="antialias"), pet[1], None)
            return ct, pet, "pet", []
        if tuple(ct[0].shape) != tuple(pet[0].shape):
            raise GridMismatchError(
                f"CT {tuple(ct[0].shape)} and PET {tuple(pet[0].shape)} are not on the same voxel grid. "
                "Load them again and choose how to resample one onto the other."
            )
        return ct, pet, None, [
            "CT and PET have the same size but different position or spacing — "
            "they were loaded without resampling and may not line up."
        ]

    def check_grid_mismatch(self, ct_path=None, pet_path=None) -> Optional[GridMismatch]:
        """How loading ``ct_path`` / ``pet_path`` would clash with the other image.

        Reads headers only (cheap, for the GUI thread). Returns None when there
        is nothing to resolve: no counterpart loaded, or the grids already match.
        """
        ct = GridInfo(*las_grid(ct_path)) if ct_path else self._source_grid("ct")
        pet = GridInfo(*las_grid(pet_path)) if pet_path else self._source_grid("pet")
        if ct is None or pet is None or ct.matches(pet):
            return None
        closes_mask = {}
        ref = self._ref_image()
        if self.tumor_mask is not None and ref is not None:
            current = GridInfo(tuple(ref.shape), ref.affine)
            for mode, target in (("ct", ct), ("pet", pet), ("none", ct)):
                closes_mask[mode] = not target.matches(current)
        return GridMismatch(ct=ct, pet=pet, closes_mask=closes_mask)

    def _source_grid(self, kind: str) -> Optional[GridInfo]:
        """Grid of the loaded CT/PET as stored on disk (before any resampling)."""
        img = self.ct_image if kind == "ct" else self.pet_image
        if img is None:
            return None
        source = self._nifti_source(kind) if self._resampled_kind() == kind else None
        if source:
            try:
                return GridInfo(*las_grid(source))
            except Exception as e:
                print(f"[SessionManager] Cannot read {source}: {e}")
        return GridInfo(tuple(img.shape), img.affine)

    def _resampled_kind(self) -> Optional[str]:
        """Which in-memory image ("ct" / "pet") is a resampled copy, if any."""
        return {"ct": "pet", "pet": "ct"}.get(self.resample_mode)

    def _nifti_source(self, kind: str) -> Optional[str]:
        """Path of the current NIfTI session's CT/PET file (None for DICOM)."""
        if self.current_session_id is None:
            return None
        session = self.repository.get_by_id(self.current_session_id)
        if session is None or session.ct_series_uid or session.pet_series_uid:
            return None
        return self._existing_path(session.ct_path if kind == "ct" else session.pet_path)

    def _apply_dicom_study(self, study):
        self.ct_image, self.pet_image = study.ct, study.pet
        self._ct_back_ornt = study.back_ornt if study.ct is not None else None
        self._pet_back_ornt = study.back_ornt if study.pet is not None else None
        self.suv_info = study.suv
        self.load_warnings = list(study.warnings)

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
        if session.tumor_seg_path and Path(session.tumor_seg_path).parent.exists():
            seg_path = Path(session.tumor_seg_path)
        else:
            seg_path = self._default_seg_path(session)
        seg_path = self._write_tumor_mask(seg_path)
        self.repository.update(self.current_session_id, tumor_seg_path=str(seg_path))
        self.tumor_dirty = False
        print(f"[SessionManager] Saved session {self.current_session_id} → {seg_path}")

    def _store_resampled_segmentation(self, source_path: str):
        """Save a segmentation resampled at load time and make it the session's file.

        Written where the AI segmentation goes (``<CT>_Segmentation.nii.gz``
        next to the CT, overwriting it), but never over the file it was read
        from. The session then points to the new file, not the source.
        """
        session = self.repository.get_by_id(self.current_session_id)
        target = self._default_seg_path(session)
        if Path(source_path).resolve() == target.resolve():
            target = target.with_name(f"{_stem_from_path(target)}_resampled.nii.gz")
        target = self._write_tumor_mask(target)
        self.repository.update(self.current_session_id, tumor_seg_path=str(target))
        self.tumor_dirty = False
        print(f"[SessionManager] Resampled segmentation from {source_path} saved → {target}")

    def _default_seg_path(self, session) -> Path:
        """Where a new segmentation file goes: next to the image whose grid it is on."""
        is_dicom = bool(session.ct_series_uid or session.pet_series_uid)
        ref_path = session.ct_path or (session.pet_path if is_dicom else None)
        if not is_dicom and self.resample_mode == "pet" and session.pet_path:
            # CT was resampled onto the PET grid: the mask lives on the PET grid,
            # so it is written next to the PET file it lines up with.
            ref_path = session.pet_path
        if ref_path and Path(ref_path).exists():
            return FileManager.get_segmentation_path(Path(ref_path))
        # Fallback: write to the old session storage dir
        return FileManager.get_file_path(self.current_session_id, "tumor_seg")

    def _write_tumor_mask(self, seg_path: Path) -> Path:
        """Write the tumor mask in the source grid's voxel order; returns the path used."""
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
        return Path(seg_path)

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
        """Forget the lesion statistics (new session / new mask from disk)."""
        self.lesion_stats = None
        self.lesion_stats_stale = False

    def invalidate_lesion_stats(self):
        """The mask was edited: keep showing the old values, recompute on save."""
        if self.lesion_stats is not None:
            self.lesion_stats_stale = True

    def update_lesion_stats(self):
        """(Re)compute the per-lesion statistics if they are missing or stale.

        Runs in worker threads right after a load or a save. Without PET or a
        tumor mask the statistics are ``None``. A failure is logged and leaves
        them ``None`` — it must not fail the load / save that triggered it.
        """
        if self.pet_image is None or self.tumor_mask is None:
            self.clear_lesion_data()
            return
        if self.lesion_stats is not None and not self.lesion_stats_stale:
            return
        try:
            self.lesion_stats = ReportEngine.compute_lesion_stats(
                self.get_pet_data(), self.get_tumor_mask_data(), self.pet_image.affine
            )
        except Exception:
            import traceback
            traceback.print_exc()
            self.clear_lesion_data()
            return
        self.lesion_stats_stale = False
        print(f"[SessionManager] Lesion stats: {len(self.lesion_stats['lesions'])} lesion(s), "
              f"tTLG={self.lesion_stats['tTLG']:.2f}")

    # ── Mask helpers ──────────────────────────────────────────────────────

    def set_tumor_mask(self, mask_array: np.ndarray):
        ref = self._ref_image()
        if ref is None:
            raise ValueError("A CT or PET image must be loaded to set the mask (need affine).")
        self.tumor_mask = nib.Nifti1Image(np.asarray(mask_array, dtype=np.uint8), ref.affine)
        self.tumor_dirty = True
        self.invalidate_lesion_stats()

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
            # In memory only: the file is written by the Segmentation tab's save
            # button. An untouched empty mask has nothing to lose, so it is not dirty.
            print(f"[SessionManager] Creating new zeroed Tumor Mask ({ref.shape})")
            self.tumor_mask = nib.Nifti1Image(np.zeros(ref.shape, dtype=np.uint8, order="F"), ref.affine)
        if self.roi_mask is None:
            self.roi_mask = np.zeros(ref.shape, dtype=np.uint8, order="F")

    def merge_roi_into_tumor(self) -> Optional[np.ndarray]:
        if self.roi_mask is None or self.tumor_mask is None:
            return self.get_tumor_mask_data()
        merged = np.maximum(self.get_tumor_mask_data(), self.roi_mask)
        self.set_tumor_mask(merged)
        self.clear_roi_mask()
        return merged
