"""
Worker for converting DICOM series to NIfTI in a background thread.

Auto-detects CT and PET series within the input folder by scanning
subdirectories and reading the DICOM Modality tag.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class DicomConversionWorker(QThread):
    """
    Runs DICOM → NIfTI conversion in a background thread.

    Emits:
        sig_log(str)                — progress / info messages for the log area
        sig_finished(str, str)      — (ct_nii_path, pet_nii_path) on success
                                      either may be "" if not found
        sig_error(str)              — error message on failure
    """

    sig_log      = pyqtSignal(str)
    sig_finished = pyqtSignal(str, str)   # ct_path, pet_or_suv_path
    sig_error    = pyqtSignal(str)

    def __init__(
        self,
        dcm_root: str,
        out_dir: str,
        pid_str: str = "PATIENT",
        do_suv: bool = True,
        do_resample: bool = True,
        resample_mode: str = "ct",
        parent=None,
    ):
        """
        Args:
            dcm_root:       Root folder to search for CT / PET DICOM series.
            out_dir:        Destination folder for .nii.gz output files.
            pid_str:        Patient ID prefix used in output filenames.
            do_suv:         Compute SUV from raw PET counts.
            do_resample:    Resample images after conversion.
            resample_mode:  "ct"  → upsample PET/SUV to CT grid (default)
                            "pet" → downsample CT to PET grid
        """
        super().__init__(parent)
        self.dcm_root      = dcm_root
        self.out_dir       = out_dir
        self.pid_str       = pid_str
        self.do_suv        = do_suv
        self.do_resample   = do_resample
        self.resample_mode = resample_mode
        self._tmp_dirs     = []

    # ------------------------------------------------------------------

    def run(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.sig_error.emit(str(exc))
        finally:
            self._cleanup_tmp()

    def _log(self, msg: str):
        print(f"[DicomConversionWorker] {msg}")
        self.sig_log.emit(msg)

    def _cleanup_tmp(self):
        import shutil
        for d in self._tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._tmp_dirs = []

    # ------------------------------------------------------------------

    def _run_pipeline(self):
        from DICOM_converter.converter import dicom_series_to_nifti, dicom_pet_series_to_nifti
        from DICOM_converter.suv_utils import reconstruct_suv_nifti
        from DICOM_converter.resample_utils import (
            upsample_pet_to_ct, downsample_ct_to_pet, clip_ct, clip_suv
        )
        from DICOM_converter.dicom_utils import get_dcm_files, sort_by_instance_number

        os.makedirs(self.out_dir, exist_ok=True)

        # ── Step 1: Detect CT and PET series ──────────────────────────
        self._log("Scanning for DICOM series...")
        ct_dir, pet_dir = self._detect_series(self.dcm_root)

        # Flat mixed directory: same folder holds both modalities
        if ct_dir and pet_dir and ct_dir == pet_dir:
            self._log("  Flat mixed directory — splitting by modality & series...")
            ct_dir, pet_dir = self._split_series(ct_dir)

        if ct_dir is None and pet_dir is None:
            raise RuntimeError(
                "No CT or PET DICOM series found in the selected folder.\n"
                "Make sure the folder contains .dcm files, "
                "or subdirectories named with 'CT' / 'PET'."
            )

        pid = self.pid_str
        ct_nii      = os.path.join(self.out_dir, f"{pid}_CT.nii.gz")
        pet_nii     = os.path.join(self.out_dir, f"{pid}_PET.nii.gz")
        suv_nii     = os.path.join(self.out_dir, f"{pid}_SUV.nii.gz")
        suv_interp  = os.path.join(self.out_dir, f"{pid}_SUVinterp.nii.gz")
        ct_interp   = os.path.join(self.out_dir, f"{pid}_CTinterp.nii.gz")

        # ── Step 2: Convert CT ─────────────────────────────────────────
        if ct_dir:
            self._log(f"Converting CT series: {ct_dir}")
            dicom_series_to_nifti(ct_dir, ct_nii)
            self._log(f"  → {ct_nii}")
        else:
            self._log("No CT series found — skipping CT conversion.")
            ct_nii = ""

        # ── Step 3: Convert PET ────────────────────────────────────────
        if pet_dir:
            self._log(f"Converting PET series: {pet_dir}")
            dicom_pet_series_to_nifti(pet_dir, pet_nii)
            self._log(f"  → {pet_nii}")
        else:
            self._log("No PET series found — skipping PET conversion.")
            pet_nii = ""

        final_pet_path = pet_nii
        final_ct_path  = ct_nii

        # ── Step 4: SUV reconstruction ─────────────────────────────────
        if self.do_suv and pet_dir and pet_nii:
            self._log("Reconstructing SUV...")
            dcm_files = sort_by_instance_number(get_dcm_files(pet_dir))
            _, estimated = reconstruct_suv_nifti(pet_nii, dcm_files[0], suv_nii)
            if estimated:
                self._log("  [warning] Radiopharmaceutical metadata missing — "
                          "SUV computed with fallback values.")
            self._log(f"  → {suv_nii}")
            final_pet_path = suv_nii

        # ── Step 5: Resample ──────────────────────────────────────────
        if self.do_resample and final_ct_path and final_pet_path:
            if self.resample_mode == "pet":
                # Downsample CT → PET grid
                self._log("Resampling CT to PET grid...")
                downsample_ct_to_pet(final_ct_path, final_pet_path, ct_interp)
                clip_ct(ct_interp)
                clip_suv(final_pet_path)
                self._log(f"  → {ct_interp}")
                final_ct_path = ct_interp
            else:
                # Upsample PET/SUV → CT grid (default)
                self._log("Resampling PET/SUV to CT grid...")
                upsample_pet_to_ct(final_pet_path, final_ct_path, suv_interp)
                clip_ct(final_ct_path)
                clip_suv(suv_interp)
                self._log(f"  → {suv_interp}")
                final_pet_path = suv_interp

        self._log("Conversion complete.")
        self.sig_finished.emit(final_ct_path, final_pet_path)

    # ------------------------------------------------------------------
    # Series detection
    # ------------------------------------------------------------------

    def _detect_series(self, root: str):
        """
        Return (ct_dir, pet_dir) by searching root recursively.

        Strategy:
        1. Subdirectory name contains "CT" → CT candidate; "PET" → PET candidate.
           WB (whole-body) subdirs ranked first; alphabetical tie-break.
        2. For unresolved slots, read Modality tag from files in each subdir.
        3. If a single flat directory holds both modalities, return (root, root)
           so the caller can call _split_series() to separate them.
        """
        ct_dir = pet_dir = None

        # Collect all leaf directories containing .dcm files, sorted for
        # deterministic behaviour across OS/filesystem orderings.
        dcm_dirs = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            if any(f.lower().endswith(".dcm") for f in filenames):
                dcm_dirs.append(dirpath)
        dcm_dirs.sort()

        if not dcm_dirs:
            return None, None

        self._log(f"  Found {len(dcm_dirs)} folder(s) with DICOM files.")

        def _wb_key(d):
            return (0 if "WB" in os.path.basename(d).upper() else 1, d)

        # Pass 1: match by directory name
        ct_candidates  = sorted(
            [d for d in dcm_dirs if "CT"  in os.path.basename(d).upper()], key=_wb_key
        )
        pet_candidates = sorted(
            [d for d in dcm_dirs if "PET" in os.path.basename(d).upper()], key=_wb_key
        )

        if ct_candidates:
            ct_dir = ct_candidates[0]
            self._log(f"  CT series (by name): {os.path.basename(ct_dir)}")
            if len(ct_candidates) > 1:
                self._log(f"  Skipping CT folders: "
                          f"{', '.join(os.path.basename(d) for d in ct_candidates[1:])}")

        if pet_candidates:
            pet_dir = pet_candidates[0]
            self._log(f"  PET series (by name): {os.path.basename(pet_dir)}")
            if len(pet_candidates) > 1:
                self._log(f"  Skipping PET folders: "
                          f"{', '.join(os.path.basename(d) for d in pet_candidates[1:])}")

        if ct_dir and pet_dir:
            return ct_dir, pet_dir

        # Pass 2: match by Modality tag for unresolved slots
        import pydicom
        for d in dcm_dirs:
            if d == ct_dir or d == pet_dir:
                continue
            dcm_files = sorted(
                f for f in os.listdir(d) if f.lower().endswith(".dcm")
            )
            found_ct = found_pet = False
            for fname in dcm_files:
                if found_ct and found_pet:
                    break
                try:
                    ds = pydicom.dcmread(
                        os.path.join(d, fname), stop_before_pixels=True
                    )
                    modality = getattr(ds, "Modality", "").upper()
                    if not found_ct and modality == "CT":
                        found_ct = True
                    if not found_pet and modality in ("PT", "PET"):
                        found_pet = True
                except Exception:
                    pass

            if found_ct and ct_dir is None:
                ct_dir = d
                self._log(f"  CT series (by Modality tag): {os.path.basename(d)}")
            if found_pet and pet_dir is None:
                pet_dir = d
                self._log(f"  PET series (by Modality tag): {os.path.basename(d)}")

            if ct_dir and pet_dir:
                return ct_dir, pet_dir

        # Pass 3: flat mixed dir — check if assigned dir also holds the other modality
        if ct_dir is not None and pet_dir is None:
            for fname in sorted(f for f in os.listdir(ct_dir) if f.lower().endswith(".dcm")):
                try:
                    ds = pydicom.dcmread(os.path.join(ct_dir, fname), stop_before_pixels=True)
                    if getattr(ds, "Modality", "").upper() in ("PT", "PET"):
                        pet_dir = ct_dir
                        self._log("  PET files found in same flat directory as CT.")
                        break
                except Exception:
                    pass

        elif pet_dir is not None and ct_dir is None:
            for fname in sorted(f for f in os.listdir(pet_dir) if f.lower().endswith(".dcm")):
                try:
                    ds = pydicom.dcmread(os.path.join(pet_dir, fname), stop_before_pixels=True)
                    if getattr(ds, "Modality", "").upper() == "CT":
                        ct_dir = pet_dir
                        self._log("  CT files found in same flat directory as PET.")
                        break
                except Exception:
                    pass

        return ct_dir, pet_dir

    # ------------------------------------------------------------------
    # Flat-directory splitter
    # ------------------------------------------------------------------

    def _split_series(self, mixed_dir: str):
        """
        Group .dcm files in mixed_dir by (Modality, SeriesInstanceUID),
        pick the best CT and PET series (WB preferred, then most slices),
        and symlink them into temporary subdirectories inside self.out_dir.

        Returns (ct_tmp_dir, pet_tmp_dir) — either may be None.
        """
        import pydicom
        import tempfile

        ct_by_series:  dict = {}   # series_uid -> [path, ...]
        pet_by_series: dict = {}

        for fname in sorted(f for f in os.listdir(mixed_dir) if f.lower().endswith(".dcm")):
            fpath = os.path.join(mixed_dir, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                modality  = getattr(ds, "Modality", "UNKNOWN").upper()
                series_uid = str(getattr(ds, "SeriesInstanceUID", fname))
                if modality == "CT":
                    ct_by_series.setdefault(series_uid, []).append(fpath)
                elif modality in ("PT", "PET"):
                    pet_by_series.setdefault(series_uid, []).append(fpath)
            except Exception:
                pass

        def _pick_best(series_dict):
            if not series_dict:
                return None
            def _rank(item):
                uid, files = item
                try:
                    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
                    desc = getattr(ds, "SeriesDescription", "").upper()
                    return (0 if "WB" in desc else 1, -len(files))
                except Exception:
                    return (1, -len(files))
            ranked = sorted(series_dict.items(), key=_rank)
            best_uid, best_files = ranked[0]
            if len(ranked) > 1:
                self._log(f"  Skipping {len(ranked) - 1} other series.")
            return best_files

        ct_files  = _pick_best(ct_by_series)
        pet_files = _pick_best(pet_by_series)

        ct_tmp = pet_tmp = None

        if ct_files:
            ct_tmp = tempfile.mkdtemp(prefix="petct_ct_", dir=self.out_dir)
            self._tmp_dirs.append(ct_tmp)
            for p in ct_files:
                os.symlink(p, os.path.join(ct_tmp, os.path.basename(p)))
            self._log(f"  CT: {len(ct_files)} slices isolated.")

        if pet_files:
            pet_tmp = tempfile.mkdtemp(prefix="petct_pet_", dir=self.out_dir)
            self._tmp_dirs.append(pet_tmp)
            for p in pet_files:
                os.symlink(p, os.path.join(pet_tmp, os.path.basename(p)))
            self._log(f"  PET: {len(pet_files)} slices isolated.")

        return ct_tmp, pet_tmp
