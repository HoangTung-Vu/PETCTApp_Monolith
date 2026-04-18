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
        parent=None,
    ):
        """
        Args:
            dcm_root:     Root folder to search for CT / PET DICOM series.
            out_dir:      Destination folder for .nii.gz output files.
            pid_str:      Patient ID prefix used in output filenames.
            do_suv:       Compute SUV from raw PET counts.
            do_resample:  Upsample PET/SUV to CT grid after conversion.
        """
        super().__init__(parent)
        self.dcm_root   = dcm_root
        self.out_dir    = out_dir
        self.pid_str    = pid_str
        self.do_suv     = do_suv
        self.do_resample = do_resample

    # ------------------------------------------------------------------

    def run(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.sig_error.emit(str(exc))

    def _log(self, msg: str):
        print(f"[DicomConversionWorker] {msg}")
        self.sig_log.emit(msg)

    # ------------------------------------------------------------------

    def _run_pipeline(self):
        from DICOM_converter.converter import dicom_series_to_nifti, dicom_pet_series_to_nifti
        from DICOM_converter.suv_utils import reconstruct_suv_nifti
        from DICOM_converter.resample_utils import upsample_pet_to_ct, clip_ct, clip_suv
        from DICOM_converter.dicom_utils import get_dcm_files, sort_by_instance_number

        os.makedirs(self.out_dir, exist_ok=True)

        # ── Step 1: Detect CT and PET series ──────────────────────────
        self._log("Scanning for DICOM series...")
        ct_dir, pet_dir = self._detect_series(self.dcm_root)

        if ct_dir is None and pet_dir is None:
            raise RuntimeError(
                "No CT or PET DICOM series found in the selected folder.\n"
                "Make sure the folder contains .dcm files, "
                "or subdirectories named with 'CT' / 'PET'."
            )

        pid = self.pid_str
        ct_nii  = os.path.join(self.out_dir, f"{pid}_CT.nii.gz")
        pet_nii = os.path.join(self.out_dir, f"{pid}_PET.nii.gz")
        suv_nii = os.path.join(self.out_dir, f"{pid}_SUV.nii.gz")
        suv_interp = os.path.join(self.out_dir, f"{pid}_SUVinterp.nii.gz")

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

        # ── Step 5: Resample SUV → CT grid ────────────────────────────
        if self.do_resample and ct_nii and final_pet_path:
            self._log("Resampling PET/SUV to CT grid...")
            upsample_pet_to_ct(final_pet_path, ct_nii, suv_interp)
            clip_ct(ct_nii)
            clip_suv(suv_interp)
            self._log(f"  → {suv_interp}")
            final_pet_path = suv_interp

        self._log("Conversion complete.")
        self.sig_finished.emit(ct_nii, final_pet_path)

    # ------------------------------------------------------------------
    # Series detection
    # ------------------------------------------------------------------

    def _detect_series(self, root: str):
        """
        Return (ct_dir, pet_dir) by searching root recursively.

        Strategy (in priority order):
        1. Subdirectory whose name contains "CT" → CT series
           Subdirectory whose name contains "PET" → PET series
        2. If none found by name, read the Modality tag from the first
           .dcm file in each subdirectory.
        3. If root itself contains .dcm files, try to split by Modality.

        Directories are sorted alphabetically so that selection is deterministic
        when multiple CT or PET series exist (e.g. WB + HEADNECK).
        The first match in alphabetical order wins; a warning is emitted when
        more than one candidate exists.
        """
        ct_dir = pet_dir = None

        # Collect all leaf directories that contain .dcm files, sorted for
        # deterministic behaviour across OS/filesystem orderings.
        dcm_dirs = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()  # ensure os.walk itself descends in sorted order
            if any(f.lower().endswith(".dcm") for f in filenames):
                dcm_dirs.append(dirpath)
        dcm_dirs.sort()

        if not dcm_dirs:
            return None, None

        self._log(f"  Found {len(dcm_dirs)} folder(s) with DICOM files.")

        # Pass 1: match by directory name — WB (whole body) series preferred.
        # Folders whose name contains "WB" are ranked first; among ties, alphabetical.
        def _wb_key(d):
            return (0 if "WB" in os.path.basename(d).upper() else 1, d)

        ct_candidates = []
        pet_candidates = []
        for d in dcm_dirs:
            name = os.path.basename(d).upper()
            if "CT" in name:
                ct_candidates.append(d)
            if "PET" in name:
                pet_candidates.append(d)

        ct_candidates.sort(key=_wb_key)
        pet_candidates.sort(key=_wb_key)

        if ct_candidates:
            ct_dir = ct_candidates[0]
            self._log(f"  CT series (by name): {os.path.basename(ct_dir)}")
            skipped = [os.path.basename(d) for d in ct_candidates[1:]]
            if skipped:
                self._log(f"  Skipping other CT folders: {', '.join(skipped)}")

        if pet_candidates:
            pet_dir = pet_candidates[0]
            self._log(f"  PET series (by name): {os.path.basename(pet_dir)}")
            skipped = [os.path.basename(d) for d in pet_candidates[1:]]
            if skipped:
                self._log(f"  Skipping other PET folders: {', '.join(skipped)}")

        # Pass 2: match by Modality tag for unresolved slots
        if ct_dir is None or pet_dir is None:
            import pydicom
            for d in dcm_dirs:
                if d == ct_dir or d == pet_dir:
                    continue
                dcm_files = sorted(
                    f for f in os.listdir(d) if f.lower().endswith(".dcm")
                )
                if not dcm_files:
                    continue
                try:
                    ds = pydicom.dcmread(
                        os.path.join(d, dcm_files[0]), stop_before_pixels=True
                    )
                    modality = getattr(ds, "Modality", "").upper()
                    if ct_dir is None and modality == "CT":
                        ct_dir = d
                        self._log(f"  CT series detected (by Modality tag): {d}")
                    elif pet_dir is None and modality in ("PT", "PET"):
                        pet_dir = d
                        self._log(f"  PET series detected (by Modality tag): {d}")
                except Exception:
                    pass

        return ct_dir, pet_dir
