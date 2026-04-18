"""
High-level pipeline: raw DICOM tree → anonymized NIfTI pairs ready for inference.

Expected input tree:
    src_dcm/
      <PatientFolder>/
        <ExamDate>/
          <CTSeriesFolder>/   ← folder name must contain "CT"
            *.dcm
          <PETSeriesFolder>/  ← folder name must contain "PET"
            *.dcm

Output tree:
    dst_nii/
      <PID>/
        <ExamDate>/
          <PID>_<ExamDate>_CT.nii.gz
          <PID>_<ExamDate>_PET.nii.gz
          <PID>_<ExamDate>_SUV.nii.gz
          <PID>_<ExamDate>_CTres.nii.gz      (CT downsampled to PET grid)
          <PID>_<ExamDate>_SUVinterp.nii.gz  (SUV upsampled to CT grid)
"""

import os
from pathlib import Path
from typing import Dict, Optional

from .anonymizer import load_pid_mapping
from .converter import dicom_series_to_nifti, dicom_pet_series_to_nifti
from .suv_utils import reconstruct_suv_nifti
from .resample_utils import downsample_ct_to_pet, upsample_pet_to_ct, clip_ct, clip_suv
from .dicom_utils import get_dcm_files, sort_by_instance_number


def run_full_pipeline(
    src_dcm: str,
    dst_nii: str,
    pid_mapping: Dict[str, str],
    do_suv: bool = True,
    do_resample: bool = True,
) -> None:
    """
    Run the complete DICOM → NIfTI pipeline for an anonymized dataset.

    Args:
        src_dcm: Root of anonymized DICOM tree (PID folders at top level).
        dst_nii: Root of NIfTI output tree.
        pid_mapping: Dict mapping patient_folder_name → PID string.
                     Build with create_pid_csv() or load_pid_mapping().
        do_suv: Whether to compute SUV from raw PET.
        do_resample: Whether to produce CTres and SUVinterp volumes.
    """
    os.makedirs(dst_nii, exist_ok=True)

    for patient_folder in sorted(os.listdir(src_dcm)):
        patient_src = os.path.join(src_dcm, patient_folder)
        if not os.path.isdir(patient_src):
            continue

        pid_str = pid_mapping.get(patient_folder, patient_folder)
        if "Exclude" in patient_folder:
            continue

        exam_folders = sorted(os.listdir(patient_src))
        for exam_folder in exam_folders:
            exam_src = os.path.join(patient_src, exam_folder)
            if not os.path.isdir(exam_src):
                continue

            exam_dst = os.path.join(dst_nii, pid_str, exam_folder)
            os.makedirs(exam_dst, exist_ok=True)

            prefix = f"{pid_str}_{exam_folder}"
            ct_nii   = os.path.join(exam_dst, f"{prefix}_CT.nii.gz")
            pet_nii  = os.path.join(exam_dst, f"{prefix}_PET.nii.gz")
            suv_nii  = os.path.join(exam_dst, f"{prefix}_SUV.nii.gz")
            ctres    = os.path.join(exam_dst, f"{prefix}_CTres.nii.gz")
            suvinterp = os.path.join(exam_dst, f"{prefix}_SUVinterp.nii.gz")

            ct_dcm_dir = pet_dcm_dir = None
            for series_folder in os.listdir(exam_src):
                series_path = os.path.join(exam_src, series_folder)
                if not os.path.isdir(series_path):
                    continue
                if "CT" in series_folder:
                    ct_dcm_dir = series_path
                elif "PET" in series_folder:
                    pet_dcm_dir = series_path

            # --- CT conversion ---
            if ct_dcm_dir and not os.path.exists(ct_nii):
                print(f"[pipeline] Converting CT: {ct_dcm_dir}")
                dicom_series_to_nifti(ct_dcm_dir, ct_nii)

            # --- PET conversion ---
            if pet_dcm_dir and not os.path.exists(pet_nii):
                print(f"[pipeline] Converting PET: {pet_dcm_dir}")
                dicom_pet_series_to_nifti(pet_dcm_dir, pet_nii)

            # --- SUV reconstruction ---
            if do_suv and pet_dcm_dir and os.path.exists(pet_nii) and not os.path.exists(suv_nii):
                dcm_files = sort_by_instance_number(get_dcm_files(pet_dcm_dir))
                print(f"[pipeline] Reconstructing SUV: {pet_nii}")
                _, estimated = reconstruct_suv_nifti(pet_nii, dcm_files[0], suv_nii)
                if estimated:
                    print(f"  [warning] SUV used fallback metadata for {prefix}")

            # --- Resampling ---
            if do_resample and os.path.exists(ct_nii) and os.path.exists(pet_nii):
                if not os.path.exists(ctres):
                    print(f"[pipeline] Downsampling CT → PET grid: {prefix}")
                    downsample_ct_to_pet(ct_nii, pet_nii, ctres, default_value=-1024.0)
                    clip_ct(ctres)

            if do_resample and os.path.exists(suv_nii) and os.path.exists(ct_nii):
                if not os.path.exists(suvinterp):
                    print(f"[pipeline] Upsampling SUV → CT grid: {prefix}")
                    upsample_pet_to_ct(suv_nii, ct_nii, suvinterp)
                    clip_ct(ct_nii)
                    clip_suv(suvinterp)

            print(f"[pipeline] Done: {prefix}")
