"""
DICOM anonymization utilities.

Replaces patient-identifying tags with a numeric PID string.
"""

import csv
import os
from typing import Dict, Optional

import pydicom

# Tags to blank out (set to empty string)
_BLANK_TAGS = [
    "InstitutionName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "PatientBirthDate",
]

# Tags to replace with PID string
_PID_TAGS = ["PatientName", "PatientID"]


def anonymize_dicom_file(
    src_path: str, dst_path: str, pid_str: str
) -> None:
    """
    Read a single DICOM file, replace PHI tags, and save to dst_path.

    Args:
        src_path: Source .dcm file path.
        dst_path: Destination .dcm file path (will be overwritten).
        pid_str: The anonymous ID to write into PatientName / PatientID.
    """
    ds = pydicom.dcmread(src_path)

    for tag in _BLANK_TAGS:
        if hasattr(ds, tag):
            setattr(ds, tag, "")

    for tag in _PID_TAGS:
        if hasattr(ds, tag):
            setattr(ds, tag, pid_str)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ds.save_as(dst_path)


def anonymize_dicom_series(
    src_scan_dir: str, dst_scan_dir: str, pid_str: str, skip_suffix: str = "(2).dcm"
) -> None:
    """
    Anonymize all DICOM files in a series directory.

    Directory structure expected (and mirrored to dst):
        src_scan_dir/<PatientName>/<ExamDate>/<SeriesName>/*.dcm

    Args:
        src_scan_dir: Root source directory containing patient sub-folders.
        dst_scan_dir: Root destination directory.
        pid_str: Anonymous patient ID (e.g. "0042").
        skip_suffix: Files ending with this suffix are skipped (duplicate detection).
    """
    for patient_folder in os.listdir(src_scan_dir):
        pid_str_mapped = pid_str  # caller is responsible for mapping
        patient_src = os.path.join(src_scan_dir, patient_folder)
        patient_dst = os.path.join(dst_scan_dir, pid_str_mapped)

        if not os.path.isdir(patient_src):
            continue

        for exam_folder in os.listdir(patient_src):
            exam_src = os.path.join(patient_src, exam_folder)
            exam_dst = os.path.join(patient_dst, exam_folder)

            if not os.path.isdir(exam_src):
                continue

            for series_folder in os.listdir(exam_src):
                if "CT" not in series_folder and "PET" not in series_folder:
                    continue
                series_src = os.path.join(exam_src, series_folder)
                series_dst = os.path.join(exam_dst, series_folder)
                os.makedirs(series_dst, exist_ok=True)

                for dcm_filename in os.listdir(series_src):
                    if dcm_filename.endswith(skip_suffix):
                        continue
                    anonymize_dicom_file(
                        os.path.join(series_src, dcm_filename),
                        os.path.join(series_dst, dcm_filename),
                        pid_str_mapped,
                    )


def create_pid_csv(src_dir: str, csv_path: str, pid_start: int = 0) -> Dict[str, str]:
    """
    Scan src_dir (one sub-folder per patient), read PatientName from the first
    DICOM file in each folder, and write a CSV mapping sequential PID → PatientName.

    Returns:
        dict mapping patient_folder_name → zero-padded PID string.
    """
    mapping: Dict[str, str] = {}
    pid = pid_start

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PID", "PatientName"])

        for folder_name in sorted(os.listdir(src_dir)):
            folder_path = os.path.join(src_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            dcm_files = sorted(os.listdir(folder_path))
            if not dcm_files:
                continue

            ds = pydicom.dcmread(
                os.path.join(folder_path, dcm_files[0]), stop_before_pixels=True
            )
            patient_name = (
                str(ds.PatientName).replace("^^^^", "").strip()
                if hasattr(ds, "PatientName")
                else folder_name
            )
            pid += 1
            pid_str = str(pid).zfill(4)
            mapping[folder_name] = pid_str
            writer.writerow([pid_str, patient_name])

    return mapping


def load_pid_mapping(csv_path: str) -> Dict[str, str]:
    """
    Load a CSV produced by create_pid_csv and return a dict
    mapping PatientName → PID string.
    """
    mapping: Dict[str, str] = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) >= 2:
                pid_str = str(row[0]).zfill(4)
                patient_name = row[1]
                mapping[patient_name] = pid_str
    return mapping
