"""
Utilities for reading, sorting, and organizing raw DICOM files.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

import pydicom


def sort_by_instance_number(dcm_paths: List[str]) -> List[str]:
    """Sort a list of DICOM file paths by InstanceNumber tag."""
    entries = []
    for path in dcm_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        entries.append((path, int(ds.InstanceNumber)))
    entries.sort(key=lambda x: x[1])
    return [e[0] for e in entries]


def read_first_dicom(dcm_dir: str) -> pydicom.Dataset:
    """
    Read and return the first DICOM dataset found in a directory.
    Files are sorted alphabetically before picking the first.
    """
    files = sorted(
        f for f in os.listdir(dcm_dir) if not f.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No files found in {dcm_dir}")
    return pydicom.dcmread(os.path.join(dcm_dir, files[0]), stop_before_pixels=True)


def get_dcm_files(dcm_dir: str) -> List[str]:
    """Return all .dcm file paths in a directory (non-recursive)."""
    return [
        os.path.join(dcm_dir, f)
        for f in os.listdir(dcm_dir)
        if f.lower().endswith(".dcm")
    ]


def get_patient_name(ds: pydicom.Dataset) -> str:
    """Extract and clean PatientName from a DICOM dataset."""
    name = str(ds.PatientName) if hasattr(ds, "PatientName") else "UNKNOWN"
    return name.replace("^^^^", "").strip()


def get_acquisition_date(ds: pydicom.Dataset) -> str:
    """Extract AcquisitionDate from a DICOM dataset."""
    return str(ds.AcquisitionDate) if hasattr(ds, "AcquisitionDate") else "00000000"


def organize_dcm_to_subdir(src_dir: str, dst_dir: str) -> None:
    """
    Organize a flat directory of DICOM files (named *_CT_* / *_PET_*)
    into a structured tree:

        dst_dir/
          <PatientName>/
            <AcquisitionDate>/
              <SeriesName>/
                *.dcm

    Each source folder is treated as one flat collection of DICOM files.
    """
    os.makedirs(dst_dir, exist_ok=True)

    for folder_name in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        dcm_files = sorted(os.listdir(folder_path))
        if not dcm_files:
            continue

        first_ds = pydicom.dcmread(
            os.path.join(folder_path, dcm_files[0]), stop_before_pixels=True
        )
        patient_name = get_patient_name(first_ds)
        acq_date = get_acquisition_date(first_ds)

        for dcm_filename in dcm_files:
            # Determine series name from filename (expects format: SERIES_NAME_<idx>.dcm)
            if "CT" not in dcm_filename and "PET" not in dcm_filename:
                continue
            suffix = "_" + dcm_filename.split("_")[-1]
            series_name = dcm_filename[: -len(suffix)]

            series_dst = os.path.join(dst_dir, patient_name, acq_date, series_name)
            os.makedirs(series_dst, exist_ok=True)
            shutil.copy(os.path.join(folder_path, dcm_filename), series_dst)
