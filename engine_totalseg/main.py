"""TotalSegmentator Engine Backend — FastAPI server with 2 segmentation endpoints."""

import io
from typing import List

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from src.engine import TotalSegEngine

app = FastAPI(title="TotalSegmentator Engine", version="0.1.0")

# Lazy-initialized engine
_engine: TotalSegEngine | None = None


def get_engine() -> TotalSegEngine:
    global _engine
    if _engine is None:
        _engine = TotalSegEngine(task="total", fast=True, device="auto")
    return _engine


def _load_nifti_from_upload(file: UploadFile) -> nib.Nifti1Image:
    """Read an uploaded .nii.gz file into a nibabel Nifti1Image."""
    data = file.file.read()
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def _nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to .nii.gz bytes."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


# ──── Health Check ────

@app.get("/health")
def health():
    return {"status": "ok"}


# ──── Endpoints ────
# TotalSegmentator only supports run and run_nib (no probability methods)

@app.post("/run")
async def run(files: List[UploadFile] = File(...)):
    """Run TotalSegmentator on uploaded CT NIfTI file. Returns mask as .nii.gz."""
    try:
        if len(files) != 1:
            raise ValueError("TotalSegmentator expects exactly 1 CT image")
        image = _load_nifti_from_upload(files[0])
        engine = get_engine()
        result = engine.run(image)
        return Response(content=_nifti_to_bytes(result), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_nib")
async def run_nib(files: List[UploadFile] = File(...)):
    """Alias for /run."""
    try:
        if len(files) != 1:
            raise ValueError("TotalSegmentator expects exactly 1 CT image")
        image = _load_nifti_from_upload(files[0])
        engine = get_engine()
        result = engine.run_nib(image)
        return Response(content=_nifti_to_bytes(result), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from src import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
