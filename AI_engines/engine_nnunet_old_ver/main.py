"""nnUNet Engine Backend — FastAPI server with 8 segmentation endpoints."""

import io
import traceback
from typing import List

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from src.engine import NNUNetEngine

app = FastAPI(title="nnUNet Engine Old Ver", version="0.1.0")

# Lazy-initialized engines (heavy model loading on first request)
_engines: dict[str, NNUNetEngine] = {}

MODELS = [
    "nnUNetTrainer_150epochs__nnUNetPlans__3d_fullres",
    "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"
]


def get_engine(model_name: str) -> NNUNetEngine:
    global _engines
    if model_name not in _engines:
        _engines[model_name] = NNUNetEngine(model_name=model_name, device="auto")
    return _engines[model_name]


def _load_nifti_from_upload(file: UploadFile) -> nib.Nifti1Image:
    """Read an uploaded .nii.gz file into a nibabel Nifti1Image."""
    data = file.file.read()
    # Decompress if it's a .nii.gz
    if file.filename and file.filename.endswith(".gz"):
        import gzip
        try:
            data = gzip.decompress(data)
        except OSError:
            pass # Maybe not actually gzipped
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def _nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to .nii.gz bytes."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def _npz_response(prob: np.ndarray, affine: np.ndarray) -> Response:
    """Pack a probability array + affine into a .npz response."""
    bio = io.BytesIO()
    np.savez_compressed(bio, prob=prob, affine=affine)
    bio.seek(0)
    return Response(content=bio.read(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=result.npz"})


# ──── Health Check ────

@app.get("/health")
def health():
    return {"status": "ok"}


# ──── Endpoints Helper ────
def create_endpoints_for_model(model_name: str):
    
    @app.post(f"/{model_name}/run", tags=[model_name])
    async def run(files: List[UploadFile] = File(...)):
        """Run segmentation on uploaded NIfTI files. Returns mask as .nii.gz."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]
            engine = get_engine(model_name)
            result = engine.run(images)
            return Response(content=_nifti_to_bytes(result), media_type="application/octet-stream",
                            headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/{model_name}/run_nib", tags=[model_name])
    async def run_nib(files: List[UploadFile] = File(...)):
        """Alias for run — both accept NIfTI uploads."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]
            engine = get_engine(model_name)
            result = engine.run_nib(images)
            return Response(content=_nifti_to_bytes(result), media_type="application/octet-stream",
                            headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/{model_name}/run_prob", tags=[model_name])
    async def run_prob(files: List[UploadFile] = File(...), single_channel: bool = Form(True)):
        """Run segmentation and return probability maps as .npz."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]
            engine = get_engine(model_name)
            prob = engine.run_prob(images, single_channel=single_channel)
            affine = images[0].affine
            return _npz_response(prob, affine)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/{model_name}/run_nib_prob", tags=[model_name])
    async def run_nib_prob(files: List[UploadFile] = File(...), single_channel: bool = Form(True)):
        """Run segmentation on NIfTI images and return probability maps as .npz."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]
            engine = get_engine(model_name)
            prob = engine.run_nib_prob(images, single_channel=single_channel)
            affine = images[0].affine
            return _npz_response(prob, affine)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

for m in MODELS:
    create_endpoints_for_model(m)


if __name__ == "__main__":
    import uvicorn
    from src import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
