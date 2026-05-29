"""nnUNet Engine Backend — FastAPI server with 8 segmentation endpoints."""

import gzip
import io
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from src.engine import NNUNetEngine

app = FastAPI(title="nnUNet Engine Old Ver", version="0.1.0")

# Singleton engine cache — model weights live here for the process lifetime.
_engines: dict[str, NNUNetEngine] = {}

# ──── Async job registry (polling API) ────
# Each job: {status: queued|running|done|error, stage, progress(0-100), detail, result(bytes)}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
# One inference at a time — the GPU/predictor is shared, so jobs queue here. This
# also makes setting predictor._progress_callback race-free (single active job).
_executor = ThreadPoolExecutor(max_workers=1)

MODELS = [
    "nnUNetTrainer_150epochs__nnUNetPlans__3d_fullres",
    "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"
]

# Models to warm-load into VRAM at startup. Edit env to skip unused ones and save VRAM.
import os as _os
_preload = _os.getenv("PRELOAD_MODELS")
PRELOAD_MODELS = [m.strip() for m in _preload.split(",")] if _preload else MODELS


def get_engine(model_name: str) -> NNUNetEngine:
    global _engines
    if model_name not in _engines:
        _engines[model_name] = NNUNetEngine(model_name=model_name, device="auto")
    return _engines[model_name]


@app.on_event("startup")
def preload_models():
    """Warm-load model weights to VRAM so first request is not paying init cost."""
    print(f"[startup] Pre-loading {len(PRELOAD_MODELS)} model(s) into VRAM...")
    for model_name in PRELOAD_MODELS:
        try:
            engine = get_engine(model_name)
            engine._init_predictor()
            print(f"[startup] OK loaded: {model_name}")
        except Exception as e:
            traceback.print_exc()
            print(f"[startup] FAILED to load {model_name}: {e}")


def _load_nifti_from_bytes(data: bytes) -> nib.Nifti1Image:
    """Deserialize NIfTI bytes (gzipped or raw) into a nibabel Nifti1Image."""
    # Detect gzip via magic bytes (1f 8b) instead of filename — filename is unreliable
    # and the failing try/except on every request was wasted CPU.
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def _load_nifti_from_upload(file: UploadFile) -> nib.Nifti1Image:
    """Read an uploaded NIfTI upload (gzipped or raw) into a nibabel Nifti1Image."""
    return _load_nifti_from_bytes(file.file.read())


def _nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to .nii.gz bytes."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def _npz_response(prob: np.ndarray, affine: np.ndarray) -> Response:
    """Pack a probability array + affine into a .npz response.

    Uses np.savez (no compression) — on localhost transfer is ~100x faster than
    the zlib pass np.savez_compressed would do on 200-400MB of float32.
    """
    bio = io.BytesIO()
    np.savez(bio, prob=prob, affine=affine)
    return Response(content=bio.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=result.npz"})


# ──── Async job helpers ────

def _set_job(job_id: str, **kw):
    """Thread-safe update of a job's state."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)


def _run_job(job_id: str, model_name: str, image_bytes: List[bytes]):
    """Background worker: run inference and stream progress into the job registry."""
    try:
        _set_job(job_id, status="running", stage="preprocessing", progress=0)
        images = [_load_nifti_from_bytes(b) for b in image_bytes]
        engine = get_engine(model_name)

        def _cb(done: int, total: int):
            pct = int(done / total * 100) if total else 0
            _set_job(job_id, status="running", stage="inference", progress=pct)

        result = engine.run(images, progress_callback=_cb)
        _set_job(job_id, stage="postprocessing", progress=99)
        mask_bytes = _nifti_to_bytes(result)
        _set_job(job_id, status="done", stage="done", progress=100, result=mask_bytes)
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status="error", detail=str(e))


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

    # ──── Polling API: submit → progress → result ────

    @app.post(f"/{model_name}/jobs", tags=[model_name])
    async def submit_job(files: List[UploadFile] = File(...)):
        """Accept NIfTI uploads and queue an async inference job. Returns a job_id.

        The upload is fully read here, so a 200 response means the client's upload
        is complete; inference then runs in the background and is polled via the
        progress/result endpoints below.
        """
        image_bytes = [await f.read() for f in files]
        job_id = uuid.uuid4().hex
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "stage": "queued",
                             "progress": 0, "detail": None, "result": None}
        _executor.submit(_run_job, job_id, model_name, image_bytes)
        return {"job_id": job_id}

    @app.get(f"/{model_name}/jobs/{{job_id}}/progress", tags=[model_name])
    async def job_progress(job_id: str):
        """Poll job status/progress. Returns status, stage, progress(0-100), detail."""
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            return {k: job[k] for k in ("status", "stage", "progress", "detail")}

    @app.get(f"/{model_name}/jobs/{{job_id}}/result", tags=[model_name])
    async def job_result(job_id: str):
        """Fetch the finished mask as .nii.gz. 409 if not ready, 500 if the job errored."""
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            status = job["status"]
            detail = job["detail"]
            result = job["result"]
            if status == "done":
                _jobs.pop(job_id, None)  # free memory once delivered
        if status == "error":
            raise HTTPException(status_code=500, detail=detail or "Inference failed")
        if status != "done":
            raise HTTPException(status_code=409, detail="Job not finished")
        return Response(content=result, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})

for m in MODELS:
    create_endpoints_for_model(m)


if __name__ == "__main__":
    import uvicorn
    from src import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
