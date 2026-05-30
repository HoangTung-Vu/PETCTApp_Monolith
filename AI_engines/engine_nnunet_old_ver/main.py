"""nnUNet Engine Backend — FastAPI server.

One model is active at a time, served by a pool of N concurrent engine workers
(N derived from free VRAM). Requests beyond N queue. Calling an inference
endpoint for a non-active model transparently switches the active model.
"""

import gzip
import io
import logging
import os
import threading
import traceback
import uuid
from typing import List

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.pool import EnginePool, ModelBusyError

# ──── Logging (all levels via ENGINE_LOG_LEVEL, default INFO) ────
logging.basicConfig(
    level=os.getenv("ENGINE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("engine.api")

app = FastAPI(title="nnUNet Engine Old Ver", version="0.2.0")

MODELS = [
    "nnUNetTrainer_150epochs__nnUNetPlans__3d_fullres",
    "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres",
]

# Default active model: from .env (client uses this one), else the DicewBCELoss variant.
DEFAULT_MODEL = os.getenv("ENGINE_NNUNET_MODEL", MODELS[1])

# The pool owns all engines + the inference executor for the active model.
POOL = EnginePool(models=MODELS, default_model=DEFAULT_MODEL, device="auto")

# ──── Async job registry (polling API) ────
# Each job: {status: queued|running|done|error, stage, progress(0-100), detail, result(bytes)}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.on_event("startup")
def on_startup():
    """Load the default model into the pool (estimate VRAM + spawn N workers)."""
    logger.info("Starting nnUNet engine | default model=%s | device=%s", DEFAULT_MODEL, POOL.device)
    try:
        POOL.ensure_model(DEFAULT_MODEL)
        logger.info("Startup complete: %s", POOL.status())
    except Exception:
        logger.exception("Failed to initialize default model '%s' at startup", DEFAULT_MODEL)


# ──── NIfTI (de)serialization ────

def _load_nifti_from_bytes(data: bytes) -> nib.Nifti1Image:
    """Deserialize NIfTI bytes (gzipped or raw) into a nibabel Nifti1Image."""
    # Detect gzip via magic bytes (1f 8b) instead of filename — filename is unreliable.
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def _load_nifti_from_upload(file: UploadFile) -> nib.Nifti1Image:
    """Read an uploaded NIfTI file (gzipped or raw) into a nibabel Nifti1Image."""
    return _load_nifti_from_bytes(file.file.read())


def _nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to gzipped .nii.gz bytes.

    The lesion mask is a sparse uint8 full-volume, so gzip shrinks it ~1000x.
    The client auto-detects the gzip magic bytes and decompresses on receipt.
    """
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return gzip.compress(bio.getvalue(), compresslevel=6)


def _npz_response(prob: np.ndarray, affine: np.ndarray) -> Response:
    """Pack a probability array + affine into a .npz response (no compression)."""
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
    """Background worker (runs in the pool executor on a checked-out engine)."""
    engine = POOL.checkout()
    try:
        _set_job(job_id, status="running", stage="preprocessing", progress=0)
        images = [_load_nifti_from_bytes(b) for b in image_bytes]

        def _cb(done: int, total: int):
            pct = int(done / total * 100) if total else 0
            _set_job(job_id, status="running", stage="inference", progress=pct)

        result = engine.run(images, progress_callback=_cb)
        _set_job(job_id, stage="postprocessing", progress=99)
        mask_bytes = _nifti_to_bytes(result)
        _set_job(job_id, status="done", stage="done", progress=100, result=mask_bytes)
        logger.info("Job %s done [model=%s]", job_id, model_name)
    except Exception as e:
        logger.exception("Job %s failed [model=%s]", job_id, model_name)
        _set_job(job_id, status="error", detail=str(e))
    finally:
        POOL.checkin(engine)


# ──── Health + model management ────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """List available models plus the currently active one and worker count."""
    return {"available": POOL.available_models(), **POOL.status()}


class SetModelRequest(BaseModel):
    # Silence pydantic v2's protected 'model_' namespace warning for this field.
    model_config = {"protected_namespaces": ()}
    model_name: str


@app.post("/models/active")
def set_active_model(req: SetModelRequest):
    """Switch the active model: tear down old engines, re-estimate VRAM, respawn N."""
    try:
        POOL.ensure_model(req.model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    logger.info("Active model set to '%s' via API", req.model_name)
    return POOL.status()


# ──── Per-model inference endpoints ────

def create_endpoints_for_model(model_name: str):

    @app.post(f"/{model_name}/run", tags=[model_name])
    def run(files: List[UploadFile] = File(...)):
        """Run segmentation on uploaded NIfTI files. Returns mask as .nii.gz."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]

            def _task():
                engine = POOL.checkout()
                try:
                    return engine.run(images)
                finally:
                    POOL.checkin(engine)

            result = POOL.submit(model_name, _task).result()
            return Response(content=_nifti_to_bytes(result), media_type="application/octet-stream",
                            headers={"Content-Disposition": "attachment; filename=mask.nii.gz"})
        except ModelBusyError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/{model_name}/run_nib", tags=[model_name])
    def run_nib(files: List[UploadFile] = File(...)):
        """Alias for run — both accept NIfTI uploads."""
        return run(files)

    @app.post(f"/{model_name}/run_prob", tags=[model_name])
    def run_prob(files: List[UploadFile] = File(...), single_channel: bool = Form(True)):
        """Run segmentation and return probability maps as .npz."""
        try:
            images = [_load_nifti_from_upload(f) for f in files]

            def _task():
                engine = POOL.checkout()
                try:
                    return engine.run_prob(images, single_channel=single_channel)
                finally:
                    POOL.checkin(engine)

            prob = POOL.submit(model_name, _task).result()
            return _npz_response(prob, images[0].affine)
        except ModelBusyError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(f"/{model_name}/run_nib_prob", tags=[model_name])
    def run_nib_prob(files: List[UploadFile] = File(...), single_channel: bool = Form(True)):
        """Run segmentation on NIfTI images and return probability maps as .npz."""
        return run_prob(files, single_channel=single_channel)

    # ──── Polling API: submit → progress → result ────

    @app.post(f"/{model_name}/jobs", tags=[model_name])
    def submit_job(files: List[UploadFile] = File(...)):
        """Accept NIfTI uploads and queue an async inference job. Returns a job_id.

        The upload is fully read here, so a 200 response means the client's
        upload is complete; inference then runs in the background and is polled
        via the progress/result endpoints below. Submitting for a non-active
        model transparently switches the active model first.
        """
        image_bytes = [f.file.read() for f in files]
        job_id = uuid.uuid4().hex
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "stage": "queued",
                             "progress": 0, "detail": None, "result": None}
        try:
            POOL.submit(model_name, _run_job, job_id, model_name, image_bytes)
        except ModelBusyError as e:
            with _jobs_lock:
                _jobs.pop(job_id, None)  # don't leave an orphan "queued" job
            raise HTTPException(status_code=409, detail=str(e))
        logger.info("Queued job %s [model=%s]", job_id, model_name)
        return {"job_id": job_id}

    @app.get(f"/{model_name}/jobs/{{job_id}}/progress", tags=[model_name])
    def job_progress(job_id: str):
        """Poll job status/progress. Returns status, stage, progress(0-100), detail."""
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            return {k: job[k] for k in ("status", "stage", "progress", "detail")}

    @app.get(f"/{model_name}/jobs/{{job_id}}/result", tags=[model_name])
    def job_result(job_id: str):
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
