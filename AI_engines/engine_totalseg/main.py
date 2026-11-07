"""TotalSegmentator Engine Backend — FastAPI server.

Runs TotalSegmentator task "total" at default settings (all 5 part models, 1.5 mm,
117 classes) and returns the multilabel mask exactly as
``TotalSegmentator -i ct.nii.gz -o organ.nii.gz --ml`` writes it: uint8, CT grid,
label names in the NIfTI header.

The HTTP contract mirrors the nnU-Net engine, so the app's polling client works
against the base URL ``http://<host>:8105/total``::

    POST /total/jobs  ->  GET /total/jobs/{id}/progress  ->  GET /total/jobs/{id}/result

Upload exactly one CT (multipart field ``files``). One inference runs at a time;
further requests queue on a single-worker executor.

GPU host:
    docker run -d --name totalseg-engine --gpus all --ipc=host -p 8105:8105 \
        hoangtungmvu/totalseg-engine:2.18.0
"""

import logging
import os
import signal
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src import TASK
from src.engine import TotalSegEngine

# nnU-Net (inside TotalSegmentator) opens multiprocessing Pools with the default start
# method (fork), e.g. a 12-worker Pool just to list the input files of every part model.
# A forked worker inherits uvicorn's SIGTERM handler, which only sets a flag, so
# Pool.terminate() cannot kill a worker that was still starting up (slow when this
# process holds GBs and a CUDA context) and the inference thread waits in join()
# forever: the job stays at "part 2/5" with an idle GPU. Forked children get the
# default SIGTERM action back. (Making spawn the default start method also ends the
# hang, but then clients polling /progress got "connection reset" after every part.)
os.register_at_fork(after_in_child=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL))

# ──── Logging (all levels via ENGINE_LOG_LEVEL, default INFO) ────
logging.basicConfig(
    level=os.getenv("ENGINE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("engine.api")

ENGINE = TotalSegEngine()

# One TotalSegmentator run at a time: it loads its 5 models per call and needs the
# whole GPU / a lot of RAM for a full-body CT. Extra requests wait in this queue.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="totalseg")
_inflight = 0  # submitted but not finished (running or queued)
_inflight_lock = threading.Lock()

# ──── Async job registry (polling API) ────
# Each job: {status: queued|running|done|error, stage, progress(0-100), part, detail, result(bytes)}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    missing = ENGINE.missing_weights()
    logger.info("Starting TotalSegmentator %s engine | task=%s | device=%s",
                ENGINE.version, TASK, ENGINE.device)
    if missing:
        logger.error("Weights missing for datasets %s — the first run will try to download them", missing)
    yield
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="TotalSegmentator Engine", version="0.1.0", lifespan=lifespan)


def _submit(fn, *args) -> Future:
    """Queue ``fn(*args)`` on the single inference worker, tracking the in-flight count."""
    global _inflight
    with _inflight_lock:
        _inflight += 1
    fut = EXECUTOR.submit(fn, *args)
    fut.add_done_callback(_on_task_done)
    return fut


def _on_task_done(_fut: Future):
    global _inflight
    with _inflight_lock:
        _inflight -= 1


def _read_single_ct(files: List[UploadFile]) -> bytes:
    """Return the bytes of the one uploaded CT; 400 for any other file count."""
    if len(files) != 1:
        raise HTTPException(status_code=400,
                            detail=f"Upload exactly one CT NIfTI file (got {len(files)}).")
    return files[0].file.read()


def _mask_response(mask_bytes: bytes) -> Response:
    return Response(content=mask_bytes, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=organ.nii.gz"})


# ──── Async job helpers ────

def _set_job(job_id: str, **kw):
    """Thread-safe update of a job's state."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)


def _run_job(job_id: str, ct_bytes: bytes):
    """Background worker (runs on the single inference thread)."""
    try:
        _set_job(job_id, status="running", stage="preprocessing", progress=0)

        def _cb(stage: str, pct: int, part: Optional[str]):
            _set_job(job_id, status="running", stage=stage, progress=pct, part=part)

        mask_bytes = ENGINE.run(ct_bytes, progress_callback=_cb)
        _set_job(job_id, status="done", stage="done", progress=100, part=None, result=mask_bytes)
        logger.info("Job %s done (%.1f KB mask)", job_id, len(mask_bytes) / 1e3)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _set_job(job_id, status="error", detail=str(e))


# ──── Health + info ────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """Engine info: the one task served, device, version, weights and queue state."""
    with _inflight_lock:
        inflight = _inflight
    return {
        "available": [TASK],
        "active_model": TASK,
        "device": ENGINE.device,
        "totalsegmentator_version": ENGINE.version,
        "fast": False,
        "num_classes": len(ENGINE.labels()),
        "missing_weights": ENGINE.missing_weights(),
        "inflight": inflight,
    }


@app.get(f"/{TASK}/labels", tags=[TASK])
def labels():
    """Label id -> structure name (same table as in the output file header)."""
    return ENGINE.labels()


# ──── Synchronous inference ────

@app.post(f"/{TASK}/run", tags=[TASK])
def run(files: List[UploadFile] = File(...)):
    """Segment one uploaded CT and return the multilabel mask (.nii.gz)."""
    ct_bytes = _read_single_ct(files)
    try:
        return _mask_response(_submit(ENGINE.run, ct_bytes).result())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"/{TASK}/run_nib", tags=[TASK])
def run_nib(files: List[UploadFile] = File(...)):
    """Alias for run (both accept NIfTI uploads)."""
    return run(files)


# ──── Polling API: submit → progress → result ────

@app.post(f"/{TASK}/jobs", tags=[TASK])
def submit_job(files: List[UploadFile] = File(...)):
    """Accept one CT upload and queue an inference job. Returns a job_id.

    The upload is fully read here, so a 200 response means the upload is complete;
    inference then runs in the background and is polled via the endpoints below.
    """
    ct_bytes = _read_single_ct(files)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0,
                         "part": None, "detail": None, "result": None}
    _submit(_run_job, job_id, ct_bytes)
    logger.info("Queued job %s (%.1f MB)", job_id, len(ct_bytes) / 1e6)
    return {"job_id": job_id}


@app.get(f"/{TASK}/jobs/{{job_id}}/progress", tags=[TASK])
def job_progress(job_id: str):
    """Poll job state: status, stage, progress (0-100), part ("k/5" during inference), detail."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {k: job[k] for k in ("status", "stage", "progress", "part", "detail")}


@app.get(f"/{TASK}/jobs/{{job_id}}/result", tags=[TASK])
def job_result(job_id: str):
    """Fetch the finished mask as .nii.gz. 409 if not ready, 500 if the job errored."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        status, detail, result = job["status"], job["detail"], job["result"]
        if status in ("done", "error"):
            _jobs.pop(job_id, None)  # free memory once delivered
    if status == "error":
        raise HTTPException(status_code=500, detail=detail or "Inference failed")
    if status != "done":
        raise HTTPException(status_code=409, detail="Job not finished")
    return _mask_response(result)


if __name__ == "__main__":
    import uvicorn
    from src import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
