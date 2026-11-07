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

Each job runs in its own worker process (src/worker.py, started by src/runner.py),
killed and restarted when it makes no progress. This process never imports torch or
TotalSegmentator, so inference cannot block the API: /health and /progress answer
during a job. src/runner.py explains why.

GPU host:
    docker run -d --name totalseg-engine --gpus all --ipc=host -p 8105:8105 \
        hoangtungmvu/totalseg-engine:2.18.0

Settings (env): ENGINE_STALL_TIMEOUT (s without progress before a restart; default 600
on a GPU, off on a CPU; 0 = off), ENGINE_MAX_ATTEMPTS (default 2), ENGINE_LOG_LEVEL.
"""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src import MAX_ATTEMPTS, TASK, runner, stall_timeout_s
from src.runner import WorkerError

# ──── Logging (all levels via ENGINE_LOG_LEVEL, default INFO) ────
logging.basicConfig(
    level=os.getenv("ENGINE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("engine.api")

# version, device, labels, missing_weights: read from a worker at start-up (lifespan)
INFO: dict = {}
STALL_TIMEOUT_S = 0.0

# One TotalSegmentator run at a time: it loads its 5 models per call and needs the
# whole GPU / a lot of RAM for a full-body CT. Extra requests wait in this queue.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="totalseg")
_inflight = 0  # submitted but not finished (running or queued)
_inflight_lock = threading.Lock()

# ──── Async job registry (polling API) ────
# Each job: {status: queued|running|done|error, stage, progress(0-100), part, detail,
#            result(bytes), finished_at(monotonic s)}
# Finished jobs stay RESULT_TTL_S, so a client whose connection dropped can fetch the
# result (or the error) again.
RESULT_TTL_S = 3600
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global STALL_TIMEOUT_S
    INFO.update(runner.probe_info())  # raises -> the server does not start
    STALL_TIMEOUT_S = stall_timeout_s(INFO["device"])
    logger.info("Starting TotalSegmentator %s engine | task=%s | device=%s | "
                "stall timeout %s | attempts %d", INFO["version"], TASK, INFO["device"],
                f"{STALL_TIMEOUT_S:.0f} s" if STALL_TIMEOUT_S else "off", MAX_ATTEMPTS)
    if INFO["missing_weights"]:
        logger.error("Weights missing for datasets %s — the first run will try to download them",
                     INFO["missing_weights"])
    yield
    runner.kill_current()
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="TotalSegmentator Engine", version="0.2.0", lifespan=lifespan)


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


def _segment(ct_bytes: bytes, on_progress: Optional[runner.ProgressCallback] = None) -> bytes:
    return runner.run_isolated(ct_bytes, on_progress,
                               stall_timeout_s=STALL_TIMEOUT_S, max_attempts=MAX_ATTEMPTS)


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


def _purge_finished_jobs():
    """Drop finished jobs older than RESULT_TTL_S."""
    now = time.monotonic()
    with _jobs_lock:
        expired = [jid for jid, job in _jobs.items()
                   if job["finished_at"] is not None and now - job["finished_at"] > RESULT_TTL_S]
        for jid in expired:
            del _jobs[jid]
    if expired:
        logger.info("Dropped %d finished job(s) older than %d s", len(expired), RESULT_TTL_S)


def _run_job(job_id: str, ct_bytes: bytes):
    """Background worker (runs on the single inference thread)."""
    t0 = time.monotonic()
    try:
        _set_job(job_id, status="running", stage="preprocessing", progress=0)

        def _cb(stage: str, pct: int, part: Optional[str], detail: Optional[str]):
            # A restart note stays in "detail" for the rest of the job.
            extra = {"detail": detail} if detail else {}
            _set_job(job_id, status="running", stage=stage, progress=pct, part=part, **extra)

        mask_bytes = _segment(ct_bytes, _cb)
        _set_job(job_id, status="done", stage="done", progress=100, part=None,
                 result=mask_bytes, finished_at=time.monotonic())
        logger.info("Job %s done in %.0f s (%.1f KB mask)", job_id, time.monotonic() - t0,
                    len(mask_bytes) / 1e3)
    except Exception as e:
        if isinstance(e, WorkerError):
            logger.error("Job %s failed after %.0f s: %s", job_id, time.monotonic() - t0, e)
        else:
            logger.exception("Job %s failed", job_id)
        _set_job(job_id, status="error", detail=str(e), finished_at=time.monotonic())


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
        "device": INFO["device"],
        "totalsegmentator_version": INFO["version"],
        "fast": False,
        "num_classes": len(INFO["labels"]),
        "missing_weights": INFO["missing_weights"],
        "inflight": inflight,
    }


@app.get(f"/{TASK}/labels", tags=[TASK])
def labels():
    """Label id -> structure name (same table as in the output file header)."""
    return INFO["labels"]


# ──── Synchronous inference ────

@app.post(f"/{TASK}/run", tags=[TASK])
def run(files: List[UploadFile] = File(...)):
    """Segment one uploaded CT and return the multilabel mask (.nii.gz)."""
    ct_bytes = _read_single_ct(files)
    try:
        return _mask_response(_submit(_segment, ct_bytes).result())
    except WorkerError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Synchronous run failed")
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
    _purge_finished_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0,
                         "part": None, "detail": None, "result": None, "finished_at": None}
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
    """Fetch the finished mask as .nii.gz. 409 if not ready, 500 if the job errored.

    Repeatable: the result stays available for RESULT_TTL_S after the job finished.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        status, detail, result = job["status"], job["detail"], job["result"]
    if status == "error":
        raise HTTPException(status_code=500, detail=detail or "Inference failed")
    if status != "done":
        raise HTTPException(status_code=409, detail="Job not finished")
    return _mask_response(result)


if __name__ == "__main__":
    import uvicorn
    from src import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
