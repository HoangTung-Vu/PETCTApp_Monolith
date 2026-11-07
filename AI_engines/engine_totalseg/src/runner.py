"""Runs each TotalSegmentator job in its own worker process (``python -m src.worker``).

Standard library only, like the rest of the API process: torch and TotalSegmentator
are imported in the worker, never here.

Why a process per job. For every part model nnU-Net starts child processes: a
default-context (fork) Pool(12) to list the input files, a Manager, a spawned
preprocessing process and a spawned export pool, which is sent the logits (several
GB) by pickling. Run inside the uvicorn process, as before, this
  * hung jobs at the start of a part (22/56/73 %): children forked from a process
    with a CUDA context, uvicorn's threads, signal handlers and open client sockets
    can block for good in join()/recv(), and every later job queued behind them;
  * stopped the API for ~10-20 s at every part change (clients got "connection
    reset"): the pickling and the children starting up took the GIL and the CPU
    from uvicorn's event loop.
The worker starts every child with "spawn", shares nothing with the API process but
one pipe, and is killed with all its children when it hangs.

Watchdog: progress events arrive at each part change. When none arrives for
``stall_timeout_s``, the worker's process group is killed and the job runs again
(``max_attempts`` runs in total); then the job fails with "no progress for N s at
part k/5". A worker that crashes or reports an error is not retried.
"""

import json
import logging
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("engine.runner")

APP_DIR = Path(__file__).resolve().parent.parent  # holds src/: cwd of the worker
WORKER_CMD = [sys.executable, "-m", "src.worker"]

# (stage, progress 0-100, part "k/5" or None, detail or None)
ProgressCallback = Callable[[str, int, Optional[str], Optional[str]], None]

_GROUP_EXIT_GRACE_S = 5.0  # per step when ending a worker's process group
_EXIT_TIMEOUT_S = 120.0  # a worker that closed its pipe should be gone by then

_proc: Optional[subprocess.Popen] = None  # worker of the running job
_proc_lock = threading.Lock()
_stopping = False


class WorkerError(RuntimeError):
    """The job failed. The message is meant for the client (the job's "detail")."""


def probe_info(timeout_s: float = 600) -> dict:
    """Engine info (version, device, labels, missing_weights) from a short-lived worker.

    Run once at start-up; raises if the worker cannot import TotalSegmentator.
    """
    t0 = time.monotonic()
    proc = subprocess.run(WORKER_CMD + ["info"], cwd=APP_DIR, stdout=subprocess.PIPE,
                          timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"'python -m src.worker info' exited with code {proc.returncode}; "
                           f"see its traceback above")
    info = json.loads(proc.stdout)
    info["labels"] = {int(k): v for k, v in info["labels"].items()}
    logger.info("Worker info in %.1f s", time.monotonic() - t0)
    return info


def kill_current() -> None:
    """Kill the running job's worker and its children; used at server shutdown."""
    global _stopping
    with _proc_lock:
        _stopping = True
        proc = _proc
    if proc is not None:
        logger.warning("Shutting down: killing worker %d", proc.pid)
        _signal_group(proc.pid, signal.SIGKILL)


def run_isolated(ct_bytes: bytes, on_progress: Optional[ProgressCallback] = None, *,
                 stall_timeout_s: float = 0.0, max_attempts: int = 1) -> bytes:
    """Segment one CT (NIfTI bytes) in a worker process; returns the .nii.gz mask bytes.

    Raises WorkerError when the worker fails or stalls on every attempt.
    """
    report = on_progress or (lambda *_: None)
    # Input, output, and TMPDIR of every attempt (TotalSegmentator's temp folders,
    # multiprocessing's pymp-*): a killed worker leaves no files behind.
    job_dir = Path(tempfile.mkdtemp(prefix="totalseg_job_"))
    try:
        in_path, out_path = job_dir / "ct_upload", job_dir / "organ.nii.gz"
        in_path.write_bytes(ct_bytes)
        for attempt in range(1, max_attempts + 1):
            stalled_at = _run_attempt(job_dir, attempt, in_path, out_path, report, stall_timeout_s)
            if stalled_at is None:
                return out_path.read_bytes()
            msg = f"no progress for {stall_timeout_s:.0f} s at {stalled_at}"
            if attempt == max_attempts:
                raise WorkerError(f"TotalSegmentator stalled: {msg} "
                                  f"(gave up after {max_attempts} attempt(s))")
            logger.warning("Worker stalled (%s): killed; starting attempt %d/%d",
                           msg, attempt + 1, max_attempts)
            report("restarting", 0, None,
                   f"Restarted: {msg} (attempt {attempt + 1}/{max_attempts})")
        raise AssertionError("unreachable")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _run_attempt(job_dir: Path, attempt: int, in_path: Path, out_path: Path,
                 report: ProgressCallback, stall_timeout_s: float) -> Optional[str]:
    """Run one worker to the end. Returns None on success, or where it stalled
    ("part 2/5", "preprocessing", ...) after killing it. Raises WorkerError on failure."""
    global _proc
    tmp = job_dir / f"tmp{attempt}"
    tmp.mkdir()
    out_path.unlink(missing_ok=True)
    env = dict(os.environ, TMPDIR=str(tmp))

    r, w = os.pipe()
    try:
        with _proc_lock:
            if _stopping:
                raise WorkerError("Engine is shutting down")
            # Own session = own process group: os.killpg() reaches the worker and every
            # process it starts. close_fds (default) keeps the API's sockets out of it.
            proc = subprocess.Popen(
                WORKER_CMD + ["run", str(in_path), str(out_path), "--progress-fd", str(w)],
                cwd=APP_DIR, env=env, pass_fds=(w,), start_new_session=True)
            _proc = proc
    except BaseException:
        os.close(r)
        raise
    finally:
        os.close(w)
    logger.info("Attempt %d: worker pid %d started", attempt, proc.pid)

    where, error = "worker start-up", None
    last_event = time.monotonic()
    stalled = False
    buf = b""
    sel = selectors.DefaultSelector()
    sel.register(r, selectors.EVENT_READ)
    try:
        while True:
            if sel.select(timeout=1.0):
                chunk = os.read(r, 65536)
                if not chunk:
                    break  # EOF: the worker exited (only it holds the write end)
                *lines, buf = (buf + chunk).split(b"\n")
                for line in lines:
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        logger.warning("Bad progress line from worker: %r", line[:200])
                        continue
                    last_event = time.monotonic()
                    if "error" in ev:
                        error = ev["error"]
                        continue
                    stage, pct, part = ev.get("stage", ""), int(ev.get("progress", 0)), ev.get("part")
                    new_where = f"part {part}" if part else stage
                    if new_where != where:
                        logger.info("Worker %d: %s (%d %%)", proc.pid, new_where, pct)
                    where = new_where
                    report(stage, pct, part, None)
            elif proc.poll() is not None:
                break
            elif stall_timeout_s and time.monotonic() - last_event > stall_timeout_s:
                stalled = True
                logger.error("Worker %d: no progress for %.0f s at %s; killing it",
                             proc.pid, stall_timeout_s, where)
                break
        if not stalled:
            try:  # pipe closed or worker gone: it is exiting
                proc.wait(timeout=_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("Worker %d still running %.0f s after closing its pipe; "
                               "killing it", proc.pid, _EXIT_TIMEOUT_S)
    finally:
        sel.close()
        os.close(r)
        _end_group(proc)
        with _proc_lock:
            _proc = None

    if _stopping:
        raise WorkerError("Engine is shutting down")
    if stalled:
        return where
    rc = proc.returncode
    if error is None and out_path.exists():
        # The mask is renamed into place complete, so it is good even if the worker
        # then hung on exit and had to be killed.
        if rc != 0:
            logger.warning("Worker %d wrote the mask, then exited with %d", proc.pid, rc)
        return None
    if error:
        raise WorkerError(f"TotalSegmentator failed at {where}: {error}")
    if rc == -signal.SIGKILL:
        raise WorkerError(f"TotalSegmentator worker was killed (SIGKILL) at {where}; "
                          f"most likely out of memory")
    if rc < 0:
        raise WorkerError(f"TotalSegmentator worker died from {signal.Signals(-rc).name} at {where}")
    raise WorkerError(f"TotalSegmentator worker exited with code {rc} at {where}"
                      + ("" if rc else " without writing the mask"))


def _end_group(proc: subprocess.Popen) -> None:
    """Make sure the worker and everything it started are gone, and reap them.

    A worker still running (stalled) gets SIGTERM, then SIGKILL. SIGTERM first because
    multiprocessing's resource tracker ignores it: it outlives the others and unlinks
    the semaphores they leave in /dev/shm (shared with the host under --ipc=host).
    In the container this process is PID 1, so children orphaned by the worker are
    re-parented here and must be reaped, or they stay as zombies.
    """
    if proc.poll() is None:
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=_GROUP_EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
    proc.wait()
    if not _wait_group_empty(proc.pid, _GROUP_EXIT_GRACE_S):
        logger.warning("Worker %d left processes running; killing them", proc.pid)
        _signal_group(proc.pid, signal.SIGKILL)
        _wait_group_empty(proc.pid, _GROUP_EXIT_GRACE_S)


def _wait_group_empty(pgid: int, timeout_s: float) -> bool:
    """Wait (reaping) until no process of group ``pgid`` is left; False on timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        _reap(pgid)
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return True  # pgid reused by a process we don't own: ours are gone
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def _reap(pgid: int) -> None:
    """Collect exited members of group ``pgid`` that are children of this process."""
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
