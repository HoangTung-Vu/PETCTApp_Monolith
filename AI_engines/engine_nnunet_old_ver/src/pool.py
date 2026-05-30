"""Engine pool — one active model at a time, served by N concurrent workers.

A single model is "active". For it the pool holds N independent
:class:`NNUNetEngine` instances (each owns its own predictor + GPU weights, so
each carries its own ``_progress_callback`` and concurrent jobs never race).
N is derived from free VRAM and the per-inference estimate. Requests run on a
``ThreadPoolExecutor(max_workers=N)`` and check out a free engine from a
size-N queue; with at most N concurrent tasks and N engines, "process N at a
time, queue the rest" falls out for free (no separate batch loop).

Switching models tears down the old engines (freeing VRAM), re-estimates N for
the new model and respawns — but ONLY when the pool is idle. If any inference
is still in flight, a switch is *rejected* with :class:`ModelBusyError` rather
than blocking the caller. Real usage is "almost always one model, switches
rare", so reject-when-busy is both simpler and matches the invariant we want:
never swap the active model out from under a running inference.

In-flight accounting: ``_inflight`` counts inference tasks that have been
submitted to the executor but not yet finished (running *or* queued). It is
incremented in ``submit`` just before ``executor.submit`` and decremented by the
future's done-callback, so ``_inflight == 0`` exactly means the executor has no
pending/running work — i.e. no engine is inferring and a switch is safe.

Locking — two locks, deliberately:
  * ``_switch_lock``  serializes model switches and submit-admission. It is the
    ONLY lock held across the slow parts (engine teardown + weight loading).
    ``submit``/``ensure_model`` take it.
  * ``_state_lock``   guards quick reads/swaps of the pool's pointers
    (active_model/engines/free/executor/n/_inflight). It is held for
    microseconds only and is NEVER held during a teardown or spawn.

So ``status``, ``checkout`` and ``checkin`` (and therefore the polling
endpoints) keep working during a switch — only other ``submit`` callers wait.
Because a switch is only permitted when ``_inflight == 0``, the executor is
already idle, so ``shutdown(wait=True)`` in the teardown is a defensive no-op.
"""

import gc
import logging
import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import torch

from . import setup_nnunet_env
from .engine import NNUNetEngine
from .vram import estimate_vram_usage_inference_gb

logger = logging.getLogger("engine.pool")

# Hard cap on concurrent engines regardless of how much VRAM is free.
MAX_ENGINES = int(os.getenv("ENGINE_MAX_WORKERS", "4"))
# Headroom (GB) added to the per-engine estimate when dividing free VRAM.
VRAM_HEADROOM_GB = 1.0


class ModelBusyError(RuntimeError):
    """Raised when a model switch is requested while inference is in flight."""


class EnginePool:
    def __init__(self, models: List[str], default_model: str, device: str = "auto"):
        setup_nnunet_env()  # ensure nnUNet_results env is set before resolving folders
        self.models = list(models)
        self.default_model = default_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # See module docstring for the two-lock rationale.
        self._switch_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self.active_model: Optional[str] = None
        self.engines: List[NNUNetEngine] = []
        self.free: "queue.Queue[NNUNetEngine]" = queue.Queue()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.n = 0
        self.estimate_gb: Optional[float] = None
        # Inference tasks submitted-but-not-finished (running or queued). A model
        # switch is only allowed when this is 0. Guarded by _state_lock.
        self._inflight = 0

    # ──── introspection (quick, never blocks on a switch) ────

    def available_models(self) -> List[str]:
        return list(self.models)

    def status(self) -> dict:
        with self._state_lock:
            return {
                "active_model": self.active_model,
                "num_engines": self.n,
                "estimate_gb": round(self.estimate_gb, 3) if self.estimate_gb else None,
                "free_engines": self.free.qsize(),
                "inflight": self._inflight,
                "device": str(self.device),
                "max_engines": MAX_ENGINES,
            }

    def _model_folder(self, model_name: str) -> Path:
        return Path(os.environ["nnUNet_results"]) / model_name

    # ──── per-request engine handout (quick locks only) ────

    def checkout(self) -> NNUNetEngine:
        with self._state_lock:
            free = self.free
        eng = free.get()  # block (without any lock held) until an engine is free
        logger.debug("Engine checked out (%d free remaining)", free.qsize())
        return eng

    def checkin(self, engine: NNUNetEngine):
        with self._state_lock:
            free = self.free
        free.put(engine)
        logger.debug("Engine checked in (%d free now)", free.qsize())

    # ──── model lifecycle (slow work guarded by _switch_lock, NOT _state_lock) ────

    def ensure_model(self, model_name: str):
        """Make ``model_name`` the active model, switching the pool if needed.

        Rejects with :class:`ModelBusyError` if a switch is required while any
        inference is in flight (``_inflight > 0``).
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.models}")
        with self._switch_lock:
            with self._state_lock:
                active = self.active_model
                executor = self.executor
                inflight = self._inflight
            if model_name != active or executor is None:
                if inflight > 0:
                    raise ModelBusyError(
                        f"Cannot switch to '{model_name}': {inflight} inference "
                        f"job(s) in progress on '{active}'. Try again once idle.")
                self._switch_to(model_name)

    def submit(self, model_name: str, fn: Callable, *args) -> Future:
        """Ensure ``model_name`` is active and submit ``fn(*args)`` to the pool.

        Holding ``_switch_lock`` across ensure + submit guarantees the task lands
        on the executor whose engines match ``model_name`` (a concurrent switch
        cannot interleave). If reaching ``model_name`` would require switching
        while inference is in flight, raises :class:`ModelBusyError` instead.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.models}")
        with self._switch_lock:
            with self._state_lock:
                active = self.active_model
                executor = self.executor
                inflight = self._inflight
            if model_name != active or executor is None:
                if inflight > 0:
                    raise ModelBusyError(
                        f"Cannot switch to '{model_name}': {inflight} inference "
                        f"job(s) in progress on '{active}'. Try again once idle.")
                self._switch_to(model_name)
                with self._state_lock:
                    executor = self.executor
            assert executor is not None  # set by _switch_to or already active
            # Count this task in flight BEFORE it lands on the executor; the
            # done-callback decrements it (so _inflight == 0 <=> executor idle).
            with self._state_lock:
                self._inflight += 1
                inflight_now = self._inflight
            try:
                fut = executor.submit(fn, *args)
            except BaseException:
                with self._state_lock:
                    self._inflight -= 1
                raise
            fut.add_done_callback(self._on_task_done)
            logger.debug("Submitting task for model '%s' (inflight=%d)", model_name, inflight_now)
            return fut

    def _on_task_done(self, _fut: Future):
        """Decrement the in-flight count when an inference task finishes.

        Registered via ``Future.add_done_callback``, so it fires exactly once per
        submitted task — including if the task had already completed by the time
        the callback was attached.
        """
        with self._state_lock:
            self._inflight -= 1

    def _teardown(self):
        """Drain the executor and free all current engines + their VRAM.

        Only ``_state_lock`` (briefly) wraps the pointer reads/resets; the slow
        ``shutdown(wait=True)`` and ``unload`` run with no lock held so in-flight
        jobs can still checkin/checkout and status() stays responsive.
        """
        with self._state_lock:
            executor = self.executor
            engines = self.engines
            old_model = self.active_model
            self.executor = None  # no new submits land here; running tasks don't read it

        if executor is not None:
            logger.debug("Draining executor before teardown...")
            executor.shutdown(wait=True)  # waits for in-flight inference to finish
        if engines:
            logger.info("Tearing down %d engine(s) for model '%s'", len(engines), old_model)
            for eng in engines:
                try:
                    eng.unload()
                except Exception:
                    logger.exception("Error unloading an engine")

        with self._state_lock:
            self.engines = []
            self.free = queue.Queue()
            self.n = 0
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info(self.device)[0] / 1e9
            logger.info("VRAM after teardown: %.2f GB free", free_gb)

    def _switch_to(self, model_name: str):
        """Tear down the current pool and build a fresh one for ``model_name``.

        Must be called holding ``_switch_lock``. The drain + estimate + spawn run
        without ``_state_lock``; only the final atomic swap takes it.
        """
        with self._state_lock:
            old = self.active_model
        logger.info("Switching active model: %s -> %s", old, model_name)

        self._teardown()
        with self._state_lock:
            self.active_model = None

        n, est_gb = self._compute_n(model_name)

        logger.info("Spawning %d engine worker(s) for '%s'...", n, model_name)
        engines: List[NNUNetEngine] = []
        for i in range(n):
            eng = NNUNetEngine(model_name=model_name, device=str(self.device))
            eng._init_predictor()  # load weights into (GPU) memory now
            engines.append(eng)
            logger.info("  engine %d/%d ready for '%s'", i + 1, n, model_name)

        new_free: "queue.Queue[NNUNetEngine]" = queue.Queue()
        for eng in engines:
            new_free.put(eng)
        new_executor = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix=f"infer-{model_name[:12]}")

        # Atomic swap — quick, under _state_lock.
        with self._state_lock:
            self.engines = engines
            self.free = new_free
            self.executor = new_executor
            self.n = n
            self.estimate_gb = est_gb
            self.active_model = model_name
        logger.info("Active model is now '%s' with %d concurrent engine worker(s)", model_name, n)

    def _compute_n(self, model_name: str):
        """Return (n_engines, per_engine_estimate_gb). CPU -> exactly 1, no estimate."""
        if self.device.type != "cuda":
            logger.warning("No GPU detected -> running a single engine, concurrency disabled")
            return 1, None

        model_folder = self._model_folder(model_name)
        est = estimate_vram_usage_inference_gb(str(model_folder), self.device)
        est_gb = est["estimate_gb"]
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        free_gb = free_bytes / 1e9
        n = int(free_gb // (est_gb + VRAM_HEADROOM_GB))
        n = max(1, min(MAX_ENGINES, n))
        logger.info(
            "VRAM budget for '%s': estimate=%.2f GB (+%.1f headroom) | free=%.2f/%.2f GB "
            "| n = min(%d, floor(%.2f / %.2f)) = %d",
            model_name, est_gb, VRAM_HEADROOM_GB, free_gb, total_bytes / 1e9,
            MAX_ENGINES, free_gb, est_gb + VRAM_HEADROOM_GB, n,
        )
        return n, est_gb
