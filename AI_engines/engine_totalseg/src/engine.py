"""TotalSegmentator engine — task "total" at default settings via the Python API.

``run()`` is the in-process equivalent of::

    TotalSegmentator -i ct.nii.gz -o organ.nii.gz --ml

and returns the exact bytes TotalSegmentator writes: a uint8 multilabel mask
(0..117, ``class_map["total"]``) on the input CT grid, with the label names in the
NIfTI header (CaretExtension XML). Nothing is re-encoded on the way out.

Imported only by the worker process (src/worker.py, one per job): the API process
never loads torch or TotalSegmentator (see src/runner.py).
"""

import gc
import logging
import tempfile
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from totalsegmentator.config import get_weights_dir, setup_nnunet, setup_totalseg

# nnunetv2 reads the nnUNet_* env vars when it is imported. TotalSegmentator sets them
# in setup_nnunet() and only then imports its nnunet module, so do the same here.
setup_nnunet()
setup_totalseg()

import totalsegmentator.nnunet as ts_nnunet  # noqa: E402
from totalsegmentator.map_tasks_config import TASK_CONFIGS  # noqa: E402
from totalsegmentator.map_to_binary import class_map  # noqa: E402
from totalsegmentator.python_api import totalsegmentator  # noqa: E402

from . import TASK  # noqa: E402

logger = logging.getLogger("engine.totalseg")

# Part models the default (non-fast) task runs: (291, 292, 293, 294, 295) for "total".
EXPECTED_DATASETS = tuple(TASK_CONFIGS[TASK]["sub_modes"]["default"]["task_id"])

_GZIP_MAGIC = b"\x1f\x8b"

# (stage, percent, part) -> None, e.g. ("inference", 39, "3/5")
ProgressCallback = Callable[[str, int, Optional[str]], None]

# ──── Per-part progress hook ────
# For the multi-model "total" task, nnUNet_predict_image() calls the module-level
# nnUNetv2_predict() once per part model, in the calling thread. Wrapping it gives
# "part k of 5" progress without touching TotalSegmentator's internals. The hook is
# thread-local, so it only reports for the run that installed it.
_hook = threading.local()
_orig_nnunetv2_predict = ts_nnunet.nnUNetv2_predict


def _nnunetv2_predict_with_progress(*args, **kwargs):
    on_part = getattr(_hook, "on_part", None)
    if on_part:
        on_part(started=True)
    result = _orig_nnunetv2_predict(*args, **kwargs)
    if on_part:
        on_part(started=False)
    return result


ts_nnunet.nnUNetv2_predict = _nnunetv2_predict_with_progress


class TotalSegEngine:
    def __init__(self):
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.version = version("TotalSegmentator")

    def missing_weights(self) -> List[int]:
        """Part-model dataset ids whose weights are not on disk (baked in at build)."""
        wd = get_weights_dir()
        return [d for d in EXPECTED_DATASETS if not any(wd.glob(f"Dataset{d}_*"))]

    @staticmethod
    def labels() -> Dict[int, str]:
        """Label id -> structure name, same table as the output file's header."""
        return {int(k): v for k, v in class_map[TASK].items()}

    def run(self, ct_bytes: bytes, progress_callback: Optional[ProgressCallback] = None) -> bytes:
        """Segment one CT (NIfTI bytes, gzipped or raw). Returns the .nii.gz mask bytes."""

        def report(stage: str, pct: int, part: Optional[str] = None):
            if progress_callback:
                progress_callback(stage, pct, part)

        n_parts = len(EXPECTED_DATASETS)
        done = 0

        # 0-5 % preprocessing (canonical + resample to 1.5 mm), 5-90 % the 5 part
        # models, 90-100 % merge + resample back to the CT grid + save.
        def on_part(started: bool):
            nonlocal done
            if started:
                report("inference", 5 + 85 * done // n_parts, f"{done + 1}/{n_parts}")
            else:
                done += 1
                if done < n_parts:
                    report("inference", 5 + 85 * done // n_parts, f"{done + 1}/{n_parts}")
                else:
                    report("postprocessing", 90)

        report("preprocessing", 0)
        with tempfile.TemporaryDirectory(prefix="totalseg_") as td:
            # Keep the upload as-is: the suffix only tells TotalSegmentator it is NIfTI.
            in_path = Path(td) / ("ct.nii.gz" if ct_bytes[:2] == _GZIP_MAGIC else "ct.nii")
            in_path.write_bytes(ct_bytes)
            out_path = Path(td) / "organ.nii.gz"

            logger.info("Running TotalSegmentator task=%s device=%s (%.1f MB input)",
                        TASK, self.device, len(ct_bytes) / 1e6)
            _hook.on_part = on_part
            try:
                # Everything except input/output/ml/device/quiet stays at the
                # TotalSegmentator default (fast=False, roi_subset=None, ...).
                # nr_thr_saving only sizes nnU-Net's export pool here (ml=True): one
                # CT is one export task, so 1 process gives the same file without
                # spawning 5 more processes that each import torch, at every part.
                totalsegmentator(input=in_path, output=out_path, ml=True, task=TASK,
                                 device=self.device, quiet=True, nr_thr_saving=1)
            finally:
                _hook.on_part = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if done != n_parts:
                logger.warning("Expected %d part models, progress hook saw %d", n_parts, done)
            report("postprocessing", 99)
            return out_path.read_bytes()
