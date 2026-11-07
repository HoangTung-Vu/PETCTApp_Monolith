"""TotalSegmentator engine — settings shared by main.py, runner.py and engine.py.

Standard library only: the API process (main.py, runner.py) imports this package and
must stay free of torch (see runner.py).
"""

import os

# Port (8104 is the nnU-Net engine)
PORT = int(os.getenv("ENGINE_TOTALSEG_PORT", "8105"))

# TotalSegmentator task served by this engine. Default (non-fast) "total" = 117 classes
# predicted by 5 part models at 1.5 mm: organs, vertebrae, cardiac, muscles, ribs.
TASK = "total"

# Watchdog (runner.py): a job whose worker sends no progress for this many seconds is
# killed and started again, up to MAX_ATTEMPTS runs in total. Progress arrives at each
# part change: a part takes ~1-1.5 min on a GPU, the final merge + save ~1.5 min.
DEFAULT_STALL_TIMEOUT_S = 600.0

# Runs per job, counting the first (2 = one restart after a stall).
MAX_ATTEMPTS = max(1, int(os.getenv("ENGINE_MAX_ATTEMPTS", "2")))


def stall_timeout_s(device: str) -> float:
    """Watchdog limit in seconds, 0 = off.

    ``ENGINE_STALL_TIMEOUT`` sets it explicitly (0 turns it off). Unset, it is
    DEFAULT_STALL_TIMEOUT_S on a GPU and off on a CPU-only host, where one part model
    can take longer than that.
    """
    env = os.getenv("ENGINE_STALL_TIMEOUT", "").strip()
    if env:
        return max(0.0, float(env))
    return DEFAULT_STALL_TIMEOUT_S if device == "gpu" else 0.0
