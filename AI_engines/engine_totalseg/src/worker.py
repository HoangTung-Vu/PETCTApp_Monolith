"""TotalSegmentator worker process: one per job, started by runner.py (cwd = /app).

    python -m src.worker info
        Prints {"version", "device", "labels", "missing_weights"} as JSON on stdout.

    python -m src.worker run IN OUT --progress-fd N
        Segments the CT in file IN (NIfTI bytes as uploaded), writes the mask to OUT and
        exits 0. Progress goes to fd N as JSON lines {"stage", "progress", "part"}; on
        failure one {"error": "..."} line, then exit 1.

Only the standard library is imported at module level. Children started with the
"spawn" method import the main module again (as ``__mp_main__``), and nnU-Net starts
several per part model: a 12-process Pool to list the input files, a Manager, a
preprocessing process and the export pool. The rest is imported inside the commands.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _info() -> int:
    # Whatever the imports print on stdout goes to the log (stderr) instead, so stdout
    # carries only the JSON line runner.probe_info() reads.
    json_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    from .engine import TotalSegEngine

    engine = TotalSegEngine()
    json.dump({"version": engine.version, "device": engine.device,
               "labels": engine.labels(), "missing_weights": engine.missing_weights()}, json_out)
    json_out.write("\n")
    json_out.close()
    return 0


def _run(in_path: Path, out_path: Path, progress_fd: int) -> int:
    import multiprocessing

    # Every process nnU-Net starts in here is spawned (a fresh interpreter), never
    # forked from this one, which holds a CUDA context and threads after the first part.
    multiprocessing.set_start_method("spawn", force=True)

    # Line-buffered: each event reaches the runner as soon as it is written. Not
    # inherited by anything this process starts, so the runner sees EOF when it exits.
    os.set_inheritable(progress_fd, False)
    events = os.fdopen(progress_fd, "w", buffering=1)

    def emit(**event):
        events.write(json.dumps(event) + "\n")

    try:
        from .engine import TotalSegEngine

        mask = TotalSegEngine().run(
            in_path.read_bytes(),
            progress_callback=lambda stage, pct, part: emit(stage=stage, progress=pct, part=part))
        tmp = out_path.with_name(out_path.name + ".part")
        tmp.write_bytes(mask)
        tmp.replace(out_path)  # the runner never sees a half-written mask
        return 0
    except BaseException as e:  # noqa: BLE001 — report everything, KeyboardInterrupt included
        traceback.print_exc()
        emit(error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        events.close()


def main() -> int:
    import logging

    # Same format as main.py; stdout/stderr of this process go straight to the server log.
    logging.basicConfig(
        level=os.getenv("ENGINE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
    ap = argparse.ArgumentParser(prog="python -m src.worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    run = sub.add_parser("run")
    run.add_argument("input", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--progress-fd", type=int, required=True)
    a = ap.parse_args()

    if a.cmd == "info":
        return _info()
    return _run(a.input, a.output, a.progress_fd)


if __name__ == "__main__":
    sys.exit(main())
