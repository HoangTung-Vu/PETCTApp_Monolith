"""Client test for the TotalSegmentator engine (run on the host against a running container).

    uv run python AI_engines/engine_totalseg/tests/test_run.py \
        --ct  .../export/fdg26072803_ct.nii.gz \
        --ref .../export/fdg26072803_organ.nii.gz \
        --out /tmp/organ.nii.gz [--crop-z 60]

Checks that the engine returns the same kind of file as ``TotalSegmentator --ml``
(the reference ``*_organ.nii.gz``): uint8, on the input CT grid, labels within
0..117, the same label table in the NIfTI header. Then reports per-label Dice
against the reference. ``--crop-z N`` sends only the middle N slices (quick smoke test).

While the job runs, a second connection asks for /health every 0.5 s: the API must
keep answering (max latency < 3 s, no connection errors). The result is fetched
twice and must be the same bytes (a client may retry after a dropped connection).

``--expect-error`` (watchdog test, e.g. ENGINE_STALL_TIMEOUT=15 on a CPU): pass if
the job ends in "error" (its detail is printed); the API checks still apply.
Exit code 1 if any check fails.
"""

import argparse
import gzip
import io
import re
import sys
import threading
import time
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np

_LABEL_RE = re.compile(r'<Label Key="(\d+)"[^>]*><!\[CDATA\[(.*?)\]\]></Label>')


def label_table(img: nib.Nifti1Image) -> dict:
    """Parse TotalSegmentator's CaretExtension label table from the header extensions."""
    table = {}
    for ext in img.header.extensions:
        table.update({int(k): v for k, v in _LABEL_RE.findall(ext.get_content().decode("utf-8", "replace"))})
    return table


def to_gz_bytes(img: nib.Nifti1Image) -> bytes:
    bio = io.BytesIO()
    img.to_file_map(img.make_file_map({"image": bio, "header": bio}))
    return gzip.compress(bio.getvalue(), compresslevel=1)


def dice_per_label(pred: np.ndarray, ref: np.ndarray, n: int) -> dict:
    """Dice for every label 1..n present in pred or ref, from one confusion matrix."""
    cm = np.bincount(pred.astype(np.int64).ravel() * (n + 1) + ref.astype(np.int64).ravel(),
                     minlength=(n + 1) ** 2).reshape(n + 1, n + 1)
    out = {}
    for k in range(1, n + 1):
        p, r = cm[k, :].sum(), cm[:, k].sum()
        if p + r:
            out[k] = 2 * cm[k, k] / (p + r)
    return out


class HealthMonitor(threading.Thread):
    """Asks for /health every ``interval`` s on its own connection until stopped."""

    def __init__(self, url: str, interval: float = 0.5):
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.latencies, self.errors = [], []
        self._stop_event = threading.Event()

    def run(self):
        with httpx.Client(base_url=self.url, timeout=60) as c:
            while not self._stop_event.is_set():
                t = time.perf_counter()
                try:
                    c.get("/health").raise_for_status()
                    self.latencies.append(time.perf_counter() - t)
                except httpx.HTTPError as e:
                    self.errors.append(f"{time.perf_counter() - t:.1f}s {type(e).__name__}: {e}")
                self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()


def get_with_retry(c: httpx.Client, path: str, errors: list, retry_s: float = 120) -> httpx.Response:
    """GET, retrying dropped connections for ``retry_s`` (like the app); errors are recorded."""
    t0 = time.time()
    while True:
        try:
            return c.get(path)
        except httpx.TransportError as e:
            errors.append(f"GET {path}: {type(e).__name__}: {e}")
            if time.time() - t0 > retry_s:
                raise
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8105")
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="reference TotalSegmentator --ml output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--crop-z", type=int, default=0, help="only send the middle N slices")
    ap.add_argument("--expect-error", action="store_true",
                    help="the job must fail (watchdog test); skip the mask checks")
    a = ap.parse_args()

    # Never overwrite an existing file (in particular the CT or the reference mask).
    if a.out.exists() or a.out.resolve() in (a.ct.resolve(), a.ref.resolve()):
        print(f"Refusing to overwrite existing file: {a.out}")
        return 2

    failures = []

    def check(ok: bool, what: str):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
        if not ok:
            failures.append(what)

    ct, ref = nib.load(a.ct), nib.load(a.ref)
    ref_table = label_table(ref)
    if a.crop_z:
        z0 = (ct.shape[2] - a.crop_z) // 2
        ct, ref = ct.slicer[:, :, z0:z0 + a.crop_z], ref.slicer[:, :, z0:z0 + a.crop_z]
        payload = to_gz_bytes(ct)
    else:
        payload = a.ct.read_bytes()  # stream the file as-is, like the app does
    print(f"CT {ct.shape} {len(payload) / 1e6:.1f} MB -> {a.url}")

    poll_errors = []
    with httpx.Client(base_url=a.url, timeout=600) as c:
        print("info:", c.get("/health").json(), c.get("/models").json())
        labels = {int(k): v for k, v in c.get("/total/labels").json().items()}

        health = HealthMonitor(a.url)
        health.start()
        t0 = time.time()
        r = c.post("/total/jobs", files={"files": ("image_0.nii.gz", payload, "application/octet-stream")})
        r.raise_for_status()
        job = r.json()["job_id"]
        last, parts, stages = None, set(), []
        while True:
            st = get_with_retry(c, f"/total/jobs/{job}/progress", poll_errors).json()
            if st.get("part"):
                parts.add(st["part"])
            if not stages or stages[-1] != st["stage"]:
                stages.append(st["stage"])
            key = (st["status"], st["stage"], st["progress"], st.get("part"), st.get("detail"))
            if key != last:
                print(f"  {time.time() - t0:7.1f}s  {st}")
                last = key
            if st["status"] in ("done", "error"):
                break
            time.sleep(2)
        elapsed = time.time() - t0
        res = [get_with_retry(c, f"/total/jobs/{job}/result", poll_errors) for _ in range(2)]
        health.stop()

    lat = np.array(health.latencies) if health.latencies else np.array([np.nan])
    print(f"API during the job: {len(health.latencies)} /health answers, latency median "
          f"{np.median(lat) * 1e3:.0f} ms, max {lat.max():.2f} s; {len(health.errors)} health "
          f"error(s), {len(poll_errors)} progress/result connection error(s)")
    for e in (health.errors + poll_errors)[:10]:
        print("   ", e)
    print("API checks:")
    check(lat.max() < 3, f"/health max latency < 3 s (got {lat.max():.2f} s)")
    check(not health.errors and not poll_errors, "no connection errors during the job")
    check(res[0].status_code == res[1].status_code and res[0].content == res[1].content,
          f"result GET is repeatable (status {res[0].status_code}, {res[1].status_code})")

    if a.expect_error:
        print(f"job ended '{st['status']}' after {elapsed:.0f}s, stages {stages}")
        print("  detail:", st.get("detail"))
        check(st["status"] == "error", "job failed as expected")
        check(res[0].status_code == 500, f"result is 500 for a failed job (got {res[0].status_code})")
        print("RESULT:", "OK" if not failures else f"{len(failures)} check(s) failed")
        return 1 if failures else 0
    if st["status"] == "error":
        print("JOB ERROR:", st["detail"])
        return 1
    res = res[0]
    res.raise_for_status()
    print(f"done in {elapsed:.0f}s, {len(res.content) / 1e3:.0f} KB, stages {stages}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(res.content)
    out = nib.load(a.out)
    data = np.asanyarray(out.dataobj)
    ref_data = np.asanyarray(ref.dataobj)
    present = np.unique(data)
    out_table = label_table(out)

    print("Format checks:")
    check(res.content[:2] == b"\x1f\x8b", "result is gzip (.nii.gz)")
    check(out.header.get_data_dtype() == np.uint8, f"dtype uint8 (got {out.header.get_data_dtype()})")
    check(out.shape == ct.shape, f"shape == input CT {ct.shape} (got {out.shape})")
    check(np.allclose(out.affine, ct.affine, atol=1e-4), "affine == input CT affine")
    check(present.min() >= 0 and present.max() <= 117, f"labels within 0..117 (max {present.max()})")
    check(len(out_table) == 117 and out_table == ref_table,
          f"header label table identical to reference ({len(out_table)} vs {len(ref_table)} entries)")
    check(labels == ref_table, "/total/labels identical to reference table")
    check(len(parts) == 5, f"progress reported all 5 part models (saw {sorted(parts)})")
    if not a.crop_z:
        check(len(present) - 1 >= 100, f"100+ structures found (got {len(present) - 1}; reference "
                                       f"{len(np.unique(ref_data)) - 1})")

    d = dice_per_label(data, ref_data, 117)
    in_ref = {k: v for k, v in d.items() if (ref_data == k).any()}
    vals = np.array(list(in_ref.values()))
    worst = sorted(in_ref.items(), key=lambda kv: kv[1])[:8]
    print(f"Dice vs reference over {len(in_ref)} reference labels: mean {vals.mean():.3f}, "
          f"median {np.median(vals):.3f}, min {vals.min():.3f}")
    print(f"  liver (5): {d.get(5, float('nan')):.3f}")
    print("  lowest:", ", ".join(f"{ref_table[k]}={v:.2f}" for k, v in worst))

    print("RESULT:", "OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
