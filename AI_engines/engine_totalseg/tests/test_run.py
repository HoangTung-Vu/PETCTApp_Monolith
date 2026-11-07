"""Client test for the TotalSegmentator engine (run on the host against a running container).

    uv run python AI_engines/engine_totalseg/tests/test_run.py \
        --ct  .../export/fdg26072803_ct.nii.gz \
        --ref .../export/fdg26072803_organ.nii.gz \
        --out /tmp/organ.nii.gz [--crop-z 60]

Checks that the engine returns the same kind of file as ``TotalSegmentator --ml``
(the reference ``*_organ.nii.gz``): uint8, on the input CT grid, labels within
0..117, the same label table in the NIfTI header. Then reports per-label Dice
against the reference. ``--crop-z N`` sends only the middle N slices (quick smoke test).
Exit code 1 if any format check fails.
"""

import argparse
import gzip
import io
import re
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8105")
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="reference TotalSegmentator --ml output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--crop-z", type=int, default=0, help="only send the middle N slices")
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

    with httpx.Client(base_url=a.url, timeout=600) as c:
        print("info:", c.get("/health").json(), c.get("/models").json())
        labels = {int(k): v for k, v in c.get("/total/labels").json().items()}

        t0 = time.time()
        r = c.post("/total/jobs", files={"files": ("image_0.nii.gz", payload, "application/octet-stream")})
        r.raise_for_status()
        job = r.json()["job_id"]
        last, parts = None, set()
        while True:
            st = c.get(f"/total/jobs/{job}/progress").json()
            if st.get("part"):
                parts.add(st["part"])
            key = (st["status"], st["stage"], st["progress"], st.get("part"))
            if key != last:
                print(f"  {time.time() - t0:7.1f}s  {st}")
                last = key
            if st["status"] in ("done", "error"):
                break
            time.sleep(2)
        if st["status"] == "error":
            print("JOB ERROR:", st["detail"])
            return 1
        res = c.get(f"/total/jobs/{job}/result")
        res.raise_for_status()
        print(f"done in {time.time() - t0:.0f}s, {len(res.content) / 1e3:.0f} KB")

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
