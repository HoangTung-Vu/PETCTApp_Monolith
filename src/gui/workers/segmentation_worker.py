import secrets
from pathlib import Path

import httpx
from PyQt6.QtCore import QThread, pyqtSignal

from ...utils.nifti_utils import bytes_to_nifti, nifti_to_gzip_bytes
from .core import HTTP_TIMEOUT, _inference_lock, get_engine_url

# How often to poll the server for job progress (ms).
_POLL_INTERVAL_MS = 750
# Upload chunk size — smaller = smoother upload %, more signal emissions.
_UPLOAD_CHUNK = 1 << 20  # 1 MiB


def _resolve_source(img, stream_path=None):
    """Pick the upload source for a session image (in the app's LAS orientation).

    Default: stream ``stream_path`` from disk — the SessionManager sets it only when
    that file's bytes equal the in-memory image (a LAS NIfTI loaded unchanged). That
    skips the re-encode CPU cost, uploads the compressed file, and is read lazily in
    chunks (no full buffer in RAM).

    Fallback (DICOM sessions, reoriented NIfTI, missing file): serialize the LAS image
    in RAM as multi-member gzip, compressed on all cores. The server detects gzip by
    magic bytes and ``gzip.decompress`` reads multi-member streams.

    Returns ``("path", Path, size)`` or ``("bytes", data)``.
    """
    if stream_path:
        p = Path(stream_path)
        if p.is_file():
            return ("path", p, p.stat().st_size)
    return ("bytes", nifti_to_gzip_bytes(img))


def _build_multipart(files):
    """Build multipart/form-data segments for streaming.

    ``files`` is a list of ``(filename, source)`` where ``source`` is the tagged
    tuple from :func:`_resolve_source`. Returns ``(segments, total, content_type)``
    where ``segments`` is an un-joined list so the body streams without one giant
    buffer. Each segment is either ``("bytes", b)`` (part headers, an in-RAM payload,
    separators, trailer) or ``("path", Path, size)`` (a file streamed lazily at send
    time). ``total`` is the exact body length for ``Content-Length`` / upload %.
    """
    boundary = secrets.token_hex(16)
    segments = []
    total = 0

    def add_bytes(b):
        nonlocal total
        segments.append(("bytes", b))
        total += len(b)

    for filename, source in files:
        add_bytes(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode()
        )
        if source[0] == "path":
            segments.append(source)
            total += source[2]
        else:
            add_bytes(source[1])
        add_bytes(b"\r\n")
    add_bytes(f"--{boundary}--\r\n".encode())
    return segments, total, f"multipart/form-data; boundary={boundary}"


class SegmentationWorker(QThread):
    """Run tumor segmentation via the nnUNet engine's polling API.

    Flow: upload images (POST /jobs) → poll /progress → download /result.
    Emits granular signals so the UI can show upload/progress/done states and
    surface any server error immediately.
    """
    uploaded = pyqtSignal()             # upload accepted, job queued on the server
    progress = pyqtSignal(int, str)     # percent (0-100), stage label
    finished = pyqtSignal(object)       # (mask_nib_image, None, "tumor")
    error = pyqtSignal(str)

    def __init__(self, images, stream_paths=None):
        """``images``: nibabel image(s) to segment; ``stream_paths``: matching
        source files that may be streamed unchanged (None entries → serialize)."""
        super().__init__()
        self.images = images
        self.stream_paths = stream_paths

    def run(self):
        try:
            with _inference_lock:
                # Serializing CT/PET to NIfTI bytes is the slow pre-upload step —
                # tell the dialog so the wait reads as "Preparing", not a stall.
                self.progress.emit(0, "preparing")
                # Resolve the endpoint once per job so a GUI host change mid-run
                # doesn't split a single job across two servers.
                engine_url = get_engine_url()
                images = self.images if isinstance(self.images, list) else [self.images]
                paths = self.stream_paths or [None] * len(images)
                files = [(f"image_{i}.nii.gz", _resolve_source(img, path))
                         for i, (img, path) in enumerate(zip(images, paths))]
                segments, total, content_type = _build_multipart(files)

                with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                    # 1. Upload with byte-level progress. Streaming the segments in
                    #    chunks (no giant joined buffer) lets us report % sent; a
                    #    200 means upload finished and the job is queued. Path
                    #    segments are read lazily from disk so the file never lands
                    #    in RAM whole.
                    def _stream():
                        sent = 0
                        for seg in segments:
                            if seg[0] == "bytes":
                                buf = seg[1]
                                for i in range(0, len(buf), _UPLOAD_CHUNK):
                                    chunk = buf[i:i + _UPLOAD_CHUNK]
                                    yield chunk  # httpx sends this before resuming us
                                    sent += len(chunk)
                                    self.progress.emit(int(sent / total * 100), "upload")
                            else:  # ("path", Path, size)
                                with open(seg[1], "rb") as fh:
                                    while True:
                                        chunk = fh.read(_UPLOAD_CHUNK)
                                        if not chunk:
                                            break
                                        yield chunk
                                        sent += len(chunk)
                                        self.progress.emit(int(sent / total * 100), "upload")

                    resp = client.post(
                        f"{engine_url}/jobs",
                        content=_stream(),
                        headers={"Content-Type": content_type,
                                 "Content-Length": str(total)},
                    )
                    resp.raise_for_status()
                    job_id = resp.json()["job_id"]
                    self.uploaded.emit()

                    # 2. Poll progress until the job is done or errors out.
                    while True:
                        pr = client.get(f"{engine_url}/jobs/{job_id}/progress")
                        pr.raise_for_status()
                        state = pr.json()
                        status = state.get("status")
                        if status == "error":
                            self.error.emit(state.get("detail") or "Inference failed on the server.")
                            return
                        if status == "done":
                            self.progress.emit(100, "done")
                            break
                        self.progress.emit(int(state.get("progress", 0)), state.get("stage", ""))
                        self.msleep(_POLL_INTERVAL_MS)

                    # 3. Download the finished mask.
                    res = client.get(f"{engine_url}/jobs/{job_id}/result")
                    res.raise_for_status()
                    mask_nib = bytes_to_nifti(res.content)
                    self.finished.emit((mask_nib, None, "tumor"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
