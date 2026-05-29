import secrets

import httpx
from PyQt6.QtCore import QThread, pyqtSignal

from ...utils.nifti_utils import bytes_to_nifti, make_nifti_upload
from .core import ENGINE_NNUNET_URL, HTTP_TIMEOUT, _inference_lock

# How often to poll the server for job progress (ms).
_POLL_INTERVAL_MS = 750
# Upload chunk size — smaller = smoother upload %, more signal emissions.
_UPLOAD_CHUNK = 1 << 20  # 1 MiB


def _build_multipart(files):
    """Build multipart/form-data parts for streaming.

    Returns ``(parts, total, content_type)`` where ``parts`` is a list of byte
    segments (part headers, file payloads, trailer), kept *un-joined* so the body
    can be streamed without allocating one giant buffer. File payloads are
    referenced as-is (no copy); only ~1 MiB chunks are copied at send time.
    """
    boundary = secrets.token_hex(16)
    parts = []
    for field, (filename, data, ctype) in files:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
        )
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    total = sum(len(p) for p in parts)
    return parts, total, f"multipart/form-data; boundary={boundary}"


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

    def __init__(self, images):
        super().__init__()
        self.images = images

    def run(self):
        try:
            with _inference_lock:
                # Serializing CT/PET to NIfTI bytes is the slow pre-upload step —
                # tell the dialog so the wait reads as "Preparing", not a stall.
                self.progress.emit(0, "preparing")
                images = self.images if isinstance(self.images, list) else [self.images]
                files = [make_nifti_upload(img, f"image_{i}.nii.gz")
                         for i, img in enumerate(images)]
                parts, total, content_type = _build_multipart(files)

                with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                    # 1. Upload with byte-level progress. Streaming the parts in
                    #    chunks (no giant joined buffer) lets us report % sent; a
                    #    200 means upload finished and the job is queued.
                    def _stream():
                        sent = 0
                        for part in parts:
                            for i in range(0, len(part), _UPLOAD_CHUNK):
                                chunk = part[i:i + _UPLOAD_CHUNK]
                                yield chunk  # httpx sends this before resuming us
                                sent += len(chunk)
                                self.progress.emit(int(sent / total * 100), "upload")

                    resp = client.post(
                        f"{ENGINE_NNUNET_URL}/jobs",
                        content=_stream(),
                        headers={"Content-Type": content_type,
                                 "Content-Length": str(total)},
                    )
                    resp.raise_for_status()
                    job_id = resp.json()["job_id"]
                    self.uploaded.emit()

                    # 2. Poll progress until the job is done or errors out.
                    while True:
                        pr = client.get(f"{ENGINE_NNUNET_URL}/jobs/{job_id}/progress")
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
                    res = client.get(f"{ENGINE_NNUNET_URL}/jobs/{job_id}/result")
                    res.raise_for_status()
                    mask_nib = bytes_to_nifti(res.content)
                    self.finished.emit((mask_nib, None, "tumor"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
