import os
import threading
from urllib.parse import quote

from dotenv import load_dotenv

from ...core.config import app_settings

# Load a local .env (if present) so the engine host can be set without code edits.
load_dotenv()

# ──── Configuration ────

# The scheme is fixed at app startup (env-overridable). The GUI configures the
# IP, port and model; their env values here are only the defaults on first run.
ENGINE_NNUNET_SCHEME = os.getenv("ENGINE_NNUNET_SCHEME", "http")
ENGINE_NNUNET_MODEL = os.getenv(
    "ENGINE_NNUNET_MODEL",
    "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres",
)
# Endpoint of the TotalSegmentator engine (AI_engines/engine_totalseg). Its
# result is an organ mask (117 labels), not a tumor mask.
TOTALSEG_MODEL = "total"

# Models offered in the GUI: (model name in the URL, label, the engine's default port).
# "Other…" in the GUI accepts any other model name.
MODEL_PRESETS = [
    (ENGINE_NNUNET_MODEL, "nnU-Net tumor (custom)", "8104"),
    (TOTALSEG_MODEL, "TotalSegmentator (organs)", "8105"),
]

# Defaults used when nothing is configured ("localhost" for a local engine).
_DEFAULT_ENGINE_IP = os.getenv("ENGINE_NNUNET_IP", "localhost")
_DEFAULT_ENGINE_PORT = os.getenv("ENGINE_NNUNET_PORT", "8104")

# QSettings keys for persisting the GUI-configured IP/port/model across runs.
_SETTINGS_IP_KEY = "engine/nnunet_ip"
_SETTINGS_PORT_KEY = "engine/nnunet_port"
_SETTINGS_MODEL_KEY = "engine/model"


def is_totalseg_model(model: str) -> bool:
    """True for the TotalSegmentator engine (organ mask on the CT, CT-only input)."""
    return model == TOTALSEG_MODEL


def model_file_tag(model: str) -> str:
    """Short model name for result file names: ``<CT>_Segmentation_<tag>.nii.gz``.

    The model name in the URL is the full nnU-Net folder name
    (``<Trainer>__<Plans>__<configuration>``), too long for a file name. The
    nnU-Net preset is ``nnUNet``; another nnU-Net name keeps only its trainer
    part; any other name is used as it is.
    """
    model = (model or "").strip().strip("/")
    if model == ENGINE_NNUNET_MODEL:
        return "nnUNet"
    return model.split("__", 1)[0] or model


def _clean_ip(value: str) -> str:
    """Reduce input to a bare IP/host, dropping any scheme, port, or path.

    Tolerant so a pasted ``http://host:port/path`` still yields just ``host``;
    the canonical stored form is e.g. ``localhost`` or ``202.122.49.242``.
    """
    v = (value or "").strip()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0].strip()
    v = v.split(":", 1)[0].strip()  # drop any :port
    return v or _DEFAULT_ENGINE_IP


def _clean_port(value) -> str:
    """Keep digits only; fall back to the default port if empty/invalid."""
    digits = "".join(ch for ch in str(value or "").strip() if ch.isdigit())
    return digits or _DEFAULT_ENGINE_PORT


def _clean_model(value: str) -> str:
    """Model name as a URL path segment: trimmed, no surrounding slashes."""
    return (value or "").strip().strip("/").strip() or ENGINE_NNUNET_MODEL


def _initial(key: str, default: str, cleaner) -> str:
    """Resolve a startup value: GUI-saved value > env default."""
    saved = app_settings().value(key, "", type=str)
    return cleaner(saved) if saved else cleaner(default)


# Current engine IP/port. Mutable so the GUI can change them at runtime.
_engine_ip = _initial(_SETTINGS_IP_KEY, _DEFAULT_ENGINE_IP, _clean_ip)
_engine_port = _initial(_SETTINGS_PORT_KEY, _DEFAULT_ENGINE_PORT, _clean_port)
_engine_model = _initial(_SETTINGS_MODEL_KEY, ENGINE_NNUNET_MODEL, _clean_model)


def get_engine_ip() -> str:
    """Return the configured engine IP/host (no scheme, no port)."""
    return _engine_ip


def get_engine_port() -> str:
    """Return the configured engine port."""
    return _engine_port


def set_engine_endpoint(ip: str, port) -> tuple[str, str]:
    """Update the engine IP + port at runtime and persist them. Returns ``(ip, port)``."""
    global _engine_ip, _engine_port
    _engine_ip = _clean_ip(ip)
    _engine_port = _clean_port(port)
    s = app_settings()
    s.setValue(_SETTINGS_IP_KEY, _engine_ip)
    s.setValue(_SETTINGS_PORT_KEY, _engine_port)
    return _engine_ip, _engine_port


def get_engine_model() -> str:
    """Return the configured model name (the last URL path segment)."""
    return _engine_model


def set_engine_model(model: str) -> str:
    """Update the model at runtime and persist it. Returns the cleaned name."""
    global _engine_model
    _engine_model = _clean_model(model)
    app_settings().setValue(_SETTINGS_MODEL_KEY, _engine_model)
    return _engine_model


def get_engine_url(model: str | None = None) -> str:
    """Full endpoint = scheme://ip:port/model (``model`` defaults to the configured one)."""
    return f"{ENGINE_NNUNET_SCHEME}://{_engine_ip}:{_engine_port}/{quote(model or _engine_model, safe='/')}"


# Global inference lock — only 1 engine can infer at a time
_inference_lock = threading.Lock()

# Timeout for HTTP requests (seconds) — inference can be slow
HTTP_TIMEOUT = 600.0
