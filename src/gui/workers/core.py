import os
import threading

from dotenv import load_dotenv
from PyQt6.QtCore import QSettings

# Load a local .env (if present) so the engine host can be set without code edits.
load_dotenv()

# ──── Configuration ────

# Scheme + model are fixed at app startup (env-overridable). The GUI configures
# the IP and port; their env values here are only the defaults shown on first run.
ENGINE_NNUNET_SCHEME = os.getenv("ENGINE_NNUNET_SCHEME", "http")
ENGINE_NNUNET_MODEL = os.getenv(
    "ENGINE_NNUNET_MODEL",
    "nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres",
)

# Defaults used when nothing is configured ("localhost" for a local engine).
_DEFAULT_ENGINE_IP = os.getenv("ENGINE_NNUNET_IP", "localhost")
_DEFAULT_ENGINE_PORT = os.getenv("ENGINE_NNUNET_PORT", "8104")

# QSettings location for persisting the GUI-configured IP/port across runs.
_SETTINGS_ORG = "PETCTApp"
_SETTINGS_APP = "PETCTApp"
_SETTINGS_IP_KEY = "engine/nnunet_ip"
_SETTINGS_PORT_KEY = "engine/nnunet_port"


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


def _initial(key: str, default: str, cleaner) -> str:
    """Resolve a startup value: GUI-saved value > env default."""
    saved = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(key, "", type=str)
    return cleaner(saved) if saved else cleaner(default)


# Current engine IP/port. Mutable so the GUI can change them at runtime.
_engine_ip = _initial(_SETTINGS_IP_KEY, _DEFAULT_ENGINE_IP, _clean_ip)
_engine_port = _initial(_SETTINGS_PORT_KEY, _DEFAULT_ENGINE_PORT, _clean_port)


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
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(_SETTINGS_IP_KEY, _engine_ip)
    s.setValue(_SETTINGS_PORT_KEY, _engine_port)
    return _engine_ip, _engine_port


def get_engine_url() -> str:
    """Full endpoint = scheme://ip:port/model. Scheme/model fixed at startup."""
    return f"{ENGINE_NNUNET_SCHEME}://{_engine_ip}:{_engine_port}/{ENGINE_NNUNET_MODEL}"


# Global inference lock — only 1 engine can infer at a time
_inference_lock = threading.Lock()

# Timeout for HTTP requests (seconds) — inference can be slow
HTTP_TIMEOUT = 600.0
