import os
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings

# QSettings location shared by everything the app persists (engine endpoint,
# data folder). On Windows this is HKCU\Software\PETCTApp\PETCTApp.
SETTINGS_ORG = "PETCTApp"
SETTINGS_APP = "PETCTApp"
_STORAGE_DIR_KEY = "storage/dir"

DB_FILENAME = "petct.db"


def app_settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def is_frozen() -> bool:
    """True when running from the PyInstaller build rather than from source."""
    return getattr(sys, "frozen", False)


def user_app_dir() -> Path:
    """Per-user writable folder for logs/caches (never inside the install dir)."""
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "PETCTApp"


def default_storage_dir() -> Path:
    """Data folder proposed on first run.

    From source this stays ``<repo>/storage`` so an existing dev database keeps
    working; the installed app can't write next to itself, so it uses the
    per-user app folder instead.
    """
    if is_frozen():
        return user_app_dir()
    return Path(__file__).resolve().parent.parent.parent / "storage"


def saved_storage_dir() -> Path | None:
    value = app_settings().value(_STORAGE_DIR_KEY, "", type=str)
    return Path(value) if value else None


def save_storage_dir(path: Path) -> None:
    app_settings().setValue(_STORAGE_DIR_KEY, str(Path(path)))


def storage_dir_override() -> Path | None:
    """``PETCT_STORAGE_DIR`` wins over the saved choice (tests, scripted setups)."""
    value = os.getenv("PETCT_STORAGE_DIR")
    return Path(value) if value else None


def check_writable(path: Path) -> str | None:
    """Return an error message if the app can't keep its database in ``path``.

    SQLite needs to create journal files next to the database, so the folder
    itself must be writable, not just the .db file.
    """
    probe = Path(path) / ".petct_write_test"
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        return f"Cannot write to {path}:\n{e.strerror or e}"
    return None


class Settings:
    def __init__(self):
        # Base directory
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.reload()

    def reload(self):
        """Resolve the data folder: env override > saved choice > default.

        Nothing is created here; ``init_db`` and ``FileManager`` make the
        folders when they first write into them.
        """
        self.STORAGE_DIR = storage_dir_override() or saved_storage_dir() or default_storage_dir()

        # Subdirectories
        self.WEIGHTS_DIR = self.STORAGE_DIR / "weights"
        self.DATA_DIR = self.STORAGE_DIR / "data"  # Sessions and nii.gz files
        self.DB_PATH = self.STORAGE_DIR / DB_FILENAME

    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a specific session."""
        session_dir = self.DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

settings = Settings()
