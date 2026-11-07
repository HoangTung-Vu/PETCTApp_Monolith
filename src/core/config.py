import os
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings

# QSettings location shared by everything the app persists (engine endpoint;
# the data folder only for pre-1.2.1 Windows installs). On Windows this is
# HKCU\Software\MetLePET\MetLePET.
SETTINGS_ORG = "MetLePET"
SETTINGS_APP = "MetLePET"
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
    return base / "MetLePET"


def default_storage_dir() -> Path:
    """Folder of the session database (petct.db), fixed — never asked for.

    The installed app uses the per-user app folder (~/.local/share/MetLePET on
    Linux, %LOCALAPPDATA%\\MetLePET on Windows): outside the install dir, so a
    fresh install starts an empty database there and an update or reinstall
    keeps it. From source it is ``<repo>/storage``, so a dev run never shares
    the installed app's database.
    """
    if is_frozen():
        return user_app_dir()
    return Path(__file__).resolve().parent.parent.parent / "storage"


def _legacy_windows_storage_dir() -> Path | None:
    """Data folder picked at first run by Windows installs up to v1.2.0.

    Those versions asked for the folder and kept the answer in the registry.
    It is still used while it holds a database, so updating such an install
    opens the same sessions instead of an empty database.
    """
    if sys.platform != "win32" or not is_frozen():
        return None
    value = app_settings().value(_STORAGE_DIR_KEY, "", type=str)
    if value and (Path(value) / DB_FILENAME).is_file():
        return Path(value)
    return None


def storage_dir_override() -> Path | None:
    """``PETCT_STORAGE_DIR`` wins over everything (tests, scripted setups)."""
    value = os.getenv("PETCT_STORAGE_DIR")
    return Path(value) if value else None


class Settings:
    def __init__(self):
        # Base directory
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.reload()

    def reload(self):
        """Resolve the data folder: env override > (Windows) pre-1.2.1 choice > fixed default.

        Nothing is created here; ``init_db`` and ``FileManager`` make the
        folders when they first write into them.
        """
        self.STORAGE_DIR = (storage_dir_override() or _legacy_windows_storage_dir()
                            or default_storage_dir())

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
