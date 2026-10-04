import os
import sys

# Thread counts and the GL backend are read when numpy/Qt load, so set them
# before those imports. setdefault keeps any value the launcher scripts gave.
_CORES = str(os.cpu_count() or 4)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _CORES)
os.environ.setdefault("QT_OPENGL", "desktop")

from .core.config import is_frozen, user_app_dir

_LOG_PATH = None


def _setup_frozen_runtime():
    """The installed app has no console and no launcher script: log to a file.

    Must run before numba is imported (it reads NUMBA_CACHE_DIR at import).
    """
    global _LOG_PATH
    import faulthandler

    app_dir = user_app_dir()
    log_dir = app_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = log_dir / "app.log"
    try:
        if _LOG_PATH.exists():
            _LOG_PATH.replace(log_dir / "app.prev.log")
    except OSError:
        # Another instance still has app.log open (Windows locks it).
        _LOG_PATH = log_dir / f"app.{os.getpid()}.log"

    log = open(_LOG_PATH, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = log
    faulthandler.enable(log)        # native crashes (e.g. GL driver) land in the log too
    sys.excepthook = _excepthook

    # numba can't write its cache next to the frozen sources.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(app_dir / "numba_cache"))

    if sys.platform == "win32":
        # Same as `start /high` in scripts/start_app.bat.
        import ctypes
        HIGH_PRIORITY_CLASS = 0x80
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)


_showing_error = False


def _excepthook(exc_type, exc, tb):
    """Log uncaught errors and tell the user, instead of failing silently."""
    global _showing_error
    import traceback
    traceback.print_exception(exc_type, exc, tb)

    from PyQt6.QtWidgets import QApplication, QMessageBox
    if QApplication.instance() is None or _showing_error:
        return
    _showing_error = True
    try:
        QMessageBox.critical(
            None, "PET/CT App – Error",
            f"{exc_type.__name__}: {exc}\n\nDetails were written to:\n{_LOG_PATH}",
        )
    finally:
        _showing_error = False


if is_frozen():
    _setup_frozen_runtime()

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont
from PyQt6.QtCore import Qt
from .gui.main_window import MainWindow
from .gui.components.storage_location import prompt_first_run
from .database.db import init_db


def _make_splash_pixmap() -> QPixmap:
    pix = QPixmap(480, 110)
    pix.fill(QColor("#1e1e1e"))
    p = QPainter(pix)
    p.setPen(QColor("#cccccc"))
    p.setFont(QFont("Arial", 14))
    p.drawText(
        pix.rect(),
        Qt.AlignmentFlag.AlignCenter,
        "Metabolic Lesion Quantification\nLoading…",
    )
    p.end()
    return pix


def main():
    # Initialize Application
    app = QApplication(sys.argv)

    # First launch: ask where the database lives, then open it
    if not prompt_first_run():
        sys.exit(0)
    init_db()

    # Show splash + spinning cursor while napari viewers initialise (~5 s)
    splash = QSplashScreen(_make_splash_pixmap(), Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.setOverrideCursor(Qt.CursorShape.WaitCursor)
    app.processEvents()

    window = MainWindow()
    window.show()

    app.restoreOverrideCursor()
    splash.finish(window)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
