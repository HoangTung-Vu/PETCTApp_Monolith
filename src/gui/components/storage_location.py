"""Choosing the data folder that holds the session database (petct.db).

The choice is stored in QSettings. It is asked for once on first launch, and
can be changed from the Workflow tab; a change takes effect after a restart
because the open database connection and session list belong to the old folder.
"""

import sqlite3
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ...core.config import (
    DB_FILENAME, check_writable, default_storage_dir, is_frozen,
    save_storage_dir, saved_storage_dir, settings, storage_dir_override,
)

_TITLE = "PET/CT App – Data folder"


def _pick_folder(parent: QWidget | None, start: Path) -> Path | None:
    folder = QFileDialog.getExistingDirectory(parent, "Choose data folder", str(start))
    return Path(folder) if folder else None


def _writable(parent: QWidget | None, folder: Path) -> bool:
    error = check_writable(folder)
    if error:
        QMessageBox.warning(parent, _TITLE, f"{error}\n\nPlease choose another folder.")
        return False
    return True


def prompt_first_run(parent: QWidget | None = None) -> bool:
    """Ask for the data folder if none is configured yet.

    Returns False if the user chose to quit instead.
    """
    if storage_dir_override() or saved_storage_dir():
        return True

    folder = default_storage_dir()
    while True:
        if (folder / DB_FILENAME).exists():
            what = "An existing database (petct.db) was found here and will be used."
        else:
            what = "A new session database (petct.db) will be created here."

        box = QMessageBox(parent)
        box.setWindowTitle(_TITLE)
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Where should PET/CT App keep its session database?")
        box.setInformativeText(
            f"{folder}\n\n{what}\n\n"
            "Use a folder on a local disk. You can change it later in the Workflow tab."
        )
        btn_use = box.addButton("Use this folder", QMessageBox.ButtonRole.AcceptRole)
        btn_other = box.addButton("Choose another…", QMessageBox.ButtonRole.ActionRole)
        btn_quit = box.addButton("Quit", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_use)
        box.exec()

        clicked = box.clickedButton()
        if clicked is btn_quit:
            return False
        if clicked is btn_other:
            # Show the prompt again with the new folder so the user confirms it.
            folder = _pick_folder(parent, folder) or folder
            continue
        if _writable(parent, folder):
            save_storage_dir(folder)
            settings.reload()
            return True


def _copy_database(src: Path, dst: Path) -> None:
    """Copy via SQLite's backup API, which is consistent while ``src`` is open."""
    src_conn = sqlite3.connect(src)
    dst_conn = sqlite3.connect(dst)
    try:
        src_conn.backup(dst_conn)
    except Exception:
        dst_conn.close()
        dst.unlink(missing_ok=True)
        raise
    finally:
        dst_conn.close()
        src_conn.close()


def _restart(parent: QWidget) -> None:
    # close() runs MainWindow.closeEvent, which may be cancelled from the
    # unsaved-segmentation prompt; only relaunch once the window really closed.
    if not parent.window().close():
        return
    if is_frozen():
        QProcess.startDetached(sys.executable, sys.argv[1:])
    else:
        QProcess.startDetached(
            sys.executable, ["-m", "src.main", *sys.argv[1:]], str(settings.BASE_DIR)
        )


def change_location(parent: QWidget) -> Path | None:
    """Let the user move to another data folder. Returns the new folder, or None."""
    if storage_dir_override():
        QMessageBox.information(
            parent, _TITLE,
            "The data folder is set by the PETCT_STORAGE_DIR environment variable "
            "and can't be changed here.",
        )
        return None

    current = settings.STORAGE_DIR
    folder = _pick_folder(parent, current)
    if folder is None or folder.resolve() == current.resolve():
        return None
    if not _writable(parent, folder):
        return None

    target_db = folder / DB_FILENAME
    if target_db.exists():
        reply = QMessageBox.question(
            parent, _TITLE,
            f"{folder} already contains a database (petct.db).\n\n"
            "The app will use the sessions stored there instead of the current ones. Continue?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return None
    elif settings.DB_PATH.exists():
        box = QMessageBox(parent)
        box.setWindowTitle(_TITLE)
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("The new folder has no database yet.")
        box.setInformativeText(
            "Copy the current database there, so existing sessions come along, "
            "or start with an empty one?"
        )
        btn_copy = box.addButton("Copy current database", QMessageBox.ButtonRole.AcceptRole)
        btn_empty = box.addButton("Start empty", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(btn_copy)
        box.exec()

        clicked = box.clickedButton()
        if clicked is btn_copy:
            try:
                _copy_database(settings.DB_PATH, target_db)
            except Exception as e:
                QMessageBox.critical(parent, _TITLE, f"Could not copy the database:\n{e}")
                return None
        elif clicked is not btn_empty:
            return None

    save_storage_dir(folder)
    print(f"[Storage] Data folder set to {folder} (applies after restart)")

    reply = QMessageBox.question(
        parent, _TITLE,
        "The new data folder will be used after the app restarts.\n\nRestart now?",
    )
    if reply == QMessageBox.StandardButton.Yes:
        _restart(parent)
    return folder
