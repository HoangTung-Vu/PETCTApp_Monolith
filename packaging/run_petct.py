"""PyInstaller entry point for the desktop app.

``src.main`` uses relative imports, so it can't be the frozen script itself.
``--selfcheck`` imports and exercises the bundled libraries without opening the
GUI, then exits 0/1. The CI build runs it because its runners have no OpenGL
driver to open the napari canvases with.
"""

import multiprocessing
import sys


def _selfcheck() -> int:
    import tempfile
    import traceback
    from pathlib import Path

    try:
        import httpx, nibabel, pydicom, scipy.ndimage, skimage.measure, sqlalchemy  # noqa: F401
        import SimpleITK  # noqa: F401
        import napari  # noqa: F401
        import npe2
        from napari.utils.colormaps import ensure_colormap
        from PyQt6.QtWidgets import QApplication

        # Qt platform plugin (qwindows.dll) must load.
        app = QApplication(sys.argv[:1])

        # napari finds its builtin readers/writers through package metadata.
        pm = npe2.PluginManager.instance()
        pm.discover()
        pm.get_manifest("napari")
        ensure_colormap("hot")

        from src.database.db import init_db
        with tempfile.TemporaryDirectory() as tmp:
            init_db(Path(tmp) / "selfcheck.db")
            import src.database.db as dbm
            dbm.engine.dispose()

        print(f"[selfcheck] OK — napari {napari.__version__}, plugins: {sorted(pm._manifests)}")
        app.quit()
        return 0
    except Exception:
        traceback.print_exc()
        print("[selfcheck] FAILED")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Importing src.main also sets up logging and crash handling for the frozen app.
    from src.main import main

    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    main()
