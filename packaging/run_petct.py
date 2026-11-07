"""PyInstaller entry point for the desktop app.

``src.main`` uses relative imports, so it can't be the frozen script itself.
``--selfcheck`` imports and exercises the bundled libraries without opening the
GUI, then exits 0/1. The CI build runs it because its runners have no OpenGL
driver to open the napari canvases with.
"""

import multiprocessing
import sys


def _check_pyopengl_platforms() -> str:
    """Load PyOpenGL's Wayland (EGL) and X11 (GLX) platforms and the system GL
    libraries each one opens; raise if one is missing.

    PyOpenGL picks one from the session when napari first draws, so a check run
    without a display (CI) would otherwise test neither. The bundle must carry both
    modules (petct_app.spec) and the .deb must pull in the libraries (Depends).
    """
    from OpenGL.plugins import PlatformPlugin

    found = []
    for session, needs in (("wayland", ("GL", "EGL")), ("x11", ("GL",))):
        cls = PlatformPlugin.by_name(session).load()  # None if the module is missing
        if cls is None:
            raise RuntimeError(f"PyOpenGL platform for {session} sessions is not in the bundle")
        platform = cls()
        libs = []
        for name in needs:
            lib = getattr(platform, name)  # loads it (raises ImportError if absent)
            path = getattr(lib, "_name", "") or ""
            # Without libOpenGL/libGL the EGL platform silently falls back to GLES,
            # which vispy can't draw with.
            if lib is None or "GLES" in path:
                raise RuntimeError(f"{cls.__name__}: no {name} library (got {path or lib})")
            libs.append(path)
        found.append(f"{session}={cls.__name__}({', '.join(libs)})")
    return "; ".join(found)


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

        opengl = _check_pyopengl_platforms() if sys.platform.startswith("linux") else "n/a"

        print(f"[selfcheck] OK — napari {napari.__version__}, plugins: {sorted(pm._manifests)}, "
              f"OpenGL: {opengl}")
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
