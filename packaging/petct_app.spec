# PyInstaller spec for the PET/CT desktop app (one-folder build).
#
#   uv sync --group build
#   uv run pyinstaller packaging/petct_app.spec --noconfirm --clean
#
# Output: dist/MetLePET/MetLePET(.exe). One-folder rather than one-file: the
# napari/Qt bundle is several hundred MB and would be unpacked on every launch.
# The spec is platform-neutral so it can be tried on Linux before CI builds it
# on Windows.

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent
ICON = ROOT / "packaging" / "assets" / "petct.ico"


def _not_tests(name):
    parts = name.split(".")
    return "_tests" not in parts and "tests" not in parts and "benchmarks" not in parts


datas, binaries, hiddenimports = [], [], []

# napari and its UI stack load modules, shaders, icons and .pyi lazy-loader
# stubs at runtime, which import analysis alone doesn't see.
for pkg in ("napari", "napari_builtins", "napari_svg", "npe2", "vispy",
            "app_model", "magicgui", "superqt"):
    d, b, h = collect_all(pkg, filter_submodules=_not_tests,
                          exclude_datas=["**/_tests/**", "**/tests/**"])
    datas += d
    binaries += b
    hiddenimports += h

# npe2 discovers napari's builtin plugins through package metadata (entry
# points), and napari reads versions of its dependencies the same way.
# napari-console's metadata is dropped with its module (excluded below), or
# npe2 would log a failed plugin import on every start.
datas += [d for d in copy_metadata("napari", recursive=True)
          if "napari_console" not in Path(d[0]).name]

if sys.platform.startswith("linux"):
    # PyOpenGL imports its platform module when napari first draws, picked from the
    # session: EGL under Wayland (XDG_SESSION_TYPE=wayland, WAYLAND_DISPLAY), GLX
    # under X11. PyInstaller's OpenGL hook only adds the GLX one on Linux, so in a
    # Wayland session the 1.2.2 bundle logged "No module named
    # 'OpenGL.platform.egl'" and the 3D view failed (a source run has both).
    # run_petct.py --selfcheck loads both.
    hiddenimports += ["OpenGL.platform.egl", "OpenGL.platform.glx"]

# Not used by the app; dropping them saves a lot of space. If the self-check or
# a launch fails with ModuleNotFoundError for one of these, remove it here.
excludes = [
    "PyQt5", "PySide2", "PySide6",          # the app is PyQt6-only
    "tkinter",
    "napari_console", "IPython", "ipykernel", "qtconsole", "jupyter_client",
    "jupyter_core", "zmq",
    "napari_plugin_manager", "pip",
    "pytest", "_pytest",
]

a = Analysis(
    [str(ROOT / "packaging" / "run_petct.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
if sys.platform.startswith("linux"):
    # System libraries (GTK/GLib/GIO, X11/xcb, fonts, D-Bus, libstdc++, ...) come
    # from the machine the app runs on, as in a source run; the .deb declares
    # them in Depends (packaging/build_linux.sh). Copies taken from the build
    # machine break desktop integration on newer Ubuntu: GLib 2.72 from the
    # 22.04 runner can't load the system's dconf GIO module, so GTK falls back
    # to its light default theme and the app ignores the dark desktop setting.
    _system_lib = re.compile(r"^/(usr/)?lib(32|64)?/")
    a.binaries = [b for b in a.binaries if not _system_lib.match(b[1])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MetLePET",
    console=False,
    upx=False,          # UPX breaks Qt DLLs and trips antivirus scanners
    icon=str(ICON) if ICON.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="MetLePET",
)
