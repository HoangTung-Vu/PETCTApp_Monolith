# PyInstaller spec for the PET/CT desktop app (one-folder build).
#
#   uv sync --group build
#   uv run pyinstaller packaging/petct_app.spec --noconfirm --clean
#
# Output: dist/PETCTApp/PETCTApp(.exe). One-folder rather than one-file: the
# napari/Qt bundle is several hundred MB and would be unpacked on every launch.
# The spec is platform-neutral so it can be tried on Linux before CI builds it
# on Windows.

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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PETCTApp",
    console=False,
    upx=False,          # UPX breaks Qt DLLs and trips antivirus scanners
    icon=str(ICON) if ICON.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="PETCTApp",
)
