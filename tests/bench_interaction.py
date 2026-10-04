"""Wall-clock benchmark of the GUI hot paths, on synthetic data only.

Runs a real ``MainWindow`` (needs a display) against a throw-away SQLite DB and
times the interactions reported as slow: session load / switch, slice
scrolling, crosshair moves, brush painting, eraser + undo, threshold-preview
slider ticks and confirm-and-save. With ``--with-3d`` the edit paths are timed
a second time after the 3D view has been opened once.

Usage:
    .venv/bin/python tests/bench_interaction.py [--shape X Y Z] [--with-3d]
                                                [--out results.json]

Synthetic volumes are cached in ``$PETCT_BENCH_DIR`` (default: <tmp>/petct_bench).
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(os.getenv("PETCT_BENCH_DIR", Path(tempfile.gettempdir()) / "petct_bench"))

# Lesion centres in nibabel XYZ voxel fractions (lesion 0 is erased, 1 is thresholded).
_LESIONS = [(0.35, 0.45, 0.5), (0.65, 0.55, 0.45), (0.5, 0.4, 0.7), (0.45, 0.6, 0.25)]
_LESION_R = 8


# ── Synthetic data ───────────────────────────────────────────────────────────

def lesion_centres(shape):
    return [tuple(int(f * s) for f, s in zip(c, shape)) for c in _LESIONS]


def make_data(shape) -> dict:
    """Write LAS CT/PET/mask .nii.gz once per shape; return their paths."""
    tag = "x".join(map(str, shape))
    paths = {k: BENCH_DIR / f"bench_{tag}_{k}.nii.gz" for k in ("ct", "pet", "seg")}
    if all(p.exists() for p in paths.values()):
        return paths
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bench] generating synthetic volumes {shape} in {BENCH_DIR} ...")

    X, Y, Z = shape
    rng = np.random.default_rng(0)
    xx = np.linspace(-1, 1, X, dtype=np.float32)[:, None]
    yy = np.linspace(-1, 1, Y, dtype=np.float32)[None, :]
    body = (xx / 0.8) ** 2 + (yy / 0.6) ** 2 < 1.0          # (X, Y)

    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[body] = 40.0
    ct += rng.normal(0, 15, size=shape).astype(np.float32)

    pet = np.zeros(shape, dtype=np.float32)
    pet[body] = 1.0
    pet += np.abs(rng.normal(0, 0.1, size=shape)).astype(np.float32)

    seg = np.zeros(shape, dtype=np.uint8)
    r = _LESION_R
    zz, yg, xg = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (zz ** 2 + yg ** 2 + xg ** 2) <= r * r
    for cx, cy, cz in lesion_centres(shape):
        sl = (slice(cx - r, cx + r + 1), slice(cy - r, cy + r + 1), slice(cz - r, cz + r + 1))
        pet[sl][ball] = 8.0
        seg[sl][ball] = 1

    affine = np.diag([-0.98, 0.98, 2.0, 1.0])               # LAS
    for key, arr in (("ct", ct), ("pet", pet), ("seg", seg)):
        nib.save(nib.Nifti1Image(arr, affine), paths[key])
    return paths


# ── Harness ──────────────────────────────────────────────────────────────────

def isolate_storage(tmp: Path):
    """Point the DB and the session storage at a throw-away directory."""
    import src.database.db as dbm

    dbm.init_db(tmp / "bench.db")
    from src.core.config import settings
    settings.DATA_DIR = tmp / "data"
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)


class Bench:
    def __init__(self, app, profile=()):
        self.app = app
        self.results = {}
        self.profile = set(profile)
        self._prof = None

    def start(self, name):
        """Start timing ``name`` (and cProfile it when requested with --profile)."""
        if name in self.profile:
            import cProfile
            self._prof = cProfile.Profile()
            self._prof.enable()
        return time.perf_counter()

    def pump(self, ms=0):
        from PyQt6.QtCore import QEventLoop
        end = time.perf_counter() + ms / 1000.0
        while True:
            self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            if time.perf_counter() >= end:
                return

    def wait(self, cond, timeout=900.0):
        t0 = time.perf_counter()
        while not cond():
            self.pump()
            time.sleep(0.002)
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("bench step timed out")

    def record(self, name, seconds, n=1):
        if self._prof is not None:
            import pstats
            self._prof.disable()
            print(f"[bench] ---- profile of {name} ----")
            pstats.Stats(self._prof).sort_stats("cumulative").print_stats(35)
            self._prof = None
        self.results[name] = {"total_ms": seconds * 1000.0, "per_op_ms": seconds * 1000.0 / n, "n": n}
        print(f"[bench] {name:<34} {seconds * 1000.0:10.1f} ms   ({seconds * 1000.0 / n:8.2f} ms/op, n={n})")


class FakeMouseEvent:
    def __init__(self, position):
        self.position = position


def run(args):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtCore import QPointF, QPoint, Qt

    shape = tuple(args.shape)
    paths = make_data(shape)

    tmp = Path(tempfile.mkdtemp(prefix="petct_bench_db_"))
    isolate_storage(tmp)

    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.main_window import MainWindow

    b = Bench(app, args.profile)
    win = MainWindow()
    win.show()
    win.resize(1600, 1000)
    b.pump(500)

    # Never block on the unsaved-changes dialog.
    win._prompt_unsaved_segmentation = lambda context: True

    loads = {"n": 0}
    orig_refresh = win._do_refresh_after_load

    def counted_refresh():
        orig_refresh()
        loads["n"] += 1
    win._do_refresh_after_load = counted_refresh

    def wait_load(prev):
        b.wait(lambda: loads["n"] > prev)
        b.pump(200)                                   # let the first paints land

    lm = win.layout_manager
    sm = win.session_manager

    # ── 1. Session load (NIfTI CT + PET + mask) ──
    t0 = b.start("load_session_first")
    win._update_session_files(ct_path=paths["ct"], pet_path=paths["pet"], tumor_seg_path=paths["seg"])
    wait_load(0)
    b.record("load_session_first", time.perf_counter() - t0)
    first_id = sm.current_session_id

    # Second session, then switch back to the first one.
    win.create_new_session("Bench Doctor", "Bench Patient")
    wait_load(1)
    win._update_session_files(ct_path=paths["ct"], pet_path=paths["pet"], tumor_seg_path=paths["seg"])
    wait_load(2)
    t0 = b.start("switch_session")
    win.load_existing_session(first_id)
    wait_load(3)
    b.record("switch_session", time.perf_counter() - t0)

    # ── 2. Six-view layout ──
    views = ["axial_ct", "axial_pet", "axial_overlay", "coronal_ct", "coronal_pet", "sagittal_overlay"]
    t0 = b.start("set_6_views")
    lm.set_active_views(views)
    b.pump(200)
    b.record("set_6_views", time.perf_counter() - t0)

    # ── 3. Wheel scrolling in the axial CT view ──
    canvas = lm._fixed_view_map["axial_ct"].qt_viewer.canvas.native
    n = 50
    t0 = b.start("wheel_scroll_axial")
    for i in range(n):
        delta = -120 if i < n // 2 else 120
        ev = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, delta),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(canvas, ev)
        b.pump()
    b.record("wheel_scroll_axial", time.perf_counter() - t0, n)

    # ── 4. Crosshair moves (click/drag path) ──
    Zn, Yn, Xn = shape[2], shape[1], shape[0]
    t0 = b.start("crosshair_move")
    for i in range(n):
        lm._on_viewer_crosshair_click([Zn / 2 + (i % 10) - 5, Yn / 2 + i % 7, Xn / 2 - i % 9])
        b.pump()
    b.record("crosshair_move", time.perf_counter() - t0, n)

    def edit_paths(suffix):
        # ── 5. Brush painting on the tumor layer (Refine tab) ──
        win.control_panel.tabs.setCurrentIndex(win._TAB_REFINE)
        b.wait(lambda: sm.roi_mask is not None and win.control_panel.tabs.isEnabled())
        b.pump(200)
        win._on_manual_edit_tool("paint")
        vw = lm._fixed_view_map["axial_ct"]
        layer = vw.viewer.layers[vw.LAYER_NAMES["tumor"]]
        z = float(vw.viewer.dims.current_step[0])
        t0 = b.start(f"brush_paint{suffix}")
        for i in range(n):
            layer.paint((z, Yn * 0.3 + i, Xn * 0.3 + i), 1)
            b.pump()
        b.pump(400)                                   # debounced auto-sync
        b.record(f"brush_paint{suffix}", time.perf_counter() - t0, n)
        win._on_manual_edit_tool("pan_zoom")

        # ── 6. Threshold preview slider ticks ──
        cx, cy, cz = lesion_centres(shape)[1]
        roi_zyx = lm.get_active_mask_data_zyx("roi")
        zc, yc, xc = Zn - 1 - cz, Yn - 1 - cy, cx
        r = _LESION_R + 4
        roi_zyx[zc - r:zc + r, yc - r:yc + r, xc - r:xc + r] = 1
        for v in lm._get_visible_viewers():
            if vw.LAYER_NAMES["roi"] in v.viewer.layers:
                v.viewer.layers[vw.LAYER_NAMES["roi"]].refresh()
        win._on_refine_adaptive(0.7, "outside_isocontour", 3)
        b.wait(lambda: getattr(win, "_current_preview_dialog", None) is not None)
        b.pump(200)
        t0 = b.start(f"threshold_slider_tick{suffix}")
        for i in range(30):
            win._update_component_preview(2.0 + 0.1 * i)
            b.pump()
        b.record(f"threshold_slider_tick{suffix}", time.perf_counter() - t0, 30)
        win._current_preview_dialog.reject()
        b.pump(200)

        # ── 7. Confirm & save (merge ROI into tumor, write to disk) ──
        t0 = b.start(f"confirm_and_save{suffix}")
        win._on_confirm_and_save()
        b.wait(lambda: not win._merge_save_worker.isRunning())
        b.pump(100)
        b.record(f"confirm_and_save{suffix}", time.perf_counter() - t0)

        # ── 8. Eraser double-click on a lesion, then undo ──
        win.control_panel.tabs.setCurrentIndex(win._TAB_ERASER)
        b.pump(200)
        win.control_panel.eraser_tab.btn_eraser_toggle.setChecked(True)
        b.pump(100)
        ex, ey, ez = lesion_centres(shape)[0]
        scale = vw._scale_zyx
        pos = ((Zn - 1 - ez) * scale[0], (Yn - 1 - ey) * scale[1], ex * scale[2])
        before = len(win._eraser_undo_stack)
        t0 = b.start(f"eraser_click{suffix}")
        vw._eraser_callback(vw.viewer, FakeMouseEvent(pos))
        b.wait(lambda: len(win._eraser_undo_stack) > before)
        b.pump()
        b.record(f"eraser_click{suffix}", time.perf_counter() - t0)

        t0 = b.start(f"eraser_undo{suffix}")
        win._on_eraser_undo()
        b.pump()
        b.record(f"eraser_undo{suffix}", time.perf_counter() - t0)
        win.control_panel.eraser_tab.btn_eraser_toggle.setChecked(False)
        win.control_panel.tabs.setCurrentIndex(win._TAB_WORKFLOW)
        b.pump(200)

    edit_paths("")

    if args.with_3d:
        t0 = b.start("open_3d")
        lm.set_view_mode("3d")
        b.pump(300)
        b.record("open_3d", time.perf_counter() - t0)
        lm.set_active_views(views)
        b.pump(300)
        edit_paths("_after_3d")

    win._prompt_unsaved_segmentation = lambda context: True
    win.close()
    b.pump(200)

    if args.out:
        Path(args.out).write_text(json.dumps({"shape": shape, "results": b.results}, indent=2))
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=3, default=[512, 512, 600])
    p.add_argument("--with-3d", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--profile", nargs="*", default=[], help="step names to cProfile")
    run(p.parse_args())
