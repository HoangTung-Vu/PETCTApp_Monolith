"""Wall-clock benchmark of the GUI hot paths, on synthetic data only.

Runs a real ``MainWindow`` (needs a display) against a throw-away SQLite DB and
times the interactions reported as slow: session load / switch, slice
scrolling, crosshair moves, brush painting, island erase + undo, threshold-preview
slider ticks and confirm-and-save — plus the lesion quantification paths
(values after load / save, double-click info box toggle, report export). With ``--with-3d`` the edit paths are timed
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
    def __init__(self, position, button=1):
        self.position = position
        self.button = button


def check(ok, what):
    print(f"[bench] check {'OK  ' if ok else 'FAIL'} {what}")
    if not ok:
        CHECK_FAILURES.append(what)


CHECK_FAILURES = []


def overlay_fresh(lm, mask_type) -> bool:
    """Every visible 2D view's smooth / contour overlay draws what its Labels
    layer holds now: the cached slice paths equal a rebuild from ``layer.data``
    (a stale cache means an edit path ended without a napari event)."""
    ok = True
    for v in lm._get_visible_viewers():
        name = v.LAYER_NAMES[mask_type]
        overlay = v._mask_overlays[mask_type]
        if v.is_3d or name not in v.viewer.layers or not overlay._enabled:
            continue
        layer = v.viewer.layers[name]
        drawn = overlay._current_path(layer)
        overlay._path = None
        rebuilt = overlay._current_path(layer)
        ok &= drawn.keys() == rebuilt.keys() and all(drawn[k] == rebuilt[k] for k in rebuilt)
    return ok


def labels_fresh(lm, mask_type) -> bool:
    """Every visible 2D view shows the current mask: napari's displayed slice
    (what the texture is built from) equals the shared data. False is the
    "other views only update after a scroll" state."""
    ok = True
    for v in lm._get_visible_viewers():
        name = v.LAYER_NAMES[mask_type]
        if v.is_3d or name not in v.viewer.layers:
            continue
        sl = v.viewer.layers[name]._slice.image
        ok &= np.array_equal(np.asarray(sl.view),
                             np.asarray(v.viewer.layers[name]._raw_to_displayed(np.asarray(sl.raw))))
    return ok


def press_keys(lm, view_id, keys):
    """A key press in ``view_id`` (goes to its active layer's key bindings)."""
    lm._fixed_view_map[view_id].qt_viewer._key_map_handler.press_key(keys)


def overlay_mode_ok(lm, mask_type) -> bool:
    """2D: Labels transparent under an enabled overlay. 3D: plain Labels at the
    mask opacity (smooth / contour are 2D only)."""
    ok = True
    for v in lm._get_visible_viewers() + ([lm.viewer_3d] if lm._is_3d_loaded else []):
        name = v.LAYER_NAMES[mask_type]
        if name not in v.viewer.layers:
            continue
        layer = v.viewer.layers[name]
        opacity = v._roi_opacity if mask_type == "roi" else v._tumor_opacity
        if v.is_3d:
            ok &= abs(layer.opacity - opacity) < 1e-6
        else:
            ok &= layer.visible and layer.opacity == 0.0 and v._mask_overlays[mask_type]._enabled
    return ok


def lesion_id_paths(b, win, shape):
    """v1.3.0: Give lesion ids, the "same lesion?" prompt (no / yes), ROI merge
    into a selected id, island erase + undo of an id, and the organ layer shown after
    the tumor layer exists (the post-TotalSegmentator "Not Responding" hang)."""
    from src.core.lesion_labels import has_lesion_ids_marker

    lm, sm = win.layout_manager, win.session_manager
    rt = win.control_panel.refine_tab
    Zn, Yn, Xn = shape[2], shape[1], shape[0]

    def to_nap(v):
        x, y, z = v
        return (Zn - 1 - z, Yn - 1 - y, x)

    win.control_panel.tabs.setCurrentIndex(win._TAB_REFINE)
    b.wait(lambda: win.control_panel.tabs.isEnabled())
    b.pump(200)
    check(not sm.tumor_labeled and rt.btn_give_ids.isEnabled(),
          "binary mask: Give lesion ids enabled, single toggle")
    binary = [(l["id"], l["bbox"]) for l in sm.lesion_stats["lesions"]]

    # ── 10. "+ Another" on a binary mask → Give lesion ids, then a new id ──
    import src.gui.handlers.lesion_label_handler as llh
    from PyQt6.QtWidgets import QMessageBox
    asked_give = []
    llh.QMessageBox.question = staticmethod(
        lambda *a, **k: asked_give.append(a[1]) or QMessageBox.StandardButton.Yes)
    t0 = b.start("give_lesion_ids")
    rt._another_btn.click()
    b.wait(lambda: getattr(win, "_give_ids_worker", None) is not None
           and not win._give_ids_worker.isRunning())
    b.pump(100)
    b.record("give_lesion_ids", time.perf_counter() - t0)
    check(asked_give == ["New lesion id"], "binary + Another asked to give ids first")
    labelled = [(l["id"], l["bbox"]) for l in sm.lesion_stats["lesions"]]
    check(sm.tumor_labeled and sm.lesion_stats["labeled"] and labelled == binary,
          f"Give ids = the binary report's ids and lesions ({len(labelled)})")
    seg_path = sm.repository.get_by_id(sm.current_session_id).tumor_seg_path
    saved = nib.load(seg_path)
    check(saved.get_data_dtype() == np.int16 and has_lesion_ids_marker(saved.header),
          "saved as int16 with the lesion-id marker")
    n_ids = len(labelled)
    check(sorted(rt._label_buttons) == list(range(1, n_ids + 2)) and rt.selected_label() == n_ids + 1,
          f"one toggle per id + the new id {n_ids + 1}, selected")

    # ── 11. Brush of lesion 1 over lesion 2 → "same lesion?" ──
    answers = {"merge": False, "asked": []}

    def fake_ask(label_id, covered, what):
        answers["asked"].append((label_id, list(covered), what))
        return answers["merge"]
    win._ask_same_lesion = fake_ask

    les = {l["id"]: l for l in sm.lesion_stats["lesions"]}
    target = to_nap(les[2]["suvmax_voxel"])
    rt._select_label(1)
    win._on_manual_edit_tool("paint")
    vw = lm._fixed_view_map["axial_ct"]
    layer = vw.viewer.layers[vw.LAYER_NAMES["tumor"]]
    zyx = lm.get_active_mask_data_zyx("tumor")
    lm.jump_to_position(*[float(c) for c in target])
    b.pump(100)
    t0 = b.start("paint_over_other_lesion_no")
    layer.paint(target, 1)
    b.pump(100)
    b.record("paint_over_other_lesion_no", time.perf_counter() - t0)
    check(answers["asked"] and answers["asked"][-1][:2] == (1, [2]), "prompt asked about lesion 2")
    check(zyx[target] == 2, "No: lesion 2 kept its voxels")

    answers["merge"] = True
    n2 = int((zyx == 2).sum())
    t0 = b.start("paint_over_other_lesion_yes")
    layer.paint(target, 1)
    b.pump(100)
    b.record("paint_over_other_lesion_yes", time.perf_counter() - t0)
    check(n2 > 0 and not (zyx == 2).any(), f"Yes: all {n2} voxels of lesion 2 became lesion 1")
    win._on_manual_edit_tool("pan_zoom")
    t0 = b.start("save_merged")
    win._on_confirm_and_save()
    b.wait(lambda: not win._merge_save_worker.isRunning())
    b.pump(100)
    b.record("save_merged", time.perf_counter() - t0)
    ids_after = [l["id"] for l in sm.lesion_stats["lesions"]]
    check(2 not in ids_after and 1 in ids_after and len(ids_after) == len(labelled) - 1,
          f"merged lesion is one lesion in the values ({ids_after})")

    # ── 12. ROI thresholded into a new id from "+ Another", lesion under it kept ──
    check(n_ids + 1 not in rt._label_buttons, "an id never drawn is dropped at the next save")
    rt._another_btn.click()
    new_id = rt.selected_label()
    rt._another_btn.click()
    check(new_id == max(ids_after) + 1 and rt.selected_label() == new_id + 1
          and {new_id, new_id + 1} <= set(rt._label_buttons),
          f"each + Another creates a new id ({new_id}, {new_id + 1})")
    rt._label_buttons[new_id].click()
    les3 = to_nap(les[3]["suvmax_voxel"])
    roi_zyx = lm.get_active_mask_data_zyx("roi")
    r = _LESION_R + 4
    roi_zyx[les3[0] - r:les3[0] + r, les3[1] - r:les3[1] + r, les3[2] - r:les3[2] + r] = 1
    lm.refresh_mask("roi")
    win._sync_roi_from_viewer()                      # the red button merges the session ROI
    answers["merge"] = False
    n_asked = len(answers["asked"])
    win._on_confirm_and_save()
    b.wait(lambda: not win._merge_save_worker.isRunning())
    b.pump(100)
    zyx = lm.get_active_mask_data_zyx("tumor")
    check(len(answers["asked"]) == n_asked + 1 and answers["asked"][-1][:2] == (new_id, [3]),
          "ROI over lesion 3 asked once")
    check(zyx[les3] == 3 and (zyx == new_id).any(),
          "ROI filled background as the new id, lesion 3 kept")
    check(new_id in [l["id"] for l in sm.lesion_stats["lesions"]], "new id in the values")

    # ── 13. Island erase removes only the selected lesion; Undo restores its id ──
    win.control_panel.tabs.setCurrentIndex(win._TAB_REFINE)
    b.pump(200)
    rt.btn_manual_erase.click()
    rt._label_buttons[1].click()
    b.pump(50)
    scale = vw._scale_zyx
    pos3 = tuple(c * s_ for c, s_ in zip(les3, scale))
    vw._lesion_dblclick_cb(vw.viewer, FakeMouseEvent(pos3))
    b.pump(100)
    check(zyx[les3] == 3 and "lesion 3" in rt.lbl_manual_hint.text(),
          "double-click on lesion 3 while lesion 1 is selected erases nothing")
    rt._label_buttons[3].click()
    vw._lesion_dblclick_cb(vw.viewer, FakeMouseEvent(pos3))
    b.wait(lambda: zyx[les3] == 0)
    b.pump()
    check((zyx == new_id).any(), "island erase of lesion 3 left the new lesion next to it")
    rt.btn_manual_undo.click()
    b.pump()
    check(zyx[les3] == 3, "Undo put id 3 back (not 1)")
    rt.btn_manual_pan.click()
    win.control_panel.tabs.setCurrentIndex(win._TAB_WORKFLOW)
    b.pump(200)

    # ── 14. Organ mask shown while tumor layers exist (old: infinite loop) ──
    organ = np.zeros(shape, dtype=np.uint8, order="F")
    organ[shape[0] // 4: 3 * shape[0] // 4, shape[1] // 4: 3 * shape[1] // 4, :] = 5
    t0 = b.start("organ_mask_after_tumor_hidden")
    lm.set_organ_mask(organ)
    b.pump(50)
    b.record("organ_mask_after_tumor_hidden", time.perf_counter() - t0)
    check(lm._cached_data_zyx.get("organ") is None, "hidden organ mask builds nothing")
    t0 = b.start("organ_mask_show")
    win.control_panel.view_display_tab.chk_organ.setChecked(True)
    b.pump(100)
    b.record("organ_mask_show", time.perf_counter() - t0)
    ok = True
    for v in lm._get_visible_viewers():
        names = [l.name for l in v.viewer.layers]
        ok &= ("Organ Mask" in names and names.index("Organ Mask") < names.index("Tumor Mask")
               and v.viewer.layers["Organ Mask"].visible)
    check(ok, "organ layer shown under the tumor in every view")
    win.control_panel.view_display_tab.chk_organ.setChecked(False)
    b.pump(50)


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
    n_les = len(sm.lesion_stats["lesions"]) if sm.lesion_stats else -1
    check(n_les == len(_LESIONS), f"lesion values after load ({n_les} lesions)")
    check(win.control_panel.workflow_tab.tbl_lesions.rowCount() == n_les,
          "session-tab table filled without export")
    check(not sm.lesion_stats_stale, "values not flagged as edited after load")

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
    check(not sm.lesion_stats_stale, "changing the layout doesn't flag values as edited")

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

    # ── 4b. Double-click lesion info box (toggle on → views jump + boxes paint) ──
    lx, ly, lz = lesion_centres(shape)[2]
    lesion_pos = [Zn - 1 - lz, Yn - 1 - ly, lx]
    vw_ax = lm._fixed_view_map["axial_ct"]
    sc = vw_ax._scale_zyx
    vw_ax._lesion_dblclick_cb(vw_ax.viewer, FakeMouseEvent(
        tuple(c * s for c, s in zip(lesion_pos, sc))))       # real callback path
    b.pump(50)
    check(len(lm._shown_lesion_keys) == 1, "double-click shows the lesion box")
    check([int(round(v)) for v in lm._xhair_pos] == lesion_pos, "views jumped to the lesion")
    shown = [v for v in lm._get_visible_viewers() if v._lesion_overlay.has_infos()]
    check(len(shown) == len(views), f"box pushed to all {len(views)} visible 2D views")
    win._on_lesion_double_clicked(lesion_pos)                # toggle off
    b.pump(50)
    check(not lm._shown_lesion_keys, "second double-click hides it")
    n_t = 20
    t0 = b.start("lesion_info_toggle")
    for _ in range(n_t):
        win._on_lesion_double_clicked(lesion_pos)
        b.pump()
    b.record("lesion_info_toggle", time.perf_counter() - t0, n_t)
    # Leave lesion 3's box on: it must survive the edits below (re-keyed by SUVmax voxel).
    if not lm._shown_lesion_keys:
        win._on_lesion_double_clicked(lesion_pos)
    t0 = b.start("lesion_box_slice_repaint")
    for i in range(n):
        lm._on_viewer_crosshair_click([lesion_pos[0] + (i % 6) - 3, lesion_pos[1], lesion_pos[2]])
        b.pump()
    b.record("lesion_box_slice_repaint", time.perf_counter() - t0, n)

    def edit_paths(suffix):
        # ── 5. Brush painting on the tumor layer (Refine tab) ──
        win.control_panel.tabs.setCurrentIndex(win._TAB_REFINE)
        b.wait(lambda: sm.roi_mask is not None and win.control_panel.tabs.isEnabled())
        b.pump(200)
        check(not sm.lesion_stats_stale, "entering the Segmentation tab doesn't flag values")
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
        check(overlay_mode_ok(lm, "tumor") and overlay_mode_ok(lm, "roi"),
              f"default smooth display: Labels at opacity 0 under the overlays{suffix}")
        check(overlay_fresh(lm, "tumor"), f"tumor overlay follows the brush in every view{suffix}")

        # ── 5b. ROI brush stroke at the crosshair, then put the voxels back ──
        roi_layer = vw.viewer.layers[vw.LAYER_NAMES["roi"]]
        zr, yr, xr = (int(round(c)) for c in lm._xhair_pos)
        roi_zyx = lm.get_active_mask_data_zyx("roi")
        box = (zr, slice(max(yr - 8, 0), yr + 8), slice(max(xr - 8, 0), xr + 8))
        saved_roi = roi_zyx[box].copy()
        lm.set_drawing_tool("paint", 10, "roi")       # ROI layer active: gets the keys
        roi_layer.paint((zr, yr, xr), 1)
        b.pump(150)                                   # mask_sync refreshes the other views
        check(not np.array_equal(roi_zyx[box], saved_roi) and roi_zyx[zr, yr, xr] == 1
              and overlay_fresh(lm, "roi") and labels_fresh(lm, "roi"),
              f"ROI overlay follows a brush stroke in every view{suffix}")
        press_keys(lm, "coronal_ct", "Control-Z")     # stroke was made in axial_ct
        b.pump(50)
        check(np.array_equal(roi_zyx[box], saved_roi) and labels_fresh(lm, "roi")
              and overlay_fresh(lm, "roi"), f"ROI Ctrl+Z from another view, all views updated{suffix}")
        press_keys(lm, "axial_pet", "Control-Shift-Z")
        b.pump(50)
        check(roi_zyx[zr, yr, xr] == 1 and labels_fresh(lm, "roi") and overlay_fresh(lm, "roi"),
              f"ROI Ctrl+Shift+Z redo, all views updated{suffix}")
        lm.set_drawing_tool("pan_zoom", 10, "roi")
        roi_zyx[box] = saved_roi
        lm.refresh_mask("roi")
        b.pump(50)
        check(overlay_fresh(lm, "roi"), f"ROI overlay follows refresh_mask{suffix}")

        # ── 6. Threshold preview slider ticks ──
        cx, cy, cz = lesion_centres(shape)[1]
        roi_zyx = lm.get_active_mask_data_zyx("roi")
        zc, yc, xc = Zn - 1 - cz, Yn - 1 - cy, cx
        r = _LESION_R + 4
        roi_zyx[zc - r:zc + r, yc - r:yc + r, xc - r:xc + r] = 1
        for v in lm._get_visible_viewers():
            if vw.LAYER_NAMES["roi"] in v.viewer.layers:
                v.viewer.layers[vw.LAYER_NAMES["roi"]].refresh()
        b.pump(50)
        check(overlay_fresh(lm, "roi"), f"ROI overlay follows an in-place edit + refresh{suffix}")
        win._on_refine_adaptive(0.7, "outside_isocontour", 3)
        b.wait(lambda: getattr(win, "_current_preview_dialog", None) is not None)
        b.pump(200)
        t0 = b.start(f"threshold_slider_tick{suffix}")
        for i in range(30):
            win._update_component_preview(2.0 + 0.1 * i)
            b.pump()
        b.record(f"threshold_slider_tick{suffix}", time.perf_counter() - t0, 30)
        b.pump(50)
        check(overlay_fresh(lm, "roi"), f"ROI overlay follows the threshold preview{suffix}")
        win._current_preview_dialog.reject()
        b.pump(200)
        check(overlay_fresh(lm, "roi"), f"ROI overlay follows the preview reject{suffix}")

        # ── 7. Confirm & save (merge ROI into tumor, write to disk) ──
        check(sm.lesion_stats_stale, "brush edit marks lesion values stale")
        t0 = b.start(f"confirm_and_save{suffix}")
        win._on_confirm_and_save()
        b.wait(lambda: not win._merge_save_worker.isRunning())
        b.pump(100)
        b.record(f"confirm_and_save{suffix}", time.perf_counter() - t0)
        check(not sm.lesion_stats_stale, "red button recomputed the lesion values "
              f"({len(sm.lesion_stats['lesions'])} lesions)")
        check(len(lm._shown_lesion_keys) == 1, "lesion box survives the recompute")
        check(overlay_fresh(lm, "tumor") and overlay_fresh(lm, "roi"),
              f"both overlays follow confirm & save{suffix}")

        # ── 8. Island erase: Erase tool + double-click on a lesion, then Undo ──
        rt = win.control_panel.refine_tab
        rt.btn_manual_erase.click()
        b.pump(100)
        ex, ey, ez = lesion_centres(shape)[0]
        voxel = (Zn - 1 - ez, Yn - 1 - ey, ex)
        scale = vw._scale_zyx
        pos = tuple(c * s_ for c, s_ in zip(voxel, scale))
        zyx = lm.get_active_mask_data_zyx("tumor")
        shown_before = list(lm._shown_lesion_keys)
        t0 = b.start(f"island_erase{suffix}")
        vw._lesion_dblclick_cb(vw.viewer, FakeMouseEvent(pos))
        b.wait(lambda: zyx[voxel] == 0)
        b.pump()
        b.record(f"island_erase{suffix}", time.perf_counter() - t0)
        check(lm._shown_lesion_keys == shown_before, "erase-mode double-click doesn't toggle boxes")
        check(sm.lesion_stats_stale, "island erase marks lesion values stale")

        t0 = b.start(f"manual_undo{suffix}")
        rt.btn_manual_undo.click()
        b.pump()
        b.record(f"manual_undo{suffix}", time.perf_counter() - t0)
        check(zyx[voxel] == 1, "Undo restored the island")
        check(overlay_fresh(lm, "tumor") and labels_fresh(lm, "tumor"),
              f"tumor overlay follows island erase + Undo{suffix}")

        # Ctrl+Z / Ctrl+Shift+Z: napari's own undo refreshed only one view.
        # Empty corner of the crosshair's axial slice (Erase mode keeps
        # preserve_labels on: inside a lesion this paint would change nothing).
        zt = int(round(lm._xhair_pos[0]))
        tbox = (slice(max(zt - 8, 0), zt + 8), slice(4, 28), slice(4, 28))
        saved_t = zyx[tbox].copy()
        layer.paint((zt, 16, 16), int(lm._tumor_label))
        b.pump(150)
        painted = zyx[tbox].copy()
        check(not np.array_equal(painted, saved_t) and labels_fresh(lm, "tumor"),
              f"tumor test stroke painted and synced{suffix}")
        press_keys(lm, "sagittal_overlay", "Control-Z")
        b.pump(50)
        check(np.array_equal(zyx[tbox], saved_t) and labels_fresh(lm, "tumor")
              and overlay_fresh(lm, "tumor"), f"tumor Ctrl+Z from another view, all views updated{suffix}")
        press_keys(lm, "axial_ct", "Control-Shift-Z")
        b.pump(50)
        check(np.array_equal(zyx[tbox], painted) and labels_fresh(lm, "tumor")
              and overlay_fresh(lm, "tumor"), f"tumor Ctrl+Shift+Z redo, all views updated{suffix}")
        rt.btn_manual_undo.click()                    # Undo button: same history
        b.pump(50)
        check(np.array_equal(zyx[tbox], saved_t) and labels_fresh(lm, "tumor"),
              f"Undo button after a redo{suffix}")

        # Erase again and save with the red button → values recomputed.
        vw._lesion_dblclick_cb(vw.viewer, FakeMouseEvent(pos))
        b.wait(lambda: zyx[voxel] == 0)
        b.pump()
        erased_atoms = lm._last_tumor_stroke[2]      # to put lesion 0 back after the save
        n_before = len(sm.lesion_stats["lesions"])
        t0 = b.start(f"island_erase_save_recompute{suffix}")
        win._on_confirm_and_save()
        b.wait(lambda: not win._merge_save_worker.isRunning())
        b.pump(50)
        b.record(f"island_erase_save_recompute{suffix}", time.perf_counter() - t0)
        check(len(sm.lesion_stats["lesions"]) == n_before - 1 and not sm.lesion_stats_stale,
              "island erase + save recomputed the values (one lesion fewer)")
        zyx = lm.get_active_mask_data_zyx("tumor")
        for idx, old_vals, _ in erased_atoms:         # put lesion 0 back for later passes
            zyx[idx] = old_vals
        lm.refresh_mask("tumor")
        sm.tumor_dirty = True
        win._on_confirm_and_save()
        b.wait(lambda: not win._merge_save_worker.isRunning())
        rt.btn_manual_pan.click()
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

    # ── 9. Export: folder pick + write the stored values (no save, no compute) ──
    import src.gui.handlers.report_handler as rh
    export_root = tmp / "export"
    rh.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(export_root))
    rh.QMessageBox.information = staticmethod(lambda *a, **k: None)
    stats_before = sm.lesion_stats
    t0 = b.start("export_report")
    win._on_report_clicked()
    b.wait(lambda: not win.report_worker.isRunning())
    b.pump(50)
    b.record("export_report", time.perf_counter() - t0)
    check(sm.lesion_stats is stats_before, "export did not recompute")
    csv_path = export_root / str(sm.current_session_id) / "report_metabolic.csv"
    head = csv_path.read_text().splitlines()[0] if csv_path.exists() else ""
    check("SUVpeak" in head and "TLG" in head, "CSV has SUVpeak + TLG columns")

    lesion_id_paths(b, win, shape)

    if CHECK_FAILURES:
        print(f"[bench] {len(CHECK_FAILURES)} check(s) FAILED: {CHECK_FAILURES}")

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
