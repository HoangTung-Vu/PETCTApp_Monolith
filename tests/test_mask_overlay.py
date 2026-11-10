"""Outline extraction behind the smoothed / contour tumor and ROI mask display,
and the napari events that keep the overlay in step with the Labels layer.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_mask_overlay.py -q
"""

import os
from types import SimpleNamespace

import numpy as np

from src.gui.components.viewers.mask_overlay import mask_outlines


def _area(poly):
    y, x = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_empty_plane_has_no_outline():
    assert mask_outlines(np.zeros((16, 16), np.uint8), smooth=True) == []


def test_voxel_outline_follows_voxel_edges():
    plane = np.zeros((16, 16), np.uint8)
    plane[5:8, 6:10] = 1
    (outline,) = mask_outlines(plane, smooth=False)
    np.testing.assert_allclose(outline.min(axis=0), (4.5, 5.5))
    np.testing.assert_allclose(outline.max(axis=0), (7.5, 9.5))
    assert abs(_area(outline) - 12) < 0.2              # only the corners are chamfered


def test_hole_and_separate_lesions_give_separate_outlines():
    yy, xx = np.ogrid[:40, :40]
    plane = np.zeros((40, 40), np.uint8)
    plane[(yy - 15) ** 2 + (xx - 15) ** 2 <= 10 ** 2] = 1
    plane[(yy - 15) ** 2 + (xx - 15) ** 2 <= 3 ** 2] = 0
    plane[35, 35] = 1                                     # single voxel lesion
    for smooth in (False, True):
        outlines = mask_outlines(plane, smooth)
        assert len(outlines) == 3
        small = min(outlines, key=_area)
        np.testing.assert_allclose(small.mean(axis=0), (35, 35), atol=1e-6)


def test_smooth_outline_keeps_the_lesion_size():
    yy, xx = np.ogrid[:64, :64]
    for r in (3, 10, 25):
        plane = ((yy - 32) ** 2 + (xx - 32) ** 2 <= r * r).astype(np.uint8)
        (outline,) = mask_outlines(plane, smooth=True)
        assert abs(_area(outline) / plane.sum() - 1) < 0.12, r


def test_each_lesion_id_gets_its_own_outline():
    from src.gui.components.viewers.mask_overlay import label_outlines
    plane = np.zeros((20, 20), np.int16)
    plane[2:8, 2:8] = 3
    plane[2:8, 8:14] = 7          # touches lesion 3: still a separate outline
    outlines = label_outlines(plane, smooth=False)
    assert sorted(outlines) == [3, 7]
    assert len(outlines[3]) == 1 and len(outlines[7]) == 1
    assert abs(_area(outlines[7][0]) - 36) < 0.3
    assert label_outlines(np.zeros((5, 5), np.int16), smooth=True) == {}


# ── Overlay ↔ Labels events (napari 0.6.6) ───────────────────────────────────

def _overlay_on(layer):
    """MaskOverlay following ``layer``, showing axial slice 2 (no real viewer)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QWidget
    from src.gui.components.viewers.mask_overlay import MaskOverlay

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    dims = SimpleNamespace(displayed=(1, 2), point=(2.0, 0.0, 0.0))
    vw = SimpleNamespace(is_3d=False, viewer=SimpleNamespace(dims=dims))
    overlay = MaskOverlay(vw, parent)
    overlay.configure("contour", True, 2, 1.0, True)
    overlay.attach_layer(layer)
    return overlay, (app, parent)


def _is_fresh(overlay, layer):
    """The paths the overlay would draw now equal a rebuild from ``layer.data``."""
    drawn = overlay._current_path(layer)          # the cache, unless invalidated
    overlay._path = None
    rebuilt = overlay._current_path(layer)
    return drawn.keys() == rebuilt.keys() and all(drawn[k] == rebuilt[k] for k in rebuilt)


def test_every_edit_route_keeps_the_overlay_fresh():
    import napari

    layer = napari.layers.Labels(np.zeros((5, 16, 16), np.uint8))
    layer.opacity = 0.0                     # overlay mode: transparent, still visible
    overlay, _keep = _overlay_on(layer)
    assert overlay._current_path(layer) == {}

    # In-place write + refresh(): set_data, also at opacity 0.
    layer.data[2, 3:7, 3:7] = 1
    layer.refresh()
    assert _is_fresh(overlay, layer) and 1 in overlay._path

    # Brush / island erase route: paint + labels_update.
    idx = tuple(np.array(v) for v in np.nonzero(np.ones((1, 3, 3), bool)))
    layer.data_setitem((idx[0] + 2, idx[1] + 10, idx[2] + 10), 1)
    assert _is_fresh(overlay, layer) and overlay._path[1].elementCount() > 0

    # Undo: refresh() inside napari.
    before = overlay._current_path(layer)[1]
    layer.undo()
    assert _is_fresh(overlay, layer) and overlay._path[1] != before

    # Shape commit / load: ``layer.data = ...`` (same shape, new array).
    new = np.zeros_like(layer.data)
    new[2, 8:12, 1:5] = 1
    layer.data = new
    assert _is_fresh(overlay, layer)

    # Hidden: refresh() emits nothing; showing again refreshes.
    layer.visible = False
    layer.data[2, 13:15, 13:15] = 1
    layer.refresh()
    layer.visible = True
    assert _is_fresh(overlay, layer)

    # Replaced while hidden: only ``events.data`` fires. (Hiding already
    # clears the cache and paintEvent skips hidden layers; seed it anyway so
    # this checks the ``data`` subscription on its own.)
    layer.visible = False
    overlay._current_path(layer)
    layer.data = np.zeros_like(layer.data)
    assert _is_fresh(overlay, layer) and overlay._path == {}
    layer.visible = True

    # Contract: an in-place write without refresh() is not seen.
    overlay._current_path(layer)
    layer.data[2, 0:2, 0:2] = 1
    assert not _is_fresh(overlay, layer)


def test_hidden_labels_layer_cannot_paint():
    """Why overlay mode uses opacity 0 and never ``visible = False``."""
    import napari

    layer = napari.layers.Labels(np.zeros((3, 8, 8), np.uint8))
    layer.mode = "paint"
    layer.visible = False
    assert layer.mode == "pan_zoom"
    layer.visible = True
    assert layer.mode == "paint"


def test_roi_overlay_is_yellow():
    import napari
    from src.gui.components.viewers.viewer_widget import ViewerWidget

    layer = napari.layers.Labels(np.zeros((3, 8, 8), np.uint8))
    ViewerWidget._apply_roi_colormap(layer)
    rgba = np.asarray(layer.colormap.map(np.array([1])), dtype=float).ravel()
    np.testing.assert_allclose(rgba[:3], (1, 1, 0))


def test_colour_cache_follows_a_new_colormap():
    import napari
    from src.gui.components.viewers.viewer_widget import ViewerWidget
    from src.gui.components.viewers.label_colors import TUMOR_LABEL_COLORMAP, label_rgba

    layer = napari.layers.Labels(np.zeros((3, 8, 8), np.uint8))
    overlay, _keep = _overlay_on(layer)
    ViewerWidget._apply_roi_colormap(layer)
    assert overlay._label_rgb(layer, [1])[1] == (1.0, 1.0, 0.0)
    cmap = layer.colormap
    assert layer.colormap is cmap                    # stable between assignments
    layer.colormap = TUMOR_LABEL_COLORMAP
    np.testing.assert_allclose(overlay._label_rgb(layer, [1])[1], label_rgba(1)[:3])
