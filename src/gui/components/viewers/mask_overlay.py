"""Smoothed fill / outline of a mask (tumor or ROI), drawn over a 2D Napari canvas.

Display only. The Labels layer underneath stays the data and the editor
(brush, erase, fill, shapes, undo); in these styles it is drawn at opacity 0
and this widget redraws the current slice from ``layer.data`` whenever the
layer reports a change. What napari 0.6.6 emits (measured):

* ``events.set_data``      — slice change and every ``refresh()`` (view sync,
  island erase, undo / redo, load, threshold preview, confirm & save)
* ``events.labels_update`` — each mouse move of a brush / fill stroke
* ``events.paint``         — end of a stroke
* ``events.data``          — ``layer.data = ...``; the only event when the
  layer is hidden, since ``refresh()`` does nothing on a hidden layer
* ``events.visible``       — show / hide (showing also refreshes)

So the Labels layer must stay *visible* at opacity 0: hidden, it stops
emitting ``set_data`` and napari forces it into pan_zoom (no painting).

Only the current slice's ``QPainterPath``s are kept (one per label id, in
voxel-index coordinates); pan and zoom just change the painter transform.
Each id is drawn in its layer colormap's colour, so touching lesions with
different ids keep separate outlines and the ROI stays yellow.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QEvent, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor, QPolygonF

# Startup look of each mask, (style, smooth, contour width px), shared by the
# viewers and the View tab controls so the two never disagree.
DEFAULT_MASK_DISPLAY = {
    "tumor": ("filled", True, 2),
    "roi": ("contour", True, 2),
}
DEFAULT_MASK_OPACITY = {"tumor": 0.7, "roi": 1.0}

# Layer events that change what the overlay must draw (see module docstring).
_LAYER_EVENTS = ("set_data", "labels_update", "paint", "visible", "data")


def _smooth_closed(points: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing of a closed polygon's vertices along the outline.

    Flattens the steps marching squares leaves on shallow slopes. σ = 1 vertex
    (≈ 0.6 voxel) barely shrinks small lesions; very short outlines are kept.
    """
    if len(points) < 8:
        return points
    r = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    wrapped = np.concatenate([points[-r:], points, points[:r]])
    return np.stack([np.convolve(wrapped[:, i], kernel, mode="valid") for i in range(2)], axis=1)


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner-cutting smoothing of a closed polygon (stays inside its hull)."""
    for _ in range(iterations):
        nxt = np.roll(points, -1, axis=0)
        out = np.empty((2 * len(points), 2))
        out[0::2] = 0.75 * points + 0.25 * nxt
        out[1::2] = 0.25 * points + 0.75 * nxt
        points = out
    return points


def mask_outlines(plane: np.ndarray, smooth: bool, value: int = None) -> list:
    """Closed outlines of ``plane > 0`` (or ``plane == value``) as (N, 2)
    arrays of (row, col) indices.

    Voxel centres sit on integer indices. ``smooth`` traces the 0.5 iso-line
    (marching squares), flattens its steps and rounds it with Chaikin's corner
    cutting; otherwise the outline follows the voxel edges.
    """
    from skimage.measure import find_contours

    mask = plane > 0 if value is None else plane == value
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return []
    cols = np.flatnonzero(mask.any(axis=0))
    # One empty voxel around the crop keeps every outline closed.
    crop = np.pad(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], 1).view(np.uint8)
    origin = np.array([rows[0] - 1, cols[0] - 1], dtype=float)
    if not smooth:
        # On a 2× nearest-upsampled grid the 0.5 iso-line runs along voxel edges.
        crop = crop.repeat(2, axis=0).repeat(2, axis=1)
    outlines = []
    for c in find_contours(crop, 0.5):
        c = c[:-1]                                   # closed: last point == first
        if len(c) < 3:
            continue
        if smooth:
            c = _chaikin(_smooth_closed(c))
        else:
            c = (c - 0.5) / 2.0
        outlines.append(c + origin)
    return outlines


def label_outlines(plane: np.ndarray, smooth: bool) -> dict:
    """``{lesion id: mask_outlines(plane == id)}`` for every id in ``plane``."""
    rows = np.flatnonzero(plane.any(axis=1))
    if rows.size == 0:
        return {}
    cols = np.flatnonzero(plane.any(axis=0))
    crop = plane[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ids = [int(v) for v in np.unique(crop) if v != 0]
    if len(ids) == 1:
        return {ids[0]: mask_outlines(plane, smooth)}
    return {v: mask_outlines(plane, smooth, v) for v in ids}


class MaskOverlay(QWidget):
    """A mask layer drawn as a smoothed fill or an outline of constant pixel width."""

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref
        self._layer = None
        self._enabled = False
        self._style = "filled"            # "filled" | "contour"
        self._smooth = False
        self._width = 2                   # contour pen width, screen pixels
        self._opacity = 0.7
        self._path = None                 # {label id: QPainterPath} of the slice in _path_key
        self._path_key = None
        # {label id: (r, g, b)} of ``_rgb_cmap``. napari's DirectLabelColormap
        # (the ROI's yellow) takes ~1.3 ms per ``map()`` call — too slow for
        # every paint of every view. Assigning a colormap replaces the object.
        self._rgb_cmap = None
        self._rgb = {}

        # Completely transparent: mouse events fall through to canvas below
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        parent.installEventFilter(self)
        self.resize(parent.size())
        self.raise_()

    # ── event filter: keep overlay sized to parent canvas ──────────────────

    def eventFilter(self, source, event: QEvent):
        if source is self.parent() and event.type() == QEvent.Type.Resize:
            self.resize(source.size())
            self.update()
        return False

    # ── public API ─────────────────────────────────────────────────────────

    def configure(self, style: str, smooth: bool, width: int, opacity: float, enabled: bool):
        if smooth != self._smooth:
            self._path = None
        self._style, self._smooth, self._width = style, smooth, int(width)
        self._opacity = float(opacity)
        self._enabled = enabled
        self.update()

    def attach_layer(self, layer):
        """Follow ``layer`` (a tumor or ROI Labels layer); no-op if already attached."""
        if layer is self._layer:
            return
        self.detach()
        self._layer = layer
        for name in _LAYER_EVENTS:
            getattr(layer.events, name).connect(self.invalidate)
        self.invalidate()

    def detach(self):
        if self._layer is not None:
            for name in _LAYER_EVENTS:
                try:
                    getattr(self._layer.events, name).disconnect(self.invalidate)
                except Exception:
                    pass
        self._layer = None
        self.invalidate()

    def invalidate(self, event=None):
        """The mask, the slice or the visibility changed: rebuild on next paint."""
        self._path = None
        if self._enabled:
            self.update()

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        layer = self._layer
        if not self._enabled or layer is None or not layer.visible or self._vw.is_3d:
            return
        transform = self._vw._canvas_transform()
        if transform is None:
            return
        paths = self._current_path(layer)
        if not paths:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setTransform(transform)
        rgb = self._label_rgb(layer, paths)
        for lesion_id, path in paths.items():
            color = QColor.fromRgbF(*rgb[lesion_id], self._opacity)
            if self._style == "contour":
                pen = QPen(color, self._width)
                pen.setCosmetic(True)                 # width in screen pixels at any zoom
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
            painter.drawPath(path)
        painter.end()

    def _label_rgb(self, layer, ids) -> dict:
        """``{id: (r, g, b)}`` for ``ids`` from the layer's colormap, cached."""
        cmap = layer.colormap
        if cmap is not self._rgb_cmap:
            self._rgb_cmap, self._rgb = cmap, {}
        missing = [i for i in ids if i not in self._rgb]
        if missing:
            rgba = np.asarray(cmap.map(np.array(missing)), dtype=float).reshape(-1, 4)
            for i, c in zip(missing, rgba):
                self._rgb[i] = (float(c[0]), float(c[1]), float(c[2]))
        return self._rgb

    def _current_path(self, layer):
        dims = self._vw.viewer.dims
        displayed = tuple(dims.displayed)
        data = layer.data
        if len(displayed) != 2 or data.ndim != 3:
            return None
        axis = [d for d in range(3) if d not in displayed][0]
        index = int(round((dims.point[axis] - layer.translate[axis]) / layer.scale[axis]))
        key = (axis, index, displayed, data.shape)
        if self._path is not None and key == self._path_key:
            return self._path

        paths = {}
        if 0 <= index < data.shape[axis]:
            plane = data[tuple(index if d == axis else slice(None) for d in range(3))]
            if displayed[0] > displayed[1]:
                plane = plane.T
            for lesion_id, outlines in label_outlines(plane, self._smooth).items():
                path = QPainterPath()
                path.setFillRule(Qt.FillRule.OddEvenFill)          # holes stay empty
                for outline in outlines:
                    # (row, col) = (vertical, horizontal) → (x, y) = (col, row)
                    path.addPolygon(QPolygonF([QPointF(x, y) for y, x in outline]))
                    path.closeSubpath()
                if not path.isEmpty():
                    paths[lesion_id] = path
        self._path, self._path_key = paths, key
        return paths
