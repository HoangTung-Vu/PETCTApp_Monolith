"""Smoothed fill / outline of the tumor mask, drawn over a 2D Napari canvas.

Display only. The Labels layer underneath stays the data and the editor
(brush, erase, fill, shapes, undo); in these styles it is drawn at opacity 0
and this widget redraws the current slice from ``layer.data`` whenever the
layer reports a change:

* ``events.set_data``      — slice change and every ``refresh()`` (view sync,
  eraser, load, confirm & save)
* ``events.labels_update`` — each mouse move of a brush / fill stroke
* ``events.paint``         — end of a stroke, undo / redo

Only the current slice's ``QPainterPath`` is kept (in voxel-index
coordinates); pan and zoom just change the painter transform.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QEvent, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor, QPolygonF


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


def mask_outlines(plane: np.ndarray, smooth: bool) -> list:
    """Closed outlines of ``plane > 0`` as (N, 2) arrays of (row, col) indices.

    Voxel centres sit on integer indices. ``smooth`` traces the 0.5 iso-line
    (marching squares), flattens its steps and rounds it with Chaikin's corner
    cutting; otherwise the outline follows the voxel edges.
    """
    from skimage.measure import find_contours

    mask = plane > 0
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


class MaskOverlay(QWidget):
    """Tumor mask drawn as a smoothed fill or an outline of constant pixel width."""

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref
        self._layer = None
        self._enabled = False
        self._style = "filled"            # "filled" | "contour"
        self._smooth = False
        self._width = 2                   # contour pen width, screen pixels
        self._opacity = 0.7
        self._path = None                 # QPainterPath of the slice in _path_key
        self._path_key = None

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
        """Follow ``layer`` (the tumor Labels layer); no-op if already attached."""
        if layer is self._layer:
            return
        self.detach()
        self._layer = layer
        for name in ("set_data", "labels_update", "paint", "visible"):
            getattr(layer.events, name).connect(self.invalidate)
        self.invalidate()

    def detach(self):
        if self._layer is not None:
            for name in ("set_data", "labels_update", "paint", "visible"):
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
        path = self._current_path(layer)
        if path is None or path.isEmpty():
            return

        rgba = np.asarray(layer.colormap.map(1), dtype=float).ravel()
        color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]), self._opacity)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setTransform(transform)
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

        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)          # holes stay empty
        if 0 <= index < data.shape[axis]:
            plane = data[tuple(index if d == axis else slice(None) for d in range(3))]
            if displayed[0] > displayed[1]:
                plane = plane.T
            for outline in mask_outlines(plane, self._smooth):
                # (row, col) = (vertical, horizontal) → (x, y) = (col, row)
                path.addPolygon(QPolygonF([QPointF(x, y) for y, x in outline]))
                path.closeSubpath()
        self._path, self._path_key = path, key
        return path
