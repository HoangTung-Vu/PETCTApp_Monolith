"""Ruler overlay drawn on top of a Napari canvas.

A transparent QWidget child of the canvas native widget (same approach as
``CrosshairOverlay`` / ``ColorBarOverlay``).  It draws:

* an mm tick-scale along the bottom + right edges of the canvas (the left edge
  is reserved for the colorbar), scaling with the camera zoom, and
* the current distance measurement: start / end markers, the connecting line
  (dashed while the end point is still being chosen), and the distance label.

All measurement points are in Napari ZYX data space and are projected to canvas
pixels via ``ViewerWidget._compute_canvas_pos`` at paint time, so they stay glued
to the anatomy through any pan / zoom / slice change — and project correctly even
when the two endpoints sit on different slices.
"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush


class RulerOverlay(QWidget):
    """Edge mm-scale + two-point distance measurement, drawn over a canvas."""

    _LINE_COLOR  = QColor(0, 255, 128)       # measurement line / markers
    _TEXT_COLOR  = QColor(255, 255, 0)       # distance label
    _TICK_COLOR  = QColor(180, 220, 255, 210)  # edge ruler ticks

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref
        self._start = None       # [z, y, x] data space, or None
        self._end = None
        self._preview = None
        self._dist_text = ""
        self._enabled = False

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

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.update()

    def set_state(self, start, end, preview, dist_text: str):
        self._start = start
        self._end = end
        self._preview = preview
        self._dist_text = dist_text or ""
        self.update()

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._enabled or getattr(self._vw, "is_3d", False):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            self._draw_edge_ruler(painter)
            self._draw_measurement(painter)
        finally:
            painter.end()

    # ── edge mm-scale ─────────────────────────────────────────────────────────

    @staticmethod
    def _nice_step(target_px: float, zoom: float):
        """Pick a 1/2/5·10^k mm step whose pixel size is closest to target_px."""
        if zoom <= 0:
            return None
        raw_mm = target_px / zoom
        if raw_mm <= 0 or not math.isfinite(raw_mm):
            return None
        exp = math.floor(math.log10(raw_mm))
        base = 10 ** exp
        frac = raw_mm / base
        if frac < 1.5:
            nice = 1
        elif frac < 3.5:
            nice = 2
        elif frac < 7.5:
            nice = 5
        else:
            nice = 10
        return nice * base

    def _draw_edge_ruler(self, painter: QPainter):
        try:
            zoom = float(self._vw.viewer.camera.zoom)
            center = self._vw.viewer.camera.center
            c_vert = float(center[-2])
            c_horiz = float(center[-1])
        except Exception:
            return
        cw, ch = self.width(), self.height()
        if cw <= 0 or ch <= 0 or zoom <= 0:
            return

        step = self._nice_step(70.0, zoom)
        if not step:
            return
        minor = step / 5.0

        font = QFont("Arial", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # ── Bottom edge: horizontal world axis (center[-1]) ──
        baseline_y = ch - 1
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawLine(0, baseline_y, cw - 1, baseline_y)

        w_left = c_horiz - (cw / 2.0) / zoom
        w_right = c_horiz + (cw / 2.0) / zoom
        i0 = math.ceil(w_left / minor)
        i1 = math.floor(w_right / minor)
        for i in range(i0, i1 + 1):
            v = i * minor
            px = (v - c_horiz) * zoom + cw / 2.0
            is_major = (i % 5 == 0)
            tick_len = 10 if is_major else 5
            painter.setPen(QPen(self._TICK_COLOR, 1))
            painter.drawLine(int(px), baseline_y, int(px), baseline_y - tick_len)
            if is_major:
                label = f"{v:.0f}"
                tw = fm.horizontalAdvance(label)
                self._draw_text(painter, px - tw / 2.0, baseline_y - tick_len - 3, label)

        # ── Right edge: vertical world axis (center[-2]) ──
        baseline_x = cw - 1
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawLine(baseline_x, 0, baseline_x, ch - 1)

        w_top = c_vert - (ch / 2.0) / zoom
        w_bot = c_vert + (ch / 2.0) / zoom
        j0 = math.ceil(w_top / minor)
        j1 = math.floor(w_bot / minor)
        for j in range(j0, j1 + 1):
            v = j * minor
            py = (v - c_vert) * zoom + ch / 2.0
            is_major = (j % 5 == 0)
            tick_len = 10 if is_major else 5
            painter.setPen(QPen(self._TICK_COLOR, 1))
            painter.drawLine(baseline_x, int(py), baseline_x - tick_len, int(py))
            if is_major:
                label = f"{v:.0f}"
                tw = fm.horizontalAdvance(label)
                self._draw_text(painter, baseline_x - tick_len - 4 - tw, py + 4, label)

    # ── measurement ───────────────────────────────────────────────────────────

    def _draw_measurement(self, painter: QPainter):
        if self._start is None:
            return
        sx, sy = self._vw._compute_canvas_pos(self._start)

        if sx is not None:
            self._draw_marker(painter, sx, sy)

        other = self._end if self._end is not None else self._preview
        if other is not None:
            ox, oy = self._vw._compute_canvas_pos(other)
            if ox is not None:
                self._draw_marker(painter, ox, oy)
                if sx is not None:
                    pen = QPen(self._LINE_COLOR, 2)
                    if self._end is None:        # still choosing end → preview dashed
                        pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(int(sx), int(sy), int(ox), int(oy))
                    if self._dist_text:
                        mx = (sx + ox) / 2.0
                        my = (sy + oy) / 2.0
                        self._draw_text(painter, mx + 8, my - 8, self._dist_text,
                                        color=self._TEXT_COLOR, bold=True)

    def _draw_marker(self, painter: QPainter, x: float, y: float):
        xi, yi = int(round(x)), int(round(y))
        painter.setPen(QPen(self._LINE_COLOR, 1))
        painter.setBrush(QBrush(self._LINE_COLOR))
        painter.drawEllipse(xi - 3, yi - 3, 6, 6)
        # small cross hairs for precise centre
        painter.drawLine(xi - 7, yi, xi - 4, yi)
        painter.drawLine(xi + 4, yi, xi + 7, yi)
        painter.drawLine(xi, yi - 7, xi, yi - 4)
        painter.drawLine(xi, yi + 4, xi, yi + 7)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _draw_text(self, painter: QPainter, x: float, y: float, text: str,
                   color: QColor = None, bold: bool = False):
        """Draw text with a 1px black shadow for legibility over any background."""
        color = color or QColor(255, 255, 255)
        f = painter.font()
        f.setBold(bold)
        painter.setFont(f)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(int(x) + 1, int(y) + 1, text)
        painter.setPen(color)
        painter.drawText(int(x), int(y), text)
        if bold:
            f.setBold(False)
            painter.setFont(f)
