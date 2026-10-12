"""Lesion quantification boxes drawn on top of a Napari canvas.

A transparent QWidget child of the canvas native widget (same approach as
``RulerOverlay``). For every lesion the user toggled on (double-click), it
outlines the lesion's extent on the current slice and draws a box with its ID,
MTV, SUVmax, SUVmean, SUVpeak and TLG next to it.

A lesion is drawn only on slices that contain at least one of its voxels.
Positions are projected from Napari ZYX data space with
``ViewerWidget._compute_canvas_pos`` at paint time, so the boxes follow pan,
zoom and slice changes.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QEvent, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics


class LesionInfoOverlay(QWidget):
    """Per-lesion quantification boxes over a 2D canvas."""

    _ACCENT     = QColor(255, 212, 0)          # outline, box border, title
    _TEXT       = QColor(235, 235, 235)
    _STALE      = QColor(255, 150, 60)
    _BOX_BG     = QColor(0, 0, 0, 185)

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref
        # Each info: {"id", "lo" (z, y, x), "crop" (bool ZYX), "occupancy" [per-axis
        # 1D bool], "values" {"MTV", "SUVmax", "SUVmean", "SUVpeak", "TLG"}}
        self._infos = []
        self._stale = False

        self._font = QFont("Monospace", 10)
        self._font.setStyleHint(QFont.StyleHint.TypeWriter)
        self._title_font = QFont(self._font)
        self._title_font.setBold(True)

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

    def set_infos(self, infos, stale: bool):
        self._infos = list(infos or [])
        self._stale = stale
        self.update()

    def has_infos(self) -> bool:
        return bool(self._infos)

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._infos or getattr(self._vw, "is_3d", False):
            return
        dims = self._vw.viewer.dims
        displayed = list(dims.displayed)
        if len(displayed) < 2:
            return
        slice_axis = [d for d in range(3) if d not in displayed][0]
        index = int(dims.current_step[slice_axis])

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            for info in self._infos:
                rect = self._slice_rect(info, slice_axis, index, displayed)
                if rect is not None:
                    self._draw_lesion(painter, info, rect)
        finally:
            painter.end()

    def _slice_rect(self, info, axis: int, index: int, displayed):
        """Canvas rectangle of the lesion's extent on this slice, or None."""
        lo, crop = info["lo"], info["crop"]
        k = index - lo[axis]
        occ = info["occupancy"][axis]
        if not (0 <= k < occ.size and occ[k]):
            return None

        # In-plane extent of the lesion on this slice (axes in displayed order).
        plane = np.take(crop, k, axis=axis)
        if displayed[0] > displayed[1]:
            plane = plane.T
        rows = np.flatnonzero(plane.any(axis=1))
        cols = np.flatnonzero(plane.any(axis=0))
        v_axis, h_axis = displayed

        # Voxel edges sit half a voxel around the voxel centres.
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        p0[v_axis] = lo[v_axis] + rows[0] - 0.5
        p1[v_axis] = lo[v_axis] + rows[-1] + 0.5
        p0[h_axis] = lo[h_axis] + cols[0] - 0.5
        p1[h_axis] = lo[h_axis] + cols[-1] + 0.5
        x0, y0 = self._vw._compute_canvas_pos(p0)
        x1, y1 = self._vw._compute_canvas_pos(p1)
        if x0 is None or x1 is None:
            return None
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if not rect.intersects(QRectF(0, 0, self.width(), self.height())):
            return None         # lesion panned out of view
        return rect

    def _lines(self, info):
        v = info["values"]
        return [
            f"MTV     {v['MTV']:8.2f} mL",
            f"SUVmax  {v['SUVmax']:8.2f}",
            f"SUVmean {v['SUVmean']:8.2f}",
            f"SUVpeak {v['SUVpeak']:8.2f}",
            f"TLG     {v['TLG']:8.2f}",
        ]

    def _draw_lesion(self, painter: QPainter, info, rect: QRectF):
        # Outline of the lesion on this slice
        pen = QPen(self._ACCENT, 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(-2, -2, 2, 2))

        title = f"Lesion {info['id']}"
        lines = self._lines(info)
        stale_line = "edited - save to update" if self._stale else None

        fm = QFontMetrics(self._font)
        fm_title = QFontMetrics(self._title_font)
        line_h = fm.height()
        pad = 6
        text_w = max([fm_title.horizontalAdvance(title)]
                     + [fm.horizontalAdvance(t) for t in lines]
                     + ([fm.horizontalAdvance(stale_line)] if stale_line else []))
        n_lines = 1 + len(lines) + (1 if stale_line else 0)
        box_w = text_w + 2 * pad
        box_h = n_lines * line_h + 2 * pad

        # Right of the lesion; left of it when that would leave the canvas.
        gap = 10
        cw, ch = self.width(), self.height()
        bx = rect.right() + gap
        if bx + box_w > cw:
            bx = rect.left() - gap - box_w
        bx = max(2.0, min(bx, cw - box_w - 2))
        by = max(2.0, min(rect.top(), ch - box_h - 2))
        box = QRectF(bx, by, box_w, box_h)

        # Leader line from the box to the nearest point of the outline
        anchor_x = rect.right() if box.left() >= rect.right() else rect.left()
        anchor_y = min(max(box.top() + line_h / 2 + pad, rect.top()), rect.bottom())
        box_x = box.left() if box.left() >= rect.right() else box.right()
        painter.setPen(QPen(self._ACCENT, 1))
        painter.drawLine(int(anchor_x), int(anchor_y), int(box_x), int(anchor_y))

        painter.setPen(QPen(self._ACCENT, 1))
        painter.setBrush(self._BOX_BG)
        painter.drawRoundedRect(box, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        y = by + pad + fm.ascent()
        painter.setFont(self._title_font)
        painter.setPen(self._ACCENT)
        painter.drawText(int(bx + pad), int(y), title)
        painter.setFont(self._font)
        painter.setPen(self._TEXT)
        for t in lines:
            y += line_h
            painter.drawText(int(bx + pad), int(y), t)
        if stale_line:
            y += line_h
            painter.setPen(self._STALE)
            painter.drawText(int(bx + pad), int(y), stale_line)
