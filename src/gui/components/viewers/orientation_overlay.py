"""3D orientation overlay drawn on top of the 3D Napari canvas.

A transparent QWidget child of the canvas native widget (same approach as
``CrosshairOverlay``). It draws

* the anatomical direction at each canvas edge (R/L/A/P/S/I, combined such as
  "LA" when the view is oblique), and
* a readout of the camera orientation relative to the anterior coronal view:
  nearest standard view + off-axis angle, azimuth, elevation, roll and the view
  direction in patient RAS coordinates (see ``utils.view_orientation``).

It repaints on every camera rotation.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QEvent, QRect
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics

from ....utils.view_orientation import describe_camera


class OrientationOverlay(QWidget):
    """Edge orientation letters + numeric orientation readout for the 3D view."""

    _LETTER_COLOR = QColor(255, 220, 90)
    _TEXT_COLOR = QColor(255, 255, 136)
    _BOX_COLOR = QColor(0, 0, 0, 170)
    _BORDER_COLOR = QColor(85, 85, 85)
    _COLORBAR_RESERVE = 80    # left edge: keep clear of the ColorBarOverlay

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref
        self._enabled = False

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        parent.installEventFilter(self)
        self.resize(parent.size())
        self.raise_()
        self._vw.viewer.camera.events.angles.connect(self._on_camera_changed)
        self.setVisible(False)

    def eventFilter(self, source, event: QEvent):
        if source is self.parent() and event.type() == QEvent.Type.Resize:
            self.resize(source.size())
            self.update()
        return False

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.setVisible(enabled)
        self.update()

    def _on_camera_changed(self, event=None):
        if self._enabled:
            self.update()

    def _has_volume(self) -> bool:
        return any(layer.visible and layer.ndim == 3 and layer.data.size > 1
                   for layer in self._vw.viewer.layers if hasattr(layer, "contrast_limits"))

    def current(self):
        cam = self._vw.viewer.camera
        return describe_camera(cam.view_direction, cam.up_direction)

    # ── painting ────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._enabled or not self._has_volume():
            return
        o = self.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_edge_letters(painter, o)
        self._paint_readout(painter, o)
        painter.end()

    def _paint_edge_letters(self, painter, o):
        font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        w, h, m = self.width(), self.height(), 10
        placements = {
            "up":    (o.up,    lambda tw: (w // 2 - tw // 2, m + fm.ascent())),
            "down":  (o.down,  lambda tw: (w // 2 - tw // 2, h - m - fm.descent())),
            "left":  (o.left,  lambda tw: (self._COLORBAR_RESERVE, h // 2 + fm.ascent() // 2)),
            "right": (o.right, lambda tw: (w - m - tw, h // 2 + fm.ascent() // 2)),
        }
        for text, pos in placements.values():
            x, y = pos(fm.horizontalAdvance(text))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x + 1, y + 1, text)
            painter.setPen(self._LETTER_COLOR)
            painter.drawText(x, y, text)

    def _paint_readout(self, painter, o):
        def z(v, eps):        # no "-0.0" for values that round to zero
            return 0.0 if abs(v) < eps else v
        lx, ly, lz = (z(c, 5e-4) for c in o.look_ras)
        lines = [
            "3D ORIENTATION  (ref: anterior coronal)",
            f"View from : {o.view_from:<13} off-axis {z(o.off_axis, 0.05):5.1f}°",
            f"Azimuth   : {z(o.azimuth, 0.05):+7.1f}°   (+90 = left lateral)",
            f"Elevation : {z(o.elevation, 0.05):+7.1f}°   (+ = toward head)",
            f"Roll      : {z(o.roll, 0.05):+7.1f}°   (+ = clockwise)",
            f"Look (RAS): {lx:+.3f} {ly:+.3f} {lz:+.3f}",
        ]
        font = QFont("monospace", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(font)
        fm = QFontMetrics(font)
        pad = 6
        box_w = max(fm.horizontalAdvance(line) for line in lines) + 2 * pad
        box_h = fm.height() * len(lines) + 2 * pad
        box = QRect(8, self.height() - box_h - 8, box_w, box_h)
        painter.fillRect(box, self._BOX_COLOR)
        painter.setPen(self._BORDER_COLOR)
        painter.drawRect(box)
        painter.setPen(self._TEXT_COLOR)
        for i, line in enumerate(lines):
            painter.drawText(box.left() + pad, box.top() + pad + fm.ascent() + i * fm.height(), line)
