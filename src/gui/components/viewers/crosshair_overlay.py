"""Crosshair overlay widget drawn on top of a Napari canvas.

A transparent QWidget child of the canvas native widget.
It reads the crosshair data position from a shared list and computes canvas
pixel coordinates via the camera transform at paint time — so it stays
correctly positioned through any pan/zoom without extra event plumbing.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QPainter, QPen, QColor


class CrosshairOverlay(QWidget):
    """Full-span yellow crosshair lines drawn over a Napari canvas.

    Args:
        viewer_widget_ref: the owning ViewerWidget (for camera/dims access).
        parent: the canvas.native widget (Qt parent for geometry tracking).
    """

    # Emitted when the user left-clicks through the overlay area.
    # Coordinates are in canvas-fraction space ([0,1] × [0,1]).
    sig_clicked = pyqtSignal(float, float)   # (frac_y, frac_x)

    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref          # ViewerWidget ref
        self._xhair_data_pos = None           # [z, y, x] in Napari data space
        self._enabled = False

        # Completely transparent: mouse events fall through to canvas below
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Resize to match parent immediately and on future resizes
        parent.installEventFilter(self)
        self.resize(parent.size())
        self.raise_()

    # ── event filter: keep overlay sized to parent canvas ──────────────────

    def eventFilter(self, source, event: QEvent):
        if source is self.parent() and event.type() == QEvent.Type.Resize:
            self.resize(source.size())
            self.update()
        return False   # don't consume

    # ── public API ──────────────────────────────────────────────────────────

    def set_pos_data(self, data_pos_zyx):
        """Set crosshair position in Napari ZYX data coords and schedule repaint."""
        self._xhair_data_pos = data_pos_zyx
        self.update()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.update()

    # ── painting ────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._enabled or self._xhair_data_pos is None:
            return

        px, py = self._vw._compute_canvas_pos(self._xhair_data_pos)
        if px is None:
            return

        w, h = self.width(), self.height()
        px_i, py_i = int(round(px)), int(round(py))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        pen = QPen(QColor(255, 255, 0, 210))
        pen.setWidth(1)
        painter.setPen(pen)

        # Horizontal line (full width)
        if 0 <= py_i < h:
            painter.drawLine(0, py_i, w - 1, py_i)
        # Vertical line (full height)
        if 0 <= px_i < w:
            painter.drawLine(px_i, 0, px_i, h - 1)

        painter.end()
