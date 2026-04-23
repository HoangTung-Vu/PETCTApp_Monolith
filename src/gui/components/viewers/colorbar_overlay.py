import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QEvent, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QLinearGradient, QBrush

class ColorBarOverlay(QWidget):
    def __init__(self, viewer_widget_ref, parent: QWidget):
        super().__init__(parent)
        self._vw = viewer_widget_ref

        # Transparent to mouse events so it doesn't block interaction
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        parent.installEventFilter(self)
        self.resize(parent.size())
        self.raise_()

        # Connections to trigger repaint when contrast limits or visibility changes
        self._vw.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._vw.viewer.layers.events.removed.connect(self._on_layers_changed)
        
        # We also need a timer or just listen to existing layers once
        self._on_layers_changed()

    def _trigger_update(self, event=None):
        self.update()

    def _on_layers_changed(self, event=None):
        for layer in self._vw.viewer.layers:
            if hasattr(layer.events, 'contrast_limits'):
                try:
                    layer.events.contrast_limits.disconnect(self._trigger_update)
                except Exception:
                    pass
                layer.events.contrast_limits.connect(self._trigger_update)
            if hasattr(layer.events, 'visible'):
                try:
                    layer.events.visible.disconnect(self._trigger_update)
                except Exception:
                    pass
                layer.events.visible.connect(self._trigger_update)
            if hasattr(layer.events, 'colormap'):
                try:
                    layer.events.colormap.disconnect(self._trigger_update)
                except Exception:
                    pass
                layer.events.colormap.connect(self._trigger_update)
        self.update()

    def eventFilter(self, source, event: QEvent):
        if source is self.parent() and event.type() == QEvent.Type.Resize:
            self.resize(source.size())
            self.update()
        return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        margin = 15
        bar_width = 12
        bar_height = 100
        
        y_offset = margin
        
        ct_name = self._vw.LAYER_NAMES.get("ct")
        pet_name = self._vw.LAYER_NAMES.get("pet")
        
        for name in [pet_name, ct_name]:
            if not name or name not in self._vw.viewer.layers:
                continue
            layer = self._vw.viewer.layers[name]
            if not layer.visible or layer.opacity == 0:
                continue
                
            cmin, cmax = layer.contrast_limits
            
            # Map colors from colormap
            grad = QLinearGradient(0, 1, 0, 0) # Bottom to Top
            grad.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
            
            try:
                # evaluate the colormap at several points
                vals = np.linspace(0, 1, 32)
                mapped_colors = layer.colormap.map(vals)
                for i, color_rgba in enumerate(mapped_colors):
                    r, g, b, a = [int(v*255) for v in color_rgba]
                    grad.setColorAt(vals[i], QColor(r, g, b))
            except Exception:
                grad.setColorAt(0, Qt.GlobalColor.black)
                grad.setColorAt(1, Qt.GlobalColor.white)
                
            # Draw background black
            rect = QRect(margin, y_offset, bar_width, bar_height)
            painter.fillRect(rect, Qt.GlobalColor.black)
            
            # Draw gradient
            painter.fillRect(rect, QBrush(grad))
            
            # Draw border
            painter.setPen(QPen(QColor(128, 128, 128), 1))
            painter.drawRect(rect)
            
            # Draw Text and Ticks
            font = QFont("Arial", 10)
            painter.setFont(font)
            
            text_x = margin + bar_width + 6
            num_ticks = 5
            tick_values = np.linspace(cmin, cmax, num_ticks)
            
            for i, val in enumerate(tick_values):
                fraction = i / (num_ticks - 1)
                t_y = y_offset + bar_height - (fraction * bar_height)
                
                tick_str = f"{val:.1f}" if name != ct_name else f"{int(val)}"
                
                # Draw a small tick line
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawLine(margin + bar_width, int(t_y), margin + bar_width + 4, int(t_y))
                
                text_y = int(t_y) + 4 # approximate vertical center adjust
                
                if i == 0:
                    text_y = int(t_y)
                elif i == num_ticks - 1:
                    text_y = int(t_y) + 10
                
                # add shadow to text for visibility
                painter.setPen(Qt.GlobalColor.black)
                painter.drawText(text_x + 1, text_y + 1, tick_str)
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(text_x, text_y, tick_str)
                
            y_offset += bar_height + margin + 10

        painter.end()
