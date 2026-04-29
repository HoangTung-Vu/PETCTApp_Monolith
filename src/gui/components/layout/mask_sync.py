"""Mask synchronization mixin for LayoutManager.

Handles connecting/disconnecting mask layer data-change events and
refreshing ONLY the currently visible viewers on paint/erase.

Includes debounced auto-sync: emits ``sig_mask_painted`` after
painting finishes (300ms debounce) so MainWindow can sync back
to session manager without a manual Sync button.
"""

from PyQt6.QtCore import QTimer, pyqtSignal


class MaskSyncMixin:
    """Mixin providing mask data-change event synchronisation."""

    sig_mask_painted = pyqtSignal(str)

    def _init_mask_sync(self):
        self._paint_debounce_timer = QTimer()
        self._paint_debounce_timer.setSingleShot(True)
        self._paint_debounce_timer.setInterval(300)
        self._paint_debounce_timer.timeout.connect(self._emit_debounced_paint)
        self._last_painted_layer = None

        self._visual_throttle_timer = QTimer()
        self._visual_throttle_timer.setSingleShot(True)
        self._visual_throttle_timer.setInterval(60)
        self._visual_throttle_timer.timeout.connect(self._do_visual_refresh)
        self._pending_refresh_layers = set()

    def _get_visible_viewers(self):
        """Return viewer widgets currently shown on screen."""
        # 3D view takes precedence when the stack shows the 3D widget
        if self.stack.currentWidget() == self.view_3d_widget:
            return [self.viewer_3d]
        # Otherwise return all pool viewers assigned to the active 2D grid
        return [self._fixed_view_map[v_id] for v_id in self._active_views]

    def _connect_mask_events(self):
        """Connect mask layer data/paint events on the CURRENTLY VISIBLE viewers only."""
        self._disconnect_mask_events()
        for v in self._get_visible_viewers():
            for layer_name in ("Tumor Mask", "ROI Mask"):
                if layer_name in v.viewer.layers:
                    layer = v.viewer.layers[layer_name]
                    layer.events.data.connect(self._on_mask_data_changed)
                    if hasattr(layer.events, 'paint'):
                        layer.events.paint.connect(self._on_mask_data_changed)

    def _disconnect_mask_events(self):
        """Disconnect mask layer data/paint events from ALL pool viewers (safety)."""
        all_viewers = list(self._viewer_pool)
        if self._is_3d_loaded:
            all_viewers.append(self.viewer_3d)

        for v in all_viewers:
            for layer_name in ("Tumor Mask", "ROI Mask"):
                if layer_name in v.viewer.layers:
                    try:
                        v.viewer.layers[layer_name].events.data.disconnect(
                            self._on_mask_data_changed
                        )
                    except Exception:
                        pass
                    try:
                        if hasattr(v.viewer.layers[layer_name].events, 'paint'):
                            v.viewer.layers[layer_name].events.paint.disconnect(
                                self._on_mask_data_changed
                            )
                    except Exception:
                        pass

    def _on_mask_data_changed(self, event):
        trigger_layer = event.source
        layer_name = trigger_layer.name

        self._pending_refresh_layers.add((trigger_layer, layer_name))

        if not self._visual_throttle_timer.isActive():
            self._visual_throttle_timer.start()

        name_to_type = {"Tumor Mask": "tumor", "ROI Mask": "roi"}
        self._last_painted_layer = name_to_type.get(layer_name, "tumor")
        self._paint_debounce_timer.start()

    def _do_visual_refresh(self):
        for trigger_layer, layer_name in list(self._pending_refresh_layers):
            for v in self._get_visible_viewers():
                if layer_name in v.viewer.layers:
                    layer = v.viewer.layers[layer_name]
                    if layer is not trigger_layer:
                        layer.refresh()
        self._pending_refresh_layers.clear()

    def _emit_debounced_paint(self):
        if self._last_painted_layer:
            self.sig_mask_painted.emit(self._last_painted_layer)
