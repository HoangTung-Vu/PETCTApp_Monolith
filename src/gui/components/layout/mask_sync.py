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

    # Signal emitted after debounced paint/erase: (layer_type: str)
    sig_mask_painted = pyqtSignal(str)

    def _init_mask_sync(self):
        """Call once from LayoutManager.__init__ to set up the debounce timer."""
        self._paint_debounce_timer = QTimer()
        self._paint_debounce_timer.setSingleShot(True)
        self._paint_debounce_timer.setInterval(300)  # ms
        self._paint_debounce_timer.timeout.connect(self._emit_debounced_paint)
        self._last_painted_layer = None

    def _get_visible_viewers(self):
        """Return only the viewer widgets that belong to the currently shown layout."""
        current = self.stack.currentWidget()
        if current == self.grid_widget:
            return list(self.grid_viewers.values())
        elif current == self.overlay_widget:
            return [self.overlay_viewer]
        elif current == self.mono_widget:
            return list(self.mono_viewers.values())
        elif current == self.mono_single_widget:
            return list(self.mono_single_viewers.values())
        elif current == self.view_3d_widget:
            return [self.viewer_3d]
        return []

    def _connect_mask_events(self):
        """Connect mask layer data events on the CURRENTLY VISIBLE viewers only."""
        self._disconnect_mask_events()
        for v in self._get_visible_viewers():
            for layer_name in ("Tumor Mask",):
                if layer_name in v.viewer.layers:
                    layer = v.viewer.layers[layer_name]
                    layer.events.data.connect(self._on_mask_data_changed)

    def _disconnect_mask_events(self):
        """Disconnect mask layer data events from ALL viewers (safety)."""
        all_viewers = (
            list(self.grid_viewers.values())
            + [self.overlay_viewer]
            + list(self.mono_viewers.values())
            + list(self.mono_single_viewers.values())
        )
        if self._is_3d_loaded:
            all_viewers.append(self.viewer_3d)

        for v in all_viewers:
            for layer_name in ("Tumor Mask",):
                if layer_name in v.viewer.layers:
                    try:
                        v.viewer.layers[layer_name].events.data.disconnect(
                            self._on_mask_data_changed
                        )
                    except Exception:
                        pass

    def _on_mask_data_changed(self, event):
        """Refresh only the OTHER visible viewers sharing this mask layer.
        Also start the debounce timer for auto-sync."""
        trigger_layer = event.source
        layer_name = trigger_layer.name

        for v in self._get_visible_viewers():
            if layer_name in v.viewer.layers:
                layer = v.viewer.layers[layer_name]
                if layer is not trigger_layer:
                    layer.refresh()

        # Debounced auto-sync: determine layer type from name
        name_to_type = {"Tumor Mask": "tumor"}
        self._last_painted_layer = name_to_type.get(layer_name, "tumor")
        self._paint_debounce_timer.start()  # restarts if already running

    def _emit_debounced_paint(self):
        """Emit signal after painting stops for 300ms."""
        if self._last_painted_layer:
            self.sig_mask_painted.emit(self._last_painted_layer)
