"""Eraser tool (connected component removal) mixin for LayoutManager."""

import numpy as np


class EraserMixin:
    """Mixin providing eraser click-to-remove-component functionality."""

    def enable_eraser_click_mode(self):
        """Install single-click callback on all 2D viewers for contour erasing."""
        self._remove_eraser_callbacks()

        for v in self._get_all_2d_viewers():
            callback = self._make_eraser_callback()
            v._eraser_callback = callback
            v.viewer.mouse_double_click_callbacks.append(callback)

    def disable_eraser_click_mode(self):
        """Remove eraser click callbacks from all viewers."""
        self._remove_eraser_callbacks()

    def _remove_eraser_callbacks(self):
        for v in self._get_all_2d_viewers():
            if hasattr(v, '_eraser_callback') and v._eraser_callback is not None:
                try:
                    v.viewer.mouse_double_click_callbacks.remove(v._eraser_callback)
                except ValueError:
                    pass
                v._eraser_callback = None

    def _world_to_data(self, position):
        """Convert Napari world coordinates to integer ZYX data indices.

        Napari world coords = data_index * scale, so invert by dividing.
        Falls back to rounding if no scale is set.
        """
        scale = None
        for v in self._get_all_2d_viewers():
            s = getattr(v, '_scale_zyx', None)
            if s is not None:
                scale = s
                break

        pos = np.asarray(position, dtype=float)
        if scale is not None:
            sc = np.asarray(scale, dtype=float)
            # trim/pad if dims differ (2D viewer may give fewer coords)
            n = min(len(pos), len(sc))
            idx = pos.copy()
            idx[:n] = pos[:n] / sc[:n]
        else:
            idx = pos
        return tuple(int(round(c)) for c in idx)

    def _make_eraser_callback(self):
        """Create a mouse callback that erases the connected component at click."""
        def on_click(viewer, event):
            coord_zyx = self._world_to_data(event.position)
            z, y, x = coord_zyx
            print(f"[Eraser] Click at ZYX={coord_zyx}")

            mask_zyx = self._cached_data_zyx.get("tumor")
            if mask_zyx is None:
                print("[Eraser] No tumor mask loaded.")
                return

            if not (0 <= z < mask_zyx.shape[0] and
                    0 <= y < mask_zyx.shape[1] and
                    0 <= x < mask_zyx.shape[2]):
                print("[Eraser] Click out of bounds.")
                return

            if mask_zyx[z, y, x] == 0:
                print("[Eraser] Clicked on background (label=0), nothing to erase.")
                self.sig_eraser_background_click.emit()
                return

            from ...workers import EraserFloodWorker
            
            # Prevent multiple simultaneous eraser tasks
            if hasattr(self, '_eraser_worker') and self._eraser_worker.isRunning():
                print("[Eraser] Please wait, another eraser operation is running.")
                return

            self._eraser_worker = EraserFloodWorker(mask_zyx, coord_zyx)
            
            # Show a busy cursor/status by emitting up to the main window
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import Qt
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            def _on_component_found(component_mask_zyx):
                QApplication.restoreOverrideCursor()
                
                num_voxels = int(np.sum(component_mask_zyx))
                if num_voxels == 0:
                    return
                    
                # Identify the exact 3D indices (in ZYX) to emit
                component_indices_zyx = np.nonzero(component_mask_zyx)
                
                # Apply in-place removal to the master numpy array buffer
                mask_zyx[component_mask_zyx] = 0
                print(f"[Eraser] Removed component at {coord_zyx} ({num_voxels} voxels).")
                
                # Emit the diff directly instead of doing heavy numpy copies.
                # old_mask_xyz vs new_mask_xyz is skipped in favor of a direct diff.
                self.sig_eraser_region_removed.emit(component_indices_zyx, component_mask_zyx, mask_zyx)
                
            def _on_error(msg):
                QApplication.restoreOverrideCursor()
                print(f"[Eraser Worker Error] {msg}")

            self._eraser_worker.component_found.connect(_on_component_found)
            self._eraser_worker.error.connect(_on_error)
            
            # Safely release reference after it finishes
            self._eraser_worker.finished.connect(self._eraser_worker.deleteLater)
            
            self._eraser_worker.start()

        return on_click
