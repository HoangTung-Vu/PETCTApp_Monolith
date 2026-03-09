"""AutoPET click handling mixin for LayoutManager."""

import numpy as np


class AutoPETClickMixin:
    """Mixin providing AutoPET Interactive click handling for LayoutManager."""

    def _get_all_2d_viewers(self):
        """Collect all 2D viewer widgets."""
        viewers = list(self.grid_viewers.values())
        viewers.append(self.overlay_viewer)
        viewers.extend(self.mono_viewers.values())
        return viewers

    def enable_autopet_click_mode(self, label: str):
        """Install mouse callback on all 2D viewers.
        label: 'tumor' or 'background'. Empty string disables.
        """
        if not label:
            self.disable_autopet_click_mode()
            return

        #disable eraser when enabling autopet
        self.disable_eraser_click_mode()

        self._autopet_click_label = label
        self._remove_autopet_callbacks()

        for v in self._get_all_2d_viewers():
            callback = self._make_click_callback()
            v._autopet_callback = callback
            v.viewer.mouse_double_click_callbacks.append(callback)

    def disable_autopet_click_mode(self):
        """Remove mouse callbacks from all viewers."""
        self._autopet_click_label = None
        self._remove_autopet_callbacks()

    def _remove_autopet_callbacks(self):
        for v in self._get_all_2d_viewers():
            if hasattr(v, '_autopet_callback') and v._autopet_callback is not None:
                try:
                    v.viewer.mouse_double_click_callbacks.remove(v._autopet_callback)
                except ValueError:
                    pass
                v._autopet_callback = None

    def _world_to_data(self, world_pos):
        """Convert world-space position to data (voxel) indices using cached scale."""
        scale = None
        for v in self._get_all_2d_viewers():
            if v._scale_zyx is not None:
                scale = v._scale_zyx
                break

        if scale is not None:
            return [round(w / s) for w, s in zip(world_pos, scale)]
        else:
            return [round(c) for c in world_pos]

    def _make_click_callback(self):
        """Create a mouse callback that captures click coordinates."""
        def on_double_click(viewer, event):
            label = getattr(self, '_autopet_click_label', None)
            if label is None:
                return
            coord_zyx = self._world_to_data(event.position)
            print(f"[AutoPET] Layout Click: {label} at Napari ZYX={coord_zyx}")

            # Paint sphere into shared array (uses Napari coordinates)
            self._paint_click_sphere(coord_zyx, label)
            
            # The backend AutoPET Engine takes raw NIfTI files and converts them 
            # to SimpleITK arrays: shape = (Z, Y, X), spacing = (sz, sy, sx).
            # The NIfTI files were NOT horizontally/vertically flipped like in Napari,
            # so we must undo the Napari flips before sending the coordinates to the backend.
            
            # 1. Convert Napari coord (Z, Y, X) -> Nibabel coord (X, Y, Z)
            shape_zyx = self._ensure_click_array().shape
            from ....utils.nifti_utils import point_from_napari
            coord_xyz_nibabel = point_from_napari(coord_zyx, shape_zyx)
            
            # 2. Convert Nibabel coord (X, Y, Z) -> SimpleITK coord (Z, Y, X)
            # which is what the autoPET-interactive predictor uses internally.
            coord_zyx_sitk = [coord_xyz_nibabel[2], coord_xyz_nibabel[1], coord_xyz_nibabel[0]]

            # Emit transformed signal for main_window to track
            print(f"[AutoPET] Transformed Click sent to worker: {label} at SITK ZYX={coord_zyx_sitk}")
            self.sig_autopet_click_added.emit(coord_zyx_sitk, label)

        return on_double_click

    def _ensure_click_array(self):
        """Lazily create the shared click markers array matching image shape."""
        if not hasattr(self, '_click_markers') or self._click_markers is None:
            shape = None
            for key in ("ct", "pet"):
                d = self._cached_data.get(key)
                if d is not None:
                    from ....utils.nifti_utils import to_napari
                    shape = to_napari(d).shape
                    break
            if shape is None:
                return None
            self._click_markers = np.zeros(shape, dtype=np.uint8)
        return self._click_markers

    def _paint_click_sphere(self, coord_zyx, label: str, radius: int = 3):
        """Paint a sphere at coord into the shared markers array and push to viewers."""
        arr = self._ensure_click_array()
        if arr is None:
            return

        val = 1 if label == "tumor" else 2
        z, y, x = coord_zyx
        shape = arr.shape

        z0, z1 = max(0, z - radius), min(shape[0], z + radius + 1)
        y0, y1 = max(0, y - radius), min(shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(shape[2], x + radius + 1)

        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist_sq = (zz - z)**2 + (yy - y)**2 + (xx - x)**2
        arr[z0:z1, y0:y1, x0:x1][dist_sq <= radius**2] = val

        self._push_click_markers(arr)

    def _push_click_markers(self, arr):
        """Push click markers array to all 2D viewers."""
        for v in self._get_all_2d_viewers():
            v.load_click_markers(arr)

    def clear_autopet_clicks(self):
        """Clear all click markers from all viewers."""
        if hasattr(self, '_click_markers') and self._click_markers is not None:
            self._click_markers[:] = 0
        for v in self._get_all_2d_viewers():
            v.remove_click_markers()
        self._click_markers = None
