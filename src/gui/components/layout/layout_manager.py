"""Layout Manager — manages view switching, data loading, and display settings.

Performance optimizations:
- **Lazy loading**: Only pushes data to the CURRENTLY VISIBLE layout.
  Other layouts load on-demand when switched to via ``set_view_mode()``.
- **No cross-layout sync**: Removed ``_sync_all_layouts()`` that chained
  9 viewers together causing ~30+ cascading events per scroll.
- **One-shot sync**: When switching view mode, the current slice position
  is copied once from the old layout to the new one.
"""

from PyQt6.QtWidgets import QWidget, QGridLayout, QStackedWidget, QVBoxLayout, QApplication
from PyQt6.QtCore import pyqtSignal
import numpy as np

from ..viewers.viewer_widget import ViewerWidget
from ..viewers.viewer_sync import link_dims, link_camera, one_shot_sync_step
from .mask_sync import MaskSyncMixin
from .autopet_click_manager import AutoPETClickMixin
from .eraser_manager import EraserMixin
from ....utils.dimension_utils import get_spacing_from_affine


class LayoutManager(MaskSyncMixin, AutoPETClickMixin, EraserMixin, QWidget):
    """Manages switching between Grid, Overlay, Mono, and 3D layouts.
    Synchronizes viewers ONLY within the same visible layout.
    """

    # Signal emitted when user clicks in autopet mode: (coord_zyx_list, label)
    sig_autopet_click_added = pyqtSignal(list, str)

    # Signal emitted when eraser removes a connected component: (old_mask_xyz, new_mask_xyz)
    sig_eraser_region_removed = pyqtSignal(object, object)

    # Signal emitted after debounced paint stroke: (layer_type: str)
    sig_mask_painted = pyqtSignal(str)

    # Signal emitted when a shape is committed: (layer_type: str)
    sig_shape_committed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # Initialize Layouts
        self._init_grid_view()
        self._init_overlay_view()
        self._init_mono_view()
        self._init_3d_view()

        # Sync ONLY within the same layout (no cross-layout sync)
        self._sync_grid_views()
        self._sync_mono_views()

        # Init mask auto-sync debounce timer
        self._init_mask_sync()

        # Data Caching for Lazy Loading
        self._cached_data = {
            "ct": None, "pet": None, "affine": None,
            "tumor": None
        }
        self._cached_data_zyx = {
            "tumor": None
        }
        self._is_3d_loaded = False
        self._cached_lesion_data = None  # (bboxes, ids)

        # Track which layouts have been loaded with current data
        self._loaded_layouts = set()

        # Contrast states (Window, Level)
        self._ct_wl = (350.0, 35.0)
        self._pet_wl = (10.0, 5.0)

    # ──── View Initialization ────

    def _init_grid_view(self):
        """6-Cell Grid: Axial/Sagittal/Coronal × CT/PET."""
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        self.grid_viewers = {}
        for r in range(3):
            for c in range(2):
                v = ViewerWidget()
                self.grid_layout.addWidget(v, r, c)
                self.grid_viewers[(r, c)] = v

                if r == 0: v.set_camera_view(0)   # Axial
                if r == 1: v.set_camera_view(2)   # Sagittal
                if r == 2: v.set_camera_view(1)   # Coronal

        self.stack.addWidget(self.grid_widget)

    def _init_overlay_view(self):
        self.overlay_widget = QWidget()
        layout = QVBoxLayout(self.overlay_widget)
        self.overlay_viewer = ViewerWidget()
        self.overlay_viewer.set_camera_view(0)
        layout.addWidget(self.overlay_viewer)
        self.stack.addWidget(self.overlay_widget)

    def _init_mono_view(self):
        """Side-by-side view: CT (Left), PET (Right)."""
        self.mono_widget = QWidget()
        self.mono_layout = QGridLayout(self.mono_widget)
        self.mono_layout.setContentsMargins(0, 0, 0, 0)

        self.mono_viewers = {}
        for i in range(2):
            v = ViewerWidget()
            self.mono_layout.addWidget(v, 0, i)
            self.mono_viewers[i] = v
            v.set_camera_view(0)

        self.stack.addWidget(self.mono_widget)

    def _init_3d_view(self):
        self.view_3d_widget = QWidget()
        layout = QVBoxLayout(self.view_3d_widget)
        self.viewer_3d = ViewerWidget()
        self.viewer_3d.set_3d_view()
        layout.addWidget(self.viewer_3d)
        self.stack.addWidget(self.view_3d_widget)

    # ──── Intra-layout Sync ────

    def _sync_grid_views(self):
        """Sync slices + camera per row (CT ↔ PET)."""
        for r in range(3):
            v1 = self.grid_viewers[(r, 0)].viewer
            v2 = self.grid_viewers[(r, 1)].viewer
            link_dims(v1, v2)
            link_camera(v1, v2)

    def _sync_mono_views(self):
        v1 = self.mono_viewers[0].viewer
        v2 = self.mono_viewers[1].viewer
        link_dims(v1, v2)
        link_camera(v1, v2)

    # ──── Data Loading (Lazy) ────

    def load_data(self, ct_data, pet_data, affine, tumor_mask=None):
        """Cache data and load ONLY into the currently visible layout."""
        # Update cache
        self._cached_data["ct"] = ct_data
        self._cached_data["pet"] = pet_data
        self._cached_data["affine"] = affine
        self._cached_data["tumor"] = tumor_mask
        self._is_3d_loaded = False
        self._loaded_layouts.clear()

        # Convert masks to Napari space ONCE
        from ....utils.nifti_utils import to_napari
        # Only load Tumor Mask into viewers.
        if tumor_mask is not None:
            self._cached_data_zyx["tumor"] = to_napari(tumor_mask.astype(np.uint8))
        else:
            self._cached_data_zyx["tumor"] = None

        # Load only the visible layout
        self._load_current_layout()

        # Connect mask events for the visible layout
        self._connect_mask_events()

    def _load_current_layout(self):
        """Load cached data into whichever layout is currently visible."""
        current = self.stack.currentWidget()

        if current == self.grid_widget:
            self._load_grid()
        elif current == self.overlay_widget:
            self._load_overlay()
        elif current == self.mono_widget:
            self._load_mono()
        elif current == self.view_3d_widget:
            self._load_3d_data()

    def _load_grid(self):
        """Load data into grid viewers."""
        if "grid" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        for (r, c), widget in self.grid_viewers.items():
            if c == 0 and ct is not None:
                widget.load_image(ct, affine, "ct", "gray")
            elif c == 1 and pet is not None:
                widget.load_image(pet, affine, "pet", "jet")

            if self._cached_data_zyx["tumor"] is not None:
                widget.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")

            if r == 0: widget.set_camera_view(0)
            elif r == 1: widget.set_camera_view(2)
            elif r == 2: widget.set_camera_view(1)

            # Apply persistent contrast
            ct_name = widget.LAYER_NAMES["ct"]
            pet_name = widget.LAYER_NAMES["pet"]
            if ct is not None and ct_name in widget.viewer.layers:
                c_min = self._ct_wl[1] - (self._ct_wl[0] / 2)
                c_max = self._ct_wl[1] + (self._ct_wl[0] / 2)
                widget.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
            if pet is not None and pet_name in widget.viewer.layers:
                p_min = max(0, self._pet_wl[1] - (self._pet_wl[0] / 2))
                p_max = self._pet_wl[1] + (self._pet_wl[0] / 2)
                widget.viewer.layers[pet_name].contrast_limits = (p_min, p_max)

            widget.viewer.reset_view()
            if self._cached_lesion_data:
                widget.show_lesion_ids(*self._cached_lesion_data)

        QApplication.processEvents()

        # Dynamically adjust row stretch
        if ct is not None:
            D_x, D_y, D_z = ct.shape
            spacing_xyz = get_spacing_from_affine(affine)
            sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
            self.grid_layout.setRowStretch(0, int(D_y * sy))
            self.grid_layout.setRowStretch(1, int(D_z * sz))
            self.grid_layout.setRowStretch(2, int(D_z * sz))

        self._loaded_layouts.add("grid")

    def _load_overlay(self):
        """Load data into overlay viewer."""
        if "overlay" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.overlay_viewer.load_image(ct, affine, "ct", "gray")
        if pet is not None:
            self.overlay_viewer.load_image(pet, affine, "pet", "jet", opacity=0.5)

        if self._cached_data_zyx["tumor"] is not None:
            self.overlay_viewer.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")

        # Apply persistent contrast
        ct_name = self.overlay_viewer.LAYER_NAMES["ct"]
        pet_name = self.overlay_viewer.LAYER_NAMES["pet"]
        if ct is not None and ct_name in self.overlay_viewer.viewer.layers:
            c_min = self._ct_wl[1] - (self._ct_wl[0] / 2)
            c_max = self._ct_wl[1] + (self._ct_wl[0] / 2)
            self.overlay_viewer.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
        if pet is not None and pet_name in self.overlay_viewer.viewer.layers:
            p_min = max(0, self._pet_wl[1] - (self._pet_wl[0] / 2))
            p_max = self._pet_wl[1] + (self._pet_wl[0] / 2)
            self.overlay_viewer.viewer.layers[pet_name].contrast_limits = (p_min, p_max)

        self.overlay_viewer.viewer.reset_view()
        if self._cached_lesion_data:
            self.overlay_viewer.show_lesion_ids(*self._cached_lesion_data)
        self._loaded_layouts.add("overlay")

    def _load_mono(self):
        """Load data into mono viewers."""
        if "mono" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.mono_viewers[0].load_image(ct, affine, "ct", "gray")
        if pet is not None:
            self.mono_viewers[1].load_image(pet, affine, "pet", "jet")

        for v in self.mono_viewers.values():
            if self._cached_data_zyx["tumor"] is not None:
                v.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
            
            # Apply persistent contrast
            ct_name = v.LAYER_NAMES["ct"]
            pet_name = v.LAYER_NAMES["pet"]
            if ct is not None and ct_name in v.viewer.layers:
                c_min = self._ct_wl[1] - (self._ct_wl[0] / 2)
                c_max = self._ct_wl[1] + (self._ct_wl[0] / 2)
                v.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
            if pet is not None and pet_name in v.viewer.layers:
                p_min = max(0, self._pet_wl[1] - (self._pet_wl[0] / 2))
                p_max = self._pet_wl[1] + (self._pet_wl[0] / 2)
                v.viewer.layers[pet_name].contrast_limits = (p_min, p_max)
                
            v.viewer.reset_view()
            if self._cached_lesion_data:
                v.show_lesion_ids(*self._cached_lesion_data)

        self._loaded_layouts.add("mono")

    def _load_3d_data(self):
        """Lazy load data into 3D viewer."""
        if self._is_3d_loaded:
            return

        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.viewer_3d.load_image(ct, affine, "ct", "gray")
        if pet is not None:
            self.viewer_3d.load_image(pet, affine, "pet", "jet", opacity=0.7)

        if self._cached_data_zyx["tumor"] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
        
        if self._cached_lesion_data:
            self.viewer_3d.show_lesion_ids(*self._cached_lesion_data)

        self.viewer_3d.viewer.dims.ndisplay = 3
        self._is_3d_loaded = True
        self._loaded_layouts.add("3d")

    # ──── View Mode Switching ────

    def set_view_mode(self, mode: str):
        # Capture current slice position for one-shot sync
        old_viewers = self._get_visible_viewers()
        old_viewer = old_viewers[0].viewer if old_viewers else None

        if mode == "grid":
            self.stack.setCurrentWidget(self.grid_widget)
            self._load_grid()

        elif mode.startswith("mono"):
            self.stack.setCurrentWidget(self.mono_widget)
            self._load_mono()
            if "axial" in mode:
                for v in self.mono_viewers.values(): v.set_camera_view(0)
            elif "coronal" in mode:
                for v in self.mono_viewers.values(): v.set_camera_view(1)
            elif "sagittal" in mode:
                for v in self.mono_viewers.values(): v.set_camera_view(2)
            else:
                for v in self.mono_viewers.values(): v.set_camera_view(0)
            for v in self.mono_viewers.values(): v.viewer.reset_view()

        elif mode.startswith("overlay"):
            self.stack.setCurrentWidget(self.overlay_widget)
            self._load_overlay()
            if "axial" in mode:
                self.overlay_viewer.set_camera_view(0)
            elif "coronal" in mode:
                self.overlay_viewer.set_camera_view(1)
            elif "sagittal" in mode:
                self.overlay_viewer.set_camera_view(2)
            else:
                self.overlay_viewer.set_camera_view(0)

        elif mode == "3d":
            self.stack.setCurrentWidget(self.view_3d_widget)
            self._load_3d_data()

        # One-shot sync: copy slice position from previous layout
        if old_viewer is not None:
            new_viewers = self._get_visible_viewers()
            target_napari = [v.viewer for v in new_viewers]
            one_shot_sync_step(old_viewer, target_napari)

        # Reconnect mask events for the new layout
        self._connect_mask_events()

    # ──── Display Settings ────

    def reset_zoom(self):
        """Reset view for currently visible viewers."""
        for v in self._get_visible_viewers():
            v.viewer.reset_view()

    def set_pet_opacity(self, value: float):
        """Update opacity for 'pet' layer in all loaded viewers."""
        all_viewers = self._get_all_loaded_viewers()
        for widget in all_viewers:
            for layer in widget.viewer.layers:
                if layer.name == widget.LAYER_NAMES["pet"]:
                    layer.opacity = value

    def set_tumor_opacity(self, value: float):
        """Update opacity for 'tumor' layer in all loaded viewers."""
        all_viewers = self._get_all_loaded_viewers()
        for widget in all_viewers:
            for layer in widget.viewer.layers:
                # The name in self.LAYER_NAMES or default "Tumor Mask"
                name = widget.LAYER_NAMES.get("tumor", "tumor")
                if layer.name == name:
                    layer.opacity = value

    def set_ct_window_level(self, window: float, level: float):
        self._ct_wl = (window, level)
        min_val = level - (window / 2)
        max_val = level + (window / 2)
        self._set_contrast_limits("ct", min_val, max_val)

    def set_pet_window_level(self, window: float, level: float):
        self._pet_wl = (window, level)
        min_val = level - (window / 2)
        max_val = level + (window / 2)
        if min_val < 0:
            min_val = 0
        self._set_contrast_limits("pet", min_val, max_val)

    def _set_contrast_limits(self, layer_type: str, min_val: float, max_val: float):
        all_viewers = self._get_all_loaded_viewers()
        if not all_viewers:
            return
        name = all_viewers[0].LAYER_NAMES.get(layer_type, layer_type)
        for widget in all_viewers:
            for layer in widget.viewer.layers:
                if layer.name == name:
                    layer.contrast_limits = (min_val, max_val)

    def set_zoom(self, value: float):
        zoom_factor = 0.1 + (value / 100.0) * 4.9

        # Only set zoom on one viewer per synced pair (sync propagates)
        current = self.stack.currentWidget()
        if current == self.grid_widget:
            self.grid_viewers[(0, 0)].viewer.camera.zoom = zoom_factor
            self.grid_viewers[(1, 0)].viewer.camera.zoom = zoom_factor
            self.grid_viewers[(2, 0)].viewer.camera.zoom = zoom_factor
        elif current == self.overlay_widget:
            self.overlay_viewer.viewer.camera.zoom = zoom_factor
        elif current == self.mono_widget:
            self.mono_viewers[0].viewer.camera.zoom = zoom_factor

    def toggle_mask(self, mask_type: str, visible: bool):
        all_viewers = self._get_all_loaded_viewers()
        name_map = {"tumor": "Tumor Mask"}
        target_name = name_map.get(mask_type, mask_type)

        for widget in all_viewers:
            for layer in widget.viewer.layers:
                if layer.name == target_name:
                    layer.visible = visible

    def toggle_3d_pet(self, visible: bool):
        """Show/Hide PET layer in 3D view."""
        for layer in self.viewer_3d.viewer.layers:
            if layer.name == self.viewer_3d.LAYER_NAMES["pet"]:
                layer.visible = visible

    def set_drawing_tool(self, tool: str, brush_size: int, layer_type: str):
        """Sets the drawing tool for visible 2D viewers."""
        for v in self._get_visible_viewers():
            if tool in ("sphere", "square"):
                # Disable Napari's built-in paint mode and enable shape drag
                v.set_drawing_mode(layer_type, "pan_zoom", brush_size)
                v.enable_shape_drag(layer_type, tool)
            else:
                # For pan_zoom and paint, disable shape drag first
                v.disable_shape_drag()
                v.set_drawing_mode(layer_type, tool, brush_size)

    def disable_shape_drag(self):
        """Disables shape dragging across all currently visible viewers."""
        for v in self._get_visible_viewers():
            v.disable_shape_drag()

    def commit_shape(self, layer_type: str):
        """Commit all shapes from visibility viewers to the mask."""
        for v in self._get_visible_viewers():
            v.commit_shape_to_mask()
        
        # After burning shapes into Napari layer, triggered events will sync 
        # to session via sig_mask_painted, but we might want an explicit sync here.
        self.sig_shape_committed.emit(layer_type)

    # ──── Mask Update ────

    def update_mask(self, mask_data, mask_type):
        """Update mask in visible viewers and cache. Lazy-loads others on demand."""
        if mask_data is None:
            return
        self._cached_data[mask_type] = mask_data

        from ....utils.nifti_utils import to_napari
        data_zyx = to_napari(mask_data.astype(np.uint8))
        self._cached_data_zyx[mask_type] = data_zyx

        # Disconnect events to prevent recursion
        self._disconnect_mask_events()

        # Push to visible viewers only
        for v in self._get_visible_viewers():
            v.load_mask_zyx(data_zyx, mask_type)

        # Also push to 3D if loaded
        if self._is_3d_loaded:
            self.viewer_3d.load_mask_zyx(data_zyx, mask_type)

        # Mark other layouts as stale so they re-load on switch
        self._loaded_layouts.discard("grid")
        self._loaded_layouts.discard("overlay")
        self._loaded_layouts.discard("mono")
        # Keep current layout marked as loaded
        current = self.stack.currentWidget()
        if current == self.grid_widget:
            self._loaded_layouts.add("grid")
        elif current == self.overlay_widget:
            self._loaded_layouts.add("overlay")
        elif current == self.mono_widget:
            self._loaded_layouts.add("mono")

        # Reconnect
        self._connect_mask_events()

    # ──── Mask Retrieval ────

    def get_active_mask_data(self, layer_type: str):
        """Retrieves mask data from the currently visible viewer. Returns (X, Y, Z) array."""
        for viewer in self._get_visible_viewers():
            data = viewer.get_layer_data(layer_type)
            if data is not None:
                return data
        return self._cached_data.get(layer_type)

    def sync_mask_cache(self, mask_data, mask_type):
        """Lightweight cache sync for auto-sync during painting.

        Updates the XYZ and ZYX caches and invalidates non-visible layouts
        WITHOUT re-pushing data to visible viewers (they already share the
        painted data via _on_mask_data_changed).  Also syncs the 3D viewer
        if it has been loaded.
        """
        self._cached_data[mask_type] = mask_data

        # Grab the ZYX data directly from whichever visible viewer was painted
        # so the cache points to the SAME array object — no redundant copy.
        synced = False
        for v in self._get_visible_viewers():
            name = v.LAYER_NAMES.get(mask_type, "")
            if name in v.viewer.layers:
                self._cached_data_zyx[mask_type] = v.viewer.layers[name].data
                synced = True
                break

        if not synced:
            if mask_data is None:
                self._cached_data_zyx[mask_type] = None
                return
            from ....utils.nifti_utils import to_napari
            self._cached_data_zyx[mask_type] = to_napari(mask_data.astype(np.uint8))

        # Sync 3D viewer if loaded
        if self._is_3d_loaded and self._cached_data_zyx[mask_type] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx[mask_type], mask_type)

        # Invalidate non-visible layouts so they re-load on switch
        current = self.stack.currentWidget()
        for layout_name in ("grid", "overlay", "mono"):
            self._loaded_layouts.discard(layout_name)
        if current == self.grid_widget:
            self._loaded_layouts.add("grid")
        elif current == self.overlay_widget:
            self._loaded_layouts.add("overlay")
        elif current == self.mono_widget:
            self._loaded_layouts.add("mono")

    # ──── Viewer Clear ────

    def clear_all_viewers(self):
        """Remove all layers from ALL viewers and reset caches."""
        all_viewers = (
            list(self.grid_viewers.values())
            + [self.overlay_viewer]
            + list(self.mono_viewers.values())
            + [self.viewer_3d]
        )
        for v in all_viewers:
            v.viewer.layers.clear()

        self._cached_data = {"ct": None, "pet": None, "affine": None, "tumor": None}
        self._cached_data_zyx = {"tumor": None}
        self._is_3d_loaded = False
        self._loaded_layouts.clear()
        self._click_markers = None
        self._cached_lesion_data = None

    # ──── Helpers ────

    def _get_all_loaded_viewers(self):
        """Return all viewer widgets that have been loaded with data."""
        viewers = []
        if "grid" in self._loaded_layouts:
            viewers.extend(self.grid_viewers.values())
        if "overlay" in self._loaded_layouts:
            viewers.append(self.overlay_viewer)
        if "mono" in self._loaded_layouts:
            viewers.extend(self.mono_viewers.values())
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)
        return viewers

    # ──── Lesion ID Labels ────

    def show_lesion_ids(self, bboxes: list, lesion_ids: list):
        """Push lesion ID labels to all LOADED viewers (including 3D)."""
        if not bboxes:
            self._cached_lesion_data = ([], [])
            for v in self._get_all_loaded_viewers():
                v.show_lesion_ids([], [])
            return

        # Perform the coordinate transformation ONCE for all viewers
        # Get shape from cached CT or PET
        nib_shape = None
        if self._cached_data.get("ct") is not None:
            nib_shape = self._cached_data["ct"].shape
        elif self._cached_data.get("pet") is not None:
            nib_shape = self._cached_data["pet"].shape
            
        points = []
        id_strings = []
        
        if nib_shape is not None:
            for bbox, lid in zip(bboxes, lesion_ids):
                # Centroid in nibabel array order
                d0_c = (bbox[0] + bbox[3]) / 2.0  # X center
                d1_c = (bbox[1] + bbox[4]) / 2.0  # Y center
                d2_c = (bbox[2] + bbox[5]) / 2.0  # Z center
    
                # Step 1: transpose — napari = (Z, Y, X) = (d2_c, d1_c, d0_c)
                z_nap = d2_c
                y_nap = d1_c
                x_nap = d0_c
    
                # Step 2: flip Z (axis 0 of ZYX space, range 0..Z_size-1)
                z_nap = (nib_shape[2] - 1) - z_nap
                # Step 3: flip Y (axis 1 of ZYX space, range 0..Y_size-1)
                y_nap = (nib_shape[1] - 1) - y_nap
    
                points.append([z_nap, y_nap, x_nap])
                id_strings.append(str(lid))

        self._cached_lesion_data = (points, id_strings)

        for v in self._get_all_loaded_viewers():
            v.show_lesion_ids(points, id_strings)

    def hide_lesion_ids(self):
        """Remove lesion ID labels from all viewers."""
        self._cached_lesion_data = None
        for v in self._get_all_loaded_viewers():
            v.hide_lesion_ids()
