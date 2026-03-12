from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QObject, QEvent
import napari
import numpy as np
from typing import Optional, Tuple
import nibabel as nib

from ....utils.nifti_utils import to_napari, from_napari
from ....utils.dimension_utils import get_spacing_from_affine

class ViewerWidget(QWidget):
    """
    A unified widget containing a single Napari viewer.
    Handles data loading and basic display settings.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize Napari Viewer
        # ndisplay=2 by default
        self.viewer = napari.Viewer(show=False)
        self.qt_viewer = self.viewer.window.qt_viewer
        self.layout.addWidget(self.qt_viewer)

        self.viewer.camera.mouse_zoom = False
        
        # Add custom mouse wheel callback to swap scroll/zoom
        # Napari native callbacks still let some events through or double fire,
        # so we intercept the raw Qt wheel event on the canvas directly.
        if hasattr(self.qt_viewer, 'canvas') and hasattr(self.qt_viewer.canvas, 'native'):
            self.qt_viewer.canvas.native.installEventFilter(self)

        # Disable default double click zoom
        for cb in list(self.viewer.mouse_double_click_callbacks):
            if cb.__name__ == 'double_click_to_zoom':
                self.viewer.mouse_double_click_callbacks.remove(cb)
        
        # Keep track of layer names
        self.LAYER_NAMES = {
            "ct": "CT Image",
            "pet": "PET Image",
            "tumor": "Tumor Mask"
        }
        self.is_3d = False
        self._scale_zyx = None  # Physical spacing in Napari (Z, Y, X) order
    
    # helper proxies for compatibility (optional, or just use imported functions)
    @staticmethod
    def to_napari(data: np.ndarray) -> np.ndarray:
        return to_napari(data)

    @staticmethod
    def from_napari(data_zyx: np.ndarray) -> np.ndarray:
        return from_napari(data_zyx)

    def load_image(self, image_data: np.ndarray, affine: np.ndarray, 
                   layer_type: str, colormap: str = "gray", 
                   blending: str = "translucent",
                   opacity: float = 1.0):
        """
        Loads an image into the viewer.
        image_data: (X, Y, Z) array from nibabel
        """
        data_zyx = self.to_napari(image_data)
        
        # Extract physical spacing from affine: |col_i| for axes X, Y, Z
        spacing_xyz = get_spacing_from_affine(affine)  # (sx, sy, sz)
        # Reorder to Napari (Z, Y, X)
        self._scale_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data_zyx
            self.viewer.layers[name].scale = self._scale_zyx
            self.viewer.layers[name].colormap = colormap
            self.viewer.layers[name].blending = blending
            self.viewer.layers[name].opacity = opacity
        else:
            self.viewer.add_image(
                data_zyx,
                name=name,
                scale=self._scale_zyx,
                colormap=colormap,
                blending=blending,
                opacity=opacity,
            )
            
    def load_mask(self, mask_data: np.ndarray, layer_type: str, color: Optional[int] = None):
        """
        Loads a label mask.
        """
        # Ensure integer type for Labels layer
        mask_data = mask_data.astype(np.uint8)
        data_zyx = self.to_napari(mask_data)
        self.load_mask_zyx(data_zyx, layer_type)
            
    def load_mask_zyx(self, data_zyx: np.ndarray, layer_type: str):
        """
        Loads a label mask already in Napari space (Z, Y, X).
        """
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            # If the data is already the same object, setting it might still trigger a refresh
            # But let's check if we can avoid redundant updates
            if layer.data is not data_zyx:
                layer.data = data_zyx
        else:
            kwargs = dict(name=name, opacity=0.7)
            if self._scale_zyx is not None:
                kwargs['scale'] = self._scale_zyx
            layer = self.viewer.add_labels(data_zyx, **kwargs)
            
        # Lock editing if this is a 3D viewer
        if self.is_3d:
            layer.editable = False
            
    def get_layer_data(self, layer_type: str) -> Optional[np.ndarray]:
        """
        Returns the underlying data of a layer in Nibabel (X, Y, Z) format.
        Useful for retrieving manually drawn masks.
        """
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name in self.viewer.layers:
            data_zyx = self.viewer.layers[name].data
            return self.from_napari(data_zyx)
        return None

    def set_drawing_mode(self, layer_type: str, mode: str, brush_size: float = 10):
        """
        Sets the interaction mode for a specific layer.
        mode: 'pan_zoom', 'paint', 'erase', 'fill'
        """
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        
        # Ensure layer exists and is selected
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            self.viewer.layers.selection.active = layer
            
            if mode == "pan_zoom":
                layer.mode = "pan_zoom"
            elif mode == "paint":
                layer.mode = "paint"
                layer.brush_size = brush_size
                layer.n_edit_dimensions = 3 # Enable 3D spherical brush
            elif mode == "erase":
                layer.mode = "erase"
                layer.brush_size = brush_size
                layer.n_edit_dimensions = 3 # Enable 3D spherical eraser
            elif mode == "fill":
                layer.mode = "fill"
        else:
            print(f"Layer {name} not found in viewer.")

    def set_camera_view(self, axis: int):
        """
        Sets the camera to look along a specific axis.
        0: Axial (Z)
        1: Coronal (Y)
        2: Sagittal (X)
        """
        self.viewer.dims.ndisplay = 2
        
        if axis == 0: # Axial (Z-axis is slice)
            self.viewer.dims.order = (0, 1, 2)
        elif axis == 1: # Coronal (Y-axis is slice)
            self.viewer.dims.order = (1, 0, 2) 
        elif axis == 2: # Sagittal (X-axis is slice)
            self.viewer.dims.order = (2, 0, 1)
        
        self.viewer.reset_view()

    def set_3d_view(self):
        self.viewer.dims.ndisplay = 3
        self.is_3d = True
        # Also ensure existing layers are locked
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.editable = False

    def eventFilter(self, source, event: QEvent):
        """Intercept raw Qt events on the Napari canvas before Vispy handles them."""
        # Check if we have the canvas widget and it's a Wheel event
        if hasattr(self, 'qt_viewer') and hasattr(self.qt_viewer, 'canvas'):
            if hasattr(self.qt_viewer.canvas, 'native') and source == self.qt_viewer.canvas.native:
                if event.type() == QEvent.Type.Wheel:
                    from PyQt6.QtCore import Qt
                    
                    # Positive angle_delta is forward/up, negative is back/down
                    delta = event.angleDelta().y() 
                    if delta == 0:
                        return False

                    # Check for Ctrl modifier
                    modifiers = event.modifiers()
                    is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

                    if is_ctrl:
                        # Zoom behavior
                        zoom_factor = 1.1 if delta > 0 else 0.9
                        self.viewer.camera.zoom *= zoom_factor
                    else:
                        # Scroll behavior
                        dims_displayed = self.viewer.dims.displayed
                        all_dims = list(range(self.viewer.dims.ndim))
                        slice_dims = [d for d in all_dims if d not in dims_displayed]
                        
                        if slice_dims:
                            dim = slice_dims[0]
                            current_step = list(self.viewer.dims.current_step)
                            step_size = -1 if delta > 0 else 1 
                            
                            new_step = current_step[dim] + step_size
                            limit = self.viewer.dims.range[dim][1]
                            new_step = max(0, min(int(limit) - 1, new_step))
                            
                            current_step[dim] = new_step
                            self.viewer.dims.current_step = tuple(current_step)
                            
                    return True
                    
        return super().eventFilter(source, event)

    # ──── AutoPET Click Markers ────
    CLICK_LAYER_NAME = "Click Markers"
    
    def load_click_markers(self, data_zyx: np.ndarray):
        """Load/update the click markers labels layer (shared array)."""
        name = self.CLICK_LAYER_NAME
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.data is not data_zyx:
                layer.data = data_zyx
            else:
                layer.refresh()
        else:
            # color_map: 0=transparent, 1=red(tumor), 2=blue(background)
            kwargs = dict(name=name, opacity=0.5)
            if self._scale_zyx is not None:
                kwargs['scale'] = self._scale_zyx
            layer = self.viewer.add_labels(data_zyx, **kwargs)
            layer.color = {1: 'red', 2: 'dodgerblue'}
            layer.editable = False
    
    def remove_click_markers(self):
        """Remove the click markers layer."""
        if self.CLICK_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(self.CLICK_LAYER_NAME)
        
    # ──── Lesion ID Labels ────
    LESION_LABEL_LAYER_NAME = "Lesion IDs"

    def show_lesion_ids(self, points: list, lesion_ids: list):
        """Show lesion ID labels as points at each lesion's centroid.

        Args:
            points:     list of pre-calculated [z, y, x] coordinates in Napari space.
            lesion_ids: list of str IDs
        """
        self.hide_lesion_ids()

        if not points:
            return

        points = np.array(points)

        kwargs = dict(
            size=0,  # Hide the point marker (circle)
            face_color='transparent',
            border_color='transparent',  
            name=self.LESION_LABEL_LAYER_NAME,
            features={'lesion_id': lesion_ids},
            text={
                'string': '{lesion_id}',
                'size': 16,
                'color': 'lime',
                'anchor': 'upper_left',
                'translation': [0, 5, 0], # Offset text slightly
            },
            n_dimensional=False, # Show only on exact slice
        )
        if self._scale_zyx is not None:
            kwargs['scale'] = self._scale_zyx

        layer = self.viewer.add_points(points, **kwargs)
        layer.editable = False

    def hide_lesion_ids(self):
        """Remove the lesion ID labels layer."""
        if self.LESION_LABEL_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(self.LESION_LABEL_LAYER_NAME)

    # ──── Sphere / Square Drag Mode ────

    SHAPE_PREVIEW_LAYER = "_shape_preview"

    def enable_shape_drag(self, layer_type: str, shape: str):
        """Enable sphere or square drag mode.

        Args:
            layer_type: 'tumor' or 'organ' — target label layer to paint into.
            shape: 'sphere' or 'square'.
        """
        self.disable_shape_drag()  # Clean up any previous state

        self._shape_mode = shape              # 'sphere' | 'square'
        self._shape_target_layer = layer_type
        self._shape_dragging = False

        self._ensure_shape_preview_layer(layer_type)
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
        
        # Only set active if it's not already active to avoid potential slice jumps
        if self.viewer.layers.selection.active != shapes_layer:
            self.viewer.layers.selection.active = shapes_layer
            
        shapes_layer.mode = "add_ellipse" if shape == "sphere" else "add_rectangle"
        
        # Connect event to switch to select mode after drawing
        if not getattr(self, '_shape_added_connected', False):
            shapes_layer.events.data.connect(self._on_shape_added)
            self._shape_added_connected = True

        # Set mask layer to pan_zoom first so Napari doesn't intercept
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name in self.viewer.layers:
            self.viewer.layers[name].mode = "pan_zoom"

        # Camera mouse pan/zoom should be disabled during ROI drawing to avoid jitter
        self.viewer.camera.mouse_pan = False
        self.viewer.camera.mouse_zoom = False

    def disable_shape_drag(self):
        """Clean up shape drag mode and hide preview."""
        if hasattr(self, '_shape_mode'):
            del self._shape_mode
        if hasattr(self, '_shape_target_layer'):
            del self._shape_target_layer
        self._shape_dragging = False

        # Re-enable camera mouse interactions
        self.viewer.camera.mouse_pan = True
        self.viewer.camera.mouse_zoom = True

        # Hide preview layer instead of removing to preserve it if needed
        if self.SHAPE_PREVIEW_LAYER in self.viewer.layers:
            # BUG-9 FIX: Clear old shapes when switching tools so they don't reappear later
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].data = []
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].visible = False
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].editable = False

    def _ensure_shape_preview_layer(self, layer_type="tumor"):
        """Create or update the shared shapes layer metadata to match the target layer."""
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name not in self.viewer.layers:
            return
        target_layer = self.viewer.layers[name]

        if self.SHAPE_PREVIEW_LAYER not in self.viewer.layers:
            kwargs = dict(
                name=self.SHAPE_PREVIEW_LAYER,
                edge_color='lime',
                face_color='transparent',
                edge_width=2,
                opacity=0.8,
                ndim=target_layer.ndim,
                scale=target_layer.scale,
                translate=target_layer.translate,
                rotate=target_layer.rotate,
                shear=target_layer.shear,
                affine=target_layer.affine,
            )
            self.viewer.add_shapes([], **kwargs)
        else:
            # Sync metadata just in case
            layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
            layer.scale = target_layer.scale
            layer.translate = target_layer.translate
            layer.affine = target_layer.affine

        self.viewer.layers[self.SHAPE_PREVIEW_LAYER].visible = True
        self.viewer.layers[self.SHAPE_PREVIEW_LAYER].editable = True

    def _on_shape_added(self, event):
        """Switch to select mode and AUTO-SELECT the new shape so it can be moved immediately."""
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
        if shapes_layer.mode in ["add_ellipse", "add_rectangle"]:
            shapes_layer.mode = "select"
            # Auto-select the last added shape (the one we just drew)
            if len(shapes_layer.data) > 0:
                shapes_layer.selected_data = {len(shapes_layer.data) - 1}

    def commit_shape_to_mask(self):
        """Paint ALL shapes from the preview layer into the target mask layer using manual 3D volume logic."""
        if not hasattr(self, '_shape_target_layer') or self.SHAPE_PREVIEW_LAYER not in self.viewer.layers:
            return

        name = self.LAYER_NAMES.get(self._shape_target_layer, self._shape_target_layer)
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]

        if len(shapes_layer.data) == 0:
            return

        data = layer.data  # (Z, Y, X)
        dims_displayed = list(self.viewer.dims.displayed)
        slice_dim = [d for d in range(3) if d not in dims_displayed][0]

        committed_count = 0
        for i, shape_pts in enumerate(shapes_layer.data):
            # shape_pts is (N, 3) in LAYER coordinates.
            # Since we synced metadata between Shapes and Mask layer, 
            # these vertices are ALREADY in data coordinate units.
            data_pts = np.array(shape_pts)
            
            # Determine if it's an ellipse or rectangle via shape_type
            stype = shapes_layer.shape_type[i]
            
            # Bounding box in data coordinates
            d_min = np.min(data_pts, axis=0)
            d_max = np.max(data_pts, axis=0)
            center = (d_min + d_max) / 2
            
            if stype == 'ellipse':
                # Map to sphere logic
                dy = d_max[dims_displayed[0]] - d_min[dims_displayed[0]]
                dx = d_max[dims_displayed[1]] - d_min[dims_displayed[1]]
                radius_voxels = max(dy, dx) / 2
                self._paint_sphere_logic(data, center, radius_voxels, dims_displayed)
            else:
                # Map to box logic
                self._paint_box_logic(data, d_min, d_max, dims_displayed, slice_dim)
            committed_count += 1

        layer.data = data
        layer.refresh()
        # BUG-4 FIX: Removed manual explicit `layer.events.data(value=data)` because `layer.data = data` 
        # already automatically triggers it. Prevents double-sync debounce triggers.

        # Clear preview
        shapes_layer.data = []
        print(f"[ViewerWidget] Committed {committed_count} shapes to {name} via robust 3D logic.")

    def _paint_sphere_logic(self, data, center, radius, dims_displayed):
        r_int = int(np.ceil(radius))
        shape = data.shape
        z_min = max(0, int(center[0] - r_int))
        z_max = min(shape[0], int(center[0] + r_int + 1))
        y_min = max(0, int(center[1] - r_int))
        y_max = min(shape[1], int(center[1] + r_int + 1))
        x_min = max(0, int(center[2] - r_int))
        x_max = min(shape[2], int(center[2] + r_int + 1))

        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist_sq = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2
        sphere_mask = dist_sq <= radius**2
        data[z_min:z_max, y_min:y_max, x_min:x_max][sphere_mask] = 1

    def _paint_box_logic(self, data, d_min, d_max, dims_displayed, slice_dim):
        h = d_max[dims_displayed[0]] - d_min[dims_displayed[0]]
        w = d_max[dims_displayed[1]] - d_min[dims_displayed[1]]
        depth = max(1, int(min(h, w) / 2))
        
        slice_pos = int((d_min[slice_dim] + d_max[slice_dim]) / 2)
        s_min = max(0, slice_pos - depth)
        s_max = min(data.shape[slice_dim], slice_pos + depth + 1)

        slices = [slice(None)] * 3
        slices[slice_dim] = slice(s_min, s_max)
        slices[dims_displayed[0]] = slice(int(max(0, d_min[dims_displayed[0]])), int(min(data.shape[dims_displayed[0]], d_max[dims_displayed[0]] + 1)))
        slices[dims_displayed[1]] = slice(int(max(0, d_min[dims_displayed[1]])), int(min(data.shape[dims_displayed[1]], d_max[dims_displayed[1]] + 1)))

        data[tuple(slices)] = 1

    def _apply_shape(self, layer, start_world, end_world):
        """Deprecated: replaced by interactive Napari Shapes + commit_shape_to_mask."""
        pass

    def close(self):
        self.viewer.close()
