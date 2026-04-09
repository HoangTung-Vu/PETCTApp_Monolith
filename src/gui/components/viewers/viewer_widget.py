from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QObject, QEvent, Qt, pyqtSignal
import napari
import numpy as np
from typing import Optional, Tuple

from ....utils.nifti_utils import to_napari, from_napari
from ....utils.dimension_utils import get_spacing_from_affine
from .crosshair_overlay import CrosshairOverlay


class ViewerWidget(QWidget):
    """
    A unified widget containing a single Napari viewer.
    Handles data loading and basic display settings.
    """
    # Emitted when crosshair mode is active and mouse moves over the canvas
    sig_cursor_intensity = pyqtSignal(str)

    # Emitted when user left-clicks to move crosshair: (z, y, x) in data space
    sig_crosshair_clicked = pyqtSignal(list)

    # Emitted when slice scroll occurs: current_step tuple
    sig_slice_changed = pyqtSignal(tuple)

    # Emitted when arrow key pressed while crosshair is active: (dim_index, delta)
    sig_crosshair_arrow = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Initialize Napari Viewer
        self.viewer = napari.Viewer(show=False)
        self.qt_viewer = self.viewer.window.qt_viewer
        self.layout.addWidget(self.qt_viewer)

        self.viewer.camera.mouse_zoom = False

        # Install event filter on canvas for wheel interception
        if hasattr(self.qt_viewer, 'canvas') and hasattr(self.qt_viewer.canvas, 'native'):
            self.qt_viewer.canvas.native.installEventFilter(self)

        # Disable default double-click zoom
        for cb in list(self.viewer.mouse_double_click_callbacks):
            if cb.__name__ == 'double_click_to_zoom':
                self.viewer.mouse_double_click_callbacks.remove(cb)

        # Layer name map
        self.LAYER_NAMES = {
            "ct": "CT Image",
            "pet": "PET Image",
            "tumor": "Tumor Mask",
            "roi": "ROI Mask",
        }
        self.is_3d = False
        self._scale_zyx = None   # physical spacing (sz, sy, sx) in mm

        # ── Crosshair overlay ───────────────────────────────────────────
        self._crosshair_enabled = False   # managed by LayoutManager
        self._crosshair_overlay = None    # created after canvas is ready
        self._crosshair_overlay = CrosshairOverlay(self, self.qt_viewer.canvas.native)

        # Re-apply cursor and update overlay when camera or slice changes
        self.viewer.camera.events.zoom.connect(self._on_view_changed)
        self.viewer.camera.events.center.connect(self._on_view_changed)
        self.viewer.dims.events.current_step.connect(self._on_view_changed)
        self.viewer.dims.events.current_step.connect(self._on_slice_changed)

        # Mouse callbacks
        self._xhair_move_cb = self._make_xhair_move_cb()
        self.viewer.mouse_move_callbacks.append(self._xhair_move_cb)

        self._xhair_press_cb = self._make_xhair_press_cb()
        self.viewer.mouse_drag_callbacks.append(self._xhair_press_cb)

        self._pan_drag_cb = self._make_pan_drag_cb()
        self.viewer.mouse_drag_callbacks.append(self._pan_drag_cb)

        # Default: no pan-on-drag (panning handled manually via RMB), arrow cursor always
        self.viewer.camera.mouse_pan = False
        self._set_cursor_cross(False)

    # ── Static helpers ──────────────────────────────────────────────────

    @staticmethod
    def to_napari(data: np.ndarray) -> np.ndarray:
        return to_napari(data)

    @staticmethod
    def from_napari(data_zyx: np.ndarray) -> np.ndarray:
        return from_napari(data_zyx)

    # ── Data loading ────────────────────────────────────────────────────

    def load_image(self, image_data: np.ndarray, affine: np.ndarray,
                   layer_type: str, colormap: str = "gray",
                   blending: str = "translucent",
                   opacity: float = 1.0):
        """Load an image into the viewer. image_data: (X, Y, Z) from nibabel."""
        data_zyx = self.to_napari(image_data)

        spacing_xyz = get_spacing_from_affine(affine)
        self._scale_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

        name = self.LAYER_NAMES.get(layer_type, layer_type)

        # In 2D viewers, coplanar layers (same depth) cause depth-test failures on
        # Windows OpenGL — the second image/mask layer gets hidden behind the first.
        # "translucent_no_depth" disables depth testing so layers always composite
        # correctly. For 3D, keep the caller-supplied blending for correct occlusion.
        effective_blending = blending if self.is_3d else "translucent_no_depth"

        if name in self.viewer.layers:
            self.viewer.layers[name].data = data_zyx
            self.viewer.layers[name].scale = self._scale_zyx
            self.viewer.layers[name].colormap = colormap
            self.viewer.layers[name].blending = effective_blending
            self.viewer.layers[name].opacity = opacity
        else:
            self.viewer.add_image(
                data_zyx,
                name=name,
                scale=self._scale_zyx,
                colormap=colormap,
                blending=effective_blending,
                opacity=opacity,
                interpolation2d='nearest',
            )

    def load_mask(self, mask_data: np.ndarray, layer_type: str, color: Optional[int] = None):
        mask_data = mask_data.astype(np.uint8)
        data_zyx = self.to_napari(mask_data)
        self.load_mask_zyx(data_zyx, layer_type)

    def load_mask_zyx(self, data_zyx: np.ndarray, layer_type: str):
        """Load a label mask already in Napari space (Z, Y, X)."""
        name = self.LAYER_NAMES.get(layer_type, layer_type)

        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.data is not data_zyx:
                layer.data = data_zyx
            else:
                layer.refresh()
        else:
            kwargs = dict(name=name, opacity=0.7)
            if layer_type == "roi":
                kwargs["opacity"] = 0.9
            if self._scale_zyx is not None:
                kwargs['scale'] = self._scale_zyx
            layer = self.viewer.add_labels(data_zyx, **kwargs)

        # Always enforce yellow colormap for ROI layer
        if layer_type == "roi":
            self._apply_roi_colormap(layer)

        # In 2D viewers with multiple stacked image layers (overlay, mono_single),
        # the depth buffer written by CT/PET layers causes the mask to fail depth
        # testing on Windows OpenGL. Disabling depth testing on the Labels layer
        # ensures it always renders on top of coplanar image layers.
        if not self.is_3d:
            layer.blending = "translucent_no_depth"

        if self.is_3d:
            layer.editable = False

    # Shared across all ViewerWidget instances — created once
    _roi_colormap = None

    @staticmethod
    def _apply_roi_colormap(layer):
        """Set a yellow-only colormap on a Labels layer for ROI visualization."""
        if ViewerWidget._roi_colormap is None:
            from napari.utils.colormaps import DirectLabelColormap
            ViewerWidget._roi_colormap = DirectLabelColormap(
                color_dict={0: "transparent", 1: "yellow", None: "transparent"}
            )
        layer.colormap = ViewerWidget._roi_colormap

    def get_layer_data(self, layer_type: str) -> Optional[np.ndarray]:
        """Returns layer data in Nibabel (X, Y, Z) format."""
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name in self.viewer.layers:
            data_zyx = self.viewer.layers[name].data
            return self.from_napari(data_zyx)
        return None

    # ── Drawing mode ────────────────────────────────────────────────────

    def set_drawing_mode(self, layer_type: str, mode: str, brush_size: float = 10):
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            self.viewer.layers.selection.active = layer
            if mode == "pan_zoom":
                layer.mode = "pan_zoom"
            elif mode == "paint":
                layer.mode = "paint"
                layer.brush_size = brush_size
                layer.n_edit_dimensions = 3
            elif mode == "erase":
                layer.mode = "erase"
                layer.brush_size = brush_size
                layer.n_edit_dimensions = 3
            elif mode == "fill":
                layer.mode = "fill"

    # ── Camera view ─────────────────────────────────────────────────────

    def set_camera_view(self, axis: int):
        """axis: 0=Axial, 1=Coronal, 2=Sagittal."""
        self.viewer.dims.ndisplay = 2
        if axis == 0:
            self.viewer.dims.order = (0, 1, 2)
        elif axis == 1:
            self.viewer.dims.order = (1, 0, 2)
        elif axis == 2:
            self.viewer.dims.order = (2, 0, 1)
        self.viewer.reset_view()

    def set_3d_view(self):
        self.viewer.dims.ndisplay = 3
        self.is_3d = True
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.editable = False

    # ── Event filter (wheel + canvas resize) ────────────────────────────

    def eventFilter(self, source, event: QEvent):
        if hasattr(self, 'qt_viewer') and hasattr(self.qt_viewer, 'canvas'):
            canvas_native = getattr(self.qt_viewer.canvas, 'native', None)
            if canvas_native and source == canvas_native:
                # Suppress RMB context menu — RMB is used for panning
                if event.type() == QEvent.Type.ContextMenu:
                    return True

                # Arrow keys move crosshair when crosshair overlay is active
                if event.type() == QEvent.Type.KeyPress and self._crosshair_enabled:
                    key = event.key()
                    dims_d = list(self.viewer.dims.displayed)
                    if len(dims_d) >= 2:
                        if key == Qt.Key.Key_Left:
                            self.sig_crosshair_arrow.emit(dims_d[1], -1)
                            return True
                        elif key == Qt.Key.Key_Right:
                            self.sig_crosshair_arrow.emit(dims_d[1], +1)
                            return True
                        elif key == Qt.Key.Key_Up:
                            self.sig_crosshair_arrow.emit(dims_d[0], -1)
                            return True
                        elif key == Qt.Key.Key_Down:
                            self.sig_crosshair_arrow.emit(dims_d[0], +1)
                            return True

                if event.type() == QEvent.Type.Wheel:
                    delta = event.angleDelta().y()
                    if delta == 0:
                        return False
                    modifiers = event.modifiers()
                    is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

                    if is_ctrl:
                        zoom_factor = 1.1 if delta > 0 else 0.9
                        self.viewer.camera.zoom *= zoom_factor
                    else:
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

    # ── Crosshair cursor ────────────────────────────────────────────────

    def _set_cursor_cross(self, enabled: bool):
        """Set the canvas cursor to CrossCursor or ArrowCursor."""
        if hasattr(self.qt_viewer, 'canvas') and hasattr(self.qt_viewer.canvas, 'native'):
            from PyQt6.QtGui import QCursor
            from PyQt6.QtCore import Qt
            shape = Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
            self.qt_viewer.canvas.native.setCursor(QCursor(shape))

    def enable_crosshair_mode(self):
        """Show crosshair overlay. Arrow cursor always."""
        self._crosshair_enabled = True
        self._crosshair_overlay.set_enabled(True)
        self.viewer.camera.mouse_pan = False

        # Initialize crosshair to the center of the current view
        try:
            c_world = self.viewer.camera.center
            layer = None
            for name in list(self.LAYER_NAMES.values()):
                if name in self.viewer.layers:
                    layer = self.viewer.layers[name]
                    break
            if layer is not None:
                data_zyx = layer.world_to_data(c_world)
                pos = [float(data_zyx[0]), float(data_zyx[1]), float(data_zyx[2])]
                
                # Replace the non-displayed dimension with the current scroll step
                dims_d = list(self.viewer.dims.displayed)
                for d in range(layer.ndim):
                    if d not in dims_d:
                        pos[d] = float(self.viewer.dims.current_step[d])
                        
                # Emit clicked so LayoutManager syncs it up
                self.sig_crosshair_clicked.emit(pos)
        except Exception:
            pass

    def disable_crosshair_mode(self):
        """Hide crosshair overlay. Arrow cursor always."""
        self._crosshair_enabled = False
        self._crosshair_overlay.set_enabled(False)

    def set_pan_mode(self, pan_on: bool):
        """Pan mode: disable crosshair overlay, let camera pan on drag."""
        self._crosshair_enabled = False
        self._crosshair_overlay.set_enabled(False)
        self.viewer.camera.mouse_pan = pan_on

    def _on_view_changed(self, event=None):
        """Camera or slice changed — refresh overlay."""
        if self._crosshair_enabled:
            self._crosshair_overlay.update()

    def _on_slice_changed(self, event=None):
        """Emit step tuple when slice scrolling happens."""
        self.sig_slice_changed.emit(tuple(self.viewer.dims.current_step))

    # ── Crosshair canvas coordinate computation ─────────────────────────

    def _compute_canvas_pos(self, data_pos_zyx):
        """Convert ZYX data-space position to canvas pixel (px, py).

        Formula derived from Vispy PanZoomCamera:
          visible_width  = canvas_w / zoom
          canvas_px = (world_x - center_x) * zoom + canvas_w/2

        Returns (px, py) or (None, None) on failure.
        """
        if self._scale_zyx is None or data_pos_zyx is None:
            return None, None
        try:
            dims_d = list(self.viewer.dims.displayed)  # e.g. [1,2] axial
            if len(dims_d) < 2:
                return None, None

            canvas_native = self.qt_viewer.canvas.native
            cw = canvas_native.width()
            ch = canvas_native.height()
            if cw == 0 or ch == 0:
                return None, None

            scale = self._scale_zyx
            # World coordinates for the two displayed data dimensions
            w_vert  = float(data_pos_zyx[dims_d[0]]) * float(scale[dims_d[0]])
            w_horiz = float(data_pos_zyx[dims_d[1]]) * float(scale[dims_d[1]])

            center = self.viewer.camera.center   # 3-tuple in world coords, ordered by dims.order!
            c_vert  = float(center[-2])
            c_horiz = float(center[-1])

            zoom = float(self.viewer.camera.zoom)

            canvas_px = (w_horiz - c_horiz) * zoom + cw / 2.0
            canvas_py = (w_vert  - c_vert)  * zoom + ch / 2.0

            return canvas_px, canvas_py
        except Exception:
            return None, None

    def update_crosshair(self, data_pos_zyx):
        """Push a new crosshair position to the overlay."""
        self._crosshair_overlay.set_pos_data(data_pos_zyx)

    # ── Mouse callbacks ─────────────────────────────────────────────────

    def _make_xhair_move_cb(self):
        """Return a mouse-move callback that reads CT/PET intensity at cursor."""
        def on_move(viewer, event):
            parts = []
            for layer_type, label, fmt in [
                ("ct",  "CT",  "{:.0f} HU"),
                ("pet", "PET", "{:.2f} SUV"),
            ]:
                name = self.LAYER_NAMES.get(layer_type)
                if not name or name not in viewer.layers:
                    continue
                layer = viewer.layers[name]
                if not layer.visible:
                    continue
                try:
                    data_pos = layer.world_to_data(event.position)
                    zi = int(round(float(data_pos[0])))
                    yi = int(round(float(data_pos[1])))
                    xi = int(round(float(data_pos[2])))
                    d = layer.data
                    if 0 <= zi < d.shape[0] and 0 <= yi < d.shape[1] and 0 <= xi < d.shape[2]:
                        val = float(d[zi, yi, xi])
                        parts.append(f"{label}: {fmt.format(val)}")
                except Exception:
                    pass
            self.sig_cursor_intensity.emit("  |  ".join(parts))
        return on_move

    def _make_xhair_press_cb(self):
        """Return a drag callback (generator) that moves crosshair on left-click/drag."""
        def on_drag(viewer, event):
            if not self._crosshair_enabled:
                return
            if event.button != 1:   # left button only
                return
            ref_layer = None
            for lt in ("ct", "pet"):
                name = self.LAYER_NAMES.get(lt)
                if name and name in viewer.layers:
                    ref_layer = viewer.layers[name]
                    break
            if ref_layer is None:
                return

            def _emit_pos():
                try:
                    data_pos = ref_layer.world_to_data(event.position)
                    pos = [float(data_pos[0]), float(data_pos[1]), float(data_pos[2])]
                    shape = ref_layer.data.shape
                    pos[0] = max(0, min(shape[0] - 1, pos[0]))
                    pos[1] = max(0, min(shape[1] - 1, pos[1]))
                    pos[2] = max(0, min(shape[2] - 1, pos[2]))
                    self.sig_crosshair_clicked.emit(pos)
                except Exception:
                    pass

            _emit_pos()   # initial click
            yield         # wait for drag moves
            while event.type == 'mouse_move':
                _emit_pos()
                yield

        return on_drag

    def _make_pan_drag_cb(self):
        """Return a drag callback that pans the camera on right-click drag."""
        def on_drag(viewer, event):
            if event.button != 2:   # right mouse button only
                return
            last_canvas = np.array(event.pos[:2], dtype=float)
            yield
            while event.type == 'mouse_move':
                new_canvas = np.array(event.pos[:2], dtype=float)
                delta = new_canvas - last_canvas
                zoom = viewer.camera.zoom
                if zoom > 0:
                    center = list(viewer.camera.center)
                    # Camera center[-1] = horizontal world axis, center[-2] = vertical
                    center[-1] -= delta[0] / zoom
                    center[-2] -= delta[1] / zoom
                    viewer.camera.center = tuple(center)
                last_canvas = new_canvas
                yield
        return on_drag

    # ── AutoPET click markers ────────────────────────────────────────────

    CLICK_LAYER_NAME = "Click Markers"

    def load_click_markers(self, data_zyx: np.ndarray):
        name = self.CLICK_LAYER_NAME
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.data is not data_zyx:
                layer.data = data_zyx
            else:
                layer.refresh()
        else:
            kwargs = dict(name=name, opacity=0.5)
            if self._scale_zyx is not None:
                kwargs['scale'] = self._scale_zyx
            layer = self.viewer.add_labels(data_zyx, **kwargs)
            layer.color = {1: 'red', 2: 'dodgerblue'}
            layer.editable = False

    def remove_click_markers(self):
        if self.CLICK_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(self.CLICK_LAYER_NAME)

    # ── Lesion ID Labels ────────────────────────────────────────────────

    LESION_LABEL_LAYER_NAME = "Lesion IDs"

    def show_lesion_ids(self, points: list, lesion_ids: list):
        self.hide_lesion_ids()
        if not points:
            return
        points = np.array(points)
        kwargs = dict(
            size=0,
            face_color='transparent',
            border_color='transparent',
            name=self.LESION_LABEL_LAYER_NAME,
            features={'lesion_id': lesion_ids},
            text={
                'string': '{lesion_id}',
                'size': 16,
                'color': 'lime',
                'anchor': 'upper_left',
                'translation': [0, 5, 0],
            },
            n_dimensional=False,
        )
        if self._scale_zyx is not None:
            kwargs['scale'] = self._scale_zyx
        layer = self.viewer.add_points(points, **kwargs)
        layer.editable = False

    def hide_lesion_ids(self):
        if self.LESION_LABEL_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(self.LESION_LABEL_LAYER_NAME)

    # ── Sphere / Square Drag Mode ────────────────────────────────────────

    SHAPE_PREVIEW_LAYER = "_shape_preview"

    def enable_shape_drag(self, layer_type: str, shape: str):
        already_active = (
            getattr(self, '_shape_mode', None) is not None
            and getattr(self, '_shape_target_layer', None) == layer_type
            and self.SHAPE_PREVIEW_LAYER in self.viewer.layers
        )
        if already_active:
            # Fast path: only update the draw mode (sphere ↔ square switch)
            self._shape_mode = shape
            shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
            shapes_layer.mode = "add_ellipse" if shape == "sphere" else "add_rectangle"
            if self.viewer.layers.selection.active != shapes_layer:
                self.viewer.layers.selection.active = shapes_layer
            return

        # Full init path (first activation)
        self.disable_shape_drag()
        self._shape_mode = shape
        self._shape_target_layer = layer_type
        self._shape_dragging = False
        self._ensure_shape_preview_layer(layer_type)
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
        if self.viewer.layers.selection.active != shapes_layer:
            self.viewer.layers.selection.active = shapes_layer
        shapes_layer.mode = "add_ellipse" if shape == "sphere" else "add_rectangle"
        if not getattr(self, '_shape_added_connected', False):
            shapes_layer.events.data.connect(self._on_shape_added)
            self._shape_added_connected = True
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        if name in self.viewer.layers:
            self.viewer.layers[name].mode = "pan_zoom"
        self.viewer.camera.mouse_pan = False
        self.viewer.camera.mouse_zoom = False

    def disable_shape_drag(self):
        if hasattr(self, '_shape_mode'):
            del self._shape_mode
        if hasattr(self, '_shape_target_layer'):
            del self._shape_target_layer
        self._shape_dragging = False
        self.viewer.camera.mouse_pan = False   # default: no drag-pan; LayoutManager sets pan mode
        self.viewer.camera.mouse_zoom = True
        if self.SHAPE_PREVIEW_LAYER in self.viewer.layers:
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].data = []
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].visible = False
            self.viewer.layers[self.SHAPE_PREVIEW_LAYER].editable = False

    def _ensure_shape_preview_layer(self, layer_type="tumor"):
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
            layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
            layer.scale = target_layer.scale
            layer.translate = target_layer.translate
            layer.affine = target_layer.affine
        self.viewer.layers[self.SHAPE_PREVIEW_LAYER].visible = True
        self.viewer.layers[self.SHAPE_PREVIEW_LAYER].editable = True

    def _on_shape_added(self, event):
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
        if shapes_layer.mode in ["add_ellipse", "add_rectangle"]:
            shapes_layer.mode = "select"
            if len(shapes_layer.data) > 0:
                shapes_layer.selected_data = {len(shapes_layer.data) - 1}

    def commit_shape_to_mask(self):
        if not hasattr(self, '_shape_target_layer') or self.SHAPE_PREVIEW_LAYER not in self.viewer.layers:
            return
        name = self.LAYER_NAMES.get(self._shape_target_layer, self._shape_target_layer)
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        shapes_layer = self.viewer.layers[self.SHAPE_PREVIEW_LAYER]
        if len(shapes_layer.data) == 0:
            return
        data = layer.data
        dims_displayed = list(self.viewer.dims.displayed)
        slice_dim = [d for d in range(3) if d not in dims_displayed][0]
        committed_count = 0
        for i, shape_pts in enumerate(shapes_layer.data):
            data_pts = np.array(shape_pts)
            stype = shapes_layer.shape_type[i]
            d_min = np.min(data_pts, axis=0)
            d_max = np.max(data_pts, axis=0)
            center = (d_min + d_max) / 2
            if stype == 'ellipse':
                dy = d_max[dims_displayed[0]] - d_min[dims_displayed[0]]
                dx = d_max[dims_displayed[1]] - d_min[dims_displayed[1]]
                radius_voxels = max(dy, dx) / 2
                self._paint_sphere_logic(data, center, radius_voxels, dims_displayed)
            else:
                self._paint_box_logic(data, d_min, d_max, dims_displayed, slice_dim)
            committed_count += 1
        layer.data = data
        layer.refresh()
        shapes_layer.data = []
        print(f"[ViewerWidget] Committed {committed_count} shapes to {name}.")

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
        slices[dims_displayed[0]] = slice(int(max(0, d_min[dims_displayed[0]])),
                                          int(min(data.shape[dims_displayed[0]], d_max[dims_displayed[0]] + 1)))
        slices[dims_displayed[1]] = slice(int(max(0, d_min[dims_displayed[1]])),
                                          int(min(data.shape[dims_displayed[1]], d_max[dims_displayed[1]] + 1)))
        data[tuple(slices)] = 1

    # ── Interpolation ────────────────────────────────────────────────────

    def set_interpolation(self, enabled: bool):
        mode = "linear" if enabled else "nearest"
        for layer in self.viewer.layers:
            if hasattr(layer, "interpolation2d"):
                try:
                    layer.interpolation2d = mode
                except Exception:
                    pass

    # ── Cleanup ─────────────────────────────────────────────────────────

    def close(self):
        self.viewer.close()
