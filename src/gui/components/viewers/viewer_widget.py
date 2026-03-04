from PyQt6.QtWidgets import QWidget, QVBoxLayout
import napari
import numpy as np
from typing import Optional, Tuple
import nibabel as nib

from ....utils.nifti_utils import to_napari, from_napari

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

        # Disable default double click zoom
        for cb in list(self.viewer.mouse_double_click_callbacks):
            if cb.__name__ == 'double_click_to_zoom':
                self.viewer.mouse_double_click_callbacks.remove(cb)
        
        # Keep track of layer names
        self.LAYER_NAMES = {
            "ct": "CT Image",
            "pet": "PET Image",
            "tumor": "Tumor Mask",
            "organ": "Organ Mask"
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
        spacing_xyz = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # (sx, sy, sz)
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

    def show_lesion_ids(self, bboxes: list, lesion_ids: list):
        """Show lesion ID labels as points at each lesion's centroid.

        The bboxes come from the report engine in nibabel array order
        (dim0, dim1, dim2). ``to_napari`` converts (X,Y,Z) → (Z,Y,X) and
        then flips Z (axis 0) and Y (axis 1).  We replicate the same
        transform so the Points align with image data.

        Args:
            bboxes:     list of (d0_min, d1_min, d2_min, d0_max, d1_max, d2_max)
            lesion_ids: list of int IDs
        """
        self.hide_lesion_ids()

        if not bboxes:
            return

        # Get the nibabel-space shape so we can mirror the flip
        # Shape in nibabel order (X, Y, Z)
        nib_shape = None
        for name in ("CT Image", "PET Image"):
            if name in self.viewer.layers:
                zyx_shape = self.viewer.layers[name].data.shape
                # to_napari produced (Z', Y', X') from (X, Y, Z)
                # So nibabel shape = (X, Y, Z) = (zyx[2], zyx[1], zyx[0])
                nib_shape = (zyx_shape[2], zyx_shape[1], zyx_shape[0])
                break

        if nib_shape is None:
            return

        centroids = []
        id_strings = []

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

            centroids.append([z_nap, y_nap, x_nap])
            id_strings.append(str(lid))

        points = np.array(centroids)

        kwargs = dict(
            size=0,  # Hide the point marker (circle)
            face_color='transparent',
            border_color='transparent',  
            name=self.LESION_LABEL_LAYER_NAME,
            features={'lesion_id': id_strings},
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

    def close(self):
        self.viewer.close()
