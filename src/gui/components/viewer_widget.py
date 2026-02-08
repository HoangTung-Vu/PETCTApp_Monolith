from PyQt6.QtWidgets import QWidget, QVBoxLayout
import napari
import numpy as np
from typing import Optional, Tuple
import nibabel as nib

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
        
        # Keep track of layer names
        self.LAYER_NAMES = {
            "ct": "CT Image",
            "pet": "PET Image",
            "tumor": "Tumor Mask",
            "organ": "Organ Mask"
        }
        
    def load_image(self, image_data: np.ndarray, affine: np.ndarray, 
                   layer_type: str, colormap: str = "gray", 
                   blending: str = "translucent",
                   opacity: float = 1.0):
        """
        Loads an image into the viewer.
        image_data: (X, Y, Z) array from nibabel
        affine: (4, 4) affine matrix
        """
        # Transpose/Rotate for Napari (ZYX)
        # Nibabel (X, Y, Z) -> Napari usually expects (Z, Y, X)
        # But we also need to respect the affine. 
        # For simplicity in this specific app logic as requested:
        # "Warnings: NAPARI USE NUMPY WITH AXES (Z,Y,X)"
        
        # Transpose to (Z, Y, X)
        data_zyx = np.transpose(image_data, (2, 1, 0))
        
        # FIX: The views are currently inverted vertically. 
        # Flip Z (axis 0) and Y (axis 1) to correct this.
        # Axial (Y-X plane): Flipping Y flips it vertically.
        # Coronal (Z-X plane): Flipping Z flips it vertically.
        # Sagittal (Z-Y plane): Flipping Z flips it vertically. 
        # Note: Flipping Y also flips Sagittal horizontally, effectively rotating it 180 degrees if both are flipped.
        # This seems to be the desired correction for "all 3 views inverted".
        data_zyx = np.flip(data_zyx, axis=(0, 1))
        
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data_zyx
        else:
            self.viewer.add_image(
                data_zyx,
                name=name,
                colormap=colormap,
                blending=blending,
                opacity=opacity,
                # scale=... # Helper might be needed for spacing
            )
            
    def load_mask(self, mask_data: np.ndarray, layer_type: str, color: Optional[int] = None):
        """
        Loads a label mask.
        """
        # Ensure integer type for Labels layer
        mask_data = mask_data.astype(np.uint8)
        
        data_zyx = np.transpose(mask_data, (2, 1, 0))
        
        # Apply the same flip as the image to keep alignment
        data_zyx = np.flip(data_zyx, axis=(0, 1))
        
        name = self.LAYER_NAMES.get(layer_type, layer_type)
        
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data_zyx
        else:
            self.viewer.add_labels(
                data_zyx,
                name=name,
                opacity=0.7
            )
            
    def set_camera_view(self, axis: int):
        """
        Sets the camera to look along a specific axis.
        0: Axial (Z)
        1: Coronal (Y)
        2: Sagittal (X)
        """
        # In Napari (Z, Y, X) -> (0, 1, 2)
        # This sets the dimension that is SLICED, i.e. the viewing plane normal.
        self.viewer.dims.ndisplay = 2
        # Order the dimensions so 'axis' is the first one (the slider)
        # Default is (0, 1, 2) -> Z is slider (Axial view)
        
        if axis == 0: # Axial (Z-axis is slice)
            self.viewer.dims.order = (0, 1, 2)
        elif axis == 1: # Coronal (Y-axis is slice)
            self.viewer.dims.order = (1, 0, 2) 
        elif axis == 2: # Sagittal (X-axis is slice)
            self.viewer.dims.order = (2, 0, 1)
        
        self.viewer.reset_view()

    def set_3d_view(self):
        self.viewer.dims.ndisplay = 3
        
    def close(self):
        self.viewer.close()
