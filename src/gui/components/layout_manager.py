from PyQt6.QtWidgets import QWidget, QGridLayout, QStackedWidget, QVBoxLayout, QLabel
from .viewer_widget import ViewerWidget

class LayoutManager(QWidget):
    """
    Manages switching between Grid, Overlay, and 3D layouts.
    Synchronizes viewers.
    """
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
        
        # Setup Synchronization
        self._sync_grid_views()
        self._sync_mono_views()
        # For now, we assume we push data to all viewers when loaded.
        
    def _init_grid_view(self):
        """
        6-Cell Grid:
        Row 0: Axial CT, Axial PET
        Row 1: Sagittal CT, Sagittal PET
        Row 2: Coronal CT, Coronal PET
        """
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        
        self.grid_viewers = {} # key: (row, col)
        
        # Rows: 0=Axial, 1=Sagittal, 2=Coronal
        # Cols: 0=CT, 1=PET
        for r in range(3):
            for c in range(2):
                v = ViewerWidget()
                self.grid_layout.addWidget(v, r, c)
                self.grid_viewers[(r, c)] = v
                
                # Set orientation
                # ViewerWidget takes 0=Axial, 1=Coronal, 2=Sagittal
                # But our rows are Axial, Sagittal, Coronal (as per user request "Orthogonal Grid View... Axial, Sagittal, Coronal")
                if r == 0: v.set_camera_view(0) # Axial
                if r == 1: v.set_camera_view(2) # Sagittal 
                if r == 2: v.set_camera_view(1) # Coronal
        
        self.stack.addWidget(self.grid_widget)
        
    def _init_overlay_view(self):
        self.overlay_widget = QWidget()
        layout = QVBoxLayout(self.overlay_widget)
        self.overlay_viewer = ViewerWidget()
        self.overlay_viewer.set_camera_view(0) # Default Axial
        layout.addWidget(self.overlay_viewer)
        self.stack.addWidget(self.overlay_widget)

    def _init_mono_view(self):
        """
        Side-by-side view: CT (Left), PET (Right).
        """
        self.mono_widget = QWidget()
        self.mono_layout = QGridLayout(self.mono_widget)
        self.mono_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mono_viewers = {}
        
        # 0: CT, 1: PET
        for i in range(2):
            v = ViewerWidget()
            self.mono_layout.addWidget(v, 0, i)
            self.mono_viewers[i] = v
            v.set_camera_view(0) # Default Axial
            
        self.stack.addWidget(self.mono_widget)
        
    def _init_3d_view(self):
        self.view_3d_widget = QWidget()
        layout = QVBoxLayout(self.view_3d_widget)
        self.viewer_3d = ViewerWidget()
        self.viewer_3d.set_3d_view()
        layout.addWidget(self.viewer_3d)
        self.stack.addWidget(self.view_3d_widget)

    def _sync_grid_views(self):
        """
        Sync cursors/slices per row.
        Sync camera (zoom/pan) across all.
        """
        # Sync Slices per Row
        for r in range(3):
            v1 = self.grid_viewers[(r, 0)].viewer
            v2 = self.grid_viewers[(r, 1)].viewer
            
            self._link_dims(v1, v2)
            self._link_camera(v1, v2)

    def _link_dims(self, v1, v2):
        """Link the current step of two viewers."""
        def sync_v1_to_v2(event):
            current_step = v1.dims.current_step
            if v2.dims.current_step != current_step:
                v2.dims.current_step = current_step
                
        def sync_v2_to_v1(event):
            current_step = v2.dims.current_step
            if v1.dims.current_step != current_step:
                v1.dims.current_step = current_step

        v1.dims.events.current_step.connect(sync_v1_to_v2)
        v2.dims.events.current_step.connect(sync_v2_to_v1)

    def _link_camera(self, v1, v2):
        """Link the camera of two viewers with loop protection."""
        
        def sync_cam_v1_to_v2(event):
            # Check zoom
            if abs(v2.camera.zoom - v1.camera.zoom) > 1e-6:
                 v2.camera.zoom = v1.camera.zoom
                 
            # Check center (tuple/array)
            c1 = v1.camera.center
            c2 = v2.camera.center
            if any(abs(a - b) > 1e-6 for a, b in zip(c1, c2)):
                v2.camera.center = c1
            
        def sync_cam_v2_to_v1(event):
             # Check zoom
            if abs(v1.camera.zoom - v2.camera.zoom) > 1e-6:
                 v1.camera.zoom = v2.camera.zoom
                 
            # Check center
            c1 = v1.camera.center
            c2 = v2.camera.center
            if any(abs(a - b) > 1e-6 for a, b in zip(c1, c2)):
                v1.camera.center = c2
        
        v1.camera.events.zoom.connect(sync_cam_v1_to_v2)
        v1.camera.events.center.connect(sync_cam_v1_to_v2)
        
        v2.camera.events.zoom.connect(sync_cam_v2_to_v1)
        v2.camera.events.center.connect(sync_cam_v2_to_v1)

    def load_data(self, ct_data, pet_data, affine, tumor_mask=None, organ_mask=None):
        """
        Push data to ALL viewers. Only loads if data is not None.
        """
        # Grid Viewers
        for (r, c), widget in self.grid_viewers.items():
            if c == 0:
                if ct_data is not None:
                    widget.load_image(ct_data, affine, "ct", "gray")
            else:
                if pet_data is not None:
                    widget.load_image(pet_data, affine, "pet", "jet")
            
            # Load Masks
            if tumor_mask is not None:
                widget.load_mask(tumor_mask, "tumor")
            if organ_mask is not None:
                widget.load_mask(organ_mask, "organ")

            # Only reset view if the primary image for this widget was loaded?
            # Or just reset always? Resetting helps center the image.
            # But if we just loaded CT, resetting PET viewer (which might be empty) is fine.
            # However, if PET is empty, reset_view might do nothing or error.
            # Let's trust Napari.
            
            # Re-apply camera view to ensure it's not reset by add_image
            # Row 0=Axial(0), Row 1=Sagittal(2), Row 2=Coronal(1)
            if r == 0: widget.set_camera_view(0)
            elif r == 1: widget.set_camera_view(2)
            elif r == 2: widget.set_camera_view(1)
            
            widget.viewer.reset_view()
                
        # Overlay Viewer
        if ct_data is not None:
            self.overlay_viewer.load_image(ct_data, affine, "ct", "gray")
        if pet_data is not None:
            self.overlay_viewer.load_image(pet_data, affine, "pet", "jet", opacity=0.5)
        
        if tumor_mask is not None:
            self.overlay_viewer.load_mask(tumor_mask, "tumor")
        if organ_mask is not None:
            self.overlay_viewer.load_mask(organ_mask, "organ")

        self.overlay_viewer.viewer.reset_view()
        
        # 3D Viewer
        if ct_data is not None:
            self.viewer_3d.load_image(ct_data, affine, "ct", "gray")
        if pet_data is not None:
            self.viewer_3d.load_image(pet_data, affine, "pet", "jet", opacity=0.7) 
        
        if tumor_mask is not None:
            self.viewer_3d.load_mask(tumor_mask, "tumor")
        if organ_mask is not None:
            self.viewer_3d.load_mask(organ_mask, "organ")

        self.viewer_3d.viewer.dims.ndisplay = 3

        # Mono Viewers
        if ct_data is not None:
            self.mono_viewers[0].load_image(ct_data, affine, "ct", "gray")
        if pet_data is not None:
            self.mono_viewers[1].load_image(pet_data, affine, "pet", "jet") # No opacity needed on separate view
            
        for v in self.mono_viewers.values():
            if tumor_mask is not None:
                v.load_mask(tumor_mask, "tumor")
            if organ_mask is not None:
                v.load_mask(organ_mask, "organ")
            v.viewer.reset_view()
        
    def _sync_grid_views(self):
        """
        Sync cursors/slices per row.
        Sync camera (zoom/pan) across ALL 2D viewers.
        """
        # 1. Sync Slices per Row (dims)
        for r in range(3):
            v1 = self.grid_viewers[(r, 0)].viewer
            v2 = self.grid_viewers[(r, 1)].viewer
            self._link_dims(v1, v2)

        # 2. Sync Camera across ALL 6 viewers
        # We pick the first viewer as "master" for connection logic, or link all to each other.
        # Linking all to all is O(N^2) connections. 
        # Better: create a chain (0 -> 1 -> 2 ... -> 0) or star (0 -> all).
        # Let's try chaining (0,0) -> (0,1) -> (1,0) -> (1,1) ...
        
        viewers = [param.viewer for param in self.grid_viewers.values()]
        # Also include overlay viewer? Yes.
        viewers.append(self.overlay_viewer.viewer)
        
        for i in range(len(viewers) - 1):
             self._link_camera(viewers[i], viewers[i+1])
             
        # Also close the loop for robustness? Or just 0->All.
        # Given manual event linking, we need to be careful.
        # Let's just stick to "Reference Viewer" approach if possible, but Napari doesn't have a central cam.
        # The chain approach works if A->B->C... 

    def _sync_mono_views(self):
        v1 = self.mono_viewers[0].viewer
        v2 = self.mono_viewers[1].viewer
        self._link_dims(v1, v2)
        self._link_camera(v1, v2)

    def set_pet_opacity(self, value: float):
        """Update opacity for 'pet' layer in all viewers."""
        # Grid
        for widget in self.grid_viewers.values():
            for layer in widget.viewer.layers:
                if layer.name == widget.LAYER_NAMES["pet"]:
                    layer.opacity = value
                    
        # Overlay
        for layer in self.overlay_viewer.viewer.layers:
            if layer.name == self.overlay_viewer.LAYER_NAMES["pet"]:
                layer.opacity = value
                
        # 3D
        for layer in self.viewer_3d.viewer.layers:
            if layer.name == self.viewer_3d.LAYER_NAMES["pet"]:
                layer.opacity = value

    def set_ct_contrast(self, min_val: float, max_val: float):
        """Update contrast limits for 'ct' layer."""
        self._set_contrast_limits("ct", min_val, max_val)

    def set_pet_contrast(self, min_val: float, max_val: float):
        """Update contrast limits for 'pet' layer."""
        self._set_contrast_limits("pet", min_val, max_val)

    def set_ct_window_level(self, window: float, level: float):
        # min = level - window/2, max = level + window/2
        min_val = level - (window / 2)
        max_val = level + (window / 2)
        self.set_ct_contrast(min_val, max_val)

    def set_pet_window_level(self, window: float, level: float):
        min_val = level - (window / 2)
        max_val = level + (window / 2)
        # Ensure min is not negative for PET if desired, but let Napari handle range
        if min_val < 0: min_val = 0
        self.set_pet_contrast(min_val, max_val)

    def _set_contrast_limits(self, layer_type: str, min_val: float, max_val: float):
        # Helper to update all viewers
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        viewers.append(self.viewer_3d)
        for v in self.mono_viewers.values(): viewers.append(v)
        
        name = viewers[0].LAYER_NAMES.get(layer_type, layer_type)
        
        for widget in viewers:
            for layer in widget.viewer.layers:
                if layer.name == name:
                    layer.contrast_limits = (min_val, max_val)

    def set_zoom(self, value: float):
        # Value 0-100 mapped to zoom factor?
        # Napari zoom is raw scale. 0.1 to 10?
        # Let's say slider 0-100 maps to 0.1 - 5.0
        zoom_factor = 0.1 + (value / 100.0) * 4.9
        
        # Update ONE viewer per independent group and let sync handle it?
        # Or update all?
        # Grid: Update (0,0)
        self.grid_viewers[(0,0)].viewer.camera.zoom = zoom_factor
        # Overlay
        self.overlay_viewer.viewer.camera.zoom = zoom_factor
        # Mono
        self.mono_viewers[0].viewer.camera.zoom = zoom_factor
        # 3D usually handles its own zoom via orbit
        
    def toggle_mask(self, mask_type: str, visible: bool):
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        viewers.append(self.viewer_3d)
        for v in self.mono_viewers.values(): viewers.append(v)

        name_map = {"tumor": "Tumor Mask", "organ": "Organ Mask", "body": "Organ Mask"}
        target_name = name_map.get(mask_type, mask_type)
        
        for widget in viewers:
            for layer in widget.viewer.layers:
                if layer.name == target_name:
                    layer.visible = visible
                
    def set_view_mode(self, mode: str):
        if mode == "grid":
            self.stack.setCurrentWidget(self.grid_widget)
            
        elif mode.startswith("mono"):
            self.stack.setCurrentWidget(self.mono_widget)
            # Handle orientation
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
            # Handle orientation
            if "axial" in mode:
                self.overlay_viewer.set_camera_view(0)
            elif "coronal" in mode:
                self.overlay_viewer.set_camera_view(1)
            elif "sagittal" in mode:
                self.overlay_viewer.set_camera_view(2)
            else:
                self.overlay_viewer.set_camera_view(0) # Default
                
        elif mode == "mono":
            self.stack.setCurrentWidget(self.mono_widget)
            
        elif mode == "3d":
            self.stack.setCurrentWidget(self.view_3d_widget)
            
    def toggle_3d_pet(self, visible: bool):
        """Show/Hide PET layer in 3D view."""
        for layer in self.viewer_3d.viewer.layers:
             if layer.name == self.viewer_3d.LAYER_NAMES["pet"]:
                 layer.visible = visible
