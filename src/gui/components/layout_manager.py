from PyQt6.QtWidgets import QWidget, QGridLayout, QStackedWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal
import numpy as np
from .viewer_widget import ViewerWidget

class LayoutManager(QWidget):
    """
    Manages switching between Grid, Overlay, and 3D layouts.
    Synchronizes viewers.
    """

    # Signal emitted when user clicks in autopet mode: (coord_zyx_list, label)
    sig_autopet_click_added = pyqtSignal(list, str)

    # Signal emitted when eraser removes a connected component: (mask_array_xyz)
    sig_eraser_region_removed = pyqtSignal(object)


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
        self._sync_all_layouts()
        
        # Data Caching for Lazy Loading
        self._cached_data = {
            "ct": None, "pet": None, "affine": None,
            "tumor": None, "organ": None
        }
        # Centralized Napari-space data for sharing across viewers
        self._cached_data_zyx = {
            "tumor": None,
            "organ": None
        }
        self._is_3d_loaded = False
        
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
        Push data to 2D viewers. Cache for 3D viewer (lazy load).
        """
        # Cache data
        self._cached_data["ct"] = ct_data
        self._cached_data["pet"] = pet_data
        self._cached_data["affine"] = affine
        self._cached_data["tumor"] = tumor_mask
        self._cached_data["organ"] = organ_mask
        self._is_3d_loaded = False # Reset 3D loaded state

        # 0. Convert masks to Napari space ONCE for sharing
        from ...utils.nifti_utils import to_napari
        if tumor_mask is not None:
            self._cached_data_zyx["tumor"] = to_napari(tumor_mask.astype(np.uint8))
        if organ_mask is not None:
            self._cached_data_zyx["organ"] = to_napari(organ_mask.astype(np.uint8))

        # 1. Grid Viewers
        for (r, c), widget in self.grid_viewers.items():
            if c == 0:
                if ct_data is not None:
                    widget.load_image(ct_data, affine, "ct", "gray")
            else:
                if pet_data is not None:
                    widget.load_image(pet_data, affine, "pet", "jet")
            
            # Load Masks using shared Napari-space data
            if self._cached_data_zyx["tumor"] is not None:
                widget.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
            if self._cached_data_zyx["organ"] is not None:
                widget.load_mask_zyx(self._cached_data_zyx["organ"], "organ")
            
            # Re-apply camera view and reset
            if r == 0: widget.set_camera_view(0)
            elif r == 1: widget.set_camera_view(2)
            elif r == 2: widget.set_camera_view(1)
            
            widget.viewer.reset_view()
                
        # 2. Overlay Viewer
        if ct_data is not None:
            self.overlay_viewer.load_image(ct_data, affine, "ct", "gray")
        if pet_data is not None:
            self.overlay_viewer.load_image(pet_data, affine, "pet", "jet", opacity=0.5)
        
        if self._cached_data_zyx["tumor"] is not None:
            self.overlay_viewer.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
        if self._cached_data_zyx["organ"] is not None:
            self.overlay_viewer.load_mask_zyx(self._cached_data_zyx["organ"], "organ")

        self.overlay_viewer.viewer.reset_view()
        
        # 3. Mono Viewers
        if ct_data is not None:
            self.mono_viewers[0].load_image(ct_data, affine, "ct", "gray")
        if pet_data is not None:
            self.mono_viewers[1].load_image(pet_data, affine, "pet", "jet")
            
        for v in self.mono_viewers.values():
            if self._cached_data_zyx["tumor"] is not None:
                v.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
            if self._cached_data_zyx["organ"] is not None:
                v.load_mask_zyx(self._cached_data_zyx["organ"], "organ")
            v.viewer.reset_view()

        # 4. Connect Event Listeners for Synchronization
        self._connect_all_mask_events()

    def _load_3d_data(self):
        """Lazy load data into 3D viewer."""
        if self._is_3d_loaded:
            return
            
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]
        tumor = self._cached_data["tumor"]
        organ = self._cached_data["organ"]
        
        if ct is not None:
            self.viewer_3d.load_image(ct, affine, "ct", "gray")
        if pet is not None:
            self.viewer_3d.load_image(pet, affine, "pet", "jet", opacity=0.7) 
        
        if self._cached_data_zyx["tumor"] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
        if self._cached_data_zyx["organ"] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["organ"], "organ")

        self.viewer_3d.viewer.dims.ndisplay = 3

        
        self._is_3d_loaded = True

    def reset_zoom(self):
        """Reset view for currently active viewers."""
        # We can just reset all 2D viewers, it's cheap.
        
        # Grid
        for v in self.grid_viewers.values():
            v.viewer.reset_view()
            
        # Overlay
        self.overlay_viewer.viewer.reset_view()
        
        # Mono
        for v in self.mono_viewers.values():
            v.viewer.reset_view()
            
        print("Zoom reset for all 2D viewers.")
        
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

    def _sync_all_layouts(self):
        """
        Links all 2D viewers across different layouts to share the same 3D crosshair position.
        This provides 'Robust Sync': clicking on a point in Axial moves Coronal/Sagittal slices.
        """
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v.viewer)
        viewers.append(self.overlay_viewer.viewer)
        for v in self.mono_viewers.values(): viewers.append(v.viewer)
        
        # We create a chain of links for current_step (Z, Y, X)
        for i in range(len(viewers) - 1):
            self._link_dims_full(viewers[i], viewers[i+1])
            
    def _link_dims_full(self, v1, v2):
        """Links the full 3D position (current_step) between two viewers."""
        def sync_v1_to_v2(event):
            step1 = v1.dims.current_step
            if v2.dims.current_step != step1:
                v2.dims.current_step = step1
                
        def sync_v2_to_v1(event):
            step2 = v2.dims.current_step
            if v1.dims.current_step != step2:
                v1.dims.current_step = step2

        v1.dims.events.current_step.connect(sync_v1_to_v2)
        v2.dims.events.current_step.connect(sync_v2_to_v1)

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
                

            
        elif mode == "3d":
            self.stack.setCurrentWidget(self.view_3d_widget)
            self._load_3d_data()
            
    def update_mask(self, mask_data, mask_type):
        """
        Update mask in all active viewers and cache it.
        Respects lazy loading for 3D viewer.
        """
        # Update Cache
        self._cached_data[mask_type] = mask_data
        
        from ...utils.nifti_utils import to_napari
        data_zyx = to_napari(mask_data.astype(np.uint8))
        self._cached_data_zyx[mask_type] = data_zyx
        
        # Helper to push to a viewer list
        def push_to_viewers(viewers, m_zyx, m_type):
            for v in viewers:
                v.load_mask_zyx(m_zyx, m_type)

        # 0.5. Disconnect events to prevent recursion
        self._disconnect_all_mask_events()

        # 1. Grid
        push_to_viewers(self.grid_viewers.values(), data_zyx, mask_type)
        
        # 2. Overlay
        self.overlay_viewer.load_mask_zyx(data_zyx, mask_type)
        
        # 3. Mono
        push_to_viewers(self.mono_viewers.values(), data_zyx, mask_type)
        
        # 4. 3D (Lazy)
        if self._is_3d_loaded:
            self.viewer_3d.load_mask_zyx(data_zyx, mask_type)
            
        # Re-connect events in case layers were recreated (load_mask_zyx handles updates but safer)
        self._connect_all_mask_events()

    def _connect_all_mask_events(self):
        """Connect all mask layers across all viewers to a centralized sync handler."""
        self._disconnect_all_mask_events() # Safety first

        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        for v in self.mono_viewers.values(): viewers.append(v)
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)

        for v in viewers:
            for layer_name in ["Tumor Mask", "Organ Mask"]:
                if layer_name in v.viewer.layers:
                    layer = v.viewer.layers[layer_name]
                    layer.events.data.connect(self._on_mask_data_changed)

    def _disconnect_all_mask_events(self):
        """Disconnect all mask layers across all viewers."""
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        for v in self.mono_viewers.values(): viewers.append(v)
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)

        for v in viewers:
            for layer_name in ["Tumor Mask", "Organ Mask"]:
                if layer_name in v.viewer.layers:
                    layer = v.viewer.layers[layer_name]
                    try:
                        layer.events.data.disconnect(self._on_mask_data_changed)
                    except:
                        pass


    def _on_mask_data_changed(self, event):
        """Called when any mask layer's data is modified (e.g. painted/erased)."""
        trigger_layer = event.source
        layer_name = trigger_layer.name
        
        # Refresh all OTHER viewers that share this data
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        for v in self.mono_viewers.values(): viewers.append(v)
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)

        for v in viewers:
            if layer_name in v.viewer.layers:
                layer = v.viewer.layers[layer_name]
                if layer is not trigger_layer:
                    # If sharing the same array, refresh() is enough to redraw
                    # Napari might not always trigger a visible update if only a few pixels changed
                    # but refresh() usually works.
                    layer.refresh()
                    


    def toggle_3d_pet(self, visible: bool):
        """Show/Hide PET layer in 3D view."""
        # Update cache if needed? No, just view toggle.
        for layer in self.viewer_3d.viewer.layers:
             if layer.name == self.viewer_3d.LAYER_NAMES["pet"]:
                 layer.visible = visible
    def set_drawing_tool(self, tool: str, brush_size: int, layer_type: str):
        """
        Sets the drawing tool for all 2D viewers.
        """
        # Iterate all 2D viewers
        viewers = []
        for v in self.grid_viewers.values(): viewers.append(v)
        viewers.append(self.overlay_viewer)
        for v in self.mono_viewers.values(): viewers.append(v)
        
        # 3D viewer usually disabled for drawing


        for v in viewers:
            v.set_drawing_mode(layer_type, tool, brush_size)
            
    def get_active_mask_data(self, layer_type: str):
        """
        Retrieves the mask data from the currently active/visible viewer.
        Returns Nibabel-space (X, Y, Z) array.
        """
        current_widget = self.stack.currentWidget()
        
        # List of all potential source viewers
        potential_viewers = []
        
        # 1. Prioritize current active viewer
        if current_widget == self.overlay_widget:
            potential_viewers.append(self.overlay_viewer)
        elif current_widget == self.mono_widget:
            potential_viewers.extend(self.mono_viewers.values())
        elif current_widget == self.grid_widget:
            # Prioritize current cell if we can find it, but for simplicity just add all grid viewers
            potential_viewers.extend(self.grid_viewers.values())
        
        # 2. If current is 3D or no data found, check ALL other 2D viewers
        # Since they share the same array, any one of them will work.
        all_2d = [self.overlay_viewer] + list(self.mono_viewers.values()) + list(self.grid_viewers.values())
        for v in all_2d:
            if v not in potential_viewers:
                potential_viewers.append(v)
        
        # 3. Try to get data from the first viewer that has it
        for viewer in potential_viewers:
            data = viewer.get_layer_data(layer_type)
            if data is not None:
                return data
                
        # Fallback to cache if viewer doesn't have it or something failed
        return self._cached_data.get(layer_type)

    # ──── AutoPET Interactive Click Handling ────
    
    def _get_all_2d_viewers(self):
        """Collect all 2D viewer widgets."""
        viewers = list(self.grid_viewers.values())
        viewers.append(self.overlay_viewer)
        viewers.extend(self.mono_viewers.values())
        return viewers
    
    def enable_autopet_click_mode(self, label: str):
        """Install mouse callback on all 2D viewers.
        label: 'tumor' or 'background'
        """
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
    
    def _make_click_callback(self):
        """Create a mouse callback that captures click coordinates."""
        def on_double_click(viewer, event):
            label = getattr(self, '_autopet_click_label', None)
            if label is None:
                return
            coord_zyx = [round(c) for c in event.position]
            print(f"[AutoPET] Click: {label} at ZYX={coord_zyx}")
            
            # Paint sphere into shared array
            self._paint_click_sphere(coord_zyx, label)
            
            # Emit signal for main_window to track
            self.sig_autopet_click_added.emit(coord_zyx, label)
        
        return on_double_click
    
    def _ensure_click_array(self):
        """Lazily create the shared click markers array matching image shape."""
        if not hasattr(self, '_click_markers') or self._click_markers is None:
            # Get shape from cached data
            shape = None
            for key in ("ct", "pet"):
                d = self._cached_data.get(key)
                if d is not None:
                    from ...utils.nifti_utils import to_napari
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
        
        # Compute sphere bounds (clipped to array)
        z0, z1 = max(0, z - radius), min(shape[0], z + radius + 1)
        y0, y1 = max(0, y - radius), min(shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(shape[2], x + radius + 1)
        
        # Create sphere mask within the bounding box
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist_sq = (zz - z)**2 + (yy - y)**2 + (xx - x)**2
        arr[z0:z1, y0:y1, x0:x1][dist_sq <= radius**2] = val
        
        # Push to all viewers (same array ref → just refresh)
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

    # ──── Eraser Tool (Connected Component) ────

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

    def _make_eraser_callback(self):
        """Create a mouse callback that erases the connected component at click."""
        def on_click(viewer, event):
            coord_zyx = [round(c) for c in event.position]
            z, y, x = coord_zyx
            print(f"[Eraser] Click at ZYX={coord_zyx}")

            # Get tumor mask data from cache (Napari ZYX space)
            mask_zyx = self._cached_data_zyx.get("tumor")
            if mask_zyx is None:
                print("[Eraser] No tumor mask loaded.")
                return

            # Bounds check
            if not (0 <= z < mask_zyx.shape[0] and
                    0 <= y < mask_zyx.shape[1] and
                    0 <= x < mask_zyx.shape[2]):
                print("[Eraser] Click out of bounds.")
                return

            if mask_zyx[z, y, x] == 0:
                print("[Eraser] Clicked on background (label=0), nothing to erase.")
                return

            # Connected component analysis
            from scipy.ndimage import label as nd_label
            labeled, num_features = nd_label(mask_zyx)
            component_id = labeled[z, y, x]
            num_voxels = int(np.sum(labeled == component_id))

            # Erase the component
            mask_zyx[labeled == component_id] = 0
            print(f"[Eraser] Removed component #{component_id} ({num_voxels} voxels), "
                  f"{num_features - 1} components remaining.")

            # Refresh all viewers (they share the same array reference)
            for v in self._get_all_2d_viewers():
                tumor_name = v.LAYER_NAMES.get("tumor", "Tumor Mask")
                if tumor_name in v.viewer.layers:
                    v.viewer.layers[tumor_name].refresh()

            # Also update 3D viewer if loaded
            if self._is_3d_loaded:
                tumor_name = self.viewer_3d.LAYER_NAMES.get("tumor", "Tumor Mask")
                if tumor_name in self.viewer_3d.viewer.layers:
                    self.viewer_3d.viewer.layers[tumor_name].refresh()

            # Convert back to Nibabel space (X, Y, Z) and emit signal
            from ...utils.nifti_utils import from_napari
            mask_xyz = from_napari(mask_zyx)
            self.sig_eraser_region_removed.emit(mask_xyz)

        return on_click
