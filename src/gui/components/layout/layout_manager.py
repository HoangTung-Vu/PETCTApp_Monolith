"""Layout Manager — manages view switching, data loading, and display settings.

Performance optimizations:
- **Lazy loading**: Only pushes data to the CURRENTLY VISIBLE layout.
- **One-shot sync**: When switching view mode, slice position is copied once.
- **No cross-layout sync**: Removed cascading 9-viewer sync that caused
  ~30+ events per scroll.
- **Crosshair**: Global position tracked as (z, y, x) data coords; each
  overlay repaints itself using the Vispy camera transform at paint time.
"""

from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QStackedWidget, QVBoxLayout, QApplication,
    QLabel, QComboBox, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, QTimer, Qt
import numpy as np

from ..viewers.viewer_widget import ViewerWidget
from ..viewers.viewer_sync import link_dims, link_camera, one_shot_sync_step
from .mask_sync import MaskSyncMixin
from .autopet_click_manager import AutoPETClickMixin
from .eraser_manager import EraserMixin
from ....utils.dimension_utils import get_spacing_from_affine


class LayoutManager(MaskSyncMixin, AutoPETClickMixin, EraserMixin, QWidget):
    """Manages Grid, Overlay, Mono, MonoSingle, and 3D layouts."""

    sig_autopet_click_added    = pyqtSignal(list, str)
    sig_eraser_region_removed  = pyqtSignal(object, object)
    sig_eraser_background_click = pyqtSignal()
    sig_mask_painted           = pyqtSignal(str)
    sig_shape_committed        = pyqtSignal(str)
    sig_cursor_intensity       = pyqtSignal(str)

    # (z_vox, y_vox, x_vox, z_mm, y_mm, x_mm, hu, suv)
    sig_crosshair_pos = pyqtSignal(float, float, float, float, float, float, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Info overlay (top-right of viewer area)
        self._info_label = QLabel(self)
        self._info_label.setStyleSheet(
            "QLabel { color: #ffff88; background: rgba(0,0,0,170);"
            " font-size: 12px; font-family: monospace;"
            " padding: 6px 10px; border: 1px solid #555; }"
        )
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._info_label.setWordWrap(False)
        self._info_label.hide()
        self._info_label.raise_()

        # Initialize layouts
        self._init_grid_view()
        self._init_overlay_view()
        self._init_mono_view()
        self._init_mono_single_view()
        self._init_3d_view()

        # Sync within layouts
        self._sync_grid_views()
        self._sync_mono_views()

        self._init_mask_sync()

        # Data cache
        self._cached_data = {"ct": None, "pet": None, "affine": None, "tumor": None}
        self._cached_data_zyx = {"tumor": None}
        self._is_3d_loaded = False
        self._cached_lesion_data = None
        self._loaded_layouts = set()

        # Display state
        self._ct_wl = (350.0, 35.0)
        self._pet_wl = (10.0, 5.0)
        self._ct_colormap = "gray"
        self._pet_colormap = "jet"

        # Crosshair state
        self._crosshair_enabled = False
        self._pan_mode = False
        self._xhair_pos = [0.0, 0.0, 0.0]   # [z, y, x] in Napari data space

        # For cross-view scroll sync: subscribe to CT viewer dims in grid
        self._scroll_sync_connected = False

        self._preload_queue = []

    # ── View Initialization ──────────────────────────────────────────────

    def _init_grid_view(self):
        """6-Cell Grid: Axial/Sagittal/Coronal × CT/PET."""
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

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
        layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_viewer = ViewerWidget()
        self.overlay_viewer.set_camera_view(0)
        layout.addWidget(self.overlay_viewer)
        self.stack.addWidget(self.overlay_widget)

    def _init_mono_view(self):
        """Side-by-side: CT (Left), PET (Right)."""
        self.mono_widget = QWidget()
        self.mono_layout = QGridLayout(self.mono_widget)
        self.mono_layout.setContentsMargins(0, 0, 0, 0)
        self.mono_layout.setSpacing(1)
        self.mono_layout.setColumnStretch(0, 1)
        self.mono_layout.setColumnStretch(1, 1)
        self.mono_viewers = {}
        for i in range(2):
            v = ViewerWidget()
            self.mono_layout.addWidget(v, 0, i)
            self.mono_viewers[i] = v
            v.set_camera_view(0)
        self.stack.addWidget(self.mono_widget)

    def _init_mono_single_view(self):
        """2×2 grid: 3 orthogonal views of ONE modality + empty cell."""
        self.mono_single_widget = QWidget()
        ms_outer = QVBoxLayout(self.mono_single_widget)
        ms_outer.setContentsMargins(0, 0, 0, 0)
        ms_outer.setSpacing(0)

        # Modality selector bar
        sel_bar = QWidget()
        sel_bar.setFixedHeight(32)
        sel_lay = QHBoxLayout(sel_bar)
        sel_lay.setContentsMargins(4, 2, 4, 2)
        sel_lay.addWidget(QLabel("Modality:"))
        self._mono_single_combo = QComboBox()
        self._mono_single_combo.addItems(["CT", "PET"])
        self._mono_single_combo.currentTextChanged.connect(self._reload_mono_single)
        sel_lay.addWidget(self._mono_single_combo)
        sel_lay.addStretch()
        ms_outer.addWidget(sel_bar)

        # 2×2 grid
        ms_grid_widget = QWidget()
        ms_grid = QGridLayout(ms_grid_widget)
        ms_grid.setContentsMargins(0, 0, 0, 0)
        ms_grid.setSpacing(1)
        ms_grid.setColumnStretch(0, 1)
        ms_grid.setColumnStretch(1, 1)
        ms_grid.setRowStretch(0, 1)
        ms_grid.setRowStretch(1, 1)

        self.mono_single_viewers = {}
        # (0,0)=Axial, (0,1)=Coronal, (1,0)=Sagittal, (1,1)=empty
        axes = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
        for (r, c), axis in axes.items():
            v = ViewerWidget()
            ms_grid.addWidget(v, r, c)
            self.mono_single_viewers[(r, c)] = v
            v.set_camera_view(axis)

        # Empty placeholder cell
        empty = QLabel()
        empty.setStyleSheet("background: #111;")
        ms_grid.addWidget(empty, 1, 1)
        ms_outer.addWidget(ms_grid_widget)

        self.stack.addWidget(self.mono_single_widget)

    def _init_3d_view(self):
        self.view_3d_widget = QWidget()
        layout = QVBoxLayout(self.view_3d_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_3d = ViewerWidget()
        self.viewer_3d.set_3d_view()
        layout.addWidget(self.viewer_3d)
        self.stack.addWidget(self.view_3d_widget)

    # ── Intra-layout Sync ────────────────────────────────────────────────

    def _sync_grid_views(self):
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

    # ── Crosshair event handling ─────────────────────────────────────────

    def _connect_crosshair_events(self):
        """Connect click, scroll, and arrow-key signals from all visible viewers."""
        for vw in self._get_all_2d_viewers():
            try:
                vw.sig_crosshair_clicked.disconnect(self._on_viewer_crosshair_click)
            except (TypeError, RuntimeError):
                pass
            try:
                vw.sig_slice_changed.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                vw.sig_crosshair_arrow.disconnect(self._on_crosshair_arrow)
            except (TypeError, RuntimeError):
                pass

        for vw in self._get_visible_viewers():
            if not vw.is_3d:
                vw.sig_crosshair_clicked.connect(self._on_viewer_crosshair_click)
                vw.sig_slice_changed.connect(
                    lambda step, v=vw: self._on_viewer_slice_changed(step, v)
                )
                vw.sig_crosshair_arrow.connect(self._on_crosshair_arrow)

    def _on_viewer_slice_changed(self, current_step: tuple, viewer_widget_ref):
        """Update _xhair_pos based on the non-displayed dimensions of the viewer."""
        if not self._crosshair_enabled:
            return
            
        dims_d = list(viewer_widget_ref.viewer.dims.displayed)
        changed = False
        for d in range(viewer_widget_ref.viewer.dims.ndim):
            if d not in dims_d:
                new_val = float(current_step[d])
                if self._xhair_pos[d] != new_val:
                    self._xhair_pos[d] = new_val
                    changed = True
        
        if changed:
            self._sync_viewer_slices()
            self._emit_crosshair_coords()
            self._refresh_all_crosshairs()

    def _on_viewer_crosshair_click(self, pos_zyx: list):
        """Update crosshair position from a viewer click and sync all views."""
        self._xhair_pos = [pos_zyx[0], pos_zyx[1], pos_zyx[2]]
        self._sync_viewer_slices()
        self._refresh_all_crosshairs()

    def _on_crosshair_arrow(self, dim: int, delta: int):
        """Move crosshair by one voxel along dim when an arrow key is pressed."""
        if not self._crosshair_enabled:
            return
        # Clamp within volume bounds if available
        new_val = self._xhair_pos[dim] + delta
        # Try to get the max extent for this dimension from any loaded viewer
        for vw in self._get_visible_viewers():
            if not vw.is_3d and vw.viewer.dims.ndim > dim:
                limit = vw.viewer.dims.range[dim][1]
                new_val = max(0.0, min(float(limit) - 1, new_val))
                break
        self._xhair_pos[dim] = new_val
        self._sync_viewer_slices()
        self._refresh_all_crosshairs()
        self._emit_crosshair_coords()
        self._emit_crosshair_coords()

    def _sync_viewer_slices(self):
        """Update the slice index of all visible 2D viewers to match _xhair_pos."""
        if getattr(self, '_is_syncing_slices', False):
            return
            
        self._is_syncing_slices = True
        try:
            for vw in self._get_visible_viewers():
                if vw.is_3d:
                    continue
                
                step = list(vw.viewer.dims.current_step)
                dims_d = list(vw.viewer.dims.displayed)
                changed = False
                
                for d in range(vw.viewer.dims.ndim):
                    if d not in dims_d:
                        target_slice = int(round(self._xhair_pos[d]))
                        # Limit is the max range of the dimension
                        limit = vw.viewer.dims.range[d][1]
                        target_slice = max(0, min(int(limit) - 1, target_slice))
                        if step[d] != target_slice:
                            step[d] = target_slice
                            changed = True
                
                if changed:
                    vw.viewer.dims.current_step = tuple(step)
        except Exception:
            pass
        finally:
            self._is_syncing_slices = False

    def _refresh_all_crosshairs(self):
        """Push new crosshair position to all loaded 2D viewer overlays."""
        pos = list(self._xhair_pos)
        for vw in self._get_all_2d_viewers():
            vw.update_crosshair(pos)

    def _emit_crosshair_coords(self):
        """Emit crosshair coordinates for the info overlay."""
        z, y, x = self._xhair_pos
        affine = self._cached_data.get("affine")

        ct_data = self._cached_data.get("ct")
        pet_data = self._cached_data.get("pet")
        ref_data = ct_data if ct_data is not None else pet_data

        if affine is not None and ref_data is not None:
            # Transform from Napari (Z, Y, X) back to Nibabel (X, Y, Z)
            shape_xyz = ref_data.shape
            shape_z, shape_y, shape_x = shape_xyz[2], shape_xyz[1], shape_xyz[0]

            # Undo Napari flip on Z and Y axes
            z_prime = shape_z - 1 - z
            y_prime = shape_y - 1 - y
            x_prime = x

            # Apply affine mathematically
            vec = np.array([x_prime, y_prime, z_prime, 1.0])
            mm_coords = affine @ vec
            x_mm, y_mm, z_mm = float(mm_coords[0]), float(mm_coords[1]), float(mm_coords[2])
        else:
            z_mm = y_mm = x_mm = 0.0

        # Read HU and SUV at crosshair position — iterate ALL visible viewers
        hu_str = "---"
        suv_str = "---"
        zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
        for vw in self._get_visible_viewers():
            if vw.is_3d:
                continue
            for lt, label in [("ct", "CT"), ("pet", "PET")]:
                name = vw.LAYER_NAMES.get(lt)
                if name and name in vw.viewer.layers:
                    d = vw.viewer.layers[name].data
                    if 0 <= zi < d.shape[0] and 0 <= yi < d.shape[1] and 0 <= xi < d.shape[2]:
                        val = float(d[zi, yi, xi])
                        if label == "CT" and hu_str == "---":
                            hu_str = f"{val:.0f}"
                        elif label == "PET" and suv_str == "---":
                            suv_str = f"{val:.2f}"
            if hu_str != "---" and suv_str != "---":
                break  # Got both values, no need to check more viewers

        # In mono-single mode only one modality is shown
        if self.stack.currentWidget() == self.mono_single_widget:
            modality = self._mono_single_combo.currentText().lower()
            if modality == "ct":
                suv_str = "N/A"
            else:
                hu_str = "N/A"

        self.sig_crosshair_pos.emit(z, y, x, z_mm, y_mm, x_mm, hu_str, suv_str)
        self._update_info_label(z, y, x, z_mm, y_mm, x_mm, hu_str, suv_str)

    def _update_info_label(self, z, y, x, z_mm, y_mm, x_mm, hu_str, suv_str):
        text = (
            f"Z: {int(z):4d}  Y: {int(y):4d}  X: {int(x):4d}  [vox]\n"
            f"Z: {z_mm:6.1f}  Y: {y_mm:6.1f}  X: {x_mm:6.1f}  [mm]\n"
            f"HU: {hu_str}    SUV: {suv_str}"
        )
        self._info_label.setText(text)
        self._info_label.adjustSize()
        # Position: top-right corner
        margin = 8
        lw = self._info_label.width()
        self._info_label.move(self.width() - lw - margin, margin)
        if self._crosshair_enabled:
            self._info_label.show()
            self._info_label.raise_()

    # ── Layout resize: keep info label in top-right ──────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._info_label.isVisible():
            margin = 8
            lw = self._info_label.width()
            self._info_label.move(self.width() - lw - margin, margin)

    # ── Data Loading (Lazy) ──────────────────────────────────────────────

    def load_data(self, ct_data, pet_data, affine, tumor_mask=None):
        self._cached_data["ct"] = ct_data
        self._cached_data["pet"] = pet_data
        self._cached_data["affine"] = affine
        self._cached_data["tumor"] = tumor_mask
        self._is_3d_loaded = False
        self._loaded_layouts.clear()

        from ....utils.nifti_utils import to_napari
        if tumor_mask is not None:
            self._cached_data_zyx["tumor"] = to_napari(tumor_mask.astype(np.uint8))
        else:
            self._cached_data_zyx["tumor"] = None

        # Init crosshair at center of data
        if ct_data is not None:
            sh = ct_data.shape  # XYZ nibabel
            # Napari ZYX: Z=sh[2], Y=sh[1], X=sh[0]
            self._xhair_pos = [sh[2] / 2.0, sh[1] / 2.0, sh[0] / 2.0]
        elif pet_data is not None:
            sh = pet_data.shape
            self._xhair_pos = [sh[2] / 2.0, sh[1] / 2.0, sh[0] / 2.0]

        # Cache scale for coordinate display
        if affine is not None:
            from ....utils.dimension_utils import get_spacing_from_affine
            sxyz = get_spacing_from_affine(affine)
            self._scale_zyx = (float(sxyz[2]), float(sxyz[1]), float(sxyz[0]))

        self._load_current_layout()
        self._connect_mask_events()
        self._schedule_preload()

    def _load_current_layout(self):
        current = self.stack.currentWidget()
        if current == self.grid_widget:
            self._load_grid()
        elif current == self.overlay_widget:
            self._load_overlay()
        elif current == self.mono_widget:
            self._load_mono()
        elif current == self.mono_single_widget:
            self._load_mono_single()
        elif current == self.view_3d_widget:
            self._load_3d_data()

    def _load_grid(self):
        if "grid" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        for (r, c), widget in self.grid_viewers.items():
            if c == 0 and ct is not None:
                widget.load_image(ct, affine, "ct", self._ct_colormap)
            elif c == 1 and pet is not None:
                widget.load_image(pet, affine, "pet", self._pet_colormap)
            if self._cached_data_zyx["tumor"] is not None:
                widget.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
            if r == 0: widget.set_camera_view(0)
            elif r == 1: widget.set_camera_view(2)
            elif r == 2: widget.set_camera_view(1)

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

            if len(widget.viewer.layers) > 0:
                widget.viewer.reset_view()
            if self._cached_lesion_data:
                widget.show_lesion_ids(*self._cached_lesion_data)

        QApplication.processEvents()

        if ct is not None:
            D_x, D_y, D_z = ct.shape
            spacing_xyz = get_spacing_from_affine(affine)
            sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
            self.grid_layout.setRowStretch(0, int(D_y * sy))
            self.grid_layout.setRowStretch(1, int(D_z * sz))
            self.grid_layout.setRowStretch(2, int(D_z * sz))

        self._loaded_layouts.add("grid")

    def _load_overlay(self):
        if "overlay" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.overlay_viewer.load_image(ct, affine, "ct", self._ct_colormap)
        if pet is not None:
            self.overlay_viewer.load_image(pet, affine, "pet", self._pet_colormap, opacity=0.5)
        if self._cached_data_zyx["tumor"] is not None:
            self.overlay_viewer.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")

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

        if len(self.overlay_viewer.viewer.layers) > 0:
            self.overlay_viewer.viewer.reset_view()
        if self._cached_lesion_data:
            self.overlay_viewer.show_lesion_ids(*self._cached_lesion_data)
        self._loaded_layouts.add("overlay")

    def _load_mono(self):
        if "mono" in self._loaded_layouts:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.mono_viewers[0].load_image(ct, affine, "ct", self._ct_colormap)
        if pet is not None:
            self.mono_viewers[1].load_image(pet, affine, "pet", self._pet_colormap)

        for v in self.mono_viewers.values():
            if self._cached_data_zyx["tumor"] is not None:
                v.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
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
            if len(v.viewer.layers) > 0:
                v.viewer.reset_view()
            if self._cached_lesion_data:
                v.show_lesion_ids(*self._cached_lesion_data)

        self._loaded_layouts.add("mono")

    def _load_mono_single(self):
        if "mono_single" in self._loaded_layouts:
            return
        modality = self._mono_single_combo.currentText().lower()  # 'ct' or 'pet'
        data = self._cached_data[modality]
        affine = self._cached_data["affine"]
        if data is None or affine is None:
            return

        colormap = self._ct_colormap if modality == "ct" else self._pet_colormap
        wl = self._ct_wl if modality == "ct" else self._pet_wl
        # (0,0)=Axial axis=0, (0,1)=Coronal axis=1, (1,0)=Sagittal axis=2
        axes = {(0, 0): 0, (0, 1): 1, (1, 0): 2}

        for (r, c), vw in self.mono_single_viewers.items():
            vw.load_image(data, affine, modality, colormap)
            if self._cached_data_zyx["tumor"] is not None:
                vw.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
                # Ensure tumor mask is on top — new image layers are appended above
                # existing layers, so after adding a new modality the mask can end
                tumor_name = vw.LAYER_NAMES.get("tumor", "Tumor Mask")
                layer_names = [l.name for l in vw.viewer.layers]
                if tumor_name in layer_names:
                    idx = layer_names.index(tumor_name)
                    top = len(vw.viewer.layers) - 1
                    if idx != top:
                        vw.viewer.layers.move(idx, top)

            name = vw.LAYER_NAMES[modality]
            if name in vw.viewer.layers:
                c_min = wl[1] - wl[0] / 2
                c_max = wl[1] + wl[0] / 2
                if modality == "pet":
                    c_min = max(0, c_min)
                vw.viewer.layers[name].contrast_limits = (c_min, c_max)
                vw.viewer.layers[name].visible = True

            other_modality = "pet" if modality == "ct" else "ct"
            other_name = vw.LAYER_NAMES[other_modality]
            if other_name in vw.viewer.layers:
                vw.viewer.layers[other_name].visible = False

            if self._cached_lesion_data:
                vw.show_lesion_ids(*self._cached_lesion_data)

            axis = axes.get((r, c), 0)
            vw.set_camera_view(axis)
            if len(vw.viewer.layers) > 0:
                vw.viewer.reset_view()

        self._loaded_layouts.add("mono_single")

    def _reload_mono_single(self, modality_text):
        """Called when the modality combo changes."""
        self._loaded_layouts.discard("mono_single")
        if self.stack.currentWidget() == self.mono_single_widget:
            self._load_mono_single()
            if self._crosshair_enabled:
                self.enable_crosshair_mode()

    def _load_3d_data(self):
        if self._is_3d_loaded:
            return
        ct = self._cached_data["ct"]
        pet = self._cached_data["pet"]
        affine = self._cached_data["affine"]

        if ct is not None:
            self.viewer_3d.load_image(ct, affine, "ct", self._ct_colormap)
        if pet is not None:
            self.viewer_3d.load_image(pet, affine, "pet", self._pet_colormap, opacity=1.0)
        if self._cached_data_zyx["tumor"] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
        if self._cached_lesion_data:
            self.viewer_3d.show_lesion_ids(*self._cached_lesion_data)

        ct_name = self.viewer_3d.LAYER_NAMES["ct"]
        pet_name = self.viewer_3d.LAYER_NAMES["pet"]
        if ct is not None and ct_name in self.viewer_3d.viewer.layers:
            c_min = self._ct_wl[1] - (self._ct_wl[0] / 2)
            c_max = self._ct_wl[1] + (self._ct_wl[0] / 2)
            self.viewer_3d.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
        if pet is not None and pet_name in self.viewer_3d.viewer.layers:
            p_min = max(0, self._pet_wl[1] - (self._pet_wl[0] / 2))
            p_max = self._pet_wl[1] + (self._pet_wl[0] / 2)
            self.viewer_3d.viewer.layers[pet_name].contrast_limits = (p_min, p_max)
        if pet is not None and pet_name in self.viewer_3d.viewer.layers:
            self.viewer_3d.viewer.layers[pet_name].visible = False

        self.viewer_3d.viewer.dims.ndisplay = 3
        self._is_3d_loaded = True
        self._loaded_layouts.add("3d")

    # ── View Mode Switching ──────────────────────────────────────────────

    def set_view_mode(self, mode: str):
        old_viewers = self._get_visible_viewers()
        old_viewer = old_viewers[0].viewer if old_viewers else None

        if mode == "grid":
            self.stack.setCurrentWidget(self.grid_widget)
            self._load_grid()

        elif mode.startswith("mono_single"):
            modality = "PET" if "pet" in mode else "CT"
            self._mono_single_combo.blockSignals(True)
            self._mono_single_combo.setCurrentText(modality)
            self._mono_single_combo.blockSignals(False)
            self._loaded_layouts.discard("mono_single")
            self.stack.setCurrentWidget(self.mono_single_widget)
            self._load_mono_single()

        elif mode.startswith("mono"):
            self.stack.setCurrentWidget(self.mono_widget)
            self._load_mono()
            axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
            axis = next((v for k, v in axis_map.items() if k in mode), 0)
            for v in self.mono_viewers.values():
                v.set_camera_view(axis)
                v.viewer.reset_view()

        elif mode.startswith("overlay"):
            self.stack.setCurrentWidget(self.overlay_widget)
            self._load_overlay()
            axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
            axis = next((v for k, v in axis_map.items() if k in mode), 0)
            self.overlay_viewer.set_camera_view(axis)

        elif mode == "3d":
            self.stack.setCurrentWidget(self.view_3d_widget)
            self._load_3d_data()

        # One-shot slice sync
        if old_viewer is not None:
            new_viewers = self._get_visible_viewers()
            target_napari = [v.viewer for v in new_viewers]
            one_shot_sync_step(old_viewer, target_napari)

        self._connect_mask_events()

        if self._crosshair_enabled:
            self.enable_crosshair_mode()

    # ── Display Settings ─────────────────────────────────────────────────

    def reset_zoom(self):
        for v in self._get_visible_viewers():
            v.viewer.reset_view()

    def set_pet_opacity(self, value: float):
        for widget in self._get_all_loaded_viewers():
            for layer in widget.viewer.layers:
                if layer.name == widget.LAYER_NAMES["pet"]:
                    layer.opacity = value

    def set_tumor_opacity(self, value: float):
        for widget in self._get_all_loaded_viewers():
            name = widget.LAYER_NAMES.get("tumor", "tumor")
            for layer in widget.viewer.layers:
                if layer.name == name:
                    layer.opacity = value

    def set_ct_window_level(self, window: float, level: float):
        self._ct_wl = (window, level)
        self._set_contrast_limits("ct", level - window / 2, level + window / 2)

    def set_pet_window_level(self, window: float, level: float):
        self._pet_wl = (window, level)
        min_val = max(0, level - window / 2)
        self._set_contrast_limits("pet", min_val, level + window / 2)

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
        name_map = {"tumor": "Tumor Mask"}
        target_name = name_map.get(mask_type, mask_type)
        for widget in self._get_all_loaded_viewers():
            for layer in widget.viewer.layers:
                if layer.name == target_name:
                    layer.visible = visible

    def toggle_3d_pet(self, pet_mode: bool):
        ct_name = self.viewer_3d.LAYER_NAMES["ct"]
        pet_name = self.viewer_3d.LAYER_NAMES["pet"]
        for layer in self.viewer_3d.viewer.layers:
            if layer.name == ct_name:
                layer.visible = not pet_mode
            elif layer.name == pet_name:
                layer.visible = pet_mode

    def set_drawing_tool(self, tool: str, brush_size: int, layer_type: str):
        for v in self._get_visible_viewers():
            if tool in ("sphere", "square"):
                v.set_drawing_mode(layer_type, "pan_zoom", brush_size)
                v.enable_shape_drag(layer_type, tool)
            else:
                v.disable_shape_drag()
                v.set_drawing_mode(layer_type, tool, brush_size)

    def disable_shape_drag(self):
        for v in self._get_visible_viewers():
            v.disable_shape_drag()

    def commit_shape(self, layer_type: str):
        for v in self._get_visible_viewers():
            v.commit_shape_to_mask()
        self.sig_shape_committed.emit(layer_type)

    # ── Mask Update ──────────────────────────────────────────────────────

    def update_mask(self, mask_data, mask_type):
        if mask_data is None:
            return
        self._cached_data[mask_type] = mask_data
        from ....utils.nifti_utils import to_napari
        data_zyx = to_napari(mask_data.astype(np.uint8))
        self._cached_data_zyx[mask_type] = data_zyx
        self._disconnect_mask_events()
        for v in self._get_visible_viewers():
            v.load_mask_zyx(data_zyx, mask_type)
        if self._is_3d_loaded:
            self.viewer_3d.load_mask_zyx(data_zyx, mask_type)
        self._invalidate_non_visible_layouts()
        self._connect_mask_events()

    def _invalidate_non_visible_layouts(self):
        current = self.stack.currentWidget()
        for name in ("grid", "overlay", "mono", "mono_single"):
            self._loaded_layouts.discard(name)
        if current == self.grid_widget:
            self._loaded_layouts.add("grid")
        elif current == self.overlay_widget:
            self._loaded_layouts.add("overlay")
        elif current == self.mono_widget:
            self._loaded_layouts.add("mono")
        elif current == self.mono_single_widget:
            self._loaded_layouts.add("mono_single")

    def get_active_mask_data(self, layer_type: str):
        for viewer in self._get_visible_viewers():
            data = viewer.get_layer_data(layer_type)
            if data is not None:
                return data
        return self._cached_data.get(layer_type)

    def sync_mask_cache(self, mask_data, mask_type):
        self._cached_data[mask_type] = mask_data
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
        if self._is_3d_loaded and self._cached_data_zyx[mask_type] is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx[mask_type], mask_type)
        self._invalidate_non_visible_layouts()

    # ── Viewer Clear ─────────────────────────────────────────────────────

    def clear_all_viewers(self):
        all_viewers = (
            list(self.grid_viewers.values())
            + [self.overlay_viewer]
            + list(self.mono_viewers.values())
            + list(self.mono_single_viewers.values())
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
        self._info_label.hide()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_all_loaded_viewers(self):
        viewers = []
        if "grid" in self._loaded_layouts:
            viewers.extend(self.grid_viewers.values())
        if "overlay" in self._loaded_layouts:
            viewers.append(self.overlay_viewer)
        if "mono" in self._loaded_layouts:
            viewers.extend(self.mono_viewers.values())
        if "mono_single" in self._loaded_layouts:
            viewers.extend(self.mono_single_viewers.values())
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)
        return viewers

    def _get_all_2d_viewers(self):
        """All 2D viewer widgets (across all layouts, loaded or not)."""
        return (
            list(self.grid_viewers.values())
            + [self.overlay_viewer]
            + list(self.mono_viewers.values())
            + list(self.mono_single_viewers.values())
        )

    # ── Lesion IDs ───────────────────────────────────────────────────────

    def show_lesion_ids(self, bboxes: list, lesion_ids: list):
        if not bboxes:
            self._cached_lesion_data = ([], [])
            for v in self._get_all_loaded_viewers():
                v.show_lesion_ids([], [])
            return

        nib_shape = None
        if self._cached_data.get("ct") is not None:
            nib_shape = self._cached_data["ct"].shape
        elif self._cached_data.get("pet") is not None:
            nib_shape = self._cached_data["pet"].shape

        points = []
        id_strings = []
        if nib_shape is not None:
            for bbox, lid in zip(bboxes, lesion_ids):
                d0_c = (bbox[0] + bbox[3]) / 2.0
                d1_c = (bbox[1] + bbox[4]) / 2.0
                d2_c = (bbox[2] + bbox[5]) / 2.0
                z_nap = (nib_shape[2] - 1) - d2_c
                y_nap = (nib_shape[1] - 1) - d1_c
                x_nap = d0_c
                points.append([z_nap, y_nap, x_nap])
                id_strings.append(str(lid))

        self._cached_lesion_data = (points, id_strings)
        for v in self._get_all_loaded_viewers():
            v.show_lesion_ids(points, id_strings)

    def hide_lesion_ids(self):
        self._cached_lesion_data = None
        for v in self._get_all_loaded_viewers():
            v.hide_lesion_ids()

    # ── Colormap ─────────────────────────────────────────────────────────

    def set_ct_colormap(self, colormap: str):
        self._ct_colormap = colormap
        for widget in self._get_all_loaded_viewers():
            name = widget.LAYER_NAMES.get("ct", "CT Image")
            if name in widget.viewer.layers:
                widget.viewer.layers[name].colormap = colormap

    def set_pet_colormap(self, colormap: str):
        self._pet_colormap = colormap
        for widget in self._get_all_loaded_viewers():
            name = widget.LAYER_NAMES.get("pet", "PET Image")
            if name in widget.viewer.layers:
                widget.viewer.layers[name].colormap = colormap

    # ── Crosshair mode ───────────────────────────────────────────────────

    def enable_crosshair_mode(self):
        """Enable crosshair overlay on visible viewers."""
        self._crosshair_enabled = True
        self._pan_mode = False

        # Sync _xhair_pos from actual grid viewer dims so position is current
        if "grid" in self._loaded_layouts:
            row_to_slice_dim = {0: 0, 1: 2, 2: 1}
            for row, slice_dim in row_to_slice_dim.items():
                vw = self.grid_viewers[(row, 0)]
                step = vw.viewer.dims.current_step
                if len(step) > slice_dim:
                    self._xhair_pos[slice_dim] = float(step[slice_dim])
        # Disconnect cursor signals from all first (avoid duplicates)
        for v in self._get_all_2d_viewers():
            v.disable_crosshair_mode()
            try:
                v.sig_cursor_intensity.disconnect(self._on_viewer_cursor_intensity)
            except (TypeError, RuntimeError):
                pass

        self._connect_crosshair_events()

        pos = list(self._xhair_pos)
        for v in self._get_visible_viewers():
            if not v.is_3d:
                v.enable_crosshair_mode()
                v.update_crosshair(pos)
                v.sig_cursor_intensity.connect(self._on_viewer_cursor_intensity)

        self._info_label.show()
        self._info_label.raise_()
        self._emit_crosshair_coords()

    def disable_crosshair_mode(self):
        """Disable crosshair overlay, show small cross cursor."""
        self._crosshair_enabled = False
        self._info_label.hide()
        for v in self._get_all_2d_viewers():
            v.disable_crosshair_mode()
            try:
                v.sig_cursor_intensity.disconnect(self._on_viewer_cursor_intensity)
            except (TypeError, RuntimeError):
                pass

    def set_pan_mode(self, pan_on: bool):
        """Pan mode: no overlay, arrow cursor."""
        self._pan_mode = pan_on
        if pan_on:
            self._crosshair_enabled = False
            self._info_label.hide()
            for v in self._get_all_2d_viewers():
                v.set_pan_mode(True)
                try:
                    v.sig_cursor_intensity.disconnect(self._on_viewer_cursor_intensity)
                except (TypeError, RuntimeError):
                    pass
        else:
            self.enable_crosshair_mode()

    def _on_viewer_cursor_intensity(self, text: str):
        self.sig_cursor_intensity.emit(text)

    # ── Interpolation ────────────────────────────────────────────────────

    def set_interpolation(self, enabled: bool):
        mode = "linear" if enabled else "nearest"
        for widget in self._get_all_loaded_viewers():
            for layer in widget.viewer.layers:
                if hasattr(layer, "interpolation2d"):
                    try:
                        layer.interpolation2d = mode
                    except Exception:
                        pass

    # ── Pre-load secondary layouts ────────────────────────────────────────

    def _schedule_preload(self):
        current = self.stack.currentWidget()
        self._preload_queue = []
        if current != self.grid_widget:
            self._preload_queue.append(self._load_grid)
        if current != self.overlay_widget:
            self._preload_queue.append(self._load_overlay)
        if current != self.mono_widget:
            self._preload_queue.append(self._load_mono)
        if self._preload_queue:
            QTimer.singleShot(800, self._preload_next)

    def _preload_next(self):
        if not self._preload_queue:
            return
        if self._cached_data.get("ct") is None and self._cached_data.get("pet") is None:
            self._preload_queue.clear()
            return
        fn = self._preload_queue.pop(0)
        fn()
        if self._preload_queue:
            QTimer.singleShot(200, self._preload_next)
