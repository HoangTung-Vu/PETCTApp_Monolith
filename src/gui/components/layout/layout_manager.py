"""Layout Manager — dynamic 9-view grid driven by active-view checklist.

Architecture
------------
* A pool of 9 ``ViewerWidget`` instances is pre-created at init.
* ``set_active_views(view_list)`` assigns pool viewers to the requested views,
  builds a QGridLayout (max 2 cols, ceil(N/2) rows), loads data, and syncs
  crosshair state.
* A QStackedWidget holds [dynamic_grid_container, 3d_widget] so the 3D viewer
  can take over the whole area when requested.
* Crosshair sync: ``_xhair_pos = [z, y, x]`` in Napari data space drives all
  visible 2D viewers via ``_sync_viewer_slices()`` / ``_refresh_all_crosshairs()``.
"""

import math
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QStackedWidget, QVBoxLayout, QApplication, QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt
import numpy as np
import napari

from ..viewers.viewer_widget import ViewerWidget
from .mask_sync import MaskSyncMixin
from .eraser_manager import EraserMixin
from ....utils.dimension_utils import get_spacing_from_affine


# ── View metadata ────────────────────────────────────────────────────────────

# Plane → Napari display axis
_PLANE_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}

# Sort keys for layout order (same-plane views end up adjacent → same row)
_PLANE_ORDER = {"axial": 0, "coronal": 1, "sagittal": 2}
_MOD_ORDER   = {"ct": 0, "pet": 1, "overlay": 2}

VIEW_LABELS = {
    "axial_ct":         "Axial — CT",
    "axial_pet":        "Axial — PET",
    "axial_overlay":    "Axial — Overlay",
    "coronal_ct":       "Coronal — CT",
    "coronal_pet":      "Coronal — PET",
    "coronal_overlay":  "Coronal — Overlay",
    "sagittal_ct":      "Sagittal — CT",
    "sagittal_pet":     "Sagittal — PET",
    "sagittal_overlay": "Sagittal — Overlay",
}


def _view_axis(view_id: str) -> int:
    return _PLANE_AXIS[view_id.split("_")[0]]


def _view_modality(view_id: str) -> str:
    # "axial_ct" → "ct"  |  "sagittal_overlay" → "overlay"
    return view_id.split("_")[-1]


def _view_sort_key(view_id: str):
    parts = view_id.split("_")
    return (_PLANE_ORDER[parts[0]], _MOD_ORDER[parts[-1]])


# ── Main class ───────────────────────────────────────────────────────────────

class LayoutManager(MaskSyncMixin, EraserMixin, QWidget):
    """Dynamic multi-view layout manager with a pool of 9 ViewerWidgets."""

    sig_eraser_region_removed   = pyqtSignal(object, object, object)
    sig_eraser_background_click = pyqtSignal()
    sig_mask_painted            = pyqtSignal(str)
    sig_shape_committed         = pyqtSignal(str)
    sig_cursor_intensity        = pyqtSignal(str)

    # (z_vox, y_vox, x_vox, z_mm, y_mm, x_mm, hu_str, suv_str)
    sig_crosshair_pos = pyqtSignal(float, float, float, float, float, float, str, str)

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Stack: index 0 = 2D dynamic grid, index 1 = 3D viewer
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

        # 2D dynamic grid container
        self._grid_container = QWidget()
        self._dynamic_grid = QGridLayout(self._grid_container)
        self._dynamic_grid.setContentsMargins(0, 0, 0, 0)
        self._dynamic_grid.setSpacing(1)
        self.stack.addWidget(self._grid_container)

        # Pool of 9 reusable 2D viewers
        self._viewer_pool: list[ViewerWidget] = [ViewerWidget() for _ in range(9)]
        for vw in self._viewer_pool:
            vw.hide()

        # Fixed view assignment: view_id → ViewerWidget
        self._fixed_view_map: dict[str, ViewerWidget] = {}
        for i, view_id in enumerate(VIEW_LABELS.keys()):
            if i < len(self._viewer_pool):
                self._fixed_view_map[view_id] = self._viewer_pool[i]

        # 3D viewer
        self._init_3d_view()

        self._init_mask_sync()

        # Data cache
        self._cached_data = {"ct": None, "pet": None, "affine": None, "tumor": None, "roi": None, "ct_filename": "", "pet_filename": ""}
        self._cached_data_zyx = {"tumor": None, "roi": None}
        self._is_3d_loaded = False
        self._cached_lesion_data = None

        self._active_views: list[str] = []

        # Display state
        self._ct_wl = (350.0, 35.0)
        self._pet_wl = (10.0, 5.0)
        self._ct_colormap = "gray"
        self._pet_colormap = "jet"
        self._overlay_pet_colormap = "jet"
        self._overlay_pet_opacity = 0.5
        self._tumor_opacity = 0.7

        # Crosshair state
        self._crosshair_enabled = False
        self._pan_mode = False
        self._xhair_pos = [0.0, 0.0, 0.0]   # [z, y, x] Napari data space

        # Initialise default layout so grid cells exist before first data load
        self.set_active_views(["axial_ct", "axial_pet"])

    # ── 3D viewer init ───────────────────────────────────────────────────────

    def _init_3d_view(self):
        self.view_3d_widget = QWidget()
        layout = QVBoxLayout(self.view_3d_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_3d = ViewerWidget()
        self.viewer_3d.set_3d_view()
        layout.addWidget(self.viewer_3d)
        self.stack.addWidget(self.view_3d_widget)

    # ── Active view management ────────────────────────────────────────────────

    def set_active_views(self, view_list: list):
        """Rebuild the dynamic grid for the requested set of views."""
        # Switch stack to the 2D grid pane
        self.stack.setCurrentWidget(self._grid_container)

        # Remove all widgets from the grid without destroying them
        self._clear_dynamic_grid()

        if not view_list:
            for vw in self._viewer_pool:
                vw.hide()
            self._active_views = []
            return

        # Do not sort, preserve the requested list order
        sorted_views = list(view_list)
        n = len(sorted_views)
        cols = 1 if n == 1 else 2
        rows = math.ceil(n / cols)

        # Remove all widgets from the grid without destroying them (already done above)

        self._active_views = sorted_views

        # Hide unassigned pool viewers and clear their memory
        assigned_views = set(sorted_views)
        for view_id, vw in self._fixed_view_map.items():
            if view_id not in assigned_views:
                vw.hide()
                vw.viewer.layers.clear()

        # Place assigned viewers into grid
        for i, view_id in enumerate(sorted_views):
            row, col = divmod(i, cols)
            vw = self._fixed_view_map[view_id]
            self._dynamic_grid.addWidget(vw, row, col)
            vw.show()

        for c in range(cols):
            self._dynamic_grid.setColumnStretch(c, 1)
        for r in range(rows):
            self._dynamic_grid.setRowStretch(r, 1)

        self._active_views = sorted_views

        # Load data into viewers if already cached
        self._load_active_views()
        self._connect_mask_events()

        if self._crosshair_enabled:
            self.enable_crosshair_mode()
        else:
            self._connect_crosshair_events()

    def _clear_dynamic_grid(self):
        """Remove all widgets from the dynamic grid (does not destroy them)."""
        while self._dynamic_grid.count():
            item = self._dynamic_grid.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                self._dynamic_grid.removeWidget(w)
        for i in range(self._dynamic_grid.rowCount()):
            self._dynamic_grid.setRowStretch(i, 0)
        for i in range(self._dynamic_grid.columnCount()):
            self._dynamic_grid.setColumnStretch(i, 0)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self, ct_data=None, pet_data=None, ct_affine=None, pet_affine=None, tumor_mask=None, roi_mask=None, ct_filename: str = "", pet_filename: str = "", affine=None):
        if ct_affine is None and affine is not None:
            ct_affine = affine
        if pet_affine is None and affine is not None:
            pet_affine = affine
            
        self._cached_data["ct"] = ct_data
        self._cached_data["pet"] = pet_data
        self._cached_data["ct_affine"] = ct_affine
        self._cached_data["pet_affine"] = pet_affine
        self._cached_data["affine"] = ct_affine if ct_affine is not None else pet_affine
        self._cached_data["tumor"] = tumor_mask
        self._cached_data["roi"] = roi_mask
        self._cached_data["ct_filename"] = ct_filename
        self._cached_data["pet_filename"] = pet_filename
        self._is_3d_loaded = False

        from ....utils.nifti_utils import to_napari
        if tumor_mask is not None:
            self._cached_data_zyx["tumor"] = to_napari(tumor_mask.astype(np.uint8, copy=False))
        else:
            self._cached_data_zyx["tumor"] = None
        if roi_mask is not None:
            self._cached_data_zyx["roi"] = to_napari(roi_mask.astype(np.uint8, copy=False))
        else:
            self._cached_data_zyx["roi"] = None

        if ct_data is not None:
            self._cached_data_zyx["ct"] = to_napari(ct_data)
        else:
            self._cached_data_zyx["ct"] = None

        if pet_data is not None:
            self._cached_data_zyx["pet"] = to_napari(pet_data)
        else:
            self._cached_data_zyx["pet"] = None

        # Init crosshair at volume centre
        ref = ct_data if ct_data is not None else pet_data
        if ref is not None:
            sh = ref.shape   # XYZ nibabel
            self._xhair_pos = [sh[2] / 2.0, sh[1] / 2.0, sh[0] / 2.0]

        if affine is not None:
            sxyz = get_spacing_from_affine(affine)
            self._scale_zyx = (float(sxyz[2]), float(sxyz[1]), float(sxyz[0]))

        self._load_active_views()
        self._connect_mask_events()

    def _load_active_views(self):
        """Push cached data into all currently assigned pool viewers.

        Preserves ``_xhair_pos`` so that switching views does not jump the
        crosshair / axial slice back to the beginning.
        """
        if not self._active_views:
            return

        # ── Save crosshair position before any viewer manipulation ──
        saved_pos = list(self._xhair_pos)

        ct_zyx = self._cached_data_zyx.get("ct")
        pet_zyx = self._cached_data_zyx.get("pet")
        ct_affine = self._cached_data.get("ct_affine")
        pet_affine = self._cached_data.get("pet_affine")

        c_min = self._ct_wl[1] - self._ct_wl[0] / 2
        c_max = self._ct_wl[1] + self._ct_wl[0] / 2
        p_min = max(0.0, self._pet_wl[1] - self._pet_wl[0] / 2)
        p_max = self._pet_wl[1] + self._pet_wl[0] / 2

        # Suppress slice-changed events during loading so that reset_view()
        # triggered dim changes don't overwrite _xhair_pos.
        self._is_syncing_slices = True
        try:
            for view_id in self._active_views:
                vw = self._fixed_view_map[view_id]
                axis = _view_axis(view_id)
                modality = _view_modality(view_id)
                wants_ct  = modality in ("ct",  "overlay")
                wants_pet = modality in ("pet", "overlay")

                # Load image layers
                if wants_ct and ct_zyx is not None and ct_affine is not None:
                    cmap = "gray" if modality == "overlay" else self._ct_colormap
                    vw.load_image_zyx(ct_zyx, ct_affine, "ct", cmap)
                    ct_name = vw.LAYER_NAMES["ct"]
                    if ct_name in vw.viewer.layers:
                        vw.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
                if wants_pet and pet_zyx is not None and pet_affine is not None:
                    pet_opacity = self._overlay_pet_opacity if modality == "overlay" else 1.0
                    cmap = self._overlay_pet_colormap if modality == "overlay" else self._pet_colormap
                    vw.load_image_zyx(pet_zyx, pet_affine, "pet", cmap, opacity=pet_opacity)
                    pet_name = vw.LAYER_NAMES["pet"]
                    if pet_name in vw.viewer.layers:
                        vw.viewer.layers[pet_name].contrast_limits = (p_min, p_max)

                # Set layer visibility
                ct_name  = vw.LAYER_NAMES["ct"]
                pet_name = vw.LAYER_NAMES["pet"]
                if ct_name in vw.viewer.layers:
                    vw.viewer.layers[ct_name].visible = wants_ct
                if pet_name in vw.viewer.layers:
                    vw.viewer.layers[pet_name].visible = wants_pet

                # Load masks
                if self._cached_data_zyx.get("tumor") is not None:
                    vw.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
                if self._cached_data_zyx.get("roi") is not None:
                    vw.load_mask_zyx(self._cached_data_zyx["roi"], "roi")

                # Set camera orientation (axis order) without resetting slice
                vw.set_camera_view(axis)
                # Reset zoom only (slice will be restored below via _sync)
                if len(vw.viewer.layers) > 0:
                    vw.viewer.reset_view()

                # View name badge
                label_text = VIEW_LABELS.get(view_id, view_id)
                ct_name = self._cached_data.get("ct_filename", "")
                pet_name = self._cached_data.get("pet_filename", "")
                
                if wants_ct and not wants_pet and ct_name:
                    label_text += f" ({ct_name})"
                elif wants_pet and not wants_ct and pet_name:
                    label_text += f" ({pet_name})"
                elif wants_ct and wants_pet:
                    label_text += " (Overlay)"
                    
                vw.set_view_label(label_text)

                if self._cached_lesion_data:
                    vw.show_lesion_ids(*self._cached_lesion_data)

                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
        finally:
            # ── Restore crosshair position that may have been clobbered ──
            self._xhair_pos = saved_pos
            self._is_syncing_slices = False

        # Now sync all viewer slices to the (preserved) crosshair position
        self._sync_viewer_slices()

    # ── 3D data loading ───────────────────────────────────────────────────────

    def _load_3d_data(self):
        if self._is_3d_loaded:
            return
        ct = self._cached_data.get("ct")
        pet = self._cached_data.get("pet")
        ct_zyx = self._cached_data_zyx.get("ct")
        pet_zyx = self._cached_data_zyx.get("pet")
        ct_affine = self._cached_data.get("ct_affine")
        pet_affine = self._cached_data.get("pet_affine")

        if ct_zyx is not None and ct_affine is not None:
            self.viewer_3d.load_image_zyx(ct_zyx, ct_affine, "ct", self._ct_colormap)
        if pet_zyx is not None and pet_affine is not None:
            self.viewer_3d.load_image_zyx(pet_zyx, pet_affine, "pet", self._pet_colormap, opacity=1.0)
        if self._cached_data_zyx.get("tumor") is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["tumor"], "tumor")
        if self._cached_data_zyx.get("roi") is not None:
            self.viewer_3d.load_mask_zyx(self._cached_data_zyx["roi"], "roi")
        if self._cached_lesion_data:
            self.viewer_3d.show_lesion_ids(*self._cached_lesion_data)

        ct_name  = self.viewer_3d.LAYER_NAMES["ct"]
        pet_name = self.viewer_3d.LAYER_NAMES["pet"]
        c_min = self._ct_wl[1] - self._ct_wl[0] / 2
        c_max = self._ct_wl[1] + self._ct_wl[0] / 2
        p_min = max(0.0, self._pet_wl[1] - self._pet_wl[0] / 2)
        p_max = self._pet_wl[1] + self._pet_wl[0] / 2
        if ct is not None and ct_name in self.viewer_3d.viewer.layers:
            self.viewer_3d.viewer.layers[ct_name].contrast_limits = (c_min, c_max)
        if pet is not None and pet_name in self.viewer_3d.viewer.layers:
            self.viewer_3d.viewer.layers[pet_name].contrast_limits = (p_min, p_max)
            self.viewer_3d.viewer.layers[pet_name].visible = False

        self.viewer_3d.viewer.dims.ndisplay = 3
        self._is_3d_loaded = True

    # ── View mode (3D only) ───────────────────────────────────────────────────

    def set_view_mode(self, mode: str):
        """Handles 3D mode toggling. 2D views use set_active_views instead."""
        if mode == "3d":
            self.stack.setCurrentWidget(self.view_3d_widget)
            self._load_3d_data()
            self.viewer_3d.viewer.dims.ndisplay = 3
            self.viewer_3d.viewer.camera.mouse_pan = True
            self.viewer_3d.viewer.camera.mouse_zoom = True
            for layer in self.viewer_3d.viewer.layers:
                if isinstance(layer, napari.layers.Labels):
                    layer.editable = False
                    layer.mode = "pan_zoom"
        else:
            # Return to 2D grid (e.g. if something sends a legacy mode string)
            self.stack.setCurrentWidget(self._grid_container)

        self._connect_mask_events()
        if self._crosshair_enabled:
            self.enable_crosshair_mode()

    # ── Crosshair event handling ──────────────────────────────────────────────

    def _connect_crosshair_events(self):
        """Connect click / scroll / arrow-key signals from all visible 2D viewers."""
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
            try:
                vw.sig_camera_changed.disconnect(self._on_viewer_camera_changed)
            except (TypeError, RuntimeError):
                pass

        for vw in self._get_visible_viewers():
            if not vw.is_3d:
                vw.sig_crosshair_clicked.connect(self._on_viewer_crosshair_click)
                vw.sig_slice_changed.connect(
                    lambda step, v=vw: self._on_viewer_slice_changed(step, v)
                )
                vw.sig_crosshair_arrow.connect(self._on_crosshair_arrow)
                vw.sig_camera_changed.connect(self._on_viewer_camera_changed)

    def _on_viewer_slice_changed(self, current_step: tuple, viewer_widget_ref):
        """Update _xhair_pos when a viewer is scrolled."""
        if getattr(self, '_is_syncing_slices', False):
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

    def jump_to_position(self, z: float, y: float, x: float):
        self._xhair_pos = [z, y, x]
        self._sync_viewer_slices()
        self._refresh_all_crosshairs()
        self._emit_crosshair_coords()

    def _on_viewer_crosshair_click(self, pos_zyx: list):
        self._xhair_pos = [pos_zyx[0], pos_zyx[1], pos_zyx[2]]
        self._sync_viewer_slices()
        self._refresh_all_crosshairs()
        self._emit_crosshair_coords()

    def _on_crosshair_arrow(self, dim: int, delta: int):
        if not self._crosshair_enabled:
            return
        new_val = self._xhair_pos[dim] + delta
        for vw in self._get_visible_viewers():
            if not vw.is_3d and vw.viewer.dims.ndim > dim:
                limit = vw.viewer.dims.range[dim][1]
                new_val = max(0.0, min(float(limit) - 1, new_val))
                break
        self._xhair_pos[dim] = new_val
        self._sync_viewer_slices()
        self._refresh_all_crosshairs()
        self._emit_crosshair_coords()

    def _on_viewer_camera_changed(self, source_vw):
        if getattr(self, '_is_syncing_camera', False):
            return
        self._is_syncing_camera = True
        try:
            source_order = source_vw.viewer.dims.order
            source_zoom = source_vw.viewer.camera.zoom
            source_center = source_vw.viewer.camera.center
            for vw in self._get_visible_viewers():
                if vw is not source_vw and not vw.is_3d:
                    if vw.viewer.dims.order == source_order:
                        vw.viewer.camera.zoom = source_zoom
                        vw.viewer.camera.center = source_center
        finally:
            self._is_syncing_camera = False

    def _sync_viewer_slices(self):
        """Set slice index in all active 2D viewers to match _xhair_pos."""
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
                        target = int(round(self._xhair_pos[d]))
                        limit  = vw.viewer.dims.range[d][1]
                        target = max(0, min(int(limit) - 1, target))
                        if step[d] != target:
                            step[d] = target
                            changed = True
                if changed:
                    vw.viewer.dims.current_step = tuple(step)
        except Exception:
            pass
        finally:
            self._is_syncing_slices = False

    def _refresh_all_crosshairs(self):
        pos = list(self._xhair_pos)
        for vw in self._get_all_2d_viewers():
            vw.update_crosshair(pos)

    def _emit_crosshair_coords(self):
        z, y, x = self._xhair_pos
        affine  = self._cached_data.get("affine")
        ct_data = self._cached_data.get("ct")
        pet_data = self._cached_data.get("pet")
        ref_data = ct_data if ct_data is not None else pet_data

        if affine is not None and ref_data is not None:
            sh = ref_data.shape   # XYZ nibabel
            z_prime = sh[2] - 1 - z
            y_prime = sh[1] - 1 - y
            x_prime = x
            vec = np.array([x_prime, y_prime, z_prime, 1.0])
            mm = affine @ vec
            x_mm, y_mm, z_mm = float(mm[0]), float(mm[1]), float(mm[2])
        else:
            z_mm = y_mm = x_mm = 0.0

        hu_str = suv_str = "---"
        zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
        for vw in self._get_visible_viewers():
            if vw.is_3d:
                continue
            for lt, label in [("ct", "CT"), ("pet", "PET")]:
                name = vw.LAYER_NAMES.get(lt)
                if name and name in vw.viewer.layers:
                    layer = vw.viewer.layers[name]
                    if not layer.visible:
                        continue
                    d = layer.data
                    if 0 <= zi < d.shape[0] and 0 <= yi < d.shape[1] and 0 <= xi < d.shape[2]:
                        val = float(d[zi, yi, xi])
                        if label == "CT" and hu_str == "---":
                            hu_str = f"{val:.0f}"
                        elif label == "PET" and suv_str == "---":
                            suv_str = f"{val:.2f}"
            if hu_str != "---" and suv_str != "---":
                break

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
        margin = 8
        self._info_label.move(self.width() - self._info_label.width() - margin, margin)
        if self._crosshair_enabled:
            self._info_label.show()
            self._info_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._info_label.isVisible():
            margin = 8
            self._info_label.move(self.width() - self._info_label.width() - margin, margin)

    # ── Display settings ──────────────────────────────────────────────────────

    def reset_zoom(self):
        for v in self._get_visible_viewers():
            v.viewer.reset_view()

    def set_overlay_pet_opacity(self, value: float):
        self._overlay_pet_opacity = value
        for view_id in self._active_views:
            if view_id.endswith("_overlay"):
                vw = self._fixed_view_map[view_id]
                name = vw.LAYER_NAMES.get("pet", "PET Image")
                if name in vw.viewer.layers:
                    vw.viewer.layers[name].opacity = value

    def set_tumor_opacity(self, value: float):
        self._tumor_opacity = value
        for vw in self._get_all_loaded_viewers():
            name = vw.LAYER_NAMES.get("tumor", "Tumor Mask")
            if name in vw.viewer.layers:
                vw.viewer.layers[name].opacity = value

    def set_roi_opacity(self, value: float):
        for vw in self._get_all_loaded_viewers():
            name = vw.LAYER_NAMES.get("roi", "ROI Mask")
            if name in vw.viewer.layers:
                vw.viewer.layers[name].opacity = value

    def set_ct_window_level(self, window: float, level: float):
        self._ct_wl = (window, level)
        self._set_contrast_limits("ct", level - window / 2, level + window / 2)

    def set_pet_window_level(self, window: float, level: float):
        self._pet_wl = (window, level)
        self._set_contrast_limits("pet", max(0.0, level - window / 2), level + window / 2)

    def _set_contrast_limits(self, layer_type: str, min_val: float, max_val: float):
        loaded = self._get_all_loaded_viewers()
        if not loaded:
            return
        name = loaded[0].LAYER_NAMES.get(layer_type, layer_type)
        for vw in loaded:
            if name in vw.viewer.layers:
                vw.viewer.layers[name].contrast_limits = (min_val, max_val)

    def set_zoom(self, value: float):
        zoom_factor = 0.1 + (value / 100.0) * 4.9
        for vw in self._get_visible_viewers():
            if not vw.is_3d:
                vw.viewer.camera.zoom = zoom_factor

    def toggle_mask(self, mask_type: str, visible: bool):
        name_map = {"tumor": "Tumor Mask", "roi": "ROI Mask"}
        target_name = name_map.get(mask_type, mask_type)
        for vw in self._get_all_loaded_viewers():
            if target_name in vw.viewer.layers:
                vw.viewer.layers[target_name].visible = visible

    def toggle_3d_pet(self, pet_mode: bool):
        ct_name  = self.viewer_3d.LAYER_NAMES["ct"]
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

    def deactivate_labels(self):
        """Deactivate Labels layers in all viewers (select Image layer instead).

        Called when leaving paint/edit tabs so napari's built-in Labels drag
        behaviour doesn't hijack crosshair or pan mouse events.
        """
        for v in self._get_visible_viewers():
            v.deactivate_labels()

    # ── Mask update ───────────────────────────────────────────────────────────

    def update_mask(self, mask_data, mask_type, data_zyx=None):
        if mask_data is None:
            return

        is_same_object = (mask_data is self._cached_data.get(mask_type))
        self._cached_data[mask_type] = mask_data
        existing_zyx = self._cached_data_zyx.get(mask_type)

        if data_zyx is None:
            if is_same_object and existing_zyx is not None:
                # Array is the same object, its ZYX counterpart is already in sync or we don't need to rebuild it
                data_zyx = existing_zyx
            else:
                from ....utils.nifti_utils import to_napari
                data_zyx = to_napari(mask_data.astype(np.uint8, copy=False))

        if existing_zyx is not None and getattr(existing_zyx, 'shape', None) == data_zyx.shape:
            if data_zyx is not existing_zyx:
                np.copyto(existing_zyx, data_zyx, casting='unsafe')
            data_zyx = existing_zyx
            
        self._cached_data_zyx[mask_type] = data_zyx

        print(f"[LayoutManager] update_mask: starting viewer loop for {mask_type}")
        self._disconnect_mask_events()
        from PyQt6.QtWidgets import QApplication
        viewers = self._get_visible_viewers()
        for i, v in enumerate(viewers):
            print(f"[LayoutManager] update_mask: pushing to viewer {i+1}/{len(viewers)}...")
            v.load_mask_zyx(data_zyx, mask_type)
            QApplication.processEvents()  # Prevent OS "Not Responding" freeze during heavy layer creation
            
        print(f"[LayoutManager] update_mask: pushing to 3D viewer...")
        if self._is_3d_loaded:
            self.viewer_3d.load_mask_zyx(data_zyx, mask_type)
            
        print(f"[LayoutManager] update_mask: connecting events...")
        self._connect_mask_events()
        print(f"[LayoutManager] update_mask: finished for {mask_type}")

    def get_active_mask_data(self, layer_type: str):
        for vw in self._get_visible_viewers():
            data = vw.get_layer_data(layer_type)
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

        if self._cached_data_zyx[mask_type] is not None:
            for v in self._get_visible_viewers():
                v.load_mask_zyx(self._cached_data_zyx[mask_type], mask_type)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear_all_viewers(self):
        for vw in self._viewer_pool:
            vw.viewer.layers.clear()
        self.viewer_3d.viewer.layers.clear()
        self._cached_data = {"ct": None, "pet": None, "affine": None, "tumor": None, "roi": None, "ct_filename": "", "pet_filename": ""}
        self._cached_data_zyx = {"tumor": None, "roi": None}
        self._is_3d_loaded = False
        # Do not clear active views so layout persists between sessions
        # self._active_views = []
        self._cached_lesion_data = None
        self._info_label.hide()
        # self._clear_dynamic_grid()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_visible_viewers(self) -> list:
        """Alias for backwards compatibility with legacy handler calls."""
        return self._get_all_loaded_viewers()

    def _get_all_loaded_viewers(self) -> list:
        """All pool viewers currently assigned to an active view (+ 3D if loaded)."""
        viewers = [self._fixed_view_map[v_id] for v_id in self._active_views]
        if self._is_3d_loaded:
            viewers.append(self.viewer_3d)
        return viewers

    def _get_all_2d_viewers(self) -> list:
        """All pool viewers (regardless of assignment)."""
        return list(self._viewer_pool)

    # ── Lesion IDs ────────────────────────────────────────────────────────────

    def show_lesion_ids(self, bboxes: list, lesion_ids: list):
        if not bboxes:
            self._cached_lesion_data = ([], [])
            for v in self._get_all_loaded_viewers():
                v.show_lesion_ids([], [])
            return

        nib_shape = None
        for key in ("ct", "pet"):
            if self._cached_data.get(key) is not None:
                nib_shape = self._cached_data[key].shape
                break

        points, id_strings = [], []
        if nib_shape is not None:
            for bbox, lid in zip(bboxes, lesion_ids):
                z_nap = (nib_shape[2] - 1) - (bbox[2] + bbox[5]) / 2.0
                y_nap = (nib_shape[1] - 1) - (bbox[1] + bbox[4]) / 2.0
                x_nap = (bbox[0] + bbox[3]) / 2.0
                points.append([z_nap, y_nap, x_nap])
                id_strings.append(str(lid))

        self._cached_lesion_data = (points, id_strings)
        for v in self._get_all_loaded_viewers():
            v.show_lesion_ids(points, id_strings)

    def hide_lesion_ids(self):
        self._cached_lesion_data = None
        for v in self._get_all_loaded_viewers():
            v.hide_lesion_ids()

    # ── Colormap ──────────────────────────────────────────────────────────────

    def set_ct_colormap(self, colormap: str):
        self._ct_colormap = colormap
        for view_id in self._active_views:
            if view_id.endswith("_overlay"):
                continue
            vw = self._fixed_view_map[view_id]
            name = vw.LAYER_NAMES.get("ct", "CT Image")
            if name in vw.viewer.layers:
                vw.viewer.layers[name].colormap = colormap
        if self._is_3d_loaded:
            name = self.viewer_3d.LAYER_NAMES.get("ct", "CT Image")
            if name in self.viewer_3d.viewer.layers:
                self.viewer_3d.viewer.layers[name].colormap = colormap

    def set_pet_colormap(self, colormap: str):
        self._pet_colormap = colormap
        for view_id in self._active_views:
            if view_id.endswith("_overlay"):
                continue
            vw = self._fixed_view_map[view_id]
            name = vw.LAYER_NAMES.get("pet", "PET Image")
            if name in vw.viewer.layers:
                vw.viewer.layers[name].colormap = colormap
        if self._is_3d_loaded:
            name = self.viewer_3d.LAYER_NAMES.get("pet", "PET Image")
            if name in self.viewer_3d.viewer.layers:
                self.viewer_3d.viewer.layers[name].colormap = colormap

    def set_overlay_pet_colormap(self, colormap: str):
        self._overlay_pet_colormap = colormap
        for view_id in self._active_views:
            if view_id.endswith("_overlay"):
                vw = self._fixed_view_map[view_id]
                name = vw.LAYER_NAMES.get("pet", "PET Image")
                if name in vw.viewer.layers:
                    vw.viewer.layers[name].colormap = colormap

    # ── Crosshair mode ────────────────────────────────────────────────────────

    def enable_crosshair_mode(self):
        self._crosshair_enabled = True
        self._pan_mode = False

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

        # Deactivate Labels layers so their drag doesn't fight with crosshair
        self.deactivate_labels()

        # Ensure all viewers show the correct slice from _xhair_pos
        self._sync_viewer_slices()

        self._info_label.show()
        self._info_label.raise_()
        self._emit_crosshair_coords()

    def disable_crosshair_mode(self):
        self._crosshair_enabled = False
        self._info_label.hide()
        for v in self._get_all_2d_viewers():
            v.disable_crosshair_mode()
            try:
                v.sig_cursor_intensity.disconnect(self._on_viewer_cursor_intensity)
            except (TypeError, RuntimeError):
                pass

    def set_pan_mode(self, pan_on: bool):
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

    # ── Interpolation ─────────────────────────────────────────────────────────

    def set_interpolation(self, enabled: bool):
        mode = "linear" if enabled else "nearest"
        for vw in self._get_all_loaded_viewers():
            for layer in vw.viewer.layers:
                if hasattr(layer, "interpolation2d"):
                    try:
                        layer.interpolation2d = mode
                    except Exception:
                        pass
