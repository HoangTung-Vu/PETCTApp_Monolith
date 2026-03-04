"""Eraser tool (connected component removal) mixin for LayoutManager."""

import numpy as np


class EraserMixin:
    """Mixin providing eraser click-to-remove-component functionality."""

    def enable_eraser_click_mode(self):
        """Install single-click callback on all 2D viewers for contour erasing."""
        # BUG-07 FIX: Mutual exclusion — disable autopet when enabling eraser
        self.disable_autopet_click_mode()

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
            coord_zyx = self._world_to_data(event.position)
            z, y, x = coord_zyx
            print(f"[Eraser] Click at ZYX={coord_zyx}")

            mask_zyx = self._cached_data_zyx.get("tumor")
            if mask_zyx is None:
                print("[Eraser] No tumor mask loaded.")
                return

            if not (0 <= z < mask_zyx.shape[0] and
                    0 <= y < mask_zyx.shape[1] and
                    0 <= x < mask_zyx.shape[2]):
                print("[Eraser] Click out of bounds.")
                return

            if mask_zyx[z, y, x] == 0:
                print("[Eraser] Clicked on background (label=0), nothing to erase.")
                return

            # Identify connected component at the click point
            # Use skimage.morphology.flood(mask, (z, y, x)) for extreme speed
            # if skimage is available. Otherwise fallback to local region labeling.
            try:
                from skimage.morphology import flood
                component_mask = flood(mask_zyx, (z, y, x))
                num_voxels = int(np.sum(component_mask))
            except ImportError:
                # Fallback to scipy.ndimage.label (re-labeling whole volume is slower)
                from scipy.ndimage import label as nd_label
                labeled, num_features = nd_label(mask_zyx)
                component_id = labeled[z, y, x]
                component_mask = (labeled == component_id)
                num_voxels = int(np.sum(component_mask))

            # Convert back to Nibabel space (X, Y, Z) and emit signal
            from ....utils.nifti_utils import from_napari
            old_mask_xyz = from_napari(mask_zyx).copy()

            mask_zyx[component_mask] = 0
            print(f"[Eraser] Removed component at {coord_zyx} ({num_voxels} voxels).")

            # BUG-02 FIX: Refresh ALL loaded viewers (not just visible)
            # so layout switching shows updated data.
            for v in self._get_all_loaded_viewers():
                tumor_name = v.LAYER_NAMES.get("tumor", "Tumor Mask")
                if tumor_name in v.viewer.layers:
                    v.viewer.layers[tumor_name].refresh()

            # BUG-02 FIX: Update XYZ cache so non-loaded layouts
            # get correct data when they load.
            new_mask_xyz = from_napari(mask_zyx).copy()
            self._cached_data["tumor"] = new_mask_xyz

            self.sig_eraser_region_removed.emit(old_mask_xyz, new_mask_xyz)

        return on_click

    # ──── Lesion ID Labels ────

    def show_lesion_ids(self, bboxes: list, lesion_ids: list):
        """Push lesion ID labels to all LOADED viewers (including 3D)."""
        self._cached_lesion_data = (bboxes, lesion_ids)
        for v in self._get_all_loaded_viewers():
            v.show_lesion_ids(bboxes, lesion_ids)

    def hide_lesion_ids(self):
        """Remove lesion ID labels from all viewers."""
        self._cached_lesion_data = None
        for v in self._get_all_loaded_viewers():
            v.hide_lesion_ids()
