"""Lesion quantification boxes (double-click toggle) — mixin for LayoutManager."""

import numpy as np

from ....utils.nifti_utils import crop_to_napari


class LesionInfoMixin:
    """Keeps the session's lesions in Napari space and the set of lesions whose
    quantification box is shown on every 2D view.

    Shown lesions are remembered by their SUVmax voxel, not by ID: a recompute
    (after a save) may renumber the lesions, and the box must stay on the same
    lesion — or disappear if that lesion was erased.
    """

    def _init_lesion_info(self):
        self._lesion_entries = []        # lesions in Napari space (see set_lesions)
        self._lesion_stale = False
        self._shown_lesion_keys = []     # SUVmax voxels (z, y, x) of shown lesions

    def set_lesions(self, stats, shape_xyz, stale: bool = False):
        """Load the lesions of ``stats`` (``ReportEngine.compute_lesion_stats``).

        ``shape_xyz`` is the XYZ grid the stats were computed on.
        """
        entries = []
        if stats and shape_xyz is not None:
            _, Y, Z = shape_xyz
            for les in stats["lesions"]:
                lo, crop = crop_to_napari(les["slices"], les["mask"], shape_xyz)
                crop = np.ascontiguousarray(crop)
                vx, vy, vz = les["suvmax_voxel"]
                entries.append({
                    "id": les["id"],
                    "lo": lo,
                    "crop": crop,
                    "occupancy": [crop.any(axis=(1, 2)), crop.any(axis=(0, 2)),
                                  crop.any(axis=(0, 1))],
                    "key": (Z - 1 - vz, Y - 1 - vy, vx),
                    "values": {k: les[k] for k in ("MTV", "SUVmax", "SUVmean", "SUVpeak", "TLG")},
                })

        old_keys = self._shown_lesion_keys
        self._lesion_entries = entries
        self._lesion_stale = stale
        keys = []
        for key in old_keys:
            entry = self.lesion_at(key)
            if entry is not None and entry["key"] not in keys:
                keys.append(entry["key"])
        self._shown_lesion_keys = keys
        self._push_lesion_infos()

    def set_lesion_stale(self, stale: bool):
        """Mark the shown values as outdated (mask edited, not saved yet)."""
        if stale != self._lesion_stale:
            self._lesion_stale = stale
            self._push_lesion_infos()

    def lesion_at(self, pos_zyx):
        """The lesion entry containing voxel ``pos_zyx`` (Napari space), or None."""
        z, y, x = (int(round(c)) for c in pos_zyx)
        for e in self._lesion_entries:
            lz, ly, lx = e["lo"]
            dz, dy, dx = e["crop"].shape
            if (lz <= z < lz + dz and ly <= y < ly + dy and lx <= x < lx + dx
                    and e["crop"][z - lz, y - ly, x - lx]):
                return e
        return None

    def toggle_lesion_info(self, pos_zyx):
        """Show / hide the box of the lesion at ``pos_zyx``.

        Showing a lesion also moves every 2D view to ``pos_zyx`` so all planes
        cut through it. Returns ``(lesion_id, shown)`` or None when no lesion
        is there.
        """
        entry = self.lesion_at(pos_zyx)
        if entry is None:
            return None
        key = entry["key"]
        if key in self._shown_lesion_keys:
            self._shown_lesion_keys.remove(key)
            shown = False
        else:
            self._shown_lesion_keys.append(key)
            shown = True
            self.jump_to_position(*(float(c) for c in pos_zyx))
        self._push_lesion_infos()
        return entry["id"], shown

    def show_only_lesion(self, lesion_id: int) -> bool:
        """Show the box of lesion ``lesion_id`` alone (other boxes are hidden)
        and move every 2D view to it. Returns False when there is no such lesion.

        The views go to the lesion's SUVmax voxel: it lies inside the lesion, so
        all three planes cut through it (the bbox centre may not, e.g. for a
        curved lesion).
        """
        entry = next((e for e in self._lesion_entries if e["id"] == lesion_id), None)
        if entry is None:
            return False
        pos = [float(c) for c in entry["key"]]
        self._shown_lesion_keys = [entry["key"]]
        self.jump_to_position(*pos)
        self._center_views_on(pos)
        self._push_lesion_infos()
        return True

    def _center_views_on(self, pos_zyx):
        """Pan the 2D views where ``pos_zyx`` is off-screen (zoomed in) onto it."""
        for vw in self._get_visible_viewers():
            if not vw.is_3d:
                vw.center_on_if_offscreen(pos_zyx)

    def clear_lesion_info(self):
        self._lesion_entries = []
        self._shown_lesion_keys = []
        self._lesion_stale = False
        self._push_lesion_infos()

    def _shown_lesion_infos(self) -> list:
        by_key = {e["key"]: e for e in self._lesion_entries}
        return [by_key[k] for k in self._shown_lesion_keys if k in by_key]

    def _push_lesion_infos(self):
        infos = self._shown_lesion_infos()
        for vw in self._get_all_2d_viewers():
            vw.set_lesion_infos(infos, self._lesion_stale)

    def _on_viewer_lesion_double_click(self, pos_zyx: list):
        self.sig_lesion_double_clicked.emit(pos_zyx)
