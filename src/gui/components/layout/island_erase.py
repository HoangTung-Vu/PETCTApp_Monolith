"""Island erase + undo for the tumor mask's manual edits — mixin for LayoutManager.

With the Segmentation tab's Erase brush active, a double-click removes the
whole island under the cursor: the 26-connected region of the selected
lesion id (skimage.flood, same value). It replaces the old Eraser tab.

The removal goes through ``Labels.data_setitem``, so it lands in napari's
undo history like a brush stroke and fires the usual ``paint`` event (dirty
flag, stale lesion values, refresh of the other views). ``undo_tumor_edit``
undoes the latest stroke / island, newest first, across all 2D views.
"""

import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


class IslandEraseMixin:

    def _init_island_erase(self):
        # Tumor layers in the order their edits were made (one entry per
        # napari history item), so Undo knows which view's history to pop.
        self._tumor_edit_layers = []
        self._last_tumor_stroke = None      # (layer, time, atoms) of the latest stroke
        self._island_worker = None

    # ── History bookkeeping ──

    def _record_tumor_stroke(self, layer, atoms):
        """Called for every finished tumor stroke (paint, erase, island)."""
        self._tumor_edit_layers.append(layer)
        del self._tumor_edit_layers[:-200]
        self._last_tumor_stroke = (layer, time.monotonic(), atoms)

    def reset_tumor_history(self):
        """The tumor mask was replaced from outside the brush (load, save, Give
        ids, AI run): the napari histories no longer match the data."""
        for vw in list(self._viewer_pool) + [self.viewer_3d]:
            name = vw.LAYER_NAMES["tumor"]
            if name in vw.viewer.layers:
                layer = vw.viewer.layers[name]
                if hasattr(layer, "_reset_history"):
                    layer._reset_history()
        self._tumor_edit_layers = []
        self._last_tumor_stroke = None

    def undo_tumor_edit(self) -> bool:
        """Undo the latest tumor stroke or island erase. False if nothing to undo."""
        while self._tumor_edit_layers:
            layer = self._tumor_edit_layers.pop()
            if getattr(layer, "_undo_history", None):
                layer.undo()
                self._last_tumor_stroke = None
                self.refresh_mask("tumor")       # the other views share the array
                return True
        return False

    # ── Island erase ──

    @staticmethod
    def _stroke_covers(atoms, pos) -> bool:
        z, y, x = pos
        for indices, _, _ in atoms or ():
            idx = [np.asarray(i) for i in indices]
            if len(idx) == 3 and np.any((idx[0] == z) & (idx[1] == y) & (idx[2] == x)):
                return True
        return False

    def _on_island_erase(self, vw, pos):
        name = vw.LAYER_NAMES["tumor"]
        if name not in vw.viewer.layers:
            return
        layer = vw.viewer.layers[name]
        data = layer.data
        z, y, x = (int(c) for c in pos)
        if not (0 <= z < data.shape[0] and 0 <= y < data.shape[1] and 0 <= x < data.shape[2]):
            return
        if self._island_worker is not None and self._island_worker.isRunning():
            return

        # The first click of the double-click already ran an Erase stroke
        # there: take it back so the island is removed whole, as one edit.
        last = self._last_tumor_stroke
        interval = QApplication.doubleClickInterval() / 1000.0 + 0.3
        if (last is not None and last[0] is layer and time.monotonic() - last[1] < interval
                and self._stroke_covers(last[2], (z, y, x))):
            if self._tumor_edit_layers and self._tumor_edit_layers[-1] is layer:
                self._tumor_edit_layers.pop()
            layer.undo()
            self._last_tumor_stroke = None

        value = int(data[z, y, x])
        target = int(self._tumor_label)
        if value == 0:
            self.sig_island_erase_info.emit("No lesion under the cursor.")
            return
        if value != target:
            self.sig_island_erase_info.emit(
                f"That is lesion {value}; the selected lesion is {target}. "
                f"Select lesion {value} to erase it.")
            return

        from ...workers import EraserFloodWorker
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        worker = EraserFloodWorker(data, (z, y, x))
        self._island_worker = worker

        def on_found(indices):
            QApplication.restoreOverrideCursor()
            n = len(indices[0])
            if n and layer.data is data:
                layer.data_setitem(indices, 0)     # history + paint event
                self.sig_island_erase_info.emit(f"Erased an island of lesion {target} ({n} voxels).")
                print(f"[IslandErase] Lesion {target}: {n} voxels at {(z, y, x)} removed.")

        def on_error(msg):
            QApplication.restoreOverrideCursor()
            print(f"[IslandErase] {msg}")

        worker.component_found.connect(on_found)
        worker.error.connect(on_error)
        worker.start()
