"""LesionInfoMixin.show_only_lesion (lesion table click) — headless."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.engine.report_engine import ReportEngine  # noqa: E402
from src.gui.components.layout.lesion_info_manager import LesionInfoMixin  # noqa: E402

SHAPE = (40, 36, 30)                                   # X, Y, Z
AFFINE = np.diag([-2.0, 2.0, 3.0, 1.0])                # LAS


class _NoViewers(LesionInfoMixin):
    """LesionInfoMixin without a LayoutManager; records the slice jumps."""

    def __init__(self):
        self._init_lesion_info()
        self.jumps = []

    def _get_all_2d_viewers(self):
        return []

    def _get_visible_viewers(self):
        return []

    def jump_to_position(self, *pos):
        self.jumps.append(tuple(pos))


def _napari_zyx(voxel_xyz):
    x, y, z = voxel_xyz
    X, Y, Z = SHAPE
    return (Z - 1 - z, Y - 1 - y, x)


def _setup():
    pet = np.ones(SHAPE, dtype=np.float32, order="F")
    mask = np.zeros(SHAPE, dtype=np.uint8, order="F")
    mask[5:9, 5:9, 2:6] = 1                 # lesion 2 (low z = feet)
    pet[5:9, 5:9, 2:6] = 4.0
    pet[6, 7, 3] = 10.0
    mask[20:23, 20:23, 20:24] = 1           # lesion 1 (head)
    pet[21, 20, 22] = 8.0
    stats = ReportEngine.compute_lesion_stats(pet, mask, AFFINE)
    gui = _NoViewers()
    gui.set_lesions(stats, SHAPE)
    return gui, stats


def _shown_ids(gui):
    return [e["id"] for e in gui._shown_lesion_infos()]


def test_jumps_to_suvmax_voxel_and_shows_only_that_lesion():
    gui, stats = _setup()
    les2 = stats["lesions"][1]
    assert gui.show_only_lesion(2)
    assert gui.jumps == [tuple(float(c) for c in _napari_zyx(les2["suvmax_voxel"]))]
    assert _napari_zyx(les2["suvmax_voxel"]) == _napari_zyx((6, 7, 3))
    assert _shown_ids(gui) == [2]
    # The jump target lies inside the lesion → every plane cuts it.
    assert gui.lesion_at(gui.jumps[-1])["id"] == 2

    assert gui.show_only_lesion(1)
    assert _shown_ids(gui) == [1]
    assert gui.jumps[-1] == tuple(float(c) for c in _napari_zyx((21, 20, 22)))


def test_same_lesion_twice_stays_shown():
    gui, _ = _setup()
    gui.show_only_lesion(1)
    gui.show_only_lesion(1)
    assert _shown_ids(gui) == [1]
    assert len(gui.jumps) == 2


def test_replaces_boxes_opened_by_double_click():
    gui, _ = _setup()
    gui.toggle_lesion_info(_napari_zyx((6, 7, 3)))      # lesion 2
    gui.toggle_lesion_info(_napari_zyx((21, 20, 22)))   # lesion 1
    assert sorted(_shown_ids(gui)) == [1, 2]
    gui.show_only_lesion(2)
    assert _shown_ids(gui) == [2]


def test_unknown_id_changes_nothing():
    gui, _ = _setup()
    gui.show_only_lesion(1)
    assert not gui.show_only_lesion(99)
    assert _shown_ids(gui) == [1]
    assert len(gui.jumps) == 1


def test_no_lesions():
    gui = _NoViewers()
    gui.set_lesions(None, SHAPE)
    assert not gui.show_only_lesion(1)
    assert gui.jumps == []
