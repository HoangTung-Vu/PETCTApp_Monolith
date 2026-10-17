"""Outline extraction behind the smoothed / contour tumor mask display.

Run: .venv/bin/pytest tests/test_mask_overlay.py -q
"""

import numpy as np

from src.gui.components.viewers.mask_overlay import mask_outlines


def _area(poly):
    y, x = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_empty_plane_has_no_outline():
    assert mask_outlines(np.zeros((16, 16), np.uint8), smooth=True) == []


def test_voxel_outline_follows_voxel_edges():
    plane = np.zeros((16, 16), np.uint8)
    plane[5:8, 6:10] = 1
    (outline,) = mask_outlines(plane, smooth=False)
    np.testing.assert_allclose(outline.min(axis=0), (4.5, 5.5))
    np.testing.assert_allclose(outline.max(axis=0), (7.5, 9.5))
    assert abs(_area(outline) - 12) < 0.2              # only the corners are chamfered


def test_hole_and_separate_lesions_give_separate_outlines():
    yy, xx = np.ogrid[:40, :40]
    plane = np.zeros((40, 40), np.uint8)
    plane[(yy - 15) ** 2 + (xx - 15) ** 2 <= 10 ** 2] = 1
    plane[(yy - 15) ** 2 + (xx - 15) ** 2 <= 3 ** 2] = 0
    plane[35, 35] = 1                                     # single voxel lesion
    for smooth in (False, True):
        outlines = mask_outlines(plane, smooth)
        assert len(outlines) == 3
        small = min(outlines, key=_area)
        np.testing.assert_allclose(small.mean(axis=0), (35, 35), atol=1e-6)


def test_smooth_outline_keeps_the_lesion_size():
    yy, xx = np.ogrid[:64, :64]
    for r in (3, 10, 25):
        plane = ((yy - 32) ** 2 + (xx - 32) ** 2 <= r * r).astype(np.uint8)
        (outline,) = mask_outlines(plane, smooth=True)
        assert abs(_area(outline) / plane.sum() - 1) < 0.12, r
