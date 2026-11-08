"""Lesion id colours, shared by the tumor Labels layers, the mask overlay, the
Segmentation tab's label toggles and the lesion table.

The colormap equals napari's default Labels colormap (49 colours, seed 0.5),
so lesion 1 keeps the colour binary masks always had. Ids cycle every 49.
"""

import numpy as np
from PyQt6.QtGui import QColor
from napari.utils.colormaps import label_colormap

TUMOR_LABEL_COLORMAP = label_colormap(49, 0.5, background_value=0)


def label_rgba(label_id: int) -> np.ndarray:
    """RGBA floats (0..1) of lesion ``label_id``."""
    return np.asarray(TUMOR_LABEL_COLORMAP.map(np.array([label_id])), dtype=float)[0]


def label_qcolor(label_id: int, alpha: float = 1.0) -> QColor:
    r, g, b, _ = label_rgba(label_id)
    return QColor.fromRgbF(float(r), float(g), float(b), float(alpha))


def text_color_for(label_id: int) -> str:
    """"black" or "white", whichever reads better on the lesion's colour."""
    r, g, b, _ = label_rgba(label_id)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
