"""Anatomical description of the 3D camera (orientation readout of the 3D viewer).

Napari's 3D scene axes are the ``to_napari`` axes of the app's LAS volumes:
(z, y, x) point to Inferior, Posterior and patient Left. A scene vector
(vz, vy, vx) is therefore (-vx, -vy, -vz) in patient RAS coordinates
(+x = Right, +y = Anterior, +z = Superior).

Angles are measured from the anterior coronal view (camera in front of the
patient, head up), the view the 3D mode opens with:

* azimuth   — rotation of the camera position about the patient's S–I axis:
              0° = anterior, +90° = left lateral, −90° = right lateral, ±180° = posterior
* elevation — camera position above (+, toward the head) or below (−) the axial plane
* roll      — in-plane rotation of the displayed patient, + = clockwise on screen
              (0° when the patient's head is straight up; for views from the head
              or the feet, 0° when anterior is up)
"""

import math
from dataclasses import dataclass

import numpy as np

_AXIS_LETTERS = (("R", "L"), ("A", "P"), ("S", "I"))   # (+, −) per RAS axis
_VIEW_NAMES = {
    "A": "Anterior", "P": "Posterior",
    "L": "Left lateral", "R": "Right lateral",
    "S": "Superior", "I": "Inferior",
}


def scene_to_ras(v) -> np.ndarray:
    """Napari scene direction (z, y, x) → unit vector in patient RAS."""
    vz, vy, vx = (float(c) for c in v)
    ras = np.array([-vx, -vy, -vz])
    return ras / np.linalg.norm(ras)


def direction_letters(v_ras, min_component: float = 0.3) -> str:
    """Anatomical letters for a direction, strongest first (e.g. "L", "LA").

    An axis is named when its share of the unit vector is ≥ ``min_component``
    (≈ 17° off the plane perpendicular to it); the strongest axis always is.
    """
    v = np.asarray(v_ras, dtype=float)
    v = v / np.linalg.norm(v)
    order = np.argsort(-np.abs(v))
    letters = []
    for i, axis in enumerate(order):
        if i == 0 or abs(v[axis]) >= min_component:
            letters.append(_AXIS_LETTERS[axis][0 if v[axis] > 0 else 1])
    return "".join(letters)


@dataclass
class CameraOrientation:
    view_from: str         # nearest standard view, e.g. "Anterior"
    off_axis: float        # degrees between the camera position and that view
    azimuth: float         # degrees
    elevation: float       # degrees
    roll: float            # degrees, + = patient appears rotated clockwise
    look_ras: np.ndarray   # unit view direction (camera → scene) in RAS
    up: str                # anatomical letters at each screen edge
    down: str
    left: str
    right: str


def _signed_angle(a, b, axis) -> float:
    """Angle (deg) rotating ``a`` onto ``b`` about ``axis`` (right-hand rule)."""
    return math.degrees(math.atan2(float(np.dot(np.cross(a, b), axis)), float(np.dot(a, b))))


def describe_camera(view_direction, up_direction) -> CameraOrientation:
    """Describe napari's ``camera.view_direction`` / ``up_direction`` anatomically."""
    look = scene_to_ras(view_direction)
    up = scene_to_ras(up_direction)
    up = up - np.dot(up, look) * look                 # napari keeps them orthogonal; be safe
    up /= np.linalg.norm(up)
    right = np.cross(look, up)                        # screen right (right-handed RAS)

    pos = -look                                       # where the camera sits, seen from the patient
    azimuth = math.degrees(math.atan2(-pos[0], pos[1]))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, pos[2]))))

    axis = int(np.argmax(np.abs(pos)))
    from_letter = _AXIS_LETTERS[axis][0 if pos[axis] > 0 else 1]
    off_axis = math.degrees(math.acos(min(1.0, abs(pos[axis]))))

    # Reference "up": the head, or anterior when looking along the S–I axis.
    ref = np.array([0.0, 0.0, 1.0]) if abs(look[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    ref = ref - np.dot(ref, look) * look
    ref /= np.linalg.norm(ref)
    # The image turns opposite to the camera: rolling the camera up-vector
    # toward the screen's left makes the patient appear rotated clockwise.
    roll = _signed_angle(up, ref, look)

    return CameraOrientation(
        view_from=_VIEW_NAMES[from_letter],
        off_axis=off_axis,
        azimuth=azimuth,
        elevation=elevation,
        roll=roll,
        look_ras=look,
        up=direction_letters(up),
        down=direction_letters(-up),
        left=direction_letters(-right),
        right=direction_letters(right),
    )
