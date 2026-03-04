"""Viewer synchronization helpers.

Provides functions to link dimensions (slice position) and camera
(zoom/pan) between two Napari viewers with re-entrancy guards to
prevent cascading event storms.
"""

import numpy as np


def link_dims(v1, v2):
    """Link the current_step of two viewers (same-row pair sync).

    Uses a re-entrancy guard so that v1 → v2 update does NOT trigger
    v2 → v1 back, eliminating excessive event cascades.
    """
    _state = {"syncing": False}

    def sync_v1_to_v2(event):
        if _state["syncing"]:
            return
        _state["syncing"] = True
        try:
            step = v1.dims.current_step
            if v2.dims.current_step != step:
                v2.dims.current_step = step
        finally:
            _state["syncing"] = False

    def sync_v2_to_v1(event):
        if _state["syncing"]:
            return
        _state["syncing"] = True
        try:
            step = v2.dims.current_step
            if v1.dims.current_step != step:
                v1.dims.current_step = step
        finally:
            _state["syncing"] = False

    v1.dims.events.current_step.connect(sync_v1_to_v2)
    v2.dims.events.current_step.connect(sync_v2_to_v1)


def link_camera(v1, v2):
    """Link camera zoom and center between two viewers with re-entrancy guard."""
    _state = {"syncing": False}

    def sync_cam_v1_to_v2(event):
        if _state["syncing"]:
            return
        _state["syncing"] = True
        try:
            if abs(v2.camera.zoom - v1.camera.zoom) > 1e-6:
                v2.camera.zoom = v1.camera.zoom
            c1 = v1.camera.center
            c2 = v2.camera.center
            if any(abs(a - b) > 1e-6 for a, b in zip(c1, c2)):
                v2.camera.center = c1
        finally:
            _state["syncing"] = False

    def sync_cam_v2_to_v1(event):
        if _state["syncing"]:
            return
        _state["syncing"] = True
        try:
            if abs(v1.camera.zoom - v2.camera.zoom) > 1e-6:
                v1.camera.zoom = v2.camera.zoom
            c1 = v1.camera.center
            c2 = v2.camera.center
            if any(abs(a - b) > 1e-6 for a, b in zip(c1, c2)):
                v1.camera.center = c2
        finally:
            _state["syncing"] = False

    v1.camera.events.zoom.connect(sync_cam_v1_to_v2)
    v1.camera.events.center.connect(sync_cam_v1_to_v2)

    v2.camera.events.zoom.connect(sync_cam_v2_to_v1)
    v2.camera.events.center.connect(sync_cam_v2_to_v1)


def one_shot_sync_step(source_viewer, target_viewers):
    """Copy current_step from source to all targets (no persistent link)."""
    step = source_viewer.dims.current_step
    for v in target_viewers:
        if v is not source_viewer and v.dims.current_step != step:
            v.dims.current_step = step
