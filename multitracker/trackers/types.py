"""Common output format shared by every tracker and read by the renderer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Edge = tuple[int, int]


@dataclass
class Detection:
    """One detected object, person, hand or face.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels. ``keypoints`` is a ``(K, 3)``
    array of ``x, y, confidence`` in pixels, or ``None``. ``skeleton`` lists
    the keypoint index pairs to join with lines. ``mask`` is a boolean
    ``(H, W)`` array the size of the frame, or ``None``.
    ``keypoints_3d`` is a ``(K, 3)`` array of ``x, y, z`` in metres (x right,
    y down, z away from the camera), matching ``keypoints`` point for
    point, or ``None`` for 2D-only trackers.
    """

    box: tuple[float, float, float, float] | None = None
    score: float = 1.0
    label: str = ""
    keypoints: np.ndarray | None = None
    skeleton: tuple[Edge, ...] = ()
    mask: np.ndarray | None = None
    keypoints_3d: np.ndarray | None = None


@dataclass
class TrackerResult:
    """Everything a renderer needs to draw one frame's output."""

    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    # Lines for the viewer, e.g. "Cup: on coaster 2". Shown top-left.
    status: list[str] = field(default_factory=list)
    # A short announcement shown large for a moment, e.g. "Cup picked up".
    banner: str = ""
