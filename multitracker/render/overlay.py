"""Overlay drawing for the common tracker output format."""

from __future__ import annotations

import cv2
import numpy as np

from multitracker.trackers.types import Detection, Edge, TrackerResult

BOX_COLOUR = (0, 200, 255)
BONE_COLOUR = (255, 180, 0)
JOINT_COLOUR = (0, 255, 0)
MASK_COLOUR = np.array((255, 0, 180), dtype=np.uint8)
TEXT_COLOUR = (255, 255, 255)
KEYPOINT_MIN_CONFIDENCE = 0.5


def _line_width(frame: np.ndarray) -> int:
    """Scale stroke width with frame size so overlays read at any resolution."""
    return max(1, round(min(frame.shape[:2]) / 300))


def draw_box(frame: np.ndarray, detection: Detection) -> None:
    if detection.box is None:
        return
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    width = _line_width(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, width)
    caption = f"{detection.label} {detection.score:.2f}".strip()
    cv2.putText(frame, caption, (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2 * width + 0.3, BOX_COLOUR, width)


def _visible(point: np.ndarray) -> bool:
    return point[2] >= KEYPOINT_MIN_CONFIDENCE


def _joint_radius(frame: np.ndarray, detection: Detection) -> int:
    """Size joints by how densely packed the keypoints are.

    A body (17 points in a large box) gets big joints, a hand small ones,
    and a face (478 points in a small box) radius 0, meaning single-pixel
    dots. No tracker-specific knowledge needed.
    """
    largest = _line_width(frame) + 2
    if detection.box is None:
        return largest
    x1, y1, x2, y2 = detection.box
    area_per_point = (x2 - x1) * (y2 - y1) / len(detection.keypoints)
    return int(np.clip(np.sqrt(area_per_point) / 5, 0, largest))


def _draw_joint(frame: np.ndarray, point: np.ndarray, radius: int) -> None:
    x, y = int(point[0]), int(point[1])
    if radius == 0:
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            frame[y, x] = JOINT_COLOUR
        return
    cv2.circle(frame, (x, y), radius, JOINT_COLOUR, -1, cv2.LINE_AA)


def draw_keypoints(frame: np.ndarray, detection: Detection) -> None:
    keypoints = detection.keypoints
    radius = _joint_radius(frame, detection)
    thickness = max(1, radius - 1)
    for start, end in detection.skeleton:
        a, b = keypoints[start], keypoints[end]
        if _visible(a) and _visible(b):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     BONE_COLOUR, thickness, cv2.LINE_AA)
    for point in keypoints:
        if _visible(point):
            _draw_joint(frame, point, radius)


def draw_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    """Tint masked pixels in place."""
    frame[mask] = (frame[mask] * 0.5 + MASK_COLOUR * 0.5).astype(np.uint8)


def draw_result(frame: np.ndarray, result: TrackerResult) -> np.ndarray:
    """Draw every detection in ``result`` onto a copy of ``frame``."""
    canvas = frame.copy()
    for detection in result.detections:
        if detection.mask is not None:
            draw_mask(canvas, detection.mask)
        draw_box(canvas, detection)
        if detection.keypoints is not None:
            draw_keypoints(canvas, detection)
    return canvas


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw status lines in the top-left corner, in place."""
    for row, text in enumerate(lines):
        origin = (10, 24 + row * 22)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    TEXT_COLOUR, 1, cv2.LINE_AA)
    return frame
