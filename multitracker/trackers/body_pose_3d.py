"""3D body pose with Google MediaPipe Tasks Pose Landmarker (CPU on Windows).

Each body gets 33 landmarks twice over: 2D pixel positions for the overlay,
and 3D "world" positions in metres, centred between the hips. The 3D points
come straight from the model, so depth is an estimate from a single camera:
good for live demos, rough when limbs point at the camera.

Standalone usage::

    from multitracker.trackers.body_pose_3d import BodyPose3DTracker

    tracker = BodyPose3DTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    for body in result.detections:
        print(body.keypoints.shape)      # (33, 3): x, y pixels + visibility
        print(body.keypoints_3d.shape)   # (33, 3): x, y, z metres
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from multitracker.trackers.model_files import ensure_model
from multitracker.trackers.types import Detection, TrackerResult

_MODEL_ROOT = "https://storage.googleapis.com/mediapipe-models"
# "lite" is fastest, "heavy" most accurate; "full" sits in between.
MODEL_VARIANTS = ("lite", "full", "heavy")


def model_url(variant: str) -> str:
    return (f"{_MODEL_ROOT}/pose_landmarker/pose_landmarker_{variant}/"
            f"float16/latest/pose_landmarker_{variant}.task")


_vision = mp.tasks.vision

BODY_SKELETON = tuple((c.start, c.end)
                      for c in _vision.PoseLandmarksConnections.POSE_LANDMARKS)


def _pixel_keypoints(landmarks, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> ``(33, 3)`` x, y pixels and visibility."""
    return np.array([(p.x * width, p.y * height, p.visibility)
                     for p in landmarks], dtype=np.float32)


def _world_keypoints(landmarks) -> np.ndarray:
    """World landmarks -> ``(33, 3)`` x right, y down, z away, in metres."""
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)


def _bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = points[:, :2].min(axis=0)
    x2, y2 = points[:, :2].max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


class BodyPose3DTracker:
    """Detects bodies as 33 landmarks in 2D (pixels) and 3D (metres)."""

    name = "BodyPose3D"

    def __init__(self, variant: str = "full", num_poses: int = 2) -> None:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"variant must be one of {MODEL_VARIANTS}")
        options = _vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ensure_model(model_url(variant)))),
            running_mode=_vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self._landmarker = _vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """VIDEO mode needs strictly increasing timestamps (see
        DISCOVERY_LOG.md), so use a monotonic clock."""
        now = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(now, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def process(self, frame: np.ndarray) -> TrackerResult:
        start = time.perf_counter()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image,
                                                   self._next_timestamp_ms())
        detections = []
        for landmarks, world in zip(result.pose_landmarks,
                                    result.pose_world_landmarks):
            points = _pixel_keypoints(landmarks, width, height)
            detections.append(Detection(
                box=_bounding_box(points),
                score=float(np.mean(points[:, 2])),
                label="body 3D",
                keypoints=points,
                skeleton=BODY_SKELETON,
                keypoints_3d=_world_keypoints(world),
            ))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TrackerResult(detections=detections, inference_ms=elapsed_ms)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
