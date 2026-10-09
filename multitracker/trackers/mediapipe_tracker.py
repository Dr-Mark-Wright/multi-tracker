"""Hand and face landmarks with Google MediaPipe Tasks (CPU on Windows).

Standalone usage::

    from multitracker.trackers.mediapipe_tracker import MediaPipeTracker

    tracker = MediaPipeTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    for item in result.detections:
        print(item.label, item.keypoints.shape)   # hands (21, 3), face (478, 3)
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from multitracker.trackers.model_files import ensure_model
from multitracker.trackers.types import Detection, TrackerResult

_MODEL_ROOT = "https://storage.googleapis.com/mediapipe-models"
HAND_MODEL_URL = (f"{_MODEL_ROOT}/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")
FACE_MODEL_URL = (f"{_MODEL_ROOT}/face_landmarker/face_landmarker/"
                  "float16/latest/face_landmarker.task")

_vision = mp.tasks.vision


def _edges(connections) -> tuple[tuple[int, int], ...]:
    return tuple((c.start, c.end) for c in connections)


HAND_SKELETON = _edges(_vision.HandLandmarksConnections.HAND_CONNECTIONS)
FACE_SKELETON = _edges(_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)

# The subject's own left/right. MediaPipe's docs describe handedness for
# mirrored (selfie) input, but on unmirrored OpenCV frames from mediapipe
# 1.1 the raw labels were already correct (see DISCOVERY_LOG.md).
_HAND_LABELS = {"Left": "left hand", "Right": "right hand"}


def _to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> ``(K, 3)`` array of x, y, confidence."""
    points = np.array([(p.x * width, p.y * height, 1.0) for p in landmarks],
                      dtype=np.float32)
    return points


def _bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = points[:, :2].min(axis=0)
    x2, y2 = points[:, :2].max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _landmark_detection(points: np.ndarray, label: str, score: float,
                        skeleton: tuple[tuple[int, int], ...]) -> Detection:
    return Detection(box=_bounding_box(points), score=score, label=label,
                     keypoints=points, skeleton=skeleton)


def _hand_detections(result, width: int, height: int) -> list[Detection]:
    detections = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        category = handedness[0]
        label = _HAND_LABELS.get(category.category_name, "hand")
        points = _to_pixels(landmarks, width, height)
        detections.append(_landmark_detection(points, label, category.score,
                                              HAND_SKELETON))
    return detections


def _face_detections(result, width: int, height: int) -> list[Detection]:
    return [
        _landmark_detection(_to_pixels(landmarks, width, height), "face", 1.0,
                            FACE_SKELETON)
        for landmarks in result.face_landmarks
    ]


def _create_hand_landmarker(num_hands: int):
    options = _vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ensure_model(HAND_MODEL_URL))),
        running_mode=_vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return _vision.HandLandmarker.create_from_options(options)


def _create_face_landmarker(num_faces: int):
    # IMAGE mode: each tile is an independent image, so VIDEO-mode
    # frame-to-frame tracking would be confused by tiles.
    options = _vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ensure_model(FACE_MODEL_URL))),
        running_mode=_vision.RunningMode.IMAGE,
        num_faces=num_faces,
    )
    return _vision.FaceLandmarker.create_from_options(options)


def square_tiles(height: int, width: int) -> list[tuple[int, int, int]]:
    """Overlapping square tiles ``(top, left, size)`` covering the frame.

    MediaPipe's face detector squeezes the whole image into a small square,
    so in a tall or wide frame a distant face shrinks below what it can
    find. Square tiles keep the face at its native scale.
    """
    size = min(height, width)
    long_side = max(height, width)
    count = -(-long_side // size)  # ceiling division
    if count == 1:
        return [(0, 0, size)]
    step = (long_side - size) / (count - 1)
    offsets = [round(i * step) for i in range(count)]
    if height > width:
        return [(offset, 0, size) for offset in offsets]
    return [(0, offset, size) for offset in offsets]


def _rgb_image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb))


def _overlaps(a: Detection, b: Detection, threshold: float = 0.5) -> bool:
    """True if two boxes overlap by more than ``threshold`` of the smaller."""
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return smaller > 0 and inter_w * inter_h / smaller > threshold


def _unique(detections: list[Detection]) -> list[Detection]:
    """Drop faces found twice where tiles overlap."""
    kept: list[Detection] = []
    for detection in detections:
        if not any(_overlaps(detection, other) for other in kept):
            kept.append(detection)
    return kept


def _faces_in_tile(landmarker, rgb: np.ndarray,
                   tile: tuple[int, int, int]) -> list[Detection]:
    top, left, size = tile
    crop = rgb[top:top + size, left:left + size]
    result = landmarker.detect(_rgb_image(crop))
    detections = _face_detections(result, size, size)
    for detection in detections:
        detection.keypoints[:, 0] += left
        detection.keypoints[:, 1] += top
        detection.box = _bounding_box(detection.keypoints)
    return detections


class MediaPipeTracker:
    """Detects hand (21 points) and face (478 points) landmarks."""

    name = "MediaPipe"

    def __init__(self, num_hands: int = 2, num_faces: int = 1,
                 hands: bool = True, face: bool = True) -> None:
        self._hands = _create_hand_landmarker(num_hands) if hands else None
        self._face = _create_face_landmarker(num_faces) if face else None
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """VIDEO mode needs strictly increasing timestamps, even across
        file loops and source switches, so use a monotonic clock."""
        now = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(now, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def detect_hands(self, rgb: np.ndarray) -> list[Detection]:
        if self._hands is None:
            return []
        height, width = rgb.shape[:2]
        result = self._hands.detect_for_video(_rgb_image(rgb),
                                              self._next_timestamp_ms())
        return _hand_detections(result, width, height)

    def detect_faces(self, rgb: np.ndarray) -> list[Detection]:
        if self._face is None:
            return []
        faces = []
        for tile in square_tiles(*rgb.shape[:2]):
            faces += _faces_in_tile(self._face, rgb, tile)
        return _unique(faces)

    def process(self, frame: np.ndarray) -> TrackerResult:
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detect_hands(rgb) + self.detect_faces(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TrackerResult(detections=detections, inference_ms=elapsed_ms)

    def close(self) -> None:
        for landmarker in (self._hands, self._face):
            if landmarker is not None:
                landmarker.close()
        self._hands = self._face = None
