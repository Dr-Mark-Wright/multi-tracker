"""Full-body 2D keypoints with Ultralytics YOLO pose.

Standalone usage::

    from multitracker.trackers.yolo_pose import YoloPoseTracker

    tracker = YoloPoseTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    for person in result.detections:
        print(person.box, person.keypoints.shape)   # (17, 3)
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from multitracker.device import select_device
from multitracker.trackers.types import Detection, TrackerResult

DEFAULT_MODEL = "yolo26s-pose.pt"

# COCO 17-keypoint skeleton: face, arms, torso, legs.
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _to_detections(prediction) -> list[Detection]:
    """Convert one Ultralytics pose prediction into common detections."""
    if prediction.keypoints is None or prediction.boxes is None:
        return []
    boxes = prediction.boxes.xyxy.cpu().numpy()
    scores = prediction.boxes.conf.cpu().numpy()
    keypoints = prediction.keypoints.data.cpu().numpy()
    return [
        Detection(box=tuple(box), score=float(score), label="person",
                  keypoints=points)
        for box, score, points in zip(boxes, scores, keypoints)
    ]


class YoloPoseTracker:
    """Detects people and their 17 COCO body keypoints."""

    name = "YOLO pose"

    def __init__(self, model: str = DEFAULT_MODEL, confidence: float = 0.4,
                 device: str | None = None) -> None:
        self._device = device or select_device()
        self._model = YOLO(model)
        self._confidence = confidence

    def process(self, frame: np.ndarray) -> TrackerResult:
        prediction = self._model.predict(
            frame, conf=self._confidence, device=self._device, verbose=False,
        )[0]
        return TrackerResult(
            detections=_to_detections(prediction),
            skeleton=COCO_SKELETON,
            inference_ms=float(prediction.speed["inference"]),
        )

    def close(self) -> None:
        self._model = None
