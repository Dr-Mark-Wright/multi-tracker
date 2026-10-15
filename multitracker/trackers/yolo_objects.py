"""General object tracking: YOLO26's 80 everyday classes, with IDs and masks.

No prompts needed. Finds people, screens ("tv"), laptops, keyboards,
phones, chairs, cups and the rest of the 80 COCO classes, and ByteTrack
gives each one an ID that stays with it from frame to frame ("person 3").
The segmentation model also outlines each object with a mask.

Standalone usage::

    from multitracker.trackers.yolo_objects import YoloObjectsTracker

    tracker = YoloObjectsTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    for item in result.detections:
        print(item.label, item.box)   # e.g. "person 3", (x1, y1, x2, y2)
"""

from __future__ import annotations

import os

import numpy as np

# Never let Ultralytics pip-install packages mid-demo (ByteTrack needs
# `lap`, which is in requirements.txt).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics import YOLO  # noqa: E402  (must follow the env setting)

from multitracker.device import select_device
from multitracker.trackers.types import Detection, TrackerResult

# Medium segmentation model: a good balance of accuracy and speed on a
# desktop GPU. On laptops pass model="yolo26n-seg.pt" or "yolo26s-seg.pt".
DEFAULT_MODEL = "yolo26m-seg.pt"
# COCO calls monitors and TVs "tv"; show the word people expect.
LABEL_NAMES = {"tv": "screen"}


def _to_detections(prediction) -> list[Detection]:
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = (boxes.id.cpu().numpy().astype(int) if boxes.id is not None
           else [None] * len(xyxy))
    masks = (prediction.masks.data.cpu().numpy() > 0.5
             if prediction.masks is not None else [None] * len(xyxy))
    detections = []
    for box, score, cls, track_id, mask in zip(xyxy, scores, classes, ids,
                                               masks):
        name = prediction.names[cls]
        name = LABEL_NAMES.get(name, name)
        label = f"{name} {track_id}" if track_id is not None else name
        detections.append(Detection(box=tuple(box), score=float(score),
                                    label=label, mask=mask))
    return detections


class YoloObjectsTracker:
    """Detects and tracks 80 everyday object types, with stable IDs."""

    name = "YOLO objects"

    def __init__(self, model: str = DEFAULT_MODEL, confidence: float = 0.35,
                 device: str | None = None) -> None:
        self._device = device or select_device()
        self._model = YOLO(model)
        self._confidence = confidence

    def process(self, frame: np.ndarray) -> TrackerResult:
        prediction = self._model.track(
            frame, conf=self._confidence, device=self._device,
            persist=True, tracker="bytetrack.yaml", retina_masks=True,
            verbose=False,
        )[0]
        return TrackerResult(
            detections=_to_detections(prediction),
            inference_ms=float(prediction.speed["inference"]),
        )

    def close(self) -> None:
        self._model = None
