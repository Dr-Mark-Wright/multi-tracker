"""Open-vocabulary detection and segmentation with Ultralytics YOLOE-26.

Type any class names as text prompts; they take effect on the next frame.

Standalone usage::

    from multitracker.trackers.yoloe import YoloeTracker

    tracker = YoloeTracker(prompts=["bottle", "cup"])
    result = tracker.process(frame)       # frame: BGR numpy array
    tracker.set_prompts(["phone"])        # safe to call from another thread
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

# Never let Ultralytics pip-install packages in the middle of a demo; a
# missing dependency should fail loudly at startup instead.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics import YOLOE  # noqa: E402  (must follow the env setting)

from multitracker.device import select_device
from multitracker.trackers.types import Detection, TrackerResult

log = logging.getLogger(__name__)

# Large model: on the coaster test the small one found 0.1% of coasters,
# large 96% ("round coaster"), at ~10 ms/frame on an RTX 5090. On weaker
# GPUs or laptops pass model="yoloe-26s-seg.pt" for speed.
DEFAULT_MODEL = "yoloe-26l-seg.pt"
DEFAULT_PROMPTS = ("person",)


def clean_prompts(prompts: list[str]) -> list[str]:
    """Strip blanks and duplicates, keeping the order typed."""
    seen: dict[str, None] = {}
    for prompt in prompts:
        prompt = prompt.strip().lower()
        if prompt:
            seen.setdefault(prompt)
    return list(seen)


def _to_detections(prediction) -> list[Detection]:
    """Convert one Ultralytics segmentation prediction to detections."""
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    masks = (prediction.masks.data.cpu().numpy() > 0.5
             if prediction.masks is not None else [None] * len(xyxy))
    return [
        Detection(box=tuple(box), score=float(score),
                  label=prediction.names[cls], mask=mask)
        for box, score, cls, mask in zip(xyxy, scores, classes, masks)
    ]


class YoloeTracker:
    """Finds and segments whatever classes the text prompts name."""

    name = "YOLOE-26"

    def __init__(self, model: str = DEFAULT_MODEL,
                 prompts: list[str] | None = None,
                 confidence: float = 0.25, device: str | None = None) -> None:
        self._device = device or select_device()
        self._model = YOLOE(model)
        self._confidence = confidence
        self._pending: list[str] | None = clean_prompts(
            prompts or list(DEFAULT_PROMPTS))
        self._lock = threading.Lock()
        self.prompts: list[str] = []

    def set_prompts(self, prompts: list[str]) -> None:
        """Queue new class prompts; applied before the next frame."""
        cleaned = clean_prompts(prompts)
        if cleaned:
            with self._lock:
                self._pending = cleaned

    def _apply_pending_prompts(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None and pending != self.prompts:
            self._model.set_classes(pending)
            self.prompts = pending
            log.info("YOLOE-26 prompts: %s", ", ".join(pending))

    def process(self, frame: np.ndarray) -> TrackerResult:
        self._apply_pending_prompts()
        prediction = self._model.predict(
            frame, conf=self._confidence, device=self._device,
            retina_masks=True, verbose=False,
        )[0]
        return TrackerResult(
            detections=_to_detections(prediction),
            inference_ms=float(prediction.speed["inference"]),
        )

    def close(self) -> None:
        self._model = None
