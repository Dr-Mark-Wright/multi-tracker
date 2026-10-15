"""Cup and bottle pick-up / put-down tracking: YOLOE-26 + YOLO pose.

YOLOE-26 finds the cup, bottle and coasters; YOLO pose finds the right
wrist; ``TableObjectAnalyser`` turns both into states and events. The
status (e.g. "Cup: on coaster 2") is returned in ``TrackerResult.status``.

Standalone usage::

    from multitracker.trackers.table_objects import TableObjectsTracker

    tracker = TableObjectsTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    print(result.status)              # ["Cup: in right hand", ...]
    print(tracker.analyser.events)    # every pick-up and put-down so far
"""

from __future__ import annotations

import time

import numpy as np

from multitracker.behaviour import TableObjectAnalyser
from multitracker.trackers.types import TrackerResult
from multitracker.trackers.yolo_pose import YoloPoseTracker
from multitracker.trackers.yoloe import YoloeTracker

OBJECTS = ("cup", "bottle")
# "round coaster" found 96% of coasters with the large model; "coaster"
# alone found far fewer (see DISCOVERY_LOG.md, milestone 4).
PROMPTS = ("bottle", "cup", "round coaster")
BANNER_SECONDS = 2.0        # how long "Cup picked up" stays on screen
RIGHT_WRIST = 10            # COCO keypoint index (the subject's right)
WRIST_MIN_CONFIDENCE = 0.5


def right_wrist(result: TrackerResult) -> tuple[float, float] | None:
    """Right wrist of the most confident person, if it is visible."""
    people = [d for d in result.detections if d.keypoints is not None]
    if not people:
        return None
    x, y, confidence = max(people, key=lambda d: d.score).keypoints[RIGHT_WRIST]
    return (float(x), float(y)) if confidence >= WRIST_MIN_CONFIDENCE else None


class TableObjectsTracker:
    """Reports when the cup and bottle are picked up and put down."""

    name = "Cup & bottle"

    def __init__(self) -> None:
        self._objects = YoloeTracker(prompts=list(PROMPTS))
        self._pose = YoloPoseTracker()
        self.analyser = TableObjectAnalyser(list(OBJECTS))
        self._banner, self._banner_until = "", 0.0

    def process(self, frame: np.ndarray) -> TrackerResult:
        objects = self._objects.process(frame)
        pose = self._pose.process(frame)
        events = self.analyser.update(objects.detections, right_wrist(pose))
        now = time.monotonic()
        if events:
            self._banner = " / ".join(e.headline for e in events)
            self._banner_until = now + BANNER_SECONDS
        return TrackerResult(
            detections=objects.detections + pose.detections,
            inference_ms=objects.inference_ms + pose.inference_ms,
            status=self.analyser.status_lines(),
            banner=self._banner if now < self._banner_until else "",
        )

    def close(self) -> None:
        self._objects.close()
        self._pose.close()
