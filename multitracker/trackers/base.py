"""The interface every tracker module implements."""

from typing import Protocol

import numpy as np

from multitracker.trackers.types import TrackerResult


class Tracker(Protocol):
    """Takes a BGR frame, returns detections in the common format."""

    name: str

    def process(self, frame: np.ndarray) -> TrackerResult:
        ...

    def close(self) -> None:
        ...
