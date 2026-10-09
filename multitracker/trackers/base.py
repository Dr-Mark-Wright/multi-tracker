"""The interface every tracker module implements."""

from typing import Protocol, runtime_checkable

import numpy as np

from multitracker.trackers.types import TrackerResult


class Tracker(Protocol):
    """Takes a BGR frame, returns detections in the common format."""

    name: str

    def process(self, frame: np.ndarray) -> TrackerResult:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class PromptableTracker(Tracker, Protocol):
    """A tracker whose classes are chosen by live text prompts."""

    prompts: list[str]

    def set_prompts(self, prompts: list[str]) -> None:
        ...
