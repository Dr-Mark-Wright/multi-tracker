"""Name -> tracker factory, with imports deferred until a tracker is chosen.

Deferring imports means using one tracker never loads the others'
dependencies (e.g. YOLO pose does not need MediaPipe installed).
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from multitracker.trackers.base import PromptableTracker, Tracker
from multitracker.trackers.types import TrackerResult


def _yolo_pose() -> Tracker:
    from multitracker.trackers.yolo_pose import YoloPoseTracker
    return YoloPoseTracker()


def _mediapipe() -> Tracker:
    from multitracker.trackers.mediapipe_tracker import MediaPipeTracker
    return MediaPipeTracker()


def _yoloe() -> Tracker:
    from multitracker.trackers.yoloe import YoloeTracker
    return YoloeTracker()


def _body_pose_3d() -> Tracker:
    from multitracker.trackers.body_pose_3d import BodyPose3DTracker
    return BodyPose3DTracker()


TRACKERS: dict[str, Callable[[], Tracker]] = {
    "YOLOE-26": _yoloe,
    "YOLO pose": _yolo_pose,
    "MediaPipe": _mediapipe,
    "BodyPose3D": _body_pose_3d,
}


class TrackerSwitcher:
    """Holds the active tracker and swaps it on request.

    Trackers are created once and cached, so switching back is instant.
    ``switch`` may be called from a UI thread while ``process`` runs in the
    capture loop.
    """

    def __init__(self, factories: dict[str, Callable[[], Tracker]] = TRACKERS
                 ) -> None:
        self._factories = factories
        self._loaded: dict[str, Tracker] = {}
        self._active: Tracker | None = None
        self._prompts: list[str] = []
        self._warmed: set[str] = set()
        self._lock = threading.Lock()

    @property
    def names(self) -> list[str]:
        return list(self._factories)

    @property
    def active_name(self) -> str:
        return self._active.name if self._active else "none"

    def _load(self, name: str) -> Tracker:
        if name not in self._loaded:
            self._loaded[name] = self._factories[name]()
        return self._loaded[name]

    @property
    def prompts(self) -> list[str]:
        """Prompts of the active tracker, or [] if it takes none."""
        tracker = self._active
        if isinstance(tracker, PromptableTracker):
            return tracker.prompts
        return []

    @property
    def accepts_prompts(self) -> bool:
        """True if the active tracker takes live text prompts."""
        return isinstance(self._active, PromptableTracker)

    def switch(self, name: str) -> None:
        tracker = self._load(name)
        if self._prompts and isinstance(tracker, PromptableTracker):
            tracker.set_prompts(self._prompts)
        with self._lock:
            self._active = tracker

    def warm_up(self, name: str, frame: np.ndarray) -> None:
        """Load a tracker and run it once, without making it active.

        A GPU model's first frames can take seconds (CUDA start-up), so a
        UI calls this off its main thread before ``switch`` to keep the
        current tracker running meanwhile. Only the first call does work.
        """
        if name in self._warmed:
            return
        tracker = self._load(name)
        if self._prompts and isinstance(tracker, PromptableTracker):
            tracker.set_prompts(self._prompts)
        tracker.process(frame)
        self._warmed.add(name)

    def set_prompts(self, prompts: list[str]) -> None:
        """Remember prompts and pass them to every promptable tracker."""
        self._prompts = list(prompts)
        for tracker in list(self._loaded.values()):
            if isinstance(tracker, PromptableTracker):
                tracker.set_prompts(self._prompts)

    def process(self, frame: np.ndarray) -> TrackerResult:
        with self._lock:
            tracker = self._active
            if tracker is None:
                return TrackerResult()
            return tracker.process(frame)

    def close(self) -> None:
        with self._lock:
            for tracker in self._loaded.values():
                tracker.close()
            self._loaded.clear()
            self._active = None
