"""Background thread that runs capture -> tracker -> overlay for the UI.

The window never touches frames or trackers directly: it changes the
source and tracker through ``VideoSource`` and ``TrackerSwitcher`` (both
thread-safe) and receives finished frames through the ``frame_ready``
signal.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from multitracker.capture import SourceKind, VideoSource
from multitracker.render import draw_result
from multitracker.trackers.registry import TrackerSwitcher
from multitracker.trackers.types import TrackerResult

log = logging.getLogger(__name__)

IDLE_WAIT_S = 0.05  # how long to wait when there is no frame to read


@dataclass
class FrameStats:
    """Numbers shown in the status panel for one frame."""

    tracker: str
    detections: int
    inference_ms: float
    fps: float


class FrameRateMeter:
    """Exponentially smoothed frames-per-second counter."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        instant = 1.0 / max(now - self._last, 1e-6)
        self._last = now
        self.fps = self._smoothing * self.fps + (1 - self._smoothing) * instant
        return self.fps


def to_qimage(frame: np.ndarray) -> QImage:
    """BGR numpy frame -> QImage that owns its own pixel data."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width,
                  QImage.Format.Format_RGB888).copy()


class PipelineThread(QThread):
    """Reads, tracks and draws frames until asked to stop.

    Files loop when they end and play at their own frame rate; webcams and
    streams run as fast as they deliver frames. A tracker error is logged
    and the raw frame shown, so one bad frame never stops the demo.
    """

    frame_ready = Signal(QImage, object, object)  # image, TrackerResult, FrameStats
    error = Signal(str)

    def __init__(self, source: VideoSource, switcher: TrackerSwitcher) -> None:
        super().__init__()
        self._source = source
        self._switcher = switcher
        self._meter = FrameRateMeter()
        self.last_frame: np.ndarray | None = None  # for warming up trackers

    def _next_frame(self) -> np.ndarray | None:
        ok, frame = self._source.read()
        spec = self._source.spec
        if not ok and spec is not None and spec.kind is SourceKind.FILE:
            try:
                self._source.open(spec)  # loop the file
            except OSError:
                return None
            ok, frame = self._source.read()
        return frame if ok else None

    def _frame_interval(self) -> float:
        """Seconds per frame for files, 0 for live sources."""
        spec = self._source.spec
        if spec is None or spec.kind is not SourceKind.FILE:
            return 0.0
        return 1.0 / (self._source.fps() or 30.0)

    def _track(self, frame: np.ndarray) -> TrackerResult:
        try:
            return self._switcher.process(frame)
        except Exception as exc:  # keep the demo running whatever happens
            log.exception("Tracker failed on a frame")
            self.error.emit(f"{self._switcher.active_name} failed: {exc}")
            return TrackerResult()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            started = time.perf_counter()
            frame = self._next_frame()
            if frame is None:
                time.sleep(IDLE_WAIT_S)
                continue
            self.last_frame = frame
            result = self._track(frame)
            canvas = draw_result(frame, result, inset_3d=False)
            stats = FrameStats(self._switcher.active_name,
                               len(result.detections), result.inference_ms,
                               self._meter.tick())
            self.frame_ready.emit(to_qimage(canvas), result, stats)
            spare = self._frame_interval() - (time.perf_counter() - started)
            if spare > 0:
                time.sleep(spare)

    def stop(self) -> None:
        self.requestInterruption()
        self.wait()
