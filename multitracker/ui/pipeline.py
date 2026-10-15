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
from multitracker.render import draw_banner, draw_hud, draw_result
from multitracker.trackers.registry import TrackerSwitcher
from multitracker.trackers.types import TrackerResult

log = logging.getLogger(__name__)

IDLE_WAIT_S = 0.05  # how long to wait when there is no frame to read
# A live source (webcam or stream) is "lost" after this many failed reads
# in a row; reconnects then start after RECONNECT_FIRST_S and back off,
# doubling up to RECONNECT_MAX_S, until it returns or another is chosen.
LOST_AFTER_FAILURES = 3
RECONNECT_FIRST_S = 1.0
RECONNECT_MAX_S = 8.0
LOST_REFRESH_S = 0.2  # redraw rate of the "signal lost" frame
ERROR_REPEAT_S = 5.0  # report a repeating tracker error at most this often


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

    If a webcam or stream stops delivering frames, the last frame stays on
    screen, dimmed, with a "signal lost" banner, while the thread keeps
    trying to reconnect with growing waits.
    """

    frame_ready = Signal(QImage, object, object)  # image, TrackerResult, FrameStats
    error = Signal(str)
    source_lost = Signal(str)       # source description
    source_restored = Signal(str)

    def __init__(self, source: VideoSource, switcher: TrackerSwitcher) -> None:
        super().__init__()
        self._source = source
        self._switcher = switcher
        self._meter = FrameRateMeter()
        self.last_frame: np.ndarray | None = None  # for warming up trackers
        self.show_boxes = True  # set from the UI thread; read once per frame
        self._failures = 0
        self._lost_spec = None
        self._next_retry = 0.0
        self._retry_wait = RECONNECT_FIRST_S
        self._attempts = 0
        self._last_error_at = -ERROR_REPEAT_S

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

    def _is_live(self) -> bool:
        spec = self._source.spec
        return spec is not None and spec.kind is not SourceKind.FILE

    def _note_failure(self) -> None:
        """Count failed reads; declare a live source lost after a few."""
        self._failures += 1
        if (self._lost_spec is None and self._is_live()
                and self._failures >= LOST_AFTER_FAILURES):
            self._lost_spec = self._source.spec
            self._retry_wait = RECONNECT_FIRST_S
            self._next_retry = time.monotonic() + self._retry_wait
            self._attempts = 0
            log.warning("Lost source %s; reconnecting", self._lost_spec)
            self.source_lost.emit(str(self._lost_spec.target))

    def _try_reconnect(self) -> None:
        spec = self._lost_spec
        if self._source.spec != spec:   # user picked another source
            self._lost_spec = None
            return
        if time.monotonic() < self._next_retry:
            return
        self._attempts += 1
        try:
            reopened = self._source.reopen(spec)
        except OSError:
            self._retry_wait = min(self._retry_wait * 2, RECONNECT_MAX_S)
            self._next_retry = time.monotonic() + self._retry_wait
            return
        self._lost_spec = None
        if reopened:
            log.info("Reconnected to %s", spec)
            self.source_restored.emit(str(spec.target))

    def _show_lost(self) -> None:
        """Re-show the last frame, dimmed, with a reconnecting banner."""
        if self.last_frame is None:
            return
        canvas = (self.last_frame * 0.4).astype(np.uint8)
        draw_banner(canvas, f"Signal lost - reconnecting ({self._attempts})")
        stats = FrameStats(self._switcher.active_name, 0, 0.0, 0.0)
        self.frame_ready.emit(to_qimage(canvas), TrackerResult(), stats)

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
            now = time.monotonic()
            if now - self._last_error_at >= ERROR_REPEAT_S:
                self._last_error_at = now
                log.exception("Tracker failed on a frame")
                self.error.emit(f"{self._switcher.active_name} failed: {exc}")
            return TrackerResult()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            started = time.perf_counter()
            if self._lost_spec is not None:
                self._try_reconnect()
                if self._lost_spec is not None:
                    self._show_lost()
                    time.sleep(LOST_REFRESH_S)
                    continue
            frame = self._next_frame()
            if frame is None:
                self._note_failure()
                time.sleep(IDLE_WAIT_S)
                continue
            self._failures = 0
            self.last_frame = frame
            result = self._track(frame)
            canvas = draw_hud(draw_result(frame, result, inset_3d=False,
                                          show_boxes=self.show_boxes),
                              result.status)
            draw_banner(canvas, result.banner)
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
