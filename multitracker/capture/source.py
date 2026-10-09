"""Source description and a switchable wrapper around ``cv2.VideoCapture``."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


class SourceKind(Enum):
    """The three supported input source types."""

    WEBCAM = "webcam"
    FILE = "file"
    STREAM = "stream"


_STREAM_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://")

# FFmpeg's default open timeout is 30 s, which freezes a live demo when a
# stream URL is wrong. Fail fast instead.
STREAM_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class SourceSpec:
    """Describes which source to open, independent of how it is opened."""

    kind: SourceKind
    target: int | str

    @classmethod
    def webcam(cls, index: int) -> SourceSpec:
        return cls(SourceKind.WEBCAM, index)

    @classmethod
    def file(cls, path: str | Path) -> SourceSpec:
        return cls(SourceKind.FILE, str(path))

    @classmethod
    def stream(cls, url: str) -> SourceSpec:
        return cls(SourceKind.STREAM, url)

    @classmethod
    def parse(cls, text: str) -> SourceSpec:
        """Infer the source kind from a device index, URL or file path."""
        text = text.strip()
        if text.isdigit():
            return cls.webcam(int(text))
        if text.lower().startswith(_STREAM_PREFIXES):
            return cls.stream(text)
        return cls.file(text)

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.target}"


def _backend_for(kind: SourceKind) -> int:
    """Pick the OpenCV backend that opens each source kind most reliably."""
    if kind is SourceKind.WEBCAM:
        # DirectShow opens webcams far faster than MSMF on Windows.
        return cv2.CAP_DSHOW
    if kind is SourceKind.STREAM:
        return cv2.CAP_FFMPEG
    return cv2.CAP_ANY


def _open_params(kind: SourceKind) -> list[int]:
    """Extra ``VideoCapture`` open parameters for each source kind."""
    if kind is SourceKind.STREAM:
        return [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS]
    return []


def _open_capture(spec: SourceSpec) -> cv2.VideoCapture:
    """Create a ``VideoCapture`` for a spec; raise if it cannot be opened."""
    capture = cv2.VideoCapture(spec.target, _backend_for(spec.kind),
                               _open_params(spec.kind))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open source {spec}")
    return capture


class VideoSource:
    """A frame source whose underlying capture can be swapped at runtime.

    ``open`` may be called from one thread (e.g. the UI) while ``read`` runs
    in another (the capture loop); a lock keeps the swap atomic.
    """

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._spec: SourceSpec | None = None
        self._lock = threading.Lock()

    @property
    def spec(self) -> SourceSpec | None:
        return self._spec

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._capture is not None and self._capture.isOpened()

    def open(self, spec: SourceSpec) -> None:
        """Open a new source, replacing the current one only on success."""
        new_capture = _open_capture(spec)
        with self._lock:
            old_capture = self._capture
            self._capture, self._spec = new_capture, spec
        if old_capture is not None:
            old_capture.release()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame, or ``(False, None)`` if none is available."""
        with self._lock:
            if self._capture is None:
                return False, None
            ok, frame = self._capture.read()
        return (True, frame) if ok else (False, None)

    def fps(self) -> float:
        """Nominal frame rate reported by the source (0.0 if unknown)."""
        with self._lock:
            if self._capture is None:
                return 0.0
            return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def release(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
            self._capture, self._spec = None, None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
