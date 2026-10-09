"""Headless-pipeline check: capture -> tracker -> overlay -> window.

Example::

    python scripts/track_preview.py "media/clip.mp4"

Keys: q or Esc quits. Files loop when they end.
"""

import argparse
import logging
import time

import cv2
import numpy as np

from multitracker.capture import SourceSpec, VideoSource
from multitracker.render import draw_hud, draw_result
from multitracker.trackers.yolo_pose import YoloPoseTracker

WINDOW = "Multi-Tracker"
QUIT_KEYS = {ord("q"), 27}


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


def next_frame(source: VideoSource) -> np.ndarray | None:
    """Read a frame, reopening the source once if a file has ended."""
    ok, frame = source.read()
    if not ok and source.spec is not None:
        source.open(source.spec)
        ok, frame = source.read()
    return frame if ok else None


def window_closed(key: int) -> bool:
    return key in QUIT_KEYS or not cv2.getWindowProperty(
        WINDOW, cv2.WND_PROP_VISIBLE)


def run(source_text: str) -> None:
    tracker = YoloPoseTracker()
    meter = FrameRateMeter()
    with VideoSource() as source:
        source.open(SourceSpec.parse(source_text))
        delay_ms = max(1, int(1000 / (source.fps() or 30)))
        while True:
            frame = next_frame(source)
            if frame is None:
                break
            result = tracker.process(frame)
            canvas = draw_result(frame, result)
            draw_hud(canvas, [
                f"{tracker.name}  people: {len(result.detections)}",
                f"inference {result.inference_ms:.1f} ms  "
                f"display {meter.tick():.0f} fps",
            ])
            cv2.imshow(WINDOW, canvas)
            if window_closed(cv2.waitKey(delay_ms) & 0xFF):
                break
    tracker.close()
    cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="webcam index, file path or stream URL")
    run(parser.parse_args().source)


if __name__ == "__main__":
    main()
