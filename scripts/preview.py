"""Play sources in a window and switch between them live with number keys.

Example::

    python scripts/preview.py media/a.mp4 media/b.mp4

Keys: 1-9 switch source, q or Esc quits. Files loop when they end.
"""

import argparse

import cv2
import numpy as np

from multitracker.capture import SourceSpec, VideoSource

WINDOW = "Multi-Tracker preview"
QUIT_KEYS = {ord("q"), 27}


def switch_to(source: VideoSource, text: str) -> None:
    """Open a source, keeping the current one if the new one fails."""
    try:
        source.open(SourceSpec.parse(text))
        print(f"Switched to {source.spec}")
    except OSError as error:
        print(f"{error}; staying on {source.spec}")


def next_frame(source: VideoSource) -> np.ndarray | None:
    """Read a frame, reopening the source once if a file has ended."""
    ok, frame = source.read()
    if not ok and source.spec is not None:
        source.open(source.spec)
        ok, frame = source.read()
    return frame if ok else None


def draw_status(frame: np.ndarray, source: VideoSource) -> np.ndarray:
    text = f"{source.spec}  (keys 1-9 switch, q quits)"
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def source_index_for(key: int, count: int) -> int | None:
    """Map a number key to a source index, or None if it is not one."""
    index = key - ord("1")
    return index if 0 <= index < count else None


def run(sources: list[str]) -> None:
    with VideoSource() as source:
        switch_to(source, sources[0])
        delay_ms = max(1, int(1000 / (source.fps() or 30)))
        while True:
            frame = next_frame(source)
            if frame is not None:
                cv2.imshow(WINDOW, draw_status(frame, source))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in QUIT_KEYS or not cv2.getWindowProperty(
                    WINDOW, cv2.WND_PROP_VISIBLE):
                break
            index = source_index_for(key, len(sources))
            if index is not None:
                switch_to(source, sources[index])
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+",
                        help="webcam index, file path or stream URL")
    run(parser.parse_args().sources)


if __name__ == "__main__":
    main()
