"""Headless check of the capture layer: read, switch source, read again.

Example::

    python scripts/capture_check.py media/a.mp4 media/b.mp4 0
"""

import argparse
import time

from multitracker.capture import SourceSpec, VideoSource


def read_frames(source: VideoSource, count: int) -> tuple[int, float]:
    """Read up to ``count`` frames; return frames read and elapsed seconds."""
    start = time.perf_counter()
    frames = 0
    for _ in range(count):
        ok, _frame = source.read()
        if not ok:
            break
        frames += 1
    return frames, time.perf_counter() - start


def check_source(source: VideoSource, text: str, count: int) -> None:
    spec = SourceSpec.parse(text)
    try:
        source.open(spec)
    except OSError as error:
        print(f"[skip] {error}")
        return
    frames, seconds = read_frames(source, count)
    rate = frames / seconds if seconds else 0.0
    print(f"[ok]   {spec}: {frames} frames, {rate:.0f} fps read rate, "
          f"nominal {source.fps():.0f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+",
                        help="webcam index, file path or stream URL")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    with VideoSource() as source:
        for text in args.sources:
            check_source(source, text, args.frames)


if __name__ == "__main__":
    main()
