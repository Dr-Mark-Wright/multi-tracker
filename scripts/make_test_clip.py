"""Generate a short synthetic test clip for exercising the capture layer."""

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 640, 360, 30


def draw_frame(index: int, label: str) -> np.ndarray:
    """Draw one frame: a moving circle plus a frame counter."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x = int((index * 5) % WIDTH)
    cv2.circle(frame, (x, HEIGHT // 2), 30, (0, 200, 255), -1)
    cv2.putText(frame, f"{label} #{index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def write_clip(path: Path, seconds: int, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    for index in range(seconds * FPS):
        writer.write(draw_frame(index, label))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--label", default="test")
    args = parser.parse_args()
    write_clip(args.path, args.seconds, args.label)
    print(f"Wrote {args.path}")


if __name__ == "__main__":
    main()
