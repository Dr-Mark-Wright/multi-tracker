"""List every cup and bottle pick-up and put-down in a video file.

Example::

    python scripts/table_events.py "media/clip.mp4"

Prints one line per event with its time in the clip, then the final
state of each object.
"""

import argparse

import cv2

from multitracker.trackers.table_objects import TableObjectsTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = TableObjectsTracker()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tracker.process(frame)
    capture.release()
    # Printed at the end: an event's text can be corrected after it
    # happens (e.g. a coaster found under where the cup stood).
    for event in tracker.analyser.events:
        print(f"{event.frame / fps:6.1f} s  {event.text}")
    for line in tracker.analyser.status_lines(last_events=0):
        print(line)
    tracker.close()


if __name__ == "__main__":
    main()
