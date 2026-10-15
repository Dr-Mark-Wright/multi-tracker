"""Launch the Multi-Tracker window.

Examples::

    python -m multitracker.ui
    python -m multitracker.ui media/a.mp4 --tracker YOLOE-26 --prompts "bottle, cup"

Keys: T next tracker, B boxes on/off, M masks on/off, 3 shows or hides the 3D panel, F11 full screen,
Esc leaves full screen. Drag the divider beside a panel to resize it.
"""

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from multitracker.trackers.registry import TRACKERS
from multitracker.ui.window import MainWindow

MEDIA_DIR = Path(__file__).resolve().parents[2] / "media"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?",
                        help="webcam index, file path or stream URL to open")
    parser.add_argument("--tracker", default="YOLO pose",
                        choices=list(TRACKERS))
    parser.add_argument("--prompts", default="person",
                        help="comma-separated YOLOE-26 classes")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow(args.source, args.tracker,
                        [p.strip() for p in args.prompts.split(",")],
                        media_dir=MEDIA_DIR)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
