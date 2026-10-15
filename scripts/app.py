"""Launch the Multi-Tracker window (same as ``python -m multitracker.ui``).

Example::

    python scripts/app.py "media/a.mp4" --tracker YOLOE-26 --prompts "bottle, cup"
"""

from multitracker.ui.__main__ import main

if __name__ == "__main__":
    main()
