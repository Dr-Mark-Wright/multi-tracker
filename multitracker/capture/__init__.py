"""Capture layer: one interface over webcam, file and network sources.

Standalone usage::

    from multitracker.capture import SourceSpec, VideoSource

    source = VideoSource()
    source.open(SourceSpec.parse("0"))          # webcam 0
    ok, frame = source.read()
    source.open(SourceSpec.parse("clip.mp4"))   # switch at runtime
"""

from multitracker.capture.source import SourceKind, SourceSpec, VideoSource

__all__ = ["SourceKind", "SourceSpec", "VideoSource"]
