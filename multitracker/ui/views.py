"""Display widgets: the video view and the rotatable 3D skeleton view."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from multitracker.render.overlay import KEYPOINT_MIN_CONFIDENCE
from multitracker.trackers.types import Detection

BACKGROUND = QColor(20, 20, 24)


class VideoView(QWidget):
    """Shows the latest frame, scaled to fit with its aspect ratio kept."""

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self.setMinimumSize(320, 240)

    def show_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self._image is None:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No source open")
            return
        size = self._image.size().scaled(self.size(),
                                         Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(0, 0, size.width(), size.height())
        target.moveCenter(self.rect().center())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)


BONE_RGBA = (0.0, 0.7, 1.0, 1.0)
JOINT_RGBA = (0.0, 1.0, 0.0, 1.0)


def _to_gl(points: np.ndarray) -> np.ndarray:
    """Tracker axes (x right, y down, z away) -> GL axes (x right, y away, z up)."""
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


class SkeletonView(gl.GLViewWidget):
    """3D skeletons in metres around the hips; drag to orbit, wheel to zoom."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(240, 240)
        self.setBackgroundColor(BACKGROUND)
        self.setCameraPosition(distance=2.2, elevation=10, azimuth=-90)
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.25, 0.25)
        grid.translate(0, 0, -1.0)  # floor roughly below the hips
        self.addItem(grid)
        self._bones = gl.GLLinePlotItem(mode="lines", width=3,
                                        color=BONE_RGBA, antialias=True)
        self._joints = gl.GLScatterPlotItem(size=8, color=JOINT_RGBA)
        self.addItem(self._bones)
        self.addItem(self._joints)
        self.clear_skeletons()

    def clear_skeletons(self) -> None:
        empty = np.zeros((0, 3))
        self._bones.setData(pos=empty)
        self._joints.setData(pos=empty)

    def show_skeletons(self, detections: list[Detection]) -> None:
        bones, joints = [], []
        for detection in detections:
            if detection.keypoints_3d is None:
                continue
            points = _to_gl(detection.keypoints_3d)
            visible = (detection.keypoints[:, 2] >= KEYPOINT_MIN_CONFIDENCE
                       if detection.keypoints is not None
                       else np.ones(len(points), bool))
            joints.append(points[visible])
            for start, end in detection.skeleton:
                if visible[start] and visible[end]:
                    bones.extend((points[start], points[end]))
        if not joints:
            self.clear_skeletons()
            return
        self._bones.setData(pos=np.array(bones) if bones else np.zeros((0, 3)))
        self._joints.setData(pos=np.vstack(joints))
