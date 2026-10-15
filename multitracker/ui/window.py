"""Main window: video in the centre, controls and 3D view in side panels.

Slow work (opening a source, loading a tracker's model the first time)
runs on background threads so the window and the video never freeze. Each
change only takes effect once it has succeeded; until then the current
source and tracker keep running.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget)

from multitracker.capture import SourceKind, SourceSpec, VideoSource
from multitracker.trackers.registry import TrackerSwitcher
from multitracker.trackers.types import TrackerResult
from multitracker.ui.pipeline import FrameStats, PipelineThread
from multitracker.ui.views import SkeletonView, VideoView

WEBCAM_CHOICES = ("0", "1", "2")
VIDEO_FILTER = "Video files (*.mp4 *.mov *.avi *.mkv *.webm);;All files (*)"
MESSAGE_MS = 6000
FIRST_FRAME_WAIT_S = 3.0
CONTROLS_MAX_WIDTH = 340
SKELETON_START_WIDTH = 520


class _TaskSignals(QObject):
    done = Signal(object)    # return value
    failed = Signal(str)     # error message


class BackgroundTask:
    """Run ``work`` on a Python thread; report back on the UI thread.

    Plain Python threads, not ``QThreadPool``: loading a PyTorch
    checkpoint on a Qt pool thread crashed the app (access violation in
    ``torch.load``) when switching to YOLOE-26 after MediaPipe.
    """

    def __init__(self, work: Callable[[], object]) -> None:
        self._work = work
        self.signals = _TaskSignals()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        try:
            value = self._work()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(value)


class MainWindow(QMainWindow):
    """The Multi-Tracker live demo window."""

    def __init__(self, source_text: str | None, tracker: str,
                 prompts: list[str], media_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Tracker")
        self.resize(1400, 860)
        self._source = VideoSource()
        self._switcher = TrackerSwitcher()
        self._switcher.set_prompts(prompts)
        self._tasks: set[BackgroundTask] = set()

        self._video = VideoView()
        self.setCentralWidget(self._video)
        self._skeleton = SkeletonView()
        self._build_controls(media_dir, prompts)
        self._build_3d_panel()
        self._build_actions()

        self._pipeline = PipelineThread(self._source, self._switcher)
        self._pipeline.frame_ready.connect(self._show_frame)
        self._pipeline.error.connect(self._show_error)
        self._pipeline.source_lost.connect(
            lambda name: self.statusBar().showMessage(
                f"Lost {name}; reconnecting…"))
        self._pipeline.source_restored.connect(
            lambda name: self.statusBar().showMessage(
                f"Reconnected to {name}", MESSAGE_MS))
        self._pipeline.start()

        if source_text:
            self._source_box.setCurrentText(source_text)
            self._open_source()
        self._select_tracker(tracker)

    # ----- layout -------------------------------------------------------

    def _build_controls(self, media_dir: Path | None,
                        prompts: list[str]) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._source_group(media_dir))
        layout.addWidget(self._tracker_group(prompts))
        layout.addWidget(self._status_group())
        layout.addStretch()
        dock = QDockWidget("Controls")
        dock.setObjectName("controls")
        panel.setMaximumWidth(CONTROLS_MAX_WIDTH)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _source_group(self, media_dir: Path | None) -> QGroupBox:
        group = QGroupBox("Source")
        layout = QVBoxLayout(group)
        self._source_box = QComboBox()
        self._source_box.setEditable(True)
        self._source_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._source_box.lineEdit().setPlaceholderText(
            "webcam number, file or stream URL")
        for index in WEBCAM_CHOICES:
            self._source_box.addItem(f"Webcam {index}", index)
        if media_dir is not None and media_dir.is_dir():
            for path in sorted(media_dir.glob("*.mp4")):
                self._source_box.addItem(path.name, str(path))
        self._source_box.setMinimumContentsLength(18)
        self._source_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._source_box.lineEdit().returnPressed.connect(self._open_source)
        self._source_box.activated.connect(self._open_source)
        layout.addWidget(self._source_box)
        buttons = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._open_button = QPushButton("Open")
        self._open_button.clicked.connect(self._open_source)
        buttons.addWidget(browse)
        buttons.addWidget(self._open_button)
        layout.addLayout(buttons)
        return group

    def _tracker_group(self, prompts: list[str]) -> QGroupBox:
        group = QGroupBox("Tracker")
        layout = QFormLayout(group)
        self._tracker_box = QComboBox()
        self._tracker_box.addItems(self._switcher.names)
        self._tracker_box.activated.connect(
            lambda _: self._select_tracker(self._tracker_box.currentText()))
        layout.addRow("Tracker", self._tracker_box)
        self._prompt_edit = QLineEdit(", ".join(prompts))
        self._prompt_edit.setPlaceholderText("e.g. bottle, cup, round coaster")
        self._prompt_edit.setToolTip(
            "YOLOE-26 classes, comma-separated. Press Enter to apply.")
        self._prompt_edit.returnPressed.connect(self._apply_prompts)
        layout.addRow("Prompts", self._prompt_edit)
        return group

    def _status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QFormLayout(group)
        self._status_labels = {}
        for key in ("Source", "Tracker", "Detections", "Inference",
                    "Display", "Activity"):
            label = QLabel("–")
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addRow(key, label)
            self._status_labels[key] = label
        return group

    def _build_3d_panel(self) -> None:
        dock = QDockWidget("3D skeleton (drag to rotate)")
        dock.setObjectName("skeleton3d")
        dock.setWidget(self._skeleton)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # Drag the divider beside the panel to resize it, or use the
        # title bar's float button to give it a window of its own.
        self.resizeDocks([dock], [SKELETON_START_WIDTH],
                         Qt.Orientation.Horizontal)
        self._skeleton_dock = dock

    def _build_actions(self) -> None:
        cycle = QAction("Next tracker", self)
        cycle.setShortcut(QKeySequence("T"))
        cycle.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        cycle.triggered.connect(self._cycle_tracker)
        full = QAction("Full screen", self)
        full.setShortcut(QKeySequence("F11"))
        full.triggered.connect(self._toggle_full_screen)
        leave = QAction("Leave full screen", self)
        leave.setShortcut(QKeySequence("Esc"))
        leave.triggered.connect(self.showNormal)
        show_3d = self._skeleton_dock.toggleViewAction()
        show_3d.setShortcut(QKeySequence("3"))
        for action in (cycle, full, leave, show_3d):
            self.addAction(action)

    # ----- background work ----------------------------------------------

    def _run(self, work: Callable[[], object],
             done: Callable[[object], None],
             failed: Callable[[str], None]) -> None:
        task = BackgroundTask(work)
        self._tasks.add(task)  # keep signals alive until the task reports

        def finish(handler, value):
            self._tasks.discard(task)
            handler(value)

        task.signals.done.connect(lambda value: finish(done, value))
        task.signals.failed.connect(lambda message: finish(failed, message))
        task.thread.start()

    # ----- source -------------------------------------------------------

    def _source_text(self) -> str:
        index = self._source_box.currentIndex()
        text = self._source_box.currentText().strip()
        if index >= 0 and text == self._source_box.itemText(index):
            return self._source_box.itemData(index) or text
        return text

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "",
                                              VIDEO_FILTER)
        if path:
            self._source_box.setCurrentText(path)
            self._open_source()

    def _open_source(self) -> None:
        text = self._source_text()
        if not text:
            return
        spec = SourceSpec.parse(text)
        self._open_button.setEnabled(False)
        self.statusBar().showMessage(f"Opening {spec}…")
        self._run(lambda: self._source.open(spec),
                  lambda _: self._source_opened(spec),
                  self._source_failed)

    def _source_opened(self, spec: SourceSpec) -> None:
        self._open_button.setEnabled(True)
        label = self._status_labels["Source"]
        label.setText(Path(str(spec.target)).name
                      if spec.kind is SourceKind.FILE else str(spec.target))
        label.setToolTip(str(spec.target))
        self.statusBar().showMessage(f"Playing {spec}", MESSAGE_MS)

    def _source_failed(self, message: str) -> None:
        self._open_button.setEnabled(True)
        current = self._source.spec
        keep = f"; still showing {current.target}" if current else ""
        self.statusBar().showMessage(f"{message}{keep}", MESSAGE_MS)

    # ----- tracker ------------------------------------------------------

    def _select_tracker(self, name: str) -> None:
        self._tracker_box.setEnabled(False)
        self.statusBar().showMessage(f"Loading {name}…")
        self._run(lambda: self._load_tracker(name),
                  lambda _: self._tracker_ready(),
                  self._tracker_failed)

    def _load_tracker(self, name: str) -> None:
        """Runs on the pool: warm up on a real frame, then go live."""
        deadline = time.monotonic() + FIRST_FRAME_WAIT_S
        while self._pipeline.last_frame is None and time.monotonic() < deadline:
            time.sleep(0.05)  # at start-up the source may still be opening
        frame = self._pipeline.last_frame
        if frame is not None:
            self._switcher.warm_up(name, frame)
        self._switcher.switch(name)

    def _tracker_ready(self) -> None:
        name = self._switcher.active_name
        self._tracker_box.setEnabled(True)
        self._tracker_box.setCurrentText(name)
        self._prompt_edit.setEnabled(self._switcher.accepts_prompts)
        self.statusBar().showMessage(f"{name} ready", MESSAGE_MS)

    def _tracker_failed(self, message: str) -> None:
        self._tracker_box.setEnabled(True)
        self._tracker_box.setCurrentText(self._switcher.active_name)
        self.statusBar().showMessage(f"Could not load tracker: {message}",
                                     MESSAGE_MS)

    def _cycle_tracker(self) -> None:
        if not self._tracker_box.isEnabled():
            return  # a tracker is still loading
        names = self._switcher.names
        current = names.index(self._tracker_box.currentText())
        self._select_tracker(names[(current + 1) % len(names)])

    def _apply_prompts(self) -> None:
        prompts = [p.strip() for p in self._prompt_edit.text().split(",")]
        prompts = [p for p in prompts if p]
        if prompts:
            self._switcher.set_prompts(prompts)
            self.statusBar().showMessage(
                "Prompts: " + ", ".join(prompts), MESSAGE_MS)

    # ----- frames -------------------------------------------------------

    def _show_frame(self, image: QImage, result: TrackerResult,
                    stats: FrameStats) -> None:
        self._video.show_image(image)
        labels = self._status_labels
        labels["Tracker"].setText(stats.tracker)
        labels["Detections"].setText(str(stats.detections))
        labels["Inference"].setText(f"{stats.inference_ms:.1f} ms")
        labels["Display"].setText(f"{stats.fps:.0f} fps")
        labels["Activity"].setText("\n".join(result.status) or "–")
        if self._skeleton_dock.isVisible():
            self._skeleton.show_skeletons(result.detections)

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, MESSAGE_MS)

    def _toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:
        self._pipeline.stop()
        for task in list(self._tasks):
            task.thread.join(timeout=10)
        self._switcher.close()
        self._source.release()
        super().closeEvent(event)
