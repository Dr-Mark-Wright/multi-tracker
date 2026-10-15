"""Pick-up and put-down analysis for objects on a table (cup, bottle).

Works on plain detections, so it runs behind any tracker that finds the
objects and coasters, plus an optional right-wrist position:

- **Movement:** an object is moving when its centre travels more than a
  fraction of its own width over a few frames.
- **Right hand:** a pick-up is credited to the right hand when the right
  wrist is close to the object's box.
- **Coasters:** coasters are remembered once seen, because an object
  standing on a coaster hides it. A put-down is "on coaster N" when the
  object's centre is within a coaster's width of that coaster. A coaster
  that appears where an object stood (it was hidden underneath) is
  credited to that object's last pick-up.
- **Table height:** an object held still only counts as put down near the
  height of the coasters and the objects' starting places. Without this,
  a cup held still while drinking counted as put down at mouth height.

Standalone usage::

    from multitracker.behaviour import TableObjectAnalyser

    analyser = TableObjectAnalyser(["cup", "bottle"])
    events = analyser.update(detections, right_wrist=(x, y))
    print(analyser.status_lines())   # e.g. ["Cup: on coaster 2", ...]

Run on a clip with ``python scripts/table_events.py media/clip.mp4``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from multitracker.trackers.types import Detection

# Tuned on the 30 fps phone clip (objects ~45-65 px wide); all distances
# are relative to object or coaster width so they scale with resolution.
MOVE_WINDOW = 5            # frames over which movement is measured
MOVE_FRACTION = 0.15       # moving if centre travels > 15% of width
STILL_FRAMES = 10          # still this long (~0.3 s) -> put down
LET_GO_FRAMES = 30         # ...or this long (~1 s) with the hand still close
HAND_REACH = 1.0           # wrist within 1 object width of the box
MISSING_FRAMES = 8         # unseen this long -> treated as out of view
COASTER_MERGE = 0.5        # same coaster if centres within half a width
DUPLICATE_SCORE = 0.5      # ignore detections far weaker than the best


class ObjectState(Enum):
    RESTING = "resting"
    HELD = "held"


@dataclass
class Event:
    frame: int
    text: str  # may be corrected later, e.g. when a hidden coaster appears
    headline: str = ""  # short form for a banner, e.g. "Cup picked up"


@dataclass
class Coaster:
    centre: np.ndarray
    width: float


@dataclass
class TrackedObject:
    """What the analyser knows about one object (e.g. the cup)."""

    name: str
    state: ObjectState = ObjectState.RESTING
    centre: np.ndarray | None = None
    width: float = 0.0
    coaster: int | None = None          # index into the coaster list
    seen_frame: int = -1
    still_frames: int = 0
    by_right_hand: bool = False
    rest_centre: np.ndarray | None = None  # where it stood before pick-up
    pick_up: Event | None = None           # the last pick-up event
    history: deque = field(default_factory=lambda: deque(maxlen=MOVE_WINDOW))


def _centre(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(((x1 + x2) / 2, (y1 + y2) / 2))


def _distance_to_box(point: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return float(np.hypot(dx, dy))


class TableObjectAnalyser:
    """Tracks each named object through resting and held states."""

    def __init__(self, names: list[str], coaster_word: str = "coaster",
                 max_events: int = 50) -> None:
        self.objects = {name: TrackedObject(name) for name in names}
        self.coasters: list[Coaster] = []
        self.events: deque[Event] = deque(maxlen=max_events)
        self._coaster_word = coaster_word
        self._frame = -1
        self._boxes: dict[str, tuple] = {}
        self._start_heights: dict[str, float] = {}

    # ----- inputs -------------------------------------------------------

    def _remember_coasters(self, detections: list[Detection]) -> None:
        for det in detections:
            if self._coaster_word not in det.label or det.box is None:
                continue
            centre, width = _centre(det.box), det.box[2] - det.box[0]
            for coaster in self.coasters:
                if np.linalg.norm(coaster.centre - centre) < COASTER_MERGE * coaster.width:
                    coaster.centre = 0.9 * coaster.centre + 0.1 * centre
                    coaster.width = 0.9 * coaster.width + 0.1 * width
                    break
            else:
                self.coasters.append(Coaster(centre, width))
                self._credit_hidden_coaster(len(self.coasters) - 1)

    def _credit_hidden_coaster(self, index: int) -> None:
        """A new coaster where an object stood was hidden under it."""
        coaster = self.coasters[index]
        for obj in self.objects.values():
            if obj.rest_centre is None or obj.pick_up is None:
                continue
            reach = max(coaster.width, obj.width)
            if np.linalg.norm(coaster.centre - obj.rest_centre) < reach:
                obj.pick_up.text = obj.pick_up.text.replace(
                    "from the table", f"from coaster {index + 1}")
                obj.rest_centre = None

    def _pick_detection(self, obj: TrackedObject,
                        detections: list[Detection]) -> Detection | None:
        """The detection of this object, preferring continuity over score."""
        matches = [d for d in detections
                   if d.box is not None and obj.name in d.label
                   and self._coaster_word not in d.label]
        if not matches:
            return None
        best = max(d.score for d in matches)
        matches = [d for d in matches if d.score >= DUPLICATE_SCORE * best]
        if obj.centre is None:
            return max(matches, key=lambda d: d.score)
        return min(matches,
                   key=lambda d: np.linalg.norm(_centre(d.box) - obj.centre))

    def _coaster_under(self, obj: TrackedObject) -> int | None:
        for index, coaster in enumerate(self.coasters):
            reach = max(coaster.width, obj.width)
            if np.linalg.norm(coaster.centre - obj.centre) < reach:
                return index
        return None

    def _table_height(self, obj: TrackedObject) -> bool:
        """Is the object at the height where things rest on the table?"""
        heights = [c.centre[1] for c in self.coasters]
        heights += [y for y in self._start_heights.values()]
        if not heights:
            return True
        pad = max([c.width for c in self.coasters] + [obj.width])
        return min(heights) - pad <= obj.centre[1] <= max(heights) + pad

    def _hand_near(self, obj: TrackedObject, wrist: np.ndarray | None) -> bool:
        box = self._boxes.get(obj.name)
        if wrist is None or box is None:
            return False
        return _distance_to_box(wrist, box) < HAND_REACH * obj.width

    def _moving(self, obj: TrackedObject) -> bool:
        if len(obj.history) < 2:
            return False
        travel = np.linalg.norm(obj.history[-1] - obj.history[0])
        return travel > MOVE_FRACTION * obj.width

    # ----- state changes ------------------------------------------------

    def _log(self, text: str, headline: str) -> Event:
        event = Event(self._frame, text, headline)
        self.events.append(event)
        return event

    def _pick_up(self, obj: TrackedObject, right_hand: bool,
                 out_of_view: bool = False) -> Event:
        place = (f"coaster {obj.coaster + 1}" if obj.coaster is not None
                 else "the table")
        obj.state, obj.by_right_hand, obj.still_frames = (
            ObjectState.HELD, right_hand, 0)
        obj.rest_centre = obj.history[0] if obj.history else obj.centre
        hand = " with right hand" if right_hand else ""
        where = " (out of view)" if out_of_view else ""
        obj.pick_up = self._log(f"{obj.name.capitalize()} picked up from "
                                f"{place}{hand}{where}",
                                f"{obj.name.capitalize()} picked up")
        return obj.pick_up

    def _put_down(self, obj: TrackedObject) -> Event:
        obj.state = ObjectState.RESTING
        obj.coaster = self._coaster_under(obj)
        place = (f"on coaster {obj.coaster + 1}" if obj.coaster is not None
                 else "on the table, not on a coaster")
        return self._log(f"{obj.name.capitalize()} put down {place}",
                         f"{obj.name.capitalize()} put down")

    def _update_seen(self, obj: TrackedObject, det: Detection,
                     wrist: np.ndarray | None) -> Event | None:
        first = obj.centre is None
        obj.centre = _centre(det.box)
        obj.width = det.box[2] - det.box[0]
        obj.seen_frame = self._frame
        obj.history.append(obj.centre)
        if first:
            obj.coaster = self._coaster_under(obj)
            self._start_heights[obj.name] = float(obj.centre[1])
            return None
        moving = self._moving(obj)
        if obj.state is ObjectState.RESTING:
            if moving:
                return self._pick_up(obj, self._hand_near(obj, wrist))
            if obj.coaster is None:  # coasters can be found after the object
                obj.coaster = self._coaster_under(obj)
            return None
        obj.still_frames = 0 if moving else obj.still_frames + 1
        if not self._table_height(obj):
            return None
        let_go = not self._hand_near(obj, wrist)
        if (obj.still_frames >= LET_GO_FRAMES
                or (obj.still_frames >= STILL_FRAMES and let_go)):
            return self._put_down(obj)
        return None

    def _update_unseen(self, obj: TrackedObject,
                       wrist: np.ndarray | None) -> Event | None:
        gone = self._frame - obj.seen_frame
        if (obj.state is ObjectState.RESTING and obj.centre is not None
                and gone == MISSING_FRAMES and self._hand_near(obj, wrist)):
            obj.history.clear()
            return self._pick_up(obj, True, out_of_view=True)
        if gone == MISSING_FRAMES:
            obj.history.clear()
        return None

    def update(self, detections: list[Detection],
               right_wrist: tuple[float, float] | None = None) -> list[Event]:
        """Feed one frame; returns the events it caused (often none)."""
        self._frame += 1
        self._remember_coasters(detections)
        wrist = np.array(right_wrist) if right_wrist is not None else None
        events = []
        for obj in self.objects.values():
            det = self._pick_detection(obj, detections)
            if det is not None:
                self._boxes[obj.name] = det.box
                event = self._update_seen(obj, det, wrist)
            else:
                event = self._update_unseen(obj, wrist)
            if event is not None:
                events.append(event)
        return events

    # ----- outputs ------------------------------------------------------

    def describe(self, obj: TrackedObject) -> str:
        if obj.centre is None:
            return "not seen yet"
        if obj.state is ObjectState.HELD:
            hand = "in right hand" if obj.by_right_hand else "picked up"
            if self._frame - obj.seen_frame >= MISSING_FRAMES:
                return f"{hand} (out of view)"
            return f"{hand}, moving" if self._moving(obj) else hand
        if obj.coaster is not None:
            return f"on coaster {obj.coaster + 1}"
        if obj.pick_up is None:
            # Never moved: a coaster could be hidden underneath it.
            return "resting, no coaster visible"
        return "on the table, not on a coaster"

    def status_lines(self, last_events: int = 1) -> list[str]:
        lines = [f"{obj.name.capitalize()}: {self.describe(obj)}"
                 for obj in self.objects.values()]
        recent = list(self.events)[-last_events:] if last_events else []
        lines += [f"Last: {e.text}" for e in recent]
        return lines
