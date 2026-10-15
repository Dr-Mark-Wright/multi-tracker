"""3D body pose by lifting YOLO pose's 2D keypoints with MotionBERT.

MotionBERT (Zhu et al., ICCV 2023) looks at a whole window of 2D poses
over time and estimates the 3D pose for every frame in it. Live, the
window is the most recent frames, and the newest frame's estimate is
shown. That keeps delay at zero, at some cost to accuracy compared with
MotionBERT's offline use, where it also sees frames after the current one.

The 3D output is relative to the pelvis and scaled to metres by assuming
a 0.5 m pelvis-to-neck length, since one camera cannot measure true size.

Standalone usage::

    from multitracker.trackers.body_pose_motionbert import MotionBertTracker

    tracker = MotionBertTracker()
    result = tracker.process(frame)   # frame: BGR numpy array
    for body in result.detections:
        print(body.keypoints.shape)      # (17, 3): H36M x, y pixels + confidence
        print(body.keypoints_3d.shape)   # (17, 3): x, y, z metres
"""

from __future__ import annotations

import time
from collections import deque
from functools import partial

import numpy as np
import torch
from torch import nn

from multitracker.device import select_device
from multitracker.trackers.model_files import ensure_model
from multitracker.trackers.motionbert.dstformer import DSTformer
from multitracker.trackers.types import Detection, TrackerResult
from multitracker.trackers.yolo_pose import YoloPoseTracker

# The checkpoint MotionBERT's own docs recommend for in-the-wild video.
CHECKPOINT_URL = ("https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
                  "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/"
                  "best_epoch.bin")
CHECKPOINT_FILE = "motionbert_lite_h36m_global.bin"

# 243 frames is the model's maximum (~8 s at 30 fps). A shorter window is
# faster and adapts sooner after the person changes pose.
WINDOW = 81
MIN_FRAMES = 9          # don't estimate from less history than this
FORGET_AFTER = 15       # frames without a person before history resets
TORSO_METRES = 0.5      # assumed pelvis-to-neck length
# MotionBERT's offline script averages each estimate with a mirrored pass.
# On the phone clip that changed nothing measurable (3.1 mm/frame jitter
# either way) and cost ~7 ms per frame, so it is off unless asked for.

# H36M 17-joint skeleton, as MotionBERT outputs it.
H36M_NAMES = ("pelvis", "right hip", "right knee", "right ankle", "left hip",
              "left knee", "left ankle", "spine", "neck", "nose", "head",
              "left shoulder", "left elbow", "left wrist", "right shoulder",
              "right elbow", "right wrist")
H36M_SKELETON = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
)
PELVIS, NECK = 0, 8
_LEFT = [4, 5, 6, 11, 12, 13]
_RIGHT = [1, 2, 3, 14, 15, 16]


def coco_to_h36m(coco: np.ndarray) -> np.ndarray:
    """COCO 17 keypoints ``(17, 3)`` -> H36M 17 joints ``(17, 3)``.

    Joints H36M has but COCO lacks (pelvis, spine, neck, head) are built
    from neighbours; their confidence is the weaker of the pair.
    """
    def mid(a: int, b: int) -> np.ndarray:
        point = (coco[a] + coco[b]) / 2
        point[2] = min(coco[a, 2], coco[b, 2])
        return point

    h36m = np.zeros((17, 3), dtype=np.float32)
    h36m[0] = mid(11, 12)                      # pelvis: between the hips
    h36m[1], h36m[2], h36m[3] = coco[12], coco[14], coco[16]
    h36m[4], h36m[5], h36m[6] = coco[11], coco[13], coco[15]
    h36m[8] = mid(5, 6)                        # neck: between the shoulders
    h36m[7] = (h36m[0] + h36m[8]) / 2          # spine
    h36m[7, 2] = min(h36m[0, 2], h36m[8, 2])
    h36m[9] = coco[0]                          # nose
    h36m[10] = mid(1, 2)                       # head: between the eyes
    h36m[11], h36m[12], h36m[13] = coco[5], coco[7], coco[9]
    h36m[14], h36m[15], h36m[16] = coco[6], coco[8], coco[10]
    return h36m


def _flip(data: torch.Tensor) -> torch.Tensor:
    """Mirror left-right, as MotionBERT does for test-time augmentation."""
    flipped = data.clone()
    flipped[..., 0] *= -1
    flipped[..., _LEFT + _RIGHT, :] = flipped[..., _RIGHT + _LEFT, :]
    return flipped


def load_model(device: str) -> nn.Module:
    """DSTformer with the lite checkpoint's settings, ready for inference."""
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                      depth=5, num_heads=8, mlp_ratio=4, num_joints=17,
                      maxlen=243, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      att_fuse=True)
    path = ensure_model(CHECKPOINT_URL, filename=CHECKPOINT_FILE)
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("module."): v
               for k, v in state["model_pos"].items()}
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


class MotionBertTracker:
    """YOLO pose for 2D, MotionBERT for 3D, one person at a time."""

    name = "BodyPose3D MotionBERT"

    def __init__(self, window: int = WINDOW, flip: bool = False,
                 device: str | None = None) -> None:
        self._device = device or select_device()
        self._pose = YoloPoseTracker(device=self._device)
        self._model = load_model(self._device)
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._flip = flip
        self._missing = 0

    def _normalise(self, h36m: np.ndarray, width: int, height: int) -> np.ndarray:
        """Pixels -> MotionBERT's input space: centred, scaled by min side."""
        scale = min(width, height) / 2
        out = h36m.copy()
        out[:, :2] = (out[:, :2] - (width / 2, height / 2)) / scale
        return out

    @torch.inference_mode()
    def _lift(self) -> np.ndarray:
        """3D pose for the newest frame, pelvis-relative, in metres."""
        clip = torch.from_numpy(np.stack(self._history))[None].to(self._device)
        poses = self._model(clip)
        if self._flip:
            poses = (poses + _flip(self._model(_flip(clip)))) / 2
        pose = poses[0, -1].float().cpu().numpy()
        pose -= pose[PELVIS]
        torso = np.linalg.norm(pose[NECK])
        return pose * (TORSO_METRES / torso) if torso > 1e-6 else pose

    def process(self, frame: np.ndarray) -> TrackerResult:
        started = time.perf_counter()
        people = self._pose.process(frame).detections
        if not people:
            self._missing += 1
            if self._missing >= FORGET_AFTER:
                self._history.clear()
            return TrackerResult(
                inference_ms=(time.perf_counter() - started) * 1000)
        self._missing = 0
        person = max(people, key=lambda d: d.score)
        h36m = coco_to_h36m(person.keypoints)
        height, width = frame.shape[:2]
        self._history.append(self._normalise(h36m, width, height))
        body = Detection(box=person.box, score=person.score, label="body 3D",
                         keypoints=h36m, skeleton=H36M_SKELETON)
        if len(self._history) >= MIN_FRAMES:
            body.keypoints_3d = self._lift()
        return TrackerResult(
            detections=[body],
            inference_ms=(time.perf_counter() - started) * 1000)

    def close(self) -> None:
        self._pose.close()
        self._model = None
