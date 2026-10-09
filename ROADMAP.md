# Roadmap and Strategy

The strategic path for the Multi-Tracker Live Demonstrator: why it exists,
the decisions that shape it, and where each milestone stands. Technical
issues and their fixes live in `DISCOVERY_LOG.md`.

## Purpose

1. Teaching demos for MA Immersive and community practitioner students.
2. A testbed for the Wonderbeam installation's perception stack.
3. A showcase for the RTX 5090 desktop.

**Defining requirement:** it must work as a live, in-person demo. Source,
tracker and YOLOE-26 class prompts all change on the fly, with no restarts.

## Priorities

- **Body and hands first.** Face tracking is a bonus: it is kept, but
  small or distant faces are accepted as a known limitation (2026-10-09).
- **Every tracker liftable on its own**, so students can reuse one module
  without the whole app.
- **Never crash or freeze mid-demo**; bad sources fail fast and leave the
  current one running.

## Architecture

```
VideoSource ──frame──▶ TrackerSwitcher ──TrackerResult──▶ renderer ──▶ UI
 (capture/)            (trackers/registry.py)              (render/)
                        ├─ YOLO pose
                        ├─ MediaPipe (hands + face)
                        ├─ YOLOE-26 (live text prompts)
                        └─ BodyPose3D (MediaPipe 3D; MotionBERT planned)
```

The UI sends "switch to X" events and reads back frames with overlays. It
never knows which source or tracker is active.

## Decisions

| Decision | Choice | Status | Reasoning |
|---|---|---|---|
| Environment | Python 3.12 + `venv` (no conda) | Decided | MediaPipe and PyTorch CUDA wheels support 3.12; all packages are pip-distributed |
| GPU | PyTorch cu128 build, loud CPU-fallback warning | Decided | RTX 5090 (Blackwell, sm_120) needs CUDA 12.8+ |
| Pose model | `yolo26s-pose` | Decided | ~5.5 ms/frame on the 5090 |
| Open-vocabulary model size | `yoloe-26l-seg` | Decided | Small model missed coasters entirely; large found 96% at ~10 ms |
| Hands + face | MediaPipe Tasks 1.1 (CPU) | Decided | Matches Wonderbeam's stack; CPU-only on Windows |
| UI toolkit | PySide6 (Qt) proposed | **Open** | Threaded capture loop, native widgets; web UIs add latency |
| BodyPose3D | MediaPipe Pose 3D landmarks first, MotionBERT lifter later | Decided (2026-10-09) | MediaPipe is live straight away with no new dependencies; MotionBERT's long two-sided frame window adds delay, so it comes second as a higher-quality option and a teaching comparison |
| Open-vocabulary default | YOLOE-26 over YOLO-World | **Provisional** | Behind the tracker interface, so cheap to change |
| Licensing | Ultralytics is AGPL-3.0 | Noted | Fine for teaching; closed Wonderbeam use needs an enterprise licence |

## Milestones

| # | Milestone | Status | Commit |
|---|---|---|---|
| 1 | Capture layer with runtime source switching | Done 2026-10-09 | `2ddff82` |
| 2 | One tracker end to end (YOLO pose + overlays) | Done 2026-10-09 | `7fe4fe5` |
| 3 | Tracker-switching interface + MediaPipe | Done 2026-10-09 | `002a02e` |
| 4 | YOLOE-26 with live text prompts | Done 2026-10-09 | `f5bfb41` |
| 5a | BodyPose3D: MediaPipe 3D landmarks + 3D views | Done 2026-10-09 | this commit |
| 5b | BodyPose3D: MotionBERT lifter on YOLO pose | Next | |
| 6 | Proper UI (source, tracker, prompt controls) | Planned | |
| 7 | Live-demo hardening (dropped streams, missing webcam) | Planned | |

## Measured performance (RTX 5090, Ryzen 9 9950X3D, 478×850 phone clip)

| Tracker | Per frame | Max rate | Runs on |
|---|---|---|---|
| YOLO pose (`yolo26s-pose`) | ~5.5 ms | ~180 fps | GPU |
| MediaPipe hands | ~7.8 ms | ~130 fps | CPU |
| MediaPipe face (2 tiles) | ~5.5 ms | ~180 fps | CPU |
| MediaPipe hands + face | ~13 ms | ~75 fps | CPU |
| YOLOE-26 large (3 prompts) | ~10 ms | ~100 fps | GPU |
| BodyPose3D, MediaPipe lite | ~11 ms | ~90 fps | CPU |
| BodyPose3D, MediaPipe full (default) | ~13 ms | ~80 fps | CPU |
| BodyPose3D, MediaPipe heavy | ~27 ms | ~37 fps | CPU |

## Future ideas

- Persistent person IDs via Ultralytics' ByteTrack/BoT-SORT.
- Face ROI tracking: crop around the last known face instead of tiling,
  for better recall on distant faces at lower cost.
