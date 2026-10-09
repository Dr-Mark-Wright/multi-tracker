# Multi-Tracker Live Demonstrator

Live switching between video sources and computer-vision trackers for
teaching demos and the Wonderbeam perception stack.

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

## Capture layer check

```powershell
.venv\Scripts\python scripts\make_test_clip.py media\a.mp4 --label A
.venv\Scripts\python scripts\make_test_clip.py media\b.mp4 --label B
.venv\Scripts\python scripts\capture_check.py media\a.mp4 media\b.mp4 0
```

## Tracking check (YOLO pose)

```powershell
.venv\Scripts\python scripts\track_preview.py "media\your-clip.mp4"
```

Press `t` to cycle trackers, or start on one, e.g. 3D body pose:

```powershell
.venv\Scripts\python scripts\track_preview.py 0 --tracker BodyPose3D
```

See `DISCOVERY_LOG.md` for decisions and issues resolved along the way.
