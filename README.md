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

## Live demo window (PySide6)

```powershell
.venv\Scripts\python -m multitracker.ui "media\your-clip.mp4" --tracker YOLOE-26 --prompts "bottle, cup, round coaster"
```

Pick a webcam, file or stream URL in **Source**, choose a tracker, and type
YOLOE-26 prompts (comma-separated, Enter to apply). The 3D panel shows
BodyPose3D skeletons; drag inside it to rotate, drag its edge to resize, or
float it into its own window. Keys: `T` next tracker, `3` show/hide the 3D
panel, `F11` full screen.

### Cup and bottle analysis

Choose **Cup & bottle** in the Tracker list to see when the cup or bottle is
picked up or put down, and on which coaster. The state shows top-left on
the video, and each event flashes up as a banner. To list every event in
a clip:

```powershell
.venv\Scripts\python scripts\table_events.py "media\your-clip.mp4"
```

### A phone as the camera

Run a camera-streaming app on the phone (Android: *IP Webcam*; iPhone: an
app such as *IP Camera Lite*), on the same Wi-Fi as this PC. Type the
address it shows into **Source**, e.g. `http://192.168.1.23:8080/video`
for IP Webcam, and press **Open**. VLC's phone app only plays video, so it
can't send the camera. If the stream drops, the app shows "signal lost"
and reconnects by itself.

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
