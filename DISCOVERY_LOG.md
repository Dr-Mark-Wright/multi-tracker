# Discovery Log

Issues hit and resolved, and the reasoning behind decisions. Newest last.

## 2026-10-09 — Environment setup

- **Python 3.12 chosen over newer releases.** MediaPipe and the CUDA builds
  of PyTorch tend to lag new Python versions; 3.12 is the safest match.
- **Host CPU was running with half its cores.** Windows reported 8 cores /
  16 threads for a 16-core Ryzen 9 9950X3D. Fixed in the ASUS BIOS (CCD /
  X3D Gaming Mode setting). Worth checking on any demo machine, as it
  halves CPU-bound tracker throughput (MediaPipe runs CPU-only on Windows).
- **No standalone CUDA Toolkit needed.** PyTorch CUDA wheels bundle the
  runtime; only a recent NVIDIA driver is required. The RTX 5090
  (Blackwell) needs a cu128+ build — the default Windows wheel is CPU-only,
  which would be a silent CPU fallback.

## 2026-10-09 — Capture layer

- **One `VideoSource` wraps every source kind.** `SourceSpec` describes
  *what* to open (webcam index, file path, stream URL); `VideoSource` owns
  *how*. Callers never branch on source kind.
- **Switch only on success.** `open()` builds the new `VideoCapture` first
  and swaps it in under a lock, so a bad path or dead URL leaves the
  current source running instead of blanking the demo.
- **Webcams use DirectShow (`CAP_DSHOW`).** The Windows default (MSMF) can
  take several seconds to open a camera, which is painful mid-demo.
- **Issue: a dead stream URL froze the app for 30 s.** FFmpeg's default
  open timeout. Fixed by passing `CAP_PROP_OPEN_TIMEOUT_MSEC` and
  `CAP_PROP_READ_TIMEOUT_MSEC` (5 s) when opening streams. Verified:
  `rtsp://127.0.0.1:1/none` now fails in ~5 s and the next source opens.
  The UI must still call `open()` off the UI thread, since 5 s is too long
  to block the window.
- **Verified (OpenCV 5.0.0):** file → file → missing file → dead stream →
  webcam 0 (no camera attached) → file all handled in one process. Failed
  opens raise `OSError` and leave the previous source intact.

## 2026-10-09 — First real test clip (phone, via WhatsApp)

- **Portrait phone video works with no special handling.** OpenCV 5
  applies rotation metadata automatically (`CAP_PROP_ORIENTATION_AUTO` is
  on by default); frames arrive 478×850, upright.
- **WhatsApp recompressed the clip** (110 MB original → 7.1 MB, 478×850
  H.264, 30 fps, 38 s) because it was sent as a video, not a document.
  Fine for testing; send as *Document* when full resolution matters.

## 2026-10-09 — Milestone 2: YOLO pose end to end

- **PyTorch 2.11.0+cu128 from the PyTorch index, not PyPI.** Confirmed
  `sm_120` (Blackwell) in `torch.cuda.get_arch_list()` and the RTX 5090
  visible. `multitracker.device.select_device()` logs a loud warning if
  CUDA is ever unavailable, so a CPU fallback cannot go unnoticed.
- **Model: `yolo26s-pose.pt`** (Ultralytics 8.4.174, YOLO26 small, 23 MB,
  auto-downloaded on first use). On the phone clip: ~5.5 ms median
  inference (first frame ~11 ms warm-up), so the camera, not the GPU,
  limits frame rate.
- **Common output format first.** `TrackerResult` carries detections
  (box, score, label, keypoints, mask) plus the skeleton edges, so the
  renderer draws any tracker's output without knowing which one ran.
- **Seated subject, legs hidden by a table:** leg keypoints fall below the
  0.5 confidence threshold and are not drawn rather than guessed.
- **Licence:** Ultralytics is AGPL-3.0. Fine for teaching; Wonderbeam
  would need an Ultralytics enterprise licence if distributed closed-source.

## 2026-10-09 — Milestone 3: tracker switching + MediaPipe

- **Issue: `pip install mediapipe` failed with "Access is denied" on
  `cv2.pyd`.** A running preview window held the DLL open. Close any
  running app before installing packages.
- **Issue: two OpenCV packages fight over `cv2/`.** MediaPipe depends on
  `opencv-contrib-python`, Ultralytics on `opencv-python`; both install
  the same folder and uninstalling either breaks the other. Fix: keep both
  on the same version (5.0.0.93) and force-reinstall contrib last, since
  it is a superset. `pip check` is clean.
- **MediaPipe 1.1 has no legacy `mp.solutions`** and ships no model files.
  Uses the Tasks API; `.task` models download on first use into `models/`.
- **Skeleton edges moved from `TrackerResult` to each `Detection`.** One
  MediaPipe frame holds hands (21 points) and a face (478 points), which
  need different edges.
- **Lazy tracker imports in `registry.py`.** Choosing YOLO pose never
  imports MediaPipe and vice versa, so modules stay liftable.
- **VIDEO mode needs strictly increasing timestamps**, including across
  file loops and tracker switches. Uses a monotonic clock, bumped by 1 ms
  if two frames land in the same millisecond.
- **Issue: hand labels were swapped.** MediaPipe's docs say handedness
  assumes a mirrored (selfie) image, so the first version swapped labels.
  On real footage that was wrong: with mediapipe 1.1 on unmirrored OpenCV
  frames the raw labels already match the subject's own hands. Swap
  removed; verified on a frame with the subject facing the camera.
- **Issue: face found in 8 of 1153 frames.** Not a confidence problem
  (0.5 vs 0.2 made no difference). The face detector squeezes the whole
  frame into a small square, so a ~50 px face in a portrait 478×850 frame
  vanishes. Fix: run the face landmarker on overlapping square tiles
  (2 tiles for this clip) in IMAGE mode and drop duplicates. Result: 743
  of 1153 frames (64%); most misses are the subject looking down. Decided
  with Mark that body and hands matter more, so this is good enough.
- **Overlay points sized by keypoint density** (area per point in the
  box), so a 478-point face draws as 1 px dots and a 17-point body as
  large joints, without the renderer knowing which tracker ran.

## 2026-10-09 — Milestone 4: YOLOE-26 with live prompts

- **Issue: first YOLOE run crashed with `No module named 'clip'`.** Text
  prompts need Ultralytics' CLIP fork from GitHub. Ultralytics tried to
  pip-install it mid-run and failed because git was not on this shell's
  PATH (installed after the shell started). Installed it explicitly, and
  set `YOLO_AUTOINSTALL=False` in the YOLOE module so Ultralytics can
  never pip-install during a live demo.
- **Prompt changes are queued, applied before the next frame.**
  `set_prompts()` can be called from any thread; the capture loop applies
  it, so the model is never modified mid-inference. First change takes
  ~0.4 s (text encoder warm-up).
- **Issue: coasters barely detected** (3 of 1153 frames; Mark saw the same
  on a laptop). Not a threshold problem: the small model's coaster scores
  sit below 0.25. Tested 12 prompt wordings × model sizes on 116 frames:

  | Prompt | small | large |
  |---|---|---|
  | coaster | 0% | 84% |
  | round coaster | 22% | 95% |
  | cork coaster | 46% | 63% |
  | disc | 22% | 51% |

  Full clip, bottle + cup + coaster prompts together, coaster boxes found
  (2 visible coasters per frame): small ~0%, medium 7%, large 43%,
  x-large 55%. With **"round coaster"**: large **96%**, x-large 66%, and
  x-large mislabelled the bottle as a cup. **Chose `yoloe-26l-seg`**
  (~10 ms/frame on the 5090). Teaching point: wording matters, and a
  more descriptive prompt ("round coaster") beats the bare noun.
- **Laptops:** use `model="yoloe-26s-seg.pt"` for speed, but expect much
  weaker recall on unusual objects.

## 2026-10-09 — Milestone 5a: BodyPose3D with MediaPipe

- **Model choice:** MediaPipe Pose Landmarker first, MotionBERT later.
  MediaPipe returns 3D "world" landmarks (33 points, metres, centred
  between the hips) directly, so it works live with no new dependencies.
  MotionBERT is trained on long clips and looks at frames on both sides
  of the current one, so live use costs delay or quality.
- **`keypoints_3d` added to `Detection`**, matching `keypoints` point for
  point. 2D-only trackers leave it `None`, so nothing else changed.
- **3D views in the overlay:** when any detection has 3D points, the
  renderer draws front and side views in the bottom-right corner. The
  side view shows depth, which the 2D overlay cannot. Still no
  tracker-specific code in the renderer.
- **Results on the phone clip (1153 frames, seated subject, legs hidden
  by a table):**

  | Variant | Frames with a body | Per frame (CPU) |
  |---|---|---|
  | lite | 99% | ~11 ms |
  | full (default) | 97% | ~13 ms |
  | heavy | 100% | ~27 ms |

  Hidden legs get low visibility scores and are left out of both the
  2D overlay and the 3D views.
- **Issue: model downloads cut off at about 4 MB.** The full (9.4 MB) and
  heavy (30.7 MB) pose models failed every time, while `curl` fetched
  them fine. Cause: this PC's internet goes through a phone hotspot,
  which drops long Python downloads. Plain retries failed at the same
  point, so `ensure_model` now resumes `.part` files with an HTTP Range
  request. Both models then completed in two attempts, byte-identical to
  the `curl` copy. Also useful on venue or student Wi-Fi.


## 2026-10-15 — Milestone 6: PySide6 UI

- **Layout:** video in the centre, a Controls panel (source, tracker,
  YOLOE-26 prompts, live status) on the left, and a rotatable 3D skeleton
  panel (pyqtgraph OpenGL) on the right. Status text now lives in the
  panel instead of being drawn on the video, which also ends the
  top-left text overlap. `draw_result(..., inset_3d=False)` drops the
  corner 3D views for the UI; `track_preview.py` is unchanged.
- **Threads:** a `QThread` runs capture -> tracker -> overlay and sends
  finished frames to the window. Opening a source and loading a tracker
  run on background threads, and each change only goes live once it has
  worked, so the video never stops for them.
- **Issue: a 4-5 s freeze on the first frames of a GPU tracker.** CUDA
  start-up happens on the first inference, not at load. Fixed with
  `TrackerSwitcher.warm_up`: the new tracker runs once on the latest real
  frame in the background before `switch` makes it live. Switching now
  pauses the video for under 0.2 s.
- **Issue: access violation in `torch.load` when switching to YOLOE-26.**
  It happened every time after YOLO pose then MediaPipe, with the
  checkpoint loading on a Qt `QThreadPool` thread. A larger thread stack
  did not help, and the same sequence on plain Python threads without Qt
  ran cleanly. Background work now uses `threading.Thread`, with results
  sent back through Qt signals. Verified: three full tours of all four
  trackers with no crash.
