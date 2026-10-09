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
