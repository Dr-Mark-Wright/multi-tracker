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
