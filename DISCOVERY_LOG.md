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
