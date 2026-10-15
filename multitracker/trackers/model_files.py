"""Download model files on first use into the project's ``models`` folder."""

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _download(url: str, partial: Path) -> bool:
    """Append the rest of ``url`` to ``partial``. True when it is complete."""
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status == 200:          # server ignored the range
            have = 0
        total = have + int(response.headers.get("Content-Length", 0))
        with partial.open("ab" if have else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
    return partial.stat().st_size >= total


def ensure_model(url: str, directory: Path = MODELS_DIR,
                 attempts: int = 6, filename: str | None = None) -> Path:
    """Return the local path of a model file, downloading it if missing.

    Interrupted downloads resume where they stopped: Python downloads of
    larger files from Google's model bucket were seen to cut off at about
    4 MB every time (see DISCOVERY_LOG.md). ``filename`` names the local
    copy when the URL's own name is generic (e.g. ``best_epoch.bin``).
    """
    path = directory / (filename or url.rsplit("/", 1)[-1])
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, attempts + 1):
        log.info("Downloading %s (attempt %d of %d)", url, attempt, attempts)
        try:
            if _download(url, partial):
                partial.replace(path)
                return path
        except urllib.error.HTTPError as error:
            if error.code == 416:           # nothing left: already complete
                partial.replace(path)
                return path
            raise
        except OSError as error:
            log.warning("Download interrupted: %s", error)
        time.sleep(min(attempt, 3))
    raise OSError(f"Could not download {url} after {attempts} attempts")
