"""Download model files on first use into the project's ``models`` folder."""

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def ensure_model(url: str, directory: Path = MODELS_DIR) -> Path:
    """Return the local path of a model file, downloading it if missing."""
    path = directory / url.rsplit("/", 1)[-1]
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    return path
