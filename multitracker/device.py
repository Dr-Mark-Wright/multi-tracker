"""GPU selection that refuses to fall back to the CPU silently."""

import logging

import torch

log = logging.getLogger(__name__)


def select_device() -> str:
    """Return ``"cuda:0"`` when a GPU is usable, else warn loudly and use CPU."""
    if torch.cuda.is_available():
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return "cuda:0"
    log.warning("CUDA NOT AVAILABLE: running on CPU. Check that the CUDA "
                "build of PyTorch is installed (torch %s, CUDA %s).",
                torch.__version__, torch.version.cuda)
    return "cpu"
