"""Run provenance for learned backends: environment, backends, stage timings."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata

logger = logging.getLogger(__name__)


def learned_environment() -> dict:
    """Versions and GPU of the learned stack; None where not installed.

    Returns
    -------
    dict
        ``torch``, ``cuda``, ``cudnn``, ``gpu`` and ``hloc`` entries.
    """
    info: dict = {"torch": None, "cuda": None, "cudnn": None, "gpu": None, "hloc": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        info["gpu"] = f"{torch.cuda.get_device_name()} (sm_{major}{minor})"
    try:
        info["hloc"] = metadata.version("hloc")
    except metadata.PackageNotFoundError:
        pass
    return info


class StageTimer:
    """Wall time and torch peak VRAM per pipeline stage.

    Peak VRAM counts torch allocations only (hloc, TransMVSNet); COLMAP's own
    CUDA memory (SIFT, PatchMatch) is not visible to torch.

    Examples
    --------
    >>> timer = StageTimer()
    >>> with timer("sfm"):
    ...     pass
    >>> sorted(timer.stages["sfm"])
    ['seconds', 'torch_peak_vram_gb']
    """

    def __init__(self) -> None:
        self.stages: dict[str, dict] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        """Time the enclosed block as stage ``name``.

        Parameters
        ----------
        name : str
            Stage name in the manifest.

        Yields
        ------
        None
        """
        cuda = _cuda()
        if cuda is not None:
            cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield
        finally:
            peak = None
            if cuda is not None:
                peak = round(cuda.max_memory_allocated() / 2**30, 2)
            self.stages[name] = {
                "seconds": round(time.perf_counter() - start, 1),
                "torch_peak_vram_gb": peak,
            }
            logger.info("Stage %s: %s", name, self.stages[name])


def _cuda():
    """``torch.cuda`` when a GPU is usable, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def with_backend_provenance(
    provenance: dict,
    sfm: dict | None,
    mvs: dict | None,
    stage_timings: dict,
) -> dict:
    """Record the backends, their stats and stage timings.

    Parameters
    ----------
    provenance : dict
        Provenance block to update in place.
    sfm : dict or None
        ``{"name": ..., **stats}`` of the SfM backend; None when not run.
    mvs : dict or None
        ``{"name": ..., **stats}`` of the MVS backend; None when not run.
    stage_timings : dict
        ``StageTimer.stages``.

    Returns
    -------
    dict
        ``provenance``, for chaining.
    """
    provenance["backends"] = {"sfm": sfm, "mvs": mvs}
    provenance["stage_timings"] = stage_timings
    provenance.setdefault("environment", {}).update(learned_environment())
    return provenance
