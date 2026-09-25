"""Backend-agnostic Multi-View Stereo entry point.

Every backend works in the undistorted COLMAP workspace ``<output>/mvs`` and
writes ``<output>/dense.ply`` (points, normals, colours). Everything after
fusion (SOR, scale recovery, head crop, Poisson) only consumes that file.

Depth estimation and fusion are separate steps so ``fuse`` can re-run with
other thresholds, masks or clipping on existing depth maps (resume-mvs).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pycolmap

from learned_sfm_mvs.mvs.mask_undistortion import undistort_masks_safe

logger = logging.getLogger(__name__)

MVS_BACKENDS = ("transmvsnet", "patchmatch")
_DEVICES = ("auto", "cpu")


@dataclass(frozen=True)
class MvsInputs:
    """What every MVS backend needs, validated once.

    Parameters
    ----------
    sparse_model_path : Path
        Sparse model from SfM (``SfmResult.model_path``).
    image_dir : Path
        Root image directory.
    output_dir : Path
        Run output directory.
    mask_dir : Path or None, optional
        Original-frame masks (COLMAP ``<image name>.png``).
    fusion_masks : bool, optional
        Warp ``mask_dir`` into the workspace and restrict fusion to it.
    bbox_min, bbox_max : list[float] or None, optional
        Axis-aligned box clipping fused points; both or neither.
    device : str, optional
        ``"auto"`` (CUDA when available) or ``"cpu"``.

    Raises
    ------
    ValueError
        On a half-specified bbox or an unknown device.
    """

    sparse_model_path: Path
    image_dir: Path
    output_dir: Path
    mask_dir: Path | None = None
    fusion_masks: bool = False
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if (self.bbox_min is None) != (self.bbox_max is None):
            raise ValueError("bbox_min and bbox_max must be given together")
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {self.device!r}")

    @property
    def mvs_dir(self) -> Path:
        """Undistorted MVS workspace."""
        return self.output_dir / "mvs"

    @property
    def dense_ply(self) -> Path:
        """Fused dense cloud."""
        return self.output_dir / "dense.ply"


@dataclass(frozen=True)
class MvsResult:
    """Output of an MVS run.

    Parameters
    ----------
    dense_ply : Path
        Fused cloud.
    fusion_mask_dir : Path or None
        Warped masks fusion used, or None when fusion was unmasked.
    stats : dict, optional
        ``depth``, ``fusion`` and ``fusion_masks`` numbers for the manifest.
    """

    dense_ply: Path
    fusion_mask_dir: Path | None
    stats: dict = field(default_factory=dict)


def undistort_workspace(inputs: MvsInputs) -> None:
    """Undistort the registered images into ``inputs.mvs_dir`` (pinhole).

    Parameters
    ----------
    inputs : MvsInputs
        Run inputs.

    Raises
    ------
    FileNotFoundError
        If the sparse model or image directory does not exist.
    """
    if not inputs.sparse_model_path.exists():
        raise FileNotFoundError(f"Sparse model not found: {inputs.sparse_model_path}")
    if not inputs.image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {inputs.image_dir}")
    inputs.mvs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Undistorting images into MVS workspace '%s'", inputs.mvs_dir)
    pycolmap.undistort_images(
        output_path=inputs.mvs_dir,
        input_path=inputs.sparse_model_path,
        image_path=inputs.image_dir,
    )


def _backend(name: str) -> ModuleType:
    # Imported here: the transmvsnet backend pulls in torch only when used.
    if name == "patchmatch":
        from learned_sfm_mvs.mvs import patchmatch

        return patchmatch
    if name == "transmvsnet":
        from learned_sfm_mvs.mvs.transmvsnet import backend

        return backend
    raise ValueError(f"Unknown MVS backend {name!r}. Expected one of {MVS_BACKENDS}.")


def run_mvs(backend: str, inputs: MvsInputs, configs: dict) -> MvsResult:
    """Estimate depth maps with the selected backend, then fuse them.

    Parameters
    ----------
    backend : str
        One of ``MVS_BACKENDS``.
    inputs : MvsInputs
        Run inputs.
    configs : dict
        ``{"colmap": <colmap.yaml>, "transmvsnet": <transmvsnet.yaml section>}``.

    Returns
    -------
    MvsResult
        The fused cloud and stats.

    Raises
    ------
    ValueError
        If ``backend`` is unknown.
    """
    module = _backend(backend)
    logger.info("MVS backend: %s", backend)
    depth_stats = module.estimate_depths(inputs, configs)
    result = fuse(backend, inputs, configs)
    return MvsResult(
        result.dense_ply, result.fusion_mask_dir, {"depth": depth_stats, **result.stats}
    )


def fuse(backend: str, inputs: MvsInputs, configs: dict) -> MvsResult:
    """Fuse existing depth maps of the selected backend.

    Fusion masks are warped here, after depth estimation, because they must
    align with the undistorted workspace images. A failed warp degrades to
    unmasked fusion (see ``undistort_masks_safe``).

    Parameters
    ----------
    backend : str
        One of ``MVS_BACKENDS``.
    inputs : MvsInputs
        Run inputs; the workspace must hold the backend's depth maps.
    configs : dict
        As for ``run_mvs``.

    Returns
    -------
    MvsResult
        The fused cloud and stats.
    """
    module = _backend(backend)
    fusion_mask_dir, mask_stats = None, None
    if inputs.fusion_masks and inputs.mask_dir is None:
        logger.warning(
            "fusion_masks requested but no mask directory is available; "
            "fusing unmasked."
        )
    elif inputs.fusion_masks and inputs.mask_dir is not None:
        fusion_mask_dir, mask_stats = undistort_masks_safe(
            mask_path=inputs.mask_dir,
            original_sparse_path=inputs.sparse_model_path,
            mvs_path=inputs.mvs_dir,
        )
    fusion_stats = module.fuse(inputs, configs, fusion_mask_dir)
    return MvsResult(
        inputs.dense_ply,
        fusion_mask_dir,
        {"fusion": fusion_stats, "fusion_masks": mask_stats},
    )
