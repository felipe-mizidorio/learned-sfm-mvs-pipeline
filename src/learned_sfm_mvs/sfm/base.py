"""Backend-agnostic Structure-from-Motion entry point.

Every backend writes a COLMAP database (``<output>/database.db``) and runs the
same COLMAP incremental mapper into ``<output>/sparse/<n>/``. Downstream stages
(ArUco scale, head crop, MVS undistortion, the resume CLIs) only consume that
layout, so they never need to know which backend produced the model.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap

from learned_sfm_mvs.sfm.images import normalize_camera_params
from learned_sfm_mvs.sfm.reconstruction import run_incremental_mapping

logger = logging.getLogger(__name__)

SFM_BACKENDS = ("hloc", "colmap_sift")
_DEVICES = ("auto", "cpu")


@dataclass(frozen=True)
class SfmInputs:
    """What every SfM backend needs, validated once.

    Parameters
    ----------
    image_dir : Path
        Root image directory.
    output_dir : Path
        Run output directory.
    image_names : list[str]
        Images to reconstruct, relative to ``image_dir``, in capture order.
    mask_dir : Path or None, optional
        COLMAP masks (``<image name>.png``, 0 = ignore) applied to features.
    camera_model : str or None, optional
        COLMAP camera model shared by all images; None self-calibrates.
    camera_params : str or None, optional
        Intrinsics for ``camera_model``, comma- or space-separated.
        Normalized to the comma-separated form COLMAP parses.
    shared_camera : bool, optional
        One camera for all images (default): right for same-device captures,
        but COLMAP then skips images of a different size. False gives each
        image its own camera, for mixed-camera sets. Ignored when
        ``camera_model`` is set, which always describes one camera.
    device : str, optional
        ``"auto"`` (CUDA when available) or ``"cpu"``.

    Raises
    ------
    ValueError
        On an empty image list, unknown device, or params without a model.
    """

    image_dir: Path
    output_dir: Path
    image_names: list[str]
    mask_dir: Path | None = None
    camera_model: str | None = None
    camera_params: str | None = None
    shared_camera: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.image_names:
            raise ValueError("image_names is empty")
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {self.device!r}")
        if self.camera_params is not None:
            if self.camera_model is None:
                raise ValueError("camera_params given without camera_model")
            # Frozen dataclass: normalize through object.__setattr__.
            object.__setattr__(
                self, "camera_params", normalize_camera_params(self.camera_params)
            )

    @property
    def database_path(self) -> Path:
        """COLMAP database path."""
        return self.output_dir / "database.db"

    @property
    def sparse_dir(self) -> Path:
        """Directory of numbered sparse models."""
        return self.output_dir / "sparse"

    @property
    def camera_mode(self) -> pycolmap.CameraMode:
        """COLMAP camera mode for importing the images."""
        if self.shared_camera or self.camera_model is not None:
            return pycolmap.CameraMode.SINGLE
        return pycolmap.CameraMode.AUTO

    @property
    def colmap_device(self) -> pycolmap.Device:
        """``device`` as a pycolmap enum."""
        return pycolmap.Device.cpu if self.device == "cpu" else pycolmap.Device.auto


@dataclass(frozen=True)
class SfmResult:
    """Output of an SfM backend.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        Model with the most registered images.
    model_path : Path
        Its directory (``<output>/sparse/<n>``), input to MVS undistortion.
    num_models : int
        Number of models the mapper produced.
    stats : dict, optional
        Backend-specific numbers for the manifest.
    """

    reconstruction: pycolmap.Reconstruction
    model_path: Path
    num_models: int
    stats: dict = field(default_factory=dict)


def prepare_output(inputs: SfmInputs) -> None:
    """Create the output dir and remove a previous run's database and models.

    Stale models are a correctness problem, not clutter: a previous run with
    more models leaves ``sparse/<n>`` directories this run does not overwrite,
    and ``load_best_reconstruction`` (used by the resume CLIs) would pick them.

    Parameters
    ----------
    inputs : SfmInputs
        Run inputs.
    """
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    if inputs.database_path.exists():
        logger.warning("Removing stale database '%s'", inputs.database_path)
        inputs.database_path.unlink()
    if inputs.sparse_dir.exists():
        logger.warning("Removing stale sparse models '%s'", inputs.sparse_dir)
        shutil.rmtree(inputs.sparse_dir)


def check_all_images_imported(inputs: SfmInputs) -> None:
    """Fail if COLMAP skipped any requested image while importing.

    COLMAP only logs a skipped image (e.g. a size mismatch with a shared
    camera) and carries on, so a run could silently lose most of its frames.

    Parameters
    ----------
    inputs : SfmInputs
        Run inputs; the database must exist.

    Raises
    ------
    RuntimeError
        If any image in ``inputs.image_names`` is missing from the database.
    """
    with pycolmap.Database.open(inputs.database_path) as db:
        imported = {image.name for image in db.read_all_images()}
    missing = [name for name in inputs.image_names if name not in imported]
    if missing:
        raise RuntimeError(
            f"COLMAP did not import {len(missing)} of {len(inputs.image_names)} "
            f"images (first: {missing[:3]}). With shared_camera=True all images "
            "must have the same size; use shared_camera=False for mixed "
            "cameras. The COLMAP log above names the reason per image."
        )


def map_and_select(inputs: SfmInputs, mapping_options: dict) -> SfmResult:
    """Run incremental mapping and keep the model with most registered images.

    Parameters
    ----------
    inputs : SfmInputs
        Run inputs; the database must hold features and verified matches.
    mapping_options : dict
        ``incremental_mapping`` section of ``configs/colmap.yaml``.

    Returns
    -------
    SfmResult
        The selected model.

    Raises
    ------
    RuntimeError
        If no model could be reconstructed.
    """
    reconstructions = run_incremental_mapping(
        database_path=inputs.database_path,
        image_dir=inputs.image_dir,
        output_path=inputs.sparse_dir,
        options=mapping_options,
        device=inputs.colmap_device,
    )
    if not reconstructions:
        raise RuntimeError("No sparse model reconstructed. Check features and matches.")

    best = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    logger.info(
        "Using sparse model %d (%d/%d images registered, %d model(s))",
        best,
        reconstructions[best].num_reg_images(),
        len(inputs.image_names),
        len(reconstructions),
    )
    return SfmResult(
        reconstruction=reconstructions[best],
        model_path=inputs.sparse_dir / str(best),
        num_models=len(reconstructions),
    )


def run_sfm(backend: str, inputs: SfmInputs, configs: dict) -> SfmResult:
    """Run the selected SfM backend.

    Parameters
    ----------
    backend : str
        One of ``SFM_BACKENDS``.
    inputs : SfmInputs
        Run inputs.
    configs : dict
        ``{"colmap": <colmap.yaml>, "hloc": <hloc.yaml "hloc" section>}``.
        Both backends map with ``colmap["incremental_mapping"]``.

    Returns
    -------
    SfmResult
        The selected model.

    Raises
    ------
    ValueError
        If ``backend`` is unknown.
    """
    logger.info("SfM backend: %s", backend)
    # Imported here: hloc_backend pulls in torch only when actually used.
    if backend == "colmap_sift":
        from learned_sfm_mvs.sfm import colmap_sift

        return colmap_sift.run(inputs, configs["colmap"])
    if backend == "hloc":
        from learned_sfm_mvs.sfm import hloc_backend

        return hloc_backend.run(
            inputs, configs["hloc"], configs["colmap"]["incremental_mapping"]
        )
    raise ValueError(
        f"Unknown SfM backend {backend!r}. Expected one of {SFM_BACKENDS}."
    )
