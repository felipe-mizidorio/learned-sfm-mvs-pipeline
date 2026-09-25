"""TransMVSNet backend: learned depth maps, fused with consistency checks."""

import logging
import shutil
from pathlib import Path

import pycolmap
import torch

from learned_sfm_mvs.mvs.base import MvsInputs, undistort_workspace
from learned_sfm_mvs.mvs.transmvsnet.fusion import fuse_depth_maps
from learned_sfm_mvs.mvs.transmvsnet.inference import run_inference
from learned_sfm_mvs.mvs.transmvsnet.views import build_views

logger = logging.getLogger(__name__)


def depth_dir(inputs: MvsInputs) -> Path:
    """Where the per-view ``<image_id>.npz`` depth maps live."""
    return inputs.mvs_dir / "transmvsnet"


def select_device(inputs: MvsInputs) -> torch.device:
    """CUDA unless ``cpu`` is requested or no GPU is visible."""
    if inputs.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        logger.warning("No CUDA device: TransMVSNet runs on CPU and will be slow.")
        return torch.device("cpu")
    return torch.device("cuda")


def estimate_depths(inputs: MvsInputs, configs: dict) -> dict:
    """Undistort, prepare views and run TransMVSNet on every view.

    Parameters
    ----------
    inputs : MvsInputs
        Run inputs.
    configs : dict
        Must hold ``transmvsnet`` (``configs/transmvsnet.yaml`` section).

    Returns
    -------
    dict
        Inference stats (checkpoint digest, input size, counts, timing).

    Raises
    ------
    RuntimeError
        If no view has enough sparse points for a depth range.
    """
    cfg = configs["transmvsnet"]
    undistort_workspace(inputs)
    views = build_views(
        pycolmap.Reconstruction(inputs.mvs_dir / "sparse"), cfg["views"]
    )
    if not views:
        raise RuntimeError("No view usable for TransMVSNet (see warnings above).")

    out = depth_dir(inputs)
    if out.exists():
        # Fusion reads every npz present; a previous run's views must not mix in.
        logger.warning("Removing stale TransMVSNet depth maps '%s'", out)
        shutil.rmtree(out)
    return run_inference(inputs.mvs_dir, views, cfg, out, select_device(inputs))


def fuse(inputs: MvsInputs, configs: dict, fusion_mask_dir: Path | None) -> dict:
    """Fuse the saved depth maps into ``inputs.dense_ply``.

    Parameters
    ----------
    inputs : MvsInputs
        Run inputs.
    configs : dict
        Must hold ``transmvsnet``.
    fusion_mask_dir : Path or None
        Masks aligned to the undistorted images.

    Returns
    -------
    dict
        Point counts.
    """
    return fuse_depth_maps(
        depth_dir=depth_dir(inputs),
        workspace=inputs.mvs_dir,
        output_ply=inputs.dense_ply,
        fusion_cfg=configs["transmvsnet"]["fusion"],
        device=select_device(inputs),
        mask_dir=fusion_mask_dir,
        bbox_min=inputs.bbox_min,
        bbox_max=inputs.bbox_max,
    )
