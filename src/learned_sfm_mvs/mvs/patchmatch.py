"""COLMAP PatchMatch Stereo backend: the classic MVS of sfm-mvs-pipeline."""

from pathlib import Path

from learned_sfm_mvs.mvs.base import MvsInputs
from learned_sfm_mvs.mvs.dense_reconstruction import run_dense_reconstruction
from learned_sfm_mvs.mvs.fusion import fuse_depth_maps


def estimate_depths(inputs: MvsInputs, configs: dict) -> dict:
    """Undistort and run PatchMatch Stereo (needs CUDA).

    Parameters
    ----------
    inputs : MvsInputs
        Run inputs.
    configs : dict
        Must hold ``colmap`` (parsed ``configs/colmap.yaml``).

    Returns
    -------
    dict
        Empty: COLMAP reports its own timings in the log.
    """
    run_dense_reconstruction(
        sparse_path=inputs.sparse_model_path,
        image_dir=inputs.image_dir,
        mvs_path=inputs.mvs_dir,
        options=configs["colmap"]["patch_match_stereo"],
    )
    return {}


def fuse(inputs: MvsInputs, configs: dict, fusion_mask_dir: Path | None) -> dict:
    """COLMAP stereo fusion into ``inputs.dense_ply``.

    Parameters
    ----------
    inputs : MvsInputs
        Run inputs.
    configs : dict
        Must hold ``colmap``.
    fusion_mask_dir : Path or None
        Masks aligned to the undistorted images.

    Returns
    -------
    dict
        Number of fused points.
    """
    fused = fuse_depth_maps(
        mvs_path=inputs.mvs_dir,
        output_path=inputs.dense_ply,
        options=configs["colmap"]["stereo_fusion"],
        bbox_min=inputs.bbox_min,
        bbox_max=inputs.bbox_max,
        mask_path=fusion_mask_dir,
    )
    return {"points": fused.num_points3D()}
