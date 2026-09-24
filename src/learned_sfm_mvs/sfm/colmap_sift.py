"""COLMAP SIFT backend: the classic pipeline of sfm-mvs-pipeline."""

import logging

from learned_sfm_mvs.sfm.base import (
    SfmInputs,
    SfmResult,
    check_all_images_imported,
    map_and_select,
    prepare_output,
)
from learned_sfm_mvs.sfm.feature_extraction import extract_features
from learned_sfm_mvs.sfm.feature_matching import match_features

logger = logging.getLogger(__name__)


def run(inputs: SfmInputs, colmap_cfg: dict) -> SfmResult:
    """SIFT extraction, COLMAP matching and incremental mapping.

    Parameters
    ----------
    inputs : SfmInputs
        Run inputs.
    colmap_cfg : dict
        Parsed ``configs/colmap.yaml``.

    Returns
    -------
    SfmResult
        The model with most registered images.
    """
    prepare_output(inputs)
    extract_features(
        database_path=inputs.database_path,
        image_dir=inputs.image_dir,
        options=colmap_cfg["feature_extraction"],
        device=inputs.colmap_device,
        camera_model=inputs.camera_model,
        camera_params=inputs.camera_params,
        image_names=inputs.image_names,
        mask_path=inputs.mask_dir,
        shared_camera=inputs.shared_camera,
    )
    check_all_images_imported(inputs)
    match_features(
        database_path=inputs.database_path,
        options=colmap_cfg["feature_matching"],
        device=inputs.colmap_device,
    )
    return map_and_select(inputs, colmap_cfg["incremental_mapping"])
