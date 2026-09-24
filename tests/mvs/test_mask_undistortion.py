from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from learned_sfm_mvs.mvs.mask_undistortion import (
    undistort_masks_safe,
    undistortion_maps,
)
from learned_sfm_mvs.pipeline.orchestration import with_fusion_mask_provenance

def _camera(model_name: str, width: int, height: int, params: list[float]) -> MagicMock:
    camera = MagicMock()
    camera.model.name = model_name
    camera.width = width
    camera.height = height
    camera.params = params
    return camera


# --- undistortion_maps ---


def test_identity_when_no_distortion_and_same_intrinsics():
    cam = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, map_y = undistortion_maps(cam, cam)

    u, v = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(48, dtype=np.float32))
    np.testing.assert_allclose(map_x, u, atol=1e-4)
    np.testing.assert_allclose(map_y, v, atol=1e-4)


def test_principal_point_is_fixed_point_of_radial_distortion():
    # At the principal point r=0, so distortion has no effect regardless of k.
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, map_y = undistortion_maps(original, undistorted)

    assert map_x[24, 32] == pytest.approx(32.0, abs=1e-4)
    assert map_y[24, 32] == pytest.approx(24.0, abs=1e-4)


def test_negative_k_pulls_border_samples_inward():
    # Barrel distortion (k < 0): the original image content is compressed
    # toward the centre, so undistorted border pixels must sample INSIDE
    # the original frame (map value < pixel coordinate at the right edge).
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    map_x, _ = undistortion_maps(original, undistorted)

    assert map_x[24, 63] < 63.0


def test_unsupported_model_raises():
    original = _camera("OPENCV_FISHEYE", 64, 48, [50.0, 50.0, 32.0, 24.0, 0, 0, 0, 0])
    undistorted = _camera("PINHOLE", 64, 48, [50.0, 50.0, 32.0, 24.0])
    with pytest.raises(ValueError, match="Unsupported camera model"):
        undistortion_maps(original, undistorted)


def test_distorted_target_camera_rejected():
    original = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, -0.2])
    not_pinhole = _camera("SIMPLE_RADIAL", 64, 48, [50.0, 32.0, 24.0, 0.1])
    with pytest.raises(ValueError, match="distortion-free"):
        undistortion_maps(original, not_pinhole)


# --- degradation behaviour ---


@patch("learned_sfm_mvs.mvs.mask_undistortion.undistort_masks")
def test_unsupported_model_degrades_to_unmasked_instead_of_aborting(
    mock_undistort, tmp_path
):
    # Warping runs after PatchMatch Stereo; raising would throw away the GPU stage.
    mock_undistort.side_effect = ValueError("Unsupported camera model: OPENCV_FISHEYE")

    out_dir, stats = undistort_masks_safe(
        mask_path=tmp_path / "masks",
        original_sparse_path=tmp_path / "sparse",
        mvs_path=tmp_path / "mvs",
    )

    assert out_dir is None
    assert stats is not None
    assert "OPENCV_FISHEYE" in stats["failure"]


# --- provenance ---


def test_provenance_records_fusion_masks_disabled_explicitly():
    # An absent key is indistinguishable from a pre-feature run.
    provenance = with_fusion_mask_provenance({}, enabled=False)

    assert provenance["fusion_masks"] == {"enabled": False}


def test_provenance_records_mask_dirs_and_stats_when_enabled(tmp_path):
    provenance = with_fusion_mask_provenance(
        {},
        enabled=True,
        source_mask_dir=tmp_path / "masks",
        workspace_mask_dir=tmp_path / "mvs" / "fusion_masks",
        stats={"masks_written": 266, "masks_missing": 0},
    )

    block = provenance["fusion_masks"]
    assert block["enabled"] is True
    assert block["masks_written"] == 266
    assert block["source_mask_dir"] == str(tmp_path / "masks")
