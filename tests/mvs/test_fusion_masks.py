"""Fusion mask wiring (deferred from the Phase 2 port until fusion existed)."""

from unittest.mock import MagicMock, patch

from learned_sfm_mvs.mvs.fusion import fuse_depth_maps

_FUSION_OPTIONS = {
    "min_num_pixels": 5,
    "max_reproj_error": 2.0,
}


# --- fusion mask wiring ---


@patch("learned_sfm_mvs.mvs.fusion.pycolmap.StereoFusionOptions")
@patch("learned_sfm_mvs.mvs.fusion.pycolmap.stereo_fusion")
def test_fusion_receives_mask_path(mock_stereo_fusion, mock_fusion_opts, tmp_path):
    mvs_path = tmp_path / "mvs"
    depth_maps_dir = mvs_path / "stereo" / "depth_maps"
    depth_maps_dir.mkdir(parents=True)
    (depth_maps_dir / "image0.photometric.bin").touch()
    mask_dir = mvs_path / "fusion_masks"
    mask_dir.mkdir()
    output_path = tmp_path / "dense.ply"

    def _create_ply(*args, **kwargs):
        output_path.touch()
        return MagicMock()

    mock_stereo_fusion.side_effect = _create_ply

    fuse_depth_maps(
        mvs_path=mvs_path,
        output_path=output_path,
        options=_FUSION_OPTIONS,
        mask_path=mask_dir,
    )

    assert mock_fusion_opts.return_value.mask_path == str(mask_dir)


@patch("learned_sfm_mvs.mvs.fusion.pycolmap.StereoFusionOptions")
@patch("learned_sfm_mvs.mvs.fusion.pycolmap.stereo_fusion")
def test_fusion_mask_path_not_set_by_default(
    mock_stereo_fusion, mock_fusion_opts, tmp_path
):
    mvs_path = tmp_path / "mvs"
    depth_maps_dir = mvs_path / "stereo" / "depth_maps"
    depth_maps_dir.mkdir(parents=True)
    (depth_maps_dir / "image0.photometric.bin").touch()
    output_path = tmp_path / "dense.ply"

    def _create_ply(*args, **kwargs):
        output_path.touch()
        return MagicMock()

    mock_stereo_fusion.side_effect = _create_ply
    options_obj = mock_fusion_opts.return_value
    # Simulate the real attribute default so we can detect accidental writes.
    options_obj.mask_path = ""

    fuse_depth_maps(mvs_path=mvs_path, output_path=output_path, options=_FUSION_OPTIONS)

    assert options_obj.mask_path == ""
