import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from learned_sfm_mvs.mvs.base import MvsInputs, fuse, run_mvs

REPO = Path(__file__).resolve().parents[2]
COLMAP_CFG = {
    "patch_match_stereo": {"max_image_size": 2000},
    "stereo_fusion": {"min_num_pixels": 5},
}


def _inputs(tmp_path, **kwargs) -> MvsInputs:
    defaults = {
        "sparse_model_path": tmp_path / "sparse" / "0",
        "image_dir": tmp_path / "images",
        "output_dir": tmp_path / "out",
    }
    return MvsInputs(**{**defaults, **kwargs})


# --- MvsInputs ---


def test_inputs_reject_half_bbox(tmp_path):
    with pytest.raises(ValueError, match="together"):
        _inputs(tmp_path, bbox_min=[0, 0, 0])


def test_inputs_reject_unknown_device(tmp_path):
    with pytest.raises(ValueError, match="device"):
        _inputs(tmp_path, device="gpu")


def test_inputs_paths(tmp_path):
    inputs = _inputs(tmp_path)
    assert inputs.mvs_dir == tmp_path / "out" / "mvs"
    assert inputs.dense_ply == tmp_path / "out" / "dense.ply"


def test_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unknown MVS backend"):
        run_mvs("gipuma", _inputs(tmp_path), {})


# --- patchmatch through the shared layer ---


@patch("learned_sfm_mvs.mvs.patchmatch.fuse_depth_maps")
@patch("learned_sfm_mvs.mvs.patchmatch.run_dense_reconstruction")
def test_run_mvs_patchmatch_estimates_then_fuses(mock_dense, mock_fuse, tmp_path):
    mock_fuse.return_value.num_points3D.return_value = 1234
    inputs = _inputs(tmp_path, bbox_min=[-1, -1, -1], bbox_max=[1, 1, 1])

    result = run_mvs("patchmatch", inputs, {"colmap": COLMAP_CFG})

    dense_kwargs = mock_dense.call_args.kwargs
    assert dense_kwargs["mvs_path"] == inputs.mvs_dir
    assert dense_kwargs["options"] == {"max_image_size": 2000}
    fuse_kwargs = mock_fuse.call_args.kwargs
    assert fuse_kwargs["output_path"] == inputs.dense_ply
    assert fuse_kwargs["bbox_min"] == [-1, -1, -1]
    assert fuse_kwargs["mask_path"] is None
    assert result.dense_ply == inputs.dense_ply
    assert result.stats["fusion"] == {"points": 1234}
    assert result.stats["fusion_masks"] is None


@patch("learned_sfm_mvs.mvs.patchmatch.fuse_depth_maps")
@patch("learned_sfm_mvs.mvs.patchmatch.run_dense_reconstruction")
def test_fuse_alone_does_not_recompute_depths(mock_dense, mock_fuse, tmp_path):
    fuse("patchmatch", _inputs(tmp_path), {"colmap": COLMAP_CFG})

    mock_dense.assert_not_called()
    mock_fuse.assert_called_once()


@patch("learned_sfm_mvs.mvs.patchmatch.fuse_depth_maps")
@patch("learned_sfm_mvs.mvs.base.undistort_masks_safe")
def test_fusion_masks_are_warped_and_passed(mock_warp, mock_fuse, tmp_path):
    warped = tmp_path / "out" / "mvs" / "fusion_masks"
    mock_warp.return_value = (warped, {"masks_written": 3})
    inputs = _inputs(tmp_path, mask_dir=tmp_path / "masks", fusion_masks=True)

    result = fuse("patchmatch", inputs, {"colmap": COLMAP_CFG})

    mock_warp.assert_called_once_with(
        mask_path=tmp_path / "masks",
        original_sparse_path=inputs.sparse_model_path,
        mvs_path=inputs.mvs_dir,
    )
    assert mock_fuse.call_args.kwargs["mask_path"] == warped
    assert result.fusion_mask_dir == warped
    assert result.stats["fusion_masks"] == {"masks_written": 3}


@patch("learned_sfm_mvs.mvs.patchmatch.fuse_depth_maps")
@patch("learned_sfm_mvs.mvs.base.undistort_masks_safe")
def test_fusion_masks_without_mask_dir_fuses_unmasked(mock_warp, mock_fuse, tmp_path):
    result = fuse(
        "patchmatch", _inputs(tmp_path, fusion_masks=True), {"colmap": COLMAP_CFG}
    )

    mock_warp.assert_not_called()
    assert mock_fuse.call_args.kwargs["mask_path"] is None
    assert result.fusion_mask_dir is None


@patch("learned_sfm_mvs.mvs.patchmatch.fuse_depth_maps")
@patch("learned_sfm_mvs.mvs.base.undistort_masks_safe")
def test_masks_ignored_unless_fusion_masks_requested(mock_warp, mock_fuse, tmp_path):
    # Masks always apply to features; fusion masking is opt-in (old behaviour).
    fuse(
        "patchmatch", _inputs(tmp_path, mask_dir=tmp_path / "m"), {"colmap": COLMAP_CFG}
    )
    mock_warp.assert_not_called()


# --- transmvsnet through the shared layer (needs torch) ---


def test_run_mvs_transmvsnet_wiring(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    from learned_sfm_mvs.mvs.transmvsnet import backend

    inputs = _inputs(tmp_path, device="cpu")
    stale = backend.depth_dir(inputs) / "00000099.npz"
    stale.parent.mkdir(parents=True)
    stale.touch()
    cfg = {"views": {"num_src": 10}, "fusion": {"min_confidence": 0.03}}
    views = [MagicMock(depth_range_source="all_points")]

    with (
        patch.object(backend, "undistort_workspace") as mock_undistort,
        patch.object(backend.pycolmap, "Reconstruction"),
        patch.object(backend, "build_views", return_value=views) as mock_views,
        patch.object(
            backend, "run_inference", return_value={"depth_maps": 1}
        ) as mock_infer,
        patch.object(
            backend, "fuse_depth_maps", return_value={"points": 7}
        ) as mock_fuse,
    ):
        result = run_mvs("transmvsnet", inputs, {"transmvsnet": cfg})

    mock_undistort.assert_called_once_with(inputs)
    assert mock_views.call_args.args[1] == {"num_src": 10}
    assert mock_views.call_args.kwargs["mask_dir"] is None  # no masks given
    assert not stale.exists()  # previous run's depth maps never mix in
    infer_args = mock_infer.call_args.args
    assert infer_args[0] == inputs.mvs_dir
    assert infer_args[1] is views
    assert infer_args[3] == backend.depth_dir(inputs)
    assert str(infer_args[4]) == "cpu"
    fuse_kwargs = mock_fuse.call_args.kwargs
    assert fuse_kwargs["fusion_cfg"] == {"min_confidence": 0.03}
    assert fuse_kwargs["output_ply"] == inputs.dense_ply
    assert result.stats == {
        "depth": {"depth_maps": 1, "depth_range_sources": {"all_points": 1}},
        "fusion": {"points": 7},
        "fusion_masks": None,
    }


def test_transmvsnet_depth_ranges_use_warped_subject_masks(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    from learned_sfm_mvs.mvs.transmvsnet import backend

    inputs = _inputs(tmp_path, mask_dir=tmp_path / "masks", device="cpu")
    warped = inputs.mvs_dir / "view_masks"
    views = [
        MagicMock(depth_range_source="subject_observed"),
        MagicMock(depth_range_source="subject_projected"),
        MagicMock(depth_range_source="subject_observed"),
    ]

    with (
        patch.object(backend, "undistort_workspace"),
        patch.object(backend.pycolmap, "Reconstruction"),
        patch.object(
            backend, "undistort_masks_safe", return_value=(warped, {})
        ) as mock_warp,
        patch.object(backend, "build_views", return_value=views) as mock_views,
        patch.object(backend, "run_inference", return_value={"depth_maps": 3}),
    ):
        stats = backend.estimate_depths(inputs, {"transmvsnet": {"views": {}}})

    # Warped into their own directory: fusion masks stay opt-in.
    assert mock_warp.call_args.kwargs == {
        "mask_path": tmp_path / "masks",
        "original_sparse_path": inputs.sparse_model_path,
        "mvs_path": inputs.mvs_dir,
        "output_dir_name": "view_masks",
    }
    assert mock_views.call_args.kwargs["mask_dir"] == warped
    assert stats["depth_range_sources"] == {
        "subject_observed": 2,
        "subject_projected": 1,
    }


def test_transmvsnet_without_views_fails_clearly(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    from learned_sfm_mvs.mvs.transmvsnet import backend

    with (
        patch.object(backend, "undistort_workspace"),
        patch.object(backend.pycolmap, "Reconstruction"),
        patch.object(backend, "build_views", return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="No view usable"):
            backend.estimate_depths(_inputs(tmp_path), {"transmvsnet": {"views": {}}})


# --- shipped config ---


def _shipped_cfg() -> dict:
    return yaml.safe_load((REPO / "configs/transmvsnet.yaml").read_text())[
        "transmvsnet"
    ]


def test_shipped_config_matches_checkpoint_training():
    cfg = _shipped_cfg()
    # Values of upstream scripts/test_tnt.sh for model_bld.ckpt.
    assert cfg["model"] == {
        "ndepths": [48, 32, 8],
        "depth_interval_ratios": [4, 1, 0.5],
        "num_depth": 192,
    }
    fusion = cfg["fusion"]
    # A pixel can only agree with the source views fusion looks at.
    assert fusion["min_consistent_views"] <= cfg["views"]["num_src"]


def test_shipped_checkpoint_digest_matches_file():
    cfg = _shipped_cfg()
    checkpoint = REPO / cfg["checkpoint"]
    if not checkpoint.exists():
        pytest.skip("model_bld.ckpt not downloaded")
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() == cfg["checkpoint_sha256"]
    )
