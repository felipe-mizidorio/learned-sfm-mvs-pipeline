from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pycolmap
import pytest

from learned_sfm_mvs.sfm.base import (
    SfmInputs,
    check_all_images_imported,
    map_and_select,
    prepare_output,
    run_sfm,
)


def _inputs(tmp_path, **kwargs) -> SfmInputs:
    defaults = {
        "image_dir": tmp_path / "images",
        "output_dir": tmp_path / "out",
        "image_names": ["f1.jpg", "f2.jpg"],
    }
    return SfmInputs(**{**defaults, **kwargs})


def _model(num_reg: int) -> MagicMock:
    model = MagicMock()
    model.num_reg_images.return_value = num_reg
    return model


# --- SfmInputs ---


def test_inputs_normalize_space_separated_params(tmp_path):
    inputs = _inputs(tmp_path, camera_model="PINHOLE", camera_params="100 100 40 32")
    assert inputs.camera_params == "100,100,40,32"


def test_inputs_reject_params_without_model(tmp_path):
    with pytest.raises(ValueError, match="without camera_model"):
        _inputs(tmp_path, camera_params="100,100,40,32")


def test_inputs_reject_empty_image_list(tmp_path):
    with pytest.raises(ValueError, match="image_names is empty"):
        _inputs(tmp_path, image_names=[])


def test_inputs_reject_unknown_device(tmp_path):
    with pytest.raises(ValueError, match="device"):
        _inputs(tmp_path, device="cuda:1")


def test_inputs_colmap_device(tmp_path):
    assert _inputs(tmp_path).colmap_device == pycolmap.Device.auto
    assert _inputs(tmp_path, device="cpu").colmap_device == pycolmap.Device.cpu


# --- prepare_output ---


def test_prepare_output_removes_stale_database_and_models(tmp_path):
    inputs = _inputs(tmp_path)
    (inputs.sparse_dir / "3").mkdir(parents=True)
    inputs.database_path.touch()
    (inputs.output_dir / "dense.ply").touch()

    prepare_output(inputs)

    assert not inputs.database_path.exists()
    assert not inputs.sparse_dir.exists()
    # Only SfM outputs are cleared.
    assert (inputs.output_dir / "dense.ply").exists()


# --- map_and_select ---


@patch("learned_sfm_mvs.sfm.base.run_incremental_mapping")
def test_map_and_select_picks_most_registered(mock_map, tmp_path):
    mock_map.return_value = {0: _model(5), 1: _model(12), 2: _model(3)}
    inputs = _inputs(tmp_path, device="cpu")

    result = map_and_select(inputs, {"min_num_matches": 15})

    assert result.model_path == inputs.sparse_dir / "1"
    assert result.num_models == 3
    assert result.reconstruction is mock_map.return_value[1]
    kwargs = mock_map.call_args.kwargs
    assert kwargs["output_path"] == inputs.sparse_dir
    assert kwargs["options"] == {"min_num_matches": 15}
    assert kwargs["device"] == pycolmap.Device.cpu


@patch("learned_sfm_mvs.sfm.base.run_incremental_mapping", return_value={})
def test_map_and_select_raises_without_model(_, tmp_path):
    with pytest.raises(RuntimeError, match="No sparse model"):
        map_and_select(_inputs(tmp_path), {})


# --- run_sfm dispatch ---


def test_run_sfm_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unknown SfM backend"):
        run_sfm("orb_slam", _inputs(tmp_path), {})


@patch("learned_sfm_mvs.sfm.colmap_sift.map_and_select")
@patch("learned_sfm_mvs.sfm.colmap_sift.check_all_images_imported")
@patch("learned_sfm_mvs.sfm.colmap_sift.match_features")
@patch("learned_sfm_mvs.sfm.colmap_sift.extract_features")
def test_run_sfm_colmap_sift(mock_extract, mock_match, mock_check, mock_map, tmp_path):
    colmap_cfg = {
        "feature_extraction": {"max_num_features": 8192},
        "feature_matching": {"method": "sequential"},
        "incremental_mapping": {"min_num_matches": 15},
    }
    inputs = _inputs(
        tmp_path,
        mask_dir=tmp_path / "masks",
        camera_model="PINHOLE",
        camera_params="1 1 2 2",
    )

    result = run_sfm("colmap_sift", inputs, {"colmap": colmap_cfg})

    extract_kwargs = mock_extract.call_args.kwargs
    assert extract_kwargs["image_names"] == ["f1.jpg", "f2.jpg"]
    assert extract_kwargs["mask_path"] == tmp_path / "masks"
    assert extract_kwargs["camera_params"] == "1,1,2,2"
    assert extract_kwargs["shared_camera"] is True
    mock_check.assert_called_once_with(inputs)
    assert mock_match.call_args.kwargs["options"] == {"method": "sequential"}
    mock_map.assert_called_once_with(inputs, {"min_num_matches": 15})
    assert result is mock_map.return_value


@patch("learned_sfm_mvs.sfm.hloc_backend.run")
def test_run_sfm_hloc_uses_colmap_mapping_options(mock_run, tmp_path):
    configs = {
        "colmap": {"incremental_mapping": {"min_num_matches": 15}},
        "hloc": {"features": "aliked-n16"},
    }
    inputs = _inputs(tmp_path)

    run_sfm("hloc", inputs, configs)

    mock_run.assert_called_once_with(
        inputs, {"features": "aliked-n16"}, {"min_num_matches": 15}
    )


# --- image import check (real pycolmap) ---


def _two_sizes(tmp_path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "a.png"), np.zeros((48, 64), np.uint8))
    cv2.imwrite(str(image_dir / "b.png"), np.zeros((60, 80), np.uint8))
    return image_dir


def _import(inputs: SfmInputs) -> None:
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.Database.open(inputs.database_path).close()
    pycolmap.import_images(
        inputs.database_path,
        inputs.image_dir,
        inputs.camera_mode,
        image_names=inputs.image_names,
    )


def test_single_camera_drop_is_reported_not_silent(tmp_path):
    # COLMAP skips images whose size differs from the shared camera and only
    # logs it; the run must stop with the reason instead.
    inputs = _inputs(
        tmp_path, image_dir=_two_sizes(tmp_path), image_names=["a.png", "b.png"]
    )
    _import(inputs)

    with pytest.raises(RuntimeError, match=r"1 of 2 images.*b\.png.*shared_camera"):
        check_all_images_imported(inputs)


def test_per_image_cameras_import_mixed_sizes(tmp_path):
    inputs = _inputs(
        tmp_path,
        image_dir=_two_sizes(tmp_path),
        image_names=["a.png", "b.png"],
        shared_camera=False,
    )
    _import(inputs)

    check_all_images_imported(inputs)


def test_camera_mode(tmp_path):
    assert _inputs(tmp_path).camera_mode == pycolmap.CameraMode.SINGLE
    assert (
        _inputs(tmp_path, shared_camera=False).camera_mode == pycolmap.CameraMode.AUTO
    )
    # Explicit intrinsics describe one physical camera.
    assert (
        _inputs(tmp_path, shared_camera=False, camera_model="PINHOLE").camera_mode
        == pycolmap.CameraMode.SINGLE
    )
