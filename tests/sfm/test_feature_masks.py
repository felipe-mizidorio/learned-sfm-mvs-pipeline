import cv2
import h5py
import numpy as np
import pytest

from learned_sfm_mvs.sfm.feature_masks import filter_features_by_mask

WIDTH, HEIGHT = 40, 30
DESC_DIM = 8


def _write_features(path, name, keypoints):
    """Mimic hloc's layout: (N, 2) keypoints, (N,) scores, (D, N) descriptors."""
    n = len(keypoints)
    with h5py.File(path, "a") as fd:
        grp = fd.create_group(name)
        kp = grp.create_dataset("keypoints", data=np.asarray(keypoints, np.float16))
        kp.attrs["uncertainty"] = 1.5
        grp.create_dataset("keypoint_scores", data=np.arange(n, dtype=np.float16))
        desc = np.tile(np.arange(n, dtype=np.float16), (DESC_DIM, 1))
        grp.create_dataset("descriptors", data=desc)
        grp.create_dataset("image_size", data=np.array([WIDTH, HEIGHT]))


def _read(path, name, key) -> np.ndarray:
    with h5py.File(path) as fd:
        dataset = fd[f"{name}/{key}"]
        assert isinstance(dataset, h5py.Dataset)
        return dataset[()]


def _write_left_half_mask(mask_dir, name):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[:, : WIDTH // 2] = 255
    path = mask_dir / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def test_keeps_only_keypoints_on_non_zero_mask_pixels(tmp_path):
    features = tmp_path / "features.h5"
    # x = 5 and 19 are in the left half; 25 and 35 are not.
    _write_features(features, "f1.jpg", [[5, 3], [25, 3], [19, 29], [35, 10]])
    _write_left_half_mask(tmp_path / "masks", "f1.jpg")

    stats = filter_features_by_mask(features, ["f1.jpg"], tmp_path / "masks")

    np.testing.assert_array_equal(
        _read(features, "f1.jpg", "keypoints"), [[5, 3], [19, 29]]
    )
    np.testing.assert_array_equal(_read(features, "f1.jpg", "keypoint_scores"), [0, 2])
    # Descriptors are (D, N): columns follow the kept keypoints.
    descriptors = _read(features, "f1.jpg", "descriptors")
    assert descriptors.shape == (DESC_DIM, 2)
    np.testing.assert_array_equal(descriptors[0], [0, 2])
    np.testing.assert_array_equal(
        _read(features, "f1.jpg", "image_size"), [WIDTH, HEIGHT]
    )
    with h5py.File(features) as fd:
        keypoints = fd["f1.jpg/keypoints"]
        assert isinstance(keypoints, h5py.Dataset)
        assert keypoints.attrs["uncertainty"] == 1.5
    assert stats == {
        "masks_applied": 1,
        "masks_missing": 0,
        "keypoints_before": 4,
        "keypoints_after": 2,
    }


def test_missing_mask_keeps_all_keypoints(tmp_path):
    features = tmp_path / "features.h5"
    _write_features(features, "f1.jpg", [[5, 3], [25, 3]])
    (tmp_path / "masks").mkdir()

    stats = filter_features_by_mask(features, ["f1.jpg"], tmp_path / "masks")

    assert _read(features, "f1.jpg", "keypoints").shape == (2, 2)
    assert stats["masks_missing"] == 1
    assert stats["keypoints_after"] == 2


def test_nested_image_names(tmp_path):
    features = tmp_path / "features.h5"
    _write_features(features, "sub/f1.jpg", [[5, 3], [25, 3]])
    _write_left_half_mask(tmp_path / "masks", "sub/f1.jpg")

    filter_features_by_mask(features, ["sub/f1.jpg"], tmp_path / "masks")

    assert _read(features, "sub/f1.jpg", "keypoints").shape == (1, 2)


def test_mask_size_mismatch_raises(tmp_path):
    features = tmp_path / "features.h5"
    _write_features(features, "f1.jpg", [[5, 3]])
    (tmp_path / "masks").mkdir()
    cv2.imwrite(str(tmp_path / "masks" / "f1.jpg.png"), np.zeros((10, 10), np.uint8))

    with pytest.raises(ValueError, match="Mask for 'f1.jpg' is 10x10"):
        filter_features_by_mask(features, ["f1.jpg"], tmp_path / "masks")
