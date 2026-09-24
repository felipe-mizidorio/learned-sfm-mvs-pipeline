import pytest

from learned_sfm_mvs.sfm.images import list_images, normalize_camera_params


def test_list_images_natural_order_recursive_and_filtered(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["frame_10.jpg", "frame_9.JPG", "sub/frame_1.png", "notes.txt"]:
        (tmp_path / name).touch()

    assert list_images(tmp_path) == ["frame_9.JPG", "frame_10.jpg", "sub/frame_1.png"]


def test_list_images_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        list_images(tmp_path / "nope")


def test_list_images_no_images(tmp_path):
    (tmp_path / "notes.txt").touch()
    with pytest.raises(ValueError, match="No images found"):
        list_images(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        "3024 3024 2016 1512",
        "3024,3024,2016,1512",
        " 3024, 3024,  2016 ,1512 ",
    ],
)
def test_normalize_camera_params_accepts_spaces_and_commas(raw):
    assert normalize_camera_params(raw) == "3024,3024,2016,1512"


def test_normalize_camera_params_keeps_signs_and_exponents():
    assert normalize_camera_params("0.12 -0.05 1e-3") == "0.12,-0.05,1e-3"


@pytest.mark.parametrize("raw", ["", "   ", "3024 fx 2016"])
def test_normalize_camera_params_rejects_invalid(raw):
    with pytest.raises(ValueError):
        normalize_camera_params(raw)
