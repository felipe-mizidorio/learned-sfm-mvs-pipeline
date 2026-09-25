from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from learned_sfm_mvs.mvs.transmvsnet.inference import (  # noqa: E402
    build_sample,
    combined_confidence,
    depth_hypotheses,
    run_inference,
    scale_intrinsics,
    target_size,
    verify_checkpoint,
)
from learned_sfm_mvs.mvs.transmvsnet.views import View  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parents[2] / "models/transmvsnet/model_bld.ckpt"


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        ((1920, 1080), (1920, 1080), (1920, 1056)),  # only rounded to /32
        ((4000, 3000), (1920, 1080), (1440, 1056)),  # height-bound
        ((4000, 1000), (1920, 1080), (1920, 480)),  # width-bound
        ((640, 480), (1920, 1080), (640, 480)),  # never upscaled
    ],
)
def test_target_size(size, limit, expected):
    assert target_size(*size, *limit) == expected


def test_target_size_rejects_tiny_images():
    with pytest.raises(ValueError, match="too small"):
        target_size(20, 20, 1920, 1080)


def test_scale_intrinsics_keeps_pixel_centres_aligned():
    K = np.array([[100.0, 0, 9.5], [0, 100.0, 4.5], [0, 0, 1]])
    scaled = scale_intrinsics(K, 2.0, 2.0)
    # Downscaled pixel (0, 0) covers upscaled pixels 0..1: its centre maps
    # to 0.5, not 0.
    assert scaled[0, 2] == pytest.approx((9.5 + 0.5) * 2 - 0.5)
    assert scaled[0, 0] == pytest.approx(200.0)
    assert K[0, 2] == 9.5  # input untouched


def test_depth_hypotheses_like_upstream():
    values = depth_hypotheses(1.0, 3.0, 192)
    assert values.shape == (192,)
    assert values.dtype == np.float32
    assert values[0] == pytest.approx(1.0)
    assert values[1] - values[0] == pytest.approx(2.0 / 192, rel=1e-5)  # float32


def _view(image_id, name="a.png", width=128, height=96, src_ids=()):
    return View(
        image_id=image_id,
        name=name,
        K=np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]]),
        extrinsic=np.eye(4),
        width=width,
        height=height,
        depth_min=1.0,
        depth_max=3.0,
        src_ids=tuple(src_ids),
    )


def test_build_sample_stage_matrices():
    views = [_view(1), _view(2)]
    images = [np.zeros((48, 64, 3), np.float32)] * 2

    sample = build_sample(views, images, (64, 48), 192)

    assert sample["imgs"].shape == (1, 2, 3, 48, 64)
    assert sample["depth_values"].shape == (1, 192)
    stage3 = sample["proj_matrices"]["stage3"][0, 0, 1, :3, :3].double().numpy()
    np.testing.assert_allclose(stage3, sample["K"], rtol=1e-6)
    stage1 = sample["proj_matrices"]["stage1"][0, 0, 1, :3, :3].double().numpy()
    np.testing.assert_allclose(stage1[:2], sample["K"][:2] / 4, rtol=1e-6)
    # 128x96 -> 64x48 is a 0.5 scale.
    assert sample["K"][0, 0] == pytest.approx(50.0)


def test_combined_confidence_is_product_of_stages():
    outputs = {
        "photometric_confidence": torch.full((1, 8, 8), 0.5),
        "stage1": {"photometric_confidence": torch.full((1, 2, 2), 0.4)},
        "stage2": {"photometric_confidence": torch.full((1, 4, 4), 0.5)},
    }
    torch.testing.assert_close(combined_confidence(outputs), torch.full((8, 8), 0.1))


def test_verify_checkpoint(tmp_path):
    path = tmp_path / "x.ckpt"
    with pytest.raises(FileNotFoundError, match="models/transmvsnet"):
        verify_checkpoint(path, None)
    path.write_bytes(b"abc")
    digest = verify_checkpoint(path, None)
    assert verify_checkpoint(path, digest) == digest
    with pytest.raises(ValueError, match="SHA-256"):
        verify_checkpoint(path, "0" * 64)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="model_bld.ckpt not downloaded")
def test_run_inference_writes_depth_maps(tmp_path):
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(0)
    views = []
    for i in range(1, 4):
        name = f"f{i}.png"
        cv2.imwrite(
            str(tmp_path / "images" / name),
            rng.integers(0, 255, (96, 128, 3), dtype=np.uint8),
        )
        views.append(_view(i, name, src_ids=[j for j in range(1, 4) if j != i]))
    views.append(_view(4, "f1.png", src_ids=[]))  # no sources: skipped
    cfg = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": None,
        "model": {
            "ndepths": [48, 32, 8],
            "depth_interval_ratios": [4, 1, 0.5],
            "num_depth": 192,
        },
        "inference": {"num_view": 3, "max_width": 64, "max_height": 64, "amp": False},
    }

    stats = run_inference(tmp_path, views, cfg, tmp_path / "depth", torch.device("cpu"))

    assert stats["depth_maps"] == 3
    assert stats["views_without_sources"] == 1
    assert stats["input_size"] == [64, 32]  # 128x96 fit into 64x64, /32
    with np.load(tmp_path / "depth" / "00000001.npz") as data:
        assert data["depth"].shape == (32, 64)
        assert data["confidence"].shape == (32, 64)
        assert list(data["src_ids"]) == [2, 3]
        assert str(data["name"]) == "f1.png"
