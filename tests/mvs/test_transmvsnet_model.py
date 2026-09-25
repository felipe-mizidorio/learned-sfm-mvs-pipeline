"""Vendored TransMVSNet: imports, loads the checkpoint, runs (needs torch)."""

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from learned_sfm_mvs._vendor.transmvsnet.models import TransMVSNet  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parents[2] / "models/transmvsnet/model_bld.ckpt"
H, W, VIEWS = 64, 96, 3
DEPTH_MIN, DEPTH_MAX = 1.0, 3.0


def _model():
    return TransMVSNet(
        refine=False,
        ndepths=[48, 32, 8],
        depth_interals_ratio=[4, 1, 0.5],
        share_cr=False,
        cr_base_chs=[8, 8, 8],
        grad_method="detach",
    ).eval()


def _inputs():
    """Views on a horizontal baseline; stage matrices as upstream builds them."""
    torch.manual_seed(0)
    imgs = torch.rand(1, VIEWS, 3, H, W)
    k_quarter = torch.tensor([[20.0, 0, W / 8], [0, 20.0, H / 8], [0, 0, 1]])
    proj = torch.zeros(1, VIEWS, 2, 4, 4)
    for v in range(VIEWS):
        extrinsic = torch.eye(4)
        extrinsic[0, 3] = -0.1 * v
        proj[0, v, 0] = extrinsic
        proj[0, v, 1, :3, :3] = k_quarter
        proj[0, v, 1, 3, 3] = 1
    stages = {"stage1": proj, "stage2": proj.clone(), "stage3": proj.clone()}
    stages["stage2"][:, :, 1, :2, :] *= 2
    stages["stage3"][:, :, 1, :2, :] *= 4
    depth_values = torch.linspace(DEPTH_MIN, DEPTH_MAX, 192).unsqueeze(0)
    return imgs, stages, depth_values


def test_forward_full_resolution_depth_within_range():
    model = _model()
    with torch.no_grad(), warnings.catch_warnings():
        # Patched meshgrid/None comparisons must not warn.
        warnings.simplefilter("error")
        out = model(*_inputs())

    assert out["depth"].shape == (1, H, W)
    assert out["photometric_confidence"].shape == (1, H, W)
    assert out["stage1"]["photometric_confidence"].shape == (1, H // 4, W // 4)
    # Stages 2-3 search +-ndepth/2 * ratio * interval around the previous
    # stage's depth without clamping, so depth may leave [min, max] by that.
    interval = (DEPTH_MAX - DEPTH_MIN) / 192
    overshoot = (32 / 2 * 1 + 8 / 2 * 0.5) * interval
    assert torch.all(out["depth"] >= DEPTH_MIN - overshoot)
    assert torch.all(out["depth"] <= DEPTH_MAX + overshoot)
    conf = out["photometric_confidence"]
    assert torch.all((conf >= 0) & (conf <= 1))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="model_bld.ckpt not downloaded")
def test_checkpoint_loads_strictly():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    _model().load_state_dict(checkpoint["model"], strict=True)
