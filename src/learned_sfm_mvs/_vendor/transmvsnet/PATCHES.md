# Local patches to vendored TransMVSNet

Upstream: https://github.com/megvii-research/TransMVSNet @ `16100feb97309a846f73af4be9ea4e2e833baf5d` (MIT, see `LICENSE`).
Vendored: `models/` only (inference). Training, datasets, fusion scripts are not vendored;
the pipeline has its own data loading and fusion in `learned_sfm_mvs/mvs/transmvsnet/`.

Every change from upstream, so the copy can be re-synced:

| File | Change | Why |
|---|---|---|
| `models/__init__.py` | `from models.TransMVSNet` → `from .TransMVSNet` | Upstream assumes its repo root on `sys.path`. |
| `models/TransMVSNet.py` | `print(...)` in `__init__` → `logger.info(...)` | Library code should not print. |
| `models/TransMVSNet.py` | `view_weights == None` → `view_weights is None` (4×) | Comparing a tensor with `==` is not an identity check. |
| `models/TransMVSNet.py` | `depth_values.shapep` → `depth_values.shape` | Typo in an assert message (only hit on failure). |
| `models/module.py` | `torch.meshgrid(..., indexing="ij")` in `homo_warping` | torch 2.x warns without it; `"ij"` is the old default, so behavior is unchanged. |

Numerics are unchanged: the checkpoint loads with `strict=True` and the tests in
`tests/mvs/test_transmvsnet_model.py` run a forward pass.
