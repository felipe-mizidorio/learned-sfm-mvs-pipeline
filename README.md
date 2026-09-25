# learned-sfm-mvs-pipeline

SfM + MVS pipeline for neonatal cranial morphometry with selectable backends:

| Stage | Learned (default) | Classic |
|---|---|---|
| SfM | [hloc](https://github.com/cvg/Hierarchical-Localization) (ALIKED + LightGlue) | COLMAP SIFT |
| MVS | [TransMVSNet](https://github.com/megvii-research/TransMVSNet) | COLMAP PatchMatch |

Successor of [sfm-mvs-pipeline](https://github.com/felipe-mizidorio/sfm-mvs-pipeline). Runs in Docker on CUDA 12.8+ (target: RTX 5090).

## Model weights

hloc (ALIKED, LightGlue, NetVLAD) downloads its weights automatically into `cache/` on first use.

TransMVSNet's weights are only on Google Drive, so they are downloaded by hand:

1. Download `model_bld.ckpt` (BlendedMVS fine-tuned) from the [TransMVSNet models folder](https://drive.google.com/drive/folders/1ZJ9bx9qZENEoXv5i5izKCNszlaCNBMkJ).
2. Put it at `models/transmvsnet/model_bld.ckpt` (git-ignored, mounted into the container).

Its SHA-256 (`9423b42c…eb85`) is checked before loading; see `configs/transmvsnet.yaml`. The model code is vendored in `src/learned_sfm_mvs/_vendor/transmvsnet/` (MIT), with local patches listed in its `PATCHES.md`.

## Development

```bash
uv sync                  # core + dev (CPU)
uv sync --group learned  # + torch cu128, hloc, LightGlue
uv run pytest
```
