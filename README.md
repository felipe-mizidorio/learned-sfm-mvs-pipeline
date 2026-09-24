# learned-sfm-mvs-pipeline

SfM + MVS pipeline for neonatal cranial morphometry with selectable backends:

| Stage | Learned (default) | Classic |
|---|---|---|
| SfM | [hloc](https://github.com/cvg/Hierarchical-Localization) (SuperPoint + LightGlue) | COLMAP SIFT |
| MVS | [TransMVSNet](https://github.com/megvii-research/TransMVSNet) | COLMAP PatchMatch |

Successor of [sfm-mvs-pipeline](https://github.com/felipe-mizidorio/sfm-mvs-pipeline). Runs in Docker on CUDA 12.8+ (target: RTX 5090).

## Development

```bash
uv sync                  # core + dev (CPU)
uv sync --group learned  # + torch cu128, hloc, LightGlue
uv run pytest
```
