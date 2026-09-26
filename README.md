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

## Running

In the GPU container (set `HOST_UID`/`HOST_GID` in `.env` first, see `.env.example`):

```bash
docker compose build pipeline
docker compose run --rm pipeline python scripts/gpu_smoke.py
docker compose run --rm pipeline sfm-mvs-run --image-dir data/raw/session_01/frames --output-dir data/processed/session_01 --frames-manifest data/raw/session_01/manifest.json
```

The backends come from `configs/pipeline.yaml` (hloc + TransMVSNet) unless overridden:

| Flag | Default | Description |
|---|---|---|
| `--sfm-backend` | `hloc` | `hloc` or `colmap_sift` |
| `--mvs-backend` | `transmvsnet` | `transmvsnet` or `patchmatch` |
| `--hloc-config` / `--transmvsnet-config` | `configs/*.yaml` | Learned-backend settings |
| `--per-image-cameras` | off | One camera per image, for image sets from several cameras. Same-device captures should share one: COLMAP skips images whose size differs from a shared camera, and the run stops if it does. |
| `--device` | `auto` | `cpu` forces COLMAP stages and TransMVSNet onto the CPU |
| `--no-feature-masks` | off | Run SfM on whole frames, ignoring the manifest masks. For low-texture subjects (plain white head) whose masked region has too few features to register the cameras. The masks still reach MVS: TransMVSNet always takes each view's depth search range from the subject's sparse points, and `--fusion-masks` keeps the dense cloud on the subject. |

All other flags (camera calibration, frames manifest, head crop, bbox, fusion masks, membrane filter, `--allow-unscaled`, evaluation) behave as in [sfm-mvs-pipeline](https://github.com/felipe-mizidorio/sfm-mvs-pipeline); see `sfm-mvs-run --help`. `--camera-params` accepts comma- or space-separated values.

Head crop, no parameters: the frames-manifest masks carve the dense cloud, with or without ArUco markers. A point is kept when it lies inside the subject mask in most masked views that see it (`configs/mesh.yaml` → `silhouette_crop`). Only without usable masks is the cloud cropped to a sphere sized from the markers. (`--head-radius` from sfm-mvs-pipeline is gone.)

Resuming an existing output directory:

- `sfm-mvs-resume-mvs` re-fuses the existing depth maps and re-runs everything after fusion. It uses the MVS backend recorded in `pipeline_manifest.json` (override with `--mvs-backend`), so TransMVSNet depth maps can be re-fused with other thresholds in `transmvsnet.yaml` without re-running the network.
- `sfm-mvs-resume-dense` resumes from `dense.ply`.

Both refuse to run when `dense.ply` was already scaled to millimetres by a previous `sfm-mvs-resume-mvs`, which would scale it twice.

`pipeline_manifest.json` records, besides the scale status and per-stage counts: the backends and their stats (image pairs, masks, TransMVSNet checkpoint SHA-256, network input size, depth-range source per view, fused points), per-stage wall time and torch peak VRAM, and torch/CUDA/GPU/hloc versions.

## Development

```bash
uv sync                  # core + dev (CPU)
uv sync --group learned  # + torch cu128, hloc, LightGlue
uv run pytest
```
