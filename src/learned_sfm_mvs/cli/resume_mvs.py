"""Re-fuse existing depth maps (optionally with bbox or masks), then SOR,
scale recovery, head crop and Poisson + LCC mesh.

The MVS backend defaults to the one recorded in the previous run's
pipeline_manifest.json, so its depth maps are the ones re-fused.

Typical usage (no bbox — full fusion, SOR + automatic ArUco-derived head crop):
    uv run sfm-mvs-resume-mvs \\
        --output-dir data/processed/<session> \\
        --image-dir path/to/filtered/frames \\
        --frames-manifest path/to/manifest.json

The head crop needs no parameters: the frames-manifest masks carve the dense
cloud (silhouette crop), or, without masks, a sphere is sized from the ArUco
markers.
Optionally add --bbox-min / --bbox-max to also clip at the fusion step.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from learned_sfm_mvs.cli.guards import guard_against_double_scale
from learned_sfm_mvs.cli.options import (
    CONFIGS,
    previous_mvs_backend,
    add_backend_arguments,
    load_transmvsnet_config,
    load_yaml,
    pipeline_defaults,
)
from learned_sfm_mvs.mvs.base import MvsInputs, fuse
from learned_sfm_mvs.pipeline.post_fusion import PostFusionOptions, run_post_fusion
from learned_sfm_mvs.postprocess.membrane_filter import (
    DEFAULT_MARKER_MARGIN_MM,
    DEFAULT_PALE_THRESHOLD,
)
from learned_sfm_mvs.pipeline.run_info import StageTimer, with_backend_provenance
from learned_sfm_mvs.pipeline.orchestration import (
    build_provenance,
    with_fusion_mask_provenance,
    with_membrane_filter_provenance,
    write_pipeline_manifest,
)
from learned_sfm_mvs.scale.policy import (
    UnscaledOutputError,
)
from learned_sfm_mvs.sfm.reconstruction import load_best_reconstruction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-fuse depth maps, SOR, crop to head sphere, scale recovery, Poisson + LCC."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--frames-manifest", default=None, type=Path)
    parser.add_argument("--aruco-config", default=CONFIGS / "aruco.yaml", type=Path)
    parser.add_argument("--mesh-config", default=CONFIGS / "mesh.yaml", type=Path)
    parser.add_argument("--colmap-config", default=CONFIGS / "colmap.yaml", type=Path)
    parser.add_argument(
        "--bbox-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional fusion-time bbox min (SfM units) — coarse background cut.",
    )
    parser.add_argument(
        "--bbox-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional fusion-time bbox max (SfM units) — coarse background cut.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip stereo fusion and reuse the existing dense.ply (already in SfM units).",
    )
    parser.add_argument(
        "--fusion-masks",
        action="store_true",
        help="Warp the frames-manifest masks into the undistorted MVS workspace and "
        "restrict stereo fusion to them. Requires --frames-manifest with a 'mask_dir'.",
    )
    parser.add_argument(
        "--membrane-filter",
        action="store_true",
        help="Remove pale 'membrane' contamination from the cropped cloud before "
        "Poisson, protecting the white ArUco marker faces. OFF by default. "
        "Scene-dependent: assumes a dark subject against pale contamination.",
    )
    parser.add_argument(
        "--allow-unscaled",
        action="store_true",
        help="Continue even if metric scale recovery fails, writing output in "
        "arbitrary SfM units. OFF by default: without this flag a failed scale "
        "recovery is a hard stop, because the alternative is a complete, "
        "plausible-looking mesh whose numbers are not millimetres. Artefacts "
        "written under this flag are renamed to *.UNSCALED_sfm_units.* and the "
        "manifest records scale.status 'unscaled'.",
    )
    parser.add_argument(
        "--membrane-pale-threshold",
        type=float,
        default=DEFAULT_PALE_THRESHOLD,
        help="Mean RGB (0-255) at or above which a point counts as pale.",
    )
    parser.add_argument(
        "--membrane-marker-margin-mm",
        type=float,
        default=DEFAULT_MARKER_MARGIN_MM,
        help="Protection margin added to each marker's own corner extent, in mm.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="auto: CUDA when available; cpu forces TransMVSNet fusion onto CPU.",
    )
    add_backend_arguments(parser, sfm=False)
    return parser.parse_args()


def main() -> None:
    """Re-fuse depth maps and re-run everything after fusion."""
    args = _parse_args()

    aruco_cfg = load_yaml(args.aruco_config).get("aruco", {})
    mesh_cfg = load_yaml(args.mesh_config)
    colmap_cfg = load_yaml(args.colmap_config)
    transmvsnet_cfg = load_transmvsnet_config(args.transmvsnet_config)

    output_dir: Path = args.output_dir
    sparse_dir = output_dir / "sparse"
    dense_ply = output_dir / "dense.ply"
    mvs_backend = (
        args.mvs_backend
        or previous_mvs_backend(output_dir)
        or pipeline_defaults(args.pipeline_config)["mvs_backend"]
    )

    if args.skip_fusion:
        guard_against_double_scale(
            output_dir,
            attempted="--skip-fusion",
            remedy="Re-run without --skip-fusion to regenerate dense.ply from mvs/",
        )

    manifest_detections = None
    manifest_data = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_detections = manifest_data.get("marker_detections")

    # Masks live next to the frames the manifest describes, exactly as in
    # run_pipeline.py: <image-dir>/<manifest mask_dir>. They always feed the
    # markerless silhouette crop; fusion masking stays opt-in.
    mask_path: Path | None = None
    if manifest_data is not None and manifest_data.get("mask_dir"):
        candidate = args.image_dir / manifest_data["mask_dir"]
        if candidate.is_dir():
            mask_path = candidate
            logger.info("Using mask directory: '%s'", mask_path)
        elif args.fusion_masks:
            logger.error("Mask directory '%s' does not exist. Aborting.", candidate)
            sys.exit(1)
        else:
            logger.warning(
                "Manifest mask_dir '%s' does not exist — ignoring masks.", candidate
            )
    if args.fusion_masks:
        if manifest_data is None:
            logger.error("--fusion-masks requires --frames-manifest. Aborting.")
            sys.exit(1)
        if not manifest_data.get("mask_dir"):
            logger.error(
                "--fusion-masks requested but frames manifest '%s' has no 'mask_dir'. Aborting.",
                args.frames_manifest,
            )
            sys.exit(1)

    reconstruction, best_sparse = load_best_reconstruction(sparse_dir)
    logger.info(
        "Loaded sparse model from '%s': %d registered images",
        best_sparse,
        reconstruction.num_reg_images(),
    )

    # --- Step 1: Fusion of the existing depth maps ---
    timer = StageTimer()
    fusion_mask_dir: Path | None = None
    fusion_mask_stats: dict | None = None
    # Always recorded, so the next resume re-fuses the same backend's maps.
    mvs_provenance: dict = {"name": mvs_backend, "fusion": None}
    if args.skip_fusion:
        logger.info(
            "Skipping stereo fusion (--skip-fusion). Using existing '%s'.", dense_ply
        )
    else:
        logger.info("=== Fusion (%s) ===", mvs_backend)
        with timer("fusion"):
            fused = fuse(
                mvs_backend,
                MvsInputs(
                    sparse_model_path=best_sparse,
                    image_dir=args.image_dir,
                    output_dir=output_dir,
                    mask_dir=mask_path,
                    fusion_masks=args.fusion_masks,
                    bbox_min=args.bbox_min,
                    bbox_max=args.bbox_max,
                    device=args.device,
                ),
                {"colmap": colmap_cfg, "transmvsnet": transmvsnet_cfg},
            )
        fusion_mask_dir = fused.fusion_mask_dir
        fusion_mask_stats = fused.stats["fusion_masks"]
        mvs_provenance = {"name": mvs_backend, "fusion": fused.stats["fusion"]}

    # --- Steps 2-6: SOR, scale, head crop, membrane filter, Poisson, scale ---
    try:
        with timer("post_fusion"):
            post = run_post_fusion(
                dense_ply,
                output_dir,
                reconstruction,
                args.image_dir,
                aruco_cfg,
                mesh_cfg,
                manifest_detections,
                PostFusionOptions(
                    membrane_filter=args.membrane_filter,
                    membrane_pale_threshold=args.membrane_pale_threshold,
                    membrane_marker_margin_mm=args.membrane_marker_margin_mm,
                    allow_unscaled=args.allow_unscaled,
                    # resume-mvs has always scaled dense.ply in place (guarded above).
                    scale_dense_ply=True,
                ),
                mask_dir=mask_path,
            )
    except UnscaledOutputError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    resolved_configs = {
        "pipeline": {"mvs_backend": mvs_backend},
        "aruco": aruco_cfg,
        "colmap": colmap_cfg,
        "mesh": mesh_cfg,
    }
    if mvs_backend == "transmvsnet" and not args.skip_fusion:
        resolved_configs["transmvsnet"] = transmvsnet_cfg
    write_pipeline_manifest(
        output_dir,
        "sfm-mvs-resume-mvs",
        post.sor_stats,
        post.lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        post.scale_factor,
        scale_sanity=post.scale_sanity,
        scale_self_consistency=post.scale_self_consistency,
        scale_status=post.scale_status,
        provenance=with_backend_provenance(
            with_membrane_filter_provenance(
                with_fusion_mask_provenance(
                    build_provenance(args.frames_manifest, resolved_configs),
                    enabled=fusion_mask_dir is not None,
                    source_mask_dir=mask_path,
                    workspace_mask_dir=fusion_mask_dir,
                    stats=fusion_mask_stats,
                ),
                enabled=args.membrane_filter,
                stats=post.membrane_stats,
            ),
            sfm=None,
            mvs=mvs_provenance,
            stage_timings=timer.stages,
        ),
    )

    logger.info("Done. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
