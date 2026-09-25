"""Re-run stereo fusion (optionally with bbox), SOR, crop to head region,
then scale recovery and Poisson + LCC mesh.

Typical usage (no bbox — full fusion, SOR + automatic ArUco-derived head crop):
    uv run python scripts/resume_from_mvs.py \\
        --output-dir data/processed/<session> \\
        --image-dir path/to/filtered/frames \\
        --frames-manifest path/to/manifest.json

The head-crop radius is derived automatically from the triangulated ArUco
markers; --head-radius is a debug-only override (0 disables the crop).
Optionally add --bbox-min / --bbox-max to also clip at the fusion step.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from learned_sfm_mvs.cli.guards import guard_against_double_scale
from learned_sfm_mvs.mvs.fusion import fuse_depth_maps
from learned_sfm_mvs.mvs.mask_undistortion import undistort_masks_safe
from learned_sfm_mvs.pipeline.post_fusion import PostFusionOptions, run_post_fusion
from learned_sfm_mvs.postprocess.membrane_filter import (
    DEFAULT_MARKER_MARGIN_MM,
    DEFAULT_PALE_THRESHOLD,
)
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

_REPO_ROOT = Path(__file__).resolve().parents[3]

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
    parser.add_argument(
        "--aruco-config", default=_REPO_ROOT / "configs/aruco.yaml", type=Path
    )
    parser.add_argument(
        "--mesh-config", default=_REPO_ROOT / "configs/mesh.yaml", type=Path
    )
    parser.add_argument(
        "--colmap-config", default=_REPO_ROOT / "configs/colmap.yaml", type=Path
    )
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
        "--head-radius",
        type=float,
        default=None,
        help="DEBUG override for the spherical head-crop radius, in SfM units. "
        "Not needed in normal use: the radius is auto-derived from the "
        "triangulated ArUco markers and marker_length_mm. 0 disables the crop.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with args.aruco_config.open() as f:
        aruco_cfg = yaml.safe_load(f).get("aruco", {})
    with args.mesh_config.open() as f:
        mesh_cfg = yaml.safe_load(f)
    with args.colmap_config.open() as f:
        colmap_cfg = yaml.safe_load(f)

    output_dir: Path = args.output_dir
    mvs_dir = output_dir / "mvs"
    sparse_dir = output_dir / "sparse"
    dense_ply = output_dir / "dense.ply"

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
    # run_pipeline.py: <image-dir>/<manifest mask_dir>.
    mask_path: Path | None = None
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
        candidate = args.image_dir / manifest_data["mask_dir"]
        if not candidate.is_dir():
            logger.error("Mask directory '%s' does not exist. Aborting.", candidate)
            sys.exit(1)
        mask_path = candidate
        logger.info("Using mask directory: '%s'", mask_path)

    reconstruction, best_sparse = load_best_reconstruction(sparse_dir)
    logger.info(
        "Loaded sparse model from '%s': %d registered images",
        best_sparse,
        reconstruction.num_reg_images(),
    )

    # --- Step 1: Stereo fusion ---
    fusion_mask_dir: Path | None = None
    fusion_mask_stats: dict | None = None
    if args.skip_fusion:
        logger.info(
            "Skipping stereo fusion (--skip-fusion). Using existing '%s'.", dense_ply
        )
    else:
        if mask_path is not None:
            logger.info("=== Undistorting masks for stereo fusion ===")
            fusion_mask_dir, fusion_mask_stats = undistort_masks_safe(
                mask_path=mask_path,
                original_sparse_path=best_sparse,
                mvs_path=mvs_dir,
            )
        logger.info("=== Stereo fusion ===")
        fusion_start = time.perf_counter()
        fuse_depth_maps(
            mvs_path=mvs_dir,
            output_path=dense_ply,
            options=colmap_cfg["stereo_fusion"],
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
            mask_path=fusion_mask_dir,
        )
        logger.info("Stereo fusion took %.1f s", time.perf_counter() - fusion_start)

    # --- Steps 2-6: SOR, scale, head crop, membrane filter, Poisson, scale ---
    try:
        post = run_post_fusion(
            dense_ply,
            output_dir,
            reconstruction,
            args.image_dir,
            aruco_cfg,
            mesh_cfg,
            manifest_detections,
            PostFusionOptions(
                head_radius=args.head_radius,
                membrane_filter=args.membrane_filter,
                membrane_pale_threshold=args.membrane_pale_threshold,
                membrane_marker_margin_mm=args.membrane_marker_margin_mm,
                allow_unscaled=args.allow_unscaled,
                # resume-mvs has always scaled dense.ply in place (guarded above).
                scale_dense_ply=True,
            ),
        )
    except UnscaledOutputError as exc:
        logger.error("%s", exc)
        sys.exit(1)

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
        provenance=with_membrane_filter_provenance(
            with_fusion_mask_provenance(
                build_provenance(
                    args.frames_manifest,
                    {"aruco": aruco_cfg, "colmap": colmap_cfg, "mesh": mesh_cfg},
                ),
                enabled=fusion_mask_dir is not None,
                source_mask_dir=mask_path,
                workspace_mask_dir=fusion_mask_dir,
                stats=fusion_mask_stats,
            ),
            enabled=args.membrane_filter,
            stats=post.membrane_stats,
        ),
    )

    logger.info("Done. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
