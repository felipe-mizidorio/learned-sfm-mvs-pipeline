"""End-to-end SfM + MVS + mesh + evaluation pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

from learned_sfm_mvs.cli.options import (
    CONFIGS,
    add_backend_arguments,
    load_transmvsnet_config,
    load_yaml,
    pipeline_defaults,
)
from learned_sfm_mvs.evaluation.metrics import evaluate
from learned_sfm_mvs.mvs.base import MvsInputs, run_mvs
from learned_sfm_mvs.pipeline.orchestration import (
    build_provenance,
    with_fusion_mask_provenance,
    with_membrane_filter_provenance,
    write_pipeline_manifest,
)
from learned_sfm_mvs.pipeline.post_fusion import PostFusionOptions, run_post_fusion
from learned_sfm_mvs.pipeline.run_info import StageTimer, with_backend_provenance
from learned_sfm_mvs.postprocess.membrane_filter import (
    DEFAULT_MARKER_MARGIN_MM,
    DEFAULT_PALE_THRESHOLD,
)
from learned_sfm_mvs.scale.policy import UnscaledOutputError
from learned_sfm_mvs.sfm.base import SfmInputs, run_sfm
from learned_sfm_mvs.sfm.feature_extraction import camera_prior_from_manifest
from learned_sfm_mvs.sfm.images import list_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SfM-MVS-mesh pipeline.")
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for all pipeline outputs.",
    )
    parser.add_argument(
        "--colmap-config",
        default=CONFIGS / "colmap.yaml",
        type=Path,
        help="Path to colmap.yaml config file.",
    )
    parser.add_argument(
        "--mesh-config",
        default=CONFIGS / "mesh.yaml",
        type=Path,
        help="Path to mesh.yaml config file.",
    )
    parser.add_argument(
        "--evaluation-config",
        default=CONFIGS / "evaluation.yaml",
        type=Path,
        help="Path to evaluation.yaml config file.",
    )
    parser.add_argument(
        "--aruco-config",
        default=CONFIGS / "aruco.yaml",
        type=Path,
        help="Path to aruco.yaml config file for metric scale recovery.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth .ply for evaluation (optional).",
    )
    parser.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Stop after sparse reconstruction (useful on CPU-only machines).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="auto: CUDA when available. cpu forces COLMAP stages and "
        "TransMVSNet onto the CPU (hloc still uses CUDA when present).",
    )
    parser.add_argument(
        "--per-image-cameras",
        action="store_true",
        help="One camera per image instead of one shared camera. Only for "
        "image sets from several cameras; same-device captures should share "
        "one (COLMAP skips images whose size differs from a shared camera).",
    )
    # --- Camera calibration (G2) ---
    parser.add_argument(
        "--camera-model",
        default=None,
        type=str,
        help=(
            "COLMAP camera model name (e.g. OPENCV, PINHOLE, SIMPLE_RADIAL). "
            "When provided, --camera-params must also be given and a single "
            "shared camera is used for all images."
        ),
    )
    parser.add_argument(
        "--camera-params",
        default=None,
        type=str,
        help=(
            "Camera intrinsics matching the chosen model, comma- or "
            "space-separated. "
            "For OPENCV: 'fx fy cx cy k1 k2 p1 p2'. "
            "For PINHOLE: 'fx fy cx cy'."
        ),
    )
    # --- Preprocessing manifest (G3) ---
    parser.add_argument(
        "--frames-manifest",
        default=None,
        type=Path,
        help=(
            "Path to a JSON manifest produced by the ArUco preprocessing pipeline. "
            'Expected keys: "frames" (list of image filenames to use) and optionally '
            '"marker_detections" ({frame: [{id, corners}]}) for scale recovery.'
        ),
    )
    # --- Head crop (debug override only) ---
    parser.add_argument(
        "--head-radius",
        type=float,
        default=None,
        help="DEBUG override for the spherical head-crop radius, in SfM units. "
        "Not needed in normal use: the radius is auto-derived from the "
        "triangulated ArUco markers and marker_length_mm. 0 disables the crop.",
    )
    # --- Bounding-box clipping (G4) ---
    parser.add_argument(
        "--bbox-min",
        default=None,
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Minimum corner of axis-aligned bounding box for stereo fusion clipping.",
    )
    parser.add_argument(
        "--bbox-max",
        default=None,
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Maximum corner of axis-aligned bounding box for stereo fusion clipping.",
    )
    parser.add_argument(
        "--fusion-masks",
        action="store_true",
        help="EXPERIMENTAL. Also restrict stereo fusion to the frames-manifest masks "
        "(warped into the undistorted MVS workspace). Off by default: with the "
        "current ArUco convex-hull masks this deletes genuine head surface away "
        "from the markers without reducing silhouette bleed — see "
        "docs/fusion_masks_report.md. Masks always apply to feature extraction "
        "regardless of this flag.",
    )
    parser.add_argument(
        "--membrane-filter",
        action="store_true",
        help="Remove pale 'membrane' contamination from the cropped cloud before "
        "Poisson, protecting the white ArUco marker faces. Off by default because "
        "it is SCENE-DEPENDENT: it assumes a dark subject against pale "
        "contamination and would delete the subject in a capture where the "
        "subject is pale. See docs/membrane_filter_report.md.",
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
    add_backend_arguments(parser, sfm=True)
    return parser.parse_args()


def main() -> None:
    """Run SfM, MVS, post-fusion, manifest and optional evaluation."""
    args = _parse_args()

    colmap_cfg = load_yaml(args.colmap_config)
    mesh_cfg = load_yaml(args.mesh_config)
    eval_cfg = load_yaml(args.evaluation_config)
    aruco_cfg = load_yaml(args.aruco_config).get("aruco", {})
    defaults = pipeline_defaults(args.pipeline_config)
    sfm_backend = args.sfm_backend or defaults["sfm_backend"]
    mvs_backend = args.mvs_backend or defaults["mvs_backend"]
    hloc_cfg = load_yaml(args.hloc_config)["hloc"]
    transmvsnet_cfg = load_transmvsnet_config(args.transmvsnet_config)
    logger.info("Backends: sfm=%s, mvs=%s", sfm_backend, mvs_backend)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = output_dir / "results" / "metrics.json"

    # --- Parse preprocessing manifest (G3) ---
    manifest_frames: list[str] | None = None
    manifest_detections: dict | None = None
    manifest_data: dict = {}
    mask_path: Path | None = None
    if args.frames_manifest is not None:
        manifest_data = json.loads(args.frames_manifest.read_text())
        manifest_frames = manifest_data.get("frames")
        manifest_detections = manifest_data.get("marker_detections")
        logger.info(
            "Frames manifest loaded: %d frames, %d pre-detected markers",
            len(manifest_frames or []),
            len(manifest_detections or {}),
        )
        if manifest_data.get("mask_dir"):
            candidate_mask_path = Path(args.image_dir) / manifest_data["mask_dir"]
            if candidate_mask_path.is_dir():
                mask_path = candidate_mask_path
                logger.info("Using mask directory: '%s'", mask_path)
            else:
                logger.warning(
                    "Manifest mask_dir '%s' does not exist — ignoring masks.",
                    candidate_mask_path,
                )

    # --- Camera intrinsics: explicit flags > EXIF-derived prior > shared
    # self-calibration. Same-device video always shares one camera model.
    camera_model, camera_params = args.camera_model, args.camera_params
    if camera_model or camera_params:
        intrinsics_source = "explicit"
    else:
        prior = camera_prior_from_manifest(manifest_data)
        if prior is not None:
            camera_model, camera_params = prior
            intrinsics_source = "exif_prior"
            logger.info(
                "Camera prior from manifest focal metadata: %s (%s)",
                camera_model,
                camera_params,
            )
        else:
            intrinsics_source = (
                "self_calibration_per_image"
                if args.per_image_cameras
                else "self_calibration_shared"
            )
    logger.info("Intrinsics source: %s", intrinsics_source)

    timer = StageTimer()

    # --- Step 1/4: Structure from Motion ---
    logger.info("=== Step 1/4: SfM (%s) ===", sfm_backend)
    sfm_inputs = SfmInputs(
        image_dir=args.image_dir,
        output_dir=output_dir,
        image_names=manifest_frames or list_images(args.image_dir),
        mask_dir=mask_path,
        camera_model=camera_model,
        camera_params=camera_params,
        shared_camera=not args.per_image_cameras,
        device=args.device,
    )
    try:
        with timer("sfm"):
            sfm = run_sfm(
                sfm_backend, sfm_inputs, {"colmap": colmap_cfg, "hloc": hloc_cfg}
            )
    except RuntimeError as exc:
        logger.error("SfM failed: %s", exc)
        sys.exit(1)

    if args.skip_mvs:
        logger.info("--skip-mvs set: stopping after sparse reconstruction.")
        return

    # --- Step 2/4: Multi-View Stereo (depth maps + fusion) ---
    # Feature-extraction masks never reach MVS, so fusion masks are re-warped
    # into the undistorted workspace. Opt-in: measured on video_test_20260716_115516,
    # ArUco hull masks at fusion cut 9.3% of mesh area inside the head sphere while
    # leaving silhouette contamination flat.
    logger.info("=== Step 2/4: MVS (%s) ===", mvs_backend)
    with timer("mvs"):
        mvs = run_mvs(
            mvs_backend,
            MvsInputs(
                sparse_model_path=sfm.model_path,
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

    # --- Step 3/4: SOR, scale, head crop, membrane filter, Poisson ---
    logger.info("=== Step 3/4: Post-fusion ===")
    try:
        with timer("post_fusion"):
            post = run_post_fusion(
                mvs.dense_ply,
                output_dir,
                sfm.reconstruction,
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
                ),
            )
    except UnscaledOutputError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # --- Step 4/4: Pipeline manifest ---
    resolved_configs = {
        "pipeline": {"sfm_backend": sfm_backend, "mvs_backend": mvs_backend},
        "aruco": aruco_cfg,
        # Both SfM backends map with colmap.yaml -> incremental_mapping.
        "colmap": colmap_cfg,
        "mesh": mesh_cfg,
    }
    if sfm_backend == "hloc":
        resolved_configs["hloc"] = hloc_cfg
    if mvs_backend == "transmvsnet":
        resolved_configs["transmvsnet"] = transmvsnet_cfg
    provenance = build_provenance(args.frames_manifest, resolved_configs)
    provenance["intrinsics_source"] = intrinsics_source
    with_fusion_mask_provenance(
        provenance,
        enabled=mvs.fusion_mask_dir is not None,
        source_mask_dir=mask_path,
        workspace_mask_dir=mvs.fusion_mask_dir,
        stats=mvs.stats["fusion_masks"],
    )
    with_membrane_filter_provenance(
        provenance, enabled=args.membrane_filter, stats=post.membrane_stats
    )
    with_backend_provenance(
        provenance,
        sfm={
            "name": sfm_backend,
            "registered_images": sfm.reconstruction.num_reg_images(),
            "input_images": len(sfm_inputs.image_names),
            "num_models": sfm.num_models,
            "model_path": str(sfm.model_path),
            **sfm.stats,
        },
        mvs={
            "name": mvs_backend,
            "depth": mvs.stats["depth"],
            "fusion": mvs.stats["fusion"],
        },
        stage_timings=timer.stages,
    )
    write_pipeline_manifest(
        output_dir,
        "sfm-mvs-run",
        post.sor_stats,
        post.lcc_stats,
        mesh_cfg["poisson_surface_reconstruction"],
        post.scale_factor,
        scale_sanity=post.scale_sanity,
        scale_self_consistency=post.scale_self_consistency,
        scale_status=post.scale_status,
        provenance=provenance,
    )

    # --- Optional: Evaluation ---
    if args.ground_truth is not None:
        logger.info("=== Evaluation: computing metrics ===")
        results = evaluate(
            predicted_ply=post.mesh_ply,
            ground_truth_ply=args.ground_truth,
            options=eval_cfg["evaluation"],
        )
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to '%s'", metrics_json)

    logger.info("Pipeline complete. Outputs in '%s'", output_dir)


if __name__ == "__main__":
    main()
