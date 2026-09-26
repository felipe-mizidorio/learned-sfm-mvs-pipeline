"""Post-fusion chain shared by sfm-mvs-run and sfm-mvs-resume-mvs.

dense.ply (SfM units) → SOR → scale recovery and checks → scale policy gate
→ head crop → optional membrane filter → Poisson + LCC → metric scale applied
once per written file, or ``*.UNSCALED_sfm_units.*`` renaming.

Order matters: scale recovery runs before the crop because the automatic crop
radius is derived in millimetres; the policy gate runs before the crop and the
mesh so a failed recovery cannot produce a finished, metric-looking mesh; the
scale is applied only after meshing so no file is scaled twice.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap

from learned_sfm_mvs.pipeline.orchestration import (
    run_head_crop,
    run_membrane_filter,
    run_poisson_lcc,
    run_sor,
)
from learned_sfm_mvs.postprocess.membrane_filter import (
    DEFAULT_MARKER_MARGIN_MM,
    DEFAULT_PALE_THRESHOLD,
)
from learned_sfm_mvs.scale.aruco_scale import (
    apply_scale_to_mesh,
    apply_scale_to_ply,
    recover_scale_details_safe,
)
from learned_sfm_mvs.scale.layout_check import check_marker_layout
from learned_sfm_mvs.scale.policy import (
    enforce_scale_policy,
    resolve_scale_status,
    unscaled_artifact_path,
)
from learned_sfm_mvs.scale.self_consistency import check_scale_self_consistency

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostFusionOptions:
    """Run options of the post-fusion chain.

    Parameters
    ----------
    membrane_filter : bool, optional
        Remove pale membrane points before Poisson (scene-dependent, opt-in).
    membrane_pale_threshold : float, optional
        Mean RGB at or above which a point is pale.
    membrane_marker_margin_mm : float, optional
        Marker protection margin in millimetres.
    allow_unscaled : bool, optional
        Continue without a recovered scale, renaming outputs as unscaled.
    scale_dense_ply : bool, optional
        Also scale (or rename) dense.ply itself. resume-mvs does this, which
        is why the resume entry points guard against scaling it twice.
    """

    membrane_filter: bool = False
    membrane_pale_threshold: float = DEFAULT_PALE_THRESHOLD
    membrane_marker_margin_mm: float = DEFAULT_MARKER_MARGIN_MM
    allow_unscaled: bool = False
    scale_dense_ply: bool = False


@dataclass(frozen=True)
class PostFusionResult:
    """Outputs and manifest fields of the post-fusion chain.

    Parameters
    ----------
    mesh_ply : Path
        Final mesh (renamed when unscaled).
    scale_factor : float or None
        Recovered mm/SfM-unit factor.
    scale_status : dict
        ``resolve_scale_status`` result.
    scale_sanity : dict or None
        ``check_marker_layout`` result.
    scale_self_consistency : dict or None
        ``check_scale_self_consistency`` result.
    sor_stats : dict
        SOR and head-crop counts.
    lcc_stats : dict
        Poisson/LCC counts.
    membrane_stats : dict or None, optional
        Membrane-filter stats when it ran.
    """

    mesh_ply: Path
    scale_factor: float | None
    scale_status: dict
    scale_sanity: dict | None
    scale_self_consistency: dict | None
    sor_stats: dict
    lcc_stats: dict
    membrane_stats: dict | None = field(default=None)


def run_post_fusion(
    dense_ply: Path,
    output_dir: Path,
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    aruco_cfg: dict,
    mesh_cfg: dict,
    detections: dict | None,
    options: PostFusionOptions,
    mask_dir: Path | None = None,
) -> PostFusionResult:
    """Turn a fused dense cloud into a metric, cropped, meshed result.

    Parameters
    ----------
    dense_ply : Path
        Fused cloud in SfM units.
    output_dir : Path
        Run output directory.
    reconstruction : pycolmap.Reconstruction
        Sparse model the cloud was fused from (scale recovery, crop centre).
    image_dir : Path
        Root image directory (ArUco detection when ``detections`` is None).
    aruco_cfg : dict
        ``aruco`` section of ``configs/aruco.yaml``.
    mesh_cfg : dict
        Parsed ``configs/mesh.yaml``.
    detections : dict or None
        Pre-computed marker detections from the frames manifest.
    options : PostFusionOptions
        Run options.
    mask_dir : Path or None, optional
        Original-frame subject masks from the frames manifest; without
        markers they drive the silhouette head crop (``configs/mesh.yaml ->
        silhouette_crop``).

    Returns
    -------
    PostFusionResult
        Final mesh path and manifest fields.

    Raises
    ------
    UnscaledOutputError
        If no scale was recovered and ``options.allow_unscaled`` is False;
        raised before the crop, so no mesh is written.
    """
    mesh_ply = output_dir / "mesh.ply"

    logger.info("=== Point cloud filtering (SOR) ===")
    dense_filtered_ply, sor_stats = run_sor(
        dense_ply, output_dir, mesh_cfg["point_cloud_filtering"]
    )

    marker_length_mm = aruco_cfg.get("marker_length_mm")
    scale_factor, marker_points, corners_by_marker = recover_scale_details_safe(
        reconstruction=reconstruction,
        image_dir=image_dir,
        marker_length_mm=float(marker_length_mm) if marker_length_mm else None,
        aruco_dict_id=int(aruco_cfg.get("dict_id", 0)),
        detections=detections,
        min_views=int(aruco_cfg.get("min_views", 2)),
    )
    scale_sanity = check_marker_layout(
        corners_by_marker or {}, scale_factor, aruco_cfg.get("layout_check")
    )
    scale_self_consistency = check_scale_self_consistency(
        corners_by_marker or {}, float(marker_length_mm) if marker_length_mm else None
    )
    scale_status = resolve_scale_status(scale_factor, scale_sanity)
    enforce_scale_policy(scale_status, allow_unscaled=options.allow_unscaled)

    cropped_ply, crop_stats = run_head_crop(
        dense_filtered_ply,
        output_dir,
        reconstruction,
        scale_factor=scale_factor,
        marker_points=marker_points,
        mask_dir=mask_dir,
        silhouette_cfg=mesh_cfg.get("silhouette_crop"),
    )
    sor_stats.update(crop_stats)

    input_for_poisson = cropped_ply
    membrane_stats: dict | None = None
    if options.membrane_filter:
        input_for_poisson, membrane_stats = run_membrane_filter(
            cropped_ply,
            output_dir,
            marker_corners=corners_by_marker,
            pale_threshold=options.membrane_pale_threshold,
            marker_margin_mm=options.membrane_marker_margin_mm,
            scale_factor=scale_factor,
        )

    logger.info("=== Surface (Poisson) reconstruction + LCC ===")
    _, lcc_stats = run_poisson_lcc(
        input_for_poisson,
        mesh_ply,
        output_dir,
        mesh_cfg["poisson_surface_reconstruction"],
    )

    # Every written cloud exactly once. dict.fromkeys de-duplicates while
    # preserving order, so a run where the crop or the membrane filter was
    # skipped does not touch the same file twice.
    clouds = [dense_filtered_ply, cropped_ply, input_for_poisson]
    if options.scale_dense_ply:
        clouds.insert(0, dense_ply)
    clouds = list(dict.fromkeys(clouds))
    if scale_factor is not None:
        for ply in clouds:
            apply_scale_to_ply(ply, scale_factor)
        apply_scale_to_mesh(mesh_ply, scale_factor)
        logger.info("Applied scale %.6f mm/unit to outputs.", scale_factor)
    else:
        # Reached only under allow_unscaled: rename every artefact so a stray
        # file cannot later be mistaken for metric output.
        for ply in clouds:
            ply.rename(unscaled_artifact_path(ply))
        mesh_ply = mesh_ply.rename(unscaled_artifact_path(mesh_ply))

    return PostFusionResult(
        mesh_ply=mesh_ply,
        scale_factor=scale_factor,
        scale_status=scale_status,
        scale_sanity=scale_sanity,
        scale_self_consistency=scale_self_consistency,
        sor_stats=sor_stats,
        lcc_stats=lcc_stats,
        membrane_stats=membrane_stats,
    )
