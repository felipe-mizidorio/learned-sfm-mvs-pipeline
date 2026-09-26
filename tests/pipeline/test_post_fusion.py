"""The post-fusion chain shared by sfm-mvs-run and sfm-mvs-resume-mvs."""

from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest
import yaml

from learned_sfm_mvs.pipeline.post_fusion import PostFusionOptions, run_post_fusion
from learned_sfm_mvs.scale.policy import UnscaledOutputError

SCALE = 100.0  # mm per SfM unit
HEAD_RADIUS = 0.6  # SfM units (60 mm)
BACKGROUND_DISTANCE = 5.0  # SfM units, far outside any head crop
MESH_CFG = yaml.safe_load(
    """
poisson_surface_reconstruction:
  depth: 6
  scale: 1.1
  linear_fit: false
  density_threshold: 0.01
  keep_largest_component: true
  taubin_smoothing: {iterations: 0}
point_cloud_filtering:
  nb_neighbors: 20
  std_ratio: 2.0
"""
)
ARUCO_CFG = {"marker_length_mm": 20.0, "dict_id": 2, "min_views": 2}


def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return radius * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
    )


def _write_dense(path) -> None:
    head = _fibonacci_sphere(4000, HEAD_RADIUS)
    rng = np.random.default_rng(0)
    # A far background blob, dense enough to survive SOR.
    background = BACKGROUND_DISTANCE + rng.normal(scale=0.05, size=(400, 3))
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.vstack([head, background]))
    )
    normals = np.vstack([head / HEAD_RADIUS, np.tile([0.0, 0, 1], (400, 1))])
    cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud)


def _markers() -> tuple[np.ndarray, dict]:
    """Three 20 mm squares on the head surface, 12 corners in SfM units."""
    side = ARUCO_CFG["marker_length_mm"] / SCALE
    corners_by_marker = {}
    for mid, center in enumerate(_fibonacci_sphere(3, HEAD_RADIUS)):
        # Square in the plane z = const through the centre (planar is enough).
        offsets = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * side / 2
        corners_by_marker[mid] = {k: center + o for k, o in enumerate(offsets)}
    points = np.array([p for c in corners_by_marker.values() for p in c.values()])
    return points, corners_by_marker


def _run(tmp_path, scale, mesh_cfg=MESH_CFG, mask_dir=None, **options):
    dense = tmp_path / "dense.ply"
    _write_dense(dense)
    points, corners = _markers()
    recovered = (scale, points, corners) if scale else (None, None, None)
    with patch(
        "learned_sfm_mvs.pipeline.post_fusion.recover_scale_details_safe",
        return_value=recovered,
    ):
        return run_post_fusion(
            dense,
            tmp_path,
            MagicMock(),
            tmp_path,
            ARUCO_CFG,
            mesh_cfg,
            None,
            PostFusionOptions(**options),
            mask_dir=mask_dir,
        )


def _points(path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def test_scaled_run_crops_meshes_and_scales_each_file_once(tmp_path):
    result = _run(tmp_path, SCALE)

    assert result.scale_factor == SCALE
    assert result.scale_status["status"] == "recovered_unvalidated"
    assert result.mesh_ply == tmp_path / "mesh.ply"
    # Metric head: ~60 mm radius. Scaled twice it would be ~6000 mm.
    mesh = o3d.io.read_triangle_mesh(str(result.mesh_ply))
    extent = np.ptp(np.asarray(mesh.vertices), axis=0)
    np.testing.assert_allclose(extent, 2 * HEAD_RADIUS * SCALE, rtol=0.1)
    for name in ["dense_filtered.ply", "dense_filtered_cropped.ply"]:
        radius = np.linalg.norm(_points(tmp_path / name), axis=1)
        assert np.median(radius) == pytest.approx(HEAD_RADIUS * SCALE, rel=0.05)
    # The crop removed the background.
    cropped = _points(tmp_path / "dense_filtered_cropped.ply")
    assert np.linalg.norm(cropped, axis=1).max() < 2 * HEAD_RADIUS * SCALE
    assert result.sor_stats["head_crop"]["radius_source"] == "aruco_auto"
    # sfm-mvs-run leaves dense.ply in SfM units.
    dense = _points(tmp_path / "dense.ply")
    assert np.median(np.linalg.norm(dense, axis=1)) == pytest.approx(
        HEAD_RADIUS, rel=0.05
    )


def test_scale_dense_ply_scales_it_in_place_once(tmp_path):
    _run(tmp_path, SCALE, scale_dense_ply=True)

    dense = _points(tmp_path / "dense.ply")
    head = dense[np.linalg.norm(dense, axis=1) < 2 * HEAD_RADIUS * SCALE]
    assert np.median(np.linalg.norm(head, axis=1)) == pytest.approx(
        HEAD_RADIUS * SCALE, rel=0.05
    )


def test_unscaled_stops_before_any_mesh(tmp_path):
    with pytest.raises(UnscaledOutputError):
        _run(tmp_path, None)

    assert not (tmp_path / "mesh.ply").exists()
    assert not (tmp_path / "dense_filtered_cropped.ply").exists()


def test_allow_unscaled_renames_every_artefact(tmp_path):
    result = _run(tmp_path, None, allow_unscaled=True, scale_dense_ply=True)

    assert result.scale_status["status"] == "unscaled"
    assert result.mesh_ply == tmp_path / "mesh.UNSCALED_sfm_units.ply"
    assert result.mesh_ply.exists()
    names = {p.name for p in tmp_path.glob("*.ply")}
    assert {
        "dense.UNSCALED_sfm_units.ply",
        "dense_filtered.UNSCALED_sfm_units.ply",
        "mesh.UNSCALED_sfm_units.ply",
    } <= names
    assert "mesh.ply" not in names
    assert "dense.ply" not in names


def test_allow_unscaled_keeps_dense_ply_for_run(tmp_path):
    _run(tmp_path, None, allow_unscaled=True)
    assert (tmp_path / "dense.ply").exists()


def test_masks_and_silhouette_config_reach_the_head_crop(tmp_path):
    silhouette = {"min_views": 5, "min_inside_fraction": 0.8}
    with patch(
        "learned_sfm_mvs.pipeline.post_fusion.run_head_crop",
        side_effect=lambda ply, *args, **kwargs: (ply, {}),
    ) as mock_crop:
        _run(
            tmp_path,
            None,
            mesh_cfg={**MESH_CFG, "silhouette_crop": silhouette},
            mask_dir=tmp_path / "masks",
            allow_unscaled=True,
        )

    kwargs = mock_crop.call_args.kwargs
    assert kwargs["mask_dir"] == tmp_path / "masks"
    assert kwargs["silhouette_cfg"] == silhouette
