"""TransMVSNet fusion on a synthetic plane seen by cameras on a baseline."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learned_sfm_mvs.mvs.transmvsnet.fusion import (  # noqa: E402
    camera_facing_normals,
    check_geometric_consistency,
    fuse_depth_maps,
)

H, W = 48, 64
PLANE_Z = 5.0
K = np.array([[50.0, 0, (W - 1) / 2], [0, 50.0, (H - 1) / 2], [0, 0, 1]])
CPU = torch.device("cpu")
CFG = {
    "min_confidence": 0.5,
    "min_consistent_views": 2,
    "max_reproj_error_px": 1.0,
    "max_relative_depth_error": 0.01,
    "dedupe": False,
}


def _extrinsic(x: float) -> np.ndarray:
    """Camera at (x, 0, 0) looking down +z (world-to-camera)."""
    extrinsic = np.eye(4)
    extrinsic[0, 3] = -x
    return extrinsic


def _workspace(tmp_path, depths, confidences=None, xs=(0.0, 0.2, 0.4, 0.6)):
    """Depth maps + solid-colour images; every view is a source of every other."""
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    ids = list(range(1, len(xs) + 1))
    for i, (image_id, x, depth) in enumerate(zip(ids, xs, depths)):
        name = f"f{image_id}.png"
        cv2.imwrite(
            str(tmp_path / "images" / name), np.full((H, W, 3), 40 * i, np.uint8)
        )
        conf = np.ones((H, W)) if confidences is None else confidences[i]
        np.savez(
            depth_dir / f"{image_id:08d}.npz",
            depth=np.full((H, W), depth, np.float32),
            confidence=conf.astype(np.float32),
            K=K,
            extrinsic=_extrinsic(x),
            src_ids=np.array([s for s in ids if s != image_id]),
            name=np.array(name),
        )
    return depth_dir


def _fuse(tmp_path, depth_dir, **kwargs):
    import open3d as o3d

    out = tmp_path / "dense.ply"
    cfg = {**CFG, **kwargs.pop("cfg", {})}
    stats = fuse_depth_maps(depth_dir, tmp_path, out, cfg, CPU, **kwargs)
    cloud = o3d.io.read_point_cloud(str(out))
    return stats, np.asarray(cloud.points), np.asarray(cloud.normals)


def test_identical_views_are_fully_consistent():
    depth = torch.full((H, W), PLANE_Z, dtype=torch.float64)
    Kt = torch.from_numpy(K)
    E = torch.from_numpy(_extrinsic(0.0))
    mask, depth_back, _ = check_geometric_consistency(
        depth, Kt, E, depth, Kt, E, 1.0, 0.01
    )
    assert mask.all()
    torch.testing.assert_close(depth_back, depth)


def test_fused_points_lie_on_plane_with_camera_facing_normals(tmp_path):
    depth_dir = _workspace(tmp_path, [PLANE_Z] * 4)

    stats, xyz, normals = _fuse(tmp_path, depth_dir)

    assert stats["points"] == len(xyz) > 0
    np.testing.assert_allclose(xyz[:, 2], PLANE_Z, atol=1e-4)
    # Cameras sit at z = 0 in front of the plane.
    np.testing.assert_allclose(
        normals, np.tile([0, 0, -1], (len(normals), 1)), atol=1e-4
    )


def test_inconsistent_view_is_rejected(tmp_path):
    # View 4 believes the plane is at 7: no other view agrees with it, and it
    # agrees with nobody, so no point at z = 7 may appear.
    depth_dir = _workspace(tmp_path, [PLANE_Z, PLANE_Z, PLANE_Z, 7.0])

    _, xyz, _ = _fuse(tmp_path, depth_dir)

    assert len(xyz) > 0
    np.testing.assert_allclose(xyz[:, 2], PLANE_Z, atol=1e-4)


def test_dedupe_shrinks_cloud_without_losing_coverage(tmp_path):
    depth_dir = _workspace(tmp_path, [PLANE_Z] * 4)

    _, full, _ = _fuse(tmp_path, depth_dir)
    _, deduped, _ = _fuse(tmp_path, depth_dir, cfg={"dedupe": True})

    assert len(deduped) < len(full) / 2
    # Same x extent of the plane is still covered.
    assert deduped[:, 0].min() == pytest.approx(full[:, 0].min(), abs=0.2)
    assert deduped[:, 0].max() == pytest.approx(full[:, 0].max(), abs=0.2)


def test_low_confidence_view_contributes_nothing(tmp_path):
    confidences = [np.ones((H, W))] * 3 + [np.zeros((H, W))]
    depth_dir = _workspace(tmp_path, [PLANE_Z] * 4, confidences)

    _, with_all, _ = _fuse(tmp_path / "b", _workspace(tmp_path / "b", [PLANE_Z] * 4))
    _, without_4, _ = _fuse(tmp_path, depth_dir)

    assert 0 < len(without_4) < len(with_all)


def test_fusion_mask_and_bbox(tmp_path):
    depth_dir = _workspace(tmp_path, [PLANE_Z] * 4)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for i in range(1, 5):
        mask = np.zeros((H, W), np.uint8)
        mask[:, : W // 2] = 255
        cv2.imwrite(str(mask_dir / f"f{i}.png.png"), mask)

    _, full, _ = _fuse(tmp_path, depth_dir)
    _, masked, _ = _fuse(tmp_path, depth_dir, mask_dir=mask_dir)
    stats, clipped, _ = _fuse(
        tmp_path, depth_dir, bbox_min=[-10, -10, 0], bbox_max=[0.0, 10, 10]
    )

    assert 0 < len(masked) < len(full)
    assert np.all(clipped[:, 0] <= 0.0)
    assert stats["points_outside_bbox"] == len(full) - len(clipped) > 0


def test_nothing_kept_raises(tmp_path):
    depth_dir = _workspace(tmp_path, [PLANE_Z] * 4)
    with pytest.raises(RuntimeError, match="kept no points"):
        _fuse(tmp_path, depth_dir, cfg={"min_confidence": 2.0})


def test_camera_facing_normals_flip_toward_origin():
    # Plane z = 5 sampled on a grid, seen from the origin.
    y, x = torch.meshgrid(
        torch.arange(4.0, dtype=torch.float64),
        torch.arange(5.0, dtype=torch.float64),
        indexing="ij",
    )
    points = torch.stack([x, y, torch.full_like(x, 5.0)]).reshape(3, -1)
    normals = camera_facing_normals(points, 4, 5)
    torch.testing.assert_close(
        normals, torch.tensor([0.0, 0, -1], dtype=torch.float64)[:, None].expand(3, 20)
    )
