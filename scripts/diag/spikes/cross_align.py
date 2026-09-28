"""White mesh vs black mesh of the SAME mannequin (diag/markerless only).

The white and black captures are the same 3D-printed head, each in its own
arbitrary SfM frame and scale. This aligns a white mesh (``--source``) to the
black reference mesh (``--target``, B2 MVS: textured, good) with a similarity
ICP and reports how far the white surface lies from the black one, as a
fraction of the head radius. No ground truth STL exists, so the black MVS
mesh stands in for it; its own error (~0.7 % R vs the hull) bounds what the
number can resolve.

Initialisation: both meshes are rotated so their camera "up" (mean image-up
axis of the run's cameras) is +z and translated so the head apex (highest
point) is at the origin; then ICP with scaling is run from a grid of scales
and azimuths and the best (lowest RMSE among fits covering >= 60 % of the
source) is kept. Only the source's top ``--head-fraction`` of its height is
used (drops the neck cut / skirt).

Writes ``--out-dir/cross_<name>.json`` and ``aligned_<name>.ply`` (source mesh
in the target frame).

Usage::

    python scripts/diag/spikes/cross_align.py --source S.ply --source-run W0 \\
        --target T.ply --target-run B2 --out-dir OUT --name neus_white
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap


def _up(run_dir: Path) -> np.ndarray:
    """Mean image-up axis of the run's registered cameras (world frame)."""
    models = [
        pycolmap.Reconstruction(p)
        for p in sorted((run_dir / "sparse").iterdir())
        if p.is_dir()
    ]
    rec = max(models, key=lambda r: r.num_reg_images())
    ups = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        pose = im.cam_from_world
        pose = pose() if callable(pose) else pose
        ups.append(-pose.matrix()[1, :3])
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def _frame(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 rigid transform: up -> +z, apex -> origin."""
    z = up
    x = (
        np.cross([0.0, 1.0, 0.0], z)
        if abs(z[1]) < 0.9
        else np.cross([1.0, 0.0, 0.0], z)
    )
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(z, x), z])
    apex = points[np.argmax(points @ z)]
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = -rot @ apex
    return out


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    out = np.eye(4)
    out[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return out


def run(
    source: Path,
    source_run: Path,
    target: Path,
    target_run: Path,
    out_dir: Path,
    name: str,
    head_fraction: float = 0.8,
) -> dict:
    """Align ``source`` to ``target`` and measure the surface distance.

    Parameters
    ----------
    source, target : Path
        Meshes (white method mesh, black reference mesh).
    source_run, target_run : Path
        Pipeline runs they belong to (for the camera up axis).
    out_dir : Path
        Output directory.
    name : str
        Label for the output files.
    head_fraction : float, optional
        Share of the source's height (from the apex) that is compared.

    Returns
    -------
    dict
        Scale, fit and distance statistics.
    """
    src_mesh = o3d.io.read_triangle_mesh(str(source))
    tgt_mesh = o3d.io.read_triangle_mesh(str(target))
    src_pts = np.asarray(src_mesh.sample_points_uniformly(200_000).points)
    tgt_pts = np.asarray(tgt_mesh.sample_points_uniformly(200_000).points)
    fs, ft = _frame(src_pts, _up(source_run)), _frame(tgt_pts, _up(target_run))
    s = src_pts @ fs[:3, :3].T + fs[:3, 3]
    t = tgt_pts @ ft[:3, :3].T + ft[:3, 3]
    height = -s[:, 2].min()
    s_head = s[s[:, 2] > -head_fraction * height]
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_head))
    src = src.random_down_sample(min(1.0, 20_000 / len(s_head)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t))
    tgt = tgt.random_down_sample(min(1.0, 60_000 / len(t)))

    # Head size in each frame: horizontal radius at 30 % of the source height.
    def _radius(p: np.ndarray, depth: float) -> float:
        band = p[np.abs(p[:, 2] + depth) < 0.03 * depth / 0.3]
        return float(
            np.median(np.linalg.norm(band[:, :2] - band[:, :2].mean(0), axis=1))
        )

    r_src = _radius(s, 0.3 * height)
    scale0 = None
    best = None
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    t_height = -t[:, 2].min()
    for ratio in np.geomspace(0.3, 1.2, 10):
        # candidate: the source head height is `ratio` of the target's full height
        scale = ratio * t_height / height
        for az in range(0, 360, 30):
            init = _rz(az) @ np.diag([scale, scale, scale, 1.0])
            thr = 0.15 * scale * r_src
            for thr_k in (1.0, 0.4, 0.15):
                reg = o3d.pipelines.registration.registration_icp(
                    src,
                    tgt,
                    thr * thr_k,
                    init,
                    est,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
                )
                init = reg.transformation
            if reg.fitness >= 0.6 and (
                best is None or reg.inlier_rmse < best[1].inlier_rmse
            ):
                best, scale0 = (az, reg), scale
    if best is None:
        raise RuntimeError("no alignment covered 60 % of the source")
    az, reg = best
    T = reg.transformation
    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))

    scene = o3d.t.geometry.RaycastingScene()
    tgt_t = o3d.t.geometry.TriangleMesh.from_legacy(tgt_mesh)
    tgt_t.transform(o3d.core.Tensor(ft))
    scene.add_triangles(tgt_t)
    s_al = s_head @ T[:3, :3].T + T[:3, 3]
    d = scene.compute_distance(o3d.core.Tensor(s_al.astype(np.float32))).numpy()
    radius = r_src * scale
    rel = d / radius

    aligned = o3d.geometry.TriangleMesh(src_mesh)
    aligned.transform(np.linalg.inv(ft) @ T @ fs)
    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_dir / f"aligned_{name}.ply"), aligned)
    out = {
        "source": str(source),
        "target": str(target),
        "head_fraction": head_fraction,
        "init_scale": scale0,
        "init_azimuth_deg": az,
        "scale_source_to_target": scale,
        "icp_fitness": float(reg.fitness),
        "icp_inlier_rmse_over_radius": float(reg.inlier_rmse / radius),
        "head_radius_target_units": radius,
        "dist_median_pct_r": float(100 * np.median(rel)),
        "dist_p90_pct_r": float(100 * np.percentile(rel, 90)),
        "within_1pct_r": float(np.mean(rel < 0.01)),
        "within_2pct_r": float(np.mean(rel < 0.02)),
    }
    (out_dir / f"cross_{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--head-fraction", type=float, default=0.8)
    args = parser.parse_args()
    run(
        args.source,
        args.source_run,
        args.target,
        args.target_run,
        args.out_dir,
        args.name,
        args.head_fraction,
    )


if __name__ == "__main__":
    main()
