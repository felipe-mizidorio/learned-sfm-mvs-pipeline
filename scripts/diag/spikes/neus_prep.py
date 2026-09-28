"""Phase 2 spike S1: pipeline run -> NeuS case, and NeuS mesh -> diag dir.

NeuS (Totoro97/NeuS, pure PyTorch, trained with the subject masks) learns a
signed distance field from silhouettes and shading, so it does not need
surface texture. This script only converts data; training runs NeuS's own
``exp_runner.py`` (see ``~/diag2/run_s1.sh`` on the VM).

``prep``: every registered view of a finished pipeline run is cropped to a
square around its subject mask (``--margin`` of the mask box per side, zero
padded) and resized to ``--size`` px; the SfM intrinsics follow the crop.
Lens distortion is ignored (SIMPLE_RADIAL k is about -4e-4 on these phones).
Writes ``image/NNN.png``, ``mask/NNN.png`` and ``cameras_sphere.npz``
(``world_mat_i`` = K [R|t], ``scale_mat_i`` = the sphere around the visual
hull, radius ``--sphere-scale`` * the 99.5th percentile hull radius).

``collect``: copies a world-space mesh (NeuS, or any other method's, for a
like-for-like mesh comparison) into a diag directory together with the run's
hull files and ``--points`` uniform surface samples written as
``dense_filtered_cropped.<tag>.ply``, so ``diag.py hullnoise`` / ``roughness``
run on it.

Usage::

    python scripts/diag/spikes/neus_prep.py prep --run-dir RUN --frames-manifest M.json \\
        --image-dir IMGS --case-dir CASE
    python scripts/diag/spikes/neus_prep.py collect --run-dir RUN --mesh MESH.ply --out-dir OUT
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap


def _best_reconstruction(sparse: Path) -> pycolmap.Reconstruction:
    """Largest model under ``sparse/``."""
    models = [
        pycolmap.Reconstruction(p) for p in sorted(sparse.iterdir()) if p.is_dir()
    ]
    return max(models, key=lambda r: r.num_reg_images())


def _cam_from_world(image: pycolmap.Image) -> np.ndarray:
    """3x4 world-to-camera matrix, across pycolmap versions."""
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return pose.matrix()


def prep(
    run_dir: Path,
    frames_manifest: Path,
    image_dir: Path,
    case_dir: Path,
    size: int = 512,
    margin: float = 0.3,
    sphere_scale: float = 1.15,
) -> dict:
    """Write a NeuS case from a pipeline run.

    Parameters
    ----------
    run_dir : Path
        Finished pipeline run (``sparse/``, ``hull_points.ply``).
    frames_manifest : Path
        Frames manifest with ``mask_dir``.
    image_dir : Path
        Frames directory.
    case_dir : Path
        NeuS case directory to create.
    size : int, optional
        Crop side, pixels.
    margin : float, optional
        Crop margin as a fraction of the mask box, per side.
    sphere_scale : float, optional
        Unit-sphere radius over the hull's 99.5th percentile radius.

    Returns
    -------
    dict
        Case metadata (also ``case_dir/prep.json``).
    """
    manifest = json.loads(frames_manifest.read_text())
    mask_dir = image_dir / manifest["mask_dir"]
    rec = _best_reconstruction(run_dir / "sparse")
    images = sorted(
        (im for im in rec.images.values() if im.has_pose), key=lambda im: im.name
    )
    hull = np.asarray(o3d.io.read_point_cloud(str(run_dir / "hull_points.ply")).points)
    centre = hull.mean(axis=0)
    radius = sphere_scale * np.percentile(np.linalg.norm(hull - centre, axis=1), 99.5)
    scale_mat = np.diag([radius, radius, radius, 1.0])
    scale_mat[:3, 3] = centre

    for sub in ("image", "mask"):
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
    cams: dict[str, np.ndarray] = {}
    names = []
    for im in images:
        bgr = cv2.imread(str(image_dir / im.name))
        mask = cv2.imread(str(mask_dir / f"{im.name}.png"), cv2.IMREAD_GRAYSCALE)
        if (
            bgr is None
            or mask is None
            or not (mask > 0).any()
            or (mask > 0).mean() > 0.99
        ):
            continue
        ys, xs = np.nonzero(mask > 0)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)
        x0, y0, s = cx - side / 2, cy - side / 2, size / side
        affine = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
        crop = cv2.warpAffine(bgr, affine, (size, size), flags=cv2.INTER_AREA)
        crop_mask = cv2.warpAffine(
            (mask > 0).astype(np.uint8) * 255,
            affine,
            (size, size),
            flags=cv2.INTER_NEAREST,
        )
        idx = len(names)
        cv2.imwrite(str(case_dir / "image" / f"{idx:03d}.png"), crop)
        cv2.imwrite(
            str(case_dir / "mask" / f"{idx:03d}.png"),
            cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR),
        )
        K = np.eye(4)
        K[:3, :3] = rec.cameras[im.camera_id].calibration_matrix()
        K[0, :3] *= s
        K[1, :3] *= s
        K[0, 2] -= x0 * s
        K[1, 2] -= y0 * s
        Rt = np.eye(4)
        Rt[:3] = _cam_from_world(im)
        cams[f"world_mat_{idx}"] = K @ Rt
        cams[f"scale_mat_{idx}"] = scale_mat
        names.append(im.name)
    np.savez(case_dir / "cameras_sphere.npz", **cams)
    meta = {
        "run_dir": str(run_dir),
        "views": len(names),
        "size": size,
        "margin": margin,
        "sphere_centre_sfm": centre.tolist(),
        "sphere_radius_sfm": float(radius),
        "frames": names,
    }
    (case_dir / "prep.json").write_text(json.dumps(meta, indent=2))
    print({k: v for k, v in meta.items() if k != "frames"})
    return meta


def collect(
    run_dir: Path, mesh_path: Path, out_dir: Path, points: int, tag: str = "neus"
) -> dict:
    """Mesh + surface samples + hull files into a diag directory.

    Parameters
    ----------
    run_dir : Path
        The pipeline run the case came from (hull files).
    mesh_path : Path
        NeuS ``validate_mesh`` output in world (SfM) coordinates.
    out_dir : Path
        Diag directory to create.
    points : int
        Uniform surface samples.
    tag : str, optional
        Method name used in the output file names.

    Returns
    -------
    dict
        Mesh size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    clusters, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(clusters) != int(np.argmax(counts)))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_dir / f"mesh.{tag}.ply"), mesh)
    o3d.io.write_point_cloud(
        str(out_dir / f"dense_filtered_cropped.{tag}.ply"),
        mesh.sample_points_uniformly(points),
    )
    for name in ("hull_mesh.ply", "hull_stats.json"):
        shutil.copy2(run_dir / name, out_dir / name)
    out = {
        "mesh": str(mesh_path),
        "triangles": len(mesh.triangles),
        "area_sfm2": float(mesh.get_surface_area()),
        "components_dropped": int(len(counts) - 1),
    }
    (out_dir / f"{tag}_mesh.json").write_text(json.dumps(out, indent=2))
    print(out)
    return out


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=512)
    p = sub.add_parser("collect")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--points", type=int, default=1_000_000)
    p.add_argument("--tag", default="neus")
    args = parser.parse_args()
    if args.cmd == "prep":
        prep(
            args.run_dir, args.frames_manifest, args.image_dir, args.case_dir, args.size
        )
    else:
        collect(args.run_dir, args.mesh, args.out_dir, args.points, args.tag)


if __name__ == "__main__":
    main()
