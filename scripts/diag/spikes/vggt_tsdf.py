"""Phase 2 spike S3: VGGT depth + masked TSDF fusion (diag/markerless only).

Learned feed-forward multi-view depth for textureless subjects. Runs in its own
venv (``~/diag2/envs/vggt``: torch cu128, facebookresearch/vggt,
open3d==0.19.0 -- 0.20.0's legacy TSDF returns empty volumes -- pycolmap), not in the pipeline image. VGGT-1B weights are research-only.

Steps:

1. Pick ``--frames`` registered views of a finished pipeline run, evenly by name.
2. Crop each frame to a square around its subject mask (``--margin`` of the
   mask box on every side, zero padded) and resize to 518 px, so the head
   fills the network input.
3. One VGGT forward pass over all crops: depth, depth confidence, cameras.
4. Similarity (Umeyama) from VGGT camera centres to the run's SfM camera
   centres gives the scale that puts VGGT depth in SfM units.
5. TSDF fusion (voxel = hull radius / ``--voxels-per-radius``) of the masked,
   confidence-filtered depth, twice: with the SfM poses and intrinsics
   (``sfm_poses``) and with VGGT's own cameras mapped into the SfM frame
   (``vggt_poses``). Mesh = marching cubes, largest component.

Writes ``--out-dir/vggt_meta.json`` and, per pose set, ``--out-dir/<poses>/``
with ``mesh.vggt.ply``, ``dense_filtered_cropped.vggt.ply`` (back-projected
points, for the diag metrics) and the run's ``hull_mesh.ply`` /
``hull_stats.json`` copied, so ``diag.py hullnoise`` / ``roughness`` work on
that directory.

Usage::

    python scripts/diag/spikes/vggt_tsdf.py --run-dir RUN --frames-manifest M.json \\
        --image-dir IMGS --out-dir OUT --weights model.pt
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

SIZE = 518


def _best_reconstruction(sparse: Path) -> pycolmap.Reconstruction:
    """Largest model under ``sparse/``."""
    models = [
        pycolmap.Reconstruction(p) for p in sorted(sparse.iterdir()) if p.is_dir()
    ]
    return max(models, key=lambda r: r.num_reg_images())


def _cam_from_world(image: pycolmap.Image) -> np.ndarray:
    """4x4 world-to-camera matrix, across pycolmap versions."""
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    out = np.eye(4)
    out[:3] = pose.matrix()
    return out


def _square_crop(
    image: np.ndarray, mask: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Square crop around the mask box, zero padded, resized to ``SIZE``.

    Returns the crop, its mask, and (x0, y0, scale) mapping full-frame pixel
    ``u`` to crop pixel ``(u - x0) * scale``.
    """
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)
    x0, y0 = cx - side / 2, cy - side / 2
    scale = SIZE / side
    affine = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    crop = cv2.warpAffine(image, affine, (SIZE, SIZE), flags=cv2.INTER_AREA)
    crop_mask = cv2.warpAffine(
        mask.astype(np.uint8), affine, (SIZE, SIZE), flags=cv2.INTER_NEAREST
    )
    return crop, crop_mask > 0, float(x0), float(y0), float(scale)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity ``dst ~ s R src + t`` (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    s = float(np.trace(np.diag(sig) @ d) / (a**2).sum(1).mean())
    return s, rot, mu_d - s * rot @ mu_s


def _fuse(
    views: list[dict], voxel: float, trunc: float, key: str
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """TSDF-fuse ``views`` with the pose set ``key``; also back-project points."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    points = []
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    pix = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols)], -1).reshape(-1, 3)
    for v in views:
        K, E = v[f"K_{key}"], v[f"E_{key}"]
        depth = np.where(v["keep"], v["depth"], 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(v["rgb"])),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=1e9,
            convert_rgb_to_intensity=False,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(
            SIZE, SIZE, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        volume.integrate(rgbd, intr, E)
        sel = v["keep"].reshape(-1)
        cam = (pix[sel] @ np.linalg.inv(K).T) * depth.reshape(-1)[sel, None]
        world_from_cam = np.linalg.inv(E)
        points.append(cam @ world_from_cam[:3, :3].T + world_from_cam[:3, 3])
    mesh = volume.extract_triangle_mesh()
    clusters, counts, _ = mesh.cluster_connected_triangles()
    if len(counts):
        mesh.remove_triangles_by_mask(np.asarray(clusters) != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    return mesh, np.concatenate(points)


def run(
    run_dir: Path,
    frames_manifest: Path,
    image_dir: Path,
    out_dir: Path,
    weights: Path,
    frames: int = 100,
    margin: float = 0.3,
    conf_keep: float = 0.7,
    voxels_per_radius: float = 200.0,
) -> dict:
    """Run the spike; see the module docstring.

    Parameters
    ----------
    run_dir : Path
        Finished pipeline run (``sparse/``, ``hull_mesh.ply``, ``hull_stats.json``).
    frames_manifest : Path
        Frames manifest with ``mask_dir``.
    image_dir : Path
        Frames directory.
    out_dir : Path
        Output directory (created).
    weights : Path
        VGGT-1B ``model.pt``.
    frames : int, optional
        Views fed to VGGT in one pass (about 21 GB VRAM at 100).
    margin : float, optional
        Crop margin as a fraction of the mask box, per side.
    conf_keep : float, optional
        Share of each view's masked pixels kept, most confident first.
    voxels_per_radius : float, optional
        TSDF resolution: hull radius / voxel size.

    Returns
    -------
    dict
        Metadata written to ``vggt_meta.json``.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(frames_manifest.read_text())
    mask_dir = image_dir / manifest["mask_dir"]
    rec = _best_reconstruction(run_dir / "sparse")
    images = sorted(
        (im for im in rec.images.values() if im.has_pose), key=lambda im: im.name
    )
    picks = np.linspace(0, len(images) - 1, min(frames, len(images))).astype(int)

    views, batch = [], []
    for i in picks:
        im = images[i]
        bgr = cv2.imread(str(image_dir / im.name))
        mask = cv2.imread(str(mask_dir / f"{im.name}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None or not (mask > 0).any():
            continue
        crop, crop_mask, x0, y0, scale = _square_crop(bgr, mask > 0, margin)
        K = rec.cameras[im.camera_id].calibration_matrix().copy()
        K[0] *= scale
        K[1] *= scale
        K[0, 2] -= x0 * scale
        K[1, 2] -= y0 * scale
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        views.append(
            {
                "name": im.name,
                "rgb": rgb,
                "mask": crop_mask,
                "K_ours": K,
                "E_ours": _cam_from_world(im),
            }
        )
        batch.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)

    device = "cuda"
    model = VGGT()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()
    x = torch.stack(batch).to(device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    extr, intr = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del model, pred, x
    torch.cuda.empty_cache()

    vggt_centres = np.array([-e[:, :3].T @ e[:, 3] for e in extr])
    our_centres = np.array([np.linalg.inv(v["E_ours"])[:3, 3] for v in views])
    s, rot, t = _umeyama(vggt_centres, our_centres)
    hull_stats = json.loads((run_dir / "hull_stats.json").read_text())
    radius = (
        3
        * hull_stats["occupied_voxels"]
        * hull_stats["voxel_size_sfm"] ** 3
        / (4 * np.pi)
    ) ** (1 / 3)
    residual = np.linalg.norm(s * vggt_centres @ rot.T + t - our_centres, axis=1)

    for v, d, c, e, k in zip(views, depth, conf, extr, intr):
        in_mask = v["mask"] & (d > 0)
        cut = np.quantile(c[in_mask], 1 - conf_keep) if in_mask.any() else np.inf
        v["keep"] = in_mask & (c >= cut)
        v["depth"] = d * s
        E = np.eye(4)
        E[:3, :3] = e[:, :3] @ rot.T
        E[:3, 3] = s * e[:, 3] - e[:, :3] @ rot.T @ t
        v["E_vggt"], v["K_vggt"] = E, k

    voxel = radius / voxels_per_radius
    meta: dict = {
        "run_dir": str(run_dir),
        "views": len(views),
        "crop_margin": margin,
        "conf_keep": conf_keep,
        "voxel_sfm": voxel,
        "hull_radius_sfm": radius,
        "vggt_to_sfm_scale": s,
        "camera_centre_residual_over_radius": {
            "median": float(np.median(residual) / radius),
            "p90": float(np.percentile(residual, 90) / radius),
        },
        "peak_vram_gb": peak_gb,
    }
    for key in ("ours", "vggt"):
        name = {"ours": "sfm_poses", "vggt": "vggt_poses"}[key]
        sub = out_dir / name
        sub.mkdir(exist_ok=True)
        mesh, pts = _fuse(views, voxel, 5 * voxel, key)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(sub / "mesh.vggt.ply"), mesh)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.random_down_sample(min(1.0, 2_000_000 / max(len(pts), 1)))
        o3d.io.write_point_cloud(str(sub / "dense_filtered_cropped.vggt.ply"), pcd)
        for name in ("hull_mesh.ply", "hull_stats.json"):
            shutil.copy2(run_dir / name, sub / name)
        meta[name] = {
            "mesh_triangles": len(mesh.triangles),
            "mesh_area_sfm2": float(mesh.get_surface_area()),
            "points": len(pts),
        }
    meta["seconds"] = round(time.time() - t0, 1)
    (out_dir / "vggt_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--conf-keep", type=float, default=0.7)
    parser.add_argument("--voxels-per-radius", type=float, default=200.0)
    args = parser.parse_args()
    run(
        args.run_dir,
        args.frames_manifest,
        args.image_dir,
        args.out_dir,
        args.weights,
        args.frames,
        args.margin,
        args.conf_keep,
        args.voxels_per_radius,
    )


if __name__ == "__main__":
    main()
