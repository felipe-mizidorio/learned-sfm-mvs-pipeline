"""Markerless diagnosis helpers (diag/markerless branch only).

Subcommands, all read-only on the run they inspect:

- ``sparse``: split the sparse model into subject / background point PLYs.
- ``masks``: mask-over-frame contact sheet and per-frame mask area series.
- ``summary``: one ``diag_summary.json`` with the numbers of a run.

Usage::

    python scripts/diag/diag.py sparse  --run-dir OUT --frames-manifest M.json --image-dir IMGS
    python scripts/diag/diag.py masks   --frames-manifest M.json --image-dir IMGS --out-dir OUT
    python scripts/diag/diag.py summary --run-dir OUT --frames-manifest M.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from learned_sfm_mvs.mvs.transmvsnet.views import subject_point_ids
from learned_sfm_mvs.sfm.reconstruction import load_best_reconstruction


def _mask_dir(frames_manifest: Path, image_dir: Path) -> Path:
    manifest = json.loads(frames_manifest.read_text())
    return image_dir / manifest["mask_dir"]


def export_sparse(run_dir: Path, frames_manifest: Path, image_dir: Path) -> dict:
    """Write ``sparse_subject.ply`` and ``sparse_background.ply``.

    Parameters
    ----------
    run_dir : Path
        Pipeline output directory (holds ``sparse/``).
    frames_manifest : Path
        Frames manifest with ``mask_dir``.
    image_dir : Path
        Frames directory the manifest's ``mask_dir`` is relative to.

    Returns
    -------
    dict
        Subject and background sparse point counts.
    """
    reconstruction, _ = load_best_reconstruction(run_dir / "sparse")
    images = [im for im in reconstruction.images.values() if im.has_pose]
    subject = subject_point_ids(images, _mask_dir(frames_manifest, image_dir))
    counts = {}
    for label, keep in (("subject", True), ("background", False)):
        ids = [p for p in reconstruction.points3D if (p in subject) == keep]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.array([reconstruction.points3D[p].xyz for p in ids]).reshape(-1, 3)
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([reconstruction.points3D[p].color for p in ids]).reshape(-1, 3)
            / 255.0
        )
        o3d.io.write_point_cloud(str(run_dir / f"sparse_{label}.ply"), pcd)
        counts[f"sparse_{label}_points"] = len(ids)
    print(counts)
    return counts


def mask_overlays(
    frames_manifest: Path, image_dir: Path, out_dir: Path, tiles: int = 24
) -> dict:
    """Contact sheet of masks over frames plus the mask area series.

    Parameters
    ----------
    frames_manifest : Path
        Frames manifest (``frames``, ``mask_dir``).
    image_dir : Path
        Frames directory.
    out_dir : Path
        Where ``masks_sheet.png``, ``mask_area.csv`` and ``mask_stats.json`` go.
    tiles : int, optional
        Frames shown in the sheet, evenly spaced.

    Returns
    -------
    dict
        Mask area statistics (fraction of frame).
    """
    manifest = json.loads(frames_manifest.read_text())
    frames = manifest["frames"]
    mask_dir = image_dir / manifest["mask_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    areas = []
    for name in frames:
        mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        areas.append(float((mask > 0).mean()) if mask is not None else float("nan"))
    area = np.array(areas)
    with (out_dir / "mask_area.csv").open("w") as f:
        f.write("frame,area_fraction\n")
        f.writelines(f"{n},{a:.5f}\n" for n, a in zip(frames, areas))

    jumps = np.abs(np.diff(area)) / np.maximum(area[:-1], 1e-6)
    stats = {
        "frames": len(frames),
        "missing_masks": int(np.isnan(area).sum()),
        "full_frame_masks": int((area > 0.99).sum()),
        "area_median": float(np.nanmedian(area)),
        "area_min": float(np.nanmin(area)),
        "area_max": float(np.nanmax(area)),
        "frames_area_jump_over_20pct": int((jumps > 0.2).sum()),
        "worst_jump_frames": [
            frames[i + 1] for i in np.argsort(-np.nan_to_num(jumps))[:5]
        ],
        "mask_generation": manifest.get("mask_generation"),
    }
    (out_dir / "mask_stats.json").write_text(json.dumps(stats, indent=2))

    picks = np.linspace(0, len(frames) - 1, min(tiles, len(frames))).astype(int)
    thumbs = []
    for i in picks:
        img = cv2.imread(str(image_dir / frames[i]))
        mask = cv2.imread(str(mask_dir / f"{frames[i]}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if mask is not None:
            tint = img.copy()
            tint[mask > 0] = (0.5 * tint[mask > 0] + [0, 0, 127]).astype(np.uint8)
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(tint, contours, -1, (0, 255, 0), 3)
            img = tint
        img = cv2.resize(img, (270, int(270 * img.shape[0] / img.shape[1])))
        cv2.putText(img, frames[i], (5, 20), 0, 0.5, (255, 255, 255), 1)
        thumbs.append(img)
    cols = 6
    h = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, 0) for t in thumbs]
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
    rows = [np.hstack(thumbs[r : r + cols]) for r in range(0, len(thumbs), cols)]
    cv2.imwrite(str(out_dir / "masks_sheet.png"), np.vstack(rows))
    print(json.dumps(stats, indent=2))
    return stats


def summary(run_dir: Path, frames_manifest: Path) -> dict:
    """Collect a run's numbers into ``diag_summary.json``.

    Parameters
    ----------
    run_dir : Path
        Pipeline output directory.
    frames_manifest : Path
        Frames manifest the run used (for marker-detection counts).

    Returns
    -------
    dict
        The summary written.
    """
    m = json.loads((run_dir / "pipeline_manifest.json").read_text())
    frames = json.loads(frames_manifest.read_text())
    backends = m.get("backends", {})
    sfm, mvs = backends.get("sfm") or {}, backends.get("mvs") or {}
    meshes = sorted(run_dir.glob("mesh*.ply"))
    final = [p for p in meshes if "lcc" not in p.name]
    area = None
    if final:
        area = float(o3d.io.read_triangle_mesh(str(final[0])).get_surface_area())
    detections = frames.get("marker_detections") or {}
    out = {
        "run_dir": str(run_dir),
        "diag_toggles": m.get("diag_toggles"),
        "frames": len(frames.get("frames", [])),
        "frames_with_marker_detections": sum(1 for v in detections.values() if v),
        "registered_images": sfm.get("registered_images"),
        "num_models": sfm.get("num_models"),
        "feature_masks": sfm.get("feature_masks"),
        "depth_range_sources": (mvs.get("depth") or {}).get("depth_range_sources"),
        "source_view_overlap": (mvs.get("depth") or {}).get("diag_source_overlap"),
        "fusion": mvs.get("fusion"),
        "sor": m.get("point_cloud_filtering"),
        "head_crop": m.get("head_crop"),
        "lcc": m.get("lcc"),
        "final_mesh": final[0].name if final else None,
        "final_mesh_area_sfm_units2": area,
        "stage_timings": m.get("stage_timings"),
    }
    for name in ("sparse_subject.ply", "sparse_background.ply"):
        if (run_dir / name).exists():
            pcd = o3d.io.read_point_cloud(str(run_dir / name))
            out[name.removesuffix(".ply") + "_points"] = len(pcd.points)
    (run_dir / "diag_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def depth_views(run_dir: Path, count: int = 4, min_confidence: float = 0.03) -> None:
    """Back-project single TransMVSNet depth maps to world-frame PLYs.

    For ``count`` evenly spaced views writes ``depthview_<id>_raw.ply`` (every
    pixel inside the subject mask) and ``depthview_<id>_conf.ply`` (only
    pixels with confidence >= ``min_confidence``), coloured by the image.

    Parameters
    ----------
    run_dir : Path
        Pipeline output directory with ``mvs/transmvsnet`` and
        ``mvs/view_masks``.
    count : int, optional
        Views to export.
    min_confidence : float, optional
        Confidence cut for the ``_conf`` file (fusion's default).
    """
    mvs = run_dir / "mvs"
    paths = sorted((mvs / "transmvsnet").glob("*.npz"))
    out_dir = run_dir / "depthviews"
    out_dir.mkdir(exist_ok=True)
    for path in [paths[i] for i in np.linspace(0, len(paths) - 1, count).astype(int)]:
        with np.load(path) as d:
            depth, conf = d["depth"], d["confidence"]
            K, extrinsic, name = d["K"], d["extrinsic"], str(d["name"])
        h, w = depth.shape
        image = cv2.imread(str(mvs / "images" / name))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.imread(str(mvs / "view_masks" / f"{name}.png"), 0)
        subject = (
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if mask is not None
            else np.ones((h, w), bool)
        )
        rows, cols = np.mgrid[0:h, 0:w]
        pix = np.stack([cols, rows, np.ones_like(cols)], -1).reshape(-1, 3)
        cam = (pix @ np.linalg.inv(K).T) * depth.reshape(-1, 1)
        world_from_cam = np.linalg.inv(extrinsic)
        world = cam @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
        colors = image.reshape(-1, 3)[:, ::-1] / 255.0
        valid = subject.reshape(-1) & (depth.reshape(-1) > 0)
        for tag, keep in (
            ("raw", valid),
            ("conf", valid & (conf.reshape(-1) >= min_confidence)),
        ):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world[keep])
            pcd.colors = o3d.utility.Vector3dVector(colors[keep])
            o3d.io.write_point_cloud(
                str(out_dir / f"depthview_{path.stem}_{tag}.ply"), pcd
            )
        print(
            f"{path.stem} {name}: depth {w}x{h}, mask px {int(valid.sum())}, "
            f"conf>={min_confidence}: {int((valid & (conf.reshape(-1) >= min_confidence)).sum())}"
        )


def visual_hull(
    run_dir: Path,
    frames_manifest: Path,
    image_dir: Path,
    resolution: int = 200,
    min_inside_fraction: float = 0.9,
) -> dict:
    """Carve the subject from masks and camera poses only (no texture).

    Voxels over the bounding box of the cropped dense cloud (1-99th
    percentile, +25 % margin) are kept when they project inside the subject
    mask in at least ``min_inside_fraction`` of the >= 5 masked views that
    see them. Writes ``hull_points.ply`` (surface voxels) and
    ``hull_mesh.ply`` (Poisson on those points), in SfM units.

    Parameters
    ----------
    run_dir : Path
        Pipeline output directory (``sparse/``, cropped dense cloud).
    frames_manifest : Path
        Frames manifest with ``mask_dir``.
    image_dir : Path
        Frames directory the manifest's ``mask_dir`` is relative to.
    resolution : int, optional
        Voxels along the longest box side.
    min_inside_fraction : float, optional
        Share of views whose mask must contain a voxel.

    Returns
    -------
    dict
        Grid size and voxel counts.
    """
    from learned_sfm_mvs.postprocess.silhouette_filter import _load_mask

    reconstruction, _ = load_best_reconstruction(run_dir / "sparse")
    mask_dir = _mask_dir(frames_manifest, image_dir)
    cropped = next(run_dir.glob("dense_filtered_cropped*.ply"))
    pts = np.asarray(o3d.io.read_point_cloud(str(cropped)).points)
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
    step = (hi - lo).max() / resolution
    dims = np.ceil((hi - lo) / step).astype(int)
    axes = [lo[i] + step * (np.arange(dims[i]) + 0.5) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    in_frame = np.zeros(len(grid), np.int32)
    inside = np.zeros(len(grid), np.int32)
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        camera = reconstruction.cameras[image.camera_id]
        mask = _load_mask(mask_dir, image.name, (camera.width, camera.height))
        if mask is None:
            continue
        pose = image.cam_from_world().matrix()
        cam = grid @ pose[:, :3].T + pose[:, 3]
        front = np.flatnonzero(cam[:, 2] > 0)
        xy = np.floor(camera.img_from_cam(cam[front])).astype(np.int64)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < camera.width)
        ok &= (xy[:, 1] >= 0) & (xy[:, 1] < camera.height)
        idx = front[ok]
        in_frame[idx] += 1
        inside[idx] += mask[xy[ok, 1], xy[ok, 0]]

    occ = ((in_frame >= 5) & (inside >= min_inside_fraction * in_frame)).reshape(dims)
    padded = np.pad(occ, 1)
    interior = padded[1:-1, 1:-1, 1:-1].copy()
    for axis in range(3):
        for shift in (1, -1):
            interior &= np.roll(padded, shift, axis)[1:-1, 1:-1, 1:-1]
    surface = occ & ~interior
    surface_pts = grid[surface.reshape(-1)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    centre = grid[occ.reshape(-1)].mean(axis=0)
    normals = np.asarray(pcd.normals)
    flip = np.sum(normals * (surface_pts - centre), axis=1) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(run_dir / "hull_points.ply"), pcd)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    o3d.io.write_triangle_mesh(str(run_dir / "hull_mesh.ply"), mesh)
    stats = {
        "grid": dims.tolist(),
        "voxel_size_sfm": float(step),
        "occupied_voxels": int(occ.sum()),
        "surface_voxels": int(surface.sum()),
        "min_inside_fraction": min_inside_fraction,
    }
    (run_dir / "hull_stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)
    return stats


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sparse")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p = sub.add_parser("masks")
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("summary")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-manifest", type=Path, required=True)
    p = sub.add_parser("depthviews")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p = sub.add_parser("hull")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=200)
    args = parser.parse_args()
    if args.cmd == "depthviews":
        depth_views(args.run_dir, args.count)
    elif args.cmd == "hull":
        visual_hull(args.run_dir, args.frames_manifest, args.image_dir, args.resolution)
    elif args.cmd == "sparse":
        export_sparse(args.run_dir, args.frames_manifest, args.image_dir)
    elif args.cmd == "masks":
        mask_overlays(args.frames_manifest, args.image_dir, args.out_dir)
    else:
        summary(args.run_dir, args.frames_manifest)


if __name__ == "__main__":
    main()
