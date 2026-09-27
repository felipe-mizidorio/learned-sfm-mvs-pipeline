"""Markerless diagnosis helpers (diag/markerless branch only).

Subcommands, all read-only on the run they inspect:

- ``sparse``: split the sparse model into subject / background point PLYs.
- ``masks``: mask-over-frame contact sheet and per-frame mask area series.
- ``summary``: one ``diag_summary.json`` with the numbers of a run.
- ``imgstats``: image signal inside the subject mask (exposure, contrast,
  sharpness, SIFT density), markers excluded.
- ``coverage``: camera viewpoints around the head from the sparse model.
- ``paintout``: copy of a frames session with the ArUco markers inpainted
  (writes a new session; the input is untouched).

Usage::

    python scripts/diag/diag.py sparse  --run-dir OUT --frames-manifest M.json --image-dir IMGS
    python scripts/diag/diag.py masks   --frames-manifest M.json --image-dir IMGS --out-dir OUT
    python scripts/diag/diag.py summary --run-dir OUT --frames-manifest M.json
    python scripts/diag/diag.py imgstats --frames-manifest M.json --image-dir IMGS --out-dir OUT
    python scripts/diag/diag.py coverage --run-dir OUT --frames-manifest M.json --image-dir IMGS
    python scripts/diag/diag.py paintout --frames-manifest M.json --image-dir IMGS --out-session NEW
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


def _hull_scene(run_dir: Path) -> tuple[o3d.t.geometry.RaycastingScene, float]:
    """Raycasting scene of ``hull_mesh.ply`` and the hull's equivalent radius."""
    mesh = o3d.io.read_triangle_mesh(str(run_dir / "hull_mesh.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    stats = json.loads((run_dir / "hull_stats.json").read_text())
    volume = stats["occupied_voxels"] * stats["voxel_size_sfm"] ** 3
    return scene, float((3 * volume / (4 * np.pi)) ** (1 / 3))


def _signed_hull_distance(
    scene: o3d.t.geometry.RaycastingScene, points: np.ndarray
) -> np.ndarray:
    """Distance to the hull surface, negative inside the hull."""
    query = o3d.core.Tensor(points.astype(np.float32))
    dist = scene.compute_distance(query).numpy()
    inside = scene.compute_occupancy(query).numpy() > 0
    return np.where(inside, -dist, dist)


def hull_noise(run_dir: Path) -> dict:
    """How far single-view depth and fused points sit from the visual hull.

    Distances are divided by the hull's equivalent-sphere radius R so runs in
    different SfM units compare. Writes ``hull_noise.json``.

    Parameters
    ----------
    run_dir : Path
        Run with ``hull_mesh.ply``, ``hull_stats.json``, ``depthviews/`` and
        the cropped dense cloud.

    Returns
    -------
    dict
        Per cloud: point count, median / p90 of |d|/R, median signed d/R
        (negative = inside the hull), share within 2 % and 5 % of R.
    """
    scene, radius = _hull_scene(run_dir)
    clouds = sorted((run_dir / "depthviews").glob("*.ply"))
    clouds.append(next(run_dir.glob("dense_filtered_cropped*.ply")))
    out: dict = {"hull_radius_sfm": radius}
    for path in clouds:
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if len(pts) == 0:
            continue
        d = _signed_hull_distance(scene, pts) / radius
        a = np.abs(d)
        out[path.stem] = {
            "points": len(pts),
            "median_abs": float(np.median(a)),
            "p90_abs": float(np.percentile(a, 90)),
            "median_signed": float(np.median(d)),
            "within_2pct": float(np.mean(a < 0.02)),
            "within_5pct": float(np.mean(a < 0.05)),
        }
    (run_dir / "hull_noise.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def hull_band(run_dir: Path, bands: tuple[float, ...] = (0.02, 0.05)) -> dict:
    """Hull-guided mesh: fused points near the hull, hull where MVS is empty.

    For each band (fraction of the hull radius R) writes
    ``hullband_<pct>_mvsonly_mesh.ply`` (band-filtered fused points only) and
    ``hullband_<pct>_mesh.ply`` (plus hull surface samples with no fused
    point within the band), both Poisson depth 9, largest component.

    Parameters
    ----------
    run_dir : Path
        Run with ``hull_mesh.ply``, ``hull_stats.json`` and the cropped cloud.
    bands : tuple[float, ...], optional
        Band half-widths as fractions of R.

    Returns
    -------
    dict
        Point counts per band.
    """
    scene, radius = _hull_scene(run_dir)
    fused = o3d.io.read_point_cloud(
        str(next(run_dir.glob("dense_filtered_cropped*.ply")))
    )
    pts = np.asarray(fused.points)
    d = np.abs(_signed_hull_distance(scene, pts))
    hull = o3d.io.read_triangle_mesh(str(run_dir / "hull_mesh.ply"))
    hull.compute_vertex_normals()
    hull_samples = hull.sample_points_uniformly(200_000, use_triangle_normal=True)
    out: dict = {"hull_radius_sfm": radius, "fused_points": len(pts)}
    for band in bands:
        tag = f"{round(band * 100)}pct"
        near = fused.select_by_index(np.flatnonzero(d < band * radius).tolist())
        tree = o3d.geometry.KDTreeFlann(near)
        empty = [
            i
            for i, p in enumerate(np.asarray(hull_samples.points))
            if tree.search_radius_vector_3d(p, band * radius)[0] == 0
        ]
        fill = hull_samples.select_by_index(empty)
        for name, cloud in (("mvsonly", near), ("filled", near + fill)):
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=9
            )
            clusters, counts, _ = mesh.cluster_connected_triangles()
            mesh.remove_triangles_by_mask(
                np.asarray(clusters) != int(np.argmax(counts))
            )
            mesh.remove_unreferenced_vertices()
            suffix = "mvsonly_mesh" if name == "mvsonly" else "mesh"
            o3d.io.write_triangle_mesh(
                str(run_dir / f"hullband_{tag}_{suffix}.ply"), mesh
            )
        o3d.io.write_point_cloud(
            str(run_dir / f"hullband_{tag}_points.ply"), near + fill
        )
        out[tag] = {
            "fused_points_in_band": len(near.points),
            "fused_share_in_band": len(near.points) / max(len(pts), 1),
            "hull_fill_points": len(fill.points),
            "hull_fill_share_of_surface": len(fill.points) / len(hull_samples.points),
        }
    (run_dir / "hull_band.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def hull_band_normals(run_dir: Path, band: float = 0.05) -> dict:
    """Hull-band mesh with the normals replaced, one variant per normal source.

    Same band and hull fill as ``hull_band``, but hull-fill points are grey
    (not black) and the fused points' normals come from:

    - ``fused``: fusion's per-pixel depth-gradient normals (unchanged; only
      the fill colour differs from ``hull_band``);
    - ``hull``: the normal of the nearest hull-mesh triangle;
    - ``pca``: PCA over the 60 nearest points, flipped to agree with the
      nearest hull normal.

    Writes ``hullband2_<pct>_<source>_mesh.ply`` and ``hullband2_<pct>_points.ply``.

    Parameters
    ----------
    run_dir : Path
        Run with ``hull_mesh.ply``, ``hull_stats.json`` and the cropped cloud.
    band : float, optional
        Band half-width as a fraction of the hull radius.

    Returns
    -------
    dict
        Point counts.
    """
    scene, radius = _hull_scene(run_dir)
    fused = o3d.io.read_point_cloud(
        str(next(run_dir.glob("dense_filtered_cropped*.ply")))
    )
    d = np.abs(_signed_hull_distance(scene, np.asarray(fused.points)))
    near = fused.select_by_index(np.flatnonzero(d < band * radius).tolist())
    hull = o3d.io.read_triangle_mesh(str(run_dir / "hull_mesh.ply"))
    hull.compute_vertex_normals()
    samples = hull.sample_points_uniformly(200_000, use_triangle_normal=True)
    tree = o3d.geometry.KDTreeFlann(near)
    empty = [
        i
        for i, p in enumerate(np.asarray(samples.points))
        if tree.search_radius_vector_3d(p, band * radius)[0] == 0
    ]
    fill = samples.select_by_index(empty)
    fill.paint_uniform_color([0.6, 0.6, 0.6])

    query = o3d.core.Tensor(np.asarray(near.points, dtype=np.float32))
    hull_normals = scene.compute_closest_points(query)["primitive_normals"].numpy()
    pca = o3d.geometry.PointCloud(near)
    pca.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(60))
    pca_normals = np.asarray(pca.normals)
    flip = np.sum(pca_normals * hull_normals, axis=1) < 0
    pca_normals[flip] *= -1

    tag = f"{round(band * 100)}pct"
    for source, normals in (
        ("fused", np.asarray(near.normals)),
        ("hull", hull_normals.astype(np.float64)),
        ("pca", pca_normals),
    ):
        cloud = o3d.geometry.PointCloud(near)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud += fill
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=9
        )
        clusters, counts, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(clusters) != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(
            str(run_dir / f"hullband2_{tag}_{source}_mesh.ply"), mesh
        )
    o3d.io.write_point_cloud(str(run_dir / f"hullband2_{tag}_points.ply"), near + fill)
    out = {
        "band": band,
        "fused_points_in_band": len(near.points),
        "hull_fill_points": len(fill.points),
        "pca_normals_flipped": int(flip.sum()),
        "fused_vs_hull_normal_median_angle_deg": float(
            np.degrees(
                np.median(
                    np.arccos(
                        np.clip(
                            np.abs(np.sum(np.asarray(near.normals) * hull_normals, 1)),
                            0,
                            1,
                        )
                    )
                )
            )
        ),
    }
    (run_dir / "hull_band_normals.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _marker_mask(
    image: np.ndarray,
    head: np.ndarray,
    detections: list,
    dilate_px: int = 12,
    bright: int = 140,
) -> np.ndarray:
    """Pixels covered by ArUco markers (detected or not).

    Detected marker quads, plus every bright blob inside the head (the white
    paper of undetected or oblique markers), all dilated by ``dilate_px``.
    Marker-sized blobs (hull at most 5 % of the head) are filled as their
    convex hull so the black code cells go too; larger blobs (merged
    highlights) are painted pixel by pixel. Meant for dark subjects only: on
    a pale subject the whole head is "bright".

    Parameters
    ----------
    image : np.ndarray
        BGR frame.
    head : np.ndarray
        Boolean subject mask.
    detections : list
        ``[{id, corners}]`` for this frame, as in the frames manifest.
    dilate_px : int, optional
        Growth of the final mask, pixels.
    bright : int, optional
        Grey level above which a head pixel counts as marker paper.

    Returns
    -------
    np.ndarray
        Boolean marker mask, image-sized.
    """
    out = np.zeros(head.shape, np.uint8)
    for det in detections:
        quad = np.asarray(det["corners"], np.float32).reshape(-1, 2)
        cv2.fillConvexPoly(out, quad.astype(np.int32), 1)
    paper = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > bright) & head
    paper = cv2.dilate(paper.astype(np.uint8), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper)
    max_hull = 0.05 * head.sum()
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 30:
            continue
        blob = labels == label
        hull = cv2.convexHull(np.column_stack(np.nonzero(blob)[::-1]).astype(np.int32))
        if cv2.contourArea(hull) <= max_hull:
            cv2.fillConvexPoly(out, hull, 1)
        else:
            out[blob] = 1
    size = 2 * dilate_px + 1
    return cv2.dilate(out, np.ones((size, size), np.uint8)) > 0


def _region_stats(
    image: np.ndarray, gray: np.ndarray, region: np.ndarray, sift: cv2.SIFT
) -> dict:
    """Exposure, contrast, sharpness and SIFT density inside ``region``."""
    g = gray.astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mean = cv2.blur(g, (7, 7))
    local_std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mean * mean, 0))
    lap = cv2.Laplacian(g, cv2.CV_32F)
    px = int(region.sum())
    keypoints = sift.detect(gray, region.astype(np.uint8) * 255)
    channel_max = image.max(axis=2)[region]
    return {
        "px": px,
        "mean": float(g[region].mean()),
        "std": float(g[region].std()),
        "clipped_frac": float((channel_max >= 250).mean()),
        "crushed_frac": float((channel_max <= 5).mean()),
        "grad_median": float(np.median(grad[region])),
        "local_std_median": float(np.median(local_std[region])),
        "local_rel_contrast_median": float(
            np.median(local_std[region] / np.maximum(mean[region], 1.0))
        ),
        "laplacian_var": float(lap[region].var()),
        "sift_per_10k_px": 1e4 * len(keypoints) / max(px, 1),
    }


def image_stats(
    frames_manifest: Path,
    image_dir: Path,
    out_dir: Path,
    erode_px: int = 15,
    tiles: int = 6,
) -> dict:
    """Image signal on the subject: is the head overexposed, blurred or flat?

    Per frame, inside the subject mask eroded by ``erode_px`` (so the
    silhouette edge does not count as texture): grey mean/std, clipped and
    crushed pixel shares, gradient, local contrast (absolute and relative to
    the local mean), Laplacian variance and SIFT keypoints per 10k pixels.
    When the manifest has marker detections the same numbers are repeated on
    ``surface`` = head minus markers (``_marker_mask``). The whole-frame
    Laplacian variance (what aruco-frame-preprocessing's blur filter scores)
    is kept for comparison. Writes ``imgstats.csv``, ``imgstats.json`` and
    ``imgstats_sheet.png`` (head crops, top row as saved, bottom row CLAHE
    to show latent texture).

    Parameters
    ----------
    frames_manifest : Path
        Frames manifest (``frames``, ``mask_dir``, optional
        ``marker_detections``).
    image_dir : Path
        Frames directory.
    out_dir : Path
        Output directory.
    erode_px : int, optional
        Mask erosion, pixels.
    tiles : int, optional
        Frames in the sheet, evenly spaced.

    Returns
    -------
    dict
        Median / p10 / p90 over frames of every per-frame number.
    """
    manifest = json.loads(frames_manifest.read_text())
    frames = manifest["frames"]
    mask_dir = image_dir / manifest["mask_dir"]
    detections = manifest.get("marker_detections") or {}
    markers = any(detections.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create()
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    picks = set(np.linspace(0, len(frames) - 1, tiles).astype(int).tolist())

    rows, crops = [], []
    for i, name in enumerate(frames):
        image = cv2.imread(str(image_dir / name))
        mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or (mask > 0).mean() > 0.99:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        head = cv2.erode((mask > 0).astype(np.uint8), kernel) > 0
        if head.sum() < 1000:
            continue
        row = {"frame": name, "mask_frac": float((mask > 0).mean())}
        row["frame_laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        regions = {"head": head}
        if markers:
            regions["surface"] = head & ~_marker_mask(
                image, mask > 0, detections.get(name, [])
            )
            row["marker_frac_of_head"] = 1 - regions["surface"].sum() / head.sum()
        for label, region in regions.items():
            for key, value in _region_stats(image, gray, region, sift).items():
                row[f"{label}_{key}"] = value
        rows.append(row)
        if i in picks:
            ys, xs = np.nonzero(mask > 0)
            crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            enhanced = cv2.cvtColor(
                clahe.apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR
            )
            scale = 360 / crop.shape[0]
            size = (max(1, int(crop.shape[1] * scale)), 360)
            crops.append(
                np.vstack([cv2.resize(crop, size), cv2.resize(enhanced, size)])
            )

    keys = [k for k in rows[0] if k != "frame"]
    with (out_dir / "imgstats.csv").open("w") as f:
        f.write(",".join(["frame", *keys]) + "\n")
        f.writelines(
            ",".join([r["frame"], *(f"{r.get(k, float('nan')):.6g}" for k in keys)])
            + "\n"
            for r in rows
        )
    summary_out: dict = {"frames": len(rows), "markers_in_manifest": markers}
    for key in keys:
        values = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        summary_out[key] = {
            "median": float(np.nanmedian(values)),
            "p10": float(np.nanpercentile(values, 10)),
            "p90": float(np.nanpercentile(values, 90)),
        }
    (out_dir / "imgstats.json").write_text(json.dumps(summary_out, indent=2))
    if crops:
        cv2.imwrite(str(out_dir / "imgstats_sheet.png"), np.hstack(crops))
    print(json.dumps(summary_out, indent=2))
    return summary_out


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def coverage(run_dir: Path, frames_manifest: Path, image_dir: Path) -> dict:
    """Where the registered cameras sit around the head.

    Head centre and radius R come from the visual hull (``hull_points.ply`` /
    ``hull_stats.json``) when present, else from the subject sparse points.
    "Up" is the mean of the cameras' image-up axes. Per camera: distance / R,
    elevation, azimuth, and the angle between its optical axis and the
    direction to the head. Direction coverage counts, for 2000 directions
    around the head, the cameras within 30 degrees of it. Writes
    ``coverage.json`` and ``coverage.png`` (azimuth vs elevation, colour =
    frame order).

    Parameters
    ----------
    run_dir : Path
        Pipeline output directory (``sparse/``, optionally the hull files).
    frames_manifest : Path
        Frames manifest with ``mask_dir``.
    image_dir : Path
        Frames directory the manifest's ``mask_dir`` is relative to.

    Returns
    -------
    dict
        Coverage statistics.
    """
    reconstruction, _ = load_best_reconstruction(run_dir / "sparse")
    images = sorted(
        (im for im in reconstruction.images.values() if im.has_pose),
        key=lambda im: im.name,
    )
    if (run_dir / "hull_points.ply").exists():
        hull = np.asarray(
            o3d.io.read_point_cloud(str(run_dir / "hull_points.ply")).points
        )
        centre = hull.mean(axis=0)
        _, radius = _hull_scene(run_dir)
        source = "visual_hull"
    else:
        subject = subject_point_ids(images, _mask_dir(frames_manifest, image_dir))
        pts = np.array([reconstruction.points3D[p].xyz for p in subject])
        centre = np.median(pts, axis=0)
        radius = float(np.median(np.linalg.norm(pts - centre, axis=1)))
        source = "sparse_subject_points"

    poses = [im.cam_from_world().matrix() for im in images]
    centres = np.array([-p[:, :3].T @ p[:, 3] for p in poses])
    up = -np.mean([p[1, :3] for p in poses], axis=0)
    up /= np.linalg.norm(up)
    v = centres - centre
    dist = np.linalg.norm(v, axis=1)
    unit = v / dist[:, None]
    e1 = unit[0] - (unit[0] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    elevation = np.degrees(np.arcsin(np.clip(unit @ up, -1, 1)))
    azimuth = np.degrees(np.arctan2(unit @ e2, unit @ e1)) % 360
    optical = np.array([p[2, :3] for p in poses])
    off_axis = np.degrees(np.arccos(np.clip(np.sum(optical * -unit, 1), -1, 1)))
    steps = np.degrees(np.arccos(np.clip(np.sum(unit[1:] * unit[:-1], 1), -1, 1)))

    az_sorted = np.sort(azimuth)
    az_gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360]]))
    dirs = _fibonacci_sphere(2000)
    seen = (dirs @ unit.T > np.cos(np.radians(30))).sum(axis=1)
    height = dirs @ up

    def _share(sel: np.ndarray, k: int) -> float:
        return float(np.mean(seen[sel] >= k))

    out = {
        "run_dir": str(run_dir),
        "centre_source": source,
        "head_radius_sfm": radius,
        "registered_images": len(images),
        "distance_over_radius": {
            "min": float(dist.min() / radius),
            "median": float(np.median(dist) / radius),
            "max": float(dist.max() / radius),
        },
        "elevation_deg": {
            q: float(np.percentile(elevation, p))
            for q, p in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90))
        }
        | {"max": float(elevation.max())},
        "views_elevation_over_45deg": int((elevation > 45).sum()),
        "azimuth_empty_10deg_bins": int(
            36 - len(np.unique((azimuth // 10).astype(int)))
        ),
        "azimuth_largest_gap_deg": float(az_gaps.max()),
        "consecutive_step_deg": {
            "median": float(np.median(steps)),
            "p90": float(np.percentile(steps, 90)),
            "max": float(steps.max()),
        },
        "optical_axis_off_head_deg_median": float(np.median(off_axis)),
        "directions_seen_by_3plus_views": {
            "all": _share(np.ones(len(dirs), bool), 3),
            "upper_hemisphere": _share(height > 0, 3),
            "top_cap_30deg": _share(height > np.cos(np.radians(30)), 3),
            "equator_band_30deg": _share(np.abs(height) < np.sin(np.radians(30)), 3),
        },
    }
    canvas = np.full((400, 740, 3), 255, np.uint8)
    for deg in range(0, 361, 90):
        cv2.line(canvas, (20 + 2 * deg, 20), (20 + 2 * deg, 380), (220, 220, 220), 1)
    for deg in (-45, 0, 45, 90):
        y = 200 - 2 * deg
        cv2.line(canvas, (20, y), (740, y), (220, 220, 220), 1)
        cv2.putText(canvas, str(deg), (0, y + 4), 0, 0.35, (0, 0, 0), 1)
    for k, (az, el) in enumerate(zip(azimuth, elevation)):
        colour = cv2.applyColorMap(
            np.uint8([[255 * k / max(len(images) - 1, 1)]]), cv2.COLORMAP_VIRIDIS
        )[0, 0].tolist()
        cv2.circle(canvas, (int(20 + 2 * az), int(200 - 2 * el)), 3, colour, -1)
    cv2.imwrite(str(run_dir / "coverage.png"), canvas)
    (run_dir / "coverage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def paint_out(
    frames_manifest: Path, image_dir: Path, out_session: Path, tiles: int = 6
) -> dict:
    """New frames session with every marker inpainted (markers-as-texture test).

    Copies the session layout (``filtered/*.jpg``, ``filtered/masks``,
    ``manifest.json``) to ``out_session``; each frame has its
    ``_marker_mask`` region filled by ``cv2.inpaint`` (Telea) and is saved as
    JPEG q95 like the originals. ``marker_detections`` becomes empty for every
    frame so the pipeline runs markerless. Writes ``paintout_sheet.png``
    (before/after head crops) and ``paintout.json``.

    Parameters
    ----------
    frames_manifest : Path
        Source manifest with ``marker_detections``.
    image_dir : Path
        Source frames directory (``filtered/``).
    out_session : Path
        New session directory; must not exist.
    tiles : int, optional
        Frames in the sheet.

    Returns
    -------
    dict
        Painted share of the head per frame (median / max).
    """
    import shutil

    manifest = json.loads(frames_manifest.read_text())
    frames = manifest["frames"]
    detections = manifest.get("marker_detections") or {}
    mask_dir = image_dir / manifest["mask_dir"]
    out_images = out_session / "filtered"
    out_session.mkdir(parents=True)
    shutil.copytree(mask_dir, out_images / manifest["mask_dir"])
    picks = set(np.linspace(0, len(frames) - 1, tiles).astype(int).tolist())

    shares, crops = [], []
    for i, name in enumerate(frames):
        image = cv2.imread(str(image_dir / name))
        mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        head = mask > 0 if mask is not None else np.zeros(image.shape[:2], bool)
        paint = _marker_mask(image, head, detections.get(name, []))
        painted = cv2.inpaint(image, paint.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        cv2.imwrite(str(out_images / name), painted)
        if head.any():
            shares.append(float((paint & head).sum() / head.sum()))
        if i in picks and head.any():
            ys, xs = np.nonzero(head)
            box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            pair = np.hstack([image[box], painted[box]])
            scale = 360 / pair.shape[0]
            crops.append(cv2.resize(pair, (int(pair.shape[1] * scale), 360)))

    manifest["marker_detections"] = {name: [] for name in frames}
    manifest["diag_paintout"] = {"source_manifest": str(frames_manifest)}
    (out_session / "manifest.json").write_text(json.dumps(manifest, indent=2))
    out = {
        "frames": len(frames),
        "painted_share_of_head_median": float(np.median(shares)),
        "painted_share_of_head_max": float(np.max(shares)),
    }
    (out_session / "paintout.json").write_text(json.dumps(out, indent=2))
    if crops:
        width = max(c.shape[1] for c in crops)
        crops = [cv2.copyMakeBorder(c, 0, 0, 0, width - c.shape[1], 0) for c in crops]
        cv2.imwrite(str(out_session / "paintout_sheet.png"), np.vstack(crops))
    print(json.dumps(out, indent=2))
    return out


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
    p = sub.add_parser("hullnoise")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("hullband")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("hullbandnormals")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("imgstats")
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("coverage")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p = sub.add_parser("paintout")
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-session", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "imgstats":
        image_stats(args.frames_manifest, args.image_dir, args.out_dir)
    elif args.cmd == "coverage":
        coverage(args.run_dir, args.frames_manifest, args.image_dir)
    elif args.cmd == "paintout":
        paint_out(args.frames_manifest, args.image_dir, args.out_session)
    elif args.cmd == "hullbandnormals":
        hull_band_normals(args.run_dir)
    elif args.cmd == "hullnoise":
        hull_noise(args.run_dir)
    elif args.cmd == "hullband":
        hull_band(args.run_dir)
    elif args.cmd == "depthviews":
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
