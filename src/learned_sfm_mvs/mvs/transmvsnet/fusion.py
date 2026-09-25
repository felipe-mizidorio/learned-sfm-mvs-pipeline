"""Fuse TransMVSNet depth maps into a dense point cloud.

Vectorized torch port of upstream's "normal" filter (``test.py``,
``filter_depth``). A reference pixel is kept when its confidence exceeds
``min_confidence`` and at least ``min_consistent_views`` source depth maps
agree with it: reprojected there and back it lands within
``max_reproj_error_px`` pixels and ``max_relative_depth_error`` relative depth.
Kept depths are averaged over the agreeing views and back-projected.

Two additions over upstream:

* Normals are computed from each averaged depth map and face the camera, so
  Poisson gets consistently oriented normals.
* ``dedupe``: every pixel that agreed with an already-fused reference view is
  not emitted again when its own view is the reference. Upstream emits each
  surface point about once per agreeing view (its ``used_mask`` is commented
  out), multiplying the cloud size without adding geometry.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from learned_sfm_mvs.mvs.transmvsnet.inference import load_image

logger = logging.getLogger(__name__)


def pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Homogeneous pixel coordinates, pixel centres at integers.

    Parameters
    ----------
    height, width : int
        Map size.
    device : torch.device
        Tensor device.

    Returns
    -------
    torch.Tensor
        ``(3, H*W)`` rows x, y, 1.
    """
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    return torch.stack([x.reshape(-1), y.reshape(-1), torch.ones_like(x).reshape(-1)])


def back_project(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Camera-frame points of a depth map.

    Parameters
    ----------
    depth : torch.Tensor
        ``(H, W)`` depth.
    K : torch.Tensor
        ``(3, 3)`` intrinsics.

    Returns
    -------
    torch.Tensor
        ``(3, H*W)`` points.
    """
    height, width = depth.shape
    rays = torch.linalg.inv(K) @ pixel_grid(height, width, depth.device)
    return rays * depth.reshape(1, -1).double()


def check_geometric_consistency(
    depth_ref: torch.Tensor,
    K_ref: torch.Tensor,
    E_ref: torch.Tensor,
    depth_src: torch.Tensor,
    K_src: torch.Tensor,
    E_src: torch.Tensor,
    max_reproj_error_px: float,
    max_relative_depth_error: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-backward check of a reference depth map against a source one.

    Parameters
    ----------
    depth_ref, depth_src : torch.Tensor
        ``(H, W)`` depth maps (sizes may differ).
    K_ref, K_src : torch.Tensor
        ``(3, 3)`` intrinsics.
    E_ref, E_src : torch.Tensor
        ``(4, 4)`` world-to-camera transforms.
    max_reproj_error_px : float
        Maximum round-trip pixel distance.
    max_relative_depth_error : float
        Maximum relative depth difference.

    Returns
    -------
    mask : torch.Tensor
        ``(H, W)`` bool, reference pixels consistent with the source view.
    depth_reprojected : torch.Tensor
        ``(H, W)`` source depth brought back to the reference view, 0 where
        inconsistent.
    xy_src : torch.Tensor
        ``(2, H*W)`` where each reference pixel lands in the source view.
    """
    height, width = depth_ref.shape
    src_h, src_w = depth_src.shape
    grid = pixel_grid(height, width, depth_ref.device)

    xyz_ref = back_project(depth_ref, K_ref)
    ref_to_src = E_src @ torch.linalg.inv(E_ref)
    xyz_src = ref_to_src[:3, :3] @ xyz_ref + ref_to_src[:3, 3:]
    proj = K_src @ xyz_src
    xy_src = proj[:2] / proj[2:].clamp(min=1e-9)

    # Bilinear lookup of the source depth, zero outside (cv2.remap in upstream).
    norm = torch.stack([xy_src[0] / (src_w - 1), xy_src[1] / (src_h - 1)]) * 2 - 1
    sampled = F.grid_sample(
        depth_src[None, None].double(),
        norm.T.reshape(1, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(1, -1)

    xyz_src_back = torch.linalg.inv(K_src) @ torch.cat(
        [xy_src, torch.ones_like(sampled)]
    )
    xyz_src_back = xyz_src_back * sampled
    src_to_ref = torch.linalg.inv(ref_to_src)
    xyz_back = src_to_ref[:3, :3] @ xyz_src_back + src_to_ref[:3, 3:]
    depth_back = xyz_back[2]
    proj_back = K_ref @ xyz_back
    xy_back = proj_back[:2] / proj_back[2:].clamp(min=1e-9)

    dist = torch.linalg.norm(xy_back - grid[:2], dim=0)
    ref_depth = depth_ref.reshape(-1).double()
    relative = (depth_back - ref_depth).abs() / ref_depth.clamp(min=1e-9)
    mask = (
        (dist < max_reproj_error_px)
        & (relative < max_relative_depth_error)
        & (ref_depth > 0)
        & (proj[2] > 0)
        & (sampled[0] > 0)
    )
    depth_reprojected = torch.where(mask, depth_back, torch.zeros_like(depth_back))
    return mask.reshape(height, width), depth_reprojected.reshape(height, width), xy_src


def camera_facing_normals(
    points: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Per-pixel normals of a camera-frame point map, oriented to the camera.

    Parameters
    ----------
    points : torch.Tensor
        ``(3, H*W)`` camera-frame points.
    height, width : int
        Map size.

    Returns
    -------
    torch.Tensor
        ``(3, H*W)`` unit normals (zero where undefined).
    """
    grid = points.reshape(3, height, width)
    # Central differences, replicated at the borders.
    padded = F.pad(grid[None], (1, 1, 1, 1), mode="replicate")[0]
    d_x = padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]
    d_y = padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]
    normals = torch.linalg.cross(d_x, d_y, dim=0).reshape(3, -1)
    normals = normals / torch.linalg.norm(normals, dim=0).clamp(min=1e-12)
    # The camera sits at the origin: flip normals pointing away from it.
    flip = (normals * points).sum(0) > 0
    return torch.where(flip, -normals, normals)


def _load(depth_dir: Path) -> dict[int, dict]:
    maps = {}
    for path in sorted(depth_dir.glob("*.npz")):
        with np.load(path) as data:
            maps[int(path.stem)] = {
                "depth": data["depth"].astype(np.float32),
                "confidence": data["confidence"].astype(np.float32),
                "K": data["K"],
                "extrinsic": data["extrinsic"],
                "src_ids": [int(s) for s in data["src_ids"]],
                "name": str(data["name"]),
            }
    return maps


def _fusion_mask(
    mask_dir: Path | None, name: str, size: tuple[int, int]
) -> np.ndarray | None:
    """Undistorted workspace mask resized to the depth map, or None."""
    if mask_dir is None:
        return None
    mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0


def fuse_depth_maps(
    depth_dir: Path,
    workspace: Path,
    output_ply: Path,
    fusion_cfg: dict,
    device: torch.device,
    mask_dir: Path | None = None,
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
) -> dict:
    """Filter, average and back-project all depth maps into one cloud.

    Parameters
    ----------
    depth_dir : Path
        ``<image_id>.npz`` files from ``inference.run_inference``.
    workspace : Path
        Undistorted MVS workspace, for colours (``images/``).
    output_ply : Path
        Output cloud with colours and normals.
    fusion_cfg : dict
        ``fusion`` section of ``configs/transmvsnet.yaml``.
    device : torch.device
        Compute device.
    mask_dir : Path or None, optional
        Masks aligned to the undistorted images; only non-zero pixels fuse.
    bbox_min, bbox_max : list[float] or None, optional
        Axis-aligned box that clips fused points; both or neither.

    Returns
    -------
    dict
        Point counts for the manifest.

    Raises
    ------
    FileNotFoundError
        If ``depth_dir`` holds no depth maps.
    RuntimeError
        If no point survives filtering.
    """
    maps = _load(depth_dir)
    if not maps:
        raise FileNotFoundError(f"No TransMVSNet depth maps in '{depth_dir}'")
    min_conf = float(fusion_cfg["min_confidence"])
    min_views = int(fusion_cfg["min_consistent_views"])
    reproj_px = float(fusion_cfg["max_reproj_error_px"])
    rel_depth = float(fusion_cfg["max_relative_depth_error"])
    dedupe = bool(fusion_cfg["dedupe"])

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).to(device)

    used = {i: torch.zeros(m["depth"].shape, dtype=torch.bool) for i, m in maps.items()}
    points, colors, normals = [], [], []
    for ref_id, ref in maps.items():
        height, width = ref["depth"].shape
        depth_ref = tensor(ref["depth"])
        K_ref = tensor(ref["K"]).double()
        E_ref = tensor(ref["extrinsic"]).double()

        consistent_count = torch.zeros(
            (height, width), dtype=torch.int32, device=device
        )
        depth_sum = depth_ref.double().clone()
        landings = []
        for src_id in ref["src_ids"]:
            if src_id not in maps:
                continue
            src = maps[src_id]
            mask, depth_reprojected, xy_src = check_geometric_consistency(
                depth_ref,
                K_ref,
                E_ref,
                tensor(src["depth"]),
                tensor(src["K"]).double(),
                tensor(src["extrinsic"]).double(),
                reproj_px,
                rel_depth,
            )
            consistent_count += mask.int()
            depth_sum += depth_reprojected
            landings.append((src_id, mask.reshape(-1), xy_src))

        keep = (
            (tensor(ref["confidence"]) > min_conf)
            & (consistent_count >= min_views)
            & ~used[ref_id].to(device)
        )
        fusion_mask = _fusion_mask(mask_dir, ref["name"], (width, height))
        if fusion_mask is not None:
            keep &= tensor(fusion_mask)
        keep = keep.reshape(-1)

        if dedupe:
            # Source pixels that agreed with a kept reference pixel describe
            # the same surface point: do not emit them again later.
            for src_id, mask, xy_src in landings:
                hit = keep & mask
                src_h, src_w = maps[src_id]["depth"].shape
                cols = xy_src[0, hit].round().long().clamp(0, src_w - 1).cpu()
                rows = xy_src[1, hit].round().long().clamp(0, src_h - 1).cpu()
                used[src_id][rows, cols] = True

        depth_avg = depth_sum / (consistent_count.double() + 1)
        cam_points = back_project(depth_avg, K_ref)
        cam_normals = camera_facing_normals(cam_points, height, width)
        R_T = E_ref[:3, :3].T
        world = R_T @ (cam_points[:, keep] - E_ref[:3, 3:])
        points.append(world.T.float().cpu().numpy())
        normals.append((R_T @ cam_normals[:, keep]).T.float().cpu().numpy())
        rgb = load_image(workspace / "images" / ref["name"], (width, height))
        colors.append(rgb.reshape(-1, 3)[keep.cpu().numpy()])

    xyz = np.concatenate(points)
    rgb = np.concatenate(colors)
    nrm = np.concatenate(normals)
    num_before_bbox = len(xyz)
    if bbox_min is not None and bbox_max is not None:
        inside = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
        xyz, rgb, nrm = xyz[inside], rgb[inside], nrm[inside]
    if len(xyz) == 0:
        raise RuntimeError(
            "TransMVSNet fusion kept no points; check thresholds, masks and bbox."
        )

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    cloud.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), cloud)

    stats = {
        "depth_maps": len(maps),
        "points": len(xyz),
        "points_outside_bbox": num_before_bbox - len(xyz),
        "dedupe": dedupe,
    }
    logger.info("TransMVSNet fusion: %s -> '%s'", stats, output_ply)
    return stats
