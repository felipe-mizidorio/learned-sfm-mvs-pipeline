"""Markerless head crop from the frames-manifest masks (silhouette voting).

Without ArUco markers there is no metric scale to size a spherical crop, but
the preprocessing masks outline the subject in every frame. A point on the
subject projects inside the subject's silhouette in every view that has it in
frame, whether or not the subject occludes it there; a background point falls
outside most silhouettes. Keeping points inside the mask in most masked views
is a visual-hull test: no radius, no scale, and backend-independent (it works
on the original frames, cameras and masks).

Views whose mask is missing or covers the whole frame (a segmentation
fallback) do not vote.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

# Masks covering more of the frame are segmentation fallbacks, not a subject.
FULL_FRAME_FRACTION = 0.99
_CHUNK = 1_000_000


def _load_mask(mask_dir: Path, name: str, size: tuple[int, int]) -> np.ndarray | None:
    """Subject mask of one frame at ``size`` (width, height), or None."""
    mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    subject = mask > 0
    return None if subject.mean() > FULL_FRAME_FRACTION else subject


def silhouette_keep(
    points: np.ndarray,
    reconstruction: pycolmap.Reconstruction,
    mask_dir: Path,
    min_views: int,
    min_inside_fraction: float,
) -> tuple[np.ndarray | None, dict]:
    """Which points lie inside the subject masks in most masked views.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` points in the reconstruction's frame (SfM units).
    reconstruction : pycolmap.Reconstruction
        Sparse model with the original (distorted) cameras.
    mask_dir : Path
        Original-frame masks (COLMAP ``<image name>.png``, non-zero = subject).
    min_views : int
        Masked views that must have a point in frame for it to be judged.
    min_inside_fraction : float
        Share of those views whose mask must contain the point, in (0, 1].

    Returns
    -------
    keep : np.ndarray or None
        ``(N,)`` bool; None when no view has a usable mask.
    stats : dict
        ``head_crop`` fields for the manifest.

    Raises
    ------
    ValueError
        If ``min_inside_fraction`` is outside (0, 1].
    """
    if not 0.0 < min_inside_fraction <= 1.0:
        raise ValueError(
            f"min_inside_fraction must be in (0, 1], got {min_inside_fraction}"
        )
    in_frame = np.zeros(len(points), dtype=np.int32)
    inside = np.zeros(len(points), dtype=np.int32)
    views_used = 0
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        camera = reconstruction.cameras[image.camera_id]
        mask = _load_mask(mask_dir, image.name, (camera.width, camera.height))
        if mask is None:
            continue
        views_used += 1
        pose = image.cam_from_world().matrix()
        for start in range(0, len(points), _CHUNK):
            cam = points[start : start + _CHUNK] @ pose[:, :3].T + pose[:, 3]
            front = np.flatnonzero(cam[:, 2] > 0)
            xy = np.floor(camera.img_from_cam(cam[front])).astype(np.int64)
            valid = (
                (xy[:, 0] >= 0)
                & (xy[:, 0] < camera.width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < camera.height)
            )
            idx = start + front[valid]
            in_frame[idx] += 1
            inside[idx] += mask[xy[valid, 1], xy[valid, 0]]

    stats = {
        "method": "silhouette",
        "views_used": views_used,
        "min_views": int(min_views),
        "min_inside_fraction": float(min_inside_fraction),
        "points_before": len(points),
    }
    if views_used == 0:
        logger.warning("Silhouette crop: no usable subject mask in any view.")
        return None, stats
    keep = (in_frame >= min_views) & (inside >= min_inside_fraction * in_frame)
    stats["points_after"] = int(keep.sum())
    logger.info(
        "Silhouette crop: keeping %d / %d points inside the masks of >= %.0f%% of "
        ">= %d masked views (%d views voted).",
        stats["points_after"],
        len(points),
        100 * min_inside_fraction,
        min_views,
        views_used,
    )
    return keep, stats
