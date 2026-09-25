"""Per-view inputs for TransMVSNet from an undistorted COLMAP workspace.

Replaces upstream's colmap2mvsnet.py export: instead of writing MVSNet
``cams/*.txt`` and ``pair.txt`` files, the same quantities are computed in
memory from the sparse model that ``pycolmap.undistort_images`` writes.

Pixel convention: COLMAP places pixel (0, 0)'s centre at (0.5, 0.5); MVSNet
code (``homo_warping``, the fusion reprojection) puts it at (0, 0). ``View.K``
is therefore COLMAP's calibration shifted by -0.5 px. Upstream skips this
shift, which offsets every back-projected point by half a pixel.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

_PINHOLE_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})


@dataclass(frozen=True)
class View:
    """One registered image of the MVS workspace.

    Parameters
    ----------
    image_id : int
        COLMAP image id.
    name : str
        Image name relative to the workspace's ``images/`` directory.
    K : np.ndarray
        ``(3, 3)`` intrinsics, pixel centres at integer coordinates.
    extrinsic : np.ndarray
        ``(4, 4)`` world-to-camera transform.
    width, height : int
        Image size in pixels.
    depth_min, depth_max : float
        Depth search range, from the view's observed sparse points.
    src_ids : tuple[int, ...]
        Source-view image ids, best first.
    """

    image_id: int
    name: str
    K: np.ndarray
    extrinsic: np.ndarray
    width: int
    height: int
    depth_min: float
    depth_max: float
    src_ids: tuple[int, ...]


def view_selection_scores(
    centers: np.ndarray,
    points: np.ndarray,
    tracks: list[np.ndarray],
    theta0: float,
    sigma1: float,
    sigma2: float,
) -> np.ndarray:
    """Pairwise view scores from shared points and their triangulation angle.

    MVSNet's heuristic (colmap2mvsnet.py): each point seen by views ``i`` and
    ``j`` adds ``exp(-(theta - theta0)^2 / (2 sigma^2))``, with ``theta`` the
    angle between the two viewing rays in degrees and ``sigma = sigma1`` below
    ``theta0``, ``sigma2`` above. Small baselines (theta << theta0) triangulate
    badly and are penalized harder than wide ones.

    Parameters
    ----------
    centers : np.ndarray
        ``(N, 3)`` camera centres in world coordinates.
    points : np.ndarray
        ``(P, 3)`` 3D points.
    tracks : list[np.ndarray]
        For each point, the indices (into ``centers``) of views observing it.
    theta0, sigma1, sigma2 : float
        Heuristic parameters, in degrees.

    Returns
    -------
    np.ndarray
        ``(N, N)`` symmetric scores with a zero diagonal.
    """
    num_views = len(centers)
    scores = np.zeros((num_views, num_views))
    for point, track in zip(points, tracks):
        track = np.unique(track)
        if len(track) < 2:
            continue
        rays = centers[track] - point
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        theta = np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
        sigma = np.where(theta <= theta0, sigma1, sigma2)
        weight = np.exp(-((theta - theta0) ** 2) / (2 * sigma**2))
        np.fill_diagonal(weight, 0.0)
        scores[np.ix_(track, track)] += weight
    return scores


def top_source_views(scores: np.ndarray, num_src: int) -> list[list[int]]:
    """Best-scoring source views per view, excluding zero scores.

    Parameters
    ----------
    scores : np.ndarray
        ``(N, N)`` output of ``view_selection_scores``.
    num_src : int
        Maximum source views per view.

    Returns
    -------
    list[list[int]]
        Source-view indices per view, best first.
    """
    selected = []
    for row in scores:
        order = np.argsort(-row, kind="stable")[:num_src]
        selected.append([int(j) for j in order if row[j] > 0])
    return selected


def depth_range(
    depths: np.ndarray,
    percentiles: tuple[float, float],
    margins: tuple[float, float],
) -> tuple[float, float]:
    """Robust depth range of a view's sparse points, widened by margins.

    Parameters
    ----------
    depths : np.ndarray
        Camera-frame z of the points the view observes (positive).
    percentiles : tuple[float, float]
        Low and high percentiles, dropping outlier points.
    margins : tuple[float, float]
        Multipliers for the low and high ends (e.g. 0.75 and 1.25), since
        sparse points under-cover the dense surface.

    Returns
    -------
    tuple[float, float]
        ``(depth_min, depth_max)``.
    """
    low, high = np.percentile(depths, percentiles)
    return float(low * margins[0]), float(high * margins[1])


def build_views(reconstruction: pycolmap.Reconstruction, cfg: dict) -> list[View]:
    """Views with intrinsics, pose, depth range and source views.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        Undistorted (pinhole) sparse model of the MVS workspace.
    cfg : dict
        ``views`` section of ``configs/transmvsnet.yaml``.

    Returns
    -------
    list[View]
        Views with a usable depth range, in image-id order. Views with fewer
        than ``min_points`` observed points are dropped (logged): no depth
        range can be estimated for them.

    Raises
    ------
    ValueError
        If a camera is not pinhole (the workspace was not undistorted).
    """
    images = sorted(
        (im for im in reconstruction.images.values() if im.has_pose),
        key=lambda im: im.image_id,
    )
    index = {im.image_id: i for i, im in enumerate(images)}
    extrinsics = np.stack([_extrinsic(im) for im in images])
    rotations, translations = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    centers = -np.einsum("nji,nj->ni", rotations, translations)

    point_ids = list(reconstruction.points3D)
    points = np.array([reconstruction.points3D[p].xyz for p in point_ids])
    tracks = [
        np.array(
            [
                index[el.image_id]
                for el in reconstruction.points3D[p].track.elements
                if el.image_id in index
            ],
            dtype=int,
        )
        for p in point_ids
    ]

    scores = view_selection_scores(
        centers, points, tracks, cfg["theta0"], cfg["sigma1"], cfg["sigma2"]
    )
    sources = top_source_views(scores, int(cfg["num_src"]))

    observed: list[list[int]] = [[] for _ in images]
    for point_idx, track in enumerate(tracks):
        for view_idx in np.unique(track):
            observed[view_idx].append(point_idx)

    views = []
    for i, image in enumerate(images):
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name not in _PINHOLE_MODELS:
            raise ValueError(
                f"Camera of '{image.name}' is {camera.model.name}; TransMVSNet "
                "needs the undistorted (pinhole) workspace."
            )
        if len(observed[i]) < int(cfg["min_points"]):
            logger.warning(
                "Skipping '%s': %d sparse points, need %d for a depth range.",
                image.name,
                len(observed[i]),
                cfg["min_points"],
            )
            continue
        cam_points = points[observed[i]] @ rotations[i].T + translations[i]
        depths = cam_points[:, 2][cam_points[:, 2] > 0]
        depth_min, depth_max = depth_range(
            depths, tuple(cfg["depth_percentiles"]), tuple(cfg["depth_margins"])
        )
        K = np.array(camera.calibration_matrix(), dtype=np.float64)
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        views.append(
            View(
                image_id=image.image_id,
                name=image.name,
                K=K,
                extrinsic=extrinsics[i],
                width=int(camera.width),
                height=int(camera.height),
                depth_min=depth_min,
                depth_max=depth_max,
                src_ids=tuple(images[j].image_id for j in sources[i]),
            )
        )

    # Dropped views cannot serve as sources either.
    kept = {v.image_id for v in views}
    views = [
        View(**{**v.__dict__, "src_ids": tuple(s for s in v.src_ids if s in kept)})
        for v in views
    ]
    logger.info(
        "TransMVSNet views: %d of %d registered images, median %d source views",
        len(views),
        len(images),
        int(np.median([len(v.src_ids) for v in views])) if views else 0,
    )
    return views


def _extrinsic(image: pycolmap.Image) -> np.ndarray:
    extrinsic = np.eye(4)
    extrinsic[:3, :4] = image.cam_from_world().matrix()
    return extrinsic
