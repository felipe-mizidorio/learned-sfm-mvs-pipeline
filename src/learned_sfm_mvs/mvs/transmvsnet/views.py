"""Per-view inputs for TransMVSNet from an undistorted COLMAP workspace.

Replaces upstream's colmap2mvsnet.py export: instead of writing MVSNet
``cams/*.txt`` and ``pair.txt`` files, the same quantities are computed in
memory from the sparse model that ``pycolmap.undistort_images`` writes.

Pixel convention: COLMAP places pixel (0, 0)'s centre at (0.5, 0.5); MVSNet
code (``homo_warping``, the fusion reprojection) puts it at (0, 0). ``View.K``
is therefore COLMAP's calibration shifted by -0.5 px. Upstream skips this
shift, which offsets every back-projected point by half a pixel.

Depth ranges: with subject masks, each view searches depth only around the
subject's sparse points. Without them, a range spanning the background (e.g.
SfM run on whole frames with ``--no-feature-masks``) spreads the fixed number
of depth hypotheses over the room, leaving few on the subject.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

_PINHOLE_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})
# Masks covering more of the frame are segmentation fallbacks, not a subject.
_FULL_FRAME_FRACTION = 0.99


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
        Depth search range, from sparse points (see ``depth_range_source``).
    src_ids : tuple[int, ...]
        Source-view image ids, best first.
    depth_range_source : str, optional
        Points the range came from: ``subject_observed`` (subject points
        this view observes), ``subject_projected`` (all subject points that
        project into its mask) or ``all_points`` (every observed point).
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
    depth_range_source: str = "all_points"


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


def load_subject_mask(mask_dir: Path, name: str) -> np.ndarray | None:
    """Workspace mask of one view as bool, or None if missing or full-frame.

    Parameters
    ----------
    mask_dir : Path
        Masks aligned to the undistorted images (``<image name>.png``).
    name : str
        Image name.

    Returns
    -------
    np.ndarray or None
        ``(H, W)`` bool; None when there is no usable subject mask.
    """
    mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    subject = mask > 0
    return None if subject.mean() > _FULL_FRAME_FRACTION else subject


def inside_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Whether COLMAP pixel coordinates fall inside a mask.

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` bool.
    xy : np.ndarray
        ``(N, 2)`` coordinates, pixel (0, 0) centred at (0.5, 0.5).

    Returns
    -------
    np.ndarray
        ``(N,)`` bool; False outside the image.
    """
    cols = np.floor(xy[:, 0]).astype(int)
    rows = np.floor(xy[:, 1]).astype(int)
    height, width = mask.shape
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    result = np.zeros(len(xy), dtype=bool)
    result[valid] = mask[rows[valid], cols[valid]]
    return result


def subject_point_ids(images: list[pycolmap.Image], mask_dir: Path) -> set[int]:
    """Sparse points inside the subject mask in most views observing them.

    A point is on the subject when at least half of its observations in views
    with a usable mask fall inside that mask. Views without one do not vote.

    Parameters
    ----------
    images : list[pycolmap.Image]
        Registered images of the undistorted model, to take votes from.
    mask_dir : Path
        Masks aligned to the undistorted images.

    Returns
    -------
    set[int]
        ``point3D_id`` of the subject points.
    """
    seen: Counter[int] = Counter()
    inside: Counter[int] = Counter()
    for image in images:
        mask = load_subject_mask(mask_dir, image.name)
        if mask is None:
            continue
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            continue
        hits = inside_mask(mask, np.array([p.xy for p in observations]))
        for p, hit in zip(observations, hits):
            seen[p.point3D_id] += 1
            inside[p.point3D_id] += int(hit)
    return {pid for pid, n in seen.items() if 2 * inside[pid] >= n}


def build_views(
    reconstruction: pycolmap.Reconstruction,
    cfg: dict,
    mask_dir: Path | None = None,
) -> list[View]:
    """Views with intrinsics, pose, depth range and source views.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        Undistorted (pinhole) sparse model of the MVS workspace.
    cfg : dict
        ``views`` section of ``configs/transmvsnet.yaml``.
    mask_dir : Path or None, optional
        Subject masks aligned to the undistorted images. When given, each
        view's depth range comes from the subject's sparse points: those it
        observes if at least ``min_points``, else all subject points that
        project into its mask, else (no usable mask or too few subject
        points) every point it observes, as without masks. Source-view
        selection always uses all points.

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

    is_subject = None
    if mask_dir is not None:
        subject_ids = subject_point_ids(images, mask_dir)
        is_subject = np.array([p in subject_ids for p in point_ids], dtype=bool)
        logger.info(
            "Subject sparse points from masks: %d of %d",
            int(is_subject.sum()),
            len(point_ids),
        )
    min_points = int(cfg["min_points"])

    views = []
    for i, image in enumerate(images):
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name not in _PINHOLE_MODELS:
            raise ValueError(
                f"Camera of '{image.name}' is {camera.model.name}; TransMVSNet "
                "needs the undistorted (pinhole) workspace."
            )
        if len(observed[i]) < min_points:
            logger.warning(
                "Skipping '%s': %d sparse points, need %d for a depth range.",
                image.name,
                len(observed[i]),
                min_points,
            )
            continue
        cam_points = points[observed[i]] @ rotations[i].T + translations[i]
        depths, source = cam_points[:, 2], "all_points"
        if is_subject is not None and mask_dir is not None:
            on_subject = is_subject[observed[i]] & (cam_points[:, 2] > 0)
            if np.count_nonzero(on_subject) >= min_points:
                depths, source = cam_points[on_subject, 2], "subject_observed"
            else:
                projected = _projected_depths(
                    points[is_subject],
                    rotations[i],
                    translations[i],
                    camera,
                    load_subject_mask(mask_dir, image.name),
                )
                if len(projected) >= min_points:
                    depths, source = projected, "subject_projected"
        depths = depths[depths > 0]
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
                depth_range_source=source,
            )
        )

    # Dropped views cannot serve as sources either.
    kept = {v.image_id for v in views}
    views = [
        View(**{**v.__dict__, "src_ids": tuple(s for s in v.src_ids if s in kept)})
        for v in views
    ]
    logger.info(
        "TransMVSNet views: %d of %d registered images, median %d source views; "
        "depth ranges from %s",
        len(views),
        len(images),
        int(np.median([len(v.src_ids) for v in views])) if views else 0,
        dict(Counter(v.depth_range_source for v in views)),
    )
    return views


def _projected_depths(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera: pycolmap.Camera,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Depths of points in front of a camera that project into its mask
    (or into the image, without a usable mask)."""
    cam = points @ rotation.T + translation
    cam = cam[cam[:, 2] > 0]
    uvw = cam @ np.array(camera.calibration_matrix()).T
    xy = uvw[:, :2] / uvw[:, 2:]
    if mask is None:
        mask = np.ones((camera.height, camera.width), dtype=bool)
    return cam[inside_mask(mask, xy), 2]


def _extrinsic(image: pycolmap.Image) -> np.ndarray:
    extrinsic = np.eye(4)
    extrinsic[:3, :4] = image.cam_from_world().matrix()
    return extrinsic
