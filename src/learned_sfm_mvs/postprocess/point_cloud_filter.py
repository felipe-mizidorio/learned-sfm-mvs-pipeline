"""Statistical outlier removal for dense point clouds."""

import logging
from pathlib import Path

import open3d as o3d

logger = logging.getLogger(__name__)


def filter_point_cloud(
    input_ply: Path,
    output_ply: Path,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    """Statistical outlier removal (SOR).

    Removes points whose mean distance to their ``nb_neighbors`` nearest
    neighbours exceeds the global mean by more than ``std_ratio`` standard
    deviations.

    Parameters
    ----------
    input_ply : Path
        Input point cloud.
    output_ply : Path
        Where the filtered cloud is written.
    nb_neighbors : int
        Neighbourhood size for the mean-distance computation.
    std_ratio : float
        Rejection threshold in standard deviations.

    Returns
    -------
    o3d.geometry.PointCloud
        The filtered cloud (also written to ``output_ply``).

    Raises
    ------
    FileNotFoundError
        If ``input_ply`` does not exist.
    ValueError
        If the input cloud is empty.
    """
    if not input_ply.exists():
        raise FileNotFoundError(f"Input point cloud not found: {input_ply}")

    pcd = o3d.io.read_point_cloud(str(input_ply))
    n_before = len(pcd.points)
    if n_before == 0:
        raise ValueError(f"Point cloud '{input_ply}' contains no points.")

    logger.info(
        "SOR: %d points, nb_neighbors=%d, std_ratio=%.2f",
        n_before,
        nb_neighbors,
        std_ratio,
    )

    pcd_filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    n_after = len(pcd_filtered.points)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), pcd_filtered)

    logger.info(
        "SOR complete: %d → %d points (%d removed, %.1f%%)",
        n_before,
        n_after,
        n_before - n_after,
        100.0 * (n_before - n_after) / n_before,
    )
    return pcd_filtered
