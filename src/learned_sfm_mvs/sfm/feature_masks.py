"""Apply COLMAP-style image masks to hloc local features.

hloc has no mask support, so keypoints are filtered after extraction and
before matching: matches then only reference kept keypoints, and the COLMAP
database import sees the filtered set. Masks follow the COLMAP convention used
throughout the pipeline: ``<mask_dir>/<image name>.png``, 0 discards, non-zero
keeps. A missing mask keeps every keypoint, like COLMAP does.

Trade-off: the extractor's keypoint budget is spent before masking, so masked
background still costs detections. Blacking out the image instead would put
artificial corners on the mask border.
"""

import logging
from pathlib import Path

import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)

# hloc stores descriptors as (D, N); every other per-keypoint array is (N, ...).
_KEYPOINT_LAST_AXIS = frozenset({"descriptors"})
_NOT_PER_KEYPOINT = frozenset({"image_size"})


def filter_features_by_mask(
    features_path: Path,
    image_names: list[str],
    mask_dir: Path,
) -> dict:
    """Drop keypoints that fall on masked-out pixels, in place.

    Parameters
    ----------
    features_path : Path
        hloc local-features HDF5 file (one group per image name).
    image_names : list[str]
        Images to filter.
    mask_dir : Path
        Directory of COLMAP masks (``<image name>.png``).

    Returns
    -------
    dict
        ``masks_applied``, ``masks_missing``, ``keypoints_before`` and
        ``keypoints_after``, for the manifest.

    Raises
    ------
    ValueError
        If a mask's size differs from its image's size.
    """
    stats = {
        "masks_applied": 0,
        "masks_missing": 0,
        "keypoints_before": 0,
        "keypoints_after": 0,
    }
    with h5py.File(features_path, "r+") as fd:
        for name in image_names:
            grp = fd[name]
            assert isinstance(grp, h5py.Group)
            keypoints = np.asarray(grp["keypoints"], dtype=np.float32)
            stats["keypoints_before"] += len(keypoints)

            mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                stats["masks_missing"] += 1
                stats["keypoints_after"] += len(keypoints)
                continue

            width, height = (int(v) for v in np.asarray(grp["image_size"]))
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask for '{name}' is {mask.shape[1]}x{mask.shape[0]}, "
                    f"image is {width}x{height}"
                )

            keep = _keypoints_inside(keypoints, mask)
            _select_keypoints(grp, keep, len(keypoints))
            stats["masks_applied"] += 1
            stats["keypoints_after"] += int(keep.sum())

    logger.info(
        "Feature masks: %d applied, %d missing (unmasked); keypoints %d -> %d",
        stats["masks_applied"],
        stats["masks_missing"],
        stats["keypoints_before"],
        stats["keypoints_after"],
    )
    if stats["masks_missing"]:
        logger.warning(
            "%d of %d images have no mask and keep all keypoints.",
            stats["masks_missing"],
            len(image_names),
        )
    return stats


def _keypoints_inside(keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean mask of keypoints on non-zero mask pixels.

    hloc keypoints use pixel-centre coordinates (pixel ``(0, 0)`` spans
    ``[-0.5, 0.5)``), so rounding gives the pixel index.
    """
    height, width = mask.shape
    cols = np.clip(np.rint(keypoints[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.rint(keypoints[:, 1]).astype(int), 0, height - 1)
    return mask[rows, cols] > 0


def _select_keypoints(grp: h5py.Group, keep: np.ndarray, num_keypoints: int) -> None:
    """Rewrite every per-keypoint dataset of ``grp`` with only ``keep`` rows."""
    for key in list(grp.keys()):
        if key in _NOT_PER_KEYPOINT:
            continue
        dataset = grp[key]
        assert isinstance(dataset, h5py.Dataset)
        data = dataset[()]
        attrs = dict(dataset.attrs)
        if key in _KEYPOINT_LAST_AXIS:
            data = data[..., keep]
        elif data.ndim >= 1 and data.shape[0] == num_keypoints:
            data = data[keep]
        else:
            continue
        del grp[key]
        new = grp.create_dataset(key, data=data)
        new.attrs.update(attrs)
