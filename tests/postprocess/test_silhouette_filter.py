"""Markerless head crop: dense points kept by multi-view mask (silhouette) votes."""

import cv2
import numpy as np
import pycolmap
import pytest

from learned_sfm_mvs.postprocess.silhouette_filter import silhouette_keep


def _scene(tmp_path, masks="silhouette"):
    """Six distorted cameras around a unit-sphere 'head' with background clutter.

    Masks are the head's silhouette in each original (distorted) frame, as the
    frames manifest provides them; ``masks="full"`` writes segmentation
    fallbacks (all white) instead.
    """
    pycolmap.set_random_seed(0)
    options = pycolmap.SyntheticDatasetOptions()
    options.camera_model_id = pycolmap.CameraModelId.SIMPLE_RADIAL
    options.camera_params = [800.0, 320.0, 240.0, 0.02]
    options.camera_width, options.camera_height = 640, 480
    options.num_rigs, options.num_frames_per_rig = 1, 6
    options.num_points3D = 50
    reconstruction = pycolmap.synthesize_dataset(options)

    rng = np.random.default_rng(0)
    head = rng.normal(size=(3000, 3))
    head /= np.linalg.norm(head, axis=1, keepdims=True)
    clutter = rng.normal(size=(3000, 3))
    clutter *= rng.uniform(1.6, 3.0, size=(3000, 1)) / np.linalg.norm(
        clutter, axis=1, keepdims=True
    )

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        mask = np.zeros((camera.height, camera.width), np.uint8)
        if masks == "full":
            mask[:] = 255
        else:
            pose = image.cam_from_world().matrix()
            xy = camera.img_from_cam(head @ pose[:, :3].T + pose[:, 3])
            cv2.fillConvexPoly(mask, cv2.convexHull(xy.astype(np.int32)), 255)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        cv2.imwrite(str(mask_dir / f"{image.name}.png"), mask)
    return reconstruction, head, clutter, mask_dir


def test_keeps_the_head_and_drops_background(tmp_path):
    reconstruction, head, clutter, mask_dir = _scene(tmp_path)
    points = np.concatenate([head, clutter])

    keep, stats = silhouette_keep(points, reconstruction, mask_dir, 3, 0.8)

    assert keep is not None
    assert keep[: len(head)].mean() > 0.99
    assert keep[len(head) :].mean() < 0.05
    assert stats == {
        "method": "silhouette",
        "views_used": 6,
        "min_views": 3,
        "min_inside_fraction": 0.8,
        "points_before": len(points),
        "points_after": int(keep.sum()),
    }


def test_points_seen_by_too_few_masked_views_are_dropped(tmp_path):
    reconstruction, head, _, mask_dir = _scene(tmp_path)
    keep, _ = silhouette_keep(head, reconstruction, mask_dir, 7, 0.8)  # only 6 views
    assert not keep.any()


def test_full_frame_fallback_masks_do_not_vote(tmp_path):
    reconstruction, head, clutter, mask_dir = _scene(tmp_path, masks="full")
    keep, stats = silhouette_keep(
        np.concatenate([head, clutter]), reconstruction, mask_dir, 3, 0.8
    )
    assert keep is None
    assert stats["views_used"] == 0


def test_missing_mask_directory_entries_do_not_vote(tmp_path):
    reconstruction, head, clutter, mask_dir = _scene(tmp_path)
    names = sorted(im.name for im in reconstruction.images.values())
    for name in names[:3]:
        (mask_dir / f"{name}.png").unlink()

    keep, stats = silhouette_keep(
        np.concatenate([head, clutter]), reconstruction, mask_dir, 3, 0.8
    )

    assert stats["views_used"] == 3
    assert keep[: len(head)].mean() > 0.99


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_rejects_invalid_fraction(tmp_path, fraction):
    reconstruction, head, _, mask_dir = _scene(tmp_path)
    with pytest.raises(ValueError, match="min_inside_fraction"):
        silhouette_keep(head, reconstruction, mask_dir, 3, fraction)
