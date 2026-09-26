import math

import cv2
import numpy as np
import pycolmap
import pytest

from learned_sfm_mvs.mvs.transmvsnet.views import (
    build_views,
    depth_range,
    top_source_views,
    view_selection_scores,
)

THETA0, SIGMA1, SIGMA2 = 5.0, 1.0, 10.0
CFG = {
    "num_src": 3,
    "theta0": THETA0,
    "sigma1": SIGMA1,
    "sigma2": SIGMA2,
    "min_points": 10,
    "depth_percentiles": [1, 99],
    "depth_margins": [0.75, 1.25],
}


def _camera_at_angle(point: np.ndarray, degrees: float) -> np.ndarray:
    """Camera centre on the x axis whose ray to ``point`` makes ``degrees``
    with the ray from the origin."""
    return np.array([point[2] * math.tan(math.radians(degrees)), 0.0, 0.0])


def test_view_selection_scores_follow_the_angle_heuristic():
    point = np.array([0.0, 0.0, 10.0])
    centers = np.stack(
        [np.zeros(3), _camera_at_angle(point, 5.0), _camera_at_angle(point, 15.0)]
    )

    scores = view_selection_scores(
        centers, point[None], [np.array([0, 1, 2])], THETA0, SIGMA1, SIGMA2
    )

    # theta == theta0 is the ideal baseline.
    assert scores[0, 1] == pytest.approx(1.0)
    # 15 deg: wide side, soft sigma2.
    assert scores[0, 2] == pytest.approx(math.exp(-(10.0**2) / (2 * SIGMA2**2)))
    # 10 deg between cameras 1 and 2 (both rays in the same plane).
    assert scores[1, 2] == pytest.approx(math.exp(-(5.0**2) / (2 * SIGMA2**2)))
    np.testing.assert_allclose(scores, scores.T)
    assert np.all(np.diag(scores) == 0)


def test_narrow_baseline_scores_lower_than_equally_wide_one():
    point = np.array([0.0, 0.0, 10.0])
    centers = np.stack(
        [np.zeros(3), _camera_at_angle(point, 3.0), _camera_at_angle(point, 7.0)]
    )
    scores = view_selection_scores(
        centers, point[None], [np.array([0, 1, 2])], THETA0, SIGMA1, SIGMA2
    )
    # Both 2 deg from theta0, but sigma1 < sigma2 penalizes the narrow side.
    assert scores[0, 1] < scores[0, 2]


def test_single_view_tracks_and_duplicates_are_ignored():
    scores = view_selection_scores(
        np.eye(3), np.zeros((2, 3)), [np.array([1]), np.array([2, 2])], 5, 1, 10
    )
    assert np.all(scores == 0)


def test_top_source_views_best_first_without_zero_scores():
    scores = np.array(
        [
            [0.0, 0.2, 0.9, 0.0],
            [0.2, 0.0, 0.1, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert top_source_views(scores, 2) == [[2, 1], [0, 2], [0, 1], []]


def test_depth_range_percentiles_and_margins():
    depths = np.arange(1.0, 102.0)  # 1..101
    assert depth_range(depths, (1, 99), (0.75, 1.25)) == pytest.approx(
        (2.0 * 0.75, 100.0 * 1.25)
    )


# --- build_views on a synthesized reconstruction ---


def _synthetic(model=pycolmap.CameraModelId.PINHOLE, params=None, num_points=200):
    options = pycolmap.SyntheticDatasetOptions()
    options.camera_model_id = model
    options.camera_params = params or [800.0, 810.0, 320.5, 240.25]
    options.camera_width, options.camera_height = 640, 480
    options.num_rigs, options.num_frames_per_rig = 1, 6
    options.num_points3D = num_points
    return pycolmap.synthesize_dataset(options)


def test_build_views_matches_colmap_projection_with_half_pixel_shift():
    reconstruction = _synthetic()

    views = build_views(reconstruction, CFG)

    assert len(views) == 6
    for view in views:
        image = reconstruction.images[view.image_id]
        observed = [p for p in image.points2D if p.has_point3D()]
        xyz = np.array([reconstruction.points3D[p.point3D_id].xyz for p in observed])
        cam = xyz @ view.extrinsic[:3, :3].T + view.extrinsic[:3, 3]
        pix = cam @ view.K.T
        pix = pix[:, :2] / pix[:, 2:]
        # View.K puts pixel centres at integers; COLMAP at +0.5.
        np.testing.assert_allclose(
            pix + 0.5, np.array([p.xy for p in observed]), atol=1e-6
        )
        assert view.depth_min <= cam[:, 2].min()
        assert view.depth_max >= cam[:, 2].max()
        assert (view.width, view.height) == (640, 480)


def test_build_views_source_views_exclude_self_and_are_capped():
    views = build_views(_synthetic(), CFG)
    ids = {v.image_id for v in views}
    for view in views:
        assert view.image_id not in view.src_ids
        assert set(view.src_ids) <= ids
        assert 0 < len(view.src_ids) <= CFG["num_src"]


def test_build_views_rejects_distorted_workspace():
    reconstruction = _synthetic(
        pycolmap.CameraModelId.SIMPLE_RADIAL, [800.0, 320.0, 240.0, 0.05]
    )
    with pytest.raises(ValueError, match="undistorted"):
        build_views(reconstruction, CFG)


def test_build_views_drops_views_without_enough_points():
    reconstruction = _synthetic(num_points=5)
    assert build_views(reconstruction, CFG) == []


# --- depth ranges from subject masks ---

# Margins off, so the tests see which points set the range.
CFG_NO_MARGINS = {**CFG, "depth_margins": [1.0, 1.0]}


def _subject_scene(tmp_path):
    """Points on the unit sphere with a small cap as the 'subject', and per-view
    masks of the visible cap's silhouette (as warped workspace masks)."""
    pycolmap.set_random_seed(0)
    reconstruction = _synthetic(num_points=2000)
    # The synthetic views see through the sphere; drop what a real one hides
    # (points facing away from the camera), or far-side points would fall
    # inside the subject's silhouette.
    for image_id, image in list(reconstruction.images.items()):
        centre = image.projection_center()
        hidden = [
            idx
            for idx, p in enumerate(image.points2D)
            if p.has_point3D()
            and np.dot(
                reconstruction.points3D[p.point3D_id].xyz,
                centre - reconstruction.points3D[p.point3D_id].xyz,
            )
            <= 0
        ]
        for idx in hidden:
            reconstruction.delete_observation(image_id, idx)
    subject = {
        pid for pid, point in reconstruction.points3D.items() if point.xyz[0] > 0.9
    }
    mask_dir = tmp_path / "view_masks"
    mask_dir.mkdir()
    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        mask = np.zeros((camera.height, camera.width), np.uint8)
        xy = np.array([p.xy for p in _observed(image, subject)]).reshape(-1, 2)
        if len(xy) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(xy.astype(np.int32)), 255)
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        cv2.imwrite(str(mask_dir / f"{image.name}.png"), mask)
    return reconstruction, subject, mask_dir


def _observed(image, point_ids):
    return [p for p in image.points2D if p.has_point3D() and p.point3D_id in point_ids]


def _depths(reconstruction, view, point_ids):
    xyz = np.array([reconstruction.points3D[p].xyz for p in point_ids])
    return (xyz @ view.extrinsic[:3, :3].T + view.extrinsic[:3, 3])[:, 2]


def _assert_range_fits(view, depths):
    assert view.depth_min == pytest.approx(np.percentile(depths, 1), rel=0.05)
    assert view.depth_max == pytest.approx(np.percentile(depths, 99), rel=0.05)


def test_masks_restrict_depth_range_to_the_subject(tmp_path):
    reconstruction, subject, mask_dir = _subject_scene(tmp_path)
    unmasked = {v.image_id: v for v in build_views(reconstruction, CFG_NO_MARGINS)}

    views = build_views(reconstruction, CFG_NO_MARGINS, mask_dir=mask_dir)

    assert len(views) == 6
    seeing = 0
    for view in views:
        visible = [
            p.point3D_id
            for p in _observed(reconstruction.images[view.image_id], subject)
        ]
        before = unmasked[view.image_id]
        assert before.depth_range_source == "all_points"
        if len(visible) < CFG["min_points"]:
            assert view.depth_range_source in {"subject_projected", "all_points"}
            continue
        seeing += 1
        assert view.depth_range_source == "subject_observed"
        _assert_range_fits(view, _depths(reconstruction, view, visible))
        assert view.depth_max - view.depth_min < before.depth_max - before.depth_min
    assert seeing >= 2


def test_view_seeing_few_subject_points_uses_projected_subject_points(tmp_path):
    reconstruction, subject, mask_dir = _subject_scene(tmp_path)
    image_id = max(
        reconstruction.images,
        key=lambda i: len(_observed(reconstruction.images[i], subject)),
    )
    image = reconstruction.images[image_id]
    visible = [p.point3D_id for p in _observed(image, subject)]
    on_subject = [
        idx
        for idx, p in enumerate(image.points2D)
        if p.has_point3D() and p.point3D_id in subject
    ]
    for idx in on_subject[3:]:  # below min_points, but other views still see them
        reconstruction.delete_observation(image_id, idx)

    views = {
        v.image_id: v
        for v in build_views(reconstruction, CFG_NO_MARGINS, mask_dir=mask_dir)
    }

    view = views[image_id]
    assert view.depth_range_source == "subject_projected"
    # COLMAP drops points left with a single observation.
    surviving = [p for p in visible if p in reconstruction.points3D]
    _assert_range_fits(view, _depths(reconstruction, view, surviving))


def test_full_frame_or_missing_masks_keep_all_point_ranges(tmp_path):
    reconstruction = _synthetic()
    mask_dir = tmp_path / "view_masks"
    mask_dir.mkdir()
    names = sorted(im.name for im in reconstruction.images.values())
    for name in names[:3]:  # segmentation fallbacks; the other views have no mask
        cv2.imwrite(str(mask_dir / f"{name}.png"), np.full((480, 640), 255, np.uint8))

    expected = build_views(reconstruction, CFG)
    views = build_views(reconstruction, CFG, mask_dir=mask_dir)

    assert [(v.depth_min, v.depth_max) for v in views] == [
        (e.depth_min, e.depth_max) for e in expected
    ]
    assert {v.depth_range_source for v in views} == {"all_points"}
