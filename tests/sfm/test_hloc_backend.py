"""hloc backend wiring, with hloc replaced by fakes (no torch needed)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from learned_sfm_mvs.sfm import hloc_backend
from learned_sfm_mvs.sfm.base import SfmInputs, SfmResult
from learned_sfm_mvs.sfm.pairs import read_pairs, write_pairs

NAMES = ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"]
FEATURE_CONFS = {
    "aliked-n16": {
        "output": "feats-aliked-n16",
        "model": {"name": "aliked", "max_num_keypoints": -1},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {"output": "global-feats-netvlad", "model": {"name": "netvlad"}},
}
MATCHER_CONFS = {"aliked+lightglue": {"model": {"name": "lightglue"}}}
HLOC_CFG = {
    "features": "aliked-n16",
    "feature_overrides": {"model": {"max_num_keypoints": 4096}},
    "matcher": "aliked+lightglue",
    "pairs": {
        "strategies": ["sequential"],
        "sequential_overlap": 1,
        "retrieval_model": "netvlad",
        "retrieval_num_matched": 20,
    },
}


def _fake_hloc(retrieval_pairs=()) -> SimpleNamespace:
    def _write_retrieval(descriptors, output, **kwargs):
        write_pairs(retrieval_pairs, output)

    return SimpleNamespace(
        cuda_available=False,
        extract_features=SimpleNamespace(confs=FEATURE_CONFS, main=MagicMock()),
        match_features=SimpleNamespace(confs=MATCHER_CONFS, main=MagicMock()),
        pairs_from_retrieval=SimpleNamespace(
            main=MagicMock(side_effect=_write_retrieval)
        ),
        create_empty_db=MagicMock(),
        get_image_ids=MagicMock(return_value={n: i for i, n in enumerate(NAMES)}),
        import_features=MagicMock(),
        import_matches=MagicMock(),
        estimation_and_geometric_verification=MagicMock(),
    )


# --- build_conf ---


def test_build_conf_deep_merges_without_mutating_base():
    conf = hloc_backend.build_conf(
        FEATURE_CONFS, "aliked-n16", {"model": {"max_num_keypoints": 4096}}, "feature"
    )
    assert conf["model"] == {"name": "aliked", "max_num_keypoints": 4096}
    assert conf["preprocessing"] == {"resize_max": 1024}
    assert FEATURE_CONFS["aliked-n16"]["model"]["max_num_keypoints"] == -1


def test_build_conf_unknown_name_lists_available():
    with pytest.raises(
        ValueError, match="Unknown hloc feature 'superpoint_max'.*aliked"
    ):
        hloc_backend.build_conf(FEATURE_CONFS, "superpoint_max", None, "feature")


# --- select_pairs ---


def test_select_pairs_unions_sequential_and_retrieval(tmp_path):
    hloc = _fake_hloc(retrieval_pairs=[("f4.jpg", "f1.jpg"), ("f2.jpg", "f1.jpg")])
    cfg = {**HLOC_CFG["pairs"], "strategies": ["sequential", "retrieval"]}

    pairs = hloc_backend.select_pairs(hloc, NAMES, tmp_path, tmp_path, cfg)

    # Sequential (overlap 1) first; retrieval adds only the new long-range pair.
    assert pairs == [
        ("f1.jpg", "f2.jpg"),
        ("f2.jpg", "f3.jpg"),
        ("f3.jpg", "f4.jpg"),
        ("f4.jpg", "f1.jpg"),
    ]
    kwargs = hloc.pairs_from_retrieval.main.call_args.kwargs
    # Capped at len(NAMES) - 1: an image cannot retrieve more neighbours.
    assert kwargs["num_matched"] == 3
    assert kwargs["query_list"] == NAMES


@pytest.mark.parametrize("strategies", [[], ["sequential", "vocab_tree"]])
def test_select_pairs_rejects_bad_strategies(tmp_path, strategies):
    cfg = {**HLOC_CFG["pairs"], "strategies": strategies}
    with pytest.raises(ValueError, match="pairs.strategies"):
        hloc_backend.select_pairs(_fake_hloc(), NAMES, tmp_path, tmp_path, cfg)


# --- run ---


def _inputs(tmp_path, **kwargs) -> SfmInputs:
    return SfmInputs(
        image_dir=tmp_path / "images",
        output_dir=tmp_path / "out",
        image_names=NAMES,
        **kwargs,
    )


def _patched_run(tmp_path, hloc, inputs):
    mapped = SfmResult(MagicMock(), inputs.sparse_dir / "0", 1)
    with (
        patch.object(hloc_backend, "_load_hloc", return_value=hloc),
        patch.object(hloc_backend, "_import_images") as mock_import_images,
        patch.object(hloc_backend, "check_all_images_imported"),
        patch.object(hloc_backend, "filter_features_by_mask") as mock_masks,
        patch.object(hloc_backend, "map_and_select", return_value=mapped) as mock_map,
        patch.object(hloc_backend.pycolmap.Database, "open"),
    ):
        mock_masks.return_value = {"masks_applied": 4}
        result = hloc_backend.run(inputs, HLOC_CFG, {"min_num_matches": 15})
    return result, mock_import_images, mock_masks, mock_map


def test_run_wires_hloc_into_colmap_mapping(tmp_path):
    hloc = _fake_hloc()
    inputs = _inputs(tmp_path)
    stale = inputs.output_dir / "hloc" / "features.h5"
    stale.parent.mkdir(parents=True)
    stale.touch()

    result, mock_import_images, mock_masks, mock_map = _patched_run(
        tmp_path, hloc, inputs
    )

    work_dir = inputs.output_dir / "hloc"
    # Stale hloc outputs are cleared so features are never silently reused.
    assert not stale.exists()
    feature_conf = hloc.extract_features.main.call_args.args[0]
    assert feature_conf["model"]["max_num_keypoints"] == 4096
    assert hloc.extract_features.main.call_args.kwargs["image_list"] == NAMES
    mock_masks.assert_not_called()
    assert read_pairs(work_dir / "pairs.txt") == [
        ("f1.jpg", "f2.jpg"),
        ("f2.jpg", "f3.jpg"),
        ("f3.jpg", "f4.jpg"),
    ]
    hloc.match_features.main.assert_called_once()
    hloc.create_empty_db.assert_called_once_with(inputs.database_path)
    mock_import_images.assert_called_once_with(inputs)
    hloc.estimation_and_geometric_verification.assert_called_once_with(
        inputs.database_path, work_dir / "pairs.txt"
    )
    mock_map.assert_called_once_with(inputs, {"min_num_matches": 15})
    assert result.stats["num_pairs"] == 3
    assert result.stats["feature_masks"] is None


def test_run_filters_features_by_mask_before_matching(tmp_path):
    hloc = _fake_hloc()
    inputs = _inputs(tmp_path, mask_dir=tmp_path / "masks")

    result, _, mock_masks, _ = _patched_run(tmp_path, hloc, inputs)

    mock_masks.assert_called_once_with(
        inputs.output_dir / "hloc" / "features.h5", NAMES, tmp_path / "masks"
    )
    assert result.stats["feature_masks"] == {"masks_applied": 4}


def test_run_rejects_unknown_feature_before_touching_outputs(tmp_path):
    inputs = _inputs(tmp_path)
    inputs.output_dir.mkdir()
    inputs.database_path.touch()
    cfg = {**HLOC_CFG, "features": "superpoint_max"}

    with patch.object(hloc_backend, "_load_hloc", return_value=_fake_hloc()):
        with pytest.raises(ValueError, match="superpoint_max"):
            hloc_backend.run(inputs, cfg, {})
    # A config typo must not delete the previous run's database.
    assert inputs.database_path.exists()


def _shipped_cfg() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "hloc.yaml"
    return yaml.safe_load(path.read_text())["hloc"]


def test_shipped_hloc_config_is_valid():
    cfg = _shipped_cfg()
    assert set(cfg["pairs"]["strategies"]) <= set(hloc_backend.PAIR_STRATEGIES)
    assert cfg["features"] == "aliked-n16"
    assert cfg["matcher"] == "aliked+lightglue"


def test_shipped_hloc_config_names_exist_in_hloc():
    """Catches typos and renames in pinned hloc confs (needs the learned group)."""
    extract_features = pytest.importorskip("hloc.extract_features")
    match_features = pytest.importorskip("hloc.match_features")
    cfg = _shipped_cfg()
    assert cfg["features"] in extract_features.confs
    assert cfg["pairs"]["retrieval_model"] in extract_features.confs
    assert cfg["matcher"] in match_features.confs
    extractors = pytest.importorskip("hloc.extractors")
    base_model = pytest.importorskip("hloc.utils.base_model")
    model_name = extract_features.confs[cfg["features"]]["model"]["name"]
    extractor = base_model.dynamic_load(extractors, model_name)
    # Overrides must only use keys the extractor knows: unknown ones are
    # passed through and silently ignored or crash at model construction.
    assert set(cfg["feature_overrides"]["model"]) <= set(extractor.default_conf)
