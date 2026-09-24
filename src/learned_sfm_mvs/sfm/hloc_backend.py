"""hloc SfM backend: learned local features and LightGlue matching.

hloc extracts features, selects pairs and matches them; the COLMAP database
import, geometric verification and incremental mapping then run exactly like
the colmap_sift backend (see ``sfm/base.py``), with the same mapper options.

hloc and torch are loaded lazily (``_load_hloc``) so the CPU dev environment,
which does not install the ``learned`` group, can still import this module.
"""

import copy
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pycolmap

from learned_sfm_mvs.sfm.base import (
    SfmInputs,
    SfmResult,
    check_all_images_imported,
    map_and_select,
    prepare_output,
)
from learned_sfm_mvs.sfm.feature_masks import filter_features_by_mask
from learned_sfm_mvs.sfm.pairs import (
    Pair,
    exhaustive_pairs,
    merge_pairs,
    read_pairs,
    sequential_pairs,
    write_pairs,
)

logger = logging.getLogger(__name__)

PAIR_STRATEGIES = ("sequential", "retrieval", "exhaustive")


def _load_hloc() -> SimpleNamespace:
    """Import the hloc entry points this backend uses."""
    import torch
    from hloc import extract_features, match_features, pairs_from_retrieval
    from hloc.reconstruction import create_empty_db, get_image_ids
    from hloc.triangulation import (
        estimation_and_geometric_verification,
        import_features,
        import_matches,
    )

    return SimpleNamespace(
        cuda_available=torch.cuda.is_available(),
        extract_features=extract_features,
        match_features=match_features,
        pairs_from_retrieval=pairs_from_retrieval,
        create_empty_db=create_empty_db,
        get_image_ids=get_image_ids,
        import_features=import_features,
        import_matches=import_matches,
        estimation_and_geometric_verification=estimation_and_geometric_verification,
    )


def build_conf(confs: dict, name: str, overrides: dict | None, kind: str) -> dict:
    """Copy an hloc conf by name and deep-merge overrides into it.

    Parameters
    ----------
    confs : dict
        hloc's conf table (e.g. ``extract_features.confs``).
    name : str
        Key into ``confs``.
    overrides : dict or None
        Nested values replacing those of the base conf.
    kind : str
        What the conf is, for the error message.

    Returns
    -------
    dict
        A new conf; ``confs`` is not modified.

    Raises
    ------
    ValueError
        If ``name`` is not in ``confs``.
    """
    if name not in confs:
        raise ValueError(f"Unknown hloc {kind} {name!r}. Available: {sorted(confs)}")
    return _deep_merge(copy.deepcopy(confs[name]), overrides or {})


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def select_pairs(
    hloc: SimpleNamespace,
    names: list[str],
    image_dir: Path,
    work_dir: Path,
    pairs_cfg: dict,
) -> list[Pair]:
    """Union of the configured pair strategies.

    Parameters
    ----------
    hloc : SimpleNamespace
        Output of ``_load_hloc``.
    names : list[str]
        Image names in capture order.
    image_dir : Path
        Root image directory.
    work_dir : Path
        Scratch directory for retrieval descriptors.
    pairs_cfg : dict
        ``pairs`` section of ``configs/hloc.yaml``.

    Returns
    -------
    list[Pair]
        Unique unordered pairs.

    Raises
    ------
    ValueError
        If no strategy or an unknown strategy is configured.
    """
    strategies = list(pairs_cfg.get("strategies", []))
    unknown = [s for s in strategies if s not in PAIR_STRATEGIES]
    if not strategies or unknown:
        raise ValueError(
            f"pairs.strategies must be a non-empty subset of {PAIR_STRATEGIES}, "
            f"got {strategies}"
        )

    pair_lists: list[list[Pair]] = []
    if "exhaustive" in strategies:
        pair_lists.append(exhaustive_pairs(names))
    if "sequential" in strategies:
        pair_lists.append(sequential_pairs(names, int(pairs_cfg["sequential_overlap"])))
    if "retrieval" in strategies and len(names) > 1:
        pair_lists.append(_retrieval_pairs(hloc, names, image_dir, work_dir, pairs_cfg))

    pairs = merge_pairs(*pair_lists)
    logger.info(
        "Pairs: %d unique from %s (%s)",
        len(pairs),
        "+".join(strategies),
        ", ".join(str(len(p)) for p in pair_lists),
    )
    return pairs


def _retrieval_pairs(
    hloc: SimpleNamespace,
    names: list[str],
    image_dir: Path,
    work_dir: Path,
    pairs_cfg: dict,
) -> list[Pair]:
    conf = build_conf(
        hloc.extract_features.confs,
        pairs_cfg["retrieval_model"],
        None,
        "retrieval model",
    )
    descriptors = work_dir / "global-features.h5"
    hloc.extract_features.main(
        conf, image_dir, image_list=names, feature_path=descriptors
    )
    pairs_path = work_dir / "pairs-retrieval.txt"
    hloc.pairs_from_retrieval.main(
        descriptors,
        pairs_path,
        num_matched=min(int(pairs_cfg["retrieval_num_matched"]), len(names) - 1),
        query_list=names,
        db_list=names,
    )
    return read_pairs(pairs_path)


def run(inputs: SfmInputs, hloc_cfg: dict, mapping_options: dict) -> SfmResult:
    """Learned features and matching, then COLMAP verification and mapping.

    Parameters
    ----------
    inputs : SfmInputs
        Run inputs.
    hloc_cfg : dict
        ``hloc`` section of ``configs/hloc.yaml``.
    mapping_options : dict
        ``incremental_mapping`` section of ``configs/colmap.yaml``.

    Returns
    -------
    SfmResult
        The model with most registered images; ``stats`` holds pair and mask
        counts.

    Raises
    ------
    ValueError
        On an unknown feature, matcher, retrieval model or pair strategy.
    RuntimeError
        If no pairs or no model result.
    """
    hloc = _load_hloc()
    if inputs.device == "cpu" and hloc.cuda_available:
        logger.warning(
            "device=cpu only applies to COLMAP stages; hloc runs on CUDA when "
            "available."
        )
    names = inputs.image_names
    feature_conf = build_conf(
        hloc.extract_features.confs,
        hloc_cfg["features"],
        hloc_cfg.get("feature_overrides"),
        "feature config",
    )
    matcher_conf = build_conf(
        hloc.match_features.confs,
        hloc_cfg["matcher"],
        hloc_cfg.get("matcher_overrides"),
        "matcher",
    )

    prepare_output(inputs)
    # hloc skips images already present in an existing .h5, so a previous
    # run's features (maybe from another config) would be reused silently.
    work_dir = inputs.output_dir / "hloc"
    if work_dir.exists():
        logger.warning("Removing stale hloc outputs '%s'", work_dir)
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    features = work_dir / "features.h5"
    hloc.extract_features.main(
        feature_conf, inputs.image_dir, image_list=names, feature_path=features
    )
    mask_stats = None
    if inputs.mask_dir is not None:
        mask_stats = filter_features_by_mask(features, names, inputs.mask_dir)

    pairs = select_pairs(hloc, names, inputs.image_dir, work_dir, hloc_cfg["pairs"])
    if not pairs:
        raise RuntimeError("No image pairs selected; need at least two images.")
    pairs_path = work_dir / "pairs.txt"
    write_pairs(pairs, pairs_path)
    matches = work_dir / "matches.h5"
    hloc.match_features.main(
        matcher_conf, pairs_path, features=features, matches=matches
    )

    hloc.create_empty_db(inputs.database_path)
    _import_images(inputs)
    check_all_images_imported(inputs)
    image_ids = hloc.get_image_ids(inputs.database_path)
    with pycolmap.Database.open(inputs.database_path) as db:
        hloc.import_features(image_ids, db, features)
        hloc.import_matches(image_ids, db, pairs_path, matches)
    hloc.estimation_and_geometric_verification(inputs.database_path, pairs_path)

    result = map_and_select(inputs, mapping_options)
    stats = {
        "features": hloc_cfg["features"],
        "matcher": hloc_cfg["matcher"],
        "pair_strategies": list(hloc_cfg["pairs"]["strategies"]),
        "num_pairs": len(pairs),
        "feature_masks": mask_stats,
    }
    return SfmResult(result.reconstruction, result.model_path, result.num_models, stats)


def _import_images(inputs: SfmInputs) -> None:
    """Register images and their camera(s) in the database."""
    options = pycolmap.ImageReaderOptions()
    if inputs.camera_model is not None:
        options.camera_model = inputs.camera_model
    if inputs.camera_params is not None:
        options.camera_params = inputs.camera_params
    pycolmap.import_images(
        inputs.database_path,
        inputs.image_dir,
        inputs.camera_mode,
        image_names=inputs.image_names,
        options=options,
    )
