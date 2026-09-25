"""Argument and config helpers shared by the entry points."""

import argparse
import json
from pathlib import Path

import yaml

from learned_sfm_mvs.mvs.base import MVS_BACKENDS
from learned_sfm_mvs.sfm.base import SFM_BACKENDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"


def load_yaml(path: Path) -> dict:
    """Parse a YAML file.

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    dict
        Parsed content.
    """
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path from a config file: relative paths are repo-relative.

    Parameters
    ----------
    path : str or Path
        Path as written in the config.

    Returns
    -------
    Path
        Absolute path.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def add_backend_arguments(parser: argparse.ArgumentParser, sfm: bool) -> None:
    """Add backend selection and learned-backend config flags.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    sfm : bool
        Also add the SfM backend flags.
    """
    parser.add_argument(
        "--pipeline-config",
        default=CONFIGS / "pipeline.yaml",
        type=Path,
        help="Default backends (configs/pipeline.yaml).",
    )
    if sfm:
        parser.add_argument(
            "--sfm-backend",
            choices=SFM_BACKENDS,
            default=None,
            help="SfM backend; overrides pipeline.yaml.",
        )
        parser.add_argument(
            "--hloc-config",
            default=CONFIGS / "hloc.yaml",
            type=Path,
            help="Path to hloc.yaml (hloc SfM backend).",
        )
    parser.add_argument(
        "--mvs-backend",
        choices=MVS_BACKENDS,
        default=None,
        help="MVS backend; overrides pipeline.yaml.",
    )
    parser.add_argument(
        "--transmvsnet-config",
        default=CONFIGS / "transmvsnet.yaml",
        type=Path,
        help="Path to transmvsnet.yaml (TransMVSNet MVS backend).",
    )


def load_transmvsnet_config(path: Path) -> dict:
    """The ``transmvsnet`` section, with the checkpoint path resolved.

    Parameters
    ----------
    path : Path
        transmvsnet.yaml.

    Returns
    -------
    dict
        Config section; ``checkpoint`` is absolute.
    """
    cfg = load_yaml(path)["transmvsnet"]
    cfg["checkpoint"] = str(resolve_repo_path(cfg["checkpoint"]))
    return cfg


def pipeline_defaults(path: Path) -> dict:
    """The ``pipeline`` section of pipeline.yaml (default backends).

    Parameters
    ----------
    path : Path
        pipeline.yaml.

    Returns
    -------
    dict
        ``sfm_backend`` and ``mvs_backend``.

    Raises
    ------
    ValueError
        If a configured backend is unknown.
    """
    cfg = load_yaml(path)["pipeline"]
    if cfg["sfm_backend"] not in SFM_BACKENDS:
        raise ValueError(f"{path}: sfm_backend must be one of {SFM_BACKENDS}")
    if cfg["mvs_backend"] not in MVS_BACKENDS:
        raise ValueError(f"{path}: mvs_backend must be one of {MVS_BACKENDS}")
    return cfg


def previous_mvs_backend(output_dir: Path) -> str | None:
    """MVS backend recorded by the previous run in ``output_dir``, if any.

    Parameters
    ----------
    output_dir : Path
        Run output directory.

    Returns
    -------
    str or None
        Backend name, or None without a (backend-aware) manifest.
    """
    manifest_path = output_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        return None
    backends = json.loads(manifest_path.read_text()).get("backends") or {}
    name = (backends.get("mvs") or {}).get("name")
    return name if name in MVS_BACKENDS else None
