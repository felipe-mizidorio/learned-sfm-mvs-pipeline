"""Checks shared by the resume entry points."""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# The only entry point that scales dense.ply in place.
_SCALES_DENSE_IN_PLACE = "sfm-mvs-resume-mvs"


def guard_against_double_scale(output_dir: Path, attempted: str, remedy: str) -> None:
    """Exit if dense.ply was already scaled to millimetres by resume-mvs.

    sfm-mvs-resume-mvs scales dense.ply in place after meshing. A step that
    re-derives the scale from the (SfM-unit) sparse model and applies it to
    that cloud again would silently produce geometry scale² too large.

    Parameters
    ----------
    output_dir : Path
        Run output directory holding pipeline_manifest.json.
    attempted : str
        What the user ran, for the message.
    remedy : str
        How to get an unscaled dense.ply back, for the message.
    """
    manifest_path = output_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        return
    previous = json.loads(manifest_path.read_text())
    scale = previous.get("scale_factor_mm_per_unit")
    if previous.get("run_script") == _SCALES_DENSE_IN_PLACE and scale:
        logger.error(
            "dense.ply in '%s' was already scaled to millimetres by a previous "
            "%s run (scale %.6f mm/unit, see pipeline_manifest.json). Running "
            "%s would double-scale it. %s, or delete pipeline_manifest.json if "
            "dense.ply was replaced manually.",
            output_dir,
            _SCALES_DENSE_IN_PLACE,
            scale,
            attempted,
            remedy,
        )
        sys.exit(1)
