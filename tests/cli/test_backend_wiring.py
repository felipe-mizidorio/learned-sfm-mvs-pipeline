"""sfm-mvs-run and sfm-mvs-resume-mvs wiring, with the heavy stages stubbed."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learned_sfm_mvs.cli import resume_dense, resume_mvs, run
from learned_sfm_mvs.cli.options import REPO_ROOT
from learned_sfm_mvs.mvs.base import MvsResult
from learned_sfm_mvs.pipeline.post_fusion import PostFusionResult
from learned_sfm_mvs.sfm.base import SfmResult

SCALE_STATUS = {"status": "recovered_unvalidated", "units": "mm"}


def _post(tmp_path) -> PostFusionResult:
    return PostFusionResult(
        mesh_ply=tmp_path / "out" / "mesh.ply",
        scale_factor=100.0,
        scale_status=SCALE_STATUS,
        scale_sanity=None,
        scale_self_consistency=None,
        sor_stats={"point_cloud_filtering": {"points_after": 9}},
        lcc_stats={"lcc": {"triangles_kept": 10}},
    )


def _images(tmp_path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["f1.jpg", "f2.jpg"]:
        (image_dir / name).touch()
    return image_dir


def _main(module, argv):
    with patch.object(sys, "argv", ["prog", *argv]):
        module.main()


def _manifest(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "pipeline_manifest.json").read_text())


class _Stages:
    """Patches run.py's stages; exposes the mocks."""

    def __init__(self, tmp_path):
        reconstruction = MagicMock()
        reconstruction.num_reg_images.return_value = 2
        out = tmp_path / "out"
        self.sfm = SfmResult(reconstruction, out / "sparse" / "0", 1, {"num_pairs": 1})
        self.mvs = MvsResult(
            out / "dense.ply",
            None,
            {"depth": {"depth_maps": 2}, "fusion": {"points": 5}, "fusion_masks": None},
        )
        self.post = _post(tmp_path)

    def __enter__(self):
        self.patches = [
            patch.object(run, "run_sfm", return_value=self.sfm),
            patch.object(run, "run_mvs", return_value=self.mvs),
            patch.object(run, "run_post_fusion", return_value=self.post),
        ]
        self.run_sfm, self.run_mvs, self.run_post_fusion = (
            p.start() for p in self.patches
        )
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


# --- sfm-mvs-run ---


def test_run_defaults_to_pipeline_yaml_backends(tmp_path):
    args = [
        "--image-dir",
        str(_images(tmp_path)),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with _Stages(tmp_path) as stages:
        _main(run, args)

    backend, sfm_inputs, sfm_configs = stages.run_sfm.call_args.args
    assert backend == "hloc"
    assert sfm_inputs.image_names == ["f1.jpg", "f2.jpg"]
    assert sfm_inputs.shared_camera is True
    assert set(sfm_configs) == {"colmap", "hloc"}
    backend, mvs_inputs, mvs_configs = stages.run_mvs.call_args.args
    assert backend == "transmvsnet"
    assert mvs_inputs.sparse_model_path == stages.sfm.model_path
    # Config paths are repo-relative, whatever the working directory.
    checkpoint = Path(mvs_configs["transmvsnet"]["checkpoint"])
    assert checkpoint == REPO_ROOT / "models/transmvsnet/model_bld.ckpt"
    assert stages.run_post_fusion.call_args.args[0] == stages.mvs.dense_ply

    manifest = _manifest(tmp_path)
    assert manifest["run_script"] == "sfm-mvs-run"
    assert manifest["backends"]["sfm"]["name"] == "hloc"
    assert manifest["backends"]["sfm"]["registered_images"] == 2
    assert manifest["backends"]["sfm"]["num_pairs"] == 1
    assert manifest["backends"]["mvs"] == {
        "name": "transmvsnet",
        "depth": {"depth_maps": 2},
        "fusion": {"points": 5},
    }
    assert set(manifest["stage_timings"]) == {"sfm", "mvs", "post_fusion"}
    assert {"hloc", "transmvsnet", "pipeline"} <= set(manifest["resolved_configs"])
    assert "torch" in manifest["environment"]
    assert manifest["scale"] == SCALE_STATUS


def test_run_backend_flags_override_pipeline_yaml(tmp_path):
    args = [
        "--image-dir",
        str(_images(tmp_path)),
        "--output-dir",
        str(tmp_path / "out"),
        "--sfm-backend",
        "colmap_sift",
        "--mvs-backend",
        "patchmatch",
        "--per-image-cameras",
    ]
    with _Stages(tmp_path) as stages:
        _main(run, args)

    assert stages.run_sfm.call_args.args[0] == "colmap_sift"
    assert stages.run_sfm.call_args.args[1].shared_camera is False
    assert stages.run_mvs.call_args.args[0] == "patchmatch"
    manifest = _manifest(tmp_path)
    assert manifest["intrinsics_source"] == "self_calibration_per_image"
    # Unused learned configs are not recorded as if they had been used.
    assert "hloc" not in manifest["resolved_configs"]
    assert "transmvsnet" not in manifest["resolved_configs"]


def test_run_skip_mvs_stops_after_sfm(tmp_path):
    args = [
        "--image-dir",
        str(_images(tmp_path)),
        "--output-dir",
        str(tmp_path / "out"),
        "--skip-mvs",
    ]
    with _Stages(tmp_path) as stages:
        _main(run, args)
    stages.run_mvs.assert_not_called()


def test_run_sfm_failure_exits_nonzero(tmp_path):
    args = [
        "--image-dir",
        str(_images(tmp_path)),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with _Stages(tmp_path) as stages:
        stages.run_sfm.side_effect = RuntimeError("No sparse model reconstructed.")
        with pytest.raises(SystemExit) as exit_info:
            _main(run, args)
    assert exit_info.value.code == 1
    stages.run_mvs.assert_not_called()


# --- sfm-mvs-resume-mvs ---


def _resume(tmp_path, previous_manifest, extra=()):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    if previous_manifest is not None:
        (out / "pipeline_manifest.json").write_text(json.dumps(previous_manifest))
    fused = MvsResult(
        out / "dense.ply", None, {"fusion": {"points": 3}, "fusion_masks": None}
    )
    with (
        patch.object(
            resume_mvs,
            "load_best_reconstruction",
            return_value=(MagicMock(), out / "sparse" / "0"),
        ),
        patch.object(resume_mvs, "fuse", return_value=fused) as mock_fuse,
        patch.object(resume_mvs, "run_post_fusion", return_value=_post(tmp_path)),
    ):
        _main(
            resume_mvs, ["--output-dir", str(out), "--image-dir", str(tmp_path), *extra]
        )
    return mock_fuse, _manifest(tmp_path)


def test_resume_refuses_previous_runs_backend(tmp_path):
    previous = {
        "run_script": "sfm-mvs-run",
        "backends": {"mvs": {"name": "patchmatch"}},
    }

    mock_fuse, manifest = _resume(tmp_path, previous)

    assert mock_fuse.call_args.args[0] == "patchmatch"
    assert manifest["backends"]["mvs"] == {
        "name": "patchmatch",
        "fusion": {"points": 3},
    }
    assert "fusion" in manifest["stage_timings"]


def test_resume_without_manifest_uses_pipeline_yaml(tmp_path):
    mock_fuse, _ = _resume(tmp_path, None)
    assert mock_fuse.call_args.args[0] == "transmvsnet"


def test_resume_flag_overrides_previous_backend(tmp_path):
    previous = {"backends": {"mvs": {"name": "patchmatch"}}}
    mock_fuse, _ = _resume(tmp_path, previous, ["--mvs-backend", "transmvsnet"])
    assert mock_fuse.call_args.args[0] == "transmvsnet"


def test_resume_skip_fusion_still_records_backend(tmp_path):
    previous = {
        "run_script": "sfm-mvs-run",
        "backends": {"mvs": {"name": "patchmatch"}},
    }

    mock_fuse, manifest = _resume(tmp_path, previous, ["--skip-fusion"])

    mock_fuse.assert_not_called()
    # The next resume must still find which backend produced mvs/.
    assert manifest["backends"]["mvs"] == {"name": "patchmatch", "fusion": None}


def test_resume_dense_carries_backend_forward(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dense.ply").touch()
    (out / "pipeline_manifest.json").write_text(
        json.dumps(
            {"run_script": "sfm-mvs-run", "backends": {"mvs": {"name": "patchmatch"}}}
        )
    )
    with (
        patch.object(
            resume_dense,
            "load_best_reconstruction",
            return_value=(MagicMock(), out / "sparse" / "0"),
        ),
        patch.object(resume_dense, "run_sor", return_value=(out / "f.ply", {})),
        patch.object(
            resume_dense, "recover_scale_details_safe", return_value=(None, None, None)
        ),
        patch.object(resume_dense, "run_poisson_lcc", return_value=(MagicMock(), {})),
    ):
        _main(resume_dense, ["--output-dir", str(out), "--image-dir", str(tmp_path)])

    assert _manifest(tmp_path)["backends"]["mvs"] == {
        "name": "patchmatch",
        "fusion": None,
    }
