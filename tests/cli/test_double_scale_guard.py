"""--skip-fusion double-scale guard (deferred from the domain port until the CLI existed)."""

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_RESUME_MODULE = "learned_sfm_mvs.cli.resume_mvs"


def _run_resume_skip_fusion(output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            _RESUME_MODULE,
            "--output-dir",
            str(output_dir),
            "--image-dir",
            str(output_dir),
            "--skip-fusion",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def test_skip_fusion_refuses_already_scaled_dense(tmp_path):
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {"run_script": "sfm-mvs-resume-mvs", "scale_factor_mm_per_unit": 123.4}
        )
    )
    result = _run_resume_skip_fusion(tmp_path)
    assert result.returncode == 1
    assert "double-scale" in result.stdout + result.stderr


def test_skip_fusion_allowed_after_run_pipeline(tmp_path):
    # sfm-mvs-run never scales dense.ply itself, so --skip-fusion is safe;
    # the guard must not trigger (the script then fails later on the missing
    # sparse model, which is expected in this bare tmp dir).
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps({"run_script": "sfm-mvs-run", "scale_factor_mm_per_unit": 123.4})
    )
    result = _run_resume_skip_fusion(tmp_path)
    assert "double-scale" not in result.stdout + result.stderr
