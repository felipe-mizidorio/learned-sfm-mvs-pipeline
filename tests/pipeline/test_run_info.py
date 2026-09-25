from learned_sfm_mvs.pipeline.run_info import (
    StageTimer,
    learned_environment,
    with_backend_provenance,
)


def test_stage_timer_records_each_stage():
    timer = StageTimer()
    with timer("sfm"):
        pass
    with timer("mvs"):
        pass
    assert list(timer.stages) == ["sfm", "mvs"]
    assert timer.stages["sfm"]["seconds"] >= 0


def test_stage_timer_records_failed_stage():
    timer = StageTimer()
    try:
        with timer("mvs"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "mvs" in timer.stages


def test_learned_environment_keys():
    assert set(learned_environment()) == {"torch", "cuda", "cudnn", "gpu", "hloc"}


def test_with_backend_provenance_merges_environment():
    provenance: dict = {"environment": {"pycolmap": "4.2.0"}}

    result = with_backend_provenance(
        provenance,
        {"name": "hloc", "num_pairs": 45},
        {"name": "transmvsnet", "fusion": {"points": 10}},
        {"sfm": {"seconds": 1.0}},
    )

    assert result is provenance
    assert provenance["backends"]["sfm"] == {"name": "hloc", "num_pairs": 45}
    assert provenance["backends"]["mvs"]["name"] == "transmvsnet"
    assert provenance["stage_timings"] == {"sfm": {"seconds": 1.0}}
    assert provenance["environment"]["pycolmap"] == "4.2.0"
    assert "torch" in provenance["environment"]
