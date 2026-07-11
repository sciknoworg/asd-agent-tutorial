from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.stage2_analysis import generate_stage2_analysis_outputs
from asd_agent.bo.stage2_benchmark import (
    default_stage2_benchmark_methods,
    load_stage2_benchmark_profile,
    profile_configs,
    run_stage2_benchmark,
    save_stage2_benchmark_results,
    stage2_observation_rows,
    stage2_summary_rows,
)


def test_stage2_smoke_comparison_and_analysis_generation(tmp_path: Path) -> None:
    profile = load_stage2_benchmark_profile("smoke").model_copy(
        update={"scenarios": ["inherent_selectivity"], "budget": 3}
    )

    results = run_stage2_benchmark(profile)
    configs = profile_configs(profile)
    result_paths = save_stage2_benchmark_results(profile, results, tmp_path / "raw")
    analysis_paths = generate_stage2_analysis_outputs(
        results, tmp_path / "analysis", configs=configs
    )

    assert {result.method for result in results} == set(default_stage2_benchmark_methods())
    assert all(len(result.observations) <= profile.budget for result in results)
    assert all(result.hypervolume_by_iteration for result in results)
    assert all(path.exists() for path in result_paths)
    assert all(path.exists() for path in analysis_paths)
    assert stage2_summary_rows(results)
    assert stage2_observation_rows(results, configs)

    figure_paths = [path for path in analysis_paths if path.suffix == ".png"]
    assert figure_paths
    assert all(path.with_suffix(".csv").exists() for path in figure_paths)


def test_stage2_pilot_profile_is_tutorial_scale() -> None:
    profile = load_stage2_benchmark_profile("pilot")

    assert profile.repetitions == 3
    assert profile.budget == 6
    assert "stage2_mobo" in profile.methods
