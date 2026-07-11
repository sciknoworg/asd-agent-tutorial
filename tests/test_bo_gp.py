from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.gp import GenericGPSettings, GPFitFailure, fit_generic_stage1_gp
from asd_agent.bo.optimizers import (
    Stage1GenericGPOptimizer,
    Stage1RunnerSettings,
    compare_stage1_methods,
    run_stage1_optimization,
)
from asd_agent.bo.serialization import load_optimizer_state, save_optimizer_state
from asd_agent.bo.stage1 import Stage1Config, Stage1ExperimentRecord, Stage1VirtualLab
from asd_agent.config import load_stage1_scenario


def stage1_records() -> tuple[Stage1Config, list[Stage1ExperimentRecord]]:
    config = load_stage1_scenario("fast_mono")
    lab = Stage1VirtualLab(config, seed=10)
    records = [
        lab.observe(0.0, "gp_001"),
        lab.observe(0.8, "gp_002"),
        lab.observe(1.6, "gp_003"),
    ]
    return config, records


def test_generic_gp_posterior_is_finite() -> None:
    config, records = stage1_records()
    model = fit_generic_stage1_gp(
        records,
        config.dose_bounds_s,
        GenericGPSettings(noise_mode="known", known_noise_variance=1e-6),
        seed=123,
    )

    prediction = model.posterior([0.25, 1.25, 2.25])

    assert len(prediction.mean) == 3
    assert all(value == pytest.approx(value) for value in prediction.mean)
    assert all(value >= 0.0 for value in prediction.stddev)


def test_generic_gp_suggestion_respects_bounds_and_avoids_duplicates() -> None:
    config, records = stage1_records()
    optimizer = Stage1GenericGPOptimizer(config, seed=44)

    result = optimizer.propose(records)

    assert result.status == "proposed"
    assert result.proposal is not None
    assert config.dose_bounds_s.contains(result.proposal.dose_s)
    assert result.proposal.dose_s not in [record.dose_s for record in records]


def test_generic_gp_suggestion_is_deterministic_for_same_seed_and_data() -> None:
    config, records = stage1_records()

    first = Stage1GenericGPOptimizer(config, seed=45).propose(records)
    second = Stage1GenericGPOptimizer(config, seed=45).propose(records)

    assert first.proposal is not None
    assert second.proposal is not None
    assert first.proposal.dose_s == second.proposal.dose_s
    assert first.proposal.posterior_summaries == second.proposal.posterior_summaries


def test_stage1_runner_recommendation_references_tested_record() -> None:
    config = load_stage1_scenario("fast_mono")
    result = run_stage1_optimization(
        config,
        "generic_gp",
        Stage1RunnerSettings(budget=6, simulator_seed=12, optimizer_seed=34),
    )

    assert result.status == "success"
    assert result.recommended_experiment_id is not None
    assert result.recommended_experiment_id in {record.experiment_id for record in result.records}


def test_generic_gp_fit_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    config, records = stage1_records()

    def fail_fit(*args: object, **kwargs: object) -> object:
        raise GPFitFailure("forced fit failure")

    monkeypatch.setattr("asd_agent.bo.optimizers.fit_generic_stage1_gp", fail_fit)
    result = Stage1GenericGPOptimizer(config, seed=46).propose(records)

    assert result.status == "model_fit_failure"
    assert "forced fit failure" in result.warnings[0]


def test_generic_gp_optimizer_state_save_restore(tmp_path: Path) -> None:
    config, records = stage1_records()
    optimizer = Stage1GenericGPOptimizer(config, seed=47)
    proposal = optimizer.propose(records)
    assert proposal.status == "proposed"
    state = optimizer.get_state()
    path = tmp_path / "gp_state.json"

    save_optimizer_state(state, path)
    restored_state = load_optimizer_state(path)
    restored = Stage1GenericGPOptimizer(config, seed=1)
    restored.restore_state(restored_state)

    assert restored.get_state().state["proposed_doses_s"] == state.state["proposed_doses_s"]


def test_stage1_smoke_comparison_runs_required_scenarios() -> None:
    configs = [
        load_stage1_scenario("fast_mono"),
        load_stage1_scenario("slow_mono"),
        load_stage1_scenario("noisy"),
    ]
    settings = Stage1RunnerSettings(budget=5, simulator_seed=88, optimizer_seed=89)

    results = compare_stage1_methods(configs, settings=settings)

    assert len(results) == 6
    assert {result.method for result in results} == {"grid", "generic_gp"}
    assert all(len(result.records) <= settings.budget for result in results)
