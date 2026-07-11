import math
from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.serialization import load_optimizer_state, save_optimizer_state
from asd_agent.bo.stage2 import condition_from_stage2_decision, validate_stage2_decision
from asd_agent.bo.stage2_mobo import (
    Stage2BOSettings,
    Stage2ConstrainedMOBOOptimizer,
    Stage2ModelFitFailure,
    build_stage2_acquisition,
    candidate_cycle_values,
    duplicate_or_too_close,
    evaluate_stage2_acquisition,
    fit_stage2_surrogate,
    observations_from_experiment_records,
    observe_stage2_decision,
    run_stage2_bo,
    sobol_initial_decisions,
    stage2_outcome_constraints,
)
from asd_agent.config import load_stage2_scenario
from asd_agent.experiment_loop import run_conditions


def fast_settings() -> Stage2BOSettings:
    return Stage2BOSettings(
        experiment_budget=4,
        initial_design_size=3,
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
        acquisition_timeout_s=3.0,
        candidate_cycle_values=[30, 50, 70],
        random_fallback_points=32,
    )


def mobo_observations() -> tuple[object, Stage2BOSettings, list[object]]:
    config = load_stage2_scenario("inherent_selectivity")
    settings = fast_settings()
    decisions = sobol_initial_decisions(
        config,
        3,
        seed=101,
        cycle_values=settings.candidate_cycle_values,
    )
    observations = [
        observe_stage2_decision(
            config, decision, experiment_id=f"mobo_{index:03d}", seed=200 + index
        )
        for index, decision in enumerate(decisions, start=1)
    ]
    return config, settings, observations


def test_stage2_mobo_acquisition_value_is_finite() -> None:
    config, settings, observations = mobo_observations()
    surrogate = fit_stage2_surrogate(config, observations, settings, seed=12)
    acquisition = build_stage2_acquisition(config, surrogate, settings, seed=12)

    value = evaluate_stage2_acquisition(acquisition, observations[0].decision)

    assert math.isfinite(value)


def test_stage2_mobo_candidate_respects_bounds_and_integer_cycle_count() -> None:
    config, settings, observations = mobo_observations()
    result = Stage2ConstrainedMOBOOptimizer(config, settings, seed=13).propose(observations)

    assert result.status == "proposed"
    assert result.proposal is not None
    assert validate_stage2_decision(config, result.proposal.decision) == []
    assert result.proposal.decision.cycle_count in candidate_cycle_values(config, settings)
    assert isinstance(result.proposal.decision.cycle_count, int)


def test_stage2_mobo_outcome_constraints_use_configured_thresholds() -> None:
    torch = pytest.importorskip("torch")
    config = load_stage2_scenario("inherent_selectivity")
    constraints = stage2_outcome_constraints(config)
    feasible = torch.tensor([[[5.5, -0.4, 0.85, -900.0]]], dtype=torch.double)
    infeasible = torch.tensor([[[4.0, -0.8, 0.75, -900.0]]], dtype=torch.double)

    assert all(float(constraint(feasible).max()) <= 0.0 for constraint in constraints)
    assert any(float(constraint(infeasible).max()) > 0.0 for constraint in constraints)


def test_stage2_mobo_candidate_is_nonduplicate() -> None:
    config, settings, observations = mobo_observations()
    result = Stage2ConstrainedMOBOOptimizer(config, settings, seed=14).propose(observations)

    assert result.proposal is not None
    assert not duplicate_or_too_close(
        result.proposal.decision,
        observations,
        [],
        config,
        settings,
    )


def test_stage2_mobo_suggestion_is_deterministic_for_seed() -> None:
    config, settings, observations = mobo_observations()

    first = Stage2ConstrainedMOBOOptimizer(config, settings, seed=15).propose(observations)
    second = Stage2ConstrainedMOBOOptimizer(config, settings, seed=15).propose(observations)

    assert first.proposal is not None
    assert second.proposal is not None
    assert first.proposal.decision == second.proposal.decision
    assert first.proposal.fallback_used == second.proposal.fallback_used


def test_stage2_mobo_state_save_and_restore(tmp_path: Path) -> None:
    config, settings, observations = mobo_observations()
    optimizer = Stage2ConstrainedMOBOOptimizer(config, settings, seed=16)
    result = optimizer.propose(observations)
    assert result.proposal is not None
    state = optimizer.get_state()
    path = tmp_path / "stage2_mobo_state.json"

    save_optimizer_state(state, path)
    restored_state = load_optimizer_state(path)
    restored = Stage2ConstrainedMOBOOptimizer(config, settings, seed=1)
    restored.restore_state(restored_state)

    assert restored.get_state().state["proposals"] == state.state["proposals"]


def test_stage2_mobo_records_model_fit_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    config, settings, observations = mobo_observations()

    def fail_fit(*args: object, **kwargs: object) -> object:
        raise Stage2ModelFitFailure("forced fit failure")

    monkeypatch.setattr("asd_agent.bo.stage2_mobo.fit_stage2_surrogate", fail_fit)
    result = Stage2ConstrainedMOBOOptimizer(config, settings, seed=17).propose(observations)

    assert result.status == "proposed"
    assert result.proposal is not None
    assert result.proposal.fallback_used == "model_fit_failure"
    assert any("forced fit failure" in warning for warning in result.warnings)


def test_stage2_mobo_can_reuse_existing_ledger_records() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    decisions = sobol_initial_decisions(config, 2, seed=18, cycle_values=[30, 50])
    records = run_conditions(
        config.process,
        [condition_from_stage2_decision(config, decision) for decision in decisions],
        "legacy",
        seed=30,
    ).records

    observations = observations_from_experiment_records(config, records)

    assert [observation.experiment_id for observation in observations]
    assert all(
        "surfaces" not in str(observation.optimizer_payload()) for observation in observations
    )


def test_stage2_mobo_oracle_fields_are_not_in_optimizer_artifacts() -> None:
    config, settings, observations = mobo_observations()
    result = Stage2ConstrainedMOBOOptimizer(config, settings, seed=19).propose(observations)

    assert result.proposal is not None
    visible_text = str(result.proposal.model_dump(mode="json"))
    visible_text += str([observation.optimizer_payload() for observation in observations])
    forbidden = ["oracle_hypervolume", "selective_window_exists", "hidden_process_parameters"]
    assert all(fragment not in visible_text for fragment in forbidden)


def test_stage2_mobo_smoke_runs_required_scenarios() -> None:
    settings = Stage2BOSettings(
        experiment_budget=3,
        initial_design_size=2,
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
        acquisition_timeout_s=2.0,
        candidate_cycle_values=[30, 50, 70],
        random_fallback_points=32,
    )
    scenarios = [
        load_stage2_scenario("inherent_selectivity"),
        load_stage2_scenario("narrow_selective_window"),
        load_stage2_scenario("impossible_selectivity"),
    ]

    results = [
        run_stage2_bo(config, settings, simulator_seed=50, optimizer_seed=60)
        for config in scenarios
    ]

    assert {result.scenario_id for result in results} == {
        "bo_stage2_inherent_selectivity",
        "bo_stage2_narrow_selective_window",
        "bo_stage2_impossible_selectivity",
    }
    assert all(len(result.observations) <= settings.experiment_budget for result in results)
    assert all(result.hypervolume_by_iteration for result in results)
