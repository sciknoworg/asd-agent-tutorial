import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.gp import GenericGPSettings
from asd_agent.bo.optimizers import (
    Stage1PhysicsGPSettings,
    Stage1PhysicsInformedGPOptimizer,
    Stage1RunnerSettings,
    run_stage1_optimization,
)
from asd_agent.bo.physics_models import (
    PhysicsGPFitFailure,
    PhysicsInformedGPSettings,
    fit_physics_informed_stage1_gp,
)
from asd_agent.bo.stage1 import Stage1Config, Stage1ExperimentRecord, Stage1VirtualLab, true_growth
from asd_agent.config import load_stage1_scenario


def stage1_records(name: str = "fast_mono") -> tuple[Stage1Config, list[Stage1ExperimentRecord]]:
    config = load_stage1_scenario(name)
    lab = Stage1VirtualLab(config, seed=101)
    bounds = config.dose_bounds_s
    records = [
        lab.observe(bounds.lerp(0.0), "phys_001"),
        lab.observe(bounds.lerp(0.25), "phys_002"),
        lab.observe(bounds.lerp(0.50), "phys_003"),
    ]
    return config, records


def test_physics_gp_parameters_are_positive() -> None:
    config, records = stage1_records()

    model = fit_physics_informed_stage1_gp(
        records,
        config.dose_bounds_s,
        PhysicsInformedGPSettings(noise_mode="known", known_noise_variance=1e-6),
        seed=1,
    )

    parameters = model.physical_parameters()
    assert parameters["g_inf"] > 0.0
    assert parameters["k"] > 0.0


def test_physics_gp_posterior_is_valid() -> None:
    config, records = stage1_records()
    model = fit_physics_informed_stage1_gp(records, config.dose_bounds_s, seed=2)

    prediction = model.posterior([0.25, 1.0, 2.0])

    assert len(prediction.mean) == 3
    assert all(value == pytest.approx(value) for value in prediction.mean)
    assert all(value >= 0.0 for value in prediction.stddev)


def test_physics_gp_is_compatible_with_mono_exponential_data() -> None:
    config, records = stage1_records()
    model = fit_physics_informed_stage1_gp(records, config.dose_bounds_s, seed=3)
    dose = 1.5

    prediction = model.posterior([dose])

    assert prediction.mean[0] == pytest.approx(true_growth(config.process, dose), abs=0.35)


def test_physics_gp_handles_misspecification_gracefully() -> None:
    config = load_stage1_scenario("misspecified")

    result = run_stage1_optimization(
        config,
        "physics_gp",
        Stage1RunnerSettings(budget=5, simulator_seed=51, optimizer_seed=52),
    )

    assert result.status in {
        "success",
        "budget_exhausted",
        "no_saturation_detected",
        "model_fit_failure",
        "numerical_failure",
    }
    assert len(result.records) <= 5


def test_physics_gp_fallback_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    config, records = stage1_records()

    def fail_physics(*args: object, **kwargs: object) -> object:
        raise PhysicsGPFitFailure("forced physics failure")

    monkeypatch.setattr("asd_agent.bo.optimizers.fit_physics_informed_stage1_gp", fail_physics)
    optimizer = Stage1PhysicsInformedGPOptimizer(
        config,
        Stage1PhysicsGPSettings(
            fallback_generic_gp=GenericGPSettings(noise_mode="known", known_noise_variance=1e-6)
        ),
        seed=4,
    )

    result = optimizer.propose(records)

    assert result.status == "proposed"
    assert result.proposal is not None
    assert "fallback" in result.proposal.optimizer
    assert any("forced physics failure" in warning for warning in result.warnings)


def test_physics_gp_suggestion_is_reproducible() -> None:
    config, records = stage1_records()

    first = Stage1PhysicsInformedGPOptimizer(config, seed=5).propose(records)
    second = Stage1PhysicsInformedGPOptimizer(config, seed=5).propose(records)

    assert first.proposal is not None
    assert second.proposal is not None
    assert first.proposal.dose_s == second.proposal.dose_s


def test_stage1_result_schema_is_complete_for_bo04() -> None:
    config = load_stage1_scenario("fast_mono")

    result = run_stage1_optimization(
        config,
        "physics_gp",
        Stage1RunnerSettings(budget=6, simulator_seed=61, optimizer_seed=62),
    )
    row = result.summary_row()

    expected = {
        "success",
        "estimated_t95_s",
        "true_t95_s",
        "absolute_t95_error_s",
        "relative_t95_error",
        "growth_fraction_at_recommendation",
        "dose_overshoot_s",
        "false_saturation",
        "cumulative_dose_s",
        "cumulative_process_time_s",
        "uncertainty_coverage",
        "model_fit_warnings",
        "failure_category",
        "optimizer_wall_time_s",
    }
    assert expected <= set(row)
