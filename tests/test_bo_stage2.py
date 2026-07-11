import pytest

from asd_agent.bo.stage2 import (
    Stage2Decision,
    Stage2Outcomes,
    evaluate_stage2_constraints,
    oracle_stage2_outcomes,
    simulate_stage2,
    validate_stage2_decision,
)
from asd_agent.bo.stage2_oracle import Stage2EvaluationOracle
from asd_agent.config import load_scenario, load_stage2_scenario
from asd_agent.objective import selectivity

STAGE2_SCENARIOS = [
    "inherent_selectivity",
    "inhibitor_selectivity",
    "impossible_selectivity",
    "narrow_selective_window",
    "noisy_measurements",
    "boundary_optimum",
    "soft_selectivity_breakdown",
    "model_misspecification",
]


def test_stage2_cycle_count_must_be_integer() -> None:
    with pytest.raises(ValueError):
        Stage2Decision(precursor_dose_s=1.0, temperature_c=180.0, cycle_count=20.5)


def test_stage2_selectivity_and_zero_growth_are_stable() -> None:
    assert selectivity(0.0, 0.0) == 0.0
    config = load_stage2_scenario("inherent_selectivity")
    outcomes = Stage2Outcomes(
        ga_thickness_nm=0.0,
        nga_thickness_nm=0.0,
        selectivity=selectivity(0.0, 0.0),
        process_time_s=0.0,
    )
    evaluation = evaluate_stage2_constraints(
        config,
        Stage2Decision(precursor_dose_s=1.0, temperature_c=180.0, cycle_count=5),
        outcomes,
    )

    assert not evaluation.feasible
    assert any("GA thickness" in violation for violation in evaluation.violations)


def test_stage2_constraints_accept_feasible_oracle_point() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    decision = Stage2Decision(precursor_dose_s=8.0, temperature_c=175.0, cycle_count=70)

    outcomes = oracle_stage2_outcomes(config, decision)
    evaluation = evaluate_stage2_constraints(config, decision, outcomes)

    assert evaluation.feasible
    assert outcomes.ga_thickness_nm >= config.constraints.ga_min_nm
    assert outcomes.nga_thickness_nm <= config.constraints.nga_max_nm
    assert outcomes.selectivity >= config.constraints.selectivity_min


def test_stage2_safety_bounds_reject_parameter_and_process_time() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    unsafe = Stage2Decision(precursor_dose_s=99.0, temperature_c=180.0, cycle_count=80)
    slow = Stage2Decision(precursor_dose_s=8.0, temperature_c=180.0, cycle_count=80)

    violations = validate_stage2_decision(config, unsafe)
    slow_violations = validate_stage2_decision(config, slow)

    assert any("precursor_dose_s" in violation for violation in violations)
    assert any("process_time_s" in violation for violation in slow_violations)


def test_stage2_feasibility_matches_documented_scenarios() -> None:
    feasible = Stage2EvaluationOracle(load_stage2_scenario("inherent_selectivity")).evaluate()
    impossible = Stage2EvaluationOracle(load_stage2_scenario("impossible_selectivity")).evaluate()

    assert feasible.selective_window_exists
    assert feasible.feasible_points
    assert not impossible.selective_window_exists
    assert impossible.feasible_points == []


def test_stage2_oracle_isolation_from_optimizer_view() -> None:
    config = load_stage2_scenario("boundary_optimum")
    view_text = str(config.optimizer_view())

    forbidden = [
        "surfaces",
        "max_growth_per_cycle_nm",
        "nucleation_delay_cycles",
        "oracle_hypervolume",
        "selective_window_exists",
        "hidden_process_parameters",
    ]
    assert all(fragment not in view_text for fragment in forbidden)


def test_stage2_scenarios_load_and_reproduce_oracle() -> None:
    for scenario in STAGE2_SCENARIOS:
        config = load_stage2_scenario(scenario)
        first = Stage2EvaluationOracle(config).evaluate()
        second = Stage2EvaluationOracle(config).evaluate()

        assert first.selective_window_exists == config.metadata.feasible_window_exists
        assert first.oracle_hypervolume == second.oracle_hypervolume
        assert len(first.points) == len(second.points)


def test_stage2_noisy_measurements_are_seed_reproducible() -> None:
    config = load_stage2_scenario("noisy_measurements")
    decision = Stage2Decision(precursor_dose_s=6.0, temperature_c=180.0, cycle_count=65)

    first = simulate_stage2(config, decision, seed=123)
    second = simulate_stage2(config, decision, seed=123)

    assert first == second


def test_stage2_yaml_files_do_not_pollute_asd_scenario_loader() -> None:
    with pytest.raises(FileNotFoundError, match="Stage 2 scenario"):
        load_scenario("bo_stage2_inherent_selectivity")
