from asd_agent.bo.acquisition import (
    ThresholdAcquisitionSettings,
    choose_threshold_candidate,
    dose_grid,
    smallest_tested_recommendation,
)
from asd_agent.bo.stage1 import Stage1VirtualLab
from asd_agent.config import load_stage1_scenario


def test_threshold_rule_prefers_smallest_probable_target_crossing() -> None:
    decision = choose_threshold_candidate(
        candidate_doses_s=[0.0, 1.0, 2.0, 3.0],
        posterior_mean=[0.0, 0.4, 0.9, 1.0],
        posterior_stddev=[0.02, 0.05, 0.04, 0.04],
        target_growth=0.8,
        tested_doses_s=[0.0],
        settings=ThresholdAcquisitionSettings(target_probability=0.8),
    )

    assert decision.status == "candidate"
    assert decision.dose_s == 2.0
    assert decision.target_probability is not None
    assert decision.target_probability >= 0.8


def test_threshold_rule_explores_uncertainty_near_target_before_crossing() -> None:
    decision = choose_threshold_candidate(
        candidate_doses_s=[0.0, 1.0, 2.0, 3.0],
        posterior_mean=[0.0, 0.45, 0.72, 0.74],
        posterior_stddev=[0.01, 0.08, 0.20, 0.06],
        target_growth=0.8,
        tested_doses_s=[0.0],
        settings=ThresholdAcquisitionSettings(target_probability=0.95),
    )

    assert decision.status == "candidate"
    assert decision.dose_s == 2.0
    assert "uncertainty" in decision.rationale


def test_threshold_rule_avoids_duplicates_and_reports_exhaustion() -> None:
    decision = choose_threshold_candidate(
        candidate_doses_s=[0.0, 1.0],
        posterior_mean=[0.0, 1.0],
        posterior_stddev=[0.1, 0.1],
        target_growth=0.8,
        tested_doses_s=[0.0, 1.0],
    )

    assert decision.status == "no_valid_candidate"


def test_dose_grid_includes_bounds() -> None:
    config = load_stage1_scenario("fast_mono")

    grid = dose_grid(config.dose_bounds_s, 4)

    assert grid[0] == config.dose_bounds_s.min
    assert grid[-1] == config.dose_bounds_s.max


def test_smallest_tested_recommendation_uses_tested_experiment_id() -> None:
    config = load_stage1_scenario("fast_mono")
    lab = Stage1VirtualLab(config, seed=1)
    records = [
        lab.observe(0.0, "rec_001"),
        lab.observe(1.0, "rec_002"),
        lab.observe(2.0, "rec_003"),
    ]

    recommendation = smallest_tested_recommendation(config, records, min_observations=3)

    assert recommendation is not None
    value, experiment_id = recommendation
    assert experiment_id == "rec_003"
    assert value.recommended_dose_s == 2.0
