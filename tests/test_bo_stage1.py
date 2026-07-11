from math import isclose, log

import pytest
from pydantic import ValidationError

from asd_agent.bo.oracle import (
    Stage1EvaluationOracle,
    Stage1Recommendation,
    evaluate_recommendation,
    numerical_t95,
)
from asd_agent.bo.stage1 import Stage1VirtualLab, true_growth
from asd_agent.config import load_scenario, load_stage1_scenario


def test_mono_exponential_behavior_is_monotonic() -> None:
    config = load_stage1_scenario("fast_mono")
    doses = [0.0, 0.5, 1.0, 2.0, 4.0]
    values = [true_growth(config.process, dose) for dose in doses]

    assert values[0] == 0.0
    assert values == sorted(values)
    assert values[-1] < config.process.g_inf


def test_analytical_t95_matches_mono_exponential_formula() -> None:
    config = load_stage1_scenario("fast_mono")
    report = Stage1EvaluationOracle(config).evaluate()
    threshold = report.saturation_threshold_growth

    assert threshold is not None
    assert config.process.k is not None
    expected = -log(1.0 - threshold / config.process.g_inf) / config.process.k
    assert report.analytical_t95_s is not None
    assert isclose(report.analytical_t95_s, expected, rel_tol=1e-12)
    assert report.true_t95_s == report.analytical_t95_s


def test_numerical_t95_for_non_analytical_soft_saturation() -> None:
    config = load_stage1_scenario("soft_biexponential")
    report = Stage1EvaluationOracle(config).evaluate()

    assert report.analytical_t95_s is None
    assert report.numerical_t95_s is not None
    assert report.saturation_threshold_growth is not None
    value = true_growth(config.process, report.numerical_t95_s)
    assert isclose(value, report.saturation_threshold_growth, rel_tol=1e-10, abs_tol=1e-10)


def test_soft_saturation_curve_is_stable_and_bounded() -> None:
    config = load_stage1_scenario("soft_biexponential")
    report = Stage1EvaluationOracle(config).evaluate(curve_points=50)
    values = [point.true_growth for point in report.dense_curve]

    assert all(value >= 0.0 for value in values)
    assert values == sorted(values)
    assert max(values) <= config.process.g_inf


def test_noisy_observations_are_deterministic_for_fixed_seed() -> None:
    config = load_stage1_scenario("noisy")
    first_lab = Stage1VirtualLab(config, seed=99)
    second_lab = Stage1VirtualLab(config, seed=99)

    first = first_lab.observe(2.0, experiment_id="noise_001")
    second = second_lab.observe(2.0, experiment_id="noise_001")

    assert first.observed_growth == second.observed_growth
    assert first.observed_growth != true_growth(config.process, 2.0)


def test_non_self_limited_oracle_classification_and_metrics() -> None:
    config = load_stage1_scenario("weak_nonselflimited")
    report = Stage1EvaluationOracle(config).evaluate()
    lab = Stage1VirtualLab(config, seed=5)
    records = [lab.observe(1.0, "nsl_001"), lab.observe(4.0, "nsl_002")]
    recommendation = Stage1Recommendation(recommended_dose_s=5.0, estimated_t95_s=5.0)

    metrics = evaluate_recommendation(config, recommendation, records, report)

    assert report.true_saturation_value is None
    assert report.true_t95_s is None
    assert not report.has_meaningful_saturation_threshold
    assert metrics.false_saturation_declaration is True
    assert metrics.cumulative_dose_s == 5.0


def test_model_misspecified_saturation_uses_numerical_t95() -> None:
    config = load_stage1_scenario("misspecified")
    report = Stage1EvaluationOracle(config).evaluate()

    assert config.process.family == "misspecified_saturation"
    assert report.analytical_t95_s is None
    assert report.numerical_t95_s is not None
    assert numerical_t95(config, report.saturation_threshold_growth) == report.numerical_t95_s


def test_known_target_and_inferred_asymptote_modes_are_separate() -> None:
    known = load_stage1_scenario("fast_mono")
    inferred = load_stage1_scenario("slow_mono")

    known_view = known.optimizer_view()
    inferred_view = inferred.optimizer_view()

    assert known.objective.mode == "known_target"
    assert "target_growth" in known_view
    assert inferred.objective.mode == "inferred_asymptote"
    assert "target_growth" not in inferred_view

    payload = inferred.model_dump()
    payload["objective"]["target_growth"] = 1.0
    with pytest.raises(ValidationError):
        type(inferred).model_validate(payload)


def test_oracle_information_is_isolated_from_optimizer_view_and_records() -> None:
    config = load_stage1_scenario("fast_mono")
    lab = Stage1VirtualLab(config, seed=1)
    record = lab.observe(1.0, experiment_id="iso_001")
    visible_text = str(lab.optimizer_view()) + str(record.model_dump(mode="json"))

    forbidden_fragments = [
        "scenario_id",
        "fast_mono",
        "mono_exponential",
        "g_inf",
        "k_fast",
        "k_slow",
        "true_t95",
        "true_saturation",
        "asymptote",
    ]
    assert all(fragment not in visible_text for fragment in forbidden_fragments)

    hidden = lab.hidden_process_parameters()
    assert hidden["scenario_id"] == "bo_stage1_fast_mono"
    assert "process" in hidden


def test_stage1_yaml_files_do_not_pollute_asd_scenario_loader() -> None:
    with pytest.raises(FileNotFoundError, match="Stage 1 scenario"):
        load_scenario("bo_stage1_fast_mono")
