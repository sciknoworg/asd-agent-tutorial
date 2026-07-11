"""Evaluation-only oracle for Stage 1 saturation processes."""

from __future__ import annotations

from math import log

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.stage1 import (
    Stage1Config,
    Stage1ExperimentRecord,
    Stage1ProcessParameters,
    true_growth,
)


class Stage1CurvePoint(BaseModel):
    """One point on a dense evaluation curve."""

    dose_s: float
    true_growth: float

    model_config = ConfigDict(extra="forbid")


class Stage1OracleReport(BaseModel):
    """Evaluation-only saturation facts hidden from optimizers."""

    true_saturation_value: float | None
    analytical_t95_s: float | None
    numerical_t95_s: float | None
    true_t95_s: float | None
    saturation_threshold_growth: float | None
    has_meaningful_saturation_threshold: bool
    dense_curve: list[Stage1CurvePoint] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Stage1Recommendation(BaseModel):
    """A final Stage 1 recommendation to evaluate."""

    recommended_dose_s: float = Field(ge=0.0)
    estimated_t95_s: float | None = Field(default=None, ge=0.0)
    estimated_saturation_value: float | None = Field(default=None, gt=0.0)
    declares_saturation: bool = True

    model_config = ConfigDict(extra="forbid")


class Stage1RecommendationMetrics(BaseModel):
    """Evaluation metrics for a Stage 1 final recommendation."""

    estimated_t95_s: float | None
    true_t95_s: float | None
    absolute_t95_error_s: float | None
    relative_t95_error: float | None
    growth_fraction_at_recommendation: float | None
    dose_overshoot_s: float | None
    cumulative_dose_s: float
    cumulative_simulated_process_time_s: float
    false_saturation_declaration: bool

    model_config = ConfigDict(extra="forbid")


class Stage1EvaluationOracle:
    """Evaluation-only oracle for one Stage 1 configuration."""

    def __init__(self, config: Stage1Config) -> None:
        self.config = config

    def evaluate(self, curve_points: int = 200) -> Stage1OracleReport:
        """Return hidden saturation facts and a dense true curve."""

        saturation_value = true_saturation_value(self.config.process)
        threshold = saturation_threshold_growth(self.config, saturation_value)
        meaningful = (
            saturation_value is not None
            and threshold is not None
            and 0.0 < threshold < saturation_value
        )
        analytical = analytical_t95(self.config, threshold) if meaningful else None
        numerical = numerical_t95(self.config, threshold) if meaningful else None
        return Stage1OracleReport(
            true_saturation_value=saturation_value,
            analytical_t95_s=analytical,
            numerical_t95_s=numerical,
            true_t95_s=analytical if analytical is not None else numerical,
            saturation_threshold_growth=threshold,
            has_meaningful_saturation_threshold=meaningful,
            dense_curve=dense_curve(self.config, curve_points),
        )


def true_saturation_value(process: Stage1ProcessParameters) -> float | None:
    """Return the finite true saturation value, if the process has one."""

    if process.family == "weakly_non_self_limited":
        return None
    return process.g_inf


def saturation_threshold_growth(
    config: Stage1Config,
    saturation_value: float | None,
) -> float | None:
    """Return the evaluation threshold for known-target or inferred-asymptote mode."""

    if config.objective.mode == "known_target":
        if config.objective.target_growth is None:
            return None
        return config.objective.saturation_fraction * config.objective.target_growth
    if saturation_value is None:
        return None
    return config.objective.saturation_fraction * saturation_value


def analytical_t95(config: Stage1Config, threshold: float | None) -> float | None:
    """Return an analytical t95 when the family has a closed-form mono-exponential."""

    process = config.process
    if threshold is None or process.family not in {"mono_exponential", "noisy_saturation"}:
        return None
    if process.k is None or not 0.0 < threshold < process.g_inf:
        return None
    return -log(1.0 - threshold / process.g_inf) / process.k


def numerical_t95(config: Stage1Config, threshold: float | None) -> float | None:
    """Return a bisection t95 for monotonic finite-asymptote processes."""

    if threshold is None:
        return None
    if true_saturation_value(config.process) is None:
        return None
    if threshold <= 0.0 or threshold >= config.process.g_inf:
        return None

    lower = 0.0
    upper = max(1.0, config.dose_bounds_s.max)
    for _ in range(80):
        if true_growth(config.process, upper) >= threshold:
            break
        upper *= 2.0
    else:
        return None

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if true_growth(config.process, midpoint) >= threshold:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def dense_curve(config: Stage1Config, n_points: int = 200) -> list[Stage1CurvePoint]:
    """Return a dense true-response curve over the configured dose bounds."""

    if n_points < 2:
        raise ValueError("dense curve requires at least two points")
    lower = config.dose_bounds_s.min
    upper = config.dose_bounds_s.max
    step = (upper - lower) / float(n_points - 1)
    return [
        Stage1CurvePoint(dose_s=dose, true_growth=true_growth(config.process, dose))
        for dose in (lower + step * index for index in range(n_points))
    ]


def evaluate_recommendation(
    config: Stage1Config,
    recommendation: Stage1Recommendation,
    records: list[Stage1ExperimentRecord],
    oracle_report: Stage1OracleReport | None = None,
) -> Stage1RecommendationMetrics:
    """Calculate evaluation-only metrics for a Stage 1 recommendation."""

    report = oracle_report or Stage1EvaluationOracle(config).evaluate()
    estimated = recommendation.estimated_t95_s
    true_t95 = report.true_t95_s
    absolute_error: float | None = None
    relative_error: float | None = None
    dose_overshoot: float | None = None
    if estimated is not None and true_t95 is not None:
        absolute_error = abs(estimated - true_t95)
        relative_error = absolute_error / true_t95 if true_t95 > 0.0 else None
    if true_t95 is not None:
        dose_overshoot = recommendation.recommended_dose_s - true_t95

    growth_fraction: float | None = None
    if report.true_saturation_value is not None and report.true_saturation_value > 0.0:
        growth_fraction = (
            true_growth(config.process, recommendation.recommended_dose_s)
            / report.true_saturation_value
        )

    return Stage1RecommendationMetrics(
        estimated_t95_s=estimated,
        true_t95_s=true_t95,
        absolute_t95_error_s=absolute_error,
        relative_t95_error=relative_error,
        growth_fraction_at_recommendation=growth_fraction,
        dose_overshoot_s=dose_overshoot,
        cumulative_dose_s=sum(record.dose_s for record in records),
        cumulative_simulated_process_time_s=sum(record.process_time_s for record in records),
        false_saturation_declaration=(
            recommendation.declares_saturation and not report.has_meaningful_saturation_threshold
        ),
    )
