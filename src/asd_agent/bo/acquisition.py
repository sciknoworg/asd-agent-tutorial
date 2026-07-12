"""Threshold-oriented acquisition rules for Stage 1 saturation learning."""

from __future__ import annotations

from collections.abc import Sequence
from math import erfc, exp, isfinite, sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.oracle import Stage1Recommendation
from asd_agent.bo.stage1 import Stage1Config, Stage1ExperimentRecord
from asd_agent.models import Range

ThresholdDecisionStatus = Literal["candidate", "no_valid_candidate", "numerical_failure"]


class ThresholdAcquisitionSettings(BaseModel):
    """Settings for the Stage 1 threshold-oriented decision rule."""

    candidate_grid_size: int = Field(default=121, ge=2)
    target_probability: float = Field(default=0.80, gt=0.0, lt=1.0)
    duplicate_tolerance_s: float = Field(default=1e-8, ge=0.0)
    uncertainty_band_fraction: float = Field(default=0.20, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class ThresholdDecision(BaseModel):
    """One acquisition decision, including explicit failure states."""

    status: ThresholdDecisionStatus
    dose_s: float | None = None
    acquisition_value: float | None = None
    posterior_mean: float | None = None
    posterior_stddev: float | None = None
    target_growth: float | None = None
    target_probability: float | None = None
    rationale: str

    model_config = ConfigDict(extra="forbid")


def dose_grid(bounds: Range, n_points: int) -> list[float]:
    """Return a deterministic closed grid over the dose bounds."""

    if n_points < 2:
        raise ValueError("dose grid requires at least two points")
    step = (bounds.max - bounds.min) / float(n_points - 1)
    return [bounds.min + step * index for index in range(n_points)]


def estimate_stage1_target_growth(
    config: Stage1Config,
    records: Sequence[Stage1ExperimentRecord],
    posterior_mean: Sequence[float] = (),
) -> float | None:
    """Estimate the growth threshold using optimizer-visible information only."""

    if config.objective.mode == "known_target":
        return config.objective.target_growth

    visible_values = [record.observed_growth for record in records]
    visible_values.extend(posterior_mean)
    estimated_asymptote = max(visible_values, default=0.0)
    if estimated_asymptote <= 0.0:
        return None
    return config.objective.saturation_fraction * estimated_asymptote


def choose_threshold_candidate(
    candidate_doses_s: Sequence[float],
    posterior_mean: Sequence[float],
    posterior_stddev: Sequence[float],
    target_growth: float | None,
    tested_doses_s: Sequence[float],
    settings: ThresholdAcquisitionSettings | None = None,
) -> ThresholdDecision:
    """Select the next dose using a threshold-oriented posterior rule.

    The rule evaluates P(growth >= target) on a candidate grid. It first chooses the
    smallest untested dose whose probability exceeds the configured probability
    threshold. If no such dose exists, it samples the untested point with the largest
    uncertainty-weighted proximity to the target:

    score(t) = sigma(t) * exp(-abs(mu(t) - target) / bandwidth)
    """

    resolved_settings = settings or ThresholdAcquisitionSettings()
    if target_growth is None or not isfinite(target_growth):
        return ThresholdDecision(
            status="numerical_failure",
            target_growth=target_growth,
            rationale="No finite target growth estimate is available.",
        )
    if not (len(candidate_doses_s) == len(posterior_mean) == len(posterior_stddev)):
        raise ValueError("candidate doses, posterior means, and posterior stddevs must align")

    probabilities: list[float] = []
    valid_indexes: list[int] = []
    for index, (dose, mean, stddev) in enumerate(
        zip(candidate_doses_s, posterior_mean, posterior_stddev, strict=True)
    ):
        if not all(isfinite(value) for value in (dose, mean, stddev)):
            return ThresholdDecision(
                status="numerical_failure",
                target_growth=target_growth,
                rationale="Posterior grid contains a non-finite value.",
            )
        if not is_duplicate_dose(dose, tested_doses_s, resolved_settings.duplicate_tolerance_s):
            valid_indexes.append(index)
        probabilities.append(threshold_probability(mean, max(stddev, 0.0), target_growth))

    if not valid_indexes:
        return ThresholdDecision(
            status="no_valid_candidate",
            target_growth=target_growth,
            rationale="No untested candidate dose remains.",
        )

    eligible = [
        index
        for index in valid_indexes
        if probabilities[index] >= resolved_settings.target_probability
    ]
    if eligible:
        chosen = min(eligible, key=lambda index: candidate_doses_s[index])
        return ThresholdDecision(
            status="candidate",
            dose_s=candidate_doses_s[chosen],
            acquisition_value=probabilities[chosen],
            posterior_mean=posterior_mean[chosen],
            posterior_stddev=posterior_stddev[chosen],
            target_growth=target_growth,
            target_probability=probabilities[chosen],
            rationale="Smallest untested dose with sufficient posterior target probability.",
        )

    bandwidth = uncertainty_bandwidth(target_growth, posterior_stddev, resolved_settings)
    chosen = max(
        valid_indexes,
        key=lambda index: (
            posterior_stddev[index] * exp(-abs(posterior_mean[index] - target_growth) / bandwidth),
            -candidate_doses_s[index],
        ),
    )
    return ThresholdDecision(
        status="candidate",
        dose_s=candidate_doses_s[chosen],
        acquisition_value=(
            posterior_stddev[chosen] * exp(-abs(posterior_mean[chosen] - target_growth) / bandwidth)
        ),
        posterior_mean=posterior_mean[chosen],
        posterior_stddev=posterior_stddev[chosen],
        target_growth=target_growth,
        target_probability=probabilities[chosen],
        rationale="Untested dose expected to reduce uncertainty near the target threshold.",
    )


def threshold_probability(mean: float, stddev: float, target_growth: float) -> float:
    """Return P(Y >= target_growth) for a normal posterior."""

    if stddev <= 0.0:
        return 1.0 if mean >= target_growth else 0.0
    z_value = (target_growth - mean) / stddev
    return 0.5 * erfc(z_value / sqrt(2.0))


def is_duplicate_dose(
    dose_s: float,
    tested_doses_s: Sequence[float],
    tolerance_s: float,
) -> bool:
    """Return whether a candidate is already represented by a tested dose."""

    return any(abs(dose_s - tested) <= tolerance_s for tested in tested_doses_s)


def uncertainty_bandwidth(
    target_growth: float,
    posterior_stddev: Sequence[float],
    settings: ThresholdAcquisitionSettings,
) -> float:
    """Return a stable bandwidth for uncertainty-near-threshold scoring."""

    max_stddev = max((stddev for stddev in posterior_stddev if isfinite(stddev)), default=0.0)
    return max(abs(target_growth) * settings.uncertainty_band_fraction, max_stddev, 1e-12)


def smallest_tested_recommendation(
    config: Stage1Config,
    records: Sequence[Stage1ExperimentRecord],
    *,
    posterior_mean: Sequence[float] = (),
    target_growth: float | None = None,
    min_observations: int = 1,
) -> tuple[Stage1Recommendation, str] | None:
    """Return the smallest tested dose meeting the current threshold estimate."""

    if len(records) < min_observations:
        return None
    target = (
        target_growth
        if target_growth is not None
        else estimate_stage1_target_growth(config, records, posterior_mean)
    )
    if target is None:
        return None
    eligible = [record for record in records if record.observed_growth >= target]
    if not eligible:
        return None
    chosen = min(eligible, key=lambda record: record.dose_s)
    estimated_saturation = target / config.objective.saturation_fraction
    return (
        Stage1Recommendation(
            recommended_dose_s=chosen.dose_s,
            estimated_t95_s=chosen.dose_s,
            estimated_saturation_value=estimated_saturation,
            declares_saturation=True,
        ),
        chosen.experiment_id,
    )
