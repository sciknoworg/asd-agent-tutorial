"""Objective, safety, and failure-category helpers."""

from __future__ import annotations

from asd_agent.models import (
    ExperimentCondition,
    ExperimentRecord,
    FailureCategory,
    ObjectiveEvaluation,
    ProcessConfig,
)


def selectivity(ga_thickness_nm: float, nga_thickness_nm: float, eps: float = 1e-12) -> float:
    """Return ASD selectivity with stable handling for zero total thickness."""

    denominator = ga_thickness_nm + nga_thickness_nm
    if denominator <= eps:
        return 0.0
    return (ga_thickness_nm - nga_thickness_nm) / denominator


def validate_safety(condition: ExperimentCondition, config: ProcessConfig) -> list[str]:
    """Return safety violations for a proposed experiment."""

    safety = config.safety
    violations: list[str] = []
    checks = [
        ("precursor_dose_s", condition.precursor_dose_s, safety.precursor_dose_s),
        ("coreactant_dose_s", condition.coreactant_dose_s, safety.coreactant_dose_s),
        ("inhibitor_dose_s", condition.inhibitor_dose_s, safety.inhibitor_dose_s),
        ("temperature_c", condition.temperature_c, safety.temperature_c),
        ("cycles", float(condition.cycles), safety.cycles),
    ]
    for name, value, allowed in checks:
        if not allowed.contains(value):
            violations.append(f"{name}={value} outside [{allowed.min}, {allowed.max}]")
    return violations


def evaluate_objective(
    condition: ExperimentCondition,
    ga_thickness_nm: float,
    nga_thickness_nm: float,
    config: ProcessConfig,
) -> ObjectiveEvaluation:
    """Evaluate safety and ASD success constraints."""

    reasons = validate_safety(condition, config)
    objective = config.objective
    sel = selectivity(ga_thickness_nm, nga_thickness_nm)
    if ga_thickness_nm < objective.ga_min_nm:
        reasons.append(f"GA thickness {ga_thickness_nm:.3f} nm below {objective.ga_min_nm:.3f} nm")
    if nga_thickness_nm > objective.nga_max_nm:
        reasons.append(
            f"NGA thickness {nga_thickness_nm:.3f} nm above {objective.nga_max_nm:.3f} nm"
        )
    if sel < objective.selectivity_min:
        reasons.append(f"selectivity {sel:.3f} below {objective.selectivity_min:.3f}")
    return ObjectiveEvaluation(
        meets_safety=not validate_safety(condition, config),
        meets_objective=not reasons,
        failure_reasons=reasons,
    )


def classify_failure(records: list[ExperimentRecord], config: ProcessConfig) -> FailureCategory:
    """Classify the run outcome for benchmark summaries."""

    if any(record.meets_objective for record in records):
        return "success"
    if any(validate_safety(record.condition, config) for record in records):
        return "safety_violation"
    if not records:
        return "budget_exhausted"

    feasible_ga = [
        record for record in records if record.ga_thickness_nm >= config.objective.ga_min_nm
    ]
    high_dose_tests = [
        record
        for record in records
        if record.condition.precursor_dose_s >= config.safety.precursor_dose_s.lerp(0.65)
        and record.condition.coreactant_dose_s >= config.safety.coreactant_dose_s.lerp(0.65)
    ]
    strong_inhibitor_tests = [
        record
        for record in records
        if record.condition.inhibitor_dose_s >= config.safety.inhibitor_dose_s.lerp(0.60)
    ]
    if (
        feasible_ga
        and high_dose_tests
        and (strong_inhibitor_tests or config.scenario != "inhibitor_selectivity")
    ):
        return "no_selective_window"
    return "budget_exhausted"


def best_success(records: list[ExperimentRecord]) -> ExperimentRecord | None:
    """Return the fastest successful experiment, if one exists."""

    successes = [record for record in records if record.meets_objective]
    if not successes:
        return None
    return min(successes, key=lambda record: record.process_time_s)
