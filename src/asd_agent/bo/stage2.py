"""Stage 2 constrained multi-objective ASD problem definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from asd_agent.models import ExperimentCondition, ProcessConfig, Range
from asd_agent.objective import selectivity
from asd_agent.simulator import VirtualLab

Stage2ObjectiveName = Literal["maximize_ga", "minimize_nga", "minimize_process_time"]


def default_stage2_objectives() -> list[Stage2ObjectiveName]:
    """Return the required Stage 2 objective set."""

    return ["maximize_ga", "minimize_nga", "minimize_process_time"]


class Stage2Decision(BaseModel):
    """Optimizer-facing Stage 2 decision variables."""

    precursor_dose_s: float = Field(ge=0.0)
    temperature_c: float
    cycle_count: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")

    def rounded_key(self) -> tuple[float, float, int]:
        """Return a stable duplicate-detection key."""

        return (round(self.precursor_dose_s, 6), round(self.temperature_c, 6), self.cycle_count)


class Stage2FixedParameters(BaseModel):
    """Simulator parameters fixed outside the Stage 2 search space."""

    coreactant_dose_s: float = Field(gt=0.0)
    inhibitor_dose_s: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage2HardBounds(BaseModel):
    """Immutable hard bounds for Stage 2 optimization."""

    precursor_dose_s: Range
    temperature_c: Range
    cycle_count: Range
    max_process_time_s: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage2Constraints(BaseModel):
    """Feasibility thresholds kept in Stage 2 YAML."""

    ga_min_nm: float = Field(gt=0.0)
    nga_max_nm: float = Field(ge=0.0)
    selectivity_min: float = Field(ge=-1.0, le=1.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage2OracleGrid(BaseModel):
    """Dense mixed-variable enumeration settings for the evaluation oracle."""

    precursor_points: int = Field(default=17, ge=2)
    temperature_points: int = Field(default=17, ge=2)
    cycle_values: list[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_cycle_values(self) -> Stage2OracleGrid:
        if any(value < 0 for value in self.cycle_values):
            raise ValueError("cycle_values must be non-negative integers")
        return self


class Stage2ScenarioMetadata(BaseModel):
    """Human-facing documentation for one Stage 2 scenario."""

    scientific_interpretation: str
    hidden_process_parameters: str
    noise: str
    feasible_window_exists: bool
    expected_difficulty: str

    model_config = ConfigDict(extra="forbid")


class Stage2Config(BaseModel):
    """Complete Stage 2 constrained multi-objective ASD problem."""

    scenario_id: str
    description: str
    process: ProcessConfig
    fixed_parameters: Stage2FixedParameters
    hard_bounds: Stage2HardBounds
    constraints: Stage2Constraints
    objectives: list[Stage2ObjectiveName] = Field(default_factory=default_stage2_objectives)
    oracle_grid: Stage2OracleGrid = Field(default_factory=Stage2OracleGrid)
    metadata: Stage2ScenarioMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_objectives_and_bounds(self) -> Stage2Config:
        required = {"maximize_ga", "minimize_nga", "minimize_process_time"}
        if set(self.objectives) != required:
            raise ValueError("Stage 2 must configure GA, NGA, and process-time objectives")
        process_safety = self.process.safety
        if not process_safety.precursor_dose_s.contains(self.hard_bounds.precursor_dose_s.min):
            raise ValueError("Stage 2 precursor lower bound outside simulator safety")
        if not process_safety.precursor_dose_s.contains(self.hard_bounds.precursor_dose_s.max):
            raise ValueError("Stage 2 precursor upper bound outside simulator safety")
        if not process_safety.temperature_c.contains(self.hard_bounds.temperature_c.min):
            raise ValueError("Stage 2 temperature lower bound outside simulator safety")
        if not process_safety.temperature_c.contains(self.hard_bounds.temperature_c.max):
            raise ValueError("Stage 2 temperature upper bound outside simulator safety")
        if not process_safety.cycles.contains(float(self.hard_bounds.cycle_count.min)):
            raise ValueError("Stage 2 cycle lower bound outside simulator safety")
        if not process_safety.cycles.contains(float(self.hard_bounds.cycle_count.max)):
            raise ValueError("Stage 2 cycle upper bound outside simulator safety")
        return self

    def optimizer_view(self) -> dict[str, object]:
        """Return information available to optimizers, excluding hidden parameters."""

        return {
            "problem_kind": "stage2_constrained_multi_objective_asd",
            "scenario_id": self.scenario_id,
            "description": self.description,
            "decision_variables": {
                "precursor_dose_s": self.hard_bounds.precursor_dose_s.model_dump(mode="json"),
                "temperature_c": self.hard_bounds.temperature_c.model_dump(mode="json"),
                "cycle_count": self.hard_bounds.cycle_count.model_dump(mode="json"),
            },
            "fixed_parameters": self.fixed_parameters.model_dump(mode="json"),
            "constraints": self.constraints.model_dump(mode="json"),
            "objectives": list(self.objectives),
            "max_process_time_s": self.hard_bounds.max_process_time_s,
        }


class Stage2Outcomes(BaseModel):
    """Measured or oracle outcomes for a Stage 2 ASD decision."""

    ga_thickness_nm: float
    nga_thickness_nm: float
    selectivity: float
    process_time_s: float

    model_config = ConfigDict(extra="forbid")


class Stage2ConstraintEvaluation(BaseModel):
    """Constraint status for one Stage 2 outcome."""

    feasible: bool
    violations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def condition_from_stage2_decision(
    config: Stage2Config, decision: Stage2Decision
) -> ExperimentCondition:
    """Convert Stage 2 variables to the existing simulator condition."""

    return ExperimentCondition(
        precursor_dose_s=decision.precursor_dose_s,
        coreactant_dose_s=config.fixed_parameters.coreactant_dose_s,
        inhibitor_dose_s=config.fixed_parameters.inhibitor_dose_s,
        temperature_c=decision.temperature_c,
        cycles=decision.cycle_count,
    )


def decision_from_condition(condition: ExperimentCondition) -> Stage2Decision:
    """Convert an existing condition to Stage 2 variables."""

    return Stage2Decision(
        precursor_dose_s=condition.precursor_dose_s,
        temperature_c=condition.temperature_c,
        cycle_count=condition.cycles,
    )


def validate_stage2_decision(config: Stage2Config, decision: Stage2Decision) -> list[str]:
    """Return immutable hard-bound violations for a Stage 2 decision."""

    bounds = config.hard_bounds
    violations: list[str] = []
    if not bounds.precursor_dose_s.contains(decision.precursor_dose_s):
        violations.append(
            "precursor_dose_s="
            f"{decision.precursor_dose_s} outside "
            f"[{bounds.precursor_dose_s.min}, {bounds.precursor_dose_s.max}]"
        )
    if not bounds.temperature_c.contains(decision.temperature_c):
        violations.append(
            f"temperature_c={decision.temperature_c} outside "
            f"[{bounds.temperature_c.min}, {bounds.temperature_c.max}]"
        )
    if not bounds.cycle_count.contains(float(decision.cycle_count)):
        violations.append(
            f"cycle_count={decision.cycle_count} outside "
            f"[{bounds.cycle_count.min}, {bounds.cycle_count.max}]"
        )
    process_time = stage2_process_time(config, decision)
    if bounds.max_process_time_s is not None and process_time > bounds.max_process_time_s:
        violations.append(
            f"process_time_s={process_time:.3f} above {bounds.max_process_time_s:.3f}"
        )
    return violations


def evaluate_stage2_constraints(
    config: Stage2Config,
    decision: Stage2Decision,
    outcomes: Stage2Outcomes,
) -> Stage2ConstraintEvaluation:
    """Evaluate hard safety and feasibility constraints for Stage 2."""

    violations = validate_stage2_decision(config, decision)
    constraints = config.constraints
    if outcomes.ga_thickness_nm < constraints.ga_min_nm:
        violations.append(
            f"GA thickness {outcomes.ga_thickness_nm:.3f} nm below {constraints.ga_min_nm:.3f} nm"
        )
    if outcomes.nga_thickness_nm > constraints.nga_max_nm:
        violations.append(
            f"NGA thickness {outcomes.nga_thickness_nm:.3f} nm above "
            f"{constraints.nga_max_nm:.3f} nm"
        )
    if outcomes.selectivity < constraints.selectivity_min:
        violations.append(
            f"selectivity {outcomes.selectivity:.3f} below {constraints.selectivity_min:.3f}"
        )
    return Stage2ConstraintEvaluation(feasible=not violations, violations=violations)


def simulate_stage2(
    config: Stage2Config,
    decision: Stage2Decision,
    *,
    seed: int | None = None,
) -> Stage2Outcomes:
    """Run a measured Stage 2 decision through the existing virtual lab."""

    if validate_stage2_decision(config, decision):
        raise ValueError(f"unsafe Stage 2 decision: {validate_stage2_decision(config, decision)}")
    lab = VirtualLab(config.process, seed=seed)
    record = lab.simulate(condition_from_stage2_decision(config, decision))
    return Stage2Outcomes(
        ga_thickness_nm=record.ga_thickness_nm,
        nga_thickness_nm=record.nga_thickness_nm,
        selectivity=record.selectivity,
        process_time_s=record.process_time_s,
    )


def oracle_stage2_outcomes(config: Stage2Config, decision: Stage2Decision) -> Stage2Outcomes:
    """Return noise-free evaluation-only Stage 2 outcomes."""

    condition = condition_from_stage2_decision(config, decision)
    lab = VirtualLab(
        config.process.model_copy(update={"noise_sigma_nm": 0.0}), seed=config.process.seed
    )
    ga = lab.surface_thickness("GA", condition)
    nga = lab.surface_thickness("NGA", condition)
    return Stage2Outcomes(
        ga_thickness_nm=ga,
        nga_thickness_nm=nga,
        selectivity=selectivity(ga, nga),
        process_time_s=lab.process_time(condition),
    )


def stage2_process_time(config: Stage2Config, decision: Stage2Decision) -> float:
    """Return total simulated process time for a Stage 2 decision."""

    lab = VirtualLab(config.process, seed=config.process.seed)
    return lab.process_time(condition_from_stage2_decision(config, decision))


def cycle_grid(config: Stage2Config) -> list[int]:
    """Return integer cycle values for oracle enumeration."""

    if config.oracle_grid.cycle_values:
        return [
            value
            for value in sorted(set(config.oracle_grid.cycle_values))
            if config.hard_bounds.cycle_count.contains(float(value))
        ]
    lower = int(config.hard_bounds.cycle_count.min)
    upper = int(config.hard_bounds.cycle_count.max)
    midpoint = int(round((lower + upper) / 2.0))
    return sorted({lower, midpoint, upper})


def linspace_range(bounds: Range, n_points: int) -> list[float]:
    """Return a closed deterministic grid for a continuous range."""

    if n_points < 2:
        raise ValueError("linspace_range requires at least two points")
    step = (bounds.max - bounds.min) / float(n_points - 1)
    return [bounds.min + index * step for index in range(n_points)]


def enumerate_stage2_decisions(config: Stage2Config) -> list[Stage2Decision]:
    """Enumerate the dense mixed continuous/integer oracle grid."""

    return [
        Stage2Decision(
            precursor_dose_s=precursor,
            temperature_c=temperature,
            cycle_count=cycles,
        )
        for precursor in linspace_range(
            config.hard_bounds.precursor_dose_s,
            config.oracle_grid.precursor_points,
        )
        for temperature in linspace_range(
            config.hard_bounds.temperature_c,
            config.oracle_grid.temperature_points,
        )
        for cycles in cycle_grid(config)
    ]


def stage2_objective_vector(outcomes: Stage2Outcomes) -> tuple[float, float, float]:
    """Return objectives in minimization form for Pareto calculations."""

    return (-outcomes.ga_thickness_nm, outcomes.nga_thickness_nm, outcomes.process_time_s)


def objective_dominates(first: Sequence[float], second: Sequence[float]) -> bool:
    """Return whether the first minimization vector dominates the second."""

    return all(a <= b for a, b in zip(first, second, strict=True)) and any(
        a < b for a, b in zip(first, second, strict=True)
    )
