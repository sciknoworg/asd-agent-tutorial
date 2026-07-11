"""Evaluation-only oracle for Stage 2 constrained ASD problems."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2ConstraintEvaluation,
    Stage2Decision,
    Stage2Outcomes,
    enumerate_stage2_decisions,
    evaluate_stage2_constraints,
    objective_dominates,
    oracle_stage2_outcomes,
    stage2_objective_vector,
)


class Stage2OraclePoint(BaseModel):
    """One hidden oracle evaluation on the mixed-variable grid."""

    decision: Stage2Decision
    outcomes: Stage2Outcomes
    constraint_evaluation: Stage2ConstraintEvaluation

    model_config = ConfigDict(extra="forbid")


class Stage2OracleReport(BaseModel):
    """Evaluation-only Stage 2 report hidden from optimizers."""

    scenario_id: str
    points: list[Stage2OraclePoint] = Field(default_factory=list)
    feasible_points: list[Stage2OraclePoint] = Field(default_factory=list)
    pareto_front: list[Stage2OraclePoint] = Field(default_factory=list)
    oracle_hypervolume: float
    selective_window_exists: bool

    model_config = ConfigDict(extra="forbid")


class Stage2EvaluationOracle:
    """Dense evaluation-only oracle for one Stage 2 configuration."""

    def __init__(self, config: Stage2Config) -> None:
        self.config = config

    def evaluate(self) -> Stage2OracleReport:
        """Enumerate the hidden mixed-variable grid and summarize feasible regions."""

        points = [
            evaluate_oracle_point(self.config, decision)
            for decision in enumerate_stage2_decisions(self.config)
        ]
        feasible = [point for point in points if point.constraint_evaluation.feasible]
        pareto_source = feasible if feasible else points
        pareto = pareto_front(pareto_source)
        return Stage2OracleReport(
            scenario_id=self.config.scenario_id,
            points=points,
            feasible_points=feasible,
            pareto_front=pareto,
            oracle_hypervolume=approximate_hypervolume(pareto),
            selective_window_exists=bool(feasible),
        )


def evaluate_oracle_point(config: Stage2Config, decision: Stage2Decision) -> Stage2OraclePoint:
    """Evaluate one grid point with noise-free hidden simulator outcomes."""

    outcomes = oracle_stage2_outcomes(config, decision)
    return Stage2OraclePoint(
        decision=decision,
        outcomes=outcomes,
        constraint_evaluation=evaluate_stage2_constraints(config, decision, outcomes),
    )


def pareto_front(points: list[Stage2OraclePoint]) -> list[Stage2OraclePoint]:
    """Return non-dominated points for GA, NGA, and process time objectives."""

    vectors = [stage2_objective_vector(point.outcomes) for point in points]
    front: list[Stage2OraclePoint] = []
    for index, point in enumerate(points):
        vector = vectors[index]
        if any(
            objective_dominates(other_vector, vector)
            for other_index, other_vector in enumerate(vectors)
            if other_index != index
        ):
            continue
        front.append(point)
    return front


def approximate_hypervolume(points: list[Stage2OraclePoint], grid_size: int = 18) -> float:
    """Approximate normalized 3D hypervolume for feasible Pareto points.

    The axes are transformed into maximization benefits:
    GA growth, inverse NGA growth, and inverse process time. A regular grid estimate
    keeps the implementation deterministic and dependency-light for tutorial use.
    """

    if not points:
        return 0.0
    ga_values = [point.outcomes.ga_thickness_nm for point in points]
    nga_values = [point.outcomes.nga_thickness_nm for point in points]
    time_values = [point.outcomes.process_time_s for point in points]
    ga_min, ga_max = min(ga_values), max(ga_values)
    nga_min, nga_max = min(nga_values), max(nga_values)
    time_min, time_max = min(time_values), max(time_values)

    benefits = [
        (
            normalize(point.outcomes.ga_thickness_nm, ga_min, ga_max),
            1.0 - normalize(point.outcomes.nga_thickness_nm, nga_min, nga_max),
            1.0 - normalize(point.outcomes.process_time_s, time_min, time_max),
        )
        for point in points
    ]
    dominated = 0
    total = grid_size**3
    for i in range(grid_size):
        x = (i + 0.5) / grid_size
        for j in range(grid_size):
            y = (j + 0.5) / grid_size
            for k in range(grid_size):
                z = (k + 0.5) / grid_size
                if any(bx >= x and by >= y and bz >= z for bx, by, bz in benefits):
                    dominated += 1
    return dominated / total


def normalize(value: float, lower: float, upper: float) -> float:
    """Normalize a value to [0, 1] with stable handling for a zero range."""

    if upper <= lower:
        return 1.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))
