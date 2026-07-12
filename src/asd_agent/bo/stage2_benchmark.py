"""Stage 2 benchmark profiles and method comparisons."""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.records import RunManifest, utc_now
from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2Decision,
    cycle_grid,
    decision_from_condition,
    linspace_range,
    validate_stage2_decision,
)
from asd_agent.bo.stage2_mobo import (
    Stage2BOSettings,
    Stage2CandidateProposal,
    Stage2Observation,
    best_stage2_observation,
    candidate_cycle_values,
    duplicate_or_too_close,
    observe_stage2_decision,
    observed_hypervolume,
    run_stage2_bo,
    sobol_initial_decisions,
)
from asd_agent.bo.stage2_oracle import Stage2EvaluationOracle
from asd_agent.config import CONFIG_DIR, load_config_mapping, load_stage2_scenario
from asd_agent.heuristic_agent import candidate_plan

Stage2BenchmarkMethod = Literal["random_search", "grid_search", "rule_based", "stage2_mobo"]


def default_stage2_benchmark_methods() -> list[Stage2BenchmarkMethod]:
    """Return the BO-07 comparison methods."""

    return ["random_search", "grid_search", "rule_based", "stage2_mobo"]


class Stage2BenchmarkProfile(BaseModel):
    """Reproducible Stage 2 benchmark profile."""

    profile_id: str
    description: str
    scenarios: list[str]
    methods: list[Stage2BenchmarkMethod] = Field(default_factory=default_stage2_benchmark_methods)
    repetitions: int = Field(default=1, ge=1)
    budget: int = Field(default=4, ge=1)
    initial_design_size: int = Field(default=2, ge=0)
    simulator_seed: int = 7207
    initialization_seed: int | None = None
    optimizer_seed: int = 8207
    grid_precursor_points: int = Field(default=3, ge=2)
    grid_temperature_points: int = Field(default=3, ge=2)
    candidate_cycle_values: list[int] = Field(default_factory=list)
    mobo_qmc_samples: int = Field(default=16, ge=4)
    mobo_num_restarts: int = Field(default=1, ge=1)
    mobo_raw_samples: int = Field(default=16, ge=4)
    mobo_acquisition_timeout_s: float | None = Field(default=4.0, gt=0.0)
    random_fallback_points: int = Field(default=64, ge=1)
    boundary_tolerance_fraction: float = Field(default=0.02, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    def mobo_settings(self) -> Stage2BOSettings:
        """Return MOBO settings matched to this benchmark profile."""

        return Stage2BOSettings(
            experiment_budget=self.budget,
            initial_design_size=max(self.initial_design_size, 2),
            qmc_samples=self.mobo_qmc_samples,
            num_restarts=self.mobo_num_restarts,
            raw_samples=self.mobo_raw_samples,
            acquisition_timeout_s=self.mobo_acquisition_timeout_s,
            candidate_cycle_values=list(self.candidate_cycle_values),
            random_fallback_points=self.random_fallback_points,
        )


class Stage2BenchmarkResult(BaseModel):
    """Serializable result for one Stage 2 method/scenario/repetition run."""

    method: Stage2BenchmarkMethod
    scenario_id: str
    repetition: int
    status: str
    observations: list[Stage2Observation]
    proposals: list[Stage2CandidateProposal] = Field(default_factory=list)
    hypervolume_by_iteration: list[float] = Field(default_factory=list)
    hypervolume_auc: float
    final_hypervolume: float
    oracle_hypervolume: float
    hypervolume_regret: float
    experiments_to_first_feasible: int | None = None
    constraint_violation_count: int
    unsafe_proposal_count: int
    duplicate_proposal_count: int
    boundary_proposal_count: int
    boundary_proposal_fraction: float
    fallback_use_count: int
    model_fit_failure_count: int
    failure_category: str
    recommended_experiment_id: str | None = None
    optimizer_wall_time_s: float = Field(ge=0.0)
    simulator_seed: int
    optimizer_seed: int
    warnings: list[str] = Field(default_factory=list)
    manifest: RunManifest

    model_config = ConfigDict(extra="forbid")

    def summary_row(self) -> dict[str, object]:
        """Return one flat summary row for CSV analysis."""

        best = best_stage2_observation(self.observations)
        return {
            "method": self.method,
            "scenario_id": self.scenario_id,
            "repetition": self.repetition,
            "status": self.status,
            "failure_category": self.failure_category,
            "success": self.status == "success",
            "n_experiments": len(self.observations),
            "experiments_to_first_feasible": self.experiments_to_first_feasible or "",
            "final_hypervolume": self.final_hypervolume,
            "oracle_hypervolume": self.oracle_hypervolume,
            "hypervolume_auc": self.hypervolume_auc,
            "hypervolume_regret": self.hypervolume_regret,
            "constraint_violation_count": self.constraint_violation_count,
            "unsafe_proposal_count": self.unsafe_proposal_count,
            "duplicate_proposal_count": self.duplicate_proposal_count,
            "boundary_proposal_count": self.boundary_proposal_count,
            "boundary_proposal_fraction": self.boundary_proposal_fraction,
            "fallback_use_count": self.fallback_use_count,
            "model_fit_failure_count": self.model_fit_failure_count,
            "best_selectivity": best.outcomes.selectivity if best else 0.0,
            "ga_thickness_nm": best.outcomes.ga_thickness_nm if best else 0.0,
            "nga_thickness_nm": best.outcomes.nga_thickness_nm if best else 0.0,
            "total_process_time_s": sum(
                observation.outcomes.process_time_s for observation in self.observations
            ),
            "recommended_experiment_id": self.recommended_experiment_id or "",
            "optimizer_wall_time_s": self.optimizer_wall_time_s,
            "simulator_seed": self.simulator_seed,
            "optimizer_seed": self.optimizer_seed,
            "warnings": "; ".join(self.warnings),
        }


def load_stage2_benchmark_profile(path_or_name: str | Path) -> Stage2BenchmarkProfile:
    """Load a Stage 2 benchmark profile from YAML or a configured profile name."""

    path = Path(path_or_name)
    if not path.exists():
        stem = str(path_or_name)
        if not stem.startswith("bo_stage2_"):
            stem = f"bo_stage2_{stem}"
        if not stem.endswith("_profile"):
            stem = f"{stem}_profile"
        path = CONFIG_DIR / f"{stem}.yaml"
    return Stage2BenchmarkProfile.model_validate(load_config_mapping(path))


def run_stage2_benchmark(profile: Stage2BenchmarkProfile) -> list[Stage2BenchmarkResult]:
    """Run all configured Stage 2 scenarios, methods, and repetitions."""

    results: list[Stage2BenchmarkResult] = []
    for repetition in range(profile.repetitions):
        for scenario in profile.scenarios:
            config = load_stage2_scenario(scenario)
            for method in profile.methods:
                results.append(run_stage2_method(config, method, profile, repetition))
    return results


def run_stage2_method(
    config: Stage2Config,
    method: Stage2BenchmarkMethod,
    profile: Stage2BenchmarkProfile,
    repetition: int,
) -> Stage2BenchmarkResult:
    """Run one Stage 2 benchmark method with matched seeds and initial design."""

    simulator_seed = profile.simulator_seed + repetition
    optimizer_seed = profile.optimizer_seed + repetition
    settings = profile.mobo_settings()
    initial = matched_initial_observations(config, profile, repetition)
    if method == "stage2_mobo":
        wall_start = perf_counter()
        mobo_result = run_stage2_bo(
            config,
            settings,
            simulator_seed=simulator_seed,
            optimizer_seed=optimizer_seed,
            initial_observations=initial,
        )
        return build_stage2_benchmark_result(
            config,
            profile,
            method,
            repetition,
            mobo_result.observations,
            simulator_seed,
            optimizer_seed,
            proposals=mobo_result.proposals,
            warnings=mobo_result.warnings,
            optimizer_wall_time_s=perf_counter() - wall_start,
            forced_status=mobo_result.status,
        )

    wall_start = perf_counter()
    candidate_decisions = stage2_candidate_sequence(config, method, profile, optimizer_seed)
    observations = run_stage2_decision_sequence(
        config,
        candidate_decisions,
        method,
        profile,
        repetition,
        initial,
        simulator_seed,
    )
    return build_stage2_benchmark_result(
        config,
        profile,
        method,
        repetition,
        observations,
        simulator_seed,
        optimizer_seed,
        optimizer_wall_time_s=perf_counter() - wall_start,
    )


def matched_initial_observations(
    config: Stage2Config,
    profile: Stage2BenchmarkProfile,
    repetition: int,
) -> list[Stage2Observation]:
    """Return matched Sobol initial observations for every method."""

    if profile.initial_design_size <= 0:
        return []
    settings = profile.mobo_settings()
    decisions = sobol_initial_decisions(
        config,
        min(profile.initial_design_size, profile.budget),
        seed=(
            profile.optimizer_seed
            if profile.initialization_seed is None
            else profile.initialization_seed
        )
        + repetition,
        cycle_values=candidate_cycle_values(config, settings),
    )
    return [
        observe_stage2_decision(
            config,
            decision,
            experiment_id=f"initial_{index:03d}",
            seed=profile.simulator_seed + repetition + index - 1,
        )
        for index, decision in enumerate(decisions, start=1)
    ]


def run_stage2_decision_sequence(
    config: Stage2Config,
    decisions: Sequence[Stage2Decision],
    method: Stage2BenchmarkMethod,
    profile: Stage2BenchmarkProfile,
    repetition: int,
    initial_observations: Sequence[Stage2Observation],
    simulator_seed: int,
) -> list[Stage2Observation]:
    """Evaluate a fixed Stage 2 decision sequence to the configured budget."""

    settings = profile.mobo_settings()
    observations = [observation.model_copy(deep=True) for observation in initial_observations]
    for decision in decisions:
        if len(observations) >= profile.budget:
            break
        if validate_stage2_decision(config, decision):
            continue
        if duplicate_or_too_close(decision, observations, [], config, settings):
            continue
        observations.append(
            observe_stage2_decision(
                config,
                decision,
                experiment_id=f"{method}_{len(observations) + 1:03d}",
                seed=simulator_seed + len(observations),
            )
        )
    if len(observations) < profile.budget and method != "random_search":
        fallback = random_stage2_decisions(
            config,
            profile,
            seed=profile.optimizer_seed + repetition + 10_000,
            n_points=profile.budget * 4,
        )
        for decision in fallback:
            if len(observations) >= profile.budget:
                break
            if duplicate_or_too_close(decision, observations, [], config, settings):
                continue
            observations.append(
                observe_stage2_decision(
                    config,
                    decision,
                    experiment_id=f"{method}_fallback_{len(observations) + 1:03d}",
                    seed=simulator_seed + len(observations),
                )
            )
    return observations


def stage2_candidate_sequence(
    config: Stage2Config,
    method: Stage2BenchmarkMethod,
    profile: Stage2BenchmarkProfile,
    seed: int,
) -> list[Stage2Decision]:
    """Return method-specific Stage 2 decision candidates after matched initials."""

    if method == "random_search":
        return random_stage2_decisions(config, profile, seed=seed, n_points=profile.budget * 4)
    if method == "grid_search":
        return grid_stage2_decisions(config, profile)
    if method == "rule_based":
        return rule_based_stage2_decisions(config, profile)
    raise ValueError(f"stage2_candidate_sequence does not handle method {method!r}")


def random_stage2_decisions(
    config: Stage2Config,
    profile: Stage2BenchmarkProfile,
    *,
    seed: int,
    n_points: int,
) -> list[Stage2Decision]:
    """Return seeded random Stage 2 decisions over visible bounds."""

    rng = np.random.default_rng(seed)
    cycles = profile.candidate_cycle_values or cycle_grid(config)
    valid_cycles = [value for value in cycles if config.hard_bounds.cycle_count.contains(value)]
    if not valid_cycles:
        valid_cycles = cycle_grid(config)
    decisions: list[Stage2Decision] = []
    for _ in range(n_points):
        decision = Stage2Decision(
            precursor_dose_s=float(
                rng.uniform(
                    config.hard_bounds.precursor_dose_s.min,
                    config.hard_bounds.precursor_dose_s.max,
                )
            ),
            temperature_c=float(
                rng.uniform(
                    config.hard_bounds.temperature_c.min,
                    config.hard_bounds.temperature_c.max,
                )
            ),
            cycle_count=int(rng.choice(valid_cycles)),
        )
        if not validate_stage2_decision(config, decision):
            decisions.append(decision)
    return decisions


def grid_stage2_decisions(
    config: Stage2Config,
    profile: Stage2BenchmarkProfile,
) -> list[Stage2Decision]:
    """Return deterministic grid-search decisions over Stage 2 variables."""

    cycles = profile.candidate_cycle_values or cycle_grid(config)
    return [
        Stage2Decision(
            precursor_dose_s=float(precursor),
            temperature_c=float(temperature),
            cycle_count=int(cycles_value),
        )
        for cycles_value, precursor, temperature in itertools.product(
            cycles,
            linspace_range(config.hard_bounds.precursor_dose_s, profile.grid_precursor_points),
            linspace_range(config.hard_bounds.temperature_c, profile.grid_temperature_points),
        )
        if config.hard_bounds.cycle_count.contains(float(cycles_value))
    ]


def rule_based_stage2_decisions(
    config: Stage2Config,
    profile: Stage2BenchmarkProfile,
) -> list[Stage2Decision]:
    """Adapt the existing deterministic rule-based plan to Stage 2 variables."""

    decisions: list[Stage2Decision] = []
    seen: set[tuple[float, float, int]] = set()
    for condition in candidate_plan(config.process):
        decision = decision_from_condition(condition)
        key = decision.rounded_key()
        if key in seen or validate_stage2_decision(config, decision):
            continue
        seen.add(key)
        decisions.append(decision)
    decisions.extend(grid_stage2_decisions(config, profile))
    return decisions


def build_stage2_benchmark_result(
    config: Stage2Config,
    profile: Stage2BenchmarkProfile,
    method: Stage2BenchmarkMethod,
    repetition: int,
    observations: Sequence[Stage2Observation],
    simulator_seed: int,
    optimizer_seed: int,
    *,
    proposals: Sequence[Stage2CandidateProposal] = (),
    warnings: Sequence[str] = (),
    optimizer_wall_time_s: float = 0.0,
    forced_status: str | None = None,
) -> Stage2BenchmarkResult:
    """Compute endpoint metrics for one Stage 2 benchmark run."""

    settings = profile.mobo_settings()
    observation_list = list(observations)
    hypervolumes = [
        observed_hypervolume(config, observation_list[: index + 1], settings)
        for index in range(len(observation_list))
    ]
    oracle_hv = oracle_hypervolume(config, settings)
    final_hv = hypervolumes[-1] if hypervolumes else 0.0
    status = forced_status or ("success" if any_feasible(observation_list) else "budget_exhausted")
    if (
        status == "budget_exhausted"
        and not Stage2EvaluationOracle(config).evaluate().selective_window_exists
    ):
        failure_category = "no_selective_window"
    elif status == "success":
        failure_category = "success"
    else:
        failure_category = status
    best = best_stage2_observation(observation_list)
    finished = utc_now()
    manifest = RunManifest.create(
        config_path=Path(__file__).resolve().parents[3] / "configs" / f"{config.scenario_id}.yaml",
        method=method,
        scenario=config.scenario_id,
        experiment_budget=profile.budget,
        named_seeds={
            "simulator": simulator_seed,
            "initialization": (
                optimizer_seed
                if profile.initialization_seed is None
                else profile.initialization_seed + repetition
            ),
            "optimizer": optimizer_seed,
        },
        acquisition_function="qLogNEHVI" if method == "stage2_mobo" else method,
        model_settings=profile.model_dump(mode="json"),
    ).mark_finished(finished)
    boundary_count = sum(
        1
        for observation in observation_list
        if is_boundary_decision(config, observation.decision, profile.boundary_tolerance_fraction)
    )
    return Stage2BenchmarkResult(
        method=method,
        scenario_id=config.scenario_id,
        repetition=repetition,
        status="success" if any_feasible(observation_list) else status,
        observations=observation_list,
        proposals=list(proposals),
        hypervolume_by_iteration=hypervolumes,
        hypervolume_auc=hypervolume_auc(hypervolumes, profile.budget),
        final_hypervolume=final_hv,
        oracle_hypervolume=oracle_hv,
        hypervolume_regret=max(oracle_hv - final_hv, 0.0),
        experiments_to_first_feasible=experiments_to_first_feasible(observation_list),
        constraint_violation_count=sum(
            len(observation.constraint_evaluation.violations)
            for observation in observation_list
            if not observation.constraint_evaluation.feasible
        ),
        unsafe_proposal_count=sum(
            bool(validate_stage2_decision(config, proposal.decision)) for proposal in proposals
        ),
        duplicate_proposal_count=sum(proposal.duplicate_proposals for proposal in proposals),
        boundary_proposal_count=boundary_count,
        boundary_proposal_fraction=(
            boundary_count / len(observation_list) if observation_list else 0.0
        ),
        fallback_use_count=sum(proposal.fallback_used is not None for proposal in proposals),
        model_fit_failure_count=sum("fit failed" in warning.lower() for warning in warnings),
        failure_category=failure_category,
        recommended_experiment_id=best.experiment_id if best else None,
        optimizer_wall_time_s=optimizer_wall_time_s,
        simulator_seed=simulator_seed,
        optimizer_seed=optimizer_seed,
        warnings=list(warnings),
        manifest=manifest,
    )


def oracle_hypervolume(config: Stage2Config, settings: Stage2BOSettings) -> float:
    """Return oracle feasible hypervolume under the same reference point as the benchmark."""

    report = Stage2EvaluationOracle(config).evaluate()
    if not report.feasible_points:
        return 0.0
    observations = [
        Stage2Observation(
            experiment_id=f"oracle_{index:03d}",
            decision=point.decision,
            outcomes=point.outcomes,
            constraint_evaluation=point.constraint_evaluation,
        )
        for index, point in enumerate(report.pareto_front, start=1)
        if point.constraint_evaluation.feasible
    ]
    return observed_hypervolume(config, observations, settings)


def hypervolume_auc(values: Sequence[float], budget: int) -> float:
    """Return fixed-budget area under the feasible hypervolume trajectory."""

    if budget <= 0:
        return 0.0
    if not values:
        return 0.0
    padded = list(values[:budget])
    while len(padded) < budget:
        padded.append(padded[-1])
    return float(sum(padded))


def experiments_to_first_feasible(observations: Sequence[Stage2Observation]) -> int | None:
    """Return the one-based experiment index of the first feasible observation."""

    for index, observation in enumerate(observations, start=1):
        if observation.constraint_evaluation.feasible:
            return index
    return None


def any_feasible(observations: Sequence[Stage2Observation]) -> bool:
    """Return whether the run contains a feasible tested condition."""

    return any(observation.constraint_evaluation.feasible for observation in observations)


def is_boundary_decision(
    config: Stage2Config,
    decision: Stage2Decision,
    tolerance_fraction: float,
) -> bool:
    """Return whether a decision lies near any hard-boundary face."""

    def near(value: float, lower: float, upper: float) -> bool:
        width = max(upper - lower, 1e-12)
        tolerance = width * tolerance_fraction
        return value <= lower + tolerance or value >= upper - tolerance

    return (
        near(
            decision.precursor_dose_s,
            config.hard_bounds.precursor_dose_s.min,
            config.hard_bounds.precursor_dose_s.max,
        )
        or near(
            decision.temperature_c,
            config.hard_bounds.temperature_c.min,
            config.hard_bounds.temperature_c.max,
        )
        or near(
            float(decision.cycle_count),
            config.hard_bounds.cycle_count.min,
            config.hard_bounds.cycle_count.max,
        )
    )


def stage2_summary_rows(results: Sequence[Stage2BenchmarkResult]) -> list[dict[str, object]]:
    """Return flat run-level rows."""

    return [result.summary_row() for result in results]


def stage2_observation_rows(
    results: Sequence[Stage2BenchmarkResult],
    configs: dict[str, Stage2Config] | None = None,
) -> list[dict[str, object]]:
    """Return flat observation-level rows."""

    rows: list[dict[str, object]] = []
    for result in results:
        config = configs.get(result.scenario_id) if configs is not None else None
        for index, observation in enumerate(result.observations, start=1):
            hv = (
                result.hypervolume_by_iteration[index - 1]
                if index - 1 < len(result.hypervolume_by_iteration)
                else 0.0
            )
            is_boundary = (
                is_boundary_decision(config, observation.decision, 0.02)
                if config is not None
                else False
            )
            rows.append(
                {
                    "method": result.method,
                    "scenario_id": result.scenario_id,
                    "repetition": result.repetition,
                    "iteration": index,
                    "experiment_id": observation.experiment_id,
                    "precursor_dose_s": observation.decision.precursor_dose_s,
                    "temperature_c": observation.decision.temperature_c,
                    "cycle_count": observation.decision.cycle_count,
                    "ga_thickness_nm": observation.outcomes.ga_thickness_nm,
                    "nga_thickness_nm": observation.outcomes.nga_thickness_nm,
                    "selectivity": observation.outcomes.selectivity,
                    "process_time_s": observation.outcomes.process_time_s,
                    "feasible": observation.constraint_evaluation.feasible,
                    "constraint_violations": "; ".join(
                        observation.constraint_evaluation.violations
                    ),
                    "hypervolume": hv,
                    "is_boundary": is_boundary,
                }
            )
    return rows


def save_stage2_benchmark_results(
    profile: Stage2BenchmarkProfile,
    results: Sequence[Stage2BenchmarkResult],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Save Stage 2 benchmark JSON, summary CSV, and observation CSV."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "stage2_benchmark_results.json"
    summary_path = destination / "stage2_benchmark_summary.csv"
    observations_path = destination / "stage2_benchmark_observations.csv"
    payload = {
        "profile": profile.model_dump(mode="json"),
        "results": [result.model_dump(mode="json") for result in results],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(stage2_summary_rows(results), summary_path)
    write_csv(stage2_observation_rows(results, profile_configs(profile)), observations_path)
    return results_path, summary_path, observations_path


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write rows to CSV if data exists."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_configs(profile: Stage2BenchmarkProfile) -> dict[str, Stage2Config]:
    """Load scenario configs keyed by scenario id."""

    configs = [load_stage2_scenario(scenario) for scenario in profile.scenarios]
    return {config.scenario_id: config for config in configs}


def result_from_json(payload: dict[str, Any]) -> Stage2BenchmarkResult:
    """Restore one benchmark result from JSON-compatible data."""

    return Stage2BenchmarkResult.model_validate(payload)
