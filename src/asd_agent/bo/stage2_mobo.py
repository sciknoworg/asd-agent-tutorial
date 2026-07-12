"""Constrained multi-objective Bayesian optimization for Stage 2 ASD problems."""

from __future__ import annotations

import csv
import json
import math
import warnings as warning_capture
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import gpytorch
import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.quasirandom import SobolEngine

from asd_agent.bo.records import OptimizerState, RunManifest, utc_now
from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2ConstraintEvaluation,
    Stage2Decision,
    Stage2Outcomes,
    cycle_grid,
    decision_from_condition,
    evaluate_stage2_constraints,
    simulate_stage2,
    stage2_process_time,
    validate_stage2_decision,
)
from asd_agent.models import ExperimentRecord

try:
    from botorch.acquisition.multi_objective.logei import (  # type: ignore[attr-defined]
        qLogNoisyExpectedHypervolumeImprovement,
    )

    NEHVI_CLASS: Any = qLogNoisyExpectedHypervolumeImprovement
    ACQUISITION_NAME = "qLogNoisyExpectedHypervolumeImprovement"
except ImportError:  # pragma: no cover - only used on older BoTorch releases
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )

    NEHVI_CLASS = qNoisyExpectedHypervolumeImprovement
    ACQUISITION_NAME = "qNoisyExpectedHypervolumeImprovement"


Stage2BONoiseMode = Literal["known", "inferred"]
Stage2BOProposalStatus = Literal[
    "proposed",
    "no_valid_candidate",
    "model_fit_failure",
    "numerical_failure",
]
Stage2BORunStatus = Literal[
    "success",
    "budget_exhausted",
    "no_feasible_candidate",
    "model_fit_failure",
    "numerical_failure",
]

MODEL_OUTCOME_NAMES = [
    "ga_thickness_nm",
    "negative_nga_thickness_nm",
    "selectivity",
    "negative_process_time_s",
]
OBJECTIVE_OUTCOME_INDICES = [0, 1, 3]


class Stage2ModelFitFailure(RuntimeError):
    """Raised when the Stage 2 GP ensemble cannot be fit."""


class Stage2BOSettings(BaseModel):
    """Settings for constrained noisy Stage 2 MOBO."""

    experiment_budget: int = Field(default=8, ge=1)
    initial_design_size: int = Field(default=4, ge=2)
    noise_mode: Stage2BONoiseMode = "inferred"
    known_noise_variances: dict[str, float] = Field(default_factory=dict)
    min_known_noise_variance: float = Field(default=1e-9, gt=0.0)
    max_fit_attempts: int = Field(default=2, ge=1, le=5)
    initial_jitter: float = Field(default=1e-6, gt=0.0)
    jitter_multiplier: float = Field(default=10.0, ge=1.0)
    qmc_samples: int = Field(default=64, ge=4)
    num_restarts: int = Field(default=4, ge=1)
    raw_samples: int = Field(default=64, ge=4)
    acquisition_timeout_s: float | None = Field(default=10.0, gt=0.0)
    candidate_cycle_values: list[int] = Field(default_factory=list)
    min_candidate_distance: float = Field(default=1e-4, ge=0.0)
    duplicate_tolerance: float = Field(default=1e-6, ge=0.0)
    random_fallback_points: int = Field(default=128, ge=1)
    allow_random_fallback: bool = True
    reference_point: list[float] | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_reference_point(self) -> Stage2BOSettings:
        if self.reference_point is not None and len(self.reference_point) != 3:
            raise ValueError("reference_point must contain three objective-space values")
        if self.reference_point is not None and not all(
            math.isfinite(value) for value in self.reference_point
        ):
            raise ValueError("reference_point values must be finite")
        return self


class Stage2Observation(BaseModel):
    """Optimizer-facing Stage 2 experiment record."""

    experiment_id: str = Field(min_length=1)
    decision: Stage2Decision
    outcomes: Stage2Outcomes
    constraint_evaluation: Stage2ConstraintEvaluation
    seed: int | None = None
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_experiment_record(
        cls,
        config: Stage2Config,
        record: ExperimentRecord,
        *,
        seed: int | None = None,
    ) -> Stage2Observation:
        """Build a Stage 2 observation from a legacy experiment-ledger row."""

        decision = decision_from_condition(record.condition)
        outcomes = Stage2Outcomes(
            ga_thickness_nm=record.ga_thickness_nm,
            nga_thickness_nm=record.nga_thickness_nm,
            selectivity=record.selectivity,
            process_time_s=record.process_time_s,
        )
        return cls(
            experiment_id=record.experiment_id,
            decision=decision,
            outcomes=outcomes,
            constraint_evaluation=evaluate_stage2_constraints(config, decision, outcomes),
            seed=seed,
            timestamp=record.timestamp,
        )

    def optimizer_payload(self) -> dict[str, object]:
        """Return visible observation fields without hidden simulator parameters."""

        return {
            "experiment_id": self.experiment_id,
            "decision": self.decision.model_dump(mode="json"),
            "outcomes": self.outcomes.model_dump(mode="json"),
            "feasible": self.constraint_evaluation.feasible,
            "constraint_violations": list(self.constraint_evaluation.violations),
            "timestamp": self.timestamp,
        }


class Stage2CandidateProposal(BaseModel):
    """One constrained MOBO candidate proposal."""

    candidate_id: str = Field(min_length=1)
    decision: Stage2Decision
    optimizer: str = Field(min_length=1)
    acquisition_value: float | None = None
    feasibility_probability: float | None = None
    posterior_summaries: dict[str, float] = Field(default_factory=dict)
    training_observation_ids: list[str] = Field(default_factory=list)
    constraint_violations: list[str] = Field(default_factory=list)
    duplicate_proposals: int = 0
    fallback_used: str | None = None
    optimizer_wall_time_s: float = Field(default=0.0, ge=0.0)
    seed: int | None = None
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        *,
        decision: Stage2Decision,
        optimizer: str,
        acquisition_value: float | None = None,
        feasibility_probability: float | None = None,
        posterior_summaries: dict[str, float] | None = None,
        training_observation_ids: Sequence[str] = (),
        constraint_violations: Sequence[str] = (),
        duplicate_proposals: int = 0,
        fallback_used: str | None = None,
        optimizer_wall_time_s: float = 0.0,
        seed: int | None = None,
    ) -> Stage2CandidateProposal:
        """Create a Stage 2 proposal with a unique identifier."""

        safe_optimizer = "".join(char if char.isalnum() else "_" for char in optimizer.lower())
        return cls(
            candidate_id=f"stage2_cand_{safe_optimizer}_{uuid4().hex[:12]}",
            decision=decision,
            optimizer=optimizer,
            acquisition_value=acquisition_value,
            feasibility_probability=feasibility_probability,
            posterior_summaries=posterior_summaries or {},
            training_observation_ids=list(training_observation_ids),
            constraint_violations=list(constraint_violations),
            duplicate_proposals=duplicate_proposals,
            fallback_used=fallback_used,
            optimizer_wall_time_s=optimizer_wall_time_s,
            seed=seed,
        )


class Stage2BOProposalResult(BaseModel):
    """Result of one MOBO proposal attempt."""

    status: Stage2BOProposalStatus
    proposal: Stage2CandidateProposal | None = None
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Stage2BOResult(BaseModel):
    """Serializable result from one Stage 2 MOBO run."""

    method: Literal["stage2_mobo"] = "stage2_mobo"
    scenario_id: str
    status: Stage2BORunStatus
    observations: list[Stage2Observation]
    proposals: list[Stage2CandidateProposal] = Field(default_factory=list)
    recommended_experiment_id: str | None = None
    hypervolume_by_iteration: list[float] = Field(default_factory=list)
    failure_category: Stage2BORunStatus
    optimizer_wall_time_s: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    optimizer_state: OptimizerState
    manifest: RunManifest
    started_at: str = Field(default_factory=utc_now)
    finished_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    def summary_row(self) -> dict[str, object]:
        """Return a compact benchmark row."""

        best = best_stage2_observation(self.observations)
        return {
            "method": self.method,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "failure_category": self.failure_category,
            "n_experiments": len(self.observations),
            "success": self.status == "success",
            "recommended_experiment_id": self.recommended_experiment_id or "",
            "best_selectivity": best.outcomes.selectivity if best else 0.0,
            "ga_thickness_nm": best.outcomes.ga_thickness_nm if best else 0.0,
            "nga_thickness_nm": best.outcomes.nga_thickness_nm if best else 0.0,
            "total_process_time_s": sum(
                observation.outcomes.process_time_s for observation in self.observations
            ),
            "final_hypervolume": self.hypervolume_by_iteration[-1]
            if self.hypervolume_by_iteration
            else 0.0,
            "fallback_uses": sum(
                1 for proposal in self.proposals if proposal.fallback_used is not None
            ),
            "duplicate_proposals": sum(proposal.duplicate_proposals for proposal in self.proposals),
            "optimizer_wall_time_s": self.optimizer_wall_time_s,
        }


@dataclass(frozen=True)
class Stage2Surrogate:
    """Fitted one-GP-per-outcome Stage 2 surrogate."""

    model: ModelListGP
    train_x: torch.Tensor
    train_y: torch.Tensor
    settings: Stage2BOSettings
    fit_warnings: list[str]
    training_observation_ids: list[str]

    def posterior_summary(
        self,
        config: Stage2Config,
        decision: Stage2Decision,
    ) -> tuple[dict[str, float], float]:
        """Return posterior means, standard deviations, and feasibility probability."""

        query = stage2_input_tensor([decision])
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(query)
        mean = posterior.mean.detach().cpu().reshape(-1)
        variance = posterior.variance.clamp_min(0.0).detach().cpu().reshape(-1)
        stddev = torch.sqrt(variance)
        summary: dict[str, float] = {}
        for index, name in enumerate(MODEL_OUTCOME_NAMES):
            summary[f"{name}_mean"] = float(mean[index])
            summary[f"{name}_stddev"] = float(stddev[index])
        ga_mean = max(float(mean[0]), 0.0)
        nga_mean = max(-float(mean[1]), 0.0)
        total_mean = ga_mean + nga_mean
        derived_selectivity = 0.0 if total_mean <= 1e-12 else (ga_mean - nga_mean) / total_mean
        summary["selectivity_from_thickness_means"] = derived_selectivity
        summary["selectivity_consistency_error"] = abs(float(mean[2]) - derived_selectivity)
        feasibility = feasibility_probability(
            config,
            [float(value) for value in mean.tolist()],
            [float(value) for value in stddev.tolist()],
        )
        return summary, feasibility


class Stage2ConstrainedMOBOOptimizer:
    """Constrained noisy multi-objective Bayesian optimizer for Stage 2."""

    name = "stage2_mobo"

    def __init__(
        self,
        config: Stage2Config,
        settings: Stage2BOSettings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or Stage2BOSettings()
        self.seed = seed
        self.proposals: list[Stage2CandidateProposal] = []
        self.warnings: list[str] = []
        self.fallback_log: list[str] = []

    def propose(self, observations: Sequence[Stage2Observation]) -> Stage2BOProposalResult:
        """Fit the surrogate and propose a safe, nonduplicate Stage 2 candidate."""

        started = perf_counter()
        if len(observations) < 2:
            return self._fallback_proposal(
                observations,
                started,
                "insufficient_observations",
                ["MOBO needs at least two observations before fitting GPs."],
            )

        try:
            surrogate = fit_stage2_surrogate(
                self.config,
                observations,
                self.settings,
                seed=self.seed,
            )
            acquisition = build_stage2_acquisition(
                self.config,
                surrogate,
                self.settings,
                seed=self.seed,
            )
        except (Stage2ModelFitFailure, ValueError) as exc:
            warning = f"Stage 2 MOBO fit failed: {exc}"
            self.warnings.append(warning)
            return self._fallback_proposal(observations, started, "model_fit_failure", [warning])
        except Exception as exc:  # pragma: no cover - defensive numerical guard
            warning = f"Stage 2 acquisition setup failed: {type(exc).__name__}: {exc}"
            self.warnings.append(warning)
            return self._fallback_proposal(observations, started, "numerical_failure", [warning])

        warnings = list(surrogate.fit_warnings)
        best: tuple[Stage2Decision, float] | None = None
        duplicate_count = 0
        optimization_failures: list[str] = []
        for cycle_count in candidate_cycle_values(self.config, self.settings):
            try:
                decision, value = optimize_for_cycle_count(
                    self.config,
                    acquisition,
                    cycle_count,
                    self.settings,
                    seed=None if self.seed is None else self.seed + int(cycle_count),
                )
            except Exception as exc:  # pragma: no cover - depends on optimizer warnings
                optimization_failures.append(f"cycle {cycle_count}: {type(exc).__name__}: {exc}")
                continue
            safety_violations = validate_stage2_decision(self.config, decision)
            if safety_violations:
                continue
            if duplicate_or_too_close(
                decision,
                observations,
                self.proposals,
                self.config,
                self.settings,
            ):
                duplicate_count += 1
                continue
            if not math.isfinite(value):
                optimization_failures.append(f"cycle {cycle_count}: non-finite acquisition")
                continue
            if best is None or value > best[1]:
                best = (decision, value)

        if best is None:
            warnings.extend(optimization_failures)
            return self._fallback_proposal(
                observations,
                started,
                "random_feasible",
                warnings or ["No valid nonduplicate acquisition candidate remained."],
                duplicate_count=duplicate_count,
                training_observation_ids=surrogate.training_observation_ids,
                surrogate=surrogate,
            )

        decision, acquisition_value = best
        summary, feasibility = surrogate.posterior_summary(self.config, decision)
        proposal = Stage2CandidateProposal.create(
            decision=decision,
            optimizer=self.name,
            acquisition_value=acquisition_value,
            feasibility_probability=feasibility,
            posterior_summaries=summary,
            training_observation_ids=surrogate.training_observation_ids,
            constraint_violations=validate_stage2_decision(self.config, decision),
            duplicate_proposals=duplicate_count,
            optimizer_wall_time_s=perf_counter() - started,
            seed=self.seed,
        )
        self.proposals.append(proposal)
        self.warnings.extend(warnings)
        return Stage2BOProposalResult(status="proposed", proposal=proposal, warnings=warnings)

    def get_state(self) -> OptimizerState:
        """Return a serializable optimizer checkpoint."""

        return OptimizerState(
            optimizer=self.name,
            state={
                "settings": self.settings.model_dump(mode="json"),
                "proposals": [proposal.model_dump(mode="json") for proposal in self.proposals],
                "warnings": list(self.warnings),
                "fallback_log": list(self.fallback_log),
            },
            observation_ids=[],
            random_seed=self.seed,
        )

    def restore_state(self, state: OptimizerState) -> None:
        """Restore proposal history and warnings."""

        proposals = state.state.get("proposals", [])
        if isinstance(proposals, list):
            self.proposals = [
                Stage2CandidateProposal.model_validate(item)
                for item in proposals
                if isinstance(item, dict)
            ]
        warnings = state.state.get("warnings", [])
        self.warnings = [str(item) for item in warnings] if isinstance(warnings, list) else []
        fallback_log = state.state.get("fallback_log", [])
        self.fallback_log = (
            [str(item) for item in fallback_log] if isinstance(fallback_log, list) else []
        )

    def _fallback_proposal(
        self,
        observations: Sequence[Stage2Observation],
        started: float,
        fallback_reason: str,
        warnings: Sequence[str],
        *,
        duplicate_count: int = 0,
        training_observation_ids: Sequence[str] = (),
        surrogate: Stage2Surrogate | None = None,
    ) -> Stage2BOProposalResult:
        if not self.settings.allow_random_fallback:
            mapped_status: Stage2BOProposalStatus = (
                "model_fit_failure"
                if fallback_reason == "model_fit_failure"
                else "numerical_failure"
            )
            return Stage2BOProposalResult(status=mapped_status, warnings=list(warnings))
        decision = random_safe_fallback_decision(
            self.config,
            self.settings,
            observations,
            self.proposals,
            seed=self.seed,
        )
        if decision is None:
            return Stage2BOProposalResult(
                status="no_valid_candidate",
                warnings=list(warnings) + ["Random fallback found no valid safe candidate."],
            )
        posterior_summary: dict[str, float] = {}
        feasibility: float | None = None
        if surrogate is not None:
            posterior_summary, feasibility = surrogate.posterior_summary(self.config, decision)
        proposal = Stage2CandidateProposal.create(
            decision=decision,
            optimizer=self.name,
            feasibility_probability=feasibility,
            posterior_summaries=posterior_summary,
            training_observation_ids=training_observation_ids,
            constraint_violations=validate_stage2_decision(self.config, decision),
            duplicate_proposals=duplicate_count,
            fallback_used=fallback_reason,
            optimizer_wall_time_s=perf_counter() - started,
            seed=self.seed,
        )
        self.proposals.append(proposal)
        self.fallback_log.append(fallback_reason)
        self.warnings.extend(warnings)
        return Stage2BOProposalResult(status="proposed", proposal=proposal, warnings=list(warnings))


def fit_stage2_surrogate(
    config: Stage2Config,
    observations: Sequence[Stage2Observation],
    settings: Stage2BOSettings | None = None,
    *,
    seed: int | None = None,
) -> Stage2Surrogate:
    """Fit a float64 ModelListGP with one GP per measured outcome."""

    resolved_settings = settings or Stage2BOSettings()
    train_x, train_y = stage2_training_tensors(observations)
    train_yvar = known_noise_tensor(config, train_y, resolved_settings)
    fit_warnings: list[str] = []
    last_error: Exception | None = None

    for attempt in range(resolved_settings.max_fit_attempts):
        if seed is not None:
            torch.manual_seed(seed + attempt)
        model = build_stage2_model(config, train_x, train_y, train_yvar)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        jitter = resolved_settings.initial_jitter * (resolved_settings.jitter_multiplier**attempt)
        try:
            with (
                gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter),
                warning_capture.catch_warnings(record=True) as caught,
            ):
                warning_capture.simplefilter("always")
                fit_gpytorch_mll(mll)
            fit_warnings.extend(
                f"attempt {attempt + 1}: {item.category.__name__}: {item.message}"
                for item in caught
            )
            model.eval()
            return Stage2Surrogate(
                model=model,
                train_x=train_x,
                train_y=train_y,
                settings=resolved_settings,
                fit_warnings=fit_warnings,
                training_observation_ids=[
                    observation.experiment_id for observation in observations
                ],
            )
        except Exception as exc:  # pragma: no cover - exercised by monkeypatch tests
            last_error = exc
            fit_warnings.append(f"attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")

    message = "Stage 2 ModelListGP fitting failed"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise Stage2ModelFitFailure(message)


def build_stage2_model(
    config: Stage2Config,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
) -> ModelListGP:
    """Build one normalized-input, standardized-output Matern GP per outcome."""

    bounds = stage2_bounds_tensor(config)
    models: list[SingleTaskGP] = []
    for outcome_index in range(train_y.shape[-1]):
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        yvar = train_yvar[:, outcome_index : outcome_index + 1] if train_yvar is not None else None
        models.append(
            SingleTaskGP(
                train_X=train_x,
                train_Y=train_y[:, outcome_index : outcome_index + 1],
                train_Yvar=yvar,
                covar_module=covar_module,
                input_transform=Normalize(d=3, bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
        )
    return ModelListGP(*models)


def build_stage2_acquisition(
    config: Stage2Config,
    surrogate: Stage2Surrogate,
    settings: Stage2BOSettings | None = None,
    *,
    seed: int | None = None,
) -> Any:
    """Build constrained log-NEHVI, falling back to qNEHVI on older BoTorch."""

    resolved_settings = settings or surrogate.settings
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([resolved_settings.qmc_samples]),
        seed=seed,
    )
    objective = IdentityMCMultiOutputObjective(outcomes=OBJECTIVE_OUTCOME_INDICES)
    return NEHVI_CLASS(
        model=surrogate.model,
        ref_point=stage2_reference_point(config, resolved_settings),
        X_baseline=surrogate.train_x,
        sampler=sampler,
        objective=objective,
        constraints=stage2_outcome_constraints(config),
        prune_baseline=False,
    )


def optimize_for_cycle_count(
    config: Stage2Config,
    acquisition: Any,
    cycle_count: int,
    settings: Stage2BOSettings,
    *,
    seed: int | None = None,
) -> tuple[Stage2Decision, float]:
    """Optimize continuous variables while holding the integer cycle count fixed."""

    if seed is not None:
        torch.manual_seed(seed)
    candidates, values = optimize_acqf(
        acquisition,
        bounds=stage2_bounds_tensor(config),
        q=1,
        num_restarts=settings.num_restarts,
        raw_samples=settings.raw_samples,
        fixed_features={2: float(cycle_count)},
        options={"batch_limit": 4, "maxiter": 50},
        timeout_sec=settings.acquisition_timeout_s,
    )
    flat = candidates.detach().cpu().reshape(-1, 3)[0]
    value_tensor = values.detach().cpu().reshape(-1)[0] if values is not None else torch.nan
    decision = Stage2Decision(
        precursor_dose_s=clamp_float(
            float(flat[0]),
            config.hard_bounds.precursor_dose_s.min,
            config.hard_bounds.precursor_dose_s.max,
        ),
        temperature_c=clamp_float(
            float(flat[1]),
            config.hard_bounds.temperature_c.min,
            config.hard_bounds.temperature_c.max,
        ),
        cycle_count=int(cycle_count),
    )
    return decision, float(value_tensor)


def evaluate_stage2_acquisition(
    acquisition: Any,
    decision: Stage2Decision,
) -> float:
    """Evaluate an acquisition function at one explicit Stage 2 decision."""

    with torch.no_grad():
        value = acquisition(stage2_input_tensor([decision]).unsqueeze(0))
    return float(value.detach().cpu().reshape(-1)[0])


def run_stage2_bo(
    config: Stage2Config,
    settings: Stage2BOSettings | None = None,
    *,
    simulator_seed: int | None = None,
    optimizer_seed: int | None = None,
    initial_observations: Sequence[Stage2Observation] = (),
    optimizer_state: OptimizerState | None = None,
) -> Stage2BOResult:
    """Run one constrained Stage 2 MOBO loop on a virtual ASD scenario."""

    resolved_settings = settings or Stage2BOSettings()
    started_at = utc_now()
    wall_start = perf_counter()
    observations = list(initial_observations)
    resolved_simulator_seed = config.process.seed if simulator_seed is None else simulator_seed
    resolved_optimizer_seed = 0 if optimizer_seed is None else optimizer_seed
    manifest = RunManifest.create(
        config_path=Path(__file__).resolve().parents[3] / "configs" / f"{config.scenario_id}.yaml",
        method="stage2_mobo",
        scenario=config.scenario_id,
        experiment_budget=resolved_settings.experiment_budget,
        named_seeds={
            "simulator": resolved_simulator_seed,
            "measurement_noise": resolved_simulator_seed,
            "optimizer": resolved_optimizer_seed,
        },
        acquisition_function=ACQUISITION_NAME,
        model_settings=resolved_settings.model_dump(mode="json"),
        started_at=started_at,
    )
    warnings: list[str] = []
    hypervolume_by_iteration = [
        observed_hypervolume(config, observations[: index + 1], resolved_settings)
        for index in range(len(observations))
    ]

    initial_decisions = sobol_initial_decisions(
        config,
        resolved_settings.initial_design_size,
        seed=optimizer_seed,
        cycle_values=candidate_cycle_values(config, resolved_settings),
    )
    for decision in initial_decisions:
        if len(observations) >= min(
            resolved_settings.initial_design_size,
            resolved_settings.experiment_budget,
        ):
            break
        if duplicate_or_too_close(decision, observations, [], config, resolved_settings):
            continue
        observations.append(
            observe_stage2_decision(
                config,
                decision,
                experiment_id=stage2_experiment_id("mobo_initial", len(observations) + 1),
                seed=None if simulator_seed is None else simulator_seed + len(observations),
            )
        )
        hypervolume_by_iteration.append(
            observed_hypervolume(config, observations, resolved_settings)
        )

    optimizer = Stage2ConstrainedMOBOOptimizer(
        config,
        resolved_settings,
        seed=optimizer_seed,
    )
    if optimizer_state is not None:
        if optimizer_state.optimizer != optimizer.name:
            raise ValueError(
                f"optimizer state belongs to {optimizer_state.optimizer!r}, not {optimizer.name!r}"
            )
        optimizer.restore_state(optimizer_state)
    status: Stage2BORunStatus | None = None
    while len(observations) < resolved_settings.experiment_budget:
        proposal_result = optimizer.propose(observations)
        warnings.extend(proposal_result.warnings)
        if proposal_result.status != "proposed" or proposal_result.proposal is None:
            status = proposal_status_to_run_status(proposal_result.status)
            break
        proposal = proposal_result.proposal
        observations.append(
            observe_stage2_decision(
                config,
                proposal.decision,
                experiment_id=stage2_experiment_id("mobo", len(observations) + 1),
                seed=None if simulator_seed is None else simulator_seed + len(observations),
            )
        )
        hypervolume_by_iteration.append(
            observed_hypervolume(config, observations, resolved_settings)
        )

    recommended = best_stage2_observation(observations)
    if status is None:
        status = "success" if any_feasible(observations) else "budget_exhausted"
    if status == "budget_exhausted" and not observations:
        status = "no_feasible_candidate"

    finished_at = utc_now()
    saved_state = optimizer.get_state().model_copy(
        update={"observation_ids": [observation.experiment_id for observation in observations]}
    )
    return Stage2BOResult(
        scenario_id=config.scenario_id,
        status=status,
        observations=observations,
        proposals=list(optimizer.proposals),
        recommended_experiment_id=recommended.experiment_id if recommended else None,
        hypervolume_by_iteration=hypervolume_by_iteration,
        failure_category=status,
        optimizer_wall_time_s=perf_counter() - wall_start,
        warnings=warnings + optimizer.warnings,
        optimizer_state=saved_state,
        manifest=manifest.mark_finished(finished_at),
        started_at=started_at,
        finished_at=finished_at,
    )


def observe_stage2_decision(
    config: Stage2Config,
    decision: Stage2Decision,
    *,
    experiment_id: str,
    seed: int | None = None,
) -> Stage2Observation:
    """Run one safe Stage 2 decision through the measured simulator."""

    outcomes = simulate_stage2(config, decision, seed=seed)
    return Stage2Observation(
        experiment_id=experiment_id,
        decision=decision,
        outcomes=outcomes,
        constraint_evaluation=evaluate_stage2_constraints(config, decision, outcomes),
        seed=seed,
    )


def observations_from_experiment_records(
    config: Stage2Config,
    records: Sequence[ExperimentRecord],
) -> list[Stage2Observation]:
    """Convert legacy ASD ledger rows into Stage 2 optimizer observations."""

    return [Stage2Observation.from_experiment_record(config, record) for record in records]


def stage2_training_tensors(
    observations: Sequence[Stage2Observation],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert optimizer-facing Stage 2 observations into float64 tensors."""

    if len(observations) < 2:
        raise ValueError("Stage 2 MOBO requires at least two observations")
    train_x = stage2_input_tensor([observation.decision for observation in observations])
    train_y = torch.tensor(
        [transformed_outcome_vector(observation.outcomes) for observation in observations],
        dtype=torch.double,
    )
    return train_x, train_y


def stage2_input_tensor(decisions: Sequence[Stage2Decision]) -> torch.Tensor:
    """Return raw float64 input tensor for Stage 2 decisions."""

    return torch.tensor(
        [
            [
                decision.precursor_dose_s,
                decision.temperature_c,
                float(decision.cycle_count),
            ]
            for decision in decisions
        ],
        dtype=torch.double,
    )


def transformed_outcome_vector(outcomes: Stage2Outcomes) -> list[float]:
    """Return model outputs in maximization-friendly transformed coordinates."""

    return [
        outcomes.ga_thickness_nm,
        -outcomes.nga_thickness_nm,
        outcomes.selectivity,
        -outcomes.process_time_s,
    ]


def known_noise_tensor(
    config: Stage2Config,
    train_y: torch.Tensor,
    settings: Stage2BOSettings,
) -> torch.Tensor | None:
    """Return a known-noise tensor, or None for inferred-noise mode."""

    if settings.noise_mode == "inferred":
        return None
    variances = {
        "ga_thickness_nm": config.process.noise_sigma_nm**2,
        "negative_nga_thickness_nm": config.process.noise_sigma_nm**2,
        "selectivity": 1e-4,
        "negative_process_time_s": 1e-6,
    }
    variances.update(settings.known_noise_variances)
    values = [
        max(float(variances[name]), settings.min_known_noise_variance)
        for name in MODEL_OUTCOME_NAMES
    ]
    return torch.tensor([values for _ in range(train_y.shape[0])], dtype=torch.double)


def stage2_bounds_tensor(config: Stage2Config) -> torch.Tensor:
    """Return raw Stage 2 input bounds for BoTorch normalization and optimization."""

    return torch.tensor(
        [
            [
                config.hard_bounds.precursor_dose_s.min,
                config.hard_bounds.temperature_c.min,
                config.hard_bounds.cycle_count.min,
            ],
            [
                config.hard_bounds.precursor_dose_s.max,
                config.hard_bounds.temperature_c.max,
                config.hard_bounds.cycle_count.max,
            ],
        ],
        dtype=torch.double,
    )


def stage2_reference_point(config: Stage2Config, settings: Stage2BOSettings) -> list[float]:
    """Return the configured scientific reference point in objective coordinates."""

    if settings.reference_point is not None:
        configured = [float(value) for value in settings.reference_point]
        if configured[0] > config.constraints.ga_min_nm:
            raise ValueError("reference GA value must not exceed the feasible GA threshold")
        if configured[1] > -config.constraints.nga_max_nm:
            raise ValueError("reference negative-NGA value must be no better than feasibility")
        max_time = config.hard_bounds.max_process_time_s
        if max_time is not None and configured[2] > -max_time:
            raise ValueError("reference negative-time value must be no better than the hard limit")
        return configured
    max_time = config.hard_bounds.max_process_time_s
    if max_time is None:
        max_time = stage2_process_time(
            config,
            Stage2Decision(
                precursor_dose_s=config.hard_bounds.precursor_dose_s.max,
                temperature_c=config.hard_bounds.temperature_c.max,
                cycle_count=int(config.hard_bounds.cycle_count.max),
            ),
        )
    return [
        0.5 * config.constraints.ga_min_nm,
        -max(config.constraints.nga_max_nm * 4.0, config.constraints.nga_max_nm + 1.0),
        -float(max_time),
    ]


def stage2_outcome_constraints(
    config: Stage2Config,
) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    """Return BoTorch outcome constraints where values <= 0 are feasible."""

    return [
        lambda samples: config.constraints.ga_min_nm - samples[..., 0],
        lambda samples: -samples[..., 1] - config.constraints.nga_max_nm,
        lambda samples: config.constraints.selectivity_min - samples[..., 2],
    ]


def feasibility_probability(
    config: Stage2Config,
    posterior_mean: Sequence[float],
    posterior_stddev: Sequence[float],
) -> float:
    """Approximate feasibility probability from marginal normal posteriors."""

    if len(posterior_mean) < 3 or len(posterior_stddev) < 3:
        return 0.0
    probabilities = [
        normal_probability_at_least(
            posterior_mean[0],
            max(posterior_stddev[0], 0.0),
            config.constraints.ga_min_nm,
        ),
        normal_probability_at_least(
            posterior_mean[1],
            max(posterior_stddev[1], 0.0),
            -config.constraints.nga_max_nm,
        ),
        normal_probability_at_least(
            posterior_mean[2],
            max(posterior_stddev[2], 0.0),
            config.constraints.selectivity_min,
        ),
    ]
    product = 1.0
    for probability in probabilities:
        product *= probability
    return max(0.0, min(1.0, product))


def normal_probability_at_least(mean: float, stddev: float, threshold: float) -> float:
    """Return P(Y >= threshold) for a normal marginal."""

    if not all(math.isfinite(value) for value in (mean, stddev, threshold)):
        return 0.0
    if stddev <= 0.0:
        return 1.0 if mean >= threshold else 0.0
    z_value = (threshold - mean) / stddev
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def candidate_cycle_values(config: Stage2Config, settings: Stage2BOSettings) -> list[int]:
    """Return configured integer cycle counts for conditional acquisition optimization."""

    source = settings.candidate_cycle_values or cycle_grid(config)
    values = [
        int(value)
        for value in sorted(set(source))
        if config.hard_bounds.cycle_count.contains(float(value))
    ]
    if not values:
        lower = int(config.hard_bounds.cycle_count.min)
        upper = int(config.hard_bounds.cycle_count.max)
        values = [lower, int(round((lower + upper) / 2.0)), upper]
    return values


def duplicate_or_too_close(
    decision: Stage2Decision,
    observations: Sequence[Stage2Observation],
    proposals: Sequence[Stage2CandidateProposal],
    config: Stage2Config,
    settings: Stage2BOSettings,
) -> bool:
    """Return whether a decision duplicates or nearly duplicates prior work."""

    existing_decisions = [observation.decision for observation in observations]
    existing_decisions.extend(proposal.decision for proposal in proposals)
    if any(same_stage2_decision(decision, existing, settings) for existing in existing_decisions):
        return True
    return any(
        normalized_stage2_distance(config, decision, existing) <= settings.min_candidate_distance
        for existing in existing_decisions
    )


def same_stage2_decision(
    first: Stage2Decision,
    second: Stage2Decision,
    settings: Stage2BOSettings,
) -> bool:
    """Return whether two Stage 2 decisions are exact duplicates within tolerance."""

    return (
        abs(first.precursor_dose_s - second.precursor_dose_s) <= settings.duplicate_tolerance
        and abs(first.temperature_c - second.temperature_c) <= settings.duplicate_tolerance
        and first.cycle_count == second.cycle_count
    )


def normalized_stage2_distance(
    config: Stage2Config,
    first: Stage2Decision,
    second: Stage2Decision,
) -> float:
    """Return Euclidean distance in normalized Stage 2 coordinates."""

    spans = [
        max(
            config.hard_bounds.precursor_dose_s.max - config.hard_bounds.precursor_dose_s.min,
            1e-12,
        ),
        max(config.hard_bounds.temperature_c.max - config.hard_bounds.temperature_c.min, 1e-12),
        max(config.hard_bounds.cycle_count.max - config.hard_bounds.cycle_count.min, 1e-12),
    ]
    deltas = [
        (first.precursor_dose_s - second.precursor_dose_s) / spans[0],
        (first.temperature_c - second.temperature_c) / spans[1],
        (float(first.cycle_count) - float(second.cycle_count)) / spans[2],
    ]
    return math.sqrt(sum(delta * delta for delta in deltas))


def sobol_initial_decisions(
    config: Stage2Config,
    n_points: int,
    *,
    seed: int | None = None,
    cycle_values: Sequence[int] = (),
) -> list[Stage2Decision]:
    """Generate seeded Sobol initial Stage 2 decisions with integer cycles."""

    cycles = list(cycle_values) or candidate_cycle_values(config, Stage2BOSettings())
    engine = SobolEngine(dimension=3, scramble=True, seed=seed)  # type: ignore[no-untyped-call]
    decisions: list[Stage2Decision] = []
    seen: list[Stage2Decision] = []
    draws_needed = max(n_points * 4, n_points)
    for row in engine.draw(draws_needed).tolist():
        cycle_index = min(int(row[2] * len(cycles)), len(cycles) - 1)
        decision = Stage2Decision(
            precursor_dose_s=config.hard_bounds.precursor_dose_s.lerp(float(row[0])),
            temperature_c=config.hard_bounds.temperature_c.lerp(float(row[1])),
            cycle_count=cycles[cycle_index],
        )
        if validate_stage2_decision(config, decision):
            continue
        if any(normalized_stage2_distance(config, decision, existing) <= 1e-8 for existing in seen):
            continue
        decisions.append(decision)
        seen.append(decision)
        if len(decisions) >= n_points:
            return decisions

    for cycle_count in cycles:
        decision = Stage2Decision(
            precursor_dose_s=config.hard_bounds.precursor_dose_s.midpoint(),
            temperature_c=config.hard_bounds.temperature_c.midpoint(),
            cycle_count=cycle_count,
        )
        if not validate_stage2_decision(config, decision):
            decisions.append(decision)
        if len(decisions) >= n_points:
            break
    return decisions


def random_safe_fallback_decision(
    config: Stage2Config,
    settings: Stage2BOSettings,
    observations: Sequence[Stage2Observation],
    proposals: Sequence[Stage2CandidateProposal],
    *,
    seed: int | None = None,
) -> Stage2Decision | None:
    """Return a deterministic Sobol fallback candidate satisfying hard safety bounds."""

    cycles = candidate_cycle_values(config, settings)
    engine = SobolEngine(dimension=3, scramble=True, seed=seed)  # type: ignore[no-untyped-call]
    for row in engine.draw(settings.random_fallback_points).tolist():
        cycle_index = min(int(row[2] * len(cycles)), len(cycles) - 1)
        decision = Stage2Decision(
            precursor_dose_s=config.hard_bounds.precursor_dose_s.lerp(float(row[0])),
            temperature_c=config.hard_bounds.temperature_c.lerp(float(row[1])),
            cycle_count=cycles[cycle_index],
        )
        if validate_stage2_decision(config, decision):
            continue
        if duplicate_or_too_close(decision, observations, proposals, config, settings):
            continue
        return decision
    return None


def observed_hypervolume(
    config: Stage2Config,
    observations: Sequence[Stage2Observation],
    settings: Stage2BOSettings,
) -> float:
    """Return raw observed constrained hypervolume using the configured reference point."""

    feasible_vectors = [
        [
            observation.outcomes.ga_thickness_nm,
            -observation.outcomes.nga_thickness_nm,
            -observation.outcomes.process_time_s,
        ]
        for observation in observations
        if observation.constraint_evaluation.feasible
    ]
    if not feasible_vectors:
        return 0.0
    tensor = torch.tensor(feasible_vectors, dtype=torch.double)
    ref_point = torch.tensor(stage2_reference_point(config, settings), dtype=torch.double)
    try:
        return float(Hypervolume(ref_point).compute(tensor))
    except Exception:  # pragma: no cover - defensive guard for malformed tensors
        return 0.0


def best_stage2_observation(
    observations: Sequence[Stage2Observation],
) -> Stage2Observation | None:
    """Return the best tested Stage 2 row using feasibility first, then objectives."""

    if not observations:
        return None
    feasible = [
        observation for observation in observations if observation.constraint_evaluation.feasible
    ]
    source = feasible or list(observations)
    return max(
        source,
        key=lambda observation: (
            observation.constraint_evaluation.feasible,
            observation.outcomes.ga_thickness_nm,
            -observation.outcomes.nga_thickness_nm,
            observation.outcomes.selectivity,
            -observation.outcomes.process_time_s,
        ),
    )


def any_feasible(observations: Sequence[Stage2Observation]) -> bool:
    """Return whether at least one tested Stage 2 observation is feasible."""

    return any(observation.constraint_evaluation.feasible for observation in observations)


def proposal_status_to_run_status(status: Stage2BOProposalStatus) -> Stage2BORunStatus:
    """Map proposal-layer statuses to run-level statuses."""

    if status == "model_fit_failure":
        return "model_fit_failure"
    if status == "numerical_failure":
        return "numerical_failure"
    return "no_feasible_candidate"


def stage2_experiment_id(prefix: str, index: int) -> str:
    """Return a stable Stage 2 experiment id."""

    return f"{prefix}_{index:03d}"


def clamp_float(value: float, lower: float, upper: float) -> float:
    """Clamp a float into a closed interval."""

    return max(lower, min(upper, value))


def save_stage2_bo_results(
    results: Sequence[Stage2BOResult],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save Stage 2 MOBO smoke results as JSON and CSV summary files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "stage2_mobo_results.json"
    csv_path = destination / "stage2_mobo_summary.csv"
    json_path.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )
    rows = [result.summary_row() for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
