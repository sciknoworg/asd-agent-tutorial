"""Stage 1 grid and generic-GP active-learning optimizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from asd_agent.bo.acquisition import (
    ThresholdAcquisitionSettings,
    ThresholdDecision,
    choose_threshold_candidate,
    dose_grid,
    estimate_stage1_target_growth,
    is_duplicate_dose,
    smallest_tested_recommendation,
)
from asd_agent.bo.gp import (
    GenericGPSettings,
    GPFitFailure,
    fit_generic_stage1_gp,
)
from asd_agent.bo.oracle import (
    Stage1EvaluationOracle,
    Stage1Recommendation,
    Stage1RecommendationMetrics,
    evaluate_recommendation,
)
from asd_agent.bo.records import OptimizerState, utc_now
from asd_agent.bo.stage1 import Stage1Config, Stage1ExperimentRecord, Stage1VirtualLab

Stage1Method = Literal["grid", "generic_gp"]
Stage1OptimizationStatus = Literal[
    "success",
    "budget_exhausted",
    "no_saturation_detected",
    "model_fit_failure",
    "numerical_failure",
]
ProposalStatus = Literal["proposed", "no_valid_candidate", "model_fit_failure", "numerical_failure"]


class Stage1CandidateProposal(BaseModel):
    """One Stage 1 dose proposed by a grid or GP optimizer."""

    candidate_id: str = Field(min_length=1)
    dose_s: float = Field(ge=0.0)
    optimizer: str = Field(min_length=1)
    acquisition_value: float | None = None
    posterior_summaries: dict[str, float] = Field(default_factory=dict)
    training_observation_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        *,
        dose_s: float,
        optimizer: str,
        acquisition_value: float | None = None,
        posterior_summaries: dict[str, float] | None = None,
        training_observation_ids: Sequence[str] = (),
        seed: int | None = None,
    ) -> Stage1CandidateProposal:
        """Create a proposal with a compact unique candidate id."""

        safe_optimizer = "".join(char if char.isalnum() else "_" for char in optimizer.lower())
        return cls(
            candidate_id=f"stage1_cand_{safe_optimizer}_{uuid4().hex[:12]}",
            dose_s=dose_s,
            optimizer=optimizer,
            acquisition_value=acquisition_value,
            posterior_summaries=posterior_summaries or {},
            training_observation_ids=list(training_observation_ids),
            seed=seed,
        )


class Stage1ProposalResult(BaseModel):
    """Result of asking an optimizer for one next candidate."""

    status: ProposalStatus
    proposal: Stage1CandidateProposal | None = None
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Stage1GridSettings(BaseModel):
    """Settings for the fixed Stage 1 grid baseline."""

    n_points: int = Field(default=8, ge=2)
    lower_s: float | None = Field(default=None, ge=0.0)
    upper_s: float | None = Field(default=None, ge=0.0)
    duplicate_tolerance_s: float = Field(default=1e-8, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_bounds(self) -> Stage1GridSettings:
        if self.lower_s is not None and self.upper_s is not None and self.upper_s < self.lower_s:
            raise ValueError("grid upper_s must be greater than or equal to lower_s")
        return self


class Stage1GenericGPSettings(BaseModel):
    """Settings for the generic-GP Stage 1 optimizer."""

    gp: GenericGPSettings = Field(default_factory=GenericGPSettings)
    acquisition: ThresholdAcquisitionSettings = Field(default_factory=ThresholdAcquisitionSettings)
    min_fit_observations: int = Field(default=2, ge=2)

    model_config = ConfigDict(extra="forbid")


class Stage1RunnerSettings(BaseModel):
    """Shared settings for Stage 1 comparison runs."""

    budget: int = Field(default=8, ge=1)
    initial_dose_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.20])
    initial_doses_s: list[float] | None = None
    grid: Stage1GridSettings = Field(default_factory=Stage1GridSettings)
    generic_gp: Stage1GenericGPSettings = Field(default_factory=Stage1GenericGPSettings)
    simulator_seed: int | None = None
    optimizer_seed: int = 0
    min_recommendation_observations: int = Field(default=3, ge=1)

    model_config = ConfigDict(extra="forbid")


class Stage1OptimizationResult(BaseModel):
    """Serializable result from one Stage 1 optimizer run."""

    method: Stage1Method
    scenario_id: str
    status: Stage1OptimizationStatus
    records: list[Stage1ExperimentRecord]
    proposals: list[Stage1CandidateProposal] = Field(default_factory=list)
    recommendation: Stage1Recommendation | None = None
    recommended_experiment_id: str | None = None
    metrics: Stage1RecommendationMetrics | None = None
    warnings: list[str] = Field(default_factory=list)
    optimizer_state: OptimizerState
    started_at: str = Field(default_factory=utc_now)
    finished_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    def summary_row(self) -> dict[str, object]:
        """Return a compact row for smoke comparisons and benchmarks."""

        best = (
            max(self.records, key=lambda record: record.observed_growth) if self.records else None
        )
        return {
            "method": self.method,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "n_experiments": len(self.records),
            "recommended_experiment_id": self.recommended_experiment_id or "",
            "recommended_dose_s": (
                self.recommendation.recommended_dose_s if self.recommendation else ""
            ),
            "best_observed_growth": best.observed_growth if best else 0.0,
            "cumulative_dose_s": (
                self.metrics.cumulative_dose_s
                if self.metrics
                else sum(record.dose_s for record in self.records)
            ),
            "cumulative_process_time_s": (
                self.metrics.cumulative_simulated_process_time_s
                if self.metrics
                else sum(record.process_time_s for record in self.records)
            ),
        }


class Stage1FixedGridOptimizer:
    """Predetermined Stage 1 grid baseline with no adaptive selection."""

    name = "stage1_grid"

    def __init__(
        self,
        config: Stage1Config,
        settings: Stage1GridSettings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or Stage1GridSettings()
        self.seed = seed
        self.proposals: list[Stage1CandidateProposal] = []
        grid_bounds = config.dose_bounds_s.model_copy(
            update={
                "min": config.dose_bounds_s.min
                if self.settings.lower_s is None
                else self.settings.lower_s,
                "max": config.dose_bounds_s.max
                if self.settings.upper_s is None
                else self.settings.upper_s,
            }
        )
        self.grid = dose_grid(grid_bounds, self.settings.n_points)

    def propose(self, records: Sequence[Stage1ExperimentRecord]) -> Stage1ProposalResult:
        """Return the next untested grid point in deterministic order."""

        tested = [record.dose_s for record in records]
        for dose in self.grid:
            if not is_duplicate_dose(dose, tested, self.settings.duplicate_tolerance_s):
                proposal = Stage1CandidateProposal.create(
                    dose_s=dose,
                    optimizer=self.name,
                    training_observation_ids=[record.experiment_id for record in records],
                    seed=self.seed,
                )
                self.proposals.append(proposal)
                return Stage1ProposalResult(status="proposed", proposal=proposal)
        return Stage1ProposalResult(
            status="no_valid_candidate",
            warnings=["fixed grid has no untested dose remaining"],
        )

    def get_state(self) -> OptimizerState:
        """Return a serializable checkpoint."""

        return OptimizerState(
            optimizer=self.name,
            state={
                "settings": self.settings.model_dump(mode="json"),
                "grid": list(self.grid),
                "proposed_doses_s": [proposal.dose_s for proposal in self.proposals],
            },
            observation_ids=[],
            random_seed=self.seed,
        )

    def restore_state(self, state: OptimizerState) -> None:
        """Restore grid settings needed for deterministic continuation."""

        proposed_doses = state.state.get("proposed_doses_s", [])
        if isinstance(proposed_doses, list):
            self.proposals = [
                Stage1CandidateProposal.create(
                    dose_s=float(dose),
                    optimizer=self.name,
                    seed=state.random_seed,
                )
                for dose in proposed_doses
            ]


class Stage1GenericGPOptimizer:
    """Generic GP active learner for one-dimensional Stage 1 saturation."""

    name = "stage1_generic_gp"

    def __init__(
        self,
        config: Stage1Config,
        settings: Stage1GenericGPSettings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or Stage1GenericGPSettings()
        self.seed = seed
        self.proposals: list[Stage1CandidateProposal] = []
        self.warnings: list[str] = []

    def propose(self, records: Sequence[Stage1ExperimentRecord]) -> Stage1ProposalResult:
        """Fit the generic GP and return the next threshold-oriented candidate."""

        if len(records) < self.settings.min_fit_observations:
            return self._exploratory_proposal(records)
        try:
            gp_model = fit_generic_stage1_gp(
                records,
                self.config.dose_bounds_s,
                self.settings.gp,
                seed=self.seed,
            )
        except (GPFitFailure, ValueError) as exc:
            warning = f"generic GP fit failed: {exc}"
            self.warnings.append(warning)
            return Stage1ProposalResult(status="model_fit_failure", warnings=[warning])

        candidate_doses = dose_grid(
            self.config.dose_bounds_s,
            self.settings.acquisition.candidate_grid_size,
        )
        prediction = gp_model.posterior(candidate_doses)
        self.warnings.extend(gp_model.fit_warnings)
        target = estimate_stage1_target_growth(
            self.config,
            records,
            posterior_mean=prediction.mean,
        )
        decision = choose_threshold_candidate(
            prediction.dose_s,
            prediction.mean,
            prediction.stddev,
            target,
            [record.dose_s for record in records],
            self.settings.acquisition,
        )
        if decision.status != "candidate" or decision.dose_s is None:
            mapped_status: ProposalStatus = (
                "numerical_failure"
                if decision.status == "numerical_failure"
                else "no_valid_candidate"
            )
            return Stage1ProposalResult(status=mapped_status, warnings=[decision.rationale])
        proposal = Stage1CandidateProposal.create(
            dose_s=decision.dose_s,
            optimizer=self.name,
            acquisition_value=decision.acquisition_value,
            posterior_summaries=posterior_summary(decision),
            training_observation_ids=gp_model.training_observation_ids,
            seed=self.seed,
        )
        self.proposals.append(proposal)
        return Stage1ProposalResult(
            status="proposed", proposal=proposal, warnings=gp_model.fit_warnings
        )

    def _exploratory_proposal(
        self,
        records: Sequence[Stage1ExperimentRecord],
    ) -> Stage1ProposalResult:
        tested = [record.dose_s for record in records]
        for dose in dose_grid(self.config.dose_bounds_s, self.settings.min_fit_observations + 1):
            if not is_duplicate_dose(
                dose,
                tested,
                self.settings.acquisition.duplicate_tolerance_s,
            ):
                proposal = Stage1CandidateProposal.create(
                    dose_s=dose,
                    optimizer=self.name,
                    training_observation_ids=[record.experiment_id for record in records],
                    seed=self.seed,
                )
                self.proposals.append(proposal)
                return Stage1ProposalResult(status="proposed", proposal=proposal)
        return Stage1ProposalResult(
            status="no_valid_candidate",
            warnings=["no exploratory candidate remains"],
        )

    def get_state(self) -> OptimizerState:
        """Return a serializable checkpoint."""

        return OptimizerState(
            optimizer=self.name,
            state={
                "settings": self.settings.model_dump(mode="json"),
                "proposed_doses_s": [proposal.dose_s for proposal in self.proposals],
                "warnings": list(self.warnings),
            },
            observation_ids=[],
            random_seed=self.seed,
        )

    def restore_state(self, state: OptimizerState) -> None:
        """Restore proposal history and recorded warnings."""

        warnings = state.state.get("warnings", [])
        self.warnings = [str(item) for item in warnings] if isinstance(warnings, list) else []
        proposed_doses = state.state.get("proposed_doses_s", [])
        if isinstance(proposed_doses, list):
            self.proposals = [
                Stage1CandidateProposal.create(
                    dose_s=float(dose),
                    optimizer=self.name,
                    seed=state.random_seed,
                )
                for dose in proposed_doses
            ]


def run_stage1_optimization(
    config: Stage1Config,
    method: Stage1Method,
    settings: Stage1RunnerSettings | None = None,
) -> Stage1OptimizationResult:
    """Run one Stage 1 optimizer with matched initial observations and budget."""

    resolved_settings = settings or Stage1RunnerSettings()
    lab = Stage1VirtualLab(config, seed=resolved_settings.simulator_seed)
    optimizer: Stage1FixedGridOptimizer | Stage1GenericGPOptimizer
    if method == "grid":
        optimizer = Stage1FixedGridOptimizer(
            config,
            resolved_settings.grid,
            seed=resolved_settings.optimizer_seed,
        )
    else:
        optimizer = Stage1GenericGPOptimizer(
            config,
            resolved_settings.generic_gp,
            seed=resolved_settings.optimizer_seed,
        )

    records: list[Stage1ExperimentRecord] = []
    warnings: list[str] = []
    status: Stage1OptimizationStatus | None = None

    for dose in initial_doses(config, resolved_settings):
        if len(records) >= resolved_settings.budget:
            break
        if already_tested(
            dose, records, resolved_settings.generic_gp.acquisition.duplicate_tolerance_s
        ):
            continue
        records.append(
            lab.observe(
                dose,
                experiment_id=stage1_experiment_id(method, len(records) + 1),
                decision_rationale="Matched initial Stage 1 observation.",
            )
        )

    while len(records) < resolved_settings.budget:
        current_recommendation = smallest_tested_recommendation(
            config,
            records,
            min_observations=resolved_settings.min_recommendation_observations,
        )
        if current_recommendation is not None:
            status = "success"
            break

        proposal_result = optimizer.propose(records)
        warnings.extend(proposal_result.warnings)
        if proposal_result.status != "proposed" or proposal_result.proposal is None:
            status = proposal_status_to_optimization_status(proposal_result.status)
            break
        records.append(
            lab.observe(
                proposal_result.proposal.dose_s,
                experiment_id=stage1_experiment_id(method, len(records) + 1),
                decision_rationale=f"Stage 1 proposal from {proposal_result.proposal.optimizer}.",
            )
        )

    final_recommendation = smallest_tested_recommendation(
        config,
        records,
        min_observations=resolved_settings.min_recommendation_observations,
    )
    if final_recommendation is None:
        recommendation = None
        recommended_experiment_id = None
        metrics = None
        status = status or "budget_exhausted"
    else:
        recommendation, recommended_experiment_id = final_recommendation
        oracle_report = Stage1EvaluationOracle(config).evaluate()
        metrics = evaluate_recommendation(config, recommendation, records, oracle_report)
        status = status or "success"

    optimizer_warnings = (
        optimizer.warnings if isinstance(optimizer, Stage1GenericGPOptimizer) else []
    )
    return Stage1OptimizationResult(
        method=method,
        scenario_id=config.scenario_id,
        status=status,
        records=records,
        proposals=list(optimizer.proposals),
        recommendation=recommendation,
        recommended_experiment_id=recommended_experiment_id,
        metrics=metrics,
        warnings=warnings + optimizer_warnings,
        optimizer_state=optimizer.get_state(),
    )


def compare_stage1_methods(
    configs: Sequence[Stage1Config],
    methods: Sequence[Stage1Method] = ("grid", "generic_gp"),
    settings: Stage1RunnerSettings | None = None,
) -> list[Stage1OptimizationResult]:
    """Run matched Stage 1 comparisons for multiple configs and methods."""

    return [
        run_stage1_optimization(config, method, settings)
        for config in configs
        for method in methods
    ]


def initial_doses(config: Stage1Config, settings: Stage1RunnerSettings) -> list[float]:
    """Return explicit or fractional matched initial doses."""

    if settings.initial_doses_s is not None:
        return [dose for dose in settings.initial_doses_s if config.dose_bounds_s.contains(dose)]
    return [config.dose_bounds_s.lerp(fraction) for fraction in settings.initial_dose_fractions]


def already_tested(
    dose_s: float,
    records: Sequence[Stage1ExperimentRecord],
    tolerance_s: float,
) -> bool:
    """Return whether a dose is already in the local Stage 1 ledger."""

    return is_duplicate_dose(dose_s, [record.dose_s for record in records], tolerance_s)


def stage1_experiment_id(method: Stage1Method, index: int) -> str:
    """Return a stable experiment id for Stage 1 runner records."""

    return f"{method}_{index:03d}"


def proposal_status_to_optimization_status(status: ProposalStatus) -> Stage1OptimizationStatus:
    """Map proposal-layer failures to run-layer stopping states."""

    if status == "model_fit_failure":
        return "model_fit_failure"
    if status == "numerical_failure":
        return "numerical_failure"
    return "no_saturation_detected"


def posterior_summary(decision: ThresholdDecision) -> dict[str, float]:
    """Return finite posterior fields for a candidate proposal."""

    summary: dict[str, float] = {}
    for key, value in (
        ("posterior_mean", decision.posterior_mean),
        ("posterior_stddev", decision.posterior_stddev),
        ("target_growth", decision.target_growth),
        ("target_probability", decision.target_probability),
    ):
        if value is not None:
            summary[key] = value
    return summary
