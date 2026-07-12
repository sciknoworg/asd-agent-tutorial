"""Serializable BO proposals, observations, manifests, and run records."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.models import ExperimentCondition, ExperimentRecord, OptimizationRun

DEFAULT_DEPENDENCY_NAMES = [
    "numpy",
    "pandas",
    "pydantic",
    "pyyaml",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "torch",
    "gpytorch",
    "botorch",
    "ax-platform",
]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


class CandidateProposal(BaseModel):
    """One BO candidate before it is sent to an experiment backend."""

    candidate_id: str = Field(min_length=1)
    parameters: ExperimentCondition
    optimizer: str = Field(min_length=1)
    model_version: str | None = None
    acquisition_value: float | None = None
    feasibility_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    posterior_summaries: dict[str, float] = Field(default_factory=dict)
    training_observation_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        parameters: ExperimentCondition,
        optimizer: str,
        *,
        acquisition_value: float | None = None,
        feasibility_probability: float | None = None,
        posterior_summaries: dict[str, float] | None = None,
        model_version: str | None = None,
        training_observation_ids: Iterable[str] = (),
        seed: int | None = None,
        timestamp: str | None = None,
    ) -> CandidateProposal:
        """Create a proposal with a collision-resistant candidate id."""

        return cls(
            candidate_id=make_candidate_id(optimizer),
            parameters=parameters,
            optimizer=optimizer,
            model_version=model_version,
            acquisition_value=acquisition_value,
            feasibility_probability=feasibility_probability,
            posterior_summaries=posterior_summaries or {},
            training_observation_ids=list(training_observation_ids),
            seed=seed,
            timestamp=timestamp or utc_now(),
        )


class OptimizerObservation(BaseModel):
    """The optimizer-facing view of one completed experiment."""

    experiment_id: str
    condition: ExperimentCondition
    ga_thickness_nm: float
    nga_thickness_nm: float
    selectivity: float
    process_time_s: float
    meets_objective: bool
    failure_reasons: list[str] = Field(default_factory=list)
    timestamp: str

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_experiment_record(cls, record: ExperimentRecord) -> OptimizerObservation:
        """Build an optimizer-visible observation from the canonical ledger row."""

        return cls(
            experiment_id=record.experiment_id,
            condition=record.condition,
            ga_thickness_nm=record.ga_thickness_nm,
            nga_thickness_nm=record.nga_thickness_nm,
            selectivity=record.selectivity,
            process_time_s=record.process_time_s,
            meets_objective=record.meets_objective,
            failure_reasons=list(record.failure_reasons),
            timestamp=record.timestamp,
        )


class BOExperimentRecord(BaseModel):
    """Backward-compatible extension around the canonical experiment record."""

    schema_version: str = "bo-experiment-record-v1"
    experiment: ExperimentRecord
    proposal: CandidateProposal | None = None
    optimizer_observation: OptimizerObservation
    run_id: str | None = None
    candidate_id: str | None = None
    measurement_uncertainty: dict[str, float] = Field(default_factory=dict)
    feasibility: bool | None = None
    constraint_violations: list[str] = Field(default_factory=list)
    simulator_seed: int | None = None
    optimizer_seed: int | None = None
    agent_action: str | None = None
    concise_rationale: str = ""
    oracle_evaluation: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_experiment_record(
        cls,
        record: ExperimentRecord,
        proposal: CandidateProposal | None = None,
        *,
        run_id: str | None = None,
        measurement_uncertainty: dict[str, float] | None = None,
        simulator_seed: int | None = None,
        optimizer_seed: int | None = None,
        agent_action: str | None = None,
        oracle_evaluation: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BOExperimentRecord:
        """Wrap an existing ledger record without modifying its schema."""

        return cls(
            experiment=record,
            proposal=proposal,
            optimizer_observation=OptimizerObservation.from_experiment_record(record),
            run_id=run_id,
            candidate_id=proposal.candidate_id if proposal is not None else None,
            measurement_uncertainty=measurement_uncertainty or {},
            feasibility=record.meets_objective,
            constraint_violations=list(record.failure_reasons),
            simulator_seed=simulator_seed,
            optimizer_seed=optimizer_seed,
            agent_action=agent_action,
            concise_rationale=record.decision_rationale,
            oracle_evaluation=oracle_evaluation or {},
            metadata=metadata or {},
        )

    def optimizer_payload(self) -> dict[str, object]:
        """Return only the observation fields that optimizers may consume."""

        return self.optimizer_observation.model_dump(mode="json")


class RunManifest(BaseModel):
    """Reproducibility metadata for a BO-capable optimization run."""

    schema_version: str = "bo-run-manifest-v1"
    run_id: str = Field(min_length=1)
    git_commit: str
    configuration_path: str
    configuration_hash: str
    python_version: str
    dependency_versions: dict[str, str]
    operating_system: str
    named_seeds: dict[str, int]
    method: str
    scenario: str
    experiment_budget: int
    acquisition_function: str | None = None
    model_settings: dict[str, object] = Field(default_factory=dict)
    llm_model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        *,
        config_path: str | Path,
        method: str,
        scenario: str,
        experiment_budget: int,
        named_seeds: dict[str, int],
        acquisition_function: str | None = None,
        model_settings: dict[str, object] | None = None,
        llm_model: str | None = None,
        token_usage: dict[str, int] | None = None,
        dependency_names: Iterable[str] = DEFAULT_DEPENDENCY_NAMES,
        repo_root: str | Path | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
    ) -> RunManifest:
        """Build a manifest from the local runtime and configuration file."""

        resolved_config = Path(config_path).resolve()
        root = Path(repo_root).resolve() if repo_root is not None else resolved_config.parents[1]
        return cls(
            run_id=run_id or f"run_{uuid4().hex[:12]}",
            git_commit=current_git_commit(root),
            configuration_path=str(resolved_config),
            configuration_hash=hash_file(resolved_config),
            python_version=sys.version.split()[0],
            dependency_versions=dependency_versions(dependency_names),
            operating_system=platform.platform(),
            named_seeds=dict(named_seeds),
            method=method,
            acquisition_function=acquisition_function,
            model_settings=model_settings or {},
            scenario=scenario,
            experiment_budget=experiment_budget,
            llm_model=llm_model,
            token_usage=token_usage or {},
            started_at=started_at or utc_now(),
        )

    def mark_finished(self, finished_at: str | None = None) -> RunManifest:
        """Return a copy with the finished timestamp populated."""

        return self.model_copy(update={"finished_at": finished_at or utc_now()})


class OptimizerState(BaseModel):
    """Serializable optimizer checkpoint for future BO implementations."""

    schema_version: str = "optimizer-state-v1"
    optimizer: str
    state: dict[str, object] = Field(default_factory=dict)
    candidate_history: list[CandidateProposal] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    random_seed: int | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")


class BORunRecord(BaseModel):
    """Serializable BO run artifact that extends existing ledgers."""

    schema_version: str = "bo-run-record-v1"
    manifest: RunManifest
    records: list[BOExperimentRecord] = Field(default_factory=list)
    optimizer_state: OptimizerState | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_optimization_run(
        cls,
        run: OptimizationRun,
        manifest: RunManifest,
        *,
        optimizer_state: OptimizerState | None = None,
    ) -> BORunRecord:
        """Create a BO run record from a legacy `OptimizationRun`."""

        return cls(
            manifest=manifest,
            records=[BOExperimentRecord.from_experiment_record(record) for record in run.records],
            optimizer_state=optimizer_state,
            token_usage=dict(run.token_usage),
        )

    def optimizer_observations(self) -> list[OptimizerObservation]:
        """Return optimizer-visible observations from the run ledger."""

        return [record.optimizer_observation for record in self.records]


def make_candidate_id(optimizer: str) -> str:
    """Return a compact unique candidate id with the optimizer name embedded."""

    safe_optimizer = "".join(char if char.isalnum() else "_" for char in optimizer.lower())
    return f"cand_{safe_optimizer}_{uuid4().hex[:12]}"


def hash_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def current_git_commit(repo_root: Path) -> str:
    """Return the current Git commit, or `unknown` outside a Git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return "unknown"
    except subprocess.CalledProcessError:
        return "unknown"
    except subprocess.TimeoutExpired:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def dependency_versions(names: Iterable[str]) -> dict[str, str]:
    """Return installed package versions, using `not installed` for optional extras."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
