"""Typed data models shared by the simulator, agents, and benchmarks."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FinishStatus = Literal["success", "no_selective_window", "budget_exhausted"]
FailureCategory = Literal[
    "success",
    "no_selective_window",
    "budget_exhausted",
    "safety_violation",
]


class Range(BaseModel):
    """Closed numeric range used for safety bounds."""

    min: float
    max: float

    @model_validator(mode="after")
    def check_order(self) -> Range:
        if self.max < self.min:
            raise ValueError("range max must be greater than or equal to min")
        return self

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max

    def midpoint(self) -> float:
        return (self.min + self.max) / 2.0

    def lerp(self, fraction: float) -> float:
        fraction = max(0.0, min(1.0, fraction))
        return self.min + fraction * (self.max - self.min)


class SafetyBounds(BaseModel):
    """Safety envelope for proposed experiments."""

    precursor_dose_s: Range
    coreactant_dose_s: Range
    inhibitor_dose_s: Range
    temperature_c: Range
    cycles: Range


class ObjectiveConstraints(BaseModel):
    """Default target for successful ASD conditions."""

    ga_min_nm: float = 5.0
    nga_max_nm: float = 0.5
    selectivity_min: float = 0.80


class SurfaceParams(BaseModel):
    """Toy surface-response parameters for GA or NGA."""

    max_growth_per_cycle_nm: float = Field(gt=0)
    precursor_tau_s: float = Field(gt=0)
    coreactant_tau_s: float = Field(gt=0)
    nucleation_delay_cycles: float = Field(ge=0)
    inhibitor_sensitivity: float = Field(ge=0)
    temperature_response_enabled: bool = True
    temperature_optimum_c: float = 180.0
    temperature_width_c: float = Field(default=50.0, gt=0)
    temperature_min_factor: float = Field(default=0.5, ge=0.0, le=1.0)


class ExperimentCondition(BaseModel):
    """A proposed virtual experiment."""

    precursor_dose_s: float = Field(ge=0)
    coreactant_dose_s: float = Field(ge=0)
    inhibitor_dose_s: float = Field(default=0.0, ge=0)
    temperature_c: float
    cycles: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")

    def rounded_key(self) -> tuple[float, float, float, float, int]:
        return (
            round(self.precursor_dose_s, 6),
            round(self.coreactant_dose_s, 6),
            round(self.inhibitor_dose_s, 6),
            round(self.temperature_c, 6),
            self.cycles,
        )


class ProcessConfig(BaseModel):
    """Complete scenario configuration."""

    scenario: str
    description: str
    seed: int = 0
    noise_sigma_nm: float = Field(default=0.0, ge=0.0)
    per_cycle_overhead_s: float = Field(default=0.0, ge=0.0)
    stabilization_time_s: float = Field(default=0.0, ge=0.0)
    surfaces: dict[str, SurfaceParams]
    safety: SafetyBounds
    objective: ObjectiveConstraints = Field(default_factory=ObjectiveConstraints)

    @model_validator(mode="after")
    def require_surfaces(self) -> ProcessConfig:
        missing = {"GA", "NGA"} - set(self.surfaces)
        if missing:
            raise ValueError(f"missing required surfaces: {sorted(missing)}")
        return self


class ObjectiveEvaluation(BaseModel):
    """Objective and safety evaluation for a completed experiment."""

    meets_safety: bool
    meets_objective: bool
    failure_reasons: list[str] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    """One row in the experiment ledger."""

    experiment_id: str
    condition: ExperimentCondition
    ga_thickness_nm: float
    nga_thickness_nm: float
    selectivity: float
    process_time_s: float
    meets_objective: bool
    failure_reasons: list[str] = Field(default_factory=list)
    decision_rationale: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ProposedExperimentsDecision(BaseModel):
    """Strictly validated content of the propose_experiments tool."""

    experiments: list[ExperimentCondition] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        if sentence_count(value) > 4:
            raise ValueError("decision rationale must be at most four sentences")
        return value.strip()


class FinishOptimizationDecision(BaseModel):
    """Strictly validated content of the finish_optimization tool."""

    status: FinishStatus
    tested_experiment_id: str
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        if sentence_count(value) > 4:
            raise ValueError("decision rationale must be at most four sentences")
        return value.strip()


class OptimizationRun(BaseModel):
    """Serializable result of an optimizer run."""

    method: str
    scenario: str
    status: FinishStatus
    records: list[ExperimentRecord]
    seed: int
    model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    failure_category: FailureCategory = "budget_exhausted"
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def best_record(self) -> ExperimentRecord | None:
        if not self.records:
            return None
        return max(self.records, key=lambda record: (record.selectivity, record.ga_thickness_nm))

    def summary_row(self) -> dict[str, Any]:
        best = self.best_record
        successful = [record for record in self.records if record.meets_objective]
        recommended = successful[0] if successful else best
        return {
            "method": self.method,
            "scenario": self.scenario,
            "status": self.status,
            "failure_category": self.failure_category,
            "seed": self.seed,
            "model": self.model or "",
            "n_experiments": len(self.records),
            "success": bool(successful),
            "best_selectivity": best.selectivity if best else 0.0,
            "ga_thickness_nm": recommended.ga_thickness_nm if recommended else 0.0,
            "nga_thickness_nm": recommended.nga_thickness_nm if recommended else 0.0,
            "total_process_time_s": sum(record.process_time_s for record in self.records),
            "input_tokens": self.token_usage.get("input_tokens", 0),
            "output_tokens": self.token_usage.get("output_tokens", 0),
            "total_tokens": self.token_usage.get("total_tokens", 0),
        }


def sentence_count(text: str) -> int:
    """Small helper for enforcing stored-rationale length."""

    terminal_marks = ".!?"
    count = 0
    in_sentence = False
    for char in text.strip():
        if char in terminal_marks:
            if in_sentence:
                count += 1
                in_sentence = False
        elif not char.isspace():
            in_sentence = True
    if in_sentence:
        count += 1
    return count
