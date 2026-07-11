"""Stage 1 one-dimensional saturation processes for BO tutorials."""

from __future__ import annotations

from math import exp
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from asd_agent.bo.records import utc_now
from asd_agent.models import Range

Stage1Mode = Literal["known_target", "inferred_asymptote"]
Stage1Family = Literal[
    "mono_exponential",
    "noisy_saturation",
    "soft_biexponential",
    "weakly_non_self_limited",
    "misspecified_saturation",
]


class Stage1Objective(BaseModel):
    """Objective mode for Stage 1 saturation learning."""

    mode: Stage1Mode
    saturation_fraction: float = Field(default=0.95, gt=0.0, lt=1.0)
    target_growth: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_mode_specific_fields(self) -> Stage1Objective:
        if self.mode == "known_target" and self.target_growth is None:
            raise ValueError("known_target mode requires target_growth")
        if self.mode == "inferred_asymptote" and self.target_growth is not None:
            raise ValueError("inferred_asymptote mode must not set target_growth")
        return self


class Stage1ProcessParameters(BaseModel):
    """Hidden process parameters for a one-dimensional virtual process."""

    family: Stage1Family
    g_inf: float = Field(gt=0.0)
    k: float | None = Field(default=None, gt=0.0)
    fast_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    k_fast: float | None = Field(default=None, gt=0.0)
    k_slow: float | None = Field(default=None, gt=0.0)
    linear_slope: float = Field(default=0.0, ge=0.0)
    misspecification_power: float | None = Field(default=None, gt=0.0)
    observation_noise_sigma: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_family_parameters(self) -> Stage1ProcessParameters:
        if (
            self.family
            in {
                "mono_exponential",
                "noisy_saturation",
                "weakly_non_self_limited",
            }
            and self.k is None
        ):
            raise ValueError(f"{self.family} requires k")
        if self.family == "soft_biexponential":
            missing = [
                name
                for name, value in (
                    ("fast_fraction", self.fast_fraction),
                    ("k_fast", self.k_fast),
                    ("k_slow", self.k_slow),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"soft_biexponential missing: {missing}")
        if self.family == "weakly_non_self_limited" and self.linear_slope <= 0.0:
            raise ValueError("weakly_non_self_limited requires a positive linear_slope")
        if self.family == "misspecified_saturation":
            if self.k is None:
                raise ValueError("misspecified_saturation requires k")
            if self.misspecification_power is None:
                raise ValueError("misspecified_saturation requires misspecification_power")
        return self


class Stage1Config(BaseModel):
    """Complete configuration for a Stage 1 virtual saturation process."""

    scenario_id: str
    description: str
    seed: int = 0
    dose_bounds_s: Range
    objective: Stage1Objective
    process: Stage1ProcessParameters
    process_time_fixed_s: float = Field(default=0.0, ge=0.0)
    process_time_per_s_dose: float = Field(default=1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    def optimizer_view(self) -> dict[str, object]:
        """Return only information available to optimizers."""

        view: dict[str, object] = {
            "process_kind": "stage1_saturation",
            "mode": self.objective.mode,
            "dose_bounds_s": self.dose_bounds_s.model_dump(mode="json"),
            "saturation_fraction": self.objective.saturation_fraction,
        }
        if self.objective.mode == "known_target":
            view["target_growth"] = self.objective.target_growth
        return view


class Stage1Dose(BaseModel):
    """One proposed dose in seconds."""

    dose_s: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid")


class Stage1ExperimentRecord(BaseModel):
    """Optimizer-facing record for one Stage 1 observation."""

    experiment_id: str
    dose_s: float
    observed_growth: float
    process_time_s: float
    decision_rationale: str = ""
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")


class Stage1VirtualLab:
    """Seeded virtual lab for one-dimensional Stage 1 processes."""

    def __init__(self, config: Stage1Config, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def optimizer_view(self) -> dict[str, object]:
        """Return optimizer-facing context for this virtual process."""

        return self.config.optimizer_view()

    def hidden_process_parameters(self) -> dict[str, object]:
        """Return hidden process details for evaluation-only use."""

        return {
            "scenario_id": self.config.scenario_id,
            "description": self.config.description,
            "process": self.config.process.model_dump(mode="json"),
            "process_time_fixed_s": self.config.process_time_fixed_s,
            "process_time_per_s_dose": self.config.process_time_per_s_dose,
        }

    def observe(
        self,
        dose: Stage1Dose | float,
        experiment_id: str = "stage1_000",
        decision_rationale: str = "",
    ) -> Stage1ExperimentRecord:
        """Run one virtual dose experiment and return an optimizer-facing record."""

        condition = dose if isinstance(dose, Stage1Dose) else Stage1Dose(dose_s=dose)
        violations = validate_stage1_dose(condition, self.config)
        if violations:
            raise ValueError(f"unsafe Stage 1 dose: {violations}")
        true_value = true_growth(self.config.process, condition.dose_s)
        observed = max(0.0, true_value + self._noise())
        return Stage1ExperimentRecord(
            experiment_id=experiment_id,
            dose_s=condition.dose_s,
            observed_growth=observed,
            process_time_s=process_time(self.config, condition.dose_s),
            decision_rationale=decision_rationale,
        )

    def _noise(self) -> float:
        sigma = self.config.process.observation_noise_sigma
        if sigma == 0.0:
            return 0.0
        return float(self.rng.normal(0.0, sigma))


def validate_stage1_dose(condition: Stage1Dose, config: Stage1Config) -> list[str]:
    """Return dose-bound violations for a Stage 1 experiment."""

    if config.dose_bounds_s.contains(condition.dose_s):
        return []
    bounds = config.dose_bounds_s
    return [f"dose_s={condition.dose_s} outside [{bounds.min}, {bounds.max}]"]


def process_time(config: Stage1Config, dose_s: float) -> float:
    """Return simulated process time for one Stage 1 dose."""

    return config.process_time_fixed_s + config.process_time_per_s_dose * dose_s


def true_growth(process: Stage1ProcessParameters, dose_s: float) -> float:
    """Return the noise-free growth for a Stage 1 process."""

    dose = max(0.0, dose_s)
    if process.family in {"mono_exponential", "noisy_saturation"}:
        return process.g_inf * (1.0 - exp(-required(process.k, "k") * dose))
    if process.family == "soft_biexponential":
        fast_fraction = required(process.fast_fraction, "fast_fraction")
        slow_fraction = 1.0 - fast_fraction
        fast = fast_fraction * exp(-required(process.k_fast, "k_fast") * dose)
        slow = slow_fraction * exp(-required(process.k_slow, "k_slow") * dose)
        return process.g_inf * (1.0 - fast - slow)
    if process.family == "weakly_non_self_limited":
        saturated = process.g_inf * (1.0 - exp(-required(process.k, "k") * dose))
        return saturated + process.linear_slope * dose
    if process.family == "misspecified_saturation":
        scaled = (required(process.k, "k") * dose) ** required(
            process.misspecification_power, "misspecification_power"
        )
        return process.g_inf * (1.0 - exp(-scaled))
    raise ValueError(f"unknown Stage 1 family: {process.family}")


def required(value: float | None, name: str) -> float:
    """Return a validated optional process parameter."""

    if value is None:
        raise ValueError(f"missing required parameter: {name}")
    return value
