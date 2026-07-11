"""Virtual ASD laboratory simulator.

The equations here are intentionally simple teaching models. They are not fitted to
or intended to predict real HfO2/MoS2 chemistry.
"""

from __future__ import annotations

from math import exp

import numpy as np

from asd_agent.models import ExperimentCondition, ExperimentRecord, ProcessConfig, SurfaceParams
from asd_agent.objective import evaluate_objective, selectivity


class VirtualLab:
    """Seeded simulator for the virtual ASD laboratory."""

    def __init__(self, config: ProcessConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def simulate(
        self,
        condition: ExperimentCondition,
        experiment_id: str = "experiment_000",
        decision_rationale: str = "",
    ) -> ExperimentRecord:
        """Run one virtual experiment and return a ledger record."""

        ga_true = self.surface_thickness("GA", condition)
        nga_true = self.surface_thickness("NGA", condition)
        ga_measured = self._measure(ga_true)
        nga_measured = self._measure(nga_true)
        sel = selectivity(ga_measured, nga_measured)
        evaluation = evaluate_objective(condition, ga_measured, nga_measured, self.config)
        return ExperimentRecord(
            experiment_id=experiment_id,
            condition=condition,
            ga_thickness_nm=ga_measured,
            nga_thickness_nm=nga_measured,
            selectivity=sel,
            process_time_s=self.process_time(condition),
            meets_objective=evaluation.meets_objective,
            failure_reasons=evaluation.failure_reasons,
            decision_rationale=decision_rationale,
        )

    def surface_thickness(self, surface_name: str, condition: ExperimentCondition) -> float:
        """Return noise-free thickness for one surface after N cycles."""

        surface = self.config.surfaces[surface_name]
        active_cycles = max(0.0, condition.cycles - surface.nucleation_delay_cycles)
        precursor = saturating_response(condition.precursor_dose_s, surface.precursor_tau_s)
        coreactant = saturating_response(condition.coreactant_dose_s, surface.coreactant_tau_s)
        inhibitor = inhibitor_factor(condition.inhibitor_dose_s, surface.inhibitor_sensitivity)
        temperature = temperature_factor(condition.temperature_c, surface)
        gpc = surface.max_growth_per_cycle_nm * precursor * coreactant * inhibitor * temperature
        return max(0.0, active_cycles * gpc)

    def process_time(self, condition: ExperimentCondition) -> float:
        """Total simulated process time in seconds."""

        per_cycle = (
            condition.precursor_dose_s
            + condition.coreactant_dose_s
            + condition.inhibitor_dose_s
            + self.config.per_cycle_overhead_s
        )
        return self.config.stabilization_time_s + condition.cycles * per_cycle

    def _measure(self, true_thickness_nm: float) -> float:
        if self.config.noise_sigma_nm == 0:
            return true_thickness_nm
        noisy = true_thickness_nm + self.rng.normal(0.0, self.config.noise_sigma_nm)
        return max(0.0, float(noisy))


def saturating_response(dose_s: float, tau_s: float) -> float:
    """Monotone saturating dose response."""

    return 1.0 - exp(-max(0.0, dose_s) / tau_s)


def inhibitor_factor(inhibitor_dose_s: float, sensitivity: float) -> float:
    """Blocking factor, where larger sensitivity means stronger inhibition."""

    return exp(-max(0.0, inhibitor_dose_s) * sensitivity)


def temperature_factor(temperature_c: float, surface: SurfaceParams) -> float:
    """Optional Gaussian-like temperature response with a nonzero floor."""

    if not surface.temperature_response_enabled:
        return 1.0
    scaled_distance = (temperature_c - surface.temperature_optimum_c) / surface.temperature_width_c
    gaussian = exp(-0.5 * scaled_distance * scaled_distance)
    return surface.temperature_min_factor + (1.0 - surface.temperature_min_factor) * gaussian
