"""Random-search and grid-search baselines."""

from __future__ import annotations

import itertools

import numpy as np

from asd_agent.experiment_loop import run_conditions
from asd_agent.models import ExperimentCondition, OptimizationRun, ProcessConfig


def random_search(
    config: ProcessConfig, budget: int = 20, seed: int | None = None
) -> OptimizationRun:
    """Uniform random baseline over the configured safety bounds."""

    run_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(run_seed)
    conditions = [
        ExperimentCondition(
            precursor_dose_s=float(
                rng.uniform(config.safety.precursor_dose_s.min, config.safety.precursor_dose_s.max)
            ),
            coreactant_dose_s=float(
                rng.uniform(
                    config.safety.coreactant_dose_s.min, config.safety.coreactant_dose_s.max
                )
            ),
            inhibitor_dose_s=float(
                rng.uniform(config.safety.inhibitor_dose_s.min, config.safety.inhibitor_dose_s.max)
            ),
            temperature_c=float(
                rng.uniform(config.safety.temperature_c.min, config.safety.temperature_c.max)
            ),
            cycles=int(
                rng.integers(int(config.safety.cycles.min), int(config.safety.cycles.max) + 1)
            ),
        )
        for _ in range(budget)
    ]
    return run_conditions(config, conditions, "random_search", run_seed)


def grid_search(
    config: ProcessConfig, budget: int = 81, seed: int | None = None
) -> OptimizationRun:
    """Coarse deterministic grid baseline."""

    run_seed = config.seed if seed is None else seed
    safety = config.safety
    precursor_values = np.linspace(safety.precursor_dose_s.min, safety.precursor_dose_s.max, 3)
    coreactant_values = np.linspace(safety.coreactant_dose_s.min, safety.coreactant_dose_s.max, 3)
    inhibitor_values = np.linspace(safety.inhibitor_dose_s.min, safety.inhibitor_dose_s.max, 3)
    temperature_values = [safety.temperature_c.midpoint()]
    cycle_values = [int(round(safety.cycles.lerp(value))) for value in (0.35, 0.65, 1.0)]

    conditions = [
        ExperimentCondition(
            precursor_dose_s=float(precursor),
            coreactant_dose_s=float(coreactant),
            inhibitor_dose_s=float(inhibitor),
            temperature_c=float(temperature),
            cycles=int(cycles),
        )
        for precursor, coreactant, inhibitor, temperature, cycles in itertools.product(
            precursor_values,
            coreactant_values,
            inhibitor_values,
            temperature_values,
            cycle_values,
        )
    ]
    return run_conditions(config, conditions[:budget], "grid_search", run_seed)
