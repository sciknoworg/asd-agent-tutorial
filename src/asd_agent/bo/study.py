"""Stage 1 study profiles and deterministic comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.acquisition import ThresholdAcquisitionSettings
from asd_agent.bo.optimizers import (
    Stage1GenericGPSettings,
    Stage1GridSettings,
    Stage1Method,
    Stage1OptimizationResult,
    Stage1PhysicsGPSettings,
    Stage1RunnerSettings,
    compare_stage1_methods,
)
from asd_agent.config import CONFIG_DIR, load_config_mapping, load_stage1_scenario


def default_stage1_methods() -> list[Stage1Method]:
    """Return the BO-04 default Stage 1 comparison methods."""

    return ["grid", "generic_gp", "physics_gp"]


class Stage1StudyProfile(BaseModel):
    """Deterministic profile for a Stage 1 comparison study."""

    profile_id: str
    description: str
    scenarios: list[str]
    methods: list[Stage1Method] = Field(default_factory=default_stage1_methods)
    repetitions: int = Field(default=1, ge=1)
    budget: int = Field(default=8, ge=1)
    candidate_grid_size: int = Field(default=41, ge=2)
    initial_dose_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.2])
    simulator_seed: int = 6104
    optimizer_seed: int = 7104
    endpoint_relative_tolerance: float = Field(default=0.10, ge=0.0)
    endpoint_tolerance_s: float | None = Field(default=None, ge=0.0)
    min_recommendation_observations: int = Field(default=3, ge=1)

    model_config = ConfigDict(extra="forbid")

    def runner_settings(self, repetition: int) -> Stage1RunnerSettings:
        """Return matched runner settings for one repetition."""

        acquisition = ThresholdAcquisitionSettings(candidate_grid_size=self.candidate_grid_size)
        return Stage1RunnerSettings(
            budget=self.budget,
            initial_dose_fractions=list(self.initial_dose_fractions),
            grid=Stage1GridSettings(n_points=self.candidate_grid_size),
            generic_gp=Stage1GenericGPSettings(acquisition=acquisition),
            physics_gp=Stage1PhysicsGPSettings(acquisition=acquisition),
            simulator_seed=self.simulator_seed + repetition,
            optimizer_seed=self.optimizer_seed + repetition,
            endpoint_relative_tolerance=self.endpoint_relative_tolerance,
            endpoint_tolerance_s=self.endpoint_tolerance_s,
            min_recommendation_observations=self.min_recommendation_observations,
        )


def load_stage1_study_profile(path_or_name: str | Path) -> Stage1StudyProfile:
    """Load a Stage 1 study profile from YAML or a configured profile name."""

    path = Path(path_or_name)
    if not path.exists():
        stem = str(path_or_name)
        if not stem.startswith("bo_stage1_"):
            stem = f"bo_stage1_{stem}"
        if not stem.endswith("_profile"):
            stem = f"{stem}_profile"
        path = CONFIG_DIR / f"{stem}.yaml"
    return Stage1StudyProfile.model_validate(load_config_mapping(path))


def run_stage1_study(profile: Stage1StudyProfile) -> list[Stage1OptimizationResult]:
    """Run all configured Stage 1 scenarios, methods, and repetitions."""

    results: list[Stage1OptimizationResult] = []
    for repetition in range(profile.repetitions):
        settings = profile.runner_settings(repetition)
        configs = [load_stage1_scenario(name) for name in profile.scenarios]
        results.extend(compare_stage1_methods(configs, profile.methods, settings))
    return results


def stage1_summary_rows(
    profile: Stage1StudyProfile,
    results: list[Stage1OptimizationResult],
) -> list[dict[str, Any]]:
    """Return flat rows for CSV or plotting."""

    rows: list[dict[str, Any]] = []
    expected_per_repetition = len(profile.scenarios) * len(profile.methods)
    for index, result in enumerate(results):
        row = result.summary_row()
        row["profile_id"] = profile.profile_id
        row["repetition"] = index // expected_per_repetition
        rows.append(row)
    return rows


def save_stage1_results(
    profile: Stage1StudyProfile,
    results: list[Stage1OptimizationResult],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save Stage 1 study results as JSON and summary rows as JSON."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    full_path = destination / "stage1_results.json"
    summary_path = destination / "stage1_summary.json"
    full_path.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(stage1_summary_rows(profile, results), indent=2),
        encoding="utf-8",
    )
    return full_path, summary_path
