"""Paired research-study harness for BO tutorial experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.acquisition import ThresholdAcquisitionSettings
from asd_agent.bo.hybrid_agent import (
    FakeHybridLLM,
    HybridMode,
    HybridRunResult,
    LiteratureHit,
    MockLiteratureProvider,
    run_hybrid_optimization,
)
from asd_agent.bo.optimizers import (
    Stage1GenericGPSettings,
    Stage1GridSettings,
    Stage1Method,
    Stage1PhysicsGPSettings,
    Stage1RunnerSettings,
    compare_stage1_methods,
)
from asd_agent.bo.stage2_benchmark import (
    Stage2BenchmarkMethod,
    Stage2BenchmarkProfile,
    run_stage2_method,
)
from asd_agent.bo.stage2_mobo import Stage2BOSettings, observed_hypervolume
from asd_agent.config import (
    CONFIG_DIR,
    load_config_mapping,
    load_stage1_scenario,
    load_stage2_scenario,
)

ResearchQuestion = Literal["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]
StudyArea = Literal["stage1_saturation", "stage2_asd", "hybrid_agent"]


def default_stage1_research_methods() -> list[Stage1Method]:
    """Return default Stage 1 methods for research profiles."""

    return ["generic_gp", "physics_gp"]


def default_stage2_research_methods() -> list[Stage2BenchmarkMethod]:
    """Return default Stage 2 methods for research profiles."""

    return ["random_search", "grid_search", "rule_based", "stage2_mobo"]


def default_hybrid_research_modes() -> list[HybridMode]:
    """Return default hybrid modes for research profiles."""

    return ["bo_only", "hybrid_intervention"]


class NamedSeedSet(BaseModel):
    """Named seeds for a paired research replicate."""

    simulator: int
    measurement_noise: int
    initialization: int
    bo: int
    llm: int

    model_config = ConfigDict(extra="forbid")


class PairedSeedScheduleRow(BaseModel):
    """One scenario/repetition seed schedule row."""

    study_area: StudyArea
    scenario_id: str
    repetition: int
    pair_id: str
    seeds: NamedSeedSet

    model_config = ConfigDict(extra="forbid")


class ResearchStudyProfile(BaseModel):
    """Configurable paired research-study profile."""

    profile_id: str
    description: str
    repetitions: int = Field(ge=1)
    include_stage1: bool = True
    include_stage2: bool = True
    include_hybrid: bool = False
    stage1_scenarios: list[str] = Field(default_factory=list)
    stage2_scenarios: list[str] = Field(default_factory=list)
    hybrid_scenarios: list[str] = Field(default_factory=list)
    stage1_methods: list[Stage1Method] = Field(default_factory=default_stage1_research_methods)
    stage2_methods: list[Stage2BenchmarkMethod] = Field(
        default_factory=default_stage2_research_methods
    )
    hybrid_modes: list[HybridMode] = Field(default_factory=default_hybrid_research_modes)
    stage1_budget: int = Field(default=6, ge=1)
    stage2_budget: int = Field(default=4, ge=1)
    hybrid_budget: int = Field(default=4, ge=1)
    stage1_candidate_grid_size: int = Field(default=31, ge=2)
    stage2_initial_design_size: int = Field(default=2, ge=0)
    stage2_candidate_cycle_values: list[int] = Field(default_factory=lambda: [30, 50, 70])
    mobo_qmc_samples: int = Field(default=8, ge=4)
    mobo_num_restarts: int = Field(default=1, ge=1)
    mobo_raw_samples: int = Field(default=8, ge=4)
    mobo_acquisition_timeout_s: float | None = Field(default=2.0, gt=0.0)
    random_fallback_points: int = Field(default=32, ge=1)
    seed_base: int = 9009
    bootstrap_iterations: int = Field(default=1000, ge=10)

    model_config = ConfigDict(extra="forbid")

    def bo_settings(self, budget: int) -> Stage2BOSettings:
        """Return matched MOBO settings for Stage 2 or hybrid runs."""

        return Stage2BOSettings(
            experiment_budget=budget,
            initial_design_size=max(self.stage2_initial_design_size, 2),
            qmc_samples=self.mobo_qmc_samples,
            num_restarts=self.mobo_num_restarts,
            raw_samples=self.mobo_raw_samples,
            acquisition_timeout_s=self.mobo_acquisition_timeout_s,
            candidate_cycle_values=list(self.stage2_candidate_cycle_values),
            random_fallback_points=self.random_fallback_points,
        )


class ResearchResultRow(BaseModel):
    """One normalized row from Stage 1, Stage 2, or hybrid research runs."""

    profile_id: str
    research_question: ResearchQuestion
    study_area: StudyArea
    pair_id: str
    repetition: int
    scenario_id: str
    method: str
    seeds: NamedSeedSet
    status: str
    success: bool
    n_experiments: int
    primary_metric_name: str
    primary_metric_value: float
    failure_category: str
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


def load_research_profile(path_or_name: str | Path) -> ResearchStudyProfile:
    """Load a research profile from YAML or a configured profile name."""

    path = Path(path_or_name)
    if not path.exists():
        stem = str(path_or_name)
        if not stem.startswith("bo_research_"):
            stem = f"bo_research_{stem}"
        if not stem.endswith("_profile"):
            stem = f"{stem}_profile"
        path = CONFIG_DIR / f"{stem}.yaml"
    return ResearchStudyProfile.model_validate(load_config_mapping(path))


def paired_seed_schedule(profile: ResearchStudyProfile) -> list[PairedSeedScheduleRow]:
    """Return deterministic named seeds for all paired scenario repetitions."""

    rows: list[PairedSeedScheduleRow] = []
    for area, scenarios in [
        ("stage1_saturation", profile.stage1_scenarios if profile.include_stage1 else []),
        ("stage2_asd", profile.stage2_scenarios if profile.include_stage2 else []),
        ("hybrid_agent", profile.hybrid_scenarios if profile.include_hybrid else []),
    ]:
        for scenario_index, scenario in enumerate(scenarios):
            for repetition in range(profile.repetitions):
                base = (
                    profile.seed_base + area_offset(area) + scenario_index * 1000 + repetition * 10
                )
                pair_id = f"{area}:{scenario}:rep_{repetition:03d}"
                rows.append(
                    PairedSeedScheduleRow(
                        study_area=area,  # type: ignore[arg-type]
                        scenario_id=scenario,
                        repetition=repetition,
                        pair_id=pair_id,
                        seeds=NamedSeedSet(
                            simulator=base,
                            measurement_noise=base + 1,
                            initialization=base + 2,
                            bo=base + 3,
                            llm=base + 4,
                        ),
                    )
                )
    return rows


def run_research_study(profile: ResearchStudyProfile) -> list[ResearchResultRow]:
    """Run a configured paired research study profile."""

    rows: list[ResearchResultRow] = []
    for schedule in paired_seed_schedule(profile):
        if schedule.study_area == "stage1_saturation":
            rows.extend(run_stage1_research_pair(profile, schedule))
        elif schedule.study_area == "stage2_asd":
            rows.extend(run_stage2_research_pair(profile, schedule))
        else:
            rows.extend(run_hybrid_research_pair(profile, schedule))
    return rows


def run_stage1_research_pair(
    profile: ResearchStudyProfile,
    schedule: PairedSeedScheduleRow,
) -> list[ResearchResultRow]:
    """Run Stage 1 methods with matched seeds."""

    config = load_stage1_scenario(schedule.scenario_id)
    acquisition = ThresholdAcquisitionSettings(
        candidate_grid_size=profile.stage1_candidate_grid_size
    )
    settings = Stage1RunnerSettings(
        budget=profile.stage1_budget,
        grid=Stage1GridSettings(n_points=profile.stage1_candidate_grid_size),
        generic_gp=Stage1GenericGPSettings(acquisition=acquisition),
        physics_gp=Stage1PhysicsGPSettings(acquisition=acquisition),
        simulator_seed=schedule.seeds.measurement_noise,
        optimizer_seed=schedule.seeds.bo,
    )
    results = compare_stage1_methods([config], profile.stage1_methods, settings)
    rows: list[ResearchResultRow] = []
    for result in results:
        summary = result.summary_row()
        primary = numeric(summary.get("absolute_t95_error_s"), default=float("nan"))
        rows.append(
            ResearchResultRow(
                profile_id=profile.profile_id,
                research_question="RQ1",
                study_area="stage1_saturation",
                pair_id=schedule.pair_id,
                repetition=schedule.repetition,
                scenario_id=result.scenario_id,
                method=result.method,
                seeds=schedule.seeds,
                status=result.status,
                success=bool(summary["success"]),
                n_experiments=int(numeric(summary.get("n_experiments"))),
                primary_metric_name="absolute_t95_error_s",
                primary_metric_value=primary,
                failure_category=result.failure_category,
                metrics=coerce_metrics(summary),
            )
        )
    return rows


def run_stage2_research_pair(
    profile: ResearchStudyProfile,
    schedule: PairedSeedScheduleRow,
) -> list[ResearchResultRow]:
    """Run Stage 2 baseline and MOBO methods with matched seeds."""

    config = load_stage2_scenario(schedule.scenario_id)
    benchmark_profile = Stage2BenchmarkProfile(
        profile_id=f"{profile.profile_id}_{schedule.scenario_id}",
        description="Research harness Stage 2 subprofile.",
        scenarios=[schedule.scenario_id],
        methods=profile.stage2_methods,
        repetitions=1,
        budget=profile.stage2_budget,
        initial_design_size=profile.stage2_initial_design_size,
        simulator_seed=schedule.seeds.measurement_noise,
        optimizer_seed=schedule.seeds.bo,
        candidate_cycle_values=list(profile.stage2_candidate_cycle_values),
        mobo_qmc_samples=profile.mobo_qmc_samples,
        mobo_num_restarts=profile.mobo_num_restarts,
        mobo_raw_samples=profile.mobo_raw_samples,
        mobo_acquisition_timeout_s=profile.mobo_acquisition_timeout_s,
        random_fallback_points=profile.random_fallback_points,
    )
    rows: list[ResearchResultRow] = []
    for method in profile.stage2_methods:
        result = run_stage2_method(config, method, benchmark_profile, 0)
        summary = result.summary_row()
        rows.append(
            ResearchResultRow(
                profile_id=profile.profile_id,
                research_question="RQ2",
                study_area="stage2_asd",
                pair_id=schedule.pair_id,
                repetition=schedule.repetition,
                scenario_id=result.scenario_id,
                method=result.method,
                seeds=schedule.seeds,
                status=result.status,
                success=bool(summary["success"]),
                n_experiments=int(numeric(summary.get("n_experiments"))),
                primary_metric_name="hypervolume_auc",
                primary_metric_value=float(result.hypervolume_auc),
                failure_category=result.failure_category,
                metrics=coerce_metrics(summary),
            )
        )
    return rows


def run_hybrid_research_pair(
    profile: ResearchStudyProfile,
    schedule: PairedSeedScheduleRow,
) -> list[ResearchResultRow]:
    """Run BO-only and hybrid modes with matched seeds."""

    config = load_stage2_scenario(schedule.scenario_id)
    settings = profile.bo_settings(profile.hybrid_budget)
    literature = MockLiteratureProvider(
        {
            "area": [
                LiteratureHit(
                    source_id="mock_research_note",
                    title="Mock hybrid ASD note",
                    summary="Local deterministic note for paired hybrid studies.",
                )
            ]
        }
    )
    rows: list[ResearchResultRow] = []
    for mode in profile.hybrid_modes:
        result = run_hybrid_optimization(
            config,
            mode=mode,
            llm=FakeHybridLLM("intervention"),
            literature_provider=literature,
            bo_settings=settings,
            seed=schedule.seeds.llm if mode.startswith("hybrid") else schedule.seeds.bo,
            budget=profile.hybrid_budget,
        )
        rows.append(
            hybrid_result_row(profile, schedule, config.scenario_id, mode, settings, result)
        )
    return rows


def hybrid_result_row(
    profile: ResearchStudyProfile,
    schedule: PairedSeedScheduleRow,
    scenario_id: str,
    mode: HybridMode,
    settings: Stage2BOSettings,
    result: HybridRunResult,
) -> ResearchResultRow:
    """Convert a hybrid result into a normalized research row."""

    final_hv = observed_hypervolume(
        load_stage2_scenario(scenario_id), result.observations, settings
    )
    metrics: dict[str, float | int | str | bool] = {
        "final_hypervolume": final_hv,
        "candidate_count": len(result.candidates),
        "literature_count": len(result.literature),
        "event_count": len(result.events),
        "final_experiment_id": result.final_experiment_id or "",
    }
    return ResearchResultRow(
        profile_id=profile.profile_id,
        research_question="RQ3",
        study_area="hybrid_agent",
        pair_id=schedule.pair_id,
        repetition=schedule.repetition,
        scenario_id=scenario_id,
        method=mode,
        seeds=schedule.seeds,
        status=result.status,
        success=result.status == "success",
        n_experiments=len(result.observations),
        primary_metric_name="final_hypervolume",
        primary_metric_value=final_hv,
        failure_category=result.status,
        metrics=metrics,
    )


def save_research_rows(
    profile: ResearchStudyProfile,
    rows: Sequence[ResearchResultRow],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save research rows as JSON and CSV."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "research_results.json"
    csv_path = destination / "research_results.csv"
    json_path.write_text(
        json.dumps(
            {
                "profile": profile.model_dump(mode="json"),
                "rows": [row.model_dump(mode="json") for row in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_research_csv(rows, csv_path)
    return json_path, csv_path


def write_research_csv(rows: Sequence[ResearchResultRow], path: Path) -> None:
    """Write normalized research rows to CSV."""

    fieldnames = [
        "profile_id",
        "research_question",
        "study_area",
        "pair_id",
        "repetition",
        "scenario_id",
        "method",
        "status",
        "success",
        "n_experiments",
        "primary_metric_name",
        "primary_metric_value",
        "failure_category",
        "simulator_seed",
        "measurement_noise_seed",
        "initialization_seed",
        "bo_seed",
        "llm_seed",
        "metrics_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "profile_id": row.profile_id,
                    "research_question": row.research_question,
                    "study_area": row.study_area,
                    "pair_id": row.pair_id,
                    "repetition": row.repetition,
                    "scenario_id": row.scenario_id,
                    "method": row.method,
                    "status": row.status,
                    "success": row.success,
                    "n_experiments": row.n_experiments,
                    "primary_metric_name": row.primary_metric_name,
                    "primary_metric_value": row.primary_metric_value,
                    "failure_category": row.failure_category,
                    "simulator_seed": row.seeds.simulator,
                    "measurement_noise_seed": row.seeds.measurement_noise,
                    "initialization_seed": row.seeds.initialization,
                    "bo_seed": row.seeds.bo,
                    "llm_seed": row.seeds.llm,
                    "metrics_json": json.dumps(row.metrics, sort_keys=True),
                }
            )


def coerce_metrics(summary: dict[str, object]) -> dict[str, float | int | str | bool]:
    """Keep JSON/CSV friendly scalar metrics."""

    metrics: dict[str, float | int | str | bool] = {}
    for key, value in summary.items():
        if isinstance(value, str | int | float | bool):
            metrics[key] = value
    return metrics


def numeric(value: object, *, default: float = 0.0) -> float:
    """Convert scalar values to float, using a default for blanks."""

    if value in ("", None):
        return default
    if isinstance(value, str | int | float | bool):
        return float(value)
    return default


def area_offset(area: str) -> int:
    """Return deterministic seed offsets by study area."""

    if area == "stage1_saturation":
        return 10_000
    if area == "stage2_asd":
        return 20_000
    return 30_000
