from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.research import (
    NamedSeedSet,
    ResearchResultRow,
    ResearchStudyProfile,
    load_research_profile,
    paired_seed_schedule,
    run_research_study,
)
from asd_agent.bo.statistics import (
    analyze_research_results,
    default_comparison_specs,
    holm_adjust,
    save_research_analysis,
)


def seed_set(offset: int = 0) -> NamedSeedSet:
    return NamedSeedSet(
        simulator=100 + offset,
        measurement_noise=200 + offset,
        initialization=300 + offset,
        bo=400 + offset,
        llm=500 + offset,
    )


def research_row(
    *,
    pair_id: str,
    method: str,
    value: float,
    success: bool,
    study_area: str = "stage2_asd",
    research_question: str = "RQ2",
    metric_name: str = "hypervolume_auc",
    scenario_id: str = "bo_stage2_inherent_selectivity",
    repetition: int = 0,
    failure_category: str = "success",
) -> ResearchResultRow:
    return ResearchResultRow(
        profile_id="test",
        research_question=research_question,  # type: ignore[arg-type]
        study_area=study_area,  # type: ignore[arg-type]
        pair_id=pair_id,
        repetition=repetition,
        scenario_id=scenario_id,
        method=method,
        seeds=seed_set(repetition),
        status="success" if success else "budget_exhausted",
        success=success,
        n_experiments=3,
        primary_metric_name=metric_name,
        primary_metric_value=value,
        failure_category=failure_category,
        metrics={
            metric_name: value,
            "experiments_to_first_feasible": 2 if success else "",
        },
    )


def synthetic_rows() -> list[ResearchResultRow]:
    rows: list[ResearchResultRow] = []
    for repetition, pair_id in enumerate(["pair_0", "pair_1", "pair_2"]):
        rows.append(
            research_row(
                pair_id=pair_id,
                method="random_search",
                value=float(repetition),
                success=repetition == 0,
                repetition=repetition,
                failure_category="success" if repetition == 0 else "budget_exhausted",
            )
        )
        rows.append(
            research_row(
                pair_id=pair_id,
                method="stage2_mobo",
                value=float(repetition + 2),
                success=True,
                repetition=repetition,
            )
        )
        rows.append(
            research_row(
                pair_id=f"stage1_{pair_id}",
                method="generic_gp",
                value=2.0 + repetition,
                success=True,
                study_area="stage1_saturation",
                research_question="RQ1",
                metric_name="absolute_t95_error_s",
                scenario_id="bo_stage1_fast_mono",
                repetition=repetition,
            )
        )
        rows.append(
            research_row(
                pair_id=f"stage1_{pair_id}",
                method="physics_gp",
                value=1.0 + repetition,
                success=True,
                study_area="stage1_saturation",
                research_question="RQ1",
                metric_name="absolute_t95_error_s",
                scenario_id="bo_stage1_fast_mono",
                repetition=repetition,
            )
        )
        rows.append(
            research_row(
                pair_id=f"hybrid_{pair_id}",
                method="bo_only",
                value=1.0 + repetition,
                success=True,
                study_area="hybrid_agent",
                research_question="RQ3",
                metric_name="final_hypervolume",
                repetition=repetition,
            )
        )
        rows.append(
            research_row(
                pair_id=f"hybrid_{pair_id}",
                method="hybrid_intervention",
                value=1.5 + repetition,
                success=True,
                study_area="hybrid_agent",
                research_question="RQ3",
                metric_name="final_hypervolume",
                repetition=repetition,
            )
        )
    return rows


def test_research_profiles_and_paired_seed_schedule_are_deterministic() -> None:
    smoke = load_research_profile("smoke")
    pilot = load_research_profile("pilot")
    paper_non_llm = load_research_profile("paper_non_llm")
    paper_llm = load_research_profile("paper_llm")

    first = paired_seed_schedule(smoke)
    second = paired_seed_schedule(smoke)

    assert smoke.repetitions == 2
    assert pilot.repetitions == 20
    assert paper_non_llm.repetitions == 100
    assert paper_llm.repetitions == 30
    assert first == second
    assert len({row.pair_id for row in first}) == len(first)
    assert all(row.seeds.bo != row.seeds.llm for row in first)


def test_statistical_analysis_generation_and_holm_correction(tmp_path: Path) -> None:
    rows = synthetic_rows()
    specs = default_comparison_specs(rows)
    analysis = analyze_research_results(rows, bootstrap_iterations=50, seed=42)
    paths = save_research_analysis(rows, tmp_path, bootstrap_iterations=50, seed=42)

    assert {spec.research_question for spec in specs} == {"RQ1", "RQ2", "RQ3"}
    assert analysis.comparisons
    assert all(result.n_pairs == 3 for result in analysis.comparisons)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert all(path.exists() for path in paths.values())
    assert "Paired Effects" in paths["markdown"].read_text(encoding="utf-8")
    assert "\\begin{tabular}" in paths["latex"].read_text(encoding="utf-8")


def test_empty_research_analysis_outputs_stable_files(tmp_path: Path) -> None:
    paths = save_research_analysis([], tmp_path, bootstrap_iterations=10, seed=1)

    assert paths["json"].read_text(encoding="utf-8")
    assert paths["comparisons_csv"].read_text(encoding="utf-8").startswith("comparison_id")
    assert "Research Statistics" in paths["markdown"].read_text(encoding="utf-8")


def test_tiny_research_run_reproducibility() -> None:
    profile = ResearchStudyProfile(
        profile_id="tiny_repro",
        description="Tiny deterministic Stage 2-only reproducibility profile.",
        repetitions=1,
        include_stage1=False,
        include_stage2=True,
        include_hybrid=False,
        stage2_scenarios=["bo_stage2_inherent_selectivity"],
        stage2_methods=["random_search", "grid_search"],
        stage2_budget=2,
        stage2_initial_design_size=1,
        seed_base=1234,
        bootstrap_iterations=10,
    )

    first = run_research_study(profile)
    second = run_research_study(profile)

    stable_first = [
        (row.pair_id, row.method, row.status, row.n_experiments, row.primary_metric_value)
        for row in first
    ]
    stable_second = [
        (row.pair_id, row.method, row.status, row.n_experiments, row.primary_metric_value)
        for row in second
    ]
    assert stable_first == stable_second
    assert {row.method for row in first} == {"random_search", "grid_search"}
