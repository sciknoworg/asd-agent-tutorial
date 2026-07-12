from pathlib import Path

import pytest

from asd_agent.benchmark import run_method
from asd_agent.bo.acquisition import estimate_stage1_target_growth
from asd_agent.bo.hybrid_agent import (
    HybridAgentError,
    HybridLLMBOAgent,
    HybridSafetySettings,
    HybridToolCall,
    LocalLiteratureProvider,
    run_hybrid_optimization,
)
from asd_agent.bo.optimizers import (
    Stage1CandidateProposal,
    Stage1RunnerSettings,
    optimizer_visible_recommendation,
    run_stage1_optimization,
)
from asd_agent.bo.oracle import Stage1Recommendation, evaluate_recommendation
from asd_agent.bo.records import CandidateProposal, RunManifest
from asd_agent.bo.stage1 import Stage1ExperimentRecord
from asd_agent.bo.stage2 import Stage2Decision
from asd_agent.bo.stage2_mobo import (
    Stage2BOSettings,
    observe_stage2_decision,
    run_stage2_bo,
    stage2_reference_point,
)
from asd_agent.bo.statistics import descriptive_summary, wilson_interval
from asd_agent.cli import build_parser
from asd_agent.config import load_stage1_scenario, load_stage2_scenario
from asd_agent.models import ExperimentCondition, Range


def test_known_target_is_an_absolute_growth_threshold() -> None:
    config = load_stage1_scenario("fast_mono")

    assert config.objective.target_growth == pytest.approx(0.95)
    assert estimate_stage1_target_growth(config, []) == pytest.approx(0.95)


def test_false_saturation_detects_a_recommendation_below_true_threshold() -> None:
    config = load_stage1_scenario("fast_mono")
    record = Stage1ExperimentRecord(
        experiment_id="low",
        dose_s=0.1,
        observed_growth=0.1,
        process_time_s=1.0,
    )
    metrics = evaluate_recommendation(
        config,
        Stage1Recommendation(recommended_dose_s=0.1, estimated_t95_s=0.1),
        [record],
    )

    assert metrics.false_saturation_declaration


def test_optimizer_visible_stopping_does_not_call_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_stage1_scenario("fast_mono")
    records = [
        Stage1ExperimentRecord(
            experiment_id=f"e{index}", dose_s=dose, observed_growth=growth, process_time_s=1.0
        )
        for index, (dose, growth) in enumerate([(1.0, 0.8), (1.5, 0.93), (1.8, 0.96)])
    ]
    proposals = [
        Stage1CandidateProposal.create(
            dose_s=1.8,
            optimizer="test",
            posterior_summaries={"target_growth": 0.95},
        )
    ]
    monkeypatch.setattr(
        "asd_agent.bo.oracle.Stage1EvaluationOracle.evaluate",
        lambda _self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("oracle leak")),
    )

    recommendation = optimizer_visible_recommendation(
        config,
        records,
        proposals,
        Stage1RunnerSettings(stopping_tolerance_s=0.5),
    )

    assert recommendation is not None
    assert recommendation[1] == "e2"


def test_hybrid_bo_request_does_not_execute_a_hidden_subrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage2_scenario("inherent_selectivity")
    agent = HybridLLMBOAgent(config, seed=4)
    monkeypatch.setattr(
        "asd_agent.bo.hybrid_agent.run_stage2_bo",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hidden subrun")),
    )

    proposals = agent._run_bo_candidates(1)

    assert len(proposals) == 1
    assert not agent.observations


def test_no_window_policy_can_require_budget_exhaustion() -> None:
    agent = HybridLLMBOAgent(
        load_stage2_scenario("impossible_selectivity"),
        safety_settings=HybridSafetySettings(require_budget_exhaustion_for_no_window=True),
    )
    agent.observations = [
        observe_stage2_decision(
            agent.config,
            Stage2Decision(precursor_dose_s=1.0, temperature_c=180.0, cycle_count=30),
            experiment_id="evidence_001",
            seed=1,
        )
    ]

    with pytest.raises(HybridAgentError, match="budget exhaustion"):
        agent.apply_tool_call(
            HybridToolCall(
                name="declare_no_selective_window",
                arguments={
                    "evidence_experiment_ids": ["evidence_001"],
                    "rationale": "No tested window was found.",
                },
            ),
            budget=3,
        )


def test_reference_point_rejects_favorable_values() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    with pytest.raises(ValueError, match="reference GA"):
        stage2_reference_point(config, Stage2BOSettings(reference_point=[99.0, -1.0, -2000.0]))


def test_shared_records_cover_required_manifest_and_candidate_fields() -> None:
    condition = ExperimentCondition(
        precursor_dose_s=1.0,
        coreactant_dose_s=1.0,
        temperature_c=180.0,
        cycles=10,
    )
    proposal = CandidateProposal.create(
        condition,
        "test",
        model_version="model-v1",
        feasibility_probability=0.7,
    )
    manifest = RunManifest.create(
        config_path=Path("configs/inherent_selectivity.yaml"),
        method="test",
        scenario="inherent_selectivity",
        experiment_budget=2,
        named_seeds={"bo": 1},
        acquisition_function="threshold_probability",
        model_settings={"kernel": "matern"},
        llm_model="none",
        token_usage={"total_tokens": 0},
    )

    assert proposal.model_version == "model-v1"
    assert proposal.feasibility_probability == pytest.approx(0.7)
    assert manifest.acquisition_function == "threshold_probability"
    assert manifest.model_settings == {"kernel": "matern"}


def test_integrated_cli_exposes_required_workflow_groups() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in ("stage1", "stage2", "hybrid", "study", "lab"):
        assert command in help_text


def test_wilson_success_interval_is_finite_and_bounded() -> None:
    low, high = wilson_interval(7, 10)

    assert 0.0 <= low < 0.7 < high <= 1.0
    assert descriptive_summary([]) == []


def test_top_level_stage1_runner_resumes_saved_state() -> None:
    config = load_stage1_scenario("fast_mono")
    first = run_stage1_optimization(
        config,
        "grid",
        Stage1RunnerSettings(budget=3, min_recommendation_observations=4),
    )
    resumed = run_stage1_optimization(
        config,
        "grid",
        Stage1RunnerSettings(budget=5, min_recommendation_observations=6),
        initial_records=first.records,
        optimizer_state=first.optimizer_state,
    )

    assert len(resumed.records) >= len(first.records)
    assert len({record.dose_s for record in resumed.records}) == len(resumed.records)
    assert resumed.optimizer_state.observation_ids == [
        record.experiment_id for record in resumed.records
    ]


def test_top_level_stage2_runner_resumes_saved_state() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    first_settings = Stage2BOSettings(
        experiment_budget=2,
        initial_design_size=2,
        candidate_cycle_values=[30, 50],
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
    )
    first = run_stage2_bo(config, first_settings, simulator_seed=1, optimizer_seed=2)
    resumed = run_stage2_bo(
        config,
        first_settings.model_copy(update={"experiment_budget": 3}),
        simulator_seed=1,
        optimizer_seed=2,
        initial_observations=first.observations,
        optimizer_state=first.optimizer_state,
    )

    assert len(resumed.observations) == 3
    assert len(resumed.hypervolume_by_iteration) == 3
    assert resumed.optimizer_state.observation_ids == [
        observation.experiment_id for observation in resumed.observations
    ]


def test_local_literature_provider_supports_documented_formats(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text(
        "# Saturation note\nThreshold uncertainty matters.", encoding="utf-8"
    )
    (tmp_path / "records.jsonl").write_text(
        '{"source_id":"jsonl-1","title":"ASD window","summary":"Bound review"}\n',
        encoding="utf-8",
    )

    provider = LocalLiteratureProvider(tmp_path)

    assert provider.query("threshold")[0].source_id == "note.md"
    assert provider.query("bound")[0].source_id == "jsonl-1"


def test_bound_changes_reject_nonexistent_evidence_ids() -> None:
    agent = HybridLLMBOAgent(
        load_stage2_scenario("inherent_selectivity"), mode="hybrid_intervention"
    )

    with pytest.raises(HybridAgentError, match="bound-change evidence"):
        agent.apply_tool_call(
            HybridToolCall(
                name="change_search_bounds",
                arguments={
                    "evidence_experiment_ids": ["missing"],
                    "duration_steps": 2,
                    "rationale": "Narrow the search using tested evidence.",
                },
            ),
            budget=3,
        )


def test_legacy_grid_benchmark_honors_the_requested_budget(tmp_path: Path) -> None:
    run = run_method(
        load_stage2_scenario("inherent_selectivity").process,
        "grid_search",
        budget=4,
        seed=1,
        run_dir=tmp_path,
    )

    assert len(run.records) <= 4


def test_nonfinite_conditions_and_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentCondition(
            precursor_dose_s=1.0,
            coreactant_dose_s=1.0,
            temperature_c=float("nan"),
            cycles=10,
        )
    with pytest.raises(ValueError):
        Range(min=0.0, max=float("inf"))


def test_explanation_only_mode_does_not_change_bo_decisions() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    settings = Stage2BOSettings(
        experiment_budget=2,
        initial_design_size=2,
        candidate_cycle_values=[30, 50],
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
    )
    bo_only = run_hybrid_optimization(
        config,
        mode="bo_only",
        bo_settings=settings,
        simulator_seed=11,
        optimizer_seed=12,
        llm_seed=13,
        budget=2,
    )
    explanation_only = run_hybrid_optimization(
        config,
        mode="hybrid_explanation_only",
        bo_settings=settings,
        simulator_seed=11,
        optimizer_seed=12,
        llm_seed=13,
        budget=2,
    )

    assert [row.decision for row in explanation_only.observations] == [
        row.decision for row in bo_only.observations
    ]
    assert [row.outcomes for row in explanation_only.observations] == [
        row.outcomes for row in bo_only.observations
    ]
