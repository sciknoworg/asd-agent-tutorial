from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.hybrid_agent import (
    FakeHybridLLM,
    FinishHybridOptimizationArgs,
    HybridAgentError,
    HybridLLMBOAgent,
    HybridToolCall,
    LiteratureHit,
    LocalLiteratureProvider,
    MockLiteratureProvider,
    RunVirtualExperimentArgs,
    SoftBoundsChange,
    hybrid_tool_schemas,
    validate_soft_bounds_change,
)
from asd_agent.bo.stage2 import Stage2Decision
from asd_agent.bo.stage2_mobo import Stage2BOSettings, Stage2CandidateProposal, Stage2Observation
from asd_agent.config import load_stage2_scenario
from asd_agent.models import Range


def fast_bo_settings() -> Stage2BOSettings:
    return Stage2BOSettings(
        experiment_budget=3,
        initial_design_size=2,
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
        acquisition_timeout_s=2.0,
        candidate_cycle_values=[30, 50, 70],
        random_fallback_points=32,
    )


def feasible_decision() -> Stage2Decision:
    return Stage2Decision(precursor_dose_s=8.0, temperature_c=175.0, cycle_count=70)


def infeasible_decision() -> Stage2Decision:
    return Stage2Decision(precursor_dose_s=0.1, temperature_c=100.0, cycle_count=5)


def agent(mode: str = "hybrid_advisory") -> HybridLLMBOAgent:
    return HybridLLMBOAgent(
        load_stage2_scenario("inherent_selectivity"),
        mode=mode,  # type: ignore[arg-type]
        llm=FakeHybridLLM(),
        bo_settings=fast_bo_settings(),
        seed=123,
    )


def add_candidate(
    hybrid: HybridLLMBOAgent,
    decision: Stage2Decision,
) -> Stage2CandidateProposal:
    proposal = Stage2CandidateProposal.create(
        decision=decision,
        optimizer="test_bo",
        seed=hybrid.seed,
    )
    hybrid.candidates[proposal.candidate_id] = proposal
    return proposal


def test_hybrid_tool_schema_prevents_arbitrary_virtual_experiment_conditions() -> None:
    schemas = {schema["name"]: schema for schema in hybrid_tool_schemas()}
    run_virtual = schemas["run_virtual_experiment"]["parameters"]

    assert "candidate_id" in run_virtual["properties"]
    assert "precursor_dose_s" not in str(run_virtual)
    with pytest.raises(ValidationError):
        RunVirtualExperimentArgs.model_validate(
            {
                "candidate_id": "cand_001",
                "precursor_dose_s": 8.0,
                "rationale": "Try to bypass BO.",
            }
        )


def test_soft_bounds_must_remain_inside_immutable_hard_bounds() -> None:
    hybrid = agent("hybrid_intervention")
    original_hard = hybrid.config.hard_bounds
    valid = SoftBoundsChange(
        precursor_dose_s=Range(min=1.0, max=7.0),
        temperature_c=Range(min=120.0, max=220.0),
        cycle_values=[30, 50],
        rationale="Narrow the search without touching hard safety bounds.",
    )
    updated = validate_soft_bounds_change(hybrid.config, hybrid.soft_bounds, valid)

    assert updated.precursor_dose_s.min == 1.0
    assert hybrid.config.hard_bounds == original_hard

    invalid = SoftBoundsChange(
        precursor_dose_s=Range(min=0.0, max=99.0),
        rationale="This exceeds immutable hard bounds.",
    )
    with pytest.raises(HybridAgentError):
        validate_soft_bounds_change(hybrid.config, hybrid.soft_bounds, invalid)


def test_candidate_execution_requires_known_immutable_candidate_id() -> None:
    hybrid = agent()
    proposal = add_candidate(hybrid, feasible_decision())

    observation = hybrid.apply_tool_call(
        HybridToolCall(
            name="run_virtual_experiment",
            arguments={
                "candidate_id": proposal.candidate_id,
                "rationale": "Execute the stored BO candidate.",
            },
        ),
        budget=3,
    )

    assert isinstance(observation, Stage2Observation)
    assert observation.experiment_id == "hybrid_001"
    with pytest.raises(HybridAgentError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="run_virtual_experiment",
                arguments={
                    "candidate_id": proposal.candidate_id,
                    "rationale": "Duplicate execution is forbidden.",
                },
            ),
            budget=3,
        )
    with pytest.raises(HybridAgentError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="run_virtual_experiment",
                arguments={
                    "candidate_id": "missing",
                    "rationale": "Unknown candidate IDs cannot execute.",
                },
            ),
            budget=3,
        )


def test_finish_requires_tested_feasible_experiment() -> None:
    hybrid = agent()
    with pytest.raises(HybridAgentError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="finish_optimization",
                arguments={
                    "tested_experiment_id": "missing",
                    "rationale": "Untested recommendations are invalid.",
                },
            ),
            budget=3,
        )

    infeasible = add_candidate(hybrid, infeasible_decision())
    hybrid.apply_tool_call(
        HybridToolCall(
            name="run_virtual_experiment",
            arguments={
                "candidate_id": infeasible.candidate_id,
                "rationale": "Execute an infeasible candidate.",
            },
        ),
        budget=3,
    )
    with pytest.raises(HybridAgentError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="finish_optimization",
                arguments={
                    "tested_experiment_id": "hybrid_001",
                    "rationale": "Infeasible recommendations are invalid.",
                },
            ),
            budget=3,
        )

    feasible = add_candidate(hybrid, feasible_decision())
    hybrid.apply_tool_call(
        HybridToolCall(
            name="run_virtual_experiment",
            arguments={
                "candidate_id": feasible.candidate_id,
                "rationale": "Execute a feasible candidate.",
            },
        ),
        budget=3,
    )
    hybrid.apply_tool_call(
        HybridToolCall(
            name="finish_optimization",
            arguments={
                "tested_experiment_id": "hybrid_002",
                "rationale": "The tested candidate is feasible.",
            },
        ),
        budget=3,
    )

    assert hybrid.state == "FINISH"
    assert hybrid.final_experiment_id == "hybrid_002"
    assert FinishHybridOptimizationArgs(
        tested_experiment_id="hybrid_002",
        rationale="The tested candidate is feasible.",
    )


def test_no_window_evidence_ids_must_exist() -> None:
    hybrid = agent()
    with pytest.raises(HybridAgentError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="declare_no_selective_window",
                arguments={
                    "evidence_experiment_ids": ["missing"],
                    "rationale": "Evidence IDs must exist.",
                },
            ),
            budget=3,
        )

    proposal = add_candidate(hybrid, infeasible_decision())
    hybrid.apply_tool_call(
        HybridToolCall(
            name="run_virtual_experiment",
            arguments={
                "candidate_id": proposal.candidate_id,
                "rationale": "Create tested evidence.",
            },
        ),
        budget=3,
    )
    hybrid.apply_tool_call(
        HybridToolCall(
            name="declare_no_selective_window",
            arguments={
                "evidence_experiment_ids": ["hybrid_001"],
                "rationale": "The tested evidence did not show feasibility.",
            },
        ),
        budget=3,
    )

    assert hybrid.state == "DECLARE_NO_WINDOW"
    assert hybrid.evidence_experiment_ids == ["hybrid_001"]


def test_oracle_values_are_not_in_llm_context() -> None:
    hybrid = agent()
    context_text = hybrid.context(budget_remaining=3).model_dump_json()

    forbidden = [
        "oracle_hypervolume",
        "selective_window_exists",
        "hidden_process_parameters",
        "max_growth_per_cycle_nm",
        "nucleation_delay_cycles",
        "surfaces",
    ]
    assert all(fragment not in context_text for fragment in forbidden)


def test_malformed_tool_call_does_not_corrupt_state() -> None:
    hybrid = agent()
    before_state = hybrid.state
    before_observations = list(hybrid.observations)
    before_candidates = dict(hybrid.candidates)

    with pytest.raises(ValidationError):
        hybrid.apply_tool_call(
            HybridToolCall(
                name="run_virtual_experiment",
                arguments={
                    "candidate_id": "missing",
                    "temperature_c": 180.0,
                    "rationale": "Malformed calls should fail validation.",
                },
            ),
            budget=3,
        )

    assert hybrid.state == before_state
    assert hybrid.observations == before_observations
    assert hybrid.candidates == before_candidates


def test_fake_llm_exercises_hybrid_intervention_transitions() -> None:
    provider = MockLiteratureProvider(
        {
            "area": [
                LiteratureHit(
                    source_id="mock_asd",
                    title="Mock ASD note",
                    summary="Local educational note for hybrid testing.",
                )
            ]
        }
    )
    hybrid = HybridLLMBOAgent(
        load_stage2_scenario("inherent_selectivity"),
        mode="hybrid_intervention",
        llm=FakeHybridLLM("intervention"),
        literature_provider=provider,
        bo_settings=fast_bo_settings(),
        seed=123,
    )

    result = hybrid.run(budget=3)
    tools = [event.tool_name for event in result.events]

    assert result.status in {"success", "budget_exhausted", "no_selective_window"}
    assert "inspect_experiment_history" in tools
    assert "query_literature" in tools
    assert "change_search_bounds" in tools
    assert "run_bayesian_optimizer" in tools
    assert "run_virtual_experiment" in tools
    assert result.literature


def test_local_literature_provider_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "literature.json"
    path.write_text(
        """
        [
          {
            "source_id": "local_1",
            "title": "Area selective deposition note",
            "summary": "A local file used for fake retrieval."
          }
        ]
        """,
        encoding="utf-8",
    )

    hits = LocalLiteratureProvider(path).query("selective deposition")

    assert [hit.source_id for hit in hits] == ["local_1"]


def test_all_hybrid_modes_are_accepted() -> None:
    config = load_stage2_scenario("inherent_selectivity")
    for mode in [
        "bo_only",
        "llm_only_legacy",
        "hybrid_advisory",
        "hybrid_intervention",
        "hybrid_explanation_only",
        "rule_based_bo",
    ]:
        hybrid = HybridLLMBOAgent(
            config,
            mode=mode,  # type: ignore[arg-type]
            llm=FakeHybridLLM("default"),
            bo_settings=fast_bo_settings(),
            seed=321,
        )
        result = hybrid.run(budget=1, max_steps=8)
        assert result.mode == mode
        assert result.status in {"success", "budget_exhausted", "no_selective_window"}
