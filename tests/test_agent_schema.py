import pytest
from pydantic import ValidationError

from asd_agent.agent import FINISH_OPTIMIZATION_TOOL, PROPOSE_EXPERIMENTS_TOOL
from asd_agent.models import (
    ExperimentCondition,
    FinishOptimizationDecision,
    ProposedExperimentsDecision,
)


def test_llm_tools_are_strict_function_schemas() -> None:
    assert PROPOSE_EXPERIMENTS_TOOL["type"] == "function"
    assert PROPOSE_EXPERIMENTS_TOOL["strict"] is True
    assert FINISH_OPTIMIZATION_TOOL["strict"] is True
    assert PROPOSE_EXPERIMENTS_TOOL["parameters"]["additionalProperties"] is False
    assert FINISH_OPTIMIZATION_TOOL["parameters"]["additionalProperties"] is False


def test_propose_experiments_allows_one_to_four_conditions() -> None:
    condition = ExperimentCondition(
        precursor_dose_s=1.0,
        coreactant_dose_s=1.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=20,
    )
    decision = ProposedExperimentsDecision(
        experiments=[condition, condition, condition, condition],
        rationale="A concise test rationale.",
    )
    assert len(decision.experiments) == 4

    with pytest.raises(ValidationError):
        ProposedExperimentsDecision(
            experiments=[condition, condition, condition, condition, condition],
            rationale="Too many experiments.",
        )


def test_rationale_is_limited_to_four_sentences() -> None:
    condition = ExperimentCondition(
        precursor_dose_s=1.0,
        coreactant_dose_s=1.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=20,
    )

    with pytest.raises(ValidationError):
        ProposedExperimentsDecision(
            experiments=[condition],
            rationale="One. Two. Three. Four. Five.",
        )


def test_finish_status_schema() -> None:
    decision = FinishOptimizationDecision(
        status="budget_exhausted",
        tested_experiment_id="exp_001",
        rationale="The experiment budget is exhausted.",
    )
    assert decision.status == "budget_exhausted"
