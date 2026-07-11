import pytest

from asd_agent.agent import validate_finish_recommendation
from asd_agent.config import load_scenario
from asd_agent.experiment_loop import run_agent_loop
from asd_agent.heuristic_agent import RuleBasedAgent
from asd_agent.models import FinishOptimizationDecision


def test_rule_based_agent_succeeds_on_inherent_scenario() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    run = run_agent_loop(config, RuleBasedAgent(), budget=12, seed=1, method="rule_based")

    assert run.status == "success"
    assert any(record.meets_objective for record in run.records)


def test_final_recommendation_must_reference_tested_experiment() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    run = run_agent_loop(config, RuleBasedAgent(), budget=12, seed=1, method="rule_based")
    bad_decision = FinishOptimizationDecision(
        status="success",
        tested_experiment_id="not_in_ledger",
        rationale="This id was never tested.",
    )

    with pytest.raises(ValueError, match="untested experiment id"):
        validate_finish_recommendation(bad_decision, run.records)


def test_final_recommendation_accepts_tested_experiment() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    run = run_agent_loop(config, RuleBasedAgent(), budget=12, seed=1, method="rule_based")
    tested_id = run.records[-1].experiment_id
    decision = FinishOptimizationDecision(
        status="success",
        tested_experiment_id=tested_id,
        rationale="This id is in the ledger.",
    )

    validate_finish_recommendation(decision, run.records)
