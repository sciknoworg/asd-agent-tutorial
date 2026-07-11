"""Deterministic no-API tutorial agent."""

from __future__ import annotations

from asd_agent.agent import validate_finish_recommendation
from asd_agent.models import (
    ExperimentCondition,
    ExperimentRecord,
    FinishOptimizationDecision,
    FinishStatus,
    ProcessConfig,
    ProposedExperimentsDecision,
)
from asd_agent.objective import best_success, classify_failure


class RuleBasedAgent:
    """Small deterministic agent that mimics a cautious experimental plan."""

    def next_decision(
        self,
        config: ProcessConfig,
        ledger: list[ExperimentRecord],
        budget_remaining: int,
    ) -> ProposedExperimentsDecision | FinishOptimizationDecision:
        success = best_success(ledger)
        if success is not None:
            decision = FinishOptimizationDecision(
                status="success",
                tested_experiment_id=success.experiment_id,
                rationale=(
                    "A tested condition satisfies GA thickness, NGA limit, "
                    "selectivity, and safety constraints."
                ),
            )
            validate_finish_recommendation(decision, ledger)
            return decision

        candidates = [
            condition for condition in candidate_plan(config) if not was_tested(condition, ledger)
        ]
        if not candidates or budget_remaining <= 0:
            status: FinishStatus = (
                "no_selective_window"
                if classify_failure(ledger, config) == "no_selective_window"
                else "budget_exhausted"
            )
            tested_id = ledger[-1].experiment_id if ledger else "none"
            decision = FinishOptimizationDecision(
                status=status,
                tested_experiment_id=tested_id,
                rationale=(
                    "The tested conditions did not reveal a safe selective "
                    "window within the remaining budget."
                ),
            )
            if ledger:
                validate_finish_recommendation(decision, ledger)
            return decision

        batch = candidates[: min(4, budget_remaining)]
        return ProposedExperimentsDecision(
            experiments=batch,
            rationale=(
                "Test saturation at increasing dose, then use inhibitor and "
                "temperature sweeps if selectivity remains poor."
            ),
        )


def candidate_plan(config: ProcessConfig) -> list[ExperimentCondition]:
    """Deterministic experiment sequence used by the rule-based agent."""

    safety = config.safety
    temp_mid = safety.temperature_c.midpoint()
    temp_low = safety.temperature_c.lerp(0.30)
    temp_high = safety.temperature_c.lerp(0.70)
    inhibitor_mid = safety.inhibitor_dose_s.lerp(0.50)
    inhibitor_high = safety.inhibitor_dose_s.max
    cycles_mid = int(round(safety.cycles.lerp(0.60)))
    cycles_high = int(round(safety.cycles.max))

    raw = [
        (1.0, 1.0, 0.0, temp_mid, 30),
        (3.0, 3.0, 0.0, temp_mid, 50),
        (6.0, 6.0, 0.0, temp_mid, 70),
        (4.0, 4.0, 1.0, temp_mid, 70),
        (4.0, 4.0, inhibitor_mid, temp_mid, cycles_high),
        (6.0, 6.0, 3.0, temp_mid, cycles_high),
        (6.0, 6.0, 3.0, temp_low, cycles_high),
        (6.0, 6.0, 3.0, temp_high, cycles_high),
        (8.0, 8.0, inhibitor_high, temp_mid, cycles_high),
        (2.0, 6.0, inhibitor_high, temp_mid, cycles_high),
        (6.0, 2.0, inhibitor_high, temp_mid, cycles_high),
        (8.0, 8.0, 0.0, temp_mid, cycles_mid),
    ]
    return [
        ExperimentCondition(
            precursor_dose_s=clip(
                precursor, safety.precursor_dose_s.min, safety.precursor_dose_s.max
            ),
            coreactant_dose_s=clip(
                coreactant, safety.coreactant_dose_s.min, safety.coreactant_dose_s.max
            ),
            inhibitor_dose_s=clip(
                inhibitor, safety.inhibitor_dose_s.min, safety.inhibitor_dose_s.max
            ),
            temperature_c=clip(temperature, safety.temperature_c.min, safety.temperature_c.max),
            cycles=int(clip(cycles, safety.cycles.min, safety.cycles.max)),
        )
        for precursor, coreactant, inhibitor, temperature, cycles in raw
    ]


def was_tested(condition: ExperimentCondition, ledger: list[ExperimentRecord]) -> bool:
    tested = {record.condition.rounded_key() for record in ledger}
    return condition.rounded_key() in tested


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
