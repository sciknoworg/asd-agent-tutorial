"""Reusable optimization loop and run persistence."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from asd_agent.agent import LLMOptimizationAgent, validate_finish_recommendation
from asd_agent.models import (
    ExperimentCondition,
    ExperimentRecord,
    FinishOptimizationDecision,
    FinishStatus,
    OptimizationRun,
    ProcessConfig,
    ProposedExperimentsDecision,
)
from asd_agent.objective import classify_failure, validate_safety
from asd_agent.simulator import VirtualLab


class OptimizationAgent(Protocol):
    """Agent interface used by the virtual lab loop."""

    def next_decision(
        self,
        config: ProcessConfig,
        ledger: list[ExperimentRecord],
        budget_remaining: int,
    ) -> ProposedExperimentsDecision | FinishOptimizationDecision: ...


def run_agent_loop(
    config: ProcessConfig,
    agent: OptimizationAgent,
    budget: int = 12,
    seed: int | None = None,
    run_dir: str | Path | None = None,
    method: str = "agent",
    model: str | None = None,
) -> OptimizationRun:
    """Run an agent until success, failure, or budget exhaustion."""

    lab = VirtualLab(config, seed=seed)
    ledger: list[ExperimentRecord] = []
    output_dir = Path(run_dir) if run_dir else None
    token_usage: dict[str, int] = {}
    started_at = datetime.now(UTC).isoformat()

    while len(ledger) < budget:
        decision = agent.next_decision(config, ledger, budget - len(ledger))
        if isinstance(decision, FinishOptimizationDecision):
            if ledger:
                validate_finish_recommendation(decision, ledger)
            run = build_run(
                method,
                config,
                decision.status,
                ledger,
                seed or config.seed,
                model,
                token_usage,
                started_at,
            )
            persist_run(run, config, output_dir)
            return run

        for condition in decision.experiments[: budget - len(ledger)]:
            violations = validate_safety(condition, config)
            if violations:
                raise ValueError(f"unsafe proposed experiment: {violations}")
            experiment_id = f"{method}_{len(ledger) + 1:03d}"
            record = lab.simulate(condition, experiment_id, decision.rationale)
            ledger.append(record)
            if isinstance(agent, LLMOptimizationAgent):
                merge_token_usage(token_usage, agent.last_token_usage)
            run = build_run(
                method,
                config,
                "budget_exhausted",
                ledger,
                seed or config.seed,
                model,
                token_usage,
                started_at,
            )
            persist_run(run, config, output_dir)

    status: FinishStatus = (
        "success" if any(record.meets_objective for record in ledger) else "budget_exhausted"
    )
    failure_category = classify_failure(ledger, config)
    if failure_category == "no_selective_window":
        status = "no_selective_window"
    run = build_run(
        method,
        config,
        status,
        ledger,
        seed or config.seed,
        model,
        token_usage,
        started_at,
    )
    persist_run(run, config, output_dir)
    return run


def build_run(
    method: str,
    config: ProcessConfig,
    status: FinishStatus,
    records: list[ExperimentRecord],
    seed: int,
    model: str | None,
    token_usage: dict[str, int],
    started_at: str,
) -> OptimizationRun:
    """Create a serializable run object."""

    failure_category = classify_failure(records, config)
    if status == "success":
        failure_category = "success"
    return OptimizationRun(
        method=method,
        scenario=config.scenario,
        status=status,
        records=records,
        seed=seed,
        model=model,
        token_usage=token_usage,
        failure_category=failure_category,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
    )


def persist_run(run: OptimizationRun, config: ProcessConfig, run_dir: Path | None) -> None:
    """Persist ledger CSV/JSON and metadata after every experiment."""

    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    records_json = [record.model_dump() for record in run.records]
    (run_dir / "experiments.json").write_text(json.dumps(records_json, indent=2), encoding="utf-8")
    write_records_csv(run.records, run_dir / "experiments.csv")
    metadata = {
        "method": run.method,
        "scenario": run.scenario,
        "status": run.status,
        "failure_category": run.failure_category,
        "seed": run.seed,
        "model": run.model,
        "token_usage": run.token_usage,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "configuration": config.model_dump(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_records_csv(records: list[ExperimentRecord], path: Path) -> None:
    """Write flattened experiment records to CSV."""

    fieldnames = [
        "experiment_id",
        "precursor_dose_s",
        "coreactant_dose_s",
        "inhibitor_dose_s",
        "temperature_c",
        "cycles",
        "ga_thickness_nm",
        "nga_thickness_nm",
        "selectivity",
        "process_time_s",
        "meets_objective",
        "failure_reasons",
        "decision_rationale",
        "timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            condition = record.condition
            writer.writerow(
                {
                    "experiment_id": record.experiment_id,
                    "precursor_dose_s": condition.precursor_dose_s,
                    "coreactant_dose_s": condition.coreactant_dose_s,
                    "inhibitor_dose_s": condition.inhibitor_dose_s,
                    "temperature_c": condition.temperature_c,
                    "cycles": condition.cycles,
                    "ga_thickness_nm": record.ga_thickness_nm,
                    "nga_thickness_nm": record.nga_thickness_nm,
                    "selectivity": record.selectivity,
                    "process_time_s": record.process_time_s,
                    "meets_objective": record.meets_objective,
                    "failure_reasons": "; ".join(record.failure_reasons),
                    "decision_rationale": record.decision_rationale,
                    "timestamp": record.timestamp,
                }
            )


def merge_token_usage(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


def run_conditions(
    config: ProcessConfig,
    conditions: list[ExperimentCondition],
    method: str,
    seed: int,
) -> OptimizationRun:
    """Run a fixed list of conditions, stopping on first success."""

    lab = VirtualLab(config, seed=seed)
    records: list[ExperimentRecord] = []
    started_at = datetime.now(UTC).isoformat()
    for condition in conditions:
        violations = validate_safety(condition, config)
        if violations:
            raise ValueError(f"unsafe baseline condition: {violations}")
        record = lab.simulate(condition, f"{method}_{len(records) + 1:03d}", "Baseline condition.")
        records.append(record)
        if record.meets_objective:
            break
    status: FinishStatus = (
        "success" if any(record.meets_objective for record in records) else "budget_exhausted"
    )
    if status != "success" and classify_failure(records, config) == "no_selective_window":
        status = "no_selective_window"
    return build_run(method, config, status, records, seed, None, {}, started_at)
