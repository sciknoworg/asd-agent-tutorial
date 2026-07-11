"""LLM agent and strict tool schemas for the Responses API."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from asd_agent.models import (
    ExperimentRecord,
    FinishOptimizationDecision,
    ProcessConfig,
    ProposedExperimentsDecision,
)

PROPOSE_EXPERIMENTS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "propose_experiments",
    "description": "Propose one to four safe virtual ASD experiments to test next.",
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "experiments": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "precursor_dose_s": {"type": "number"},
                        "coreactant_dose_s": {"type": "number"},
                        "inhibitor_dose_s": {"type": "number"},
                        "temperature_c": {"type": "number"},
                        "cycles": {"type": "integer"},
                    },
                    "required": [
                        "precursor_dose_s",
                        "coreactant_dose_s",
                        "inhibitor_dose_s",
                        "temperature_c",
                        "cycles",
                    ],
                },
            },
            "rationale": {
                "type": "string",
                "description": "Concise scientific rationale, at most four sentences.",
            },
        },
        "required": ["experiments", "rationale"],
    },
}

FINISH_OPTIMIZATION_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "finish_optimization",
    "description": "Finish with a tested final experiment or an explicit failure status.",
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "no_selective_window", "budget_exhausted"],
            },
            "tested_experiment_id": {
                "type": "string",
                "description": "Experiment id already present in the ledger.",
            },
            "rationale": {
                "type": "string",
                "description": "Concise scientific rationale, at most four sentences.",
            },
        },
        "required": ["status", "tested_experiment_id", "rationale"],
    },
}


def llm_tools() -> list[dict[str, Any]]:
    """Return a defensive copy of the strict function-tool schemas."""

    return [deepcopy(PROPOSE_EXPERIMENTS_TOOL), deepcopy(FINISH_OPTIMIZATION_TOOL)]


def parse_tool_decision(
    name: str, arguments: str | dict[str, Any]
) -> ProposedExperimentsDecision | FinishOptimizationDecision:
    """Validate a tool call from the LLM."""

    payload = json.loads(arguments) if isinstance(arguments, str) else arguments
    if name == "propose_experiments":
        return ProposedExperimentsDecision.model_validate(payload)
    if name == "finish_optimization":
        return FinishOptimizationDecision.model_validate(payload)
    raise ValueError(f"unexpected tool call: {name}")


def validate_finish_recommendation(
    decision: FinishOptimizationDecision,
    ledger: list[ExperimentRecord],
) -> None:
    """Require finish_optimization to reference a tested experiment id."""

    tested_ids = {record.experiment_id for record in ledger}
    if decision.tested_experiment_id not in tested_ids:
        raise ValueError(
            "finish_optimization referenced untested experiment id "
            f"{decision.tested_experiment_id!r}"
        )


class LLMOptimizationAgent:
    """Responses API agent with strict function calling."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        if not self.model:
            raise ValueError("OPENAI_MODEL must be set for the LLM agent")
        self.last_token_usage: dict[str, int] = {}

    def next_decision(
        self,
        config: ProcessConfig,
        ledger: list[ExperimentRecord],
        budget_remaining: int,
    ) -> ProposedExperimentsDecision | FinishOptimizationDecision:
        """Ask the model for the next strict tool call."""

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install the llm extra and set OPENAI_API_KEY to use the LLM agent. "
                "Use --agent rule_based for the no-API tutorial path."
            ) from exc

        client = OpenAI()
        responses_client: Any = client.responses
        response = responses_client.create(
            model=self.model,
            instructions=system_instructions(),
            input=build_agent_input(config, ledger, budget_remaining),
            tools=llm_tools(),
            tool_choice="required",
            parallel_tool_calls=False,
        )
        self.last_token_usage = extract_token_usage(response)
        decision = first_tool_decision(response)
        if isinstance(decision, FinishOptimizationDecision):
            validate_finish_recommendation(decision, ledger)
        return decision


def system_instructions() -> str:
    """Stable instruction block for the tutorial agent."""

    return (
        "You are an ASD optimization agent operating a toy virtual laboratory. "
        "This is an educational simulator and must not be treated as real chemistry. "
        "Use exactly one tool call: propose_experiments or finish_optimization. "
        "Propose only conditions inside the supplied safety bounds. "
        "Never reveal or store hidden chain-of-thought; include only a concise rationale "
        "of at most four sentences. Finish with success only when the ledger contains a "
        "tested experiment satisfying all target constraints. If the tested space strongly "
        "suggests no safe selective window, finish with no_selective_window."
    )


def build_agent_input(
    config: ProcessConfig,
    ledger: list[ExperimentRecord],
    budget_remaining: int,
) -> str:
    """Build compact model input containing config, objective, bounds, and ledger."""

    payload = {
        "scenario": config.scenario,
        "description": config.description,
        "objective": config.objective.model_dump(),
        "safety_bounds": config.safety.model_dump(),
        "budget_remaining": budget_remaining,
        "ledger": [ledger_row(record) for record in ledger],
    }
    return json.dumps(payload, indent=2)


def ledger_row(record: ExperimentRecord) -> dict[str, Any]:
    """Compact ledger row for the LLM context."""

    return {
        "experiment_id": record.experiment_id,
        "condition": record.condition.model_dump(),
        "ga_thickness_nm": round(record.ga_thickness_nm, 4),
        "nga_thickness_nm": round(record.nga_thickness_nm, 4),
        "selectivity": round(record.selectivity, 4),
        "process_time_s": round(record.process_time_s, 2),
        "meets_objective": record.meets_objective,
        "failure_reasons": record.failure_reasons,
    }


def first_tool_decision(response: Any) -> ProposedExperimentsDecision | FinishOptimizationDecision:
    """Extract and validate the first function call from a Responses API object."""

    for item in getattr(response, "output", []):
        item_type = _get(item, "type")
        if item_type == "function_call":
            return parse_tool_decision(_get(item, "name"), _get(item, "arguments"))
    raise RuntimeError("Responses API call did not return a function tool call")


def extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token accounting without depending on a specific SDK object shape."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = _get(usage, key, None)
        if isinstance(value, int):
            result[key] = value
    return result


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
