"""Hybrid LLM-BO orchestration for the Stage 2 ASD virtual laboratory."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from asd_agent.bo.records import utc_now
from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2Decision,
    Stage2HardBounds,
    validate_stage2_decision,
)
from asd_agent.bo.stage2_mobo import (
    Stage2BOSettings,
    Stage2CandidateProposal,
    Stage2Observation,
    best_stage2_observation,
    candidate_cycle_values,
    observe_stage2_decision,
    run_stage2_bo,
)
from asd_agent.heuristic_agent import candidate_plan
from asd_agent.models import Range, sentence_count

HybridState = Literal[
    "INITIALIZE",
    "INSPECT_HISTORY",
    "REQUEST_BO_CANDIDATES",
    "REVIEW_CANDIDATES",
    "EXECUTE_CANDIDATE",
    "OBSERVE",
    "CONTINUE",
    "CHANGE_SOFT_BOUNDS",
    "FINISH",
    "DECLARE_NO_WINDOW",
]
HybridMode = Literal[
    "bo_only",
    "llm_only_legacy",
    "hybrid_advisory",
    "hybrid_intervention",
    "hybrid_explanation_only",
    "rule_based_bo",
]
HybridStatus = Literal[
    "success",
    "budget_exhausted",
    "no_selective_window",
    "malformed_tool",
]
HybridToolName = Literal[
    "inspect_experiment_history",
    "run_bayesian_optimizer",
    "run_virtual_experiment",
    "change_search_bounds",
    "finish_optimization",
    "declare_no_selective_window",
    "query_literature",
]


class HybridAgentError(RuntimeError):
    """Raised when a hybrid-agent safety rule is violated."""


class LiteratureHit(BaseModel):
    """One local or mock literature item."""

    source_id: str
    title: str
    summary: str

    model_config = ConfigDict(extra="forbid")


class LiteratureProvider(Protocol):
    """Protocol for non-mandatory literature providers."""

    def query(self, query: str, *, max_results: int = 3) -> list[LiteratureHit]: ...


class NullLiteratureProvider:
    """Literature provider that intentionally performs no retrieval."""

    def query(self, query: str, *, max_results: int = 3) -> list[LiteratureHit]:
        return []


class MockLiteratureProvider:
    """Deterministic in-memory literature provider for tests and notebooks."""

    def __init__(self, hits: Mapping[str, Sequence[LiteratureHit | Mapping[str, str]]]) -> None:
        self.hits: dict[str, list[LiteratureHit]] = {
            key.lower(): [
                hit if isinstance(hit, LiteratureHit) else LiteratureHit.model_validate(hit)
                for hit in values
            ]
            for key, values in hits.items()
        }

    def query(self, query: str, *, max_results: int = 3) -> list[LiteratureHit]:
        text = query.lower()
        matches: list[LiteratureHit] = []
        for key, values in self.hits.items():
            if key in text:
                matches.extend(values)
        return matches[:max_results]


class LocalLiteratureProvider:
    """Local JSON-backed literature provider with no live web retrieval."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.hits = self._load_hits(self.path)

    def query(self, query: str, *, max_results: int = 3) -> list[LiteratureHit]:
        terms = [term.lower() for term in query.split() if term.strip()]
        scored: list[tuple[int, LiteratureHit]] = []
        for hit in self.hits:
            haystack = f"{hit.title} {hit.summary}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, hit))
        scored.sort(key=lambda item: (-item[0], item[1].source_id))
        return [hit for _, hit in scored[:max_results]]

    @staticmethod
    def _load_hits(path: Path) -> list[LiteratureHit]:
        if not path.exists():
            return []
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("literature", [])
            if not isinstance(payload, list):
                raise ValueError("local literature JSON must contain a list")
            return [LiteratureHit.model_validate(item) for item in payload]
        hits: list[LiteratureHit] = []
        for item in sorted(path.glob("*.json")):
            hits.extend(LocalLiteratureProvider._load_hits(item))
        return hits


class Stage2SoftBounds(BaseModel):
    """Mutable search bounds constrained inside immutable hard bounds."""

    precursor_dose_s: Range
    temperature_c: Range
    cycle_values: list[int]

    model_config = ConfigDict(extra="forbid")


class SoftBoundsChange(BaseModel):
    """Candidate soft-bound changes proposed by an LLM."""

    precursor_dose_s: Range | None = None
    temperature_c: Range | None = None
    cycle_values: list[int] | None = None
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        if sentence_count(value) > 4:
            raise ValueError("decision rationale must be at most four sentences")
        return value.strip()


class InspectExperimentHistoryArgs(BaseModel):
    """Tool args for inspecting the visible history."""

    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class RunBayesianOptimizerArgs(BaseModel):
    """Tool args for requesting BO candidates."""

    max_candidates: int = Field(default=1, ge=1, le=4)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class RunVirtualExperimentArgs(BaseModel):
    """Tool args for executing exactly one immutable BO candidate."""

    candidate_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class FinishHybridOptimizationArgs(BaseModel):
    """Tool args for a successful tested final recommendation."""

    tested_experiment_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class DeclareNoSelectiveWindowArgs(BaseModel):
    """Tool args for declaring no selective window from tested evidence."""

    evidence_experiment_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class QueryLiteratureArgs(BaseModel):
    """Tool args for local/mock literature lookup."""

    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=3, ge=1, le=10)
    rationale: str = Field(min_length=1, max_length=700)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rationale")
    @classmethod
    def rationale_is_concise(cls, value: str) -> str:
        return concise(value)


class HybridToolCall(BaseModel):
    """One strict tool call selected by an LLM or deterministic policy."""

    name: HybridToolName
    arguments: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class HybridCandidateView(BaseModel):
    """LLM-visible candidate summary."""

    candidate_id: str
    decision: Stage2Decision
    acquisition_value: float | None = None
    feasibility_probability: float | None = None
    posterior_summaries: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class HybridContext(BaseModel):
    """LLM-visible context that excludes oracle and hidden simulator values."""

    mode: HybridMode
    state: HybridState
    optimizer_view: dict[str, Any]
    soft_bounds: Stage2SoftBounds
    observations: list[dict[str, object]]
    candidates: list[HybridCandidateView]
    literature: list[LiteratureHit]
    budget_remaining: int

    model_config = ConfigDict(extra="forbid")


class HybridStateEvent(BaseModel):
    """One state-machine transition."""

    from_state: HybridState
    to_state: HybridState
    tool_name: str
    status: str
    rationale: str = ""
    timestamp: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")


class HybridRunResult(BaseModel):
    """Serializable result from a hybrid LLM-BO run."""

    mode: HybridMode
    status: HybridStatus
    observations: list[Stage2Observation]
    candidates: list[Stage2CandidateProposal]
    soft_bounds: Stage2SoftBounds
    final_experiment_id: str | None = None
    evidence_experiment_ids: list[str] = Field(default_factory=list)
    events: list[HybridStateEvent] = Field(default_factory=list)
    literature: list[LiteratureHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HybridLLM(Protocol):
    """Small protocol for fake or real hybrid LLM adapters."""

    def decide(self, context: HybridContext) -> HybridToolCall: ...


class FakeHybridLLM:
    """Deterministic fake LLM for tests and notebooks."""

    def __init__(self, policy: Literal["default", "intervention", "no_window"] = "default") -> None:
        self.policy = policy
        self.queried = False
        self.changed_bounds = False

    def decide(self, context: HybridContext) -> HybridToolCall:
        if context.state == "INSPECT_HISTORY":
            return HybridToolCall(
                name="inspect_experiment_history",
                arguments={"rationale": "Review the tested evidence before choosing a tool."},
            )
        if context.state == "REQUEST_BO_CANDIDATES":
            if self.policy == "intervention" and not self.queried:
                self.queried = True
                return HybridToolCall(
                    name="query_literature",
                    arguments={
                        "query": "area selective deposition optimization",
                        "max_results": 2,
                        "rationale": "Check local notes before changing search bounds.",
                    },
                )
            if self.policy == "intervention" and not self.changed_bounds:
                self.changed_bounds = True
                return HybridToolCall(
                    name="change_search_bounds",
                    arguments={
                        "precursor_dose_s": context.soft_bounds.precursor_dose_s.model_dump(
                            mode="json"
                        ),
                        "temperature_c": context.soft_bounds.temperature_c.model_dump(mode="json"),
                        "cycle_values": context.soft_bounds.cycle_values,
                        "rationale": "Keep the current valid soft bounds for the next BO step.",
                    },
                )
            return HybridToolCall(
                name="run_bayesian_optimizer",
                arguments={
                    "max_candidates": 1,
                    "rationale": "Request a numerical candidate from BO.",
                },
            )
        if context.state == "REVIEW_CANDIDATES" and context.candidates:
            return HybridToolCall(
                name="run_virtual_experiment",
                arguments={
                    "candidate_id": context.candidates[-1].candidate_id,
                    "rationale": "Execute the immutable BO candidate.",
                },
            )
        if context.state == "CONTINUE":
            feasible = [row for row in context.observations if bool(row.get("feasible", False))]
            if feasible:
                return HybridToolCall(
                    name="finish_optimization",
                    arguments={
                        "tested_experiment_id": str(feasible[0]["experiment_id"]),
                        "rationale": "A tested feasible condition is available.",
                    },
                )
            if self.policy == "no_window" and context.observations:
                return HybridToolCall(
                    name="declare_no_selective_window",
                    arguments={
                        "evidence_experiment_ids": [
                            str(row["experiment_id"]) for row in context.observations
                        ],
                        "rationale": "The tested evidence did not reveal a feasible window.",
                    },
                )
            return HybridToolCall(
                name="run_bayesian_optimizer",
                arguments={
                    "max_candidates": 1,
                    "rationale": "Continue with another BO proposal.",
                },
            )
        return HybridToolCall(
            name="run_bayesian_optimizer",
            arguments={"max_candidates": 1, "rationale": "Default to BO continuation."},
        )


class HybridLLMBOAgent:
    """State-machine orchestrator that keeps LLM reasoning separate from BO numerics."""

    def __init__(
        self,
        config: Stage2Config,
        *,
        mode: HybridMode = "hybrid_advisory",
        llm: HybridLLM | None = None,
        literature_provider: LiteratureProvider | None = None,
        bo_settings: Stage2BOSettings | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.llm = llm or FakeHybridLLM()
        self.literature_provider = literature_provider or NullLiteratureProvider()
        self.bo_settings = bo_settings or Stage2BOSettings(
            experiment_budget=6,
            initial_design_size=2,
            qmc_samples=8,
            num_restarts=1,
            raw_samples=8,
            acquisition_timeout_s=2.0,
            random_fallback_points=32,
        )
        self.seed = config.process.seed if seed is None else seed
        self.state: HybridState = "INITIALIZE"
        self.observations: list[Stage2Observation] = []
        self.candidates: dict[str, Stage2CandidateProposal] = {}
        self.executed_candidate_ids: set[str] = set()
        self.soft_bounds = default_soft_bounds(config, self.bo_settings)
        self.events: list[HybridStateEvent] = []
        self.literature: list[LiteratureHit] = []
        self.warnings: list[str] = []
        self.final_experiment_id: str | None = None
        self.evidence_experiment_ids: list[str] = []

    def run(self, *, budget: int = 6, max_steps: int = 40) -> HybridRunResult:
        """Run the hybrid state machine without live API calls by default."""

        if self.mode == "bo_only":
            return self._run_bo_only(budget=budget)
        if self.mode == "llm_only_legacy":
            return self._run_llm_only_legacy(budget=budget)

        self._transition("INITIALIZE", "INSPECT_HISTORY", "initialize", "ok")
        for _ in range(max_steps):
            if len(self.observations) >= budget:
                return self._budget_result()
            context = self.context(budget_remaining=budget - len(self.observations))
            try:
                tool_call = self.llm.decide(context)
                self.apply_tool_call(tool_call, budget=budget)
            except Exception as exc:
                self.warnings.append(f"malformed tool ignored: {type(exc).__name__}: {exc}")
                self._transition(self.state, self.state, "malformed_tool", "ignored")
            if self.state == "FINISH":
                return HybridRunResult(
                    mode=self.mode,
                    status="success",
                    observations=self.observations,
                    candidates=list(self.candidates.values()),
                    soft_bounds=self.soft_bounds,
                    final_experiment_id=self.final_experiment_id,
                    events=self.events,
                    literature=self.literature,
                    warnings=self.warnings,
                )
            if self.state == "DECLARE_NO_WINDOW":
                return HybridRunResult(
                    mode=self.mode,
                    status="no_selective_window",
                    observations=self.observations,
                    candidates=list(self.candidates.values()),
                    soft_bounds=self.soft_bounds,
                    evidence_experiment_ids=list(self.evidence_experiment_ids),
                    events=self.events,
                    literature=self.literature,
                    warnings=self.warnings,
                )
        self.warnings.append("maximum hybrid state-machine steps reached")
        return self._budget_result()

    def apply_tool_call(self, tool_call: HybridToolCall, *, budget: int) -> object:
        """Validate and apply one strict hybrid tool call."""

        before_state = self.state
        before_observations = len(self.observations)
        before_candidates = set(self.candidates)
        try:
            result = self._apply_validated_tool_call(tool_call, budget=budget)
        except Exception:
            if (
                len(self.observations) != before_observations
                or set(self.candidates) != before_candidates
            ):
                raise AssertionError("malformed tool mutated hybrid state") from None
            self.state = before_state
            raise
        return result

    def _apply_validated_tool_call(self, tool_call: HybridToolCall, *, budget: int) -> object:
        if tool_call.name == "inspect_experiment_history":
            inspect_args = InspectExperimentHistoryArgs.model_validate(tool_call.arguments)
            self._transition(
                self.state,
                "REQUEST_BO_CANDIDATES",
                tool_call.name,
                "ok",
                inspect_args.rationale,
            )
            return self.context(budget_remaining=budget - len(self.observations))
        if tool_call.name == "query_literature":
            query_args = QueryLiteratureArgs.model_validate(tool_call.arguments)
            hits = self.literature_provider.query(
                query_args.query, max_results=query_args.max_results
            )
            self.literature.extend(hits)
            self._transition(
                self.state,
                "REQUEST_BO_CANDIDATES",
                tool_call.name,
                "ok",
                query_args.rationale,
            )
            return hits
        if tool_call.name == "change_search_bounds":
            bounds_args = SoftBoundsChange.model_validate(tool_call.arguments)
            if self.mode != "hybrid_intervention":
                self.warnings.append(
                    "change_search_bounds ignored outside hybrid_intervention mode"
                )
                self._transition(
                    self.state,
                    "REQUEST_BO_CANDIDATES",
                    tool_call.name,
                    "ignored",
                    bounds_args.rationale,
                )
                return self.soft_bounds
            self.soft_bounds = validate_soft_bounds_change(
                self.config, self.soft_bounds, bounds_args
            )
            self._transition(
                self.state,
                "REQUEST_BO_CANDIDATES",
                tool_call.name,
                "ok",
                bounds_args.rationale,
            )
            return self.soft_bounds
        if tool_call.name == "run_bayesian_optimizer":
            bo_args = RunBayesianOptimizerArgs.model_validate(tool_call.arguments)
            proposals = self._run_bo_candidates(bo_args.max_candidates)
            self._transition(
                self.state,
                "REVIEW_CANDIDATES",
                tool_call.name,
                "ok",
                bo_args.rationale,
            )
            return proposals
        if tool_call.name == "run_virtual_experiment":
            experiment_args = RunVirtualExperimentArgs.model_validate(tool_call.arguments)
            observation = self._execute_candidate(experiment_args.candidate_id)
            self._transition(
                self.state,
                "OBSERVE",
                tool_call.name,
                "ok",
                experiment_args.rationale,
            )
            self._transition("OBSERVE", "CONTINUE", "observe", "ok")
            return observation
        if tool_call.name == "finish_optimization":
            finish_args = FinishHybridOptimizationArgs.model_validate(tool_call.arguments)
            self._validate_finish(finish_args)
            self.final_experiment_id = finish_args.tested_experiment_id
            self._transition(
                self.state,
                "FINISH",
                tool_call.name,
                "ok",
                finish_args.rationale,
            )
            return finish_args
        if tool_call.name == "declare_no_selective_window":
            no_window_args = DeclareNoSelectiveWindowArgs.model_validate(tool_call.arguments)
            self._validate_no_window_evidence(no_window_args)
            self.evidence_experiment_ids = list(no_window_args.evidence_experiment_ids)
            self._transition(
                self.state,
                "DECLARE_NO_WINDOW",
                tool_call.name,
                "ok",
                no_window_args.rationale,
            )
            return no_window_args
        raise HybridAgentError(f"unsupported hybrid tool: {tool_call.name}")

    def context(self, *, budget_remaining: int) -> HybridContext:
        """Build LLM-visible context without oracle or hidden simulator parameters."""

        return HybridContext(
            mode=self.mode,
            state=self.state,
            optimizer_view=self.config.optimizer_view(),
            soft_bounds=self.soft_bounds,
            observations=[observation.optimizer_payload() for observation in self.observations],
            candidates=[
                HybridCandidateView(
                    candidate_id=candidate.candidate_id,
                    decision=candidate.decision,
                    acquisition_value=candidate.acquisition_value,
                    feasibility_probability=candidate.feasibility_probability,
                    posterior_summaries=dict(candidate.posterior_summaries),
                )
                for candidate in self.candidates.values()
            ],
            literature=list(self.literature),
            budget_remaining=budget_remaining,
        )

    def _run_bo_candidates(self, max_candidates: int) -> list[Stage2CandidateProposal]:
        proposals: list[Stage2CandidateProposal] = []
        soft_config = config_with_soft_bounds(self.config, self.soft_bounds)
        settings = self.bo_settings.model_copy(
            update={
                "experiment_budget": max(
                    self.bo_settings.experiment_budget, len(self.observations) + 1
                ),
                "candidate_cycle_values": list(self.soft_bounds.cycle_values),
            }
        )
        for index in range(max_candidates):
            result = run_stage2_bo(
                soft_config,
                settings,
                simulator_seed=self.seed,
                optimizer_seed=self.seed + index,
                initial_observations=self.observations,
            )
            if result.proposals:
                proposal = result.proposals[-1]
            else:
                best = best_stage2_observation(result.observations)
                if best is None:
                    continue
                proposal = Stage2CandidateProposal.create(
                    decision=best.decision,
                    optimizer="stage2_mobo_initial",
                    seed=self.seed + index,
                )
            if validate_stage2_decision(self.config, proposal.decision):
                continue
            self.candidates.setdefault(proposal.candidate_id, proposal)
            proposals.append(proposal)
        return proposals

    def _execute_candidate(self, candidate_id: str) -> Stage2Observation:
        if candidate_id not in self.candidates:
            raise HybridAgentError(f"unknown candidate_id {candidate_id!r}")
        if candidate_id in self.executed_candidate_ids:
            raise HybridAgentError(f"candidate_id {candidate_id!r} was already executed")
        proposal = self.candidates[candidate_id]
        violations = validate_stage2_decision(self.config, proposal.decision)
        if violations:
            raise HybridAgentError(f"candidate violates immutable hard bounds: {violations}")
        observation = observe_stage2_decision(
            self.config,
            proposal.decision,
            experiment_id=f"hybrid_{len(self.observations) + 1:03d}",
            seed=self.seed + len(self.observations),
        )
        self.executed_candidate_ids.add(candidate_id)
        self.observations.append(observation)
        return observation

    def _validate_finish(self, args: FinishHybridOptimizationArgs) -> None:
        matching = [
            obs for obs in self.observations if obs.experiment_id == args.tested_experiment_id
        ]
        if not matching:
            raise HybridAgentError(
                f"finish_optimization referenced untested id {args.tested_experiment_id!r}"
            )
        if not matching[0].constraint_evaluation.feasible:
            raise HybridAgentError("finish_optimization requires a feasible tested experiment")

    def _validate_no_window_evidence(self, args: DeclareNoSelectiveWindowArgs) -> None:
        tested = {obs.experiment_id for obs in self.observations}
        missing = [item for item in args.evidence_experiment_ids if item not in tested]
        if missing:
            raise HybridAgentError(f"evidence ids not present in ledger: {missing}")

    def _transition(
        self,
        from_state: HybridState,
        to_state: HybridState,
        tool_name: str,
        status: str,
        rationale: str = "",
    ) -> None:
        self.events.append(
            HybridStateEvent(
                from_state=from_state,
                to_state=to_state,
                tool_name=tool_name,
                status=status,
                rationale=rationale,
            )
        )
        self.state = to_state

    def _budget_result(self) -> HybridRunResult:
        best = best_stage2_observation(self.observations)
        success = best is not None and best.constraint_evaluation.feasible
        final_experiment_id = best.experiment_id if best is not None and success else None
        return HybridRunResult(
            mode=self.mode,
            status="success" if success else "budget_exhausted",
            observations=self.observations,
            candidates=list(self.candidates.values()),
            soft_bounds=self.soft_bounds,
            final_experiment_id=final_experiment_id,
            events=self.events,
            literature=self.literature,
            warnings=self.warnings,
        )

    def _run_bo_only(self, *, budget: int) -> HybridRunResult:
        result = run_stage2_bo(
            self.config,
            self.bo_settings.model_copy(update={"experiment_budget": budget}),
            simulator_seed=self.seed,
            optimizer_seed=self.seed,
        )
        status: HybridStatus = "success" if result.status == "success" else "budget_exhausted"
        return HybridRunResult(
            mode=self.mode,
            status=status,
            observations=result.observations,
            candidates=result.proposals,
            soft_bounds=self.soft_bounds,
            final_experiment_id=result.recommended_experiment_id if status == "success" else None,
            warnings=result.warnings,
        )

    def _run_llm_only_legacy(self, *, budget: int) -> HybridRunResult:
        observations: list[Stage2Observation] = []
        for condition in candidate_plan(self.config.process):
            if len(observations) >= budget:
                break
            decision = Stage2Decision(
                precursor_dose_s=condition.precursor_dose_s,
                temperature_c=condition.temperature_c,
                cycle_count=condition.cycles,
            )
            if validate_stage2_decision(self.config, decision):
                continue
            observations.append(
                observe_stage2_decision(
                    self.config,
                    decision,
                    experiment_id=f"legacy_llm_{len(observations) + 1:03d}",
                    seed=self.seed + len(observations),
                )
            )
        final = best_stage2_observation(observations)
        success = final is not None and final.constraint_evaluation.feasible
        return HybridRunResult(
            mode=self.mode,
            status="success" if success else "budget_exhausted",
            observations=observations,
            candidates=[],
            soft_bounds=self.soft_bounds,
            final_experiment_id=final.experiment_id if success and final else None,
            warnings=["llm_only_legacy preserves the separate legacy LLM comparator path."],
        )


def hybrid_tool_schemas() -> list[dict[str, Any]]:
    """Return strict function-tool schemas for the hybrid LLM-BO orchestrator."""

    return [
        function_schema("inspect_experiment_history", InspectExperimentHistoryArgs),
        function_schema("run_bayesian_optimizer", RunBayesianOptimizerArgs),
        function_schema("run_virtual_experiment", RunVirtualExperimentArgs),
        function_schema("change_search_bounds", SoftBoundsChange),
        function_schema("finish_optimization", FinishHybridOptimizationArgs),
        function_schema("declare_no_selective_window", DeclareNoSelectiveWindowArgs),
        function_schema("query_literature", QueryLiteratureArgs),
    ]


def function_schema(name: str, model: type[BaseModel]) -> dict[str, Any]:
    """Return a Responses-style strict function schema from a Pydantic model."""

    return {
        "type": "function",
        "name": name,
        "strict": True,
        "parameters": model.model_json_schema(),
    }


def default_soft_bounds(config: Stage2Config, settings: Stage2BOSettings) -> Stage2SoftBounds:
    """Return full-range soft bounds initialized from immutable hard bounds."""

    return Stage2SoftBounds(
        precursor_dose_s=config.hard_bounds.precursor_dose_s,
        temperature_c=config.hard_bounds.temperature_c,
        cycle_values=candidate_cycle_values(config, settings),
    )


def validate_soft_bounds_change(
    config: Stage2Config,
    current: Stage2SoftBounds,
    change: SoftBoundsChange,
) -> Stage2SoftBounds:
    """Validate soft-bound changes against immutable hard bounds."""

    candidate = Stage2SoftBounds(
        precursor_dose_s=change.precursor_dose_s or current.precursor_dose_s,
        temperature_c=change.temperature_c or current.temperature_c,
        cycle_values=change.cycle_values or current.cycle_values,
    )
    hard = config.hard_bounds
    if not range_inside(candidate.precursor_dose_s, hard.precursor_dose_s):
        raise HybridAgentError("precursor soft bounds must stay inside hard bounds")
    if not range_inside(candidate.temperature_c, hard.temperature_c):
        raise HybridAgentError("temperature soft bounds must stay inside hard bounds")
    if not candidate.cycle_values:
        raise HybridAgentError("soft cycle values must not be empty")
    invalid_cycles = [
        value for value in candidate.cycle_values if not hard.cycle_count.contains(float(value))
    ]
    if invalid_cycles:
        raise HybridAgentError(f"soft cycle values outside hard bounds: {invalid_cycles}")
    return candidate


def config_with_soft_bounds(config: Stage2Config, soft_bounds: Stage2SoftBounds) -> Stage2Config:
    """Return a copy whose optimizer-visible bounds are narrowed to soft bounds."""

    updated_hard = Stage2HardBounds(
        precursor_dose_s=soft_bounds.precursor_dose_s,
        temperature_c=soft_bounds.temperature_c,
        cycle_count=Range(
            min=float(min(soft_bounds.cycle_values)),
            max=float(max(soft_bounds.cycle_values)),
        ),
        max_process_time_s=config.hard_bounds.max_process_time_s,
    )
    return config.model_copy(update={"hard_bounds": updated_hard})


def range_inside(inner: Range, outer: Range) -> bool:
    """Return whether one range is fully inside another."""

    return outer.contains(inner.min) and outer.contains(inner.max)


def parse_hybrid_tool_call(name: str, arguments: str | dict[str, Any]) -> HybridToolCall:
    """Parse a strict hybrid tool call without executing it."""

    payload = json.loads(arguments) if isinstance(arguments, str) else arguments
    return HybridToolCall(name=cast(HybridToolName, name), arguments=payload)


def concise(value: str) -> str:
    """Validate concise stored rationales."""

    if sentence_count(value) > 4:
        raise ValueError("decision rationale must be at most four sentences")
    return value.strip()


def run_hybrid_optimization(
    config: Stage2Config,
    *,
    mode: HybridMode = "hybrid_advisory",
    llm: HybridLLM | None = None,
    literature_provider: LiteratureProvider | None = None,
    bo_settings: Stage2BOSettings | None = None,
    seed: int | None = None,
    budget: int = 6,
) -> HybridRunResult:
    """Convenience function for notebooks and smoke tests."""

    return HybridLLMBOAgent(
        config,
        mode=mode,
        llm=llm,
        literature_provider=literature_provider,
        bo_settings=bo_settings,
        seed=seed,
    ).run(budget=budget)
