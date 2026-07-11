# Bayesian Optimization Decision Log

This log records architectural decisions for the BO extension. BO-00 decisions are
planning decisions only and do not change runtime behavior.

## D-001 - BO-00 Is Planning Only

Decision:
- BO-00 creates roadmap and audit documents only.

Rationale:
- The staged plan should protect the existing tutorial behavior before adding new BO
  methods or dependencies.

Consequences:
- No BO modules, CLI options, dependencies, scenarios, or tests are added in BO-00.

## D-002 - Add BO Behind a New Package Layer

Decision:
- Future BO code should live under `src/asd_agent/bo/`.

Rationale:
- The current simulator, objective, agent, baseline, plotting, and benchmark modules
  are already useful public surfaces. A BO subpackage keeps new abstractions scoped.

Consequences:
- Existing imports such as `from asd_agent.simulator import VirtualLab` remain valid.
- BO modules can be optional and can guard heavy imports.

## D-003 - Preserve the Existing Ledger Schema

Decision:
- `ExperimentRecord` remains the canonical experiment ledger row.

Rationale:
- Current CSV/JSON persistence, tests, benchmark summaries, and LLM context all depend
  on this schema.

Consequences:
- BO-specific tensors, normalized coordinates, posterior diagnostics, and acquisition
  values should be recorded in separate BO metadata files or adapters unless a later
  stage explicitly approves a backward-compatible schema extension.

## D-004 - Use Existing Safety and Objective Semantics

Decision:
- BO candidates must pass `validate_safety`, and BO success should align with
  `evaluate_objective` unless a stage explicitly defines an additional objective.

Rationale:
- This preserves comparability with random search, grid search, rule-based search, and
  LLM-agent runs.

Consequences:
- BO methods should output `ExperimentCondition` values, not raw arrays, at the boundary
  with the simulator and experiment loop.

## D-005 - Keep Heavy BO Dependencies Optional

Decision:
- PyTorch, GPyTorch, BoTorch, Ax, and statistics/analysis packages should be optional
  extras added only when the relevant stage needs them.

Rationale:
- The base tutorial should remain lightweight and runnable without GPU or heavy ML
  libraries.

Consequences:
- Future modules must avoid top-level imports of optional BO dependencies from
  non-BO paths.

## D-006 - Verify Python Compatibility Before Claiming It

Decision:
- Do not claim PyTorch/GPyTorch/BoTorch compatibility with the declared Python target
  until a later stage verifies it.

Rationale:
- BO-00 baseline verification used the available bundled Python 3.12.13 runtime. The
  local `py` launcher reported Python 3.11 and no Python 3.14 runtime.

Consequences:
- BO dependency selection must include a compatibility check before modifying
  `pyproject.toml`.

## D-007 - Keep LLM and BO Responsibilities Separate

Decision:
- Deterministic BO methods should be implemented and benchmarked before any hybrid
  LLM-BO agent.

Rationale:
- This keeps the scientific optimization method separable from LLM prompting behavior
  and supports no-API reproducibility.

Consequences:
- BO-08 is the first stage that should combine LLM and BO behavior.

## D-008 - Prompt Artifacts Must Be Safe and Minimal

Decision:
- Prompt files may be added under `prompts/`, but they must not contain credentials,
  hidden chain-of-thought, or raw private logs.

Rationale:
- The repository guidance requires concise stored rationales and no hidden reasoning
  records.

Consequences:
- Prompt templates should describe allowed context, tool schemas, and output contracts.

