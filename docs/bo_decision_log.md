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

## D-009 - BO-01 Uses Optional Extras Only

Decision:
- BO dependencies are declared as optional extras: `bo`, `bo-gp`, `bo-ax`, and
  `bo-analysis`.

Rationale:
- The base tutorial should remain runnable without SciPy, PyTorch, GPyTorch, BoTorch,
  or Ax installed.

Consequences:
- BO-01 infrastructure does not import optional BO packages.
- Resolver compatibility was checked on the available Python 3.12.13 runtime; Python
  3.14 runtime compatibility remains unverified locally.

## D-010 - BO Records Wrap Existing Ledger Rows

Decision:
- `BOExperimentRecord` wraps `ExperimentRecord` rather than changing the canonical
  ledger schema.

Rationale:
- Existing CSV/JSON ledgers, demos, notebooks, benchmark summaries, and LLM context all
  depend on `ExperimentRecord` remaining stable.

Consequences:
- BO metadata such as candidate IDs, acquisition values, posterior summaries, and
  training observation IDs live in BO-specific artifacts.
- Legacy `ExperimentRecord` rows can be wrapped with no proposal attached.

## D-011 - Optimizer Visibility Is Explicit

Decision:
- `VirtualASDBackend.optimizer_view()` exposes scenario, description, objective, and
  safety bounds, while surface parameters, noise, and process-time constants remain in
  a separate simulator-only view.

Rationale:
- Future optimizer benchmarks need a clean boundary between information available to
  optimizers and hidden virtual-lab parameters.

Consequences:
- Tests assert that optimizer-facing records and backend context do not contain hidden
  simulator fields such as surface growth rates or nucleation delays.

## D-012 - Run Reproducibility Metadata Is a First-Class Artifact

Decision:
- `RunManifest`, `OptimizerState`, and `BORunRecord` are Pydantic JSON artifacts.

Rationale:
- BO benchmarks need reproducible manifests before GP state, acquisition diagnostics,
  and hybrid-agent behavior are added.

Consequences:
- Later BO stages can save optimizer checkpoints and run manifests without changing
  legacy demo or benchmark outputs.

## D-013 - Avoid Python 3.14-Only Syntax in Runtime Code

Decision:
- BO-01 avoids syntax that the available Python 3.12.13 runtime cannot parse, even when
  ruff targeting Python 3.14 can format it.

Rationale:
- Baseline verification and BO-01 tests currently run on Python 3.12.13 in this
  workspace.

Consequences:
- The code remains executable in the verified local runtime while the project still
  documents that Python 3.14 runtime compatibility has not been locally verified.

## D-014 - Stage 1 Uses a Separate One-Dimensional Dose Type

Decision:
- BO-02 introduces `Stage1Dose` and `Stage1ExperimentRecord` instead of forcing Stage
  1 processes into the ASD `ExperimentCondition` schema.

Rationale:
- Stage 1 is a one-dimensional precursor-dose teaching problem. Reusing the full ASD
  condition would add irrelevant temperature, inhibitor, coreactant, and cycle fields.

Consequences:
- Stage 1 code lives under `asd_agent.bo` and does not change the existing ASD ledger.
- Later adapters can convert Stage 1 observations into GP training arrays without
  touching `ExperimentRecord`.

## D-015 - Stage 1 Oracle Is Evaluation-Only

Decision:
- The Stage 1 oracle reports true saturation values, t95 values, dense true curves, and
  non-self-limited classifications only through evaluation-specific classes.

Rationale:
- Active-learning benchmarks need hidden truth for scoring while optimizers must work
  from safe public context and noisy observations.

Consequences:
- `Stage1Config.optimizer_view()` and `Stage1ExperimentRecord` exclude process family,
  rate constants, true t95, true asymptote, and scenario labels.
- Tests explicitly check for oracle leakage.

## D-016 - Known Target and Inferred Asymptote Remain Explicit Modes

Decision:
- `Stage1Objective` validates `known_target` and `inferred_asymptote` as distinct
  modes. Known-target mode requires `target_growth`; inferred-asymptote mode forbids
  it.

Rationale:
- The two learning tasks have different information assumptions and should not be
  silently mixed.

Consequences:
- Optimizer views expose `target_growth` only for known-target scenarios.
- Inferred-asymptote scenarios require the optimizer to infer the plateau from
  observations; the oracle still uses the true asymptote for evaluation metrics.

## D-017 - Non-Self-Limited Growth Has No Meaningful Saturation Threshold

Decision:
- The weakly non-self-limited Stage 1 family is classified as lacking a meaningful
  saturation threshold, even though it can cross finite response levels numerically.

Rationale:
- A continuing linear tail is a failure mode for saturation declaration, not a hidden
  optimum to optimize around.

Consequences:
- The oracle returns no true saturation value or true t95 for this family.
- Recommendation metrics can flag false saturation declarations.

## D-018 - BO-03 Uses Threshold-Oriented Active Learning

Decision:
- Stage 1 generic-GP acquisition evaluates posterior target probability over a finite
  candidate grid, chooses the smallest untested dose above a probability threshold, and
  otherwise samples the point with highest uncertainty-weighted proximity to the
  threshold.

Rationale:
- The Stage 1 tutorial question is "what is the smallest tested dose that reaches
  saturation?", not "where is growth largest?" A maximum-growth acquisition would
  bias comparisons toward unnecessarily high doses.

Consequences:
- The acquisition rule is implemented in `asd_agent.bo.acquisition` and is testable
  without fitting a GP.
- Candidate recommendations include posterior summaries and acquisition values when
  available.

## D-019 - Stage 1 Recommendations Must Reference Tested Rows

Decision:
- BO-03 results store `recommended_experiment_id`, and successful recommendations are
  generated only from the Stage 1 experiment ledger.

Rationale:
- This mirrors the existing LLM-agent safety rule that a final recommendation cannot
  point to an untested condition.

Consequences:
- The generic GP may propose untested candidates during the loop, but its final
  recommendation is the smallest tested row meeting the current threshold estimate.
- Tests assert that the recommended experiment ID is present in the ledger.

## D-020 - Generic GP Stays in Optional BO Modules

Decision:
- BoTorch, GPyTorch, and PyTorch imports live in `asd_agent.bo.gp` and are not imported
  by `asd_agent.bo.__init__`.

Rationale:
- Base simulator, LLM, notebook, and legacy benchmark users should not need the heavy
  GP stack unless they use BO-03 functionality.

Consequences:
- BO-03 tests use optional dependency gates where appropriate.
- Mypy ignores missing GPyTorch stubs because the installed package lacks a `py.typed`
  marker.

## D-021 - Stage 1 Grid and GP Share Initial Observations

Decision:
- BO-03 runners evaluate matched initial Stage 1 doses before method-specific
  candidate selection.

Rationale:
- Grid and adaptive comparisons should differ in selection strategy, not in starting
  information, simulator seed, or budget accounting.

Consequences:
- `Stage1RunnerSettings` records the shared budget, initial dose fractions or explicit
  doses, simulator seed, optimizer seed, and minimum observations before
  recommendation.
