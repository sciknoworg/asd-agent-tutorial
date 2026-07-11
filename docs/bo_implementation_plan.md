# Bayesian Optimization Implementation Plan

Task: BO-00 - Audit repository and define the Bayesian-optimization roadmap

This document defines a staged integration plan for Bayesian optimization (BO) in the
ASD agent tutorial. BO-00 is planning only. It does not implement BO, add runtime
dependencies, or change existing behavior.

## Integration Principles

- Preserve the existing public interfaces unless a later task explicitly approves a
  backward-compatible extension.
- Keep `ExperimentCondition`, `ExperimentRecord`, `ProcessConfig`, `SafetyBounds`,
  `VirtualLab`, `run_agent_loop`, `run_conditions`, `random_search`, `grid_search`,
  `RuleBasedAgent`, and `LLMOptimizationAgent` usable as they are today.
- Add BO code behind new modules, most likely under `src/asd_agent/bo/`, rather than
  mixing model-fitting code into simulator, objective, or ledger modules.
- Treat the existing simulator as the oracle for tutorial experiments. BO should call
  the same safety checks and virtual-lab entry points that current agents and
  baselines use.
- Keep the existing experiment ledger stable. If BO needs tensors, normalized values,
  posterior state, or acquisition diagnostics, store those in BO-specific artifacts or
  optional metadata files rather than changing `ExperimentRecord` in place.
- Keep heavy BO dependencies optional until compatibility with the project's Python
  target and operating systems has been verified.
- Do not run live OpenAI calls in BO stages unless the specific task is the hybrid
  LLM-BO integration stage and credentials are explicitly provided.
- Do not store hidden chain-of-thought in prompt or ledger artifacts.

## Proposed BO Architecture

Future BO work should be layered as follows:

1. Existing simulator and objectives remain the source of truth.
2. A BO domain layer converts `SafetyBounds` into search variables and normalizes
   `ExperimentCondition` objects for surrogate models.
3. A BO observation layer converts `ExperimentRecord` ledgers into training arrays
   without changing the ledger schema.
4. Surrogate model implementations live behind small interfaces so tutorial stages can
   compare generic GP, physics-informed GP, and later constrained multi-objective BO.
5. Acquisition and candidate-generation code proposes safe `ExperimentCondition`
   objects and reuses `validate_safety`.
6. BO optimizers expose either a simple baseline function, like `random_search`, or an
   `OptimizationAgent`-compatible `next_decision` method when they participate in
   `run_agent_loop`.
7. Benchmark and plotting additions consume existing `OptimizationRun.summary_row`
   output plus BO-specific analysis files.

## Dependency Strategy

No dependency changes are part of BO-00.

### PyTorch

- Add PyTorch only as an optional dependency extra in a later task, likely `bo`.
- Verify wheel support for the project's declared Python target before adding a
  requirement. BO-00 did not verify Python 3.14 runtime compatibility.
- Keep CPU-only installation instructions separate from CUDA-specific instructions.
- Avoid importing `torch` from package top-level modules so non-BO users can keep using
  the simulator without PyTorch installed.

### GPyTorch

- Add GPyTorch only after PyTorch compatibility is confirmed.
- Use it for Gaussian-process model definitions and likelihoods, not for domain
  normalization or ledger conversion.
- Keep the first GPyTorch wrapper minimal and covered by deterministic tests on small
  synthetic data.

### BoTorch

- Prefer BoTorch for acquisition functions, candidate generation, and constrained BO
  once the basic GP path is stable.
- Keep BoTorch-specific objects behind adapter functions so benchmark and agent code
  can remain library-agnostic.
- Record random seeds and candidate-generation settings in run metadata.

### Ax

- Treat Ax as optional and probably tutorial-facing rather than core infrastructure.
- Consider Ax only after BoTorch-based flows are working, because Ax may obscure the
  mechanics that the tutorial is meant to teach.
- If included, place it behind a separate optional extra such as `bo-ax`.

### Scientific Analysis Dependencies

- Potential later additions: `scipy`, `scikit-learn`, `statsmodels`, `seaborn`, and
  `tqdm`.
- Add analysis dependencies only when a specific stage needs them.
- Keep statistical reporting reproducible by recording seeds, repetitions, scenario
  names, method names, and versions in benchmark outputs.

## Work Packages

### BO-01: Shared BO Infrastructure

Purpose:
- Create the reusable BO substrate: search-space definitions, variable normalization,
  ledger-to-observation conversion, candidate validation, and interfaces for surrogate
  models and acquisition functions.

Dependencies:
- Existing `ProcessConfig`, `SafetyBounds`, `ExperimentCondition`, `ExperimentRecord`,
  `validate_safety`, and `run_conditions`.
- No PyTorch/GPyTorch/BoTorch dependency unless explicitly approved in BO-01.

Expected files:
- `src/asd_agent/bo/__init__.py`
- `src/asd_agent/bo/domain.py`
- `src/asd_agent/bo/observations.py`
- `src/asd_agent/bo/transforms.py`
- `src/asd_agent/bo/candidates.py`
- `tests/test_bo_domain.py`
- `tests/test_bo_observations.py`
- Documentation updates for BO concepts.

Acceptance criteria:
- Existing baseline tests remain unchanged and passing.
- Safety bounds can be converted to and from normalized search-space coordinates.
- A list of `ExperimentRecord` objects can be converted into BO training observations
  without changing the ledger format.
- Candidate conversion never produces out-of-bounds `ExperimentCondition` values.

Tests:
- Round-trip normalized and raw parameters.
- Ledger conversion for empty, one-record, and multi-record cases.
- Safety rejection for invalid normalized or raw candidates.
- Determinism of candidate utilities when seeds are supplied.

Explicit non-goals:
- No Gaussian-process fitting.
- No acquisition optimization.
- No benchmark changes beyond tests for shared utilities.
- No dependency additions unless BO-01 explicitly revises this plan.

### BO-02: Stage 1 Process Models and Oracle

Purpose:
- Define simple Stage 1 process models and an oracle interface for BO tutorials before
  using the full ASD constraints.

Dependencies:
- BO-01 shared infrastructure.
- Existing simulator equations may be reused, but Stage 1 should remain simpler than
  the full ASD scenario set.

Expected files:
- `src/asd_agent/bo/stage1.py`
- `src/asd_agent/bo/oracle.py`
- `configs/bo_stage1_*.yaml` if configuration is needed.
- `tests/test_bo_stage1.py`
- Notebook or docs additions only if approved for this stage.

Acceptance criteria:
- Stage 1 oracle returns deterministic outputs for fixed seeds and noise settings.
- Stage 1 inputs can be represented as `ExperimentCondition` or a clearly documented
  BO-specific input type with adapters.
- Stage 1 outputs identify objective values and optional noise separately.

Tests:
- Deterministic oracle behavior.
- Known optimum or reference behavior for synthetic process models.
- Bounds and invalid-input handling.

Explicit non-goals:
- No constrained multi-objective optimization.
- No LLM integration.
- No changes to the three existing ASD scenarios.

### BO-03: Stage 1 Generic GP and Grid Comparison

Purpose:
- Implement and benchmark a generic Gaussian-process BO loop on Stage 1 models, then
  compare it against grid search.

Dependencies:
- BO-01 and BO-02.
- PyTorch and GPyTorch only after compatibility is verified and dependencies are added
  in this stage or a prerequisite stage.

Expected files:
- `src/asd_agent/bo/gp.py`
- `src/asd_agent/bo/acquisition.py`
- `src/asd_agent/bo/optimizers.py`
- `tests/test_bo_gp.py`
- `tests/test_bo_acquisition.py`
- Benchmark additions under `src/asd_agent/benchmark.py` or a BO-specific benchmark
  module.

Acceptance criteria:
- Generic GP BO runs reproducibly on Stage 1 oracle problems.
- Grid-search comparison uses the same budget accounting.
- Benchmark output records success rate, experiment count, best objective, and seed.

Tests:
- GP model can fit a tiny deterministic dataset.
- Acquisition scores finite candidate sets without NaNs.
- BO loop does not repeat a tested candidate unless the candidate set is exhausted.

Explicit non-goals:
- No physics-informed kernels yet.
- No Stage 2 ASD constraints.
- No LLM-driven BO decisions.

### BO-04: Stage 1 Physics-Informed GP and Benchmark

Purpose:
- Add physics-informed surrogate options for Stage 1 and compare them against the
  generic GP and grid search.

Dependencies:
- BO-01 through BO-03.
- Stage 1 process-model assumptions documented in BO-02.

Expected files:
- `src/asd_agent/bo/physics_kernels.py`
- `src/asd_agent/bo/physics_models.py`
- `tests/test_bo_physics_gp.py`
- Benchmark outputs and plots for Stage 1 comparisons.

Acceptance criteria:
- Physics-informed model exposes the same prediction/acquisition interface as the
  generic GP path.
- Benchmarks compare generic GP, physics-informed GP, grid search, and random search
  with shared seeds and budgets.
- Documentation explains what physical assumption is encoded and why it is still a
  tutorial model.

Tests:
- Kernel/model parameter constraints.
- Deterministic predictions for tiny fixtures.
- Benchmark smoke test on a small repetition count.

Explicit non-goals:
- No claim of real HfO2/MoS2 predictive value.
- No Stage 2 constrained multi-objective BO.
- No production lab recommendations.

### BO-05: Stage 2 Problem Definition and Scenarios

Purpose:
- Define the Stage 2 ASD BO problem using GA thickness, NGA thickness, selectivity,
  process time, safety bounds, and scenario-specific objectives.

Dependencies:
- BO-01 shared infrastructure.
- Existing ASD scenarios and objective functions.
- Lessons from BO-02 through BO-04.

Expected files:
- `src/asd_agent/bo/stage2.py`
- `src/asd_agent/bo/objectives.py`
- `configs/bo_stage2_*.yaml` if separate BO configs are needed.
- `tests/test_bo_stage2_problem.py`

Acceptance criteria:
- Stage 2 problem definitions can be built from existing `ProcessConfig` objects.
- Objective and constraint outputs are explicit: GA target, NGA limit, selectivity
  target, safety feasibility, and optional process-time penalty.
- Existing scenarios remain loadable and unchanged.

Tests:
- Stage 2 construction for all three existing scenarios.
- Constraint values match existing `evaluate_objective` behavior.
- Impossible scenario remains impossible under tested bounds.

Explicit non-goals:
- No multi-objective optimizer implementation.
- No LLM integration.
- No changes to default demo or benchmark CLI behavior.

### BO-06: Stage 2 Constrained Multi-Objective BO

Purpose:
- Implement constrained BO for ASD Stage 2, balancing GA growth, NGA suppression,
  selectivity, safety, and process time.

Dependencies:
- BO-01 and BO-05.
- BoTorch after dependency compatibility is verified.

Expected files:
- `src/asd_agent/bo/constrained.py`
- `src/asd_agent/bo/multi_objective.py`
- `src/asd_agent/bo/asd_bo_agent.py`
- `tests/test_bo_constrained.py`
- `tests/test_bo_asd_agent.py`

Acceptance criteria:
- BO proposes only safe `ExperimentCondition` values.
- BO can run through existing `run_agent_loop` or a documented BO runner.
- Final recommendations still refer only to tested experiment IDs.
- Benchmark rows remain compatible with current summary fields.

Tests:
- Constraint feasibility calculations.
- Candidate safety and no-duplicate behavior.
- End-to-end smoke tests on solvable and impossible scenarios.

Explicit non-goals:
- No hidden LLM reasoning or live OpenAI calls.
- No real-lab recommendation claims.
- No changes to current random/grid/rule-based results except adding new methods.

### BO-07: Stage 2 Benchmark and Analysis

Purpose:
- Extend benchmarking and plotting to compare BO methods against existing baselines on
  the Stage 2 ASD scenarios.

Dependencies:
- BO-06.
- Existing benchmark and plotting modules.

Expected files:
- `src/asd_agent/bo/benchmark.py` or additions to `src/asd_agent/benchmark.py`
- `src/asd_agent/bo/analysis.py`
- `tests/test_bo_benchmark.py`
- Plotting additions for BO trajectories, constraint feasibility, and objective tradeoffs.

Acceptance criteria:
- Benchmarks run at least 20 independent repetitions by default or by documented CLI.
- Output includes success rate, experiments required, best selectivity, GA/NGA
  thickness, process time, and failure category.
- BO-specific plots are generated without requiring live API access.

Tests:
- Benchmark smoke run with small repetitions.
- Summary-row schema compatibility.
- Plot function smoke tests using a noninteractive backend.

Explicit non-goals:
- No LLM-BO hybrid behavior.
- No statistical publication claims beyond descriptive benchmark summaries.
- No dependency changes unrelated to analysis needs.

### BO-08: Hybrid LLM-BO Agent

Purpose:
- Combine LLM planning with BO candidate generation while preserving strict tool
  calling, tested-final-condition validation, and concise rationales.

Dependencies:
- BO-06 or BO-07.
- Existing `LLMOptimizationAgent` and strict tool schema patterns.
- OpenAI credentials only when explicitly provided.

Expected files:
- `src/asd_agent/bo/hybrid_agent.py`
- `src/asd_agent/bo/prompts.py`
- Prompt artifacts under `prompts/`
- `tests/test_bo_hybrid_schema.py`
- `tests/test_bo_hybrid_agent.py`

Acceptance criteria:
- Hybrid agent can run in no-API dry-run or mocked-LLM tests.
- LLM output never directly recommends an untested experiment.
- BO candidate recommendations remain inside safety bounds.
- Prompt records store concise decision rationales only.

Tests:
- Strict schema validation.
- Mocked LLM tool-call behavior.
- Untested-final-condition rejection.
- No-API deterministic fallback behavior.

Explicit non-goals:
- No live OpenAI calls in tests.
- No hidden chain-of-thought storage.
- No replacement of deterministic BO baselines.

### BO-09: Research-Study Harness and Statistics

Purpose:
- Add a reproducible research harness for repeated BO, baseline, and hybrid-agent
  experiments with statistical summaries.

Dependencies:
- BO-07 and BO-08.
- Scientific analysis dependencies only after explicit addition.

Expected files:
- `src/asd_agent/bo/study.py`
- `src/asd_agent/bo/statistics.py`
- `tests/test_bo_study.py`
- `tests/test_bo_statistics.py`
- `docs/study_protocol.md`

Acceptance criteria:
- Study runs record seeds, scenario versions, method versions, dependency versions,
  budgets, and failure categories.
- Statistical summaries are reproducible from saved CSV/JSON outputs.
- LLM variability analysis remains separate from deterministic BO variability.

Tests:
- Study manifest serialization.
- Statistical summary calculations on small fixed fixtures.
- Reproducibility of repeated deterministic runs.

Explicit non-goals:
- No claims that toy outcomes generalize to real chemistry.
- No live API calls in default tests.
- No publication-ready conclusions without separate review.

### BO-10: Laboratory Handoff, Notebooks, and Final Documentation

Purpose:
- Prepare the final tutorial materials, notebooks, and handoff documentation for users
  who want to understand or extend the BO workflows.

Dependencies:
- BO-01 through BO-09.

Expected files:
- New BO notebooks under `notebooks/`
- README updates
- `docs/bo_user_guide.md`
- `docs/lab_handoff.md`
- Final prompt documentation under `prompts/`

Acceptance criteria:
- Notebooks explain simulator, BO, constrained BO, hybrid LLM-BO, benchmarks, and
  failure modes without requiring live API calls unless clearly marked.
- README gives install paths for base, dev, BO, and LLM extras.
- Handoff docs state limitations and non-predictive chemistry scope.
- Full verification suite passes.

Tests:
- Notebook import/parse validation.
- Documentation smoke checks where practical.
- Full `ruff format --check`, `ruff check`, `mypy`, and `pytest`.

Explicit non-goals:
- No real-lab deployment automation.
- No undisclosed credentials or prompt logs.
- No expanded chemistry claims.

