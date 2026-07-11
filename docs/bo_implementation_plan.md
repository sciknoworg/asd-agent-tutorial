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

BO-00 made no dependency changes. BO-01 adds optional dependency extras only; the
base tutorial path remains unchanged.

BO-01 dependency resolution was performed with a pip dry run on the available
Windows Python 3.12.13 runtime. The resolved optional stack was:

- `bo`: SciPy 1.18.0 and scikit-learn 1.9.0.
- `bo-gp`: PyTorch 2.13.0, GPyTorch 1.15.2, and BoTorch 0.18.1.
- `bo-ax`: Ax platform 1.3.1.
- `bo-analysis`: statsmodels 0.14.6, seaborn 0.13.2, and tqdm 4.68.4.

These packages are declared as optional extras and are not imported by top-level
package modules. The dry run did not install the heavy BO stack. Python 3.14
runtime compatibility was not verified locally.

### PyTorch

- PyTorch is declared only in the `bo-gp` optional extra.
- BO-01 verified resolver compatibility on Python 3.12.13 only. It did not verify
  wheel availability on a Python 3.14 interpreter.
- Keep CPU-only installation instructions separate from CUDA-specific instructions.
- Avoid importing `torch` from package top-level modules so non-BO users can keep using
  the simulator without PyTorch installed.

### GPyTorch

- GPyTorch is declared only in the `bo-gp` optional extra and is not imported by BO-01
  infrastructure.
- Use it for Gaussian-process model definitions and likelihoods, not for domain
  normalization or ledger conversion.
- Keep the first GPyTorch wrapper minimal and covered by deterministic tests on small
  synthetic data.

### BoTorch

- BoTorch is declared only in the `bo-gp` optional extra.
- Prefer BoTorch for acquisition functions, candidate generation, and constrained BO
  once the basic GP path is stable.
- Keep BoTorch-specific objects behind adapter functions so benchmark and agent code
  can remain library-agnostic.
- Record random seeds and candidate-generation settings in run metadata.

### Ax

- Ax is declared separately in `bo-ax` and should remain tutorial-facing rather than
  core infrastructure.
- Consider Ax only after BoTorch-based flows are working, because Ax may obscure the
  mechanics that the tutorial is meant to teach.
- If included, place it behind a separate optional extra such as `bo-ax`.

### Scientific Analysis Dependencies

- SciPy and scikit-learn are declared in the `bo` extra for shared BO utilities and
  future non-GP comparisons.
- statsmodels, seaborn, and tqdm are declared separately in `bo-analysis`.
- Keep statistical reporting reproducible by recording seeds, repetitions, scenario
  names, method names, and versions in benchmark outputs.

## Work Packages

### BO-01: Shared BO Infrastructure

Purpose:
- Create the reusable BO substrate for later stages without implementing Gaussian
  processes or acquisition functions.
- Add typed interfaces, proposal records, optimizer-visible observations, run
  manifests, optimizer checkpoints, and run-record serialization.
- Provide a `VirtualASDBackend` that wraps the existing simulator and preserves hidden
  simulator parameters outside optimizer-facing records.

Dependencies:
- Existing `ProcessConfig`, `ExperimentCondition`, `ExperimentRecord`, `OptimizationRun`,
  `VirtualLab`, `validate_safety`, and `run_conditions`.
- Optional BO dependencies are declared in extras, but BO-01 code does not import the
  heavy GP stack.

Expected files:
- `src/asd_agent/bo/__init__.py`
- `src/asd_agent/bo/interfaces.py`
- `src/asd_agent/bo/records.py`
- `src/asd_agent/bo/backend.py`
- `src/asd_agent/bo/serialization.py`
- `tests/test_bo_backend.py`
- `tests/test_bo_records.py`
- `docs/implementation/BO-01.md`
- Updates to this roadmap and `docs/bo_decision_log.md`.

Acceptance criteria:
- Existing baseline tests remain unchanged and passing.
- Candidate proposal IDs are unique.
- BO records serialize and deserialize as JSON.
- Run manifests record commit, config hash, Python version, dependency versions,
  operating system, named seeds, method, scenario, budget, and timestamps.
- `VirtualASDBackend` executes the existing simulator and rejects unsafe candidates.
- Optimizer-facing records exclude hidden simulator surface parameters.
- Existing `ExperimentRecord` rows can be wrapped for backward compatibility.

Tests:
- Candidate ID uniqueness.
- BO record, optimizer-state, and run-record round trips.
- Manifest creation.
- Simulator backend execution and safety rejection.
- Hidden-field exclusion from optimizer-visible payloads.
- Backward-compatible wrapping of existing records.

Explicit non-goals:
- No Gaussian-process fitting.
- No acquisition optimization.
- No Stage 1 process models.
- No Stage 2 constrained optimization.
- No hybrid LLM-BO behavior.

### BO-02: Stage 1 Process Models and Oracle

Purpose:
- Define one-dimensional Stage 1 saturation-process families and an evaluation-only
  oracle for ALD precursor-dose active-learning tutorials.
- Support known-target and inferred-asymptote modes as separate configuration modes.
- Prepare recommendation metrics without implementing adaptive optimization.

Dependencies:
- BO-01 shared infrastructure.
- Existing JSON-compatible YAML configuration loading.
- NumPy and Matplotlib from the base tutorial dependencies.

Expected files:
- `src/asd_agent/bo/stage1.py`
- `src/asd_agent/bo/oracle.py`
- `configs/bo_stage1_fast_mono.yaml`
- `configs/bo_stage1_slow_mono.yaml`
- `configs/bo_stage1_soft_biexponential.yaml`
- `configs/bo_stage1_noisy.yaml`
- `configs/bo_stage1_weak_nonselflimited.yaml`
- `configs/bo_stage1_misspecified.yaml`
- `scripts/plot_stage1_processes.py`
- `tests/test_bo_stage1.py`
- `docs/implementation/BO-02.md`
- Updates to this roadmap and `docs/bo_decision_log.md`.

Acceptance criteria:
- Six configured process families are loadable through the repository configuration
  system.
- Stage 1 virtual lab observations are deterministic for fixed seeds and noise
  settings.
- The oracle reports finite saturation values, analytical t95 where available,
  numerical t95 for non-analytical monotonic models, dense true curves, and
  non-self-limited classifications.
- Optimizer-facing views and records do not expose hidden process families, rate
  constants, true t95, true asymptotes, or scenario labels.
- Recommendation metrics include estimated and true t95, absolute and relative error,
  growth fraction, dose overshoot, cumulative dose, cumulative process time, and false
  saturation declarations.

Tests:
- Monotonic mono-exponential behavior.
- Analytical t95 formula.
- Numerical t95 for soft saturation.
- Soft-saturation stability.
- Deterministic noisy observations.
- Non-self-limited oracle classification.
- Model-misspecified saturation behavior.
- Known-target and inferred-asymptote separation.
- Oracle isolation from optimizer-facing APIs.

Explicit non-goals:
- No GP fitting.
- No adaptive experiment selection.
- No Stage 2 work.
- No LLM integration.

### BO-03: Stage 1 Generic GP and Grid Comparison

Purpose:
- Implement and benchmark fixed-grid and generic Gaussian-process active-learning
  loops on Stage 1 models.
- Use a threshold-oriented decision rule because Stage 1 is about finding the smallest
  tested dose that reaches a saturation threshold, not maximizing growth.

Dependencies:
- BO-01 and BO-02.
- Optional BO-GP dependencies from BO-01: PyTorch, GPyTorch, and BoTorch.
- Local verification used the bundled Python 3.12.13 runtime with PyTorch 2.13.0+cpu,
  GPyTorch 1.15.2, and BoTorch 0.18.1.

Expected files:
- `src/asd_agent/bo/gp.py`
- `src/asd_agent/bo/acquisition.py`
- `src/asd_agent/bo/optimizers.py`
- `tests/test_bo_gp.py`
- `tests/test_bo_acquisition.py`
- `scripts/compare_stage1_methods.py`
- `docs/implementation/BO-03.md`

Acceptance criteria:
- Generic GP BO runs reproducibly on Stage 1 process-family problems using float64,
  normalized inputs, standardized outputs, a stationary Matern kernel, and explicit
  known-noise or inferred-noise modes.
- Grid-search comparison uses predetermined dose grids, shared initial observations,
  matched simulator seeds, and the same experiment budget as adaptive methods.
- Threshold decisions evaluate posterior target probability over a candidate grid,
  prefer the smallest sufficiently probable untested dose, fall back to uncertainty
  reduction near the threshold, and return explicit failures when no valid candidate
  remains.
- Final recommendations refer only to tested Stage 1 experiment IDs.
- Smoke comparison output records method, scenario, status, experiment count,
  recommendation, best observed growth, cumulative dose, and cumulative process time.

Tests:
- GP model can fit a tiny deterministic dataset.
- Acquisition scores finite candidate sets without NaNs and handles duplicate
  candidates.
- BO loop does not repeat a tested candidate unless the candidate set is exhausted.
- Deterministic suggestions, tested recommendations, fit-failure recording, state
  save/restore, and matched smoke comparisons are covered.

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
