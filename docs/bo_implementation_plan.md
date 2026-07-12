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
- Add a physics-informed Stage 1 surrogate with a trainable saturating mean and a GP
  residual.
- Complete the Stage 1 comparison framework for fixed grid, generic GP, and
  physics-informed GP.

Dependencies:
- BO-01 through BO-03.
- Stage 1 process-model assumptions documented in BO-02.
- Optional BO-GP dependencies verified in BO-03.

Expected files:
- `src/asd_agent/bo/physics_models.py`
- `src/asd_agent/bo/study.py`
- `tests/test_bo_physics_gp.py`
- `configs/bo_stage1_smoke_profile.yaml`
- `configs/bo_stage1_pilot_profile.yaml`
- `scripts/run_stage1_study.py`
- `notebooks/05_bayesian_active_learning_for_saturation.ipynb`
- `docs/implementation/BO-04.md`
- Updates to `src/asd_agent/bo/optimizers.py`, this roadmap, and the decision log.

Acceptance criteria:
- The physics-informed model uses positive constrained trainable `g_inf` and `k`
  parameters in a saturating mean, with a stationary residual GP for uncertainty.
- The implementation is accurately described as a physics-informed GP, not a fully
  Bayesian posterior over physical parameters.
- Grid, generic GP, and physics-informed GP share initial observations, budgets,
  candidate grid size, target definitions, stopping tolerances, noise streams, and
  scenario instances through `Stage1RunnerSettings`.
- Per-run summaries report success, experiments used, estimated and true t95,
  absolute and relative t95 error, growth fraction, dose overshoot, false saturation,
  cumulative dose, process time, uncertainty coverage, model-fit warning count,
  failure category, and optimizer wall time.
- Deterministic Stage 1 scripts produce posterior, t95, experiment-count,
  cumulative-dose, calibration, and misspecification figures with source data beside
  each figure.
- A CPU-only notebook demonstrates the Stage 1 active-learning workflow.

Tests:
- Positive physical parameters.
- Finite posterior predictions.
- Compatibility with mono-exponential data.
- Graceful behavior under model misspecification.
- Fallback recording when the physics-informed fit fails.
- Reproducible suggestions for fixed seeds.
- Complete Stage 1 result schema.

Explicit non-goals:
- No claim of real HfO2/MoS2 predictive value.
- No Stage 2 constrained multi-objective BO.
- No production lab recommendations.
- No paper-scale repeated experiments.

### BO-05: Stage 2 Problem Definition and Scenarios

Purpose:
- Define the Stage 2 ASD BO problem using GA thickness, NGA thickness, selectivity,
  process time, safety bounds, and scenario-specific objectives.
- Prepare the two-surface ASD simulator for constrained multi-objective optimization
  without implementing a MOBO acquisition function.

Dependencies:
- BO-01 shared infrastructure.
- Existing ASD scenarios and objective functions.
- Lessons from BO-02 through BO-04.

Expected files:
- `src/asd_agent/bo/stage2.py`
- `src/asd_agent/bo/stage2_oracle.py`
- `configs/bo_stage2_*.yaml`
- `scripts/plot_stage2_oracle_surfaces.py`
- `tests/test_bo_stage2.py`
- `docs/stage2_scenarios.md`
- `docs/implementation/BO-05.md`

Acceptance criteria:
- Stage 2 problem definitions can be built from existing `ProcessConfig` objects.
- Decision variables are exactly precursor dose, temperature, and integer cycle count.
- Measured outcomes are GA thickness, NGA thickness, selectivity, and simulated
  process time.
- Objectives are explicit: maximize useful GA growth, minimize NGA growth, and
  minimize process time. Selectivity is used as a constraint, not a sole objective.
- Constraint thresholds are kept in YAML: minimum GA thickness, maximum NGA thickness,
  and minimum selectivity.
- Hard bounds cover precursor dose, temperature, integer cycle count, and optional
  total process time.
- The evaluation-only oracle enumerates the mixed-variable grid, reports feasible
  regions, approximate Pareto front, approximate hypervolume, and whether a selective
  window exists.
- Existing scenarios remain loadable and unchanged.
- Simulator-only plots are generated for GA, NGA, selectivity, and feasibility.

Tests:
- Stage 2 construction for all three existing scenarios.
- Constraint values match existing `evaluate_objective` behavior.
- Impossible scenario remains impossible under tested bounds.
- Integer cycle validation, stable selectivity, zero-growth handling, safety bounds,
  feasibility, oracle isolation, and scenario reproducibility.

Explicit non-goals:
- No multi-objective optimizer implementation.
- No LLM integration.
- No changes to default demo or benchmark CLI behavior.
- No additional decision variables beyond precursor dose, temperature, and cycle count.

### BO-06: Stage 2 Constrained Multi-Objective BO

Purpose:
- Implement constrained noisy multi-objective BO for ASD Stage 2, balancing useful GA
  growth, NGA suppression, process time, configured feasibility constraints, and hard
  safety bounds.

Dependencies:
- BO-01 and BO-05.
- Optional BO-GP stack verified locally in prior stages: PyTorch, GPyTorch, and
  BoTorch.
- Stage 2 problem definitions, safety validation, measured outcomes, and
  evaluation-only oracle from BO-05.

Expected files:
- `src/asd_agent/bo/stage2_mobo.py`
- `tests/test_bo_stage2_mobo.py`
- `scripts/run_stage2_mobo_smoke.py`
- `docs/implementation/BO-06.md`
- Updates to this roadmap and `docs/bo_decision_log.md`.

Acceptance criteria:
- The surrogate uses one GP per measured outcome with float64 tensors, normalized
  inputs, standardized outputs, Matern kernels, known-noise or inferred-noise mode,
  deterministic seeds, fit retries, and jitter escalation.
- Acquisition uses BoTorch's log-stabilized noisy expected hypervolume improvement
  where available, with qNEHVI fallback on older releases.
- Acquisition includes outcome constraints for minimum GA, maximum NGA, and minimum
  selectivity.
- The reference point is configurable and defaults to a threshold-based scientific
  value, not a value derived from benchmark results.
- Cycle count is treated as an integer by enumerating configured or scenario cycle
  values and optimizing continuous variables conditional on each fixed count.
- Initial observations use seeded Sobol designs, and existing ASD ledger rows can be
  converted into Stage 2 observations.
- Candidate records include acquisition value, feasibility probability, posterior
  summaries, training observation IDs, duplicate counts, fallback use, and wall time.
- Runs record hypervolume by iteration, constraint violations, fallback logging, and
  tested-row recommendations.
- Small smoke optimization runs on inherent selectivity, narrow selective window, and
  impossible selectivity.

Tests:
- Finite acquisition values.
- Safety bounds and integer cycle-count compliance.
- Outcome-constraint sign conventions.
- Candidate uniqueness and deterministic suggestions.
- Optimizer state save/restore.
- Model-fit fallback recording.
- Reuse of existing ledger observations.
- Oracle isolation from optimizer artifacts.
- Smoke run across inherent selectivity, narrow selective window, and impossible
  selectivity.

Explicit non-goals:
- No hidden LLM reasoning or live OpenAI calls.
- No real-lab recommendation claims.
- No changes to current random/grid/rule-based results.
- No Stage 2 benchmark statistics beyond the small smoke runner.
- No hybrid LLM-BO behavior.

### BO-07: Stage 2 Benchmark and Analysis

Purpose:
- Extend benchmarking and plotting to compare random search, grid search,
  deterministic rule-based search, and constrained MOBO on the Stage 2 ASD scenarios.
- Use feasible-hypervolume trajectory AUC as the primary fixed-budget endpoint.

Dependencies:
- BO-06.
- Stage 2 benchmark observations, oracle hypervolume, and MOBO run records.
- Existing configuration loading and Matplotlib plotting stack.

Expected files:
- `src/asd_agent/bo/stage2_benchmark.py`
- `src/asd_agent/bo/stage2_analysis.py`
- `configs/bo_stage2_smoke_profile.yaml`
- `configs/bo_stage2_pilot_profile.yaml`
- `scripts/run_stage2_benchmark.py`
- `tests/test_bo_stage2_benchmark.py`
- `notebooks/06_constrained_multiobjective_asd_optimization.ipynb`
- `docs/implementation/BO-07.md`

Acceptance criteria:
- Stage 2 methods share scenario configs, simulator seeds, optimizer seeds, budgets,
  and matched Sobol initial designs where applicable.
- Random, grid, and rule-based methods operate on the Stage 2 variables only:
  precursor dose, temperature, and integer cycle count.
- Primary endpoint is area under the feasible hypervolume trajectory under fixed
  budget.
- Secondary endpoints include final feasible success rate, experiments to first
  feasible condition, final hypervolume, hypervolume regret, constraint-violation
  counts, boundary proposals, and failure taxonomy.
- Analysis exports search trajectories, observed and oracle Pareto fronts,
  hypervolume trajectories, hypervolume AUC, final regret, experiments to first
  feasible point, constraint-violation counts, robustness by scenario, and failure
  taxonomy.
- Every generated figure has a CSV source file beside it.
- Smoke and pilot profiles are tutorial-scale and do not run paper-scale experiments.

Tests:
- Benchmark smoke run with small repetitions.
- Summary-row schema compatibility.
- Analysis-generation smoke test using a noninteractive backend.

Explicit non-goals:
- No LLM-BO hybrid behavior.
- No statistical publication claims beyond descriptive benchmark summaries.
- No dependency changes.
- No paper-scale Stage 2 runs.

### BO-08: Hybrid LLM-BO Agent

Purpose:
- Combine LLM-style scientific planning with BO candidate generation while preserving
  strict tool calling, tested-final-condition validation, concise rationales, and a
  clear separation between LLM, BO, and backend responsibilities.

Dependencies:
- BO-06 Stage 2 constrained MOBO.
- BO-07 Stage 2 benchmark/result conventions.
- Existing strict tool schema patterns from the legacy LLM-only agent.

Expected files:
- `src/asd_agent/bo/hybrid_agent.py`
- `tests/test_bo_hybrid_agent.py`
- `notebooks/07_hybrid_llm_bo_agent.ipynb`
- `docs/implementation/BO-08.md`
- Updates to this roadmap and `docs/bo_decision_log.md`

Acceptance criteria:
- Hybrid agent can run in no-API fake-LLM mode.
- State machine covers initialize, inspect history, request BO candidates, review
  candidates, execute candidate, observe, continue, change soft bounds, finish, and
  declare no selective window.
- Strict tools include history inspection, BO execution, virtual experiment execution
  by candidate ID, soft-bound changes, finish, no-window declaration, and literature
  query.
- LLM-facing context excludes oracle values and hidden simulator parameters.
- `run_virtual_experiment` accepts immutable candidate IDs only, not arbitrary
  numerical reactor settings.
- Soft bounds are mutable only inside immutable hard bounds.
- Final success recommendations must reference tested feasible experiments.
- No-window declarations must cite existing evidence IDs.
- Malformed tool calls do not mutate optimizer or experiment state.
- Literature retrieval defaults to null/mock/local providers with no mandatory live
  web access.
- Modes are available for `bo_only`, `llm_only_legacy`, `hybrid_advisory`,
  `hybrid_intervention`, `hybrid_explanation_only`, and `rule_based_bo`.

Tests:
- Strict schema validation.
- Fake LLM tool-call behavior.
- Untested-final-condition rejection.
- Infeasible-final-condition rejection.
- Candidate-ID execution bypass prevention.
- Soft-bound validation.
- Evidence-ID validation.
- Oracle isolation.
- Malformed-tool state safety.
- Local literature provider behavior.
- No-API deterministic fallback behavior.

Explicit non-goals:
- No live OpenAI calls in tests.
- No hidden chain-of-thought storage.
- No replacement of the legacy LLM-only comparator.
- No mandatory live literature retrieval.

### BO-09: Research-Study Harness and Statistics

Purpose:
- Add a reproducible paired research harness for repeated Stage 1, Stage 2, and
  hybrid-agent experiments with statistical summaries.

Dependencies:
- BO-07 and BO-08.
- Existing BO optional dependencies. BO-09 does not add new package requirements.

Expected files:
- `src/asd_agent/bo/research.py`
- `src/asd_agent/bo/statistics.py`
- `configs/bo_research_smoke_profile.yaml`
- `configs/bo_research_pilot_profile.yaml`
- `configs/bo_research_paper_non_llm_profile.yaml`
- `configs/bo_research_paper_llm_profile.yaml`
- `scripts/run_research_study.py`
- `tests/test_bo_research.py`
- `docs/research_protocol.md`
- `notebooks/08_research_benchmark_and_statistics.ipynb`
- `docs/implementation/BO-09.md`

Acceptance criteria:
- Research profiles define smoke, pilot, paper non-LLM, and paper LLM repetition
  counts, with paper-scale profiles left unrun by default.
- Study rows record paired scenario/repetition identifiers, named seeds, method,
  status, success, primary metric, and failure category.
- Stage 1, Stage 2, and hybrid runs share matched scenario instances, initial
  designs where applicable, and deterministic noise streams.
- Statistical summaries are reproducible from normalized rows and are exported as
  CSV, JSON, Markdown, and LaTeX tables.
- The fake-LLM hybrid path is used by default; live API variability remains a
  future opt-in study design.

Tests:
- Paired seed scheduling and named seed determinism.
- Statistical summary calculations on fixed fixtures.
- Holm correction, empty-result handling, and artifact generation.
- Reproducibility of a tiny deterministic paired run.

Explicit non-goals:
- No claims that toy outcomes generalize to real chemistry.
- No live API calls in default tests.
- No publication-ready conclusions without separate review.
- No paper-scale profile execution in BO-09 verification.

### BO-10: Laboratory Handoff, Notebooks, and Final Documentation

Purpose:
- Prepare the final tutorial materials, notebooks, and handoff documentation for users
  who want to understand or extend the BO workflows.
- Add a human-operated manual laboratory handoff backend that exports validated plans
  and ingests completed measurements without autonomous reactor control.

Dependencies:
- BO-01 through BO-09.

Expected files:
- `src/asd_agent/bo/manual_lab.py`
- `scripts/run_manual_lab_smoke.py`
- `tests/test_manual_lab.py`
- `docs/lab_validation_protocol.md`
- `docs/tutorial_outline.md`
- `docs/model_cards/generic_gp.md`
- `docs/model_cards/physics_informed_gp.md`
- `docs/model_cards/constrained_mobo.md`
- `docs/model_cards/hybrid_agent.md`
- `docs/implementation/BO-10.md`
- README and `AGENTS.md` updates.

Acceptance criteria:
- `ManualLabBackend` validates Stage 2 candidates, exports CSV/JSON plans, marks
  experiments pending, imports completed measurements, validates nanometer units and
  required fields, updates the Stage 2 observation ledger, and returns observations
  that can seed continued BO.
- README gives install paths for base, dev, BO, analysis, notebooks, and LLM extras.
- Handoff docs state human-operator responsibilities, limitations, and
  non-predictive chemistry scope.
- Model cards summarize generic GP, physics-informed GP, constrained MOBO, and hybrid
  agent assumptions and limitations.
- Full verification suite passes.

Tests:
- Manual-lab plan export, hard-bound rejection, measurement validation, QC rejection,
  ledger update, and BO continuation.
- Full `ruff format --check`, `ruff check`, `mypy`, and `pytest`.
- Legacy demo, reduced benchmark, Stage 1 smoke, Stage 2 smoke, hybrid fake-LLM
  smoke, research-analysis smoke, and manual export/ingestion smoke.

Explicit non-goals:
- No real-lab deployment automation.
- No undisclosed credentials or prompt logs.
- No expanded chemistry claims.
- No live LLM calls or paper-scale runs in final verification.
