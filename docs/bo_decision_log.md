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

## D-022 - Physics-Informed GP Uses a Trainable Saturating Mean

Decision:
- BO-04 implements the Stage 1 physics-informed surrogate as a GPyTorch exact GP with
  a trainable saturating mean `g_inf * (1 - exp(-k * t))` and a Matérn residual GP.

Rationale:
- The tutorial needs an explicit physical inductive bias while preserving uncertainty
  from residual model error.

Consequences:
- `g_inf` and `k` are positive constrained trainable parameters.
- The implementation is a physics-informed GP with optimized physical parameters. It
  is not described as a fully Bayesian posterior over those parameters.

## D-023 - Custom Physics Model Uses Direct GPyTorch Training

Decision:
- The physics-informed model is fit with a short deterministic Adam loop over the
  GPyTorch marginal log likelihood instead of BoTorch's generic fitting helper.

Rationale:
- The custom GPyTorch model does not expose BoTorch-specific model conveniences such
  as `transform_inputs`.

Consequences:
- Fit retries and warnings are recorded by the BO wrapper.
- Generic-GP fallback remains available when physics-informed fitting fails.

## D-024 - BO-04 Adds a Virtual Endpoint Separate From Optimizer Visibility

Decision:
- Stage 1 study success is evaluated with a hidden virtual endpoint: a tested dose
  must reach at least 95% of true finite saturation within a configurable dose
  tolerance of the minimum true threshold dose.

Rationale:
- Optimizers should not see hidden truth, but the benchmark needs a stable evaluation
  endpoint for fair comparison.

Consequences:
- Results record both the optimizer-visible recommendation and the evaluation-only
  endpoint experiment ID when one exists.
- Non-self-limited processes remain failure cases for this endpoint.

## D-025 - Stage 1 Profiles Stay Tutorial-Scale

Decision:
- BO-04 adds smoke and pilot profiles only.

Rationale:
- The current task explicitly excludes paper-scale experiments.

Consequences:
- `bo_stage1_smoke_profile` uses one repetition and a small candidate grid.
- `bo_stage1_pilot_profile` uses three repetitions and remains CPU-friendly.

## D-026 - Stage 2 Uses Three Decision Variables

Decision:
- BO-05 exposes only precursor dose, temperature, and integer cycle count as Stage 2
  decision variables.

Rationale:
- The task scope is to prepare constrained multi-objective optimization without
  expanding the search space prematurely.

Consequences:
- Coreactant dose and inhibitor dose are fixed scenario parameters.
- Future MOBO methods must propose `Stage2Decision` values, not full
  `ExperimentCondition` rows.

## D-027 - Stage 2 Selectivity Is a Constraint, Not a Sole Objective

Decision:
- Stage 2 objectives are useful GA growth, NGA suppression, and process-time
  reduction. Selectivity remains a configured feasibility constraint.

Rationale:
- Optimizing selectivity alone can favor near-zero growth and is not the stated
  scientific goal.

Consequences:
- YAML thresholds include minimum GA thickness, maximum NGA thickness, and minimum
  selectivity.
- Pareto calculations use GA, NGA, and process time.

## D-028 - Stage 2 Oracle Is Evaluation-Only

Decision:
- The Stage 2 oracle performs dense mixed-variable enumeration and reports feasible
  regions, approximate Pareto front, approximate hypervolume, and selective-window
  existence only through `Stage2EvaluationOracle`.

Rationale:
- Optimizers need measured outcomes and visible bounds, not hidden simulator truth or
  oracle summaries.

Consequences:
- `Stage2Config.optimizer_view()` excludes surfaces, hidden process parameters,
  oracle hypervolume, and selective-window labels.
- Tests assert this isolation boundary.

## D-029 - Stage 2 Scenario YAML Documents Hidden Context

Decision:
- Each `bo_stage2_*.yaml` file includes human-facing metadata for interpretation,
  hidden process parameters, noise, feasibility, and expected difficulty.

Rationale:
- The tutorial should make failure modes and scenario intent explicit while keeping
  optimizer-facing context restricted.

Consequences:
- `docs/stage2_scenarios.md` summarizes all Stage 2 scenarios.
- The original three ASD scenario YAML files remain unchanged.

## D-030 - Stage 2 MOBO Uses Transformed Outcome Coordinates

Decision:
- BO-06 models measured outcomes as GA thickness, negative NGA thickness, selectivity,
  and negative process time.

Rationale:
- BoTorch's multi-objective acquisition functions use maximization conventions. Sign
  transforms allow GA growth, NGA suppression, and process-time reduction to share one
  objective direction without changing the measured simulator outputs.

Consequences:
- Proposal records and run records still expose measured GA, NGA, selectivity, and
  process time.
- Acquisition constraints convert the negative NGA coordinate back into the configured
  maximum-NGA feasibility test.

## D-031 - Cycle Count Is Enumerated, Not Rounded

Decision:
- Stage 2 MOBO enumerates permitted or configured cycle counts and optimizes
  precursor dose and temperature conditional on each fixed integer count.

Rationale:
- Rounding an unconstrained continuous cycle suggestion can produce misleading
  acquisition values and violates the task requirement for integer handling.

Consequences:
- Candidate cycle values default to the Stage 2 scenario grid and can be overridden in
  `Stage2BOSettings`.
- Candidate records always contain integer `Stage2Decision.cycle_count` values.

## D-032 - Use Log-NEHVI When Available

Decision:
- BO-06 uses `qLogNoisyExpectedHypervolumeImprovement` in the installed BoTorch
  0.18.1 runtime and falls back to `qNoisyExpectedHypervolumeImprovement` on older
  releases.

Rationale:
- The task requested the current supported log-stabilized noisy hypervolume
  acquisition where available.

Consequences:
- The optimizer remains portable across BoTorch releases while documenting the actual
  acquisition used in the verified local environment.
- On this Windows runtime, BoTorch fell back to a pure-Python log-EHVI path because
  the fused C++ extension could not be compiled in the user profile directory.

## D-033 - Reference Point Is Configurable and Threshold-Based

Decision:
- `Stage2BOSettings.reference_point` accepts an explicit three-objective reference
  point. When omitted, BO-06 derives a conservative default from configured
  feasibility thresholds and hard process-time bounds.

Rationale:
- The reference point should encode the scientific objective definition, not leak
  favorable values from benchmark or oracle results.

Consequences:
- Tests and smoke runs do not query `Stage2EvaluationOracle` for acquisition setup.
- Future benchmark profiles can set scenario-specific reference points without
  changing the optimizer code.

## D-034 - Random Fallback Means Hard-Safety-Feasible

Decision:
- BO-06's random fallback samples Sobol candidates that satisfy immutable hard
  parameter and process-time safety bounds and avoid duplicates.

Rationale:
- Outcome feasibility is unknown until the virtual experiment is run. Calling a
  fallback outcome-feasible before measurement would blur the optimizer/oracle
  boundary.

Consequences:
- Proposal records use `fallback_used` and `constraint_violations` to make fallback
  behavior explicit.
- The docs describe fallback candidates as hard-safety-feasible, not guaranteed
  selective-window solutions.

## D-035 - Stage 2 MOBO Is Opt-In

Decision:
- BO-06 adds `asd_agent.bo.stage2_mobo` and a smoke script but does not import the
  heavy BoTorch path from `asd_agent.bo.__init__`.

Rationale:
- Base tutorial users should not pay the PyTorch/BoTorch import cost unless they use
  the MOBO stage.

Consequences:
- Existing simulator, baseline, LLM, and notebook imports remain unchanged.
- Users import Stage 2 MOBO directly from `asd_agent.bo.stage2_mobo`.

## D-036 - BO-07 Uses Fixed-Budget Hypervolume AUC

Decision:
- Stage 2 benchmark comparisons use area under the feasible observed hypervolume
  trajectory as the primary endpoint.

Rationale:
- A final feasible hit alone does not distinguish early discovery from late discovery
  or sustained Pareto improvement. Hypervolume trajectory AUC rewards methods that
  find feasible useful tradeoffs earlier under the same budget.

Consequences:
- All methods are evaluated over a fixed budget for trajectory metrics.
- Secondary endpoints still report final success, first feasible experiment, final
  hypervolume, regret, violations, boundary proposals, and failures.

## D-037 - Stage 2 Baselines Use Stage 2 Variables Only

Decision:
- BO-07 implements Stage 2 benchmark adapters for random search, grid search, and the
  deterministic rule-based plan using precursor dose, temperature, and integer cycle
  count while keeping coreactant and inhibitor doses fixed by scenario.

Rationale:
- Stage 2 was defined as a three-variable constrained problem in BO-05. Comparing
  methods in a five-variable legacy search space would no longer match the constrained
  MOBO problem.

Consequences:
- Legacy baseline modules remain unchanged.
- Stage 2 benchmark methods live in `asd_agent.bo.stage2_benchmark`.

## D-038 - Initial Designs Are Matched Across Methods

Decision:
- BO-07 uses matched seeded Sobol initial observations for all Stage 2 benchmark
  methods in a profile.

Rationale:
- The first observations should not advantage one method through different starting
  information. After the shared initial design, each method controls its own
  additional proposals.

Consequences:
- Profile fields record simulator seeds, optimizer seeds, budget, initial design size,
  and cycle candidates.
- Random, grid, rule-based, and MOBO runs can be compared from the same starting
  evidence.

## D-039 - Analysis Artifacts Are Figure-CSV Pairs

Decision:
- Every BO-07 analysis figure is generated with a CSV source file of the same stem in
  the same directory.

Rationale:
- The tutorial should be reproducible and inspectable without reverse-engineering
  plotted values from images.

Consequences:
- The analysis module exports trajectory, Pareto, hypervolume, regret, robustness, and
  failure-taxonomy source data beside PNG figures.
- Tests assert that generated PNG files have matching CSV files.

## D-040 - BO-07 Profiles Stay Tutorial-Scale

Decision:
- BO-07 adds smoke and pilot profiles only.

Rationale:
- The task explicitly excludes paper-scale profiles.

Consequences:
- `bo_stage2_smoke_profile` is suitable for tests and notebooks.
- `bo_stage2_pilot_profile` increases scenarios, repetitions, and budget modestly but
  remains a CPU-friendly exploratory profile.

## D-041 - Hybrid Execution Uses Candidate IDs Only

Decision:
- BO-08's `run_virtual_experiment` tool accepts an immutable `candidate_id` and a
  concise rationale. It does not accept arbitrary reactor conditions.

Rationale:
- The LLM may inspect, review, explain, and decide, but numerical experimental
  conditions must be created by BO and validated before backend execution.

Consequences:
- Candidate IDs are stored in the hybrid orchestrator before execution.
- Unknown or already executed candidate IDs are rejected.
- Tests assert that raw dose, temperature, and cycle fields cannot be passed to
  `run_virtual_experiment`.

## D-042 - Soft Bounds Are Advisory Within Hard Bounds

Decision:
- The hybrid LLM can propose soft-bound changes only in `hybrid_intervention` mode,
  and those soft bounds must remain inside immutable Stage 2 hard bounds.

Rationale:
- This allows scientific steering without weakening safety constraints.

Consequences:
- The original Stage 2 configuration remains unchanged.
- Soft-bound changes narrow the optimizer-facing search copy only.

## D-043 - Final Hybrid Success Requires Tested Feasibility

Decision:
- `finish_optimization` is accepted only when the referenced experiment ID exists in
  the hybrid ledger and its measured outcomes satisfy configured constraints.

Rationale:
- This extends the repository's tested-final-condition rule to the hybrid LLM-BO
  workflow.

Consequences:
- Untested and infeasible final recommendations raise validation errors.
- Budget exhaustion can still return a best observed row, but not a successful final
  recommendation.

## D-044 - Literature Retrieval Is Optional And Local By Default

Decision:
- BO-08 implements `NullLiteratureProvider`, `MockLiteratureProvider`, and
  `LocalLiteratureProvider`, with no mandatory live web retrieval.

Rationale:
- The tutorial should remain reproducible and runnable without network access or
  credentials.

Consequences:
- Tests and notebooks use fake or local literature providers.
- Future live retrieval can be added as an optional provider without changing the
  hybrid state machine.

## D-045 - Fake LLM Exercises The Hybrid State Machine

Decision:
- BO-08 includes a deterministic `FakeHybridLLM` that can inspect history, query
  local literature, request BO, change soft bounds, execute candidate IDs, finish, or
  declare no selective window.

Rationale:
- The hybrid orchestration needs test coverage without live OpenAI calls.

Consequences:
- `notebooks/07_hybrid_llm_bo_agent.ipynb` runs with the fake LLM by default.
- Tests cover malformed-tool handling and state-transition safety.

## D-046 - BO-09 Uses Normalized Research Rows

Decision:
- BO-09 stores Stage 1, Stage 2, and hybrid outcomes as `ResearchResultRow`
  records with a shared schema for pair ID, named seeds, method, status, success,
  primary metric, metrics payload, and failure category.

Rationale:
- The statistical harness should not depend on hidden simulator fields or
  method-specific internal objects after a run is complete.

Consequences:
- CSV, JSON, Markdown, and LaTeX reports can be regenerated from normalized rows.
- Oracle-only values remain outside optimizer-facing and LLM-facing records.

## D-047 - Research Comparisons Are Paired By Scenario And Repetition

Decision:
- Every research profile creates deterministic `pair_id` values and named seeds for
  simulator, measurement noise, initialization, BO, and LLM behavior.

Rationale:
- Paired comparisons reduce variance and make method differences easier to audit.

Consequences:
- Methods within a pair reuse matched scenario instances, initial designs where
  applicable, and noise streams.
- Tests assert deterministic schedule generation.

## D-048 - Paper Profiles Are Configuration Only In BO-09

Decision:
- BO-09 defines smoke, pilot, paper non-LLM, and paper LLM profiles, but routine
  verification runs only smoke or smaller checks.

Rationale:
- The repository needs reproducible paper-scale settings without turning every
  development check into a long benchmark.

Consequences:
- `paper_non_llm` defaults to 100 paired repetitions.
- `paper_llm` defaults to 30 paired repetitions and uses the fake LLM unless a
  future task explicitly adds live API evaluation.

## D-049 - Statistical Tests Are Hypothesis-Support Tools, Not Conclusions

Decision:
- BO-09 implements paired effect estimates, bootstrap confidence intervals,
  McNemar paired-success comparisons, Wilcoxon secondary tests, Holm correction,
  cumulative success curves, and failure summaries.

Rationale:
- The tutorial should support research-style analysis while keeping conclusions
  separate from code-generation smoke checks.

Consequences:
- `docs/research_protocol.md` frames RQ1-RQ5 as testable hypotheses.
- Generated reports explicitly avoid claims about real ASD chemistry.
