# Research Protocol

This protocol defines reproducible virtual studies for the ASD agent tutorial.
The simulator is educational and does not predict real HfO2/MoS2 chemistry.

## Research Questions

RQ1: Physics-informed versus generic saturation learning.

Hypothesis: a physics-informed GP residual model will require fewer measurements
and produce lower t95 error than a generic stationary GP on mono-exponential and
soft-saturation Stage 1 processes. This is a testable hypothesis, not a reported
conclusion.

RQ2: Constrained MOBO versus random, grid, and rule-based search.

Hypothesis: constrained multi-objective BO will achieve larger feasible
hypervolume AUC under a fixed budget than random search, grid search, and the
deterministic rule-based method on feasible Stage 2 scenarios.

RQ3: Hybrid orchestration versus BO alone.

Hypothesis: a hybrid LLM-BO orchestrator can improve robustness on difficult
scenarios by inspecting history and adjusting validated soft bounds, while BO
alone remains the primary numerical comparator.

RQ4: Robustness to noise, misspecification, narrow windows, boundaries, poor
initial designs, and impossible scenarios.

Hypothesis: method rankings and failure categories will change across these
stressors, and impossible scenarios should produce explicit no-window or budget
failure categories rather than untested final recommendations.

RQ5: Uncertainty and feasibility calibration.

Hypothesis: uncertainty summaries and feasibility probabilities will be better
calibrated on processes matching model assumptions than on noisy or
misspecified scenarios.

## Profiles

The harness defines four configurable profiles:

| Profile | Default repetitions | Purpose |
| --- | ---: | --- |
| `bo_research_smoke` | 2 | Fast CI/local smoke checks. |
| `bo_research_pilot` | 20 | Pilot-scale paired checks before larger runs. |
| `bo_research_paper_non_llm` | 100 | Paper-scale non-LLM Stage 1 and Stage 2 runs. |
| `bo_research_paper_llm` | 30 | Paper-scale hybrid-agent runs. |

Paper profiles are configuration only. They should not be run in routine tests.

## Paired Design

Each scenario and repetition receives a `pair_id`. All methods compared within
that pair reuse matched scenario instances, initial designs where applicable,
and deterministic noise streams. Named seeds are recorded for simulator,
measurement noise, initialization, BO, and LLM behavior.

The optimizer-facing records contain measured outcomes, candidate metadata, and
concise rationales only. Hidden simulator parameters and oracle-only quantities
must not be passed into optimizers or LLM prompts.

## Primary Endpoints

RQ1 uses paired t95 estimation error and saturation-recommendation outcomes.
RQ2 uses area under the feasible hypervolume trajectory under a fixed budget.
RQ3 uses final feasible hypervolume and success under matched hybrid scenarios.
RQ4 uses the failure taxonomy by scenario and method. RQ5 uses calibration
summaries where posterior or feasibility information is available.

## Statistical Analyses

The default analysis reports paired treatment-minus-control effect estimates,
paired bootstrap 95% confidence intervals, exact McNemar comparisons for paired
success, Wilcoxon signed-rank tests as secondary analyses, Holm correction within
research-question families, cumulative success curves, and failure-category
summaries.

Generated artifacts include CSV, JSON, Markdown, and LaTeX tables. Source rows
are saved separately from analysis outputs so statistical reports can be
regenerated without rerunning experiments.

## Reporting Rules

Report hypotheses separately from results. Do not describe smoke or pilot runs
as paper-scale evidence. Do not claim real chemistry prediction. Live LLM calls
are not part of default BO-09 profiles or tests; fake and deterministic adapters
are used unless a future task explicitly configures live API evaluation.
