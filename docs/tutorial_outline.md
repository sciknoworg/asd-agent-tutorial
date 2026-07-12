# Tutorial Outline

This outline summarizes the completed ASD agent tutorial and the BO extension.

## Part 1: Virtual Laboratory

- Define GA and NGA surfaces.
- Explain saturating precursor and coreactant response.
- Show nucleation delays, inhibitor blocking, temperature response, and noise.
- Emphasize that the model is educational and non-predictive.

Notebook: `notebooks/01_understanding_the_virtual_lab.ipynb`

## Part 2: Baseline Optimization

- Run deterministic rule-based search.
- Compare random and grid baselines.
- Inspect ledgers, success criteria, and failure categories.

Notebook: `notebooks/02_rule_based_optimization.ipynb`

## Part 3: LLM Agent Loop

- Explain strict tools: `propose_experiments` and `finish_optimization`.
- Separate Codex as software-building agent from the scientific optimization agent.
- Demonstrate no-API deterministic alternatives.

Notebook: `notebooks/03_llm_agent.ipynb`

## Part 4: Legacy Benchmarking

- Compare random, grid, rule-based, and optional LLM methods.
- Generate summary plots and discuss failure modes.

Notebook: `notebooks/04_benchmark_and_failure_analysis.ipynb`

## Part 5: Stage 1 Saturation Learning

- Define one-dimensional saturation-process families.
- Compare fixed grid, generic GP, and physics-informed GP.
- Discuss t95 estimation, cumulative dose, calibration, and misspecification.

Notebook: `notebooks/05_bayesian_active_learning_for_saturation.ipynb`

## Part 6: Stage 2 Constrained MOBO

- Define objectives and constraints for two-surface ASD optimization.
- Explain why selectivity alone is insufficient.
- Compare constrained MOBO against random, grid, and rule-based search.

Notebook: `notebooks/06_constrained_multiobjective_asd_optimization.ipynb`

## Part 7: Hybrid LLM-BO Agent

- Use BO as the numerical optimizer.
- Use the LLM to inspect history, request candidates, explain choices, and propose
  bounded interventions.
- Run the fake LLM by default.

Notebook: `notebooks/07_hybrid_llm_bo_agent.ipynb`

## Part 8: Research Harness

- Configure paired smoke, pilot, and paper profiles.
- Generate paired statistical summaries and failure taxonomies.
- Keep hypotheses separate from conclusions.

Notebook: `notebooks/08_research_benchmark_and_statistics.ipynb`

## Part 9: Manual Laboratory Handoff

- Export validated manual lab plans.
- Import completed measurements.
- Continue BO from accepted human-operated measurements.
- Document remaining work before any real laboratory validation.

Primary docs: `docs/lab_validation_protocol.md` and model cards under
`docs/model_cards/`.
