# ASD Agent Tutorial

Educational Python tutorial for an AI-agent loop in a virtual area-selective deposition
(ASD) laboratory. The project is inspired by a 2026 ALD optimization paper in which
reasoning-model agents iteratively propose dose times, receive measurements, and decide
whether a process is saturated enough to recommend or abandon.

This repository does **not** predict real HfO2/MoS2 chemistry. It uses a toy simulator
to teach agent design, optimization, benchmarking, reproducibility, and failure modes.

## Python Target

The tutorial targets Python 3.14+, following the latest stable CPython release line
available from python.org on July 10, 2026: Python 3.14.6. Python 3.15 is still
pre-release at that date.

The BO integration has also been exercised in this repository's development
environment with Python 3.12.13. That records the actually tested local runtime; it
does not replace the declared project target.

## Quick Start

```bash
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[dev,llm,notebooks]"
```

For Bayesian-optimization tutorials and smoke checks, install the optional BO stack:

```bash
.venv\Scripts\python -m pip install -e ".[dev,bo,bo-gp,bo-analysis,notebooks]"
```

`bo-gp` installs PyTorch, GPyTorch, and BoTorch. `llm` is only needed for live OpenAI
agent experiments; all default BO, hybrid, and research smoke commands run without
live API calls.

Run a no-API deterministic demo:

```bash
asd-demo --scenario inherent_selectivity --agent rule_based
```

Run the benchmark with at least 20 repetitions:

```bash
asd-benchmark --repetitions 20 --output-dir results/benchmark_demo
```

Use the LLM agent:

```bash
set OPENAI_API_KEY=sk-...
set OPENAI_MODEL=gpt-5.6
asd-demo --scenario inhibitor_selectivity --agent llm
```

The LLM path uses the OpenAI Responses API with strict function tools:
`propose_experiments` and `finish_optimization`.

## Bayesian Optimization Extension

The BO tutorial layers are:

- Stage 1 saturation learning with fixed grid, generic GP, and physics-informed GP.
- Stage 2 constrained multi-objective ASD optimization with random, grid, rule-based,
  and constrained MOBO methods.
- Hybrid LLM-BO orchestration with BO as the numerical optimizer and a deterministic
  fake LLM for no-API tests.
- Paired research profiles and statistical analysis.
- Manual laboratory handoff for human-operated validation planning.

Useful smoke commands from a source checkout:

```bash
python scripts/run_stage1_study.py --profile smoke --output-dir results/bo04_stage1_smoke
python scripts/run_stage2_benchmark.py --profile smoke --output-dir results/bo07_stage2_smoke
python scripts/run_hybrid_fake_llm_smoke.py --output-dir results/bo10_hybrid_fake_llm_smoke
python scripts/run_research_study.py --profile smoke --output-dir results/bo09_research_smoke
python scripts/run_manual_lab_smoke.py --output-dir results/bo10_manual_lab_smoke
```

Paper-scale profiles are configured but intentionally not run by default:

```bash
python scripts/run_research_study.py --profile paper_non_llm --output-dir results/paper_non_llm
python scripts/run_research_study.py --profile paper_llm --output-dir results/paper_llm
```

Run those only after confirming compute budget, dependency versions, and study design.

## What Is Modeled

- Growth area (GA) and non-growth area (NGA, representing MoS2 in the tutorial story).
- Saturating precursor-dose and coreactant-dose responses.
- Surface-specific maximum growth per cycle.
- Surface-specific nucleation delay.
- Optional inhibitor blocking, stronger on NGA in the inhibitor scenario.
- Optional temperature response.
- Configurable Gaussian measurement noise.
- Thickness after `N` cycles.
- Selectivity `(GA - NGA) / (GA + NGA)`, with zero returned when both are zero.

## Scenarios

- `inherent_selectivity`: GA nucleates quickly; NGA has a long nucleation delay.
- `inhibitor_selectivity`: inhibitor strongly blocks NGA and weakly affects GA.
- `impossible_selectivity`: tested bounds contain no meaningful selective window.

## Default Success Criteria

- GA thickness >= 5 nm.
- NGA thickness <= 0.5 nm.
- Selectivity >= 0.80.
- Temperature and dose times inside configured safety bounds.

## Repository Layout

```text
configs/        Scenario definitions
src/asd_agent/  Simulator, agents, baselines, BO, benchmark, plotting
notebooks/      Tutorial notebooks
docs/           Protocols, model cards, implementation notes
tests/          Pytest suite
results/        Generated ledgers, metadata, plots, and benchmark outputs
```

## Reproducibility

Every run records configuration, timestamp, seed, model name, token usage, and an
experiment ledger as CSV and JSON. Random-search and simulator noise use explicit
NumPy seeds. The default demo mode uses the deterministic rule-based agent and does
not require API credentials.

BO research runs additionally record paired scenario/repetition IDs and named seeds
for simulator, measurement noise, initialization, BO, and LLM behavior. Statistical
analysis outputs are regenerated from saved CSV/JSON rows.

## Manual Laboratory Handoff

`ManualLabBackend` exports validated Stage 2 optimizer candidates as CSV and JSON lab
plans, marks them pending, imports completed human-operated measurements, validates
nanometer units and required fields, and converts accepted measurements into Stage 2
observations so BO can continue.

This is a handoff and record-keeping workflow only. It does not control a reactor,
does not replace laboratory safety review, and does not validate the toy simulator as
real chemistry. See `docs/lab_validation_protocol.md`.

## Limitations

- The simulator is educational and non-predictive.
- Oracle information is for evaluation only and must not enter optimizer or LLM inputs.
- Smoke and pilot profiles are software checks, not research conclusions.
- Live LLM calls are opt-in and require environment credentials.
- Real laboratory validation requires independent safety review, instrument mapping,
  measurement uncertainty protocols, and human sign-off.

## Source Inspiration

The high-level agent loop is inspired by Angel Yanguas-Gil, "Performance of AI agents
based on reasoning language models on ALD process optimization tasks," J. Vac. Sci.
Technol. A 44, 043410 (2026), DOI: 10.1116/6.0005313. The simulator and scenarios here
are intentionally simplified and educational.
