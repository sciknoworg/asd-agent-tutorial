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

## Quick Start

```bash
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[dev,llm,notebooks]"
```

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
src/asd_agent/  Simulator, agents, baselines, benchmark, plotting
notebooks/      Four tutorial notebooks
tests/          Pytest suite
results/        Generated ledgers, metadata, plots, and benchmark outputs
```

## Reproducibility

Every run records configuration, timestamp, seed, model name, token usage, and an
experiment ledger as CSV and JSON. Random-search and simulator noise use explicit
NumPy seeds. The default demo mode uses the deterministic rule-based agent and does
not require API credentials.

## Source Inspiration

The high-level agent loop is inspired by Angel Yanguas-Gil, "Performance of AI agents
based on reasoning language models on ALD process optimization tasks," J. Vac. Sci.
Technol. A 44, 043410 (2026), DOI: 10.1116/6.0005313. The simulator and scenarios here
are intentionally simplified and educational.

