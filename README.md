# ASD Agent Tutorial

Reproducible Python software for studying AI-agent loops, Bayesian active learning,
constrained multi-objective Bayesian optimization, and manual laboratory handoff in an
educational area-selective deposition (ASD) simulator.

> This project does not predict real HfO2/MoS2 chemistry. Its equations and scenarios
> are teaching models for software design, optimization, benchmarking, and failure
> analysis. Do not use its recommendations as laboratory process guidance.

## Features

- Seeded two-surface virtual laboratory with GA and NGA growth, nucleation delays,
  saturating doses, inhibitor blocking, temperature response, noise, and process time.
- Legacy random search, grid search, deterministic rule-based agent, and optional
  OpenAI Responses API agent.
- Stage 1 fixed-grid, generic-GP, and physics-informed-GP saturation learning.
- Stage 2 constrained noisy multi-objective BO over precursor dose, temperature, and
  integer cycle count.
- Hybrid LLM-BO orchestration with immutable candidate IDs and a fake LLM by default.
- Paired research profiles, bootstrap confidence intervals, McNemar comparisons,
  Wilcoxon tests, Holm correction, and publication-table exports.
- Human-operated laboratory plan export and validated measurement ingestion. No
  autonomous hardware control is included.
- Eight executable tutorial notebooks and deterministic plotting scripts.

## Requirements

- Python 3.12 through 3.14.
- The full suite is currently verified on Python 3.12.13.
- CPU execution is supported. A GPU is optional and not required by the smoke profiles.

Python 3.13 and 3.14 are compatibility targets, not verified environments in the
current repository record.

## Installation

Create a virtual environment and install the package in editable mode:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev,bo,bo-gp,bo-analysis,notebooks]"
```

Optional extras:

| Extra | Purpose |
| --- | --- |
| `bo` | NumPy/SciPy analysis helpers |
| `bo-gp` | PyTorch, GPyTorch, and BoTorch models |
| `bo-ax` | Optional Ax experiments; not required by the implemented runners |
| `bo-analysis` | Statistical and plotting utilities |
| `llm` | Optional live OpenAI Responses API paths |
| `notebooks` | JupyterLab and notebook file support |
| `dev` | pytest, Ruff, mypy, and type stubs |

## Quick Start

Run the no-API legacy tutorial:

```powershell
asd-demo --scenario inherent_selectivity --agent rule_based
asd-benchmark --repetitions 20 --output-dir results/legacy_benchmark
```

Run Bayesian-optimization workflows through the integrated CLI:

```powershell
asd-agent stage1 run --profile smoke --scenario fast_mono --method physics_gp `
  --output-dir results/stage1_smoke

asd-agent stage2 run --profile smoke --scenario inherent_selectivity `
  --method stage2_mobo --output-dir results/stage2_smoke

asd-agent hybrid run --scenario narrow_selective_window `
  --mode hybrid_advisory --fake-llm --output-dir results/hybrid_smoke
```

The fake LLM is deterministic and makes no network calls.

## CLI

```text
asd-agent stage1 run|compare
asd-agent stage2 run|compare
asd-agent hybrid run
asd-agent study run|analyze
asd-agent lab export|ingest
```

Show command-specific options with `asd-agent <group> <command> --help`. Seeds,
budgets, scenarios, profiles, and output directories are explicit command arguments or
validated YAML values.

### Research studies

```powershell
asd-agent study run --config smoke --output-dir results/study_smoke
asd-agent study analyze --input results/study_smoke/research_results.json `
  --output results/study_smoke/analysis
```

Larger profiles are configured but are never run automatically:

```powershell
python scripts/run_stage1_study.py --profile pilot --output-dir results/stage1_pilot
python scripts/run_stage2_benchmark.py --profile pilot --output-dir results/stage2_pilot
python scripts/run_research_study.py --profile paper_non_llm `
  --output-dir results/paper_non_llm
python scripts/run_research_study.py --profile paper_llm `
  --output-dir results/paper_llm
```

Review compute budgets, resolved dependencies, and the research protocol before
starting pilot or paper-scale profiles.

### Optional live LLM

Live calls require the `llm` extra, environment credentials, a model name, and an
explicit live flag:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "your-responses-api-model"
asd-agent hybrid run --scenario narrow_selective_window `
  --mode hybrid_advisory --live-llm --output-dir results/hybrid_live
```

Tests and default smoke commands never make live API calls. The live hybrid adapter is
import- and schema-tested but has not been exercised against the API in this repository.

## Manual Laboratory Handoff

Export one validated, pending experiment plan:

```powershell
asd-agent lab export --scenario inherent_selectivity --run-id manual_001 `
  --output results/manual_001
```

After a human operator completes and reviews the measurement template, import it in a
later process:

```powershell
asd-agent lab ingest --scenario inherent_selectivity `
  --plan results/manual_001/manual_lab_plan.json `
  --measurements results/manual_001/completed_measurements.csv `
  --output results/manual_001/optimizer_observations.json
```

Imported measurements require GA/NGA thickness in nanometers, a measurement method,
replicate ID, quality-control status, and operator notes. See
[`docs/lab_validation_protocol.md`](docs/lab_validation_protocol.md).

## Configuration

YAML files in `configs/` define:

- the three legacy ASD scenarios;
- six Stage 1 saturation-process families;
- eight Stage 2 constrained optimization scenarios;
- Stage 1 and Stage 2 smoke/pilot profiles;
- paired smoke, pilot, paper non-LLM, and paper LLM study profiles.

Hidden simulator parameters remain in the backend configuration. Optimizers and LLM
contexts receive explicit optimizer-facing views that exclude oracle values and hidden
process constants.

## Outputs and Reproducibility

Generated files are written below `results/` and ignored by Git except for
`results/.gitkeep`. Depending on the command, outputs include:

- experiment ledgers and optimizer proposals in JSON/CSV;
- run manifests with Git commit, configuration hash, Python/dependency versions,
  named seeds, method settings, and timestamps;
- hypervolume trajectories, failure categories, and constraint outcomes;
- CSV source data beside PNG, SVG, and PDF figures;
- JSON, CSV, Markdown, and LaTeX statistical tables;
- manual laboratory plans, templates, and imported observations.

Every stochastic simulator and optimizer path accepts an explicit seed. Paired studies
persist separate simulator, measurement-noise, initialization, BO, and LLM seed streams.

## Repository Layout

```text
configs/             Scenario and study YAML
docs/                Protocols, implementation records, and model cards
notebooks/           Eight tutorial notebooks
prompts/             Prompt-data guidance; no agent traces
scripts/             Reproducible smoke, benchmark, and plotting entry points
src/asd_agent/       Installable package
src/asd_agent/bo/    BO, hybrid, statistics, oracle, and lab-handoff modules
tests/               Unit and integration tests
results/             Ignored generated outputs
```

## Development

Run the complete local verification suite:

```powershell
ruff format --check .
ruff check .
mypy src
pytest
```

The test suite covers simulator behavior, safety bounds, schemas, oracle isolation,
tested recommendations, GP posteriors, mixed-variable MOBO, hybrid bypass prevention,
paired statistics, save/restore, and manual-lab ingestion.

## Documentation

- [Research protocol](docs/research_protocol.md)
- [Tutorial outline](docs/tutorial_outline.md)
- [Laboratory validation protocol](docs/lab_validation_protocol.md)
- [Generic GP model card](docs/model_cards/generic_gp.md)
- [Physics-informed GP model card](docs/model_cards/physics_informed_gp.md)
- [Constrained MOBO model card](docs/model_cards/constrained_mobo.md)
- [Hybrid agent model card](docs/model_cards/hybrid_agent.md)

## Limitations

- The simulator is not calibrated to experimental chemistry.
- Oracle fronts and true saturation values exist only for retrospective virtual
  evaluation and are not laboratory ground truth.
- Independent Stage 2 outcome GPs do not exactly preserve the algebraic relationship
  between GA, NGA, and selectivity; consistency diagnostics are recorded.
- Smoke and pilot profiles verify software pipelines but are not research conclusions.
- Python 3.13/3.14, GPU execution, Ax integration, live LLM behavior, and paper-scale
  runtime have not been verified in the current environment.
- Manual plan validation is not a substitute for institutional safety review, equipment
  limits, calibration, replicates, uncertainty analysis, or qualified human approval.

## Scientific Inspiration

The high-level agent loop is inspired by Angel Yanguas-Gil, “Performance of AI agents
based on reasoning language models on ALD process optimization tasks,” *Journal of
Vacuum Science & Technology A* 44, 043410 (2026), DOI: 10.1116/6.0005313.

The implementation, equations, parameter values, scenarios, and benchmark claims in
this repository are independent educational constructs.
