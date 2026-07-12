# Long-Prompt Requirements and Scientific Audit

Date: 2026-07-12

This audit compared the repository with `long-prompt-BO.txt`, inspected the simulator,
Stage 1 and Stage 2 models, hybrid safety path, research statistics, manual handoff,
configuration files, tests, scripts, and notebook code cells, and then corrected
confirmed defects. The repository remains an educational virtual laboratory and does
not predict real HfO2/MoS2 chemistry.

## Corrected Findings

- Removed Stage 1 oracle leakage from online stopping. The optimizer-visible stopping
  rule now uses tested observations and saved posterior target estimates; true t95 is
  used only in retrospective endpoint evaluation.
- Defined `known_target.target_growth` as the absolute external growth threshold and
  aligned the two known-target scenarios with the 0.95 target.
- Corrected false-saturation evaluation so a declared recommendation below the true
  virtual threshold is classified as false saturation.
- Changed the primary Stage 1 tolerance default to a relative 10% t95 tolerance while
  preserving an optional absolute-tolerance compatibility field.
- Replaced hidden hybrid BO sub-runs with proposal-only optimizer calls. The backend is
  now the only component that executes a selected candidate.
- Made explanation-only mode reuse unchanged BO decisions and added independent
  no-window policy checks, oracle-only correctness recording, bound-change evidence
  validation, split named seeds, and cumulative token/tool metrics.
- Added an optional strict OpenAI Responses API hybrid adapter. Live calls still require
  an explicit flag and were not executed during this audit.
- Added finite-number validation, fixed-parameter safety validation, integer cycle-bound
  validation, scientifically conservative reference-point checks, and a selectivity
  posterior-consistency diagnostic.
- Added actual top-level Stage 1 and Stage 2 state resume, existing-observation
  hypervolume restoration, and populated observation IDs in checkpoints.
- Extended candidate, experiment, and manifest schemas with model, feasibility,
  uncertainty, seed, acquisition, model-settings, LLM, and token metadata while keeping
  legacy fields valid.
- Made manual plans reloadable across processes and required operator notes on imported
  measurements.
- Added an integrated `asd-agent` command tree and corrected the legacy grid benchmark
  so its `--budget` argument is honored.
- Added descriptive statistics, Wilson success intervals, vector plot exports, missing
  Stage 1 trajectory/saturation/failure outputs, and documented the independent
  selectivity-GP limitation.
- Corrected Notebook 08 source-checkout imports and rewrote the README as software
  documentation.

## Verification

- Python: 3.12.13.
- `ruff format --check .`: passed.
- `ruff check .`: passed.
- `mypy src`: passed for 34 source files.
- `pytest`: 107 tests passed with two documented numerical/performance warnings.
- Legacy deterministic demo: passed.
- Reduced matched-budget legacy benchmark: passed.
- Stage 1 smoke: passed and emitted CSV plus PNG/SVG/PDF artifacts.
- Stage 2 smoke: 12 runs passed and emitted source data plus PNG/SVG/PDF artifacts.
- Hybrid fake-LLM smoke: passed without API calls.
- Research smoke: 16 paired rows plus JSON, CSV, Markdown, and LaTeX outputs.
- Manual export/ingestion smoke: passed.
- Notebooks 05-08: all code cells executed in small CPU-only mode.

## Remaining Limitations

- Python 3.13 and 3.14 are declared compatibility targets but were not verified here.
- Live OpenAI behavior, token accounting against a real response, and API failure modes
  were not exercised.
- Paper-scale profiles were not run, and no statistical-power claim is made.
- Ax, qParEGO, and every proposed ablation combination are not implemented as runnable
  comparison methods.
- The dedicated Stage 3 publication-figure suite described in the long prompt is not a
  standalone generator; hybrid event and metric data are persisted for future plots.
- The direct selectivity GP can be inconsistent with GA/NGA posterior samples. A
  diagnostic is recorded, but a sample-derived selectivity constraint remains future
  work.
- Physics-informed mean parameters are optimized point estimates, not a fully Bayesian
  posterior.
- The original live-LLM Notebook 03 was structurally checked but not executed because
  no live API call was authorized.
- Manual laboratory files and validation rules have not been tested with a real
  instrument, operator workflow, safety system, or experimental dataset.
