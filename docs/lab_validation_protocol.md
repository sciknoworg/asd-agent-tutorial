# Laboratory Validation Protocol

This document describes a human-operated validation workflow for the ASD tutorial.
It is not autonomous reactor-control software, and it does not certify that the toy
model predicts real HfO2/MoS2 chemistry.

## Scope

The manual handoff workflow is intended to:

- receive validated optimizer candidates;
- export operator-readable lab plans;
- track planned experiments as pending;
- import completed measurements;
- validate units, required fields, and quality-control status;
- convert accepted measurements into optimizer-visible observations;
- allow Bayesian optimization to continue from the updated ledger.

The workflow does not open valves, set reactor hardware, collect measurements, or
make safety decisions for a real instrument.

## Candidate Export

`ManualLabBackend` accepts Stage 2 candidates containing:

- candidate ID;
- precursor dose in seconds;
- temperature in degrees Celsius;
- integer cycle count;
- optimizer name.

The backend validates candidates against immutable Stage 2 hard bounds before
creating a pending plan. Exported plans include experiment ID, run ID, candidate ID,
fixed settings, requested measurements, safety notes, status, operator notes, and
timestamps.

## Operator Review

Before execution, a human operator should review:

- local laboratory standard operating procedures;
- actual precursor, coreactant, substrate, carrier-gas, purge, and exhaust safety;
- tool-specific temperature, dose, pressure, and cycle-count limits;
- material compatibility and sample handling;
- whether the requested condition is appropriate for the real tool.

The tutorial-exported safety notes are reminders, not a replacement for laboratory
safety review.

## Measurement Import

Completed measurement rows must include:

- experiment ID;
- GA thickness;
- NGA thickness;
- units;
- uncertainty when available;
- measurement method;
- replicate identifier;
- quality-control status;
- operator notes.

Thickness units must be nanometers. Rows marked with failed quality control are not
added to the optimizer ledger. Accepted rows are converted into Stage 2 observations,
with selectivity and configured constraints evaluated from the imported values.

## Reproducibility

Every plan and measurement file should be archived with:

- exported CSV and JSON plan;
- completed measurement CSV or JSON;
- scenario YAML hash or copy;
- software commit hash;
- run ID;
- operator and instrument notes;
- measurement-method metadata.

The repository `results/` directory is ignored by git, so laboratory artifacts should
be copied into an institution-approved record system when they become real lab data.

## Minimum Readiness Before Real Validation

Before using this framework around real laboratory work, the team should complete:

- independent safety review by qualified lab staff;
- unit and metadata mapping to the local instrument;
- measurement-method uncertainty protocol;
- replicate and randomization plan;
- dry run with mock measurements;
- access-control and data-retention plan;
- explicit sign-off that no autonomous control is enabled.
