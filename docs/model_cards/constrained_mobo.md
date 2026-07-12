# Model Card: Constrained Stage 2 MOBO

## Intended Use

The constrained MOBO method optimizes the Stage 2 virtual ASD problem with GA growth,
NGA suppression, and process time as separate objectives under feasibility
constraints. It is a tutorial benchmark method, not a reactor-control policy.

## Inputs

- Precursor dose.
- Temperature.
- Integer cycle count.
- Measured GA thickness, NGA thickness, selectivity, and process time.
- Scenario-specific hard bounds and feasibility thresholds.

## Outputs

- Valid nonduplicate candidate proposals.
- Feasibility probabilities and posterior summaries.
- Feasible hypervolume trajectories.
- Recommended tested observations when feasible.

## Assumptions

- The bounded virtual search space is adequate for small CPU-only studies.
- One GP per measured outcome is transparent enough for tutorial use.
- Integer cycle count is handled by conditional enumeration.
- Selectivity is currently modeled as a separate outcome because the selected BoTorch acquisition
  API consumes explicit outcome constraints. Candidate diagnostics report selectivity recomputed
  from posterior GA/NGA means and the discrepancy from the direct selectivity GP.

## Known Limitations

- It depends on optional PyTorch, GPyTorch, and BoTorch packages.
- Small smoke profiles are not paper-scale evidence.
- Pure-Python acquisition fallback may be slower when BoTorch extensions cannot compile.
- Oracle values are for evaluation only and must not enter optimizer inputs.
- Independent outcome GPs do not preserve the exact algebraic relationship between GA, NGA, and
  selectivity in posterior samples. The consistency diagnostic reveals, but does not remove, this
  approximation.

## Reproducibility

Runs record seeds, initial designs, settings, warnings, hypervolume trajectories, and
failure categories.
