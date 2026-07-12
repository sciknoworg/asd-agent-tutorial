# Model Card: Physics-Informed Stage 1 GP

## Intended Use

The physics-informed GP studies active learning when a saturating mean function is
available. It is an educational residual GP around a trainable mono-exponential
mean, not a fully Bayesian posterior over physical parameters.

## Inputs

- Precursor dose in seconds.
- Observed growth response.
- Positive-constrained saturation and rate parameters.
- GP residual settings and fallback policy.

## Outputs

- Posterior mean and uncertainty.
- Trainable physical-mean parameters.
- Threshold-oriented candidate proposals.
- Fit warnings and fallback records.

## Assumptions

- A monotonic saturating trend is a useful inductive bias.
- Residual deviations can be represented by a GP.
- The virtual process is educational and simplified.

## Known Limitations

- The physical parameters are fitted point estimates.
- Strongly misspecified or non-self-limited processes can still mislead the model.
- Fallback to a generic GP is recorded when fitting fails.

## Reproducibility

Runs record seeds, settings, fit warnings, fallback use, observations, and metrics.
