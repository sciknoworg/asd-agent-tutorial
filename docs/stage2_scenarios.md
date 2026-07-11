# Stage 2 ASD Scenarios

These scenarios are educational virtual-laboratory problems for constrained
multi-objective optimization. They do not predict real HfO2/MoS2 chemistry.

All Stage 2 scenarios expose only three decision variables to optimizers:

- precursor dose time;
- temperature;
- integer cycle count.

Coreactant dose and inhibitor dose are fixed per scenario. Hidden process parameters
remain in the simulator configuration and are used only by the virtual lab and
evaluation oracle.

## Scenario Summary

| Scenario | Scientific Interpretation | Hidden Process Parameters | Noise | Feasibility Thresholds | Feasible Window | Difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| `bo_stage2_inherent_selectivity` | GA accumulates film before delayed NGA growth begins. | NGA has a much longer nucleation delay than GA. | Gaussian, sigma 0.02 nm. | GA >= 5.0 nm; NGA <= 0.5 nm; selectivity >= 0.80. | Yes | easy |
| `bo_stage2_inhibitor_selectivity` | A fixed inhibitor dose creates differential blocking. | NGA has much stronger inhibitor sensitivity than GA. | Gaussian, sigma 0.02 nm. | GA >= 5.0 nm; NGA <= 0.5 nm; selectivity >= 0.80. | Yes | moderate |
| `bo_stage2_impossible_selectivity` | Similar surfaces prevent selective ASD under tested variables. | GA and NGA have nearly matched kinetics and temperature responses. | Gaussian, sigma 0.02 nm. | GA >= 5.0 nm; NGA <= 0.5 nm; selectivity >= 0.80. | No | impossible |
| `bo_stage2_narrow_selective_window` | Useful growth exists only shortly before NGA nucleation. | NGA nucleation delay is close to the GA threshold cycle count. | Gaussian, sigma 0.015 nm. | GA >= 5.0 nm; NGA <= 0.30 nm; selectivity >= 0.88. | Yes | hard |
| `bo_stage2_noisy_measurements` | A feasible inherent-selectivity window is obscured by noise. | Same qualitative process as inherent selectivity, with larger measurement noise. | Gaussian, sigma 0.08 nm. | GA >= 5.0 nm; NGA <= 0.5 nm; selectivity >= 0.80. | Yes | moderate |
| `bo_stage2_boundary_optimum` | The best feasible region lies near the high-temperature hard bound. | GA temperature optimum is near the upper bound while NGA optimum is lower. | Gaussian, sigma 0.02 nm. | GA >= 4.5 nm; NGA <= 0.40 nm; selectivity >= 0.82. | Yes | hard |
| `bo_stage2_soft_selectivity_breakdown` | Selectivity degrades smoothly under aggressive conditions. | NGA has a higher temperature optimum and moderate nucleation delay. | Gaussian, sigma 0.02 nm. | GA >= 4.0 nm; NGA <= 0.60 nm; selectivity >= 0.75. | Yes | moderate |
| `bo_stage2_model_misspecification` | Offset optima stress surrogates that assume simple monotonic behavior. | GA and NGA have separated temperature optima and different precursor time constants. | Gaussian, sigma 0.03 nm. | GA >= 4.2 nm; NGA <= 0.55 nm; selectivity >= 0.78. | Yes | hard |

## Visible Bounds

Visible bounds are encoded in each `bo_stage2_*.yaml` file under `hard_bounds`.
They include precursor dose, temperature, integer cycle count, and a maximum process
time when applicable.

## Oracle Boundary

The Stage 2 oracle performs dense mixed-variable enumeration to produce a feasible
region map, approximate Pareto front, approximate hypervolume, and feasibility label.
These values are evaluation-only and are not included in `Stage2Config.optimizer_view()`.
