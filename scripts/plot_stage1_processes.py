from __future__ import annotations

import argparse
from pathlib import Path

from asd_agent.bo.oracle import Stage1EvaluationOracle
from asd_agent.config import load_stage1_scenario

DEFAULT_SCENARIOS = [
    "fast_mono",
    "slow_mono",
    "soft_biexponential",
    "noisy",
    "weak_nonselflimited",
    "misspecified",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BO Stage 1 process families.")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", default="results/bo02_stage1_process_families.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    for scenario in args.scenarios:
        config = load_stage1_scenario(scenario)
        report = Stage1EvaluationOracle(config).evaluate(curve_points=240)
        axis.plot(
            [point.dose_s for point in report.dense_curve],
            [point.true_growth for point in report.dense_curve],
            label=scenario,
        )

    axis.set_xlabel("Precursor dose time (s)")
    axis.set_ylabel("Growth response (arb. units)")
    axis.set_title("Stage 1 saturation-process families")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
