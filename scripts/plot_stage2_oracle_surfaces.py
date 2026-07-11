"""Generate simulator-only Stage 2 ASD response-surface plots."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from asd_agent.bo.stage2 import (
        Stage2Decision,
        evaluate_stage2_constraints,
        oracle_stage2_outcomes,
    )
    from asd_agent.config import load_stage2_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="narrow_selective_window")
    parser.add_argument("--cycle-count", type=int, default=None)
    parser.add_argument("--output-dir", default="results/bo05_stage2_surfaces")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    config = load_stage2_scenario(args.scenario)
    cycle_count = args.cycle_count
    if cycle_count is None:
        cycle_count = int(
            round((config.hard_bounds.cycle_count.min + config.hard_bounds.cycle_count.max) / 2)
        )
    precursor_values = linspace(
        config.hard_bounds.precursor_dose_s.min, config.hard_bounds.precursor_dose_s.max, 41
    )
    temperature_values = linspace(
        config.hard_bounds.temperature_c.min, config.hard_bounds.temperature_c.max, 41
    )
    rows: list[dict[str, object]] = []
    for temperature in temperature_values:
        for precursor in precursor_values:
            decision = Stage2Decision(
                precursor_dose_s=precursor,
                temperature_c=temperature,
                cycle_count=cycle_count,
            )
            outcomes = oracle_stage2_outcomes(config, decision)
            constraints = evaluate_stage2_constraints(config, decision, outcomes)
            rows.append(
                {
                    "precursor_dose_s": precursor,
                    "temperature_c": temperature,
                    "cycle_count": cycle_count,
                    "ga_thickness_nm": outcomes.ga_thickness_nm,
                    "nga_thickness_nm": outcomes.nga_thickness_nm,
                    "selectivity": outcomes.selectivity,
                    "process_time_s": outcomes.process_time_s,
                    "feasible": constraints.feasible,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stage2_surface_source.csv", rows)
    plot_metric(
        plt,
        rows,
        precursor_values,
        temperature_values,
        "ga_thickness_nm",
        output_dir / "ga_response_surface.png",
    )
    plot_metric(
        plt,
        rows,
        precursor_values,
        temperature_values,
        "nga_thickness_nm",
        output_dir / "nga_response_surface.png",
    )
    plot_metric(
        plt,
        rows,
        precursor_values,
        temperature_values,
        "selectivity",
        output_dir / "selectivity_surface.png",
    )
    plot_metric(
        plt,
        rows,
        precursor_values,
        temperature_values,
        "feasible",
        output_dir / "feasible_region_map.png",
    )
    print(f"Wrote Stage 2 simulator-only plots to {output_dir}")


def linspace(lower: float, upper: float, n_points: int) -> list[float]:
    """Return a closed deterministic grid."""

    step = (upper - lower) / float(n_points - 1)
    return [lower + index * step for index in range(n_points)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write plot source rows."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    plt: object,
    rows: list[dict[str, object]],
    precursor_values: list[float],
    temperature_values: list[float],
    metric: str,
    path: Path,
) -> None:
    """Plot one metric as a precursor-temperature surface."""

    value_by_key = {
        (float(row["temperature_c"]), float(row["precursor_dose_s"])): float(row[metric])
        if metric != "feasible"
        else float(bool(row[metric]))
        for row in rows
    }
    matrix = [
        [value_by_key[(temperature, precursor)] for precursor in precursor_values]
        for temperature in temperature_values
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[
            min(precursor_values),
            max(precursor_values),
            min(temperature_values),
            max(temperature_values),
        ],
    )
    axis.set_xlabel("precursor dose (s)")
    axis.set_ylabel("temperature (C)")
    axis.set_title(metric.replace("_", " "))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
