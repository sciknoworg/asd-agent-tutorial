"""Run BO-04 Stage 1 study profiles and generate deterministic figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from asd_agent.bo.acquisition import dose_grid
    from asd_agent.bo.gp import fit_generic_stage1_gp
    from asd_agent.bo.oracle import Stage1EvaluationOracle
    from asd_agent.bo.physics_models import fit_physics_informed_stage1_gp
    from asd_agent.bo.stage1 import true_growth
    from asd_agent.bo.study import (
        load_stage1_study_profile,
        run_stage1_study,
        save_stage1_results,
        stage1_summary_rows,
    )
    from asd_agent.config import load_stage1_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--output-dir", default="results/bo04_stage1_smoke")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    profile = load_stage1_study_profile(args.profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_stage1_study(profile)
    save_stage1_results(profile, results, output_dir)
    rows = stage1_summary_rows(profile, results)
    write_csv(output_dir / "stage1_summary.csv", rows)

    make_bar_plot(
        plt,
        rows,
        "n_experiments",
        "Experiments to success or stop",
        output_dir / "experiments_to_success",
    )
    make_bar_plot(
        plt,
        rows,
        "absolute_t95_error_s",
        "Absolute t95 error (s)",
        output_dir / "t95_error",
    )
    make_bar_plot(
        plt,
        rows,
        "cumulative_dose_s",
        "Cumulative dose (s)",
        output_dir / "cumulative_dose",
    )
    make_scatter_plot(
        plt,
        rows,
        "estimated_t95_s",
        "true_t95_s",
        "Estimated versus true t95",
        output_dir / "estimated_vs_true_t95",
    )
    make_bar_plot(
        plt,
        rows,
        "uncertainty_coverage",
        "Uncertainty coverage",
        output_dir / "calibration",
    )
    misspec_rows = [
        row
        for row in rows
        if "misspecified" in str(row["scenario_id"]) or "soft" in str(row["scenario_id"])
    ]
    make_bar_plot(
        plt,
        misspec_rows or rows,
        "absolute_t95_error_s",
        "Model-misspecification comparison",
        output_dir / "model_misspecification",
    )

    first_config = load_stage1_scenario(profile.scenarios[0])
    posterior_rows: list[dict[str, object]] = []
    for result in results:
        if result.scenario_id != first_config.scenario_id or result.method == "grid":
            continue
        candidate_doses = dose_grid(first_config.dose_bounds_s, profile.candidate_grid_size)
        try:
            if result.method == "generic_gp":
                model = fit_generic_stage1_gp(result.records, first_config.dose_bounds_s)
            else:
                model = fit_physics_informed_stage1_gp(result.records, first_config.dose_bounds_s)
            prediction = model.posterior(candidate_doses)
        except Exception as exc:
            posterior_rows.append(
                {
                    "method": result.method,
                    "dose_s": "",
                    "posterior_mean": "",
                    "posterior_stddev": "",
                    "true_growth": "",
                    "warning": str(exc),
                }
            )
            continue
        report = Stage1EvaluationOracle(first_config).evaluate(
            curve_points=profile.candidate_grid_size
        )
        for dose, mean, stddev, curve_point in zip(
            prediction.dose_s,
            prediction.mean,
            prediction.stddev,
            report.dense_curve,
            strict=True,
        ):
            posterior_rows.append(
                {
                    "method": result.method,
                    "dose_s": dose,
                    "posterior_mean": mean,
                    "posterior_stddev": stddev,
                    "true_growth": true_growth(first_config.process, curve_point.dose_s),
                    "warning": "",
                }
            )
    make_posterior_plot(plt, posterior_rows, output_dir / "sequential_posterior")
    make_experiment_trajectory_plot(plt, results, output_dir / "experiment_trajectory")
    make_saturation_samples_plot(plt, results, output_dir / "saturation_with_samples")
    write_failure_rate_table(results, output_dir / "failure_rates.csv")

    print(f"Wrote BO-04 Stage 1 outputs to {output_dir}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_bar_plot(
    plt: object,
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    output_base: Path,
) -> None:
    """Make a deterministic method/scenario bar plot."""

    plot_rows = [
        {
            "label": f"{row['method']} / {str(row['scenario_id']).replace('bo_stage1_', '')}",
            metric: row.get(metric, ""),
        }
        for row in rows
        if numeric(row.get(metric)) is not None
    ]
    write_csv(output_base.with_suffix(".csv"), plot_rows)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    labels = [str(row["label"]) for row in plot_rows]
    values = [numeric(row[metric]) or 0.0 for row in plot_rows]
    axis.bar(labels, values, color="#4c78a8")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=150)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def make_scatter_plot(
    plt: object,
    rows: list[dict[str, object]],
    x_metric: str,
    y_metric: str,
    title: str,
    output_base: Path,
) -> None:
    """Make a deterministic scatter plot with source CSV."""

    plot_rows = [
        row
        for row in rows
        if numeric(row.get(x_metric)) is not None and numeric(row.get(y_metric)) is not None
    ]
    write_csv(output_base.with_suffix(".csv"), plot_rows)
    figure, axis = plt.subplots(figsize=(5, 5))
    xs = [numeric(row[x_metric]) or 0.0 for row in plot_rows]
    ys = [numeric(row[y_metric]) or 0.0 for row in plot_rows]
    axis.scatter(xs, ys, color="#59a14f")
    if xs and ys:
        lower = min(xs + ys)
        upper = max(xs + ys)
        axis.plot([lower, upper], [lower, upper], color="#555555", linewidth=1)
    axis.set_xlabel(x_metric)
    axis.set_ylabel(y_metric)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=150)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def make_posterior_plot(
    plt: object,
    rows: list[dict[str, object]],
    output_base: Path,
) -> None:
    """Plot final sequential posterior curves for GP methods."""

    write_csv(output_base.with_suffix(".csv"), rows)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for method in sorted({str(row["method"]) for row in rows if row.get("dose_s") != ""}):
        method_rows = [row for row in rows if row["method"] == method and row.get("dose_s") != ""]
        xs = [numeric(row["dose_s"]) or 0.0 for row in method_rows]
        means = [numeric(row["posterior_mean"]) or 0.0 for row in method_rows]
        stddevs = [numeric(row["posterior_stddev"]) or 0.0 for row in method_rows]
        axis.plot(xs, means, label=method)
        lower = [mean - 2.0 * stddev for mean, stddev in zip(means, stddevs, strict=True)]
        upper = [mean + 2.0 * stddev for mean, stddev in zip(means, stddevs, strict=True)]
        axis.fill_between(xs, lower, upper, alpha=0.15)
    true_rows = [row for row in rows if row.get("dose_s") != ""]
    if true_rows:
        seen: set[float] = set()
        xs: list[float] = []
        ys: list[float] = []
        for row in true_rows:
            dose = numeric(row["dose_s"])
            if dose is None or dose in seen:
                continue
            seen.add(dose)
            xs.append(dose)
            ys.append(numeric(row["true_growth"]) or 0.0)
        axis.plot(xs, ys, color="#222222", linestyle="--", label="true process")
    axis.set_xlabel("dose_s")
    axis.set_ylabel("growth")
    axis.set_title("Sequential posterior snapshot")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=150)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def make_experiment_trajectory_plot(
    plt: object,
    results: list[Any],
    output_base: Path,
) -> None:
    """Plot sequential dose choices and export the observation-level source data."""

    rows = [
        {
            "scenario_id": result.scenario_id,
            "method": result.method,
            "iteration": index,
            "experiment_id": record.experiment_id,
            "dose_s": record.dose_s,
            "observed_growth": record.observed_growth,
        }
        for result in results
        for index, record in enumerate(result.records, start=1)
    ]
    write_csv(output_base.with_suffix(".csv"), rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key in sorted({f"{row['scenario_id']} | {row['method']}" for row in rows}):
        selected = [row for row in rows if f"{row['scenario_id']} | {row['method']}" == key]
        axis.plot(
            [int(row["iteration"]) for row in selected],
            [float(row["dose_s"]) for row in selected],
            marker="o",
            linewidth=1.0,
            label=key,
        )
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Precursor dose (s)")
    axis.set_title("Stage 1 experiment trajectories")
    if rows:
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=150)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def make_saturation_samples_plot(
    plt: object,
    results: list[Any],
    output_base: Path,
) -> None:
    """Plot the first scenario's true curve and sequential sampled observations."""

    from asd_agent.bo.oracle import Stage1EvaluationOracle
    from asd_agent.config import load_stage1_scenario

    if not results:
        write_csv(output_base.with_suffix(".csv"), [])
        return
    scenario_id = str(results[0].scenario_id)
    config = load_stage1_scenario(scenario_id.removeprefix("bo_stage1_"))
    report = Stage1EvaluationOracle(config).evaluate(curve_points=200)
    rows: list[dict[str, object]] = [
        {
            "scenario_id": scenario_id,
            "method": "oracle_curve",
            "kind": "true_curve",
            "iteration": "",
            "dose_s": point.dose_s,
            "growth": point.true_growth,
        }
        for point in report.dense_curve
    ]
    selected_results = [result for result in results if result.scenario_id == scenario_id]
    rows.extend(
        {
            "scenario_id": scenario_id,
            "method": result.method,
            "kind": "observation",
            "iteration": index,
            "dose_s": record.dose_s,
            "growth": record.observed_growth,
        }
        for result in selected_results
        for index, record in enumerate(result.records, start=1)
    )
    write_csv(output_base.with_suffix(".csv"), rows)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    curve_rows = [row for row in rows if row["kind"] == "true_curve"]
    axis.plot(
        [float(row["dose_s"]) for row in curve_rows],
        [float(row["growth"]) for row in curve_rows],
        color="#222222",
        label="true virtual process",
    )
    for method in sorted({str(row["method"]) for row in rows if row["kind"] == "observation"}):
        method_rows = [
            row for row in rows if row["kind"] == "observation" and row["method"] == method
        ]
        axis.scatter(
            [float(row["dose_s"]) for row in method_rows],
            [float(row["growth"]) for row in method_rows],
            label=method,
        )
    axis.set_xlabel("Precursor dose (s)")
    axis.set_ylabel("Growth (arbitrary units)")
    axis.set_title(f"Saturation samples: {scenario_id}")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=150)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)


def write_failure_rate_table(results: list[Any], path: Path) -> None:
    """Export method/scenario failure counts and rates."""

    grouped: dict[tuple[str, str], list[Any]] = {}
    for result in results:
        grouped.setdefault((str(result.scenario_id), str(result.method)), []).append(result)
    rows = [
        {
            "scenario_id": scenario_id,
            "method": method,
            "runs": len(group),
            "failures": sum(result.status != "success" for result in group),
            "failure_rate": sum(result.status != "success" for result in group) / len(group),
        }
        for (scenario_id, method), group in sorted(grouped.items())
    ]
    write_csv(path, rows)


def numeric(value: object) -> float | None:
    """Return a finite float or None."""

    if value in {"", None}:
        return None
    try:
        number = float(value)
    except TypeError:
        return None
    except ValueError:
        return None
    if number != number:
        return None
    return number


if __name__ == "__main__":
    main()
