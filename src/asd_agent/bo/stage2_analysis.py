"""Analysis exports for Stage 2 constrained ASD benchmark results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asd_agent.bo.stage2 import Stage2Config, objective_dominates, stage2_objective_vector
from asd_agent.bo.stage2_benchmark import (
    Stage2BenchmarkResult,
    profile_configs,
    stage2_observation_rows,
    write_csv,
)
from asd_agent.bo.stage2_mobo import Stage2Observation
from asd_agent.bo.stage2_oracle import Stage2EvaluationOracle


def generate_stage2_analysis_outputs(
    results: Sequence[Stage2BenchmarkResult],
    output_dir: str | Path,
    *,
    configs: dict[str, Stage2Config] | None = None,
) -> list[Path]:
    """Generate BO-07 CSV and figure outputs for a Stage 2 benchmark."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_configs = configs or infer_configs_from_results(results)
    written: list[Path] = []
    written.extend(write_search_trajectories(results, resolved_configs, destination))
    written.extend(write_pareto_fronts(results, resolved_configs, destination))
    written.extend(write_hypervolume_by_iteration(results, destination))
    written.extend(write_metric_bar(results, "hypervolume_auc", "Hypervolume AUC", destination))
    written.extend(
        write_metric_bar(
            results,
            "hypervolume_regret",
            "Final Regret",
            destination,
            stem="final_regret",
        )
    )
    written.extend(
        write_metric_bar(
            results,
            "experiments_to_first_feasible",
            "Experiments To First Feasible",
            destination,
        )
    )
    written.extend(
        write_metric_bar(
            results,
            "constraint_violation_count",
            "Constraint-Violation Counts",
            destination,
        )
    )
    written.extend(write_robustness_by_scenario(results, destination))
    written.extend(write_failure_taxonomy(results, destination))
    return written


def infer_configs_from_results(
    results: Sequence[Stage2BenchmarkResult],
) -> dict[str, Stage2Config]:
    """Load configs for scenario IDs represented in benchmark results."""

    from asd_agent.config import load_stage2_scenario

    configs: dict[str, Stage2Config] = {}
    for result in results:
        if result.scenario_id in configs:
            continue
        configs[result.scenario_id] = load_stage2_scenario(result.scenario_id)
    return configs


def write_search_trajectories(
    results: Sequence[Stage2BenchmarkResult],
    configs: dict[str, Stage2Config],
    output_dir: Path,
) -> list[Path]:
    """Write search trajectory scatter plot and source data."""

    rows = stage2_observation_rows(results, configs)
    csv_path = output_dir / "search_trajectories.csv"
    png_path = output_dir / "search_trajectories.png"
    write_csv(rows, csv_path)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    for feasible, marker, label in [(True, "o", "feasible"), (False, "x", "infeasible")]:
        subset = [row for row in rows if bool(row["feasible"]) is feasible]
        if not subset:
            continue
        ax.scatter(
            [as_float(row["precursor_dose_s"]) for row in subset],
            [as_float(row["temperature_c"]) for row in subset],
            marker=marker,
            alpha=0.75,
            label=label,
        )
    ax.set_xlabel("Precursor dose (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Search Trajectories")
    ax.legend(loc="best")
    fig.tight_layout()
    figure_paths = save_figure(fig, png_path)
    plt.close(fig)
    return [csv_path, *figure_paths]


def write_pareto_fronts(
    results: Sequence[Stage2BenchmarkResult],
    configs: dict[str, Stage2Config],
    output_dir: Path,
) -> list[Path]:
    """Write observed and oracle Pareto-front source data and plot."""

    rows = pareto_front_rows(results, configs)
    csv_path = output_dir / "pareto_fronts.csv"
    png_path = output_dir / "pareto_fronts.png"
    write_csv(rows, csv_path)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    for source, marker in [("oracle", "x"), ("observed", "o")]:
        subset = [row for row in rows if row["front_source"] == source]
        if not subset:
            continue
        ax.scatter(
            [as_float(row["nga_thickness_nm"]) for row in subset],
            [as_float(row["ga_thickness_nm"]) for row in subset],
            marker=marker,
            alpha=0.75,
            label=source,
        )
    ax.set_xlabel("NGA thickness (nm)")
    ax.set_ylabel("GA thickness (nm)")
    ax.set_title("Observed and Oracle Pareto Fronts")
    ax.legend(loc="best")
    fig.tight_layout()
    figure_paths = save_figure(fig, png_path)
    plt.close(fig)
    return [csv_path, *figure_paths]


def pareto_front_rows(
    results: Sequence[Stage2BenchmarkResult],
    configs: dict[str, Stage2Config],
) -> list[dict[str, object]]:
    """Return observed and oracle Pareto-front rows."""

    rows: list[dict[str, object]] = []
    for result in results:
        estimated = observed_pareto_observations(result)
        for observation in estimated:
            rows.append(
                {
                    "front_source": "observed",
                    "method": result.method,
                    "scenario_id": result.scenario_id,
                    "repetition": result.repetition,
                    "precursor_dose_s": observation.decision.precursor_dose_s,
                    "temperature_c": observation.decision.temperature_c,
                    "cycle_count": observation.decision.cycle_count,
                    "ga_thickness_nm": observation.outcomes.ga_thickness_nm,
                    "nga_thickness_nm": observation.outcomes.nga_thickness_nm,
                    "selectivity": observation.outcomes.selectivity,
                    "process_time_s": observation.outcomes.process_time_s,
                    "feasible": observation.constraint_evaluation.feasible,
                }
            )
    for scenario_id, config in configs.items():
        report = Stage2EvaluationOracle(config).evaluate()
        for point in report.pareto_front:
            rows.append(
                {
                    "front_source": "oracle",
                    "method": "oracle",
                    "scenario_id": scenario_id,
                    "repetition": "",
                    "precursor_dose_s": point.decision.precursor_dose_s,
                    "temperature_c": point.decision.temperature_c,
                    "cycle_count": point.decision.cycle_count,
                    "ga_thickness_nm": point.outcomes.ga_thickness_nm,
                    "nga_thickness_nm": point.outcomes.nga_thickness_nm,
                    "selectivity": point.outcomes.selectivity,
                    "process_time_s": point.outcomes.process_time_s,
                    "feasible": point.constraint_evaluation.feasible,
                }
            )
    return rows


def observed_pareto_observations(
    result: Stage2BenchmarkResult,
) -> list[Stage2Observation]:
    """Return non-dominated observed rows for one benchmark result."""

    source = [obs for obs in result.observations if obs.constraint_evaluation.feasible]
    if not source:
        source = list(result.observations)
    vectors = [stage2_objective_vector(obs.outcomes) for obs in source]
    front: list[Stage2Observation] = []
    for index, observation in enumerate(source):
        vector = vectors[index]
        dominated = any(
            objective_dominates(other_vector, vector)
            for other_index, other_vector in enumerate(vectors)
            if other_index != index
        )
        if not dominated:
            front.append(observation)
    return front


def write_hypervolume_by_iteration(
    results: Sequence[Stage2BenchmarkResult],
    output_dir: Path,
) -> list[Path]:
    """Write hypervolume trajectory plot and source data."""

    rows: list[dict[str, object]] = []
    for result in results:
        for index, value in enumerate(result.hypervolume_by_iteration, start=1):
            rows.append(
                {
                    "method": result.method,
                    "scenario_id": result.scenario_id,
                    "repetition": result.repetition,
                    "iteration": index,
                    "hypervolume": value,
                }
            )
    csv_path = output_dir / "hypervolume_by_iteration.csv"
    png_path = output_dir / "hypervolume_by_iteration.png"
    write_csv(rows, csv_path)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in sorted({line_key(row) for row in rows}):
        subset = [row for row in rows if line_key(row) == key]
        ax.plot(
            [as_float(row["iteration"]) for row in subset],
            [as_float(row["hypervolume"]) for row in subset],
            marker="o",
            linewidth=1.2,
            label=key,
        )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Feasible hypervolume")
    ax.set_title("Hypervolume By Iteration")
    if rows:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    figure_paths = save_figure(fig, png_path)
    plt.close(fig)
    return [csv_path, *figure_paths]


def write_metric_bar(
    results: Sequence[Stage2BenchmarkResult],
    metric: str,
    title: str,
    output_dir: Path,
    *,
    stem: str | None = None,
) -> list[Path]:
    """Write one run-level metric bar plot and source data."""

    rows = aggregate_metric_rows(results, metric)
    file_stem = stem or metric
    csv_path = output_dir / f"{file_stem}.csv"
    png_path = output_dir / f"{file_stem}.png"
    write_csv(rows, csv_path)
    labels = [f"{row['scenario_id']}\n{row['method']}" for row in rows]
    values = [as_float(row["mean"]) for row in rows]
    return [csv_path, *plot_bar(labels, values, title, metric, png_path)]


def aggregate_metric_rows(
    results: Sequence[Stage2BenchmarkResult],
    metric: str,
) -> list[dict[str, object]]:
    """Aggregate a run-level metric by scenario and method."""

    groups: dict[tuple[str, str], list[float]] = {}
    for result in results:
        value = result.summary_row()[metric]
        if value == "":
            continue
        groups.setdefault((result.scenario_id, result.method), []).append(as_float(value))
    return [
        {
            "scenario_id": scenario_id,
            "method": method,
            "mean": sum(values) / len(values),
            "n": len(values),
        }
        for (scenario_id, method), values in sorted(groups.items())
        if values
    ]


def write_robustness_by_scenario(
    results: Sequence[Stage2BenchmarkResult],
    output_dir: Path,
) -> list[Path]:
    """Write robustness summary and plot."""

    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[Stage2BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.scenario_id, result.method), []).append(result)
    for (scenario_id, method), bucket in sorted(groups.items()):
        rows.append(
            {
                "scenario_id": scenario_id,
                "method": method,
                "runs": len(bucket),
                "success_rate": sum(1 for item in bucket if item.status == "success") / len(bucket),
                "mean_hypervolume_auc": sum(item.hypervolume_auc for item in bucket) / len(bucket),
                "mean_final_regret": sum(item.hypervolume_regret for item in bucket) / len(bucket),
            }
        )
    csv_path = output_dir / "robustness_by_scenario.csv"
    png_path = output_dir / "robustness_by_scenario.png"
    write_csv(rows, csv_path)
    labels = [f"{row['scenario_id']}\n{row['method']}" for row in rows]
    values = [as_float(row["success_rate"]) for row in rows]
    return [
        csv_path,
        *plot_bar(labels, values, "Robustness By Scenario", "success rate", png_path),
    ]


def write_failure_taxonomy(
    results: Sequence[Stage2BenchmarkResult],
    output_dir: Path,
) -> list[Path]:
    """Write failure taxonomy counts and plot."""

    counts: dict[tuple[str, str], int] = {}
    for result in results:
        counts[(result.method, result.failure_category)] = (
            counts.get((result.method, result.failure_category), 0) + 1
        )
    rows = [
        {"method": method, "failure_category": category, "count": count}
        for (method, category), count in sorted(counts.items())
    ]
    csv_path = output_dir / "failure_taxonomy.csv"
    png_path = output_dir / "failure_taxonomy.png"
    write_csv(rows, csv_path)
    labels = [f"{row['method']}\n{row['failure_category']}" for row in rows]
    values = [as_float(row["count"]) for row in rows]
    return [csv_path, *plot_bar(labels, values, "Failure Taxonomy", "count", png_path)]


def plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    path: Path,
) -> list[Path]:
    """Write a compact bar chart."""

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4.5))
    ax.bar(list(range(len(values))), values)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    paths = save_figure(fig, path)
    plt.close(fig)
    return paths


def save_figure(fig: Any, png_path: Path) -> list[Path]:
    """Save raster and vector variants of a publication figure."""

    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    return [png_path, svg_path, pdf_path]


def line_key(row: dict[str, object]) -> str:
    """Return a compact line label."""

    return f"{row['scenario_id']} | {row['method']} | r{row['repetition']}"


def as_float(value: object) -> float:
    """Convert a scalar CSV value to float."""

    if isinstance(value, str | int | float | bool):
        if value == "":
            return 0.0
        return float(value)
    raise TypeError(f"cannot convert {value!r} to float")


def pyplot() -> Any:
    """Import pyplot lazily with a noninteractive backend."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


__all__ = [
    "generate_stage2_analysis_outputs",
    "infer_configs_from_results",
    "pareto_front_rows",
    "profile_configs",
]
