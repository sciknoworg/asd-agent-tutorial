"""Plotting utilities for tutorial notebooks and benchmark scripts."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from asd_agent.models import ExperimentCondition, ExperimentRecord, ProcessConfig
from asd_agent.simulator import VirtualLab


def plot_thickness_curves(config: ProcessConfig, output_path: str | Path) -> None:
    """Plot GA and NGA thickness versus cycle count for a representative condition."""

    plt = _matplotlib()
    lab = VirtualLab(config, seed=config.seed)
    cycles = np.arange(int(config.safety.cycles.min), int(config.safety.cycles.max) + 1)
    condition_base = {
        "precursor_dose_s": config.safety.precursor_dose_s.lerp(0.75),
        "coreactant_dose_s": config.safety.coreactant_dose_s.lerp(0.75),
        "inhibitor_dose_s": config.safety.inhibitor_dose_s.lerp(0.60),
        "temperature_c": config.safety.temperature_c.midpoint(),
    }
    ga = []
    nga = []
    for cycle_count in cycles:
        condition = ExperimentCondition(**condition_base, cycles=int(cycle_count))
        ga.append(lab.surface_thickness("GA", condition))
        nga.append(lab.surface_thickness("NGA", condition))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(cycles, ga, label="GA", linewidth=2)
    ax.plot(cycles, nga, label="NGA", linewidth=2)
    ax.axhline(config.objective.ga_min_nm, color="tab:blue", linestyle="--", linewidth=1)
    ax.axhline(config.objective.nga_max_nm, color="tab:orange", linestyle="--", linewidth=1)
    ax.set_xlabel("ALD cycles")
    ax.set_ylabel("Thickness (nm)")
    ax.set_title(f"Thickness curves: {config.scenario}")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)


def plot_selectivity_heatmap(config: ProcessConfig, output_path: str | Path) -> None:
    """Plot selectivity across precursor and coreactant dose space."""

    plt = _matplotlib()
    lab = VirtualLab(config, seed=config.seed)
    p_values = np.linspace(
        config.safety.precursor_dose_s.min, config.safety.precursor_dose_s.max, 40
    )
    c_values = np.linspace(
        config.safety.coreactant_dose_s.min, config.safety.coreactant_dose_s.max, 40
    )
    heatmap = np.zeros((len(c_values), len(p_values)))
    for row, coreactant in enumerate(c_values):
        for col, precursor in enumerate(p_values):
            record = lab.simulate(
                ExperimentCondition(
                    precursor_dose_s=float(precursor),
                    coreactant_dose_s=float(coreactant),
                    inhibitor_dose_s=config.safety.inhibitor_dose_s.lerp(0.60),
                    temperature_c=config.safety.temperature_c.midpoint(),
                    cycles=int(config.safety.cycles.max),
                ),
                experiment_id="heatmap",
            )
            heatmap[row, col] = record.selectivity

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[p_values.min(), p_values.max(), c_values.min(), c_values.max()],
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    ax.set_xlabel("Precursor dose (s)")
    ax.set_ylabel("Coreactant dose (s)")
    ax.set_title(f"Selectivity heatmap: {config.scenario}")
    fig.colorbar(image, ax=ax, label="Selectivity")
    fig.tight_layout()
    _save(fig, output_path)


def plot_trajectory(records: list[ExperimentRecord], output_path: str | Path) -> None:
    """Plot experiment trajectory through precursor/coreactant space."""

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    x = [record.condition.precursor_dose_s for record in records]
    y = [record.condition.coreactant_dose_s for record in records]
    colors = [record.selectivity for record in records]
    scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=70, edgecolor="black")
    ax.plot(x, y, color="0.5", linewidth=1)
    for index, record in enumerate(records, start=1):
        ax.annotate(
            str(index), (record.condition.precursor_dose_s, record.condition.coreactant_dose_s)
        )
    ax.set_xlabel("Precursor dose (s)")
    ax.set_ylabel("Coreactant dose (s)")
    ax.set_title("Experiment trajectory")
    fig.colorbar(scatter, ax=ax, label="Selectivity")
    fig.tight_layout()
    _save(fig, output_path)


def plot_experiments_required(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Plot average number of experiments per method."""

    plt = _matplotlib()
    grouped = _group_numeric(rows, "method", "n_experiments")
    labels = list(grouped)
    values = [sum(items) / len(items) for items in grouped.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("Experiments")
    ax.set_title("Experiments required per method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_success_rate(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Plot success-rate comparison by method."""

    plt = _matplotlib()
    grouped = _group_numeric(rows, "method", "success")
    labels = list(grouped)
    values = [sum(items) / len(items) for items in grouped.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="tab:green")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title("Success-rate comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_llm_variability(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Plot variability across repeated LLM runs when LLM rows are present."""

    plt = _matplotlib()
    llm_rows = [row for row in rows if str(row.get("method", "")).startswith("llm")]
    fig, ax = plt.subplots(figsize=(6, 4))
    if llm_rows:
        values = [_as_float(row.get("best_selectivity", 0.0)) for row in llm_rows]
        ax.hist(values, bins=min(10, max(3, len(values))), color="tab:purple", edgecolor="black")
    ax.set_xlabel("Best selectivity")
    ax.set_ylabel("Count")
    ax.set_title("Variability across repeated LLM runs")
    fig.tight_layout()
    _save(fig, output_path)


def read_benchmark_rows(path: str | Path) -> list[dict[str, object]]:
    """Read benchmark CSV rows with basic numeric coercion."""

    rows: list[dict[str, object]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: _coerce(value) for key, value in row.items()})
    return rows


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install matplotlib to generate tutorial plots.") from exc
    return plt


def _save(fig: object, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)  # type: ignore[attr-defined]
    fig.clf()  # type: ignore[attr-defined]


def _group_numeric(
    rows: Iterable[dict[str, object]],
    group_key: str,
    value_key: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(_as_float(row[value_key]))
    return grouped


def _as_float(value: object) -> float:
    if isinstance(value, str | int | float | bool):
        return float(value)
    raise TypeError(f"cannot convert {value!r} to float")


def _coerce(value: str) -> object:
    if value in {"True", "False"}:
        return value == "True"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
