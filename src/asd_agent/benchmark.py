"""Benchmark script for baseline, rule-based, and optional LLM agents."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from asd_agent.agent import LLMOptimizationAgent
from asd_agent.baselines import grid_search, random_search
from asd_agent.config import load_scenario
from asd_agent.experiment_loop import run_agent_loop
from asd_agent.heuristic_agent import RuleBasedAgent
from asd_agent.models import OptimizationRun, ProcessConfig
from asd_agent.plotting import (
    plot_experiments_required,
    plot_llm_variability,
    plot_success_rate,
)

DEFAULT_SCENARIOS = ["inherent_selectivity", "inhibitor_selectivity", "impossible_selectivity"]
DEFAULT_METHODS = ["random_search", "grid_search", "rule_based"]


def run_benchmark(
    scenarios: list[str] | None = None,
    methods: list[str] | None = None,
    repetitions: int = 20,
    output_dir: str | Path = "results/benchmark",
    budget: int = 20,
) -> list[dict[str, object]]:
    """Run independent repetitions and return summary rows."""

    selected_scenarios = scenarios or DEFAULT_SCENARIOS
    selected_methods = methods or DEFAULT_METHODS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[OptimizationRun] = []

    for scenario in selected_scenarios:
        config = load_scenario(scenario)
        for repetition in range(repetitions):
            seed = config.seed + 1000 * repetition
            for method in selected_methods:
                run_dir = out / "runs" / scenario / method / f"rep_{repetition:03d}"
                run = run_method(config, method, budget=budget, seed=seed, run_dir=run_dir)
                runs.append(run)

    rows = [run.summary_row() for run in runs]
    write_summary(rows, out / "benchmark_summary.csv")
    (out / "benchmark_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_benchmark_plots(rows, out / "plots")
    return rows


def run_method(
    config: ProcessConfig,
    method: str,
    budget: int,
    seed: int,
    run_dir: Path,
) -> OptimizationRun:
    if method == "random_search":
        run = random_search(config, budget=budget, seed=seed)
        return _persist_baseline(run, config, run_dir)
    if method == "grid_search":
        run = grid_search(config, budget=max(budget, 81), seed=seed)
        return _persist_baseline(run, config, run_dir)
    if method == "rule_based":
        return run_agent_loop(
            config, RuleBasedAgent(), budget=budget, seed=seed, run_dir=run_dir, method=method
        )
    if method == "llm":
        model = os.environ.get("OPENAI_MODEL")
        return run_agent_loop(
            config,
            LLMOptimizationAgent(model=model),
            budget=budget,
            seed=seed,
            run_dir=run_dir,
            method=method,
            model=model,
        )
    raise ValueError(f"unknown benchmark method: {method}")


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_benchmark_plots(rows: list[dict[str, object]], plot_dir: Path) -> None:
    try:
        plot_experiments_required(rows, plot_dir / "experiments_required_per_method.png")
        plot_success_rate(rows, plot_dir / "success_rate_comparison.png")
        plot_llm_variability(rows, plot_dir / "llm_variability.png")
    except ModuleNotFoundError:
        return


def _persist_baseline(
    run: OptimizationRun,
    config: ProcessConfig,
    run_dir: Path,
) -> OptimizationRun:
    from asd_agent.experiment_loop import persist_run

    persist_run(run, config, run_dir)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASD tutorial optimizers.")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--include-llm", action="store_true")
    args = parser.parse_args()

    methods = list(args.methods)
    if args.include_llm and "llm" not in methods:
        methods.append("llm")
    rows = run_benchmark(args.scenarios, methods, args.repetitions, args.output_dir, args.budget)
    print(json.dumps(aggregate(rows), indent=2))


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row["method"])
        bucket = result.setdefault(method, {"runs": 0.0, "successes": 0.0, "mean_experiments": 0.0})
        bucket["runs"] += 1
        bucket["successes"] += 1 if row["success"] else 0
        bucket["mean_experiments"] += _as_float(row["n_experiments"])
    for bucket in result.values():
        bucket["success_rate"] = bucket["successes"] / bucket["runs"]
        bucket["mean_experiments"] = bucket["mean_experiments"] / bucket["runs"]
    return result


def _as_float(value: object) -> float:
    if isinstance(value, str | int | float | bool):
        return float(value)
    raise TypeError(f"cannot convert {value!r} to float")


if __name__ == "__main__":
    main()
