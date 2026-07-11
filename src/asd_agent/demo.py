"""Command-line demo entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from asd_agent.agent import LLMOptimizationAgent
from asd_agent.config import load_scenario
from asd_agent.experiment_loop import OptimizationAgent, run_agent_loop
from asd_agent.heuristic_agent import RuleBasedAgent
from asd_agent.models import OptimizationRun, ProcessConfig
from asd_agent.plotting import plot_selectivity_heatmap, plot_thickness_curves, plot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ASD virtual-lab optimization demo.")
    parser.add_argument("--scenario", default="inherent_selectivity")
    parser.add_argument("--agent", choices=["rule_based", "llm"], default="rule_based")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="results/demo")
    args = parser.parse_args()

    config = load_scenario(args.scenario)
    seed = config.seed if args.seed is None else args.seed
    model = None
    agent: OptimizationAgent
    if args.agent == "llm":
        model = os.environ.get("OPENAI_MODEL")
        agent = LLMOptimizationAgent(model=model)
    else:
        agent = RuleBasedAgent()

    run_dir = Path(args.output_dir) / args.scenario / args.agent
    run = run_agent_loop(
        config=config,
        agent=agent,
        budget=args.budget,
        seed=seed,
        run_dir=run_dir,
        method=args.agent,
        model=model,
    )
    make_demo_plots(config, run, run_dir)
    print(json.dumps(run.summary_row(), indent=2))
    print(f"Ledger written to {run_dir}")


def make_demo_plots(config: ProcessConfig, run: OptimizationRun, run_dir: Path) -> None:
    try:
        plot_thickness_curves(config, run_dir / "thickness_curves.png")
        plot_selectivity_heatmap(config, run_dir / "selectivity_heatmap.png")
        plot_trajectory(run.records, run_dir / "experiment_trajectory.png")
    except ModuleNotFoundError:
        return


if __name__ == "__main__":
    main()
