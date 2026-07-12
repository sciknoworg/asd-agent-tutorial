"""Run a small constrained Stage 2 MOBO smoke optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Stage 2 scenario name. May be supplied more than once.",
    )
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--initial-design-size", type=int, default=3)
    parser.add_argument("--optimizer-seed", type=int, default=6206)
    parser.add_argument("--simulator-seed", type=int, default=5206)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "bo06_stage2_mobo_smoke",
    )
    return parser.parse_args()


def main() -> None:
    from asd_agent.bo.stage2_mobo import (
        Stage2BOSettings,
        run_stage2_bo,
        save_stage2_bo_results,
    )
    from asd_agent.config import load_stage2_scenario

    args = parse_args()
    scenarios = args.scenario or [
        "inherent_selectivity",
        "narrow_selective_window",
        "impossible_selectivity",
    ]
    settings = Stage2BOSettings(
        experiment_budget=args.budget,
        initial_design_size=args.initial_design_size,
        qmc_samples=16,
        num_restarts=1,
        raw_samples=16,
        acquisition_timeout_s=4.0,
        random_fallback_points=64,
    )
    results = [
        run_stage2_bo(
            load_stage2_scenario(scenario),
            settings,
            simulator_seed=args.simulator_seed + index,
            optimizer_seed=args.optimizer_seed + index,
        )
        for index, scenario in enumerate(scenarios)
    ]
    json_path, csv_path = save_stage2_bo_results(results, args.output_dir)
    print(f"Wrote Stage 2 MOBO smoke results to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
