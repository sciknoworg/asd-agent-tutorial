"""Run a small BO-03 Stage 1 grid-vs-GP smoke comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from asd_agent.bo.optimizers import Stage1RunnerSettings, compare_stage1_methods
    from asd_agent.config import load_stage1_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Stage 1 scenario name without the bo_stage1_ prefix.",
    )
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--output-json", default="results/bo03_stage1_smoke.json")
    parser.add_argument("--output-csv", default="results/bo03_stage1_smoke.csv")
    args = parser.parse_args()

    scenario_names = args.scenario or ["fast_mono", "slow_mono", "noisy"]
    configs = [load_stage1_scenario(name) for name in scenario_names]
    settings = Stage1RunnerSettings(
        budget=args.budget,
        simulator_seed=5103,
        optimizer_seed=7103,
    )
    results = compare_stage1_methods(configs, methods=("grid", "generic_gp"), settings=settings)

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )
    rows = [result.summary_row() for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
