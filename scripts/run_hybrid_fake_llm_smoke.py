"""Run a no-API hybrid LLM-BO smoke check with the deterministic fake LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="inherent_selectivity")
    parser.add_argument(
        "--mode",
        choices=["bo_only", "hybrid_advisory", "hybrid_intervention", "hybrid_explanation_only"],
        default="hybrid_intervention",
    )
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8108)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "bo10_hybrid_fake_llm_smoke",
    )
    return parser.parse_args()


def main() -> None:
    from asd_agent.bo.hybrid_agent import FakeHybridLLM, run_hybrid_optimization
    from asd_agent.bo.stage2_mobo import Stage2BOSettings
    from asd_agent.config import load_stage2_scenario

    args = parse_args()
    config = load_stage2_scenario(args.scenario)
    settings = Stage2BOSettings(
        experiment_budget=args.budget,
        initial_design_size=2,
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
        acquisition_timeout_s=2.0,
        candidate_cycle_values=[30, 50, 70],
        random_fallback_points=32,
    )
    result = run_hybrid_optimization(
        config,
        mode=args.mode,
        llm=FakeHybridLLM("intervention"),
        bo_settings=settings,
        seed=args.seed,
        budget=args.budget,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "hybrid_fake_llm_result.json"
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scenario": config.scenario_id,
                "mode": result.mode,
                "status": result.status,
                "observations": len(result.observations),
                "candidates": len(result.candidates),
                "events": len(result.events),
                "result": str(result_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
