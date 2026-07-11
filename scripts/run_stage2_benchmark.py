"""Run Stage 2 benchmark profiles and generate analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asd_agent.bo.stage2_analysis import generate_stage2_analysis_outputs
from asd_agent.bo.stage2_benchmark import (
    load_stage2_benchmark_profile,
    profile_configs,
    run_stage2_benchmark,
    save_stage2_benchmark_results,
    stage2_summary_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "bo07_stage2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_stage2_benchmark_profile(args.profile)
    results = run_stage2_benchmark(profile)
    results_path, summary_path, observations_path = save_stage2_benchmark_results(
        profile,
        results,
        args.output_dir,
    )
    analysis_paths = generate_stage2_analysis_outputs(
        results,
        args.output_dir / "analysis",
        configs=profile_configs(profile),
    )
    print(
        json.dumps(
            {
                "profile": profile.profile_id,
                "runs": len(results),
                "summary": stage2_summary_rows(results),
                "artifacts": [
                    str(results_path),
                    str(summary_path),
                    str(observations_path),
                    *[str(path) for path in analysis_paths],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
