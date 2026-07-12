"""Run paired BO research profiles and generate statistical tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "bo09_research")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the profile bootstrap count for quick local checks.",
    )
    return parser.parse_args()


def main() -> None:
    from asd_agent.bo.research import (
        load_research_profile,
        run_research_study,
        save_research_rows,
    )
    from asd_agent.bo.statistics import save_research_analysis

    args = parse_args()
    profile = load_research_profile(args.profile)
    rows = run_research_study(profile)
    results_json, results_csv = save_research_rows(profile, rows, args.output_dir)
    analysis_paths = save_research_analysis(
        rows,
        args.output_dir / "analysis",
        bootstrap_iterations=args.bootstrap_iterations or profile.bootstrap_iterations,
        seed=profile.seed_base,
    )
    print(
        json.dumps(
            {
                "profile": profile.profile_id,
                "rows": len(rows),
                "results": [str(results_json), str(results_csv)],
                "analysis": {key: str(path) for key, path in analysis_paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
