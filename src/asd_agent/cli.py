"""Integrated command-line interface for tutorial, BO, study, and lab workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the educational command tree without importing optional BO packages."""

    parser = argparse.ArgumentParser(
        prog="asd-agent",
        description=(
            "Educational virtual ASD optimization. Seeds and budgets are explicit; "
            "generated outputs are written below the selected output directory."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    stage1 = commands.add_parser("stage1", help="Run one-dimensional saturation learning.")
    stage1_commands = stage1.add_subparsers(dest="stage1_command", required=True)
    for name in ("run", "compare"):
        command = stage1_commands.add_parser(name)
        command.add_argument("--profile", "--study", default="smoke")
        command.add_argument("--scenario")
        command.add_argument(
            "--method",
            choices=["grid", "generic_gp", "physics_gp"],
        )
        command.add_argument("--output-dir", type=Path, default=Path("results/stage1"))

    stage2 = commands.add_parser("stage2", help="Run constrained multi-objective ASD search.")
    stage2_commands = stage2.add_subparsers(dest="stage2_command", required=True)
    for name in ("run", "compare"):
        command = stage2_commands.add_parser(name)
        command.add_argument("--profile", default="smoke")
        command.add_argument("--scenario")
        command.add_argument(
            "--method",
            choices=["random_search", "grid_search", "rule_based", "stage2_mobo"],
        )
        command.add_argument("--output-dir", type=Path, default=Path("results/stage2"))

    hybrid = commands.add_parser(
        "hybrid",
        help="Run hybrid LLM-BO orchestration; fake LLM is the default.",
    )
    hybrid_commands = hybrid.add_subparsers(dest="hybrid_command", required=True)
    hybrid_run = hybrid_commands.add_parser("run")
    hybrid_run.add_argument("--scenario", default="narrow_selective_window")
    hybrid_run.add_argument(
        "--mode",
        default="hybrid_advisory",
        choices=[
            "bo_only",
            "llm_only_legacy",
            "hybrid_advisory",
            "hybrid_intervention",
            "hybrid_explanation_only",
            "rule_based_bo",
        ],
    )
    hybrid_run.add_argument("--profile")
    hybrid_run.add_argument("--budget", type=int)
    hybrid_run.add_argument("--seed", type=int, default=8808)
    llm_mode = hybrid_run.add_mutually_exclusive_group()
    llm_mode.add_argument("--fake-llm", action="store_true")
    llm_mode.add_argument(
        "--live-llm",
        action="store_true",
        help="Make live OpenAI Responses API calls using OPENAI_MODEL and environment credentials.",
    )
    hybrid_run.add_argument("--output-dir", type=Path, default=Path("results/hybrid"))

    study = commands.add_parser("study", help="Run or analyze paired research profiles.")
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_run = study_commands.add_parser("run")
    study_run.add_argument("--config", "--profile", default="smoke")
    study_run.add_argument("--output-dir", type=Path, default=Path("results/study"))
    study_analyze = study_commands.add_parser("analyze")
    study_analyze.add_argument("--input", type=Path, required=True)
    study_analyze.add_argument("--output", type=Path, required=True)
    study_analyze.add_argument("--bootstrap-iterations", type=int, default=1000)

    lab = commands.add_parser(
        "lab", help="Export and ingest human-operated lab plans; no reactor control."
    )
    lab_commands = lab.add_subparsers(dest="lab_command", required=True)
    lab_export = lab_commands.add_parser("export")
    lab_export.add_argument("--scenario", default="inherent_selectivity")
    lab_export.add_argument("--run-id", required=True)
    lab_export.add_argument("--seed", type=int, default=1010)
    lab_export.add_argument("--output", type=Path, required=True)
    lab_ingest = lab_commands.add_parser("ingest")
    lab_ingest.add_argument("--scenario", required=True)
    lab_ingest.add_argument("--plan", type=Path, required=True)
    lab_ingest.add_argument("--measurements", type=Path, required=True)
    lab_ingest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch one integrated CLI command."""

    args = build_parser().parse_args(argv)
    if args.command == "stage1":
        run_stage1(args)
    elif args.command == "stage2":
        run_stage2(args)
    elif args.command == "hybrid":
        run_hybrid(args)
    elif args.command == "study":
        run_study(args)
    else:
        run_lab(args)


def run_stage1(args: argparse.Namespace) -> None:
    from asd_agent.bo.study import load_stage1_study_profile, run_stage1_study, save_stage1_results

    profile = load_stage1_study_profile(args.profile)
    updates: dict[str, object] = {}
    if args.scenario:
        updates["scenarios"] = [args.scenario]
    if args.method:
        updates["methods"] = [args.method]
    resolved = profile.model_copy(update=updates)
    results = run_stage1_study(resolved)
    paths = save_stage1_results(resolved, results, args.output_dir)
    print_paths(paths)


def run_stage2(args: argparse.Namespace) -> None:
    from asd_agent.bo.stage2_analysis import generate_stage2_analysis_outputs
    from asd_agent.bo.stage2_benchmark import (
        load_stage2_benchmark_profile,
        run_stage2_benchmark,
        save_stage2_benchmark_results,
    )

    profile = load_stage2_benchmark_profile(args.profile)
    updates: dict[str, object] = {}
    if args.scenario:
        updates["scenarios"] = [args.scenario]
    if args.method:
        updates["methods"] = [args.method]
    resolved = profile.model_copy(update=updates)
    results = run_stage2_benchmark(resolved)
    paths = list(save_stage2_benchmark_results(resolved, results, args.output_dir))
    paths.extend(generate_stage2_analysis_outputs(results, args.output_dir))
    print_paths(paths)


def run_hybrid(args: argparse.Namespace) -> None:
    from asd_agent.agent import LLMOptimizationAgent
    from asd_agent.bo.hybrid_agent import FakeHybridLLM, ResponsesHybridLLM, run_hybrid_optimization
    from asd_agent.bo.research import load_research_profile
    from asd_agent.config import load_stage2_scenario

    llm = ResponsesHybridLLM() if args.live_llm else FakeHybridLLM()
    legacy_agent = (
        LLMOptimizationAgent() if args.live_llm and args.mode == "llm_only_legacy" else None
    )
    budget = args.budget or 4
    bo_settings = None
    if args.profile:
        profile = load_research_profile(args.profile)
        budget = args.budget or profile.hybrid_budget
        bo_settings = profile.bo_settings(budget)
    result = run_hybrid_optimization(
        load_stage2_scenario(args.scenario),
        mode=args.mode,
        llm=llm,
        legacy_agent=legacy_agent,
        bo_settings=bo_settings,
        seed=args.seed,
        budget=budget,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "hybrid_result.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print_paths([path])


def run_study(args: argparse.Namespace) -> None:
    from asd_agent.bo.research import (
        ResearchResultRow,
        load_research_profile,
        run_research_study,
        save_research_rows,
    )
    from asd_agent.bo.statistics import save_research_analysis

    if args.study_command == "run":
        profile = load_research_profile(args.config)
        rows = run_research_study(profile)
        result_paths = save_research_rows(profile, rows, args.output_dir)
        print_paths(result_paths)
        return
    payload: Any = json.loads(args.input.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("research input must contain a rows list")
    rows = [ResearchResultRow.model_validate(row) for row in raw_rows]
    analysis_paths = save_research_analysis(
        rows,
        args.output,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print_paths(analysis_paths.values())


def run_lab(args: argparse.Namespace) -> None:
    from asd_agent.bo.manual_lab import ManualCandidate, ManualLabBackend
    from asd_agent.bo.stage2_mobo import Stage2BOSettings, Stage2ConstrainedMOBOOptimizer
    from asd_agent.config import load_stage2_scenario

    config = load_stage2_scenario(args.scenario)
    if args.lab_command == "export":
        optimizer = Stage2ConstrainedMOBOOptimizer(
            config,
            Stage2BOSettings(candidate_cycle_values=[]),
            seed=args.seed,
        )
        proposal_result = optimizer.propose([])
        if proposal_result.proposal is None:
            raise RuntimeError("BO did not return a safe candidate for manual export")
        backend = ManualLabBackend(config, run_id=args.run_id)
        backend.receive_candidate(
            ManualCandidate(
                candidate_id=proposal_result.proposal.candidate_id,
                decision=proposal_result.proposal.decision,
                optimizer=proposal_result.proposal.optimizer,
            )
        )
        print_paths(
            [*backend.export_plan(args.output), backend.export_measurement_template(args.output)]
        )
        return
    backend = ManualLabBackend.from_plan_json(config, args.plan)
    observations = backend.import_completed_measurements(args.measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([row.model_dump(mode="json") for row in observations], indent=2),
        encoding="utf-8",
    )
    print_paths([args.output])


def print_paths(paths: Any) -> None:
    """Print generated artifacts for command-line users."""

    for path in paths:
        print(Path(path).resolve())


if __name__ == "__main__":
    main()
