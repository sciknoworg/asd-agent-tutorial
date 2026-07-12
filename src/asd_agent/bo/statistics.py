"""Statistical analysis helpers for paired BO research studies."""

from __future__ import annotations

import csv
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.research import ResearchQuestion, ResearchResultRow, StudyArea


class ComparisonSpec(BaseModel):
    """One paired statistical comparison."""

    comparison_id: str
    research_question: ResearchQuestion
    study_area: StudyArea
    control_method: str
    treatment_method: str
    metric: str
    higher_is_better: bool = True

    model_config = ConfigDict(extra="forbid")


class PairedEffectResult(BaseModel):
    """Effect estimate and paired statistical summaries for one comparison."""

    comparison_id: str
    research_question: ResearchQuestion
    study_area: StudyArea
    control_method: str
    treatment_method: str
    metric: str
    higher_is_better: bool
    n_pairs: int
    control_mean: float
    treatment_mean: float
    paired_effect: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    control_successes: int
    treatment_successes: int
    discordant_control_only: int
    discordant_treatment_only: int
    mcnemar_p_value: float
    wilcoxon_p_value: float
    holm_adjusted_wilcoxon_p_value: float

    model_config = ConfigDict(extra="forbid")


class CumulativeSuccessPoint(BaseModel):
    """Success fraction by experiment number."""

    study_area: StudyArea
    scenario_id: str
    method: str
    iteration: int
    success_rate: float
    n_runs: int

    model_config = ConfigDict(extra="forbid")


class FailureSummaryRow(BaseModel):
    """Failure-category count for one method/scenario."""

    study_area: StudyArea
    scenario_id: str
    method: str
    failure_category: str
    count: int
    total: int
    fraction: float

    model_config = ConfigDict(extra="forbid")


class DescriptiveSummaryRow(BaseModel):
    """Distribution and success summary for one method/scenario."""

    study_area: StudyArea
    scenario_id: str
    method: str
    metric: str
    n: int
    mean: float
    standard_deviation: float
    median: float
    q1: float
    q3: float
    success_rate: float
    success_ci_low: float
    success_ci_high: float

    model_config = ConfigDict(extra="forbid")


class ResearchAnalysis(BaseModel):
    """Serializable research-analysis payload."""

    comparisons: list[PairedEffectResult] = Field(default_factory=list)
    cumulative_success: list[CumulativeSuccessPoint] = Field(default_factory=list)
    failure_summary: list[FailureSummaryRow] = Field(default_factory=list)
    descriptive_summary: list[DescriptiveSummaryRow] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def default_comparison_specs(rows: Sequence[ResearchResultRow]) -> list[ComparisonSpec]:
    """Return default paired comparisons supported by the available rows."""

    methods_by_area: dict[StudyArea, set[str]] = defaultdict(set)
    for row in rows:
        methods_by_area[row.study_area].add(row.method)

    specs: list[ComparisonSpec] = []
    stage1_methods = methods_by_area.get("stage1_saturation", set())
    if {"generic_gp", "physics_gp"}.issubset(stage1_methods):
        specs.append(
            ComparisonSpec(
                comparison_id="rq1_physics_gp_vs_generic_gp_t95_error",
                research_question="RQ1",
                study_area="stage1_saturation",
                control_method="generic_gp",
                treatment_method="physics_gp",
                metric="absolute_t95_error_s",
                higher_is_better=False,
            )
        )

    stage2_methods = methods_by_area.get("stage2_asd", set())
    for baseline in ("random_search", "grid_search", "rule_based"):
        if baseline in stage2_methods and "stage2_mobo" in stage2_methods:
            specs.append(
                ComparisonSpec(
                    comparison_id=f"rq2_stage2_mobo_vs_{baseline}_hv_auc",
                    research_question="RQ2",
                    study_area="stage2_asd",
                    control_method=baseline,
                    treatment_method="stage2_mobo",
                    metric="hypervolume_auc",
                    higher_is_better=True,
                )
            )

    hybrid_methods = methods_by_area.get("hybrid_agent", set())
    if {"bo_only", "hybrid_intervention"}.issubset(hybrid_methods):
        specs.append(
            ComparisonSpec(
                comparison_id="rq3_hybrid_intervention_vs_bo_only_final_hv",
                research_question="RQ3",
                study_area="hybrid_agent",
                control_method="bo_only",
                treatment_method="hybrid_intervention",
                metric="final_hypervolume",
                higher_is_better=True,
            )
        )
    return specs


def analyze_research_results(
    rows: Sequence[ResearchResultRow],
    *,
    comparison_specs: Sequence[ComparisonSpec] | None = None,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> ResearchAnalysis:
    """Analyze paired research rows without accessing hidden simulator fields."""

    specs = (
        list(comparison_specs) if comparison_specs is not None else default_comparison_specs(rows)
    )
    comparisons = [
        paired_effect_estimate(rows, spec, bootstrap_iterations=bootstrap_iterations, seed=seed)
        for spec in specs
    ]
    comparisons = apply_holm_correction(comparisons)
    return ResearchAnalysis(
        comparisons=comparisons,
        cumulative_success=cumulative_success_curve(rows),
        failure_summary=failure_category_summary(rows),
        descriptive_summary=descriptive_summary(rows),
    )


def paired_effect_estimate(
    rows: Sequence[ResearchResultRow],
    spec: ComparisonSpec,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> PairedEffectResult:
    """Compute paired treatment-minus-control effect estimates."""

    by_pair: dict[str, dict[str, ResearchResultRow]] = defaultdict(dict)
    for row in rows:
        if row.study_area == spec.study_area:
            by_pair[row.pair_id][row.method] = row

    control_values: list[float] = []
    treatment_values: list[float] = []
    control_success: list[bool] = []
    treatment_success: list[bool] = []
    for methods in by_pair.values():
        control = methods.get(spec.control_method)
        treatment = methods.get(spec.treatment_method)
        if control is None or treatment is None:
            continue
        control_metric = metric_value(control, spec.metric)
        treatment_metric = metric_value(treatment, spec.metric)
        if not (math.isfinite(control_metric) and math.isfinite(treatment_metric)):
            continue
        control_values.append(control_metric)
        treatment_values.append(treatment_metric)
        control_success.append(control.success)
        treatment_success.append(treatment.success)

    diffs = [
        treatment - control
        for control, treatment in zip(control_values, treatment_values, strict=True)
    ]
    ci_low, ci_high = bootstrap_ci(diffs, iterations=bootstrap_iterations, seed=seed)
    control_only = sum(c and not t for c, t in zip(control_success, treatment_success, strict=True))
    treatment_only = sum(
        t and not c for c, t in zip(control_success, treatment_success, strict=True)
    )
    return PairedEffectResult(
        comparison_id=spec.comparison_id,
        research_question=spec.research_question,
        study_area=spec.study_area,
        control_method=spec.control_method,
        treatment_method=spec.treatment_method,
        metric=spec.metric,
        higher_is_better=spec.higher_is_better,
        n_pairs=len(diffs),
        control_mean=safe_mean(control_values),
        treatment_mean=safe_mean(treatment_values),
        paired_effect=safe_mean(diffs),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        control_successes=sum(control_success),
        treatment_successes=sum(treatment_success),
        discordant_control_only=control_only,
        discordant_treatment_only=treatment_only,
        mcnemar_p_value=mcnemar_p_value(control_success, treatment_success),
        wilcoxon_p_value=wilcoxon_p_value(diffs),
        holm_adjusted_wilcoxon_p_value=math.nan,
    )


def bootstrap_ci(
    paired_differences: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a paired bootstrap confidence interval for mean differences."""

    finite = np.asarray(
        [value for value in paired_differences if math.isfinite(value)], dtype=float
    )
    if finite.size == 0:
        return math.nan, math.nan
    if finite.size == 1:
        value = float(finite[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(iterations, finite.size), replace=True)
    means = draws.mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def mcnemar_p_value(control_success: Sequence[bool], treatment_success: Sequence[bool]) -> float:
    """Return an exact McNemar p-value for paired binary success."""

    control_only = sum(c and not t for c, t in zip(control_success, treatment_success, strict=True))
    treatment_only = sum(
        t and not c for c, t in zip(control_success, treatment_success, strict=True)
    )
    discordant = control_only + treatment_only
    if discordant == 0:
        return 1.0
    return exact_two_sided_binomial_p(min(control_only, treatment_only), discordant)


def wilcoxon_p_value(paired_differences: Sequence[float]) -> float:
    """Return a Wilcoxon signed-rank p-value, with a sign-test fallback."""

    finite = [value for value in paired_differences if math.isfinite(value) and value != 0.0]
    if not finite:
        return 1.0
    try:
        scipy_stats: Any = importlib.import_module("scipy.stats")
        result = scipy_stats.wilcoxon(finite, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        positive = sum(value > 0.0 for value in finite)
        negative = len(finite) - positive
        return exact_two_sided_binomial_p(min(positive, negative), len(finite))


def apply_holm_correction(results: Sequence[PairedEffectResult]) -> list[PairedEffectResult]:
    """Apply Holm correction to Wilcoxon p-values within each research question."""

    adjusted = list(results)
    by_question: dict[ResearchQuestion, list[int]] = defaultdict(list)
    for index, result in enumerate(adjusted):
        by_question[result.research_question].append(index)
    for indexes in by_question.values():
        p_values = [adjusted[index].wilcoxon_p_value for index in indexes]
        corrected = holm_adjust(p_values)
        for index, p_value in zip(indexes, corrected, strict=True):
            adjusted[index] = adjusted[index].model_copy(
                update={"holm_adjusted_wilcoxon_p_value": p_value}
            )
    return adjusted


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""

    if not p_values:
        return []
    clean = [p if math.isfinite(p) else 1.0 for p in p_values]
    order = sorted(range(len(clean)), key=lambda index: clean[index])
    adjusted = [1.0] * len(clean)
    running_max = 0.0
    total = len(clean)
    for rank, index in enumerate(order):
        value = min((total - rank) * clean[index], 1.0)
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def cumulative_success_curve(
    rows: Sequence[ResearchResultRow],
) -> list[CumulativeSuccessPoint]:
    """Return cumulative success rates by tested experiment count."""

    grouped: dict[tuple[StudyArea, str, str], list[ResearchResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.study_area, row.scenario_id, row.method)].append(row)

    points: list[CumulativeSuccessPoint] = []
    for (study_area, scenario_id, method), group in sorted(grouped.items()):
        max_iteration = max((max(row.n_experiments, 1) for row in group), default=0)
        for iteration in range(1, max_iteration + 1):
            successes = sum(success_iteration(row) <= iteration for row in group)
            points.append(
                CumulativeSuccessPoint(
                    study_area=study_area,
                    scenario_id=scenario_id,
                    method=method,
                    iteration=iteration,
                    success_rate=successes / len(group) if group else 0.0,
                    n_runs=len(group),
                )
            )
    return points


def failure_category_summary(rows: Sequence[ResearchResultRow]) -> list[FailureSummaryRow]:
    """Return counts and fractions by failure category."""

    grouped: dict[tuple[StudyArea, str, str], list[ResearchResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.study_area, row.scenario_id, row.method)].append(row)

    summaries: list[FailureSummaryRow] = []
    for (study_area, scenario_id, method), group in sorted(grouped.items()):
        total = len(group)
        counts: dict[str, int] = defaultdict(int)
        for row in group:
            counts[row.failure_category] += 1
        for failure_category, count in sorted(counts.items()):
            summaries.append(
                FailureSummaryRow(
                    study_area=study_area,
                    scenario_id=scenario_id,
                    method=method,
                    failure_category=failure_category,
                    count=count,
                    total=total,
                    fraction=count / total if total else 0.0,
                )
            )
    return summaries


def descriptive_summary(rows: Sequence[ResearchResultRow]) -> list[DescriptiveSummaryRow]:
    """Return median, IQR, mean, SD, and Wilson success intervals."""

    grouped: dict[tuple[StudyArea, str, str, str], list[ResearchResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.study_area, row.scenario_id, row.method, row.primary_metric_name)].append(row)
    summaries: list[DescriptiveSummaryRow] = []
    for (study_area, scenario_id, method, metric), group in sorted(grouped.items()):
        values = np.asarray(
            [row.primary_metric_value for row in group if math.isfinite(row.primary_metric_value)],
            dtype=float,
        )
        successes = sum(row.success for row in group)
        ci_low, ci_high = wilson_interval(successes, len(group))
        summaries.append(
            DescriptiveSummaryRow(
                study_area=study_area,
                scenario_id=scenario_id,
                method=method,
                metric=metric,
                n=int(values.size),
                mean=float(values.mean()) if values.size else math.nan,
                standard_deviation=float(values.std(ddof=1)) if values.size > 1 else 0.0,
                median=float(np.median(values)) if values.size else math.nan,
                q1=float(np.quantile(values, 0.25)) if values.size else math.nan,
                q3=float(np.quantile(values, 0.75)) if values.size else math.nan,
                success_rate=successes / len(group) if group else math.nan,
                success_ci_low=ci_low,
                success_ci_high=ci_high,
            )
        )
    return summaries


def wilson_interval(
    successes: int, total: int, z_value: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial success rate."""

    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def save_research_analysis(
    rows: Sequence[ResearchResultRow],
    output_dir: str | Path,
    *,
    comparison_specs: Sequence[ComparisonSpec] | None = None,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Path]:
    """Generate CSV, JSON, Markdown, and LaTeX analysis artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    analysis = analyze_research_results(
        rows,
        comparison_specs=comparison_specs,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    comparison_rows = [result.model_dump(mode="json") for result in analysis.comparisons]
    cumulative_rows = [point.model_dump(mode="json") for point in analysis.cumulative_success]
    failure_rows = [row.model_dump(mode="json") for row in analysis.failure_summary]
    descriptive_rows = [row.model_dump(mode="json") for row in analysis.descriptive_summary]

    paths = {
        "json": destination / "research_statistics.json",
        "comparisons_csv": destination / "paired_effects.csv",
        "cumulative_csv": destination / "cumulative_success.csv",
        "failures_csv": destination / "failure_summary.csv",
        "descriptive_csv": destination / "descriptive_summary.csv",
        "markdown": destination / "research_statistics.md",
        "latex": destination / "research_statistics.tex",
    }
    paths["json"].write_text(
        json.dumps(analysis.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    write_dict_rows(comparison_rows, paths["comparisons_csv"], paired_effect_fieldnames())
    write_dict_rows(cumulative_rows, paths["cumulative_csv"], cumulative_fieldnames())
    write_dict_rows(failure_rows, paths["failures_csv"], failure_fieldnames())
    write_dict_rows(descriptive_rows, paths["descriptive_csv"], descriptive_fieldnames())
    paths["markdown"].write_text(markdown_report(analysis), encoding="utf-8")
    paths["latex"].write_text(latex_report(analysis), encoding="utf-8")
    return paths


def metric_value(row: ResearchResultRow, metric: str) -> float:
    """Return a finite metric value from a normalized research row when possible."""

    if metric == "primary_metric_value":
        return row.primary_metric_value
    if metric == "success":
        return 1.0 if row.success else 0.0
    value = row.metrics.get(metric)
    if isinstance(value, str | int | float | bool):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def success_iteration(row: ResearchResultRow) -> int:
    """Return the iteration where success appeared, or a sentinel beyond the run."""

    if not row.success:
        return row.n_experiments + 1
    first_feasible = row.metrics.get("experiments_to_first_feasible")
    if isinstance(first_feasible, str) and first_feasible.strip() == "":
        return row.n_experiments
    if isinstance(first_feasible, str | int | float | bool):
        return max(1, int(float(first_feasible)))
    return row.n_experiments


def exact_two_sided_binomial_p(k: int, n: int) -> float:
    """Return exact two-sided binomial p-value for p=0.5."""

    tail = float(sum(math.comb(n, value) for value in range(0, k + 1))) / float(2**n)
    return float(min(1.0, 2.0 * tail))


def safe_mean(values: Sequence[float]) -> float:
    """Return the finite mean or NaN."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def write_dict_rows(rows: Sequence[dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with stable headers, even when the result is empty."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_report(analysis: ResearchAnalysis) -> str:
    """Return a compact Markdown statistical report."""

    lines = [
        "# Research Statistics",
        "",
        "These tables summarize paired tutorial benchmarks. They are hypotheses tests, not",
        "claims about real ASD chemistry.",
        "",
        "## Paired Effects",
        "",
        markdown_table(
            [result.model_dump(mode="json") for result in analysis.comparisons],
            [
                "comparison_id",
                "n_pairs",
                "control_mean",
                "treatment_mean",
                "paired_effect",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "mcnemar_p_value",
                "holm_adjusted_wilcoxon_p_value",
            ],
        ),
        "",
        "## Descriptive Statistics",
        "",
        markdown_table(
            [row.model_dump(mode="json") for row in analysis.descriptive_summary],
            [
                "scenario_id",
                "method",
                "n",
                "median",
                "q1",
                "q3",
                "mean",
                "standard_deviation",
                "success_rate",
                "success_ci_low",
                "success_ci_high",
            ],
        ),
        "",
        "## Failure Categories",
        "",
        markdown_table(
            [row.model_dump(mode="json") for row in analysis.failure_summary],
            ["study_area", "scenario_id", "method", "failure_category", "count", "fraction"],
        ),
        "",
    ]
    return "\n".join(lines)


def latex_report(analysis: ResearchAnalysis) -> str:
    """Return a compact LaTeX table report."""

    comparison_rows = [result.model_dump(mode="json") for result in analysis.comparisons]
    return "\n".join(
        [
            "% Auto-generated by asd_agent.bo.statistics.",
            "\\begin{tabular}{lrrrrr}",
            "\\hline",
            "Comparison & N & Control & Treatment & Effect & Holm p \\\\",
            "\\hline",
            *[
                (
                    f"{latex_escape(str(row['comparison_id']))} & {row['n_pairs']} & "
                    f"{format_number(row['control_mean'])} & "
                    f"{format_number(row['treatment_mean'])} & "
                    f"{format_number(row['paired_effect'])} & "
                    f"{format_number(row['holm_adjusted_wilcoxon_p_value'])} \\\\"
                )
                for row in comparison_rows
            ],
            "\\hline",
            "\\end{tabular}",
            "",
        ]
    )


def markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    """Return a simple Markdown table with stable columns."""

    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    if not rows:
        return "\n".join([header, separator])
    body = [
        "| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def format_cell(value: Any) -> str:
    """Format scalar table cells."""

    if isinstance(value, float):
        return format_number(value)
    return str(value)


def format_number(value: Any) -> str:
    """Format numeric values with NaN protection."""

    try:
        numeric_value = float(value)
    except TypeError:
        return ""
    except ValueError:
        return ""
    if not math.isfinite(numeric_value):
        return ""
    return f"{numeric_value:.4g}"


def latex_escape(value: str) -> str:
    """Escape the small subset of characters used in generated table labels."""

    return value.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def paired_effect_fieldnames() -> list[str]:
    """Return stable paired-effect CSV headers."""

    return [
        "comparison_id",
        "research_question",
        "study_area",
        "control_method",
        "treatment_method",
        "metric",
        "higher_is_better",
        "n_pairs",
        "control_mean",
        "treatment_mean",
        "paired_effect",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "control_successes",
        "treatment_successes",
        "discordant_control_only",
        "discordant_treatment_only",
        "mcnemar_p_value",
        "wilcoxon_p_value",
        "holm_adjusted_wilcoxon_p_value",
    ]


def cumulative_fieldnames() -> list[str]:
    """Return stable cumulative-success CSV headers."""

    return ["study_area", "scenario_id", "method", "iteration", "success_rate", "n_runs"]


def failure_fieldnames() -> list[str]:
    """Return stable failure-summary CSV headers."""

    return ["study_area", "scenario_id", "method", "failure_category", "count", "total", "fraction"]


def descriptive_fieldnames() -> list[str]:
    """Return stable descriptive-statistics CSV headers."""

    return [
        "study_area",
        "scenario_id",
        "method",
        "metric",
        "n",
        "mean",
        "standard_deviation",
        "median",
        "q1",
        "q3",
        "success_rate",
        "success_ci_low",
        "success_ci_high",
    ]


def rows_from_json(payload: Iterable[dict[str, Any]]) -> list[ResearchResultRow]:
    """Restore normalized research rows from a JSON payload."""

    return [ResearchResultRow.model_validate(item) for item in payload]
