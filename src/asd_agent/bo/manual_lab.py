"""Manual laboratory handoff backend for Stage 2 BO candidates."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from asd_agent.bo.records import utc_now
from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2Decision,
    Stage2Outcomes,
    condition_from_stage2_decision,
    evaluate_stage2_constraints,
    stage2_process_time,
    validate_stage2_decision,
)
from asd_agent.objective import selectivity

if TYPE_CHECKING:
    from asd_agent.bo.stage2_mobo import Stage2Observation

ManualExperimentStatus = Literal["pending", "completed"]
QualityControlStatus = Literal["pass", "warning", "fail"]


class ManualLabError(ValueError):
    """Raised when a manual-lab plan or measurement is invalid."""


class ManualCandidate(BaseModel):
    """Small optimizer-candidate view accepted by `ManualLabBackend`."""

    candidate_id: str = Field(min_length=1)
    decision: Stage2Decision
    optimizer: str = Field(default="manual_candidate", min_length=1)

    model_config = ConfigDict(extra="forbid")


class ManualLabPlanRow(BaseModel):
    """One pending or completed laboratory plan row."""

    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    optimizer: str = Field(min_length=1)
    precursor_dose_s: float
    temperature_c: float
    cycle_count: int
    fixed_settings: dict[str, float]
    requested_measurements: list[str]
    safety_notes: list[str]
    status: ManualExperimentStatus = "pending"
    operator_notes: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    def decision(self) -> Stage2Decision:
        """Return the optimizer decision represented by this lab plan."""

        return Stage2Decision(
            precursor_dose_s=self.precursor_dose_s,
            temperature_c=self.temperature_c,
            cycle_count=self.cycle_count,
        )

    def csv_row(self) -> dict[str, object]:
        """Return a flat CSV row for operators."""

        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "optimizer": self.optimizer,
            "precursor_dose_s": self.precursor_dose_s,
            "temperature_c": self.temperature_c,
            "cycle_count": self.cycle_count,
            "fixed_settings_json": json.dumps(self.fixed_settings, sort_keys=True),
            "requested_measurements_json": json.dumps(self.requested_measurements),
            "safety_notes_json": json.dumps(self.safety_notes),
            "status": self.status,
            "operator_notes": self.operator_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ManualMeasurementRecord(BaseModel):
    """Completed measurement row imported from a human-operated lab."""

    experiment_id: str = Field(min_length=1)
    ga_thickness: float = Field(ge=0.0)
    nga_thickness: float = Field(ge=0.0)
    units: str
    ga_uncertainty: float | None = Field(default=None, ge=0.0)
    nga_uncertainty: float | None = Field(default=None, ge=0.0)
    measurement_method: str = Field(min_length=1)
    replicate_identifier: str = Field(min_length=1)
    quality_control_status: QualityControlStatus
    operator_notes: str = Field(min_length=1)
    measured_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="forbid")

    @field_validator("units")
    @classmethod
    def units_must_be_nanometers(cls, value: str) -> str:
        """Normalize and validate thickness units."""

        normalized = value.strip().lower()
        if normalized not in {"nm", "nanometer", "nanometers"}:
            raise ValueError("manual measurements must use nanometer units")
        return "nm"

    def csv_row(self) -> dict[str, object]:
        """Return a flat CSV row matching the ingestion schema."""

        return {
            "experiment_id": self.experiment_id,
            "ga_thickness": self.ga_thickness,
            "nga_thickness": self.nga_thickness,
            "units": self.units,
            "ga_uncertainty": "" if self.ga_uncertainty is None else self.ga_uncertainty,
            "nga_uncertainty": "" if self.nga_uncertainty is None else self.nga_uncertainty,
            "measurement_method": self.measurement_method,
            "replicate_identifier": self.replicate_identifier,
            "quality_control_status": self.quality_control_status,
            "operator_notes": self.operator_notes,
            "measured_at": self.measured_at,
        }


class ManualLabBackend:
    """Manual handoff backend that keeps human lab execution outside BO."""

    def __init__(
        self,
        config: Stage2Config,
        *,
        run_id: str | None = None,
        requested_measurements: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or f"manual_run_{uuid4().hex[:12]}"
        self.requested_measurements = list(
            requested_measurements
            or [
                "GA thickness (nm)",
                "NGA thickness (nm)",
                "measurement uncertainty when available",
            ]
        )
        self.plans: dict[str, ManualLabPlanRow] = {}
        self.ledger: list[Stage2Observation] = []
        self.measurements: dict[str, ManualMeasurementRecord] = {}

    @classmethod
    def from_plan_json(
        cls,
        config: Stage2Config,
        path: str | Path,
    ) -> ManualLabBackend:
        """Restore pending/completed plan state exported by a previous process."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ManualLabError("manual lab plan JSON must contain an object")
        scenario_id = payload.get("scenario_id")
        if scenario_id != config.scenario_id:
            raise ManualLabError(
                f"plan scenario {scenario_id!r} does not match {config.scenario_id!r}"
            )
        run_id = payload.get("run_id")
        raw_plans = payload.get("plans")
        if not isinstance(run_id, str) or not run_id:
            raise ManualLabError("manual lab plan JSON requires a run_id")
        if not isinstance(raw_plans, list):
            raise ManualLabError("manual lab plan JSON requires a plans list")
        backend = cls(config, run_id=run_id)
        for raw_plan in raw_plans:
            plan = ManualLabPlanRow.model_validate(raw_plan)
            if plan.run_id != run_id:
                raise ManualLabError("manual lab plan row has a mismatched run_id")
            if plan.experiment_id in backend.plans:
                raise ManualLabError(f"duplicate plan experiment_id {plan.experiment_id!r}")
            backend.plans[plan.experiment_id] = plan
        return backend

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""

        return "manual_lab"

    def optimizer_view(self) -> dict[str, object]:
        """Return optimizer-visible lab context without hidden simulator parameters."""

        return {
            "backend": self.backend_name,
            "run_id": self.run_id,
            "scenario_id": self.config.scenario_id,
            "decision_variables": self.config.optimizer_view()["decision_variables"],
            "fixed_parameters": self.fixed_settings(),
            "pending_experiment_ids": [
                plan.experiment_id for plan in self.plans.values() if plan.status == "pending"
            ],
            "completed_experiment_ids": [observation.experiment_id for observation in self.ledger],
        }

    def fixed_settings(self) -> dict[str, float]:
        """Return Stage 2 fixed reactor settings for operator plans."""

        condition = condition_from_stage2_decision(
            self.config,
            Stage2Decision(
                precursor_dose_s=self.config.hard_bounds.precursor_dose_s.min,
                temperature_c=self.config.hard_bounds.temperature_c.min,
                cycle_count=int(self.config.hard_bounds.cycle_count.min),
            ),
        )
        return {
            "coreactant_dose_s": condition.coreactant_dose_s,
            "inhibitor_dose_s": condition.inhibitor_dose_s,
        }

    def receive_candidate(
        self,
        candidate: ManualCandidate | Mapping[str, object] | object,
        *,
        experiment_id: str | None = None,
        operator_notes: str = "",
    ) -> ManualLabPlanRow:
        """Validate an optimizer candidate and mark the lab experiment pending."""

        manual_candidate = coerce_manual_candidate(candidate)
        violations = validate_stage2_decision(self.config, manual_candidate.decision)
        if violations:
            raise ManualLabError(f"candidate violates hard bounds: {violations}")
        resolved_experiment_id = experiment_id or f"manual_{len(self.plans) + 1:03d}"
        if resolved_experiment_id in self.plans:
            raise ManualLabError(f"duplicate manual experiment_id {resolved_experiment_id!r}")
        plan = ManualLabPlanRow(
            experiment_id=resolved_experiment_id,
            run_id=self.run_id,
            candidate_id=manual_candidate.candidate_id,
            optimizer=manual_candidate.optimizer,
            precursor_dose_s=manual_candidate.decision.precursor_dose_s,
            temperature_c=manual_candidate.decision.temperature_c,
            cycle_count=manual_candidate.decision.cycle_count,
            fixed_settings=self.fixed_settings(),
            requested_measurements=list(self.requested_measurements),
            safety_notes=[
                "Validated against immutable Stage 2 hard bounds.",
                "Manual execution only; no autonomous reactor control is implemented.",
            ],
            status="pending",
            operator_notes=operator_notes,
        )
        self.plans[plan.experiment_id] = plan
        return plan

    def export_plan(
        self,
        output_dir: str | Path,
        *,
        plans: Sequence[ManualLabPlanRow] | None = None,
    ) -> tuple[Path, Path]:
        """Export pending or supplied laboratory plans to CSV and JSON."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        selected = list(plans) if plans is not None else list(self.plans.values())
        csv_path = destination / "manual_lab_plan.csv"
        json_path = destination / "manual_lab_plan.json"
        write_plan_csv(selected, csv_path)
        json_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "scenario_id": self.config.scenario_id,
                    "plans": [plan.model_dump(mode="json") for plan in selected],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return csv_path, json_path

    def export_measurement_template(self, output_dir: str | Path) -> Path:
        """Write a CSV template for operators to fill in completed measurements."""

        path = Path(output_dir) / "manual_lab_measurements_template.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "experiment_id": plan.experiment_id,
                "ga_thickness": "",
                "nga_thickness": "",
                "units": "nm",
                "ga_uncertainty": "",
                "nga_uncertainty": "",
                "measurement_method": "",
                "replicate_identifier": "",
                "quality_control_status": "pass",
                "operator_notes": "",
                "measured_at": "",
            }
            for plan in self.plans.values()
            if plan.status == "pending"
        ]
        write_dict_csv(rows, path, measurement_fieldnames())
        return path

    def import_completed_measurements(
        self,
        source: str | Path | Sequence[ManualMeasurementRecord | Mapping[str, object]],
    ) -> list[Stage2Observation]:
        """Import completed manual measurements and append Stage 2 observations."""

        measurements = load_measurements(source)
        imported: list[Stage2Observation] = []
        for measurement in measurements:
            imported.append(self._ingest_measurement(measurement))
        return imported

    def optimizer_observations(self) -> list[Stage2Observation]:
        """Return completed manual observations for BO continuation."""

        return [observation.model_copy(deep=True) for observation in self.ledger]

    def _ingest_measurement(self, measurement: ManualMeasurementRecord) -> Stage2Observation:
        from asd_agent.bo.stage2_mobo import Stage2Observation

        if measurement.experiment_id not in self.plans:
            raise ManualLabError(
                f"measurement references unknown experiment_id {measurement.experiment_id!r}"
            )
        if measurement.experiment_id in self.measurements:
            raise ManualLabError(
                f"duplicate completed measurement for {measurement.experiment_id!r}"
            )
        if measurement.quality_control_status == "fail":
            raise ManualLabError(
                f"quality-control status is fail for {measurement.experiment_id!r}; "
                "do not add failed measurements to the optimizer ledger"
            )
        plan = self.plans[measurement.experiment_id]
        decision = plan.decision()
        outcomes = Stage2Outcomes(
            ga_thickness_nm=measurement.ga_thickness,
            nga_thickness_nm=measurement.nga_thickness,
            selectivity=selectivity(measurement.ga_thickness, measurement.nga_thickness),
            process_time_s=stage2_process_time(self.config, decision),
        )
        observation = Stage2Observation(
            experiment_id=plan.experiment_id,
            decision=decision,
            outcomes=outcomes,
            constraint_evaluation=evaluate_stage2_constraints(self.config, decision, outcomes),
        )
        self.measurements[measurement.experiment_id] = measurement
        self.ledger.append(observation)
        self.plans[measurement.experiment_id] = plan.model_copy(
            update={"status": "completed", "updated_at": utc_now()}
        )
        return observation


def coerce_manual_candidate(
    candidate: ManualCandidate | Mapping[str, object] | object,
) -> ManualCandidate:
    """Convert a manual candidate, mapping, or Stage 2 proposal-like object."""

    if isinstance(candidate, ManualCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        return ManualCandidate.model_validate(candidate)
    candidate_id = getattr(candidate, "candidate_id", None)
    decision = getattr(candidate, "decision", None)
    optimizer = getattr(candidate, "optimizer", "unknown_optimizer")
    if not isinstance(candidate_id, str) or not isinstance(decision, Stage2Decision):
        raise ManualLabError("manual lab candidates must provide candidate_id and Stage2Decision")
    return ManualCandidate(candidate_id=candidate_id, decision=decision, optimizer=str(optimizer))


def load_measurements(
    source: str | Path | Sequence[ManualMeasurementRecord | Mapping[str, object]],
) -> list[ManualMeasurementRecord]:
    """Load manual measurements from JSON, CSV, or in-memory rows."""

    if isinstance(source, str | Path):
        path = Path(source)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_rows = (
                payload.get("measurements", payload) if isinstance(payload, dict) else payload
            )
            if not isinstance(raw_rows, list):
                raise ManualLabError("manual measurement JSON must contain a list")
            return [
                ManualMeasurementRecord.model_validate(clean_measurement_row(row))
                for row in raw_rows
            ]
        if path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8") as handle:
                return [
                    ManualMeasurementRecord.model_validate(clean_measurement_row(row))
                    for row in csv.DictReader(handle)
                ]
        raise ManualLabError("manual measurements must be imported from .csv or .json")
    return [ManualMeasurementRecord.model_validate(clean_measurement_row(row)) for row in source]


def clean_measurement_row(row: ManualMeasurementRecord | Mapping[str, object]) -> dict[str, object]:
    """Normalize optional blank fields before Pydantic validation."""

    if isinstance(row, ManualMeasurementRecord):
        return row.model_dump(mode="json")
    cleaned = dict(row)
    for key in ("ga_uncertainty", "nga_uncertainty", "measured_at"):
        if cleaned.get(key) == "":
            cleaned.pop(key)
    return cleaned


def write_plan_csv(plans: Sequence[ManualLabPlanRow], path: Path) -> None:
    """Write manual lab plans to CSV with stable headers."""

    fieldnames = [
        "experiment_id",
        "run_id",
        "candidate_id",
        "optimizer",
        "precursor_dose_s",
        "temperature_c",
        "cycle_count",
        "fixed_settings_json",
        "requested_measurements_json",
        "safety_notes_json",
        "status",
        "operator_notes",
        "created_at",
        "updated_at",
    ]
    write_dict_csv([plan.csv_row() for plan in plans], path, fieldnames)


def write_measurement_csv(
    measurements: Sequence[ManualMeasurementRecord],
    path: str | Path,
) -> None:
    """Write completed manual measurements to CSV."""

    write_dict_csv(
        [measurement.csv_row() for measurement in measurements],
        Path(path),
        measurement_fieldnames(),
    )


def write_measurement_json(
    measurements: Sequence[ManualMeasurementRecord],
    path: str | Path,
) -> None:
    """Write completed manual measurements to JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"measurements": [measurement.model_dump(mode="json") for measurement in measurements]},
            indent=2,
        ),
        encoding="utf-8",
    )


def write_dict_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    fieldnames: Sequence[str],
) -> None:
    """Write dictionaries to CSV with stable headers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def measurement_fieldnames() -> list[str]:
    """Return required manual measurement CSV headers."""

    return [
        "experiment_id",
        "ga_thickness",
        "nga_thickness",
        "units",
        "ga_uncertainty",
        "nga_uncertainty",
        "measurement_method",
        "replicate_identifier",
        "quality_control_status",
        "operator_notes",
        "measured_at",
    ]
