from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("torch")

from asd_agent.bo.manual_lab import (
    ManualCandidate,
    ManualLabBackend,
    ManualLabError,
    ManualMeasurementRecord,
)
from asd_agent.bo.stage2 import Stage2Decision
from asd_agent.bo.stage2_mobo import Stage2BOSettings, run_stage2_bo
from asd_agent.config import load_stage2_scenario


def valid_decision() -> Stage2Decision:
    return Stage2Decision(precursor_dose_s=5.0, temperature_c=175.0, cycle_count=70)


def valid_measurement(experiment_id: str = "manual_001") -> ManualMeasurementRecord:
    return ManualMeasurementRecord(
        experiment_id=experiment_id,
        ga_thickness=5.4,
        nga_thickness=0.25,
        units="nm",
        ga_uncertainty=0.05,
        nga_uncertainty=0.05,
        measurement_method="ellipsometry",
        replicate_identifier="replicate_001",
        quality_control_status="pass",
        operator_notes="QC passed.",
    )


def backend() -> ManualLabBackend:
    return ManualLabBackend(load_stage2_scenario("inherent_selectivity"), run_id="test_manual_run")


def test_manual_lab_exports_pending_plan_and_template(tmp_path: Path) -> None:
    lab = backend()
    plan = lab.receive_candidate(
        ManualCandidate(
            candidate_id="cand_manual_001",
            decision=valid_decision(),
            optimizer="test_optimizer",
        ),
        experiment_id="manual_001",
        operator_notes="Operator should review the safety notes.",
    )

    plan_csv, plan_json = lab.export_plan(tmp_path)
    template_csv = lab.export_measurement_template(tmp_path)

    assert plan.status == "pending"
    assert plan_csv.exists()
    assert plan_json.exists()
    assert template_csv.exists()
    assert "fixed_settings_json" in plan_csv.read_text(encoding="utf-8")
    assert "requested_measurements_json" in plan_csv.read_text(encoding="utf-8")
    assert "manual_001" in template_csv.read_text(encoding="utf-8")
    assert lab.optimizer_view()["pending_experiment_ids"] == ["manual_001"]

    restored = ManualLabBackend.from_plan_json(lab.config, plan_json)
    assert restored.run_id == lab.run_id
    assert restored.plans["manual_001"].status == "pending"


def test_manual_lab_rejects_out_of_bounds_candidate() -> None:
    lab = backend()
    with pytest.raises(ManualLabError):
        lab.receive_candidate(
            ManualCandidate(
                candidate_id="unsafe",
                decision=Stage2Decision(
                    precursor_dose_s=999.0,
                    temperature_c=175.0,
                    cycle_count=70,
                ),
                optimizer="test_optimizer",
            )
        )


def test_manual_lab_import_validates_units_and_required_fields() -> None:
    lab = backend()
    lab.receive_candidate(
        ManualCandidate(
            candidate_id="cand_manual_001",
            decision=valid_decision(),
            optimizer="test_optimizer",
        ),
        experiment_id="manual_001",
    )

    bad_units = valid_measurement().model_copy(update={"units": "micrometer"})
    with pytest.raises(ValueError):
        lab.import_completed_measurements([bad_units.model_dump(mode="json")])

    missing_required = valid_measurement().model_dump(mode="json")
    missing_required.pop("measurement_method")
    with pytest.raises(ValueError):
        lab.import_completed_measurements([missing_required])


def test_manual_lab_rejects_failed_quality_control() -> None:
    lab = backend()
    lab.receive_candidate(
        ManualCandidate(
            candidate_id="cand_manual_001",
            decision=valid_decision(),
            optimizer="test_optimizer",
        ),
        experiment_id="manual_001",
    )
    failed = valid_measurement().model_copy(update={"quality_control_status": "fail"})

    with pytest.raises(ManualLabError):
        lab.import_completed_measurements([failed])

    assert not lab.ledger


def test_manual_lab_import_updates_ledger_and_allows_bo_continuation() -> None:
    lab = backend()
    lab.receive_candidate(
        ManualCandidate(
            candidate_id="cand_manual_001",
            decision=valid_decision(),
            optimizer="test_optimizer",
        ),
        experiment_id="manual_001",
    )

    observations = lab.import_completed_measurements([valid_measurement()])

    assert observations[0].experiment_id == "manual_001"
    assert observations[0].constraint_evaluation.feasible
    assert lab.plans["manual_001"].status == "completed"
    assert lab.optimizer_view()["completed_experiment_ids"] == ["manual_001"]

    settings = Stage2BOSettings(
        experiment_budget=3,
        initial_design_size=2,
        qmc_samples=8,
        num_restarts=1,
        raw_samples=8,
        acquisition_timeout_s=2.0,
        candidate_cycle_values=[30, 50, 70],
        random_fallback_points=16,
    )
    result = run_stage2_bo(
        lab.config,
        settings,
        simulator_seed=120,
        optimizer_seed=220,
        initial_observations=lab.optimizer_observations(),
    )

    assert result.observations[0].experiment_id == "manual_001"
    assert len(result.observations) <= settings.experiment_budget
