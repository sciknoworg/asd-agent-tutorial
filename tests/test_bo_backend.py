import pytest

from asd_agent.bo.backend import VirtualASDBackend
from asd_agent.bo.records import CandidateProposal
from asd_agent.config import load_scenario
from asd_agent.models import ExperimentCondition


def safe_condition() -> ExperimentCondition:
    return ExperimentCondition(
        precursor_dose_s=6.0,
        coreactant_dose_s=6.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=70,
    )


def test_virtual_asd_backend_executes_existing_simulator() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    backend = VirtualASDBackend(config, seed=1)
    proposal = CandidateProposal.create(
        safe_condition(),
        "bo_test",
        seed=1,
        training_observation_ids=["exp_000"],
    )

    record = backend.run_experiment(proposal, experiment_id="bo_001")

    assert record.experiment.experiment_id == "bo_001"
    assert record.proposal == proposal
    assert record.optimizer_observation.ga_thickness_nm == record.experiment.ga_thickness_nm
    assert record.experiment.ga_thickness_nm > 0.0


def test_optimizer_view_excludes_hidden_simulator_fields() -> None:
    config = load_scenario("inhibitor_selectivity")
    backend = VirtualASDBackend(config, seed=2)

    visible = backend.optimizer_view()
    visible_text = str(visible)

    assert "safety_bounds" in visible
    assert "objective" in visible
    assert "surfaces" not in visible
    assert "max_growth_per_cycle_nm" not in visible_text
    assert "nucleation_delay_cycles" not in visible_text
    assert "inhibitor_sensitivity" not in visible_text

    hidden = backend.hidden_simulator_parameters()
    assert "surfaces" in hidden
    assert "noise_sigma_nm" in hidden


def test_bo_backend_rejects_unsafe_candidate() -> None:
    config = load_scenario("inherent_selectivity")
    backend = VirtualASDBackend(config, seed=1)
    unsafe = ExperimentCondition(
        precursor_dose_s=999.0,
        coreactant_dose_s=1.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=20,
    )
    proposal = CandidateProposal.create(unsafe, "bo_test", seed=1)

    with pytest.raises(ValueError, match="unsafe BO candidate"):
        backend.run_experiment(proposal)
