from pathlib import Path

from asd_agent.bo.records import (
    BOExperimentRecord,
    BORunRecord,
    CandidateProposal,
    OptimizerState,
    RunManifest,
)
from asd_agent.bo.serialization import (
    load_bo_records,
    load_optimizer_state,
    load_run_record,
    save_bo_records,
    save_optimizer_state,
    save_run_record,
)
from asd_agent.config import load_scenario
from asd_agent.experiment_loop import run_conditions
from asd_agent.models import ExperimentCondition
from asd_agent.simulator import VirtualLab


def bo_condition() -> ExperimentCondition:
    return ExperimentCondition(
        precursor_dose_s=6.0,
        coreactant_dose_s=6.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=70,
    )


def test_candidate_ids_are_unique() -> None:
    condition = bo_condition()
    first = CandidateProposal.create(condition, "test_optimizer", seed=11)
    second = CandidateProposal.create(condition, "test_optimizer", seed=11)

    assert first.candidate_id != second.candidate_id
    assert first.parameters == second.parameters


def test_bo_record_serialization_round_trip(tmp_path: Path) -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    lab = VirtualLab(config, seed=1)
    proposal = CandidateProposal.create(bo_condition(), "test_optimizer", seed=1)
    record = lab.simulate(proposal.parameters, "exp_001", "Test proposal.")
    bo_record = BOExperimentRecord.from_experiment_record(record, proposal=proposal)
    path = tmp_path / "bo_records.json"

    save_bo_records([bo_record], path)
    restored = load_bo_records(path)

    assert restored == [bo_record]
    assert restored[0].experiment.experiment_id == "exp_001"


def test_run_manifest_creation_records_runtime_metadata() -> None:
    path = Path("configs/inherent_selectivity.yaml")
    manifest = RunManifest.create(
        config_path=path,
        method="bo_test",
        scenario="inherent_selectivity",
        experiment_budget=5,
        named_seeds={"simulator": 123, "optimizer": 456},
        dependency_names=["numpy", "scipy"],
        repo_root=Path.cwd(),
        run_id="run_test",
    )

    assert manifest.run_id == "run_test"
    assert manifest.configuration_path.endswith("inherent_selectivity.yaml")
    assert len(manifest.configuration_hash) == 64
    assert manifest.dependency_versions["numpy"] != "not installed"
    assert "scipy" in manifest.dependency_versions
    assert manifest.named_seeds == {"simulator": 123, "optimizer": 456}


def test_optimizer_state_and_run_record_save_restore(tmp_path: Path) -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    run = run_conditions(config, [bo_condition()], "legacy", seed=1)
    manifest = RunManifest.create(
        config_path=Path("configs/inherent_selectivity.yaml"),
        method="legacy",
        scenario=config.scenario,
        experiment_budget=1,
        named_seeds={"simulator": 1},
        dependency_names=["numpy"],
        repo_root=Path.cwd(),
        run_id="run_legacy",
    )
    state = OptimizerState(
        optimizer="test_optimizer",
        state={"phase": "initial"},
        observation_ids=["legacy_001"],
        random_seed=1,
    )
    run_record = BORunRecord.from_optimization_run(run, manifest, optimizer_state=state)

    state_path = tmp_path / "optimizer_state.json"
    run_path = tmp_path / "run_record.json"
    save_optimizer_state(state, state_path)
    save_run_record(run_record, run_path)

    assert load_optimizer_state(state_path) == state
    restored_run = load_run_record(run_path)
    assert restored_run == run_record
    assert restored_run.optimizer_observations()[0].experiment_id == "legacy_001"


def test_legacy_experiment_record_can_be_wrapped_without_proposal() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    record = VirtualLab(config, seed=1).simulate(bo_condition(), "legacy_001", "Legacy row.")

    wrapped = BOExperimentRecord.from_experiment_record(record)

    assert wrapped.proposal is None
    assert wrapped.experiment == record
    assert wrapped.optimizer_observation.experiment_id == record.experiment_id
