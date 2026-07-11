import pytest

from asd_agent.baselines import grid_search
from asd_agent.config import load_scenario
from asd_agent.models import ExperimentCondition
from asd_agent.objective import validate_safety


def test_impossible_scenario_reports_no_selective_window() -> None:
    config = load_scenario("impossible_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    run = grid_search(config, budget=81, seed=123)

    assert run.status == "no_selective_window"
    assert not any(record.meets_objective for record in run.records)


def test_safety_bounds_reject_out_of_range_dose_and_temperature() -> None:
    config = load_scenario("inherent_selectivity")
    condition = ExperimentCondition(
        precursor_dose_s=999.0,
        coreactant_dose_s=1.0,
        inhibitor_dose_s=0.0,
        temperature_c=-20.0,
        cycles=20,
    )

    violations = validate_safety(condition, config)
    assert any("precursor_dose_s" in violation for violation in violations)
    assert any("temperature_c" in violation for violation in violations)


def test_safety_model_rejects_invalid_negative_condition() -> None:
    with pytest.raises(ValueError):
        ExperimentCondition(
            precursor_dose_s=-1.0,
            coreactant_dose_s=1.0,
            inhibitor_dose_s=0.0,
            temperature_c=180.0,
            cycles=20,
        )
