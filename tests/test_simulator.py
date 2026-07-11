from asd_agent.config import load_scenario
from asd_agent.models import ExperimentCondition
from asd_agent.objective import selectivity
from asd_agent.simulator import VirtualLab, saturating_response


def test_saturation_behavior_is_monotone_and_bounded() -> None:
    low = saturating_response(0.5, 1.0)
    high = saturating_response(6.0, 1.0)
    very_high = saturating_response(12.0, 1.0)

    assert 0.0 < low < high < 1.0
    assert very_high > 0.999


def test_nucleation_delay_blocks_growth_until_delay_passes() -> None:
    config = load_scenario("inherent_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    lab = VirtualLab(config, seed=1)
    before_delay = ExperimentCondition(
        precursor_dose_s=8.0,
        coreactant_dose_s=8.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=60,
    )
    after_delay = before_delay.model_copy(update={"cycles": 75})

    assert lab.surface_thickness("NGA", before_delay) == 0.0
    assert lab.surface_thickness("NGA", after_delay) > 0.0


def test_inhibitor_selectivity_blocks_nga_more_than_ga() -> None:
    config = load_scenario("inhibitor_selectivity").model_copy(update={"noise_sigma_nm": 0.0})
    lab = VirtualLab(config, seed=1)
    base = ExperimentCondition(
        precursor_dose_s=6.0,
        coreactant_dose_s=6.0,
        inhibitor_dose_s=0.0,
        temperature_c=180.0,
        cycles=80,
    )
    inhibited = base.model_copy(update={"inhibitor_dose_s": 3.0})

    base_record = lab.simulate(base)
    inhibited_record = lab.simulate(inhibited)

    ga_fraction = inhibited_record.ga_thickness_nm / base_record.ga_thickness_nm
    nga_fraction = inhibited_record.nga_thickness_nm / base_record.nga_thickness_nm
    assert nga_fraction < ga_fraction
    assert inhibited_record.selectivity > base_record.selectivity


def test_selectivity_is_stable_when_both_thicknesses_are_zero() -> None:
    assert selectivity(0.0, 0.0) == 0.0
