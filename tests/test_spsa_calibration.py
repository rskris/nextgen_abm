"""Tests for Automated SPSA Calibration Loop."""

import pytest
from nextgen_abm.traffic.network import HierarchicalMultiModalNetwork
from nextgen_abm.traffic.equilibrium import HouseholdAgent
from nextgen_abm.household.household_milp import MemberAgenda
from nextgen_abm.household.assets import VehicleAsset
from nextgen_abm.scheduler.milp import ActivityDefinition
from nextgen_abm.validation.spsa_calibration import SPSACalibrator


def test_spsa_calibration_loop_and_geh():
    net = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network()
    calibrator = SPSACalibrator(network=net)

    # 1. Test GEH statistic math
    geh_zero = calibrator.compute_geh(3500.0, 3500.0)
    assert geh_zero == 0.0

    geh_diff = calibrator.compute_geh(4000.0, 3500.0)
    assert 5.0 < geh_diff < 10.0

    # 2. Test SPSA calibration run
    home_loc = [("bldg_home", (-119.82, 34.43))]
    work_loc = [("bldg_work", (-119.70, 34.42))]

    m = MemberAgenda(
        person_id="calib_p1",
        is_adult=True,
        has_license=True,
        activities=[
            ActivityDefinition("act_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("act_work", "work", True, work_loc, 6.0, 9.0, 8.0, 18.0),
            ActivityDefinition("act_home2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
        ],
    )
    hh = HouseholdAgent(
        household_id="calib_hh",
        members=[m],
        vehicles=[VehicleAsset("calib_v1", "calib_hh", "ice", "bldg_home", "bldg_home")],
    )

    targets = {
        "US101_Salinas": 3000.0,
        "US101_Fairview": 3200.0,
    }

    result = calibrator.calibrate(households=[hh], pems_targets=targets, max_iterations=2)

    assert result.iterations_run == 2
    assert len(result.loss_history) >= 2
    assert "vot_car" in result.best_parameters
    assert "corridor_capacity_freeway_vph" in result.best_parameters
    assert len(result.geh_statistics) == 2
