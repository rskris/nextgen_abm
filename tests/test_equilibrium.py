"""Tests for Day-to-Day Evolutionary Replanning Engine."""

import pytest
from nextgen_abm.traffic.network import HierarchicalMultiModalNetwork
from nextgen_abm.traffic.equilibrium import DayToDayEquilibriumEngine, HouseholdAgent
from nextgen_abm.household.household_milp import MemberAgenda, JointActivitySpec, EscortSpec
from nextgen_abm.household.assets import VehicleAsset
from nextgen_abm.scheduler.milp import ActivityDefinition


def test_day_to_day_replanning_loop():
    net = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network()
    engine = DayToDayEquilibriumEngine(network=net)

    home_loc = [("bldg_home", (-119.82, 34.43))]
    work_loc = [("bldg_work", (-119.70, 34.42))]

    # Household 1
    m1 = MemberAgenda(
        person_id="hh1_p1",
        is_adult=True,
        has_license=True,
        activities=[
            ActivityDefinition("hh1_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("hh1_work", "work", True, work_loc, 6.0, 9.0, 8.0, 18.0),
            ActivityDefinition("hh1_home2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
        ],
    )
    hh1 = HouseholdAgent(
        household_id="hh1",
        members=[m1],
        vehicles=[VehicleAsset("veh1", "hh1", "ev", "bldg_home", "bldg_home")],
    )

    # Household 2
    m2 = MemberAgenda(
        person_id="hh2_p1",
        is_adult=True,
        has_license=True,
        activities=[
            ActivityDefinition("hh2_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("hh2_work", "work", True, work_loc, 6.0, 9.0, 8.5, 18.5),
            ActivityDefinition("hh2_home2", "home_night", True, home_loc, 4.0, 10.0, 17.5, 24.0),
        ],
    )
    hh2 = HouseholdAgent(
        household_id="hh2",
        members=[m2],
        vehicles=[VehicleAsset("veh2", "hh2", "ice", "bldg_home", "bldg_home")],
    )

    metrics = engine.run_equilibrium(households=[hh1, hh2], max_iterations=2)

    assert len(metrics) == 2
    assert metrics[0].iteration == 1
    assert metrics[1].iteration == 2
    assert metrics[0].total_trips >= 2
    assert metrics[0].total_vht_hours > 0.0
    assert metrics[0].total_vkt_km > 0.0
    assert metrics[1].relative_gap >= 0.0
    assert len(hh1.plans) >= 1
