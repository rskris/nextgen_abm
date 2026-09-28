"""Tests for Unified Joint Household MILP Solver."""

import pytest
from nextgen_abm.household.household_milp import (
    UnifiedHouseholdMILP,
    MemberAgenda,
    JointActivitySpec,
    EscortSpec,
)
from nextgen_abm.scheduler.milp import ActivityDefinition
from nextgen_abm.household.assets import VehicleAsset


def test_unified_household_milp_solve_time_and_coordination():
    solver = UnifiedHouseholdMILP()

    # Define household with Adult 1, Adult 2, and Child 1
    home_loc = [("bldg_home", (-119.82, 34.43))]
    work1_loc = [("bldg_work1", (-119.70, 34.41))]
    work2_loc = [("bldg_work2", (-119.85, 34.42))]
    school_loc = [("bldg_school", (-119.81, 34.44))]

    # Adult 1: Home -> Work -> Home
    m1 = MemberAgenda(
        person_id="p1_adult1",
        is_adult=True,
        has_license=True,
        activities=[
            ActivityDefinition("p1_act_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("p1_act_work", "work", True, work1_loc, 6.0, 9.0, 8.0, 18.0),
            ActivityDefinition("p1_act_home2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
        ],
    )

    # Adult 2: Home -> Work -> Home
    m2 = MemberAgenda(
        person_id="p2_adult2",
        is_adult=True,
        has_license=True,
        activities=[
            ActivityDefinition("p2_act_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("p2_act_work", "work", True, work2_loc, 6.0, 9.0, 8.5, 18.5),
            ActivityDefinition("p2_act_home2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
        ],
    )

    # Child 1: Home -> School -> Home
    c1 = MemberAgenda(
        person_id="c1_child",
        is_adult=False,
        has_license=False,
        activities=[
            ActivityDefinition("c1_act_home1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
            ActivityDefinition("c1_act_school", "school", True, school_loc, 6.0, 7.0, 8.0, 16.0),
            ActivityDefinition("c1_act_home2", "home_night", True, home_loc, 4.0, 10.0, 15.0, 24.0),
        ],
        school_act_idx=1,
    )

    # Household vehicles: 1 EV
    vehicles = [
        VehicleAsset("veh_ev_01", "hh_01", "ev", "bldg_home", "bldg_home")
    ]

    # School escort spec: one adult must escort child to school by 08:15
    escorts = [
        EscortSpec(
            child_id="c1_child",
            school_act_idx=1,
            target_school_arrival=8.25,
            school_coords=(-119.81, 34.44),
            school_building_id="bldg_school",
            eligible_driver_ids=["p1_adult1", "p2_adult2"],
        )
    ]

    # Joint dinner synchronization
    joint_acts = [
        JointActivitySpec(
            name="family_dinner",
            target_hour=19.0,
            participant_ids=["p1_adult1", "p2_adult2"],
            tolerance_hours=0.5,
            bonus_utility=20.0,
        )
    ]

    result = solver.solve_household(
        household_id="hh_01",
        members=[m1, m2, c1],
        vehicles=vehicles,
        joint_activities=joint_acts,
        escorts=escorts,
    )

    assert result.status == "optimal"
    assert result.solve_time_ms < 50.0, f"Solve time {result.solve_time_ms}ms exceeded 50ms limit"
    assert len(result.member_schedules) == 3
    assert len(result.escort_results) == 1
    assert result.escort_results[0]["status"] == "escorted_by_adult"
    assert result.escort_results[0]["driver_id"] in ["p1_adult1", "p2_adult2"]

    # Verify no license child cannot drive solo auto
    child_legs = result.member_schedules["c1_child"].legs
    for leg in child_legs:
        assert leg.chosen_mode in ["walk", "bike", "transit"]
