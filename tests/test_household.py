"""Test household vehicle conservation, EV charging, escorting, and BEAM export."""

import pytest
from pathlib import Path
from nextgen_abm.household.assets import HouseholdVehicleManager, VehicleAsset
from nextgen_abm.household.ev import EVBatteryTracker
from nextgen_abm.household.escort import HouseholdCoordinator
from nextgen_abm.household.beam_export import BEAMExporter
from nextgen_abm.scheduler.milp import ScheduledPlan, ScheduledActivity, ScheduledLeg


def test_vehicle_space_time_conservation():
    # Household with 1 car starting at Home
    car = VehicleAsset(
        vehicle_id="car_01",
        household_id="hh_01",
        vehicle_type="ice",
        initial_location="bldg_home",
        current_location="bldg_home"
    )
    mgr = HouseholdVehicleManager("hh_01", [car])

    # Person 1 drives car from Home to Work from 08:00 to 08:45
    success = mgr.record_trip_leg("car_01", "p1", "bldg_home", "bldg_work", 8.0, 8.75)
    assert success is True
    assert car.current_location == "bldg_work"

    # Person 2 at Home attempts to drive car_01 at 09:00 -> MUST FAIL! Car is at Work!
    can_res, reason = mgr.can_reserve_vehicle("car_01", "p2", "bldg_home", 9.0, 9.5)
    assert can_res is False
    assert "bldg_work" in reason

    # Person 1 drives car back Home from 17:00 to 17:45 -> Succeeds!
    success_return = mgr.record_trip_leg("car_01", "p1", "bldg_work", "bldg_home", 17.0, 17.75)
    assert success_return is True
    assert car.current_location == "bldg_home"

    # Now Person 2 can take it for an evening grocery trip from Home at 18:30!
    can_evening, _ = mgr.can_reserve_vehicle("car_01", "p2", "bldg_home", 18.5, 19.5)
    assert can_evening is True


def test_ev_battery_soc_and_charging():
    # 65 kWh battery starting at 80% SoC (52 kWh)
    ev = EVBatteryTracker(vehicle_id="ev_01", battery_capacity_kwh=65.0, initial_soc_pct=80.0)
    assert ev.current_soc_pct == 80.0

    # Morning commute consumes 12 kWh
    ok, soc = ev.consume_energy(12.0)
    assert ok is True
    assert round(soc, 1) == round(((52.0 - 12.0) / 65.0) * 100.0, 1)

    # Workplace charging session: 4 hours at 6.6 kW adds up to ~26.4 kWh
    session = ev.plan_charging_session("workplace_l2", "bldg_work", start_hour=9.0, duration_hours=4.0, target_soc_pct=95.0)
    assert session.energy_added_kwh > 0
    assert ev.current_soc_pct > 80.0

    # Attempt massive trip that drains battery below 15% reserve threshold -> should fail!
    ok_drain, _ = ev.consume_energy(60.0)
    assert ok_drain is False


def test_school_escort_chain_optimization():
    coord = HouseholdCoordinator()

    home = (-119.850, 34.420)
    school = (-119.830, 34.425)

    # Worker 1 works close to school; Worker 2 works 20 miles away
    adults = [
        {"person_id": "adult_near", "work_coords": (-119.820, 34.430)},
        {"person_id": "adult_far", "work_coords": (-120.435, 34.950)},
    ]

    escort = coord.assign_school_escort_driver(
        home_coords=home,
        school_coords=school,
        school_start_hour=8.25,  # 08:15 AM
        adult_workers=adults,
        child_person_id="child_01"
    )

    # The adult with the nearby workplace should be chosen to minimize detour!
    assert escort.driver_person_id == "adult_near"
    assert escort.school_arrival_hour <= 8.25
    assert escort.detour_minutes < 15.0


def test_beam_exporter(tmp_path: Path):
    exporter = BEAMExporter(output_dir=tmp_path)

    plan = ScheduledPlan(
        agent_id="test_agent",
        total_utility=85.0,
        solve_time_ms=12.5,
        activities=[
            ScheduledActivity("a1", "home", "bldg_home", (-119.85, 34.42), 6.0, 8.0, 2.0),
            ScheduledActivity("a2", "work", "bldg_work", (-119.83, 34.43), 8.5, 17.0, 8.5),
        ],
        legs=[
            ScheduledLeg("a1", "a2", "car", 8.0, 8.5, 30.0, 5.2)
        ]
    )

    # Test plans.xml export
    xml_path = exporter.export_plans_xml([plan])
    assert xml_path.exists()
    xml_content = xml_path.read_text()
    assert "<person id=\"test_agent\">" in xml_content
    assert "<activity type=\"home\"" in xml_content
    assert "<leg mode=\"car\"" in xml_content

    # Test beamVehicles.csv export
    vehicles = [
        {"vehicle_id": "v1", "vehicle_type": "ev", "initial_soc_pct": 90.0, "household_id": "hh1"},
        {"vehicle_id": "v2", "vehicle_type": "ice", "initial_soc_pct": 0.0, "household_id": "hh2"}
    ]
    csv_path = exporter.export_beam_vehicles_csv(vehicles)
    assert csv_path.exists()
    csv_content = csv_path.read_text()
    assert "BEV_Standard" in csv_content
    assert "ICE_Midsize" in csv_content
