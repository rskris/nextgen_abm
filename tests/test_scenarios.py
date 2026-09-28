"""Test regional policy scenario evaluators for Santa Barbara County."""

import pytest
from nextgen_abm.policy.scenarios import PolicyScenarioEngine, ScenarioResult


def test_us101_peak_spreading_evaluation():
    engine = PolicyScenarioEngine()
    result = engine.evaluate_us101_peak_spreading(n_commuters=500, pct_flexible_workers=0.40)

    assert isinstance(result, ScenarioResult)
    assert result.scenario_id == "scenario_us101_peak_spreading"
    assert result.key_metrics["total_commuters"] == 500
    assert result.key_metrics["peak_volume_reduction_pct"] > 10.0
    assert "US-101" in result.summary_findings


def test_pacific_surfliner_clock_face_evaluation():
    engine = PolicyScenarioEngine()
    result = engine.evaluate_pacific_surfliner_clock_face(
        n_corridor_commuters=3000,
        baseline_headway_min=120,
        clock_face_headway_min=60
    )

    assert isinstance(result, ScenarioResult)
    assert result.key_metrics["new_rail_share_pct"] > result.key_metrics["baseline_rail_share_pct"]
    assert result.key_metrics["daily_diverted_car_trips"] > 0
    assert result.key_metrics["estimated_households_shedding_vehicle"] > 0
    assert result.key_metrics["annual_corridor_vmt_reduced"] > 1000000


def test_ucsb_staggered_classes_evaluation():
    engine = PolicyScenarioEngine()
    result = engine.evaluate_ucsb_staggered_classes(n_students=150)

    assert isinstance(result, ScenarioResult)
    assert result.key_metrics["peak_flow_reduction_pct"] > 0.0
    assert "Pardall Gate" in result.summary_findings


def test_jobs_housing_relocation_evaluation():
    engine = PolicyScenarioEngine()
    result = engine.evaluate_jobs_housing_relocation(new_south_coast_units=3000)

    assert isinstance(result, ScenarioResult)
    assert result.key_metrics["commuters_relocated_from_north_county"] > 1000
    assert result.key_metrics["annual_vmt_eliminated"] > 10000000
    assert result.key_metrics["annual_co2_metric_tons_avoided"] > 1000.0


def test_ev_grid_load_profiles_evaluation():
    engine = PolicyScenarioEngine()
    result = engine.evaluate_ev_grid_load_profiles(n_ev_fleet=10000, pct_workplace_charging=0.40)

    assert isinstance(result, ScenarioResult)
    assert result.key_metrics["unmanaged_evening_peak_mw"] > 0.0
    assert result.key_metrics["peak_grid_stress_reduction_pct"] > 50.0
