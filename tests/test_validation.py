"""Test CHTS duration calibration, PeMS GEH statistics, and PyDeck dashboard rendering."""

import pytest
from pathlib import Path
from nextgen_abm.spatial.overture import OvertureClient
from nextgen_abm.spatial.capacity import BuildingCapacityModel
from nextgen_abm.validation.calibration import SurveyCalibrationEngine
from nextgen_abm.validation.traffic_validation import PeMSValidator
from nextgen_abm.validation.dashboard import DashboardRenderer


def test_chts_duration_calibration():
    engine = SurveyCalibrationEngine()

    simulated = {
        "work": [7.5, 8.0, 7.2, 7.8, 8.1],
        "grocery": [0.6, 0.7, 0.5, 0.8],
        "dining": [1.0, 1.2, 0.9, 1.3],
    }

    metrics = engine.evaluate_durations(simulated)

    assert "work" in metrics
    assert "grocery" in metrics
    assert "dining" in metrics

    work_metric = metrics["work"]
    assert work_metric.simulated_mean_hours > 7.0
    assert work_metric.mean_abs_error_hours < 0.5
    assert work_metric.is_calibrated is True


def test_pems_traffic_volume_validation():
    validator = PeMSValidator()

    # Perfect or near-perfect match
    modeled = {
        "pems_us101_gaviota": 2680,     # observed: 2650
        "pems_us101_storke": 4900,      # observed: 4850
        "pems_us101_patterson": 5050,   # observed: 5100
        "pems_us101_milpas": 5350,      # observed: 5400
        "pems_us101_carpinteria": 4150  # observed: 4200
    }

    results = validator.validate_station_volumes(modeled)
    assert len(results) == 5

    for station_id, res in results.items():
        assert res.geh_statistic < 5.0  # Must satisfy FHWA GEH < 5.0 validation threshold!
        assert res.is_fhwa_validated is True


def test_pydeck_dashboard_rendering(tmp_path: Path):
    import pandas as pd
    renderer = DashboardRenderer(output_dir=tmp_path)

    # 1. Render OD flows map (default representative)
    od_html_path = renderer.render_od_flow_map(filename="test_od_flow.html")
    assert od_html_path.exists()
    assert od_html_path.stat().st_size > 1000

    # 1b. Render flow deck from raw schedules DataFrame
    sched_df = pd.DataFrame([
        {"household_id": "hh1", "person_id": "p1", "act_id": "a1", "act_type": "home", "location_id": "home_loc", "lon": -119.82, "lat": 34.43, "start_hour": 6.0, "end_hour": 8.0, "duration_hours": 2.0},
        {"household_id": "hh1", "person_id": "p1", "act_id": "a2", "act_type": "work", "location_id": "work_loc", "lon": -119.70, "lat": 34.42, "start_hour": 8.5, "end_hour": 17.0, "duration_hours": 8.5},
        {"household_id": "hh1", "person_id": "p1", "act_id": "a3", "act_type": "home", "location_id": "home_loc", "lon": -119.82, "lat": 34.43, "start_hour": 17.5, "end_hour": 24.0, "duration_hours": 6.5},
    ])
    sched_html_path = renderer.render_flow_deck(flows_df=sched_df, filename="test_sched_deck.html", metrics_summary={"total_vht_hours": 10.5}, geh_scores={"stn1": 2.1})
    assert sched_html_path.exists()
    assert "deck-hud-overlay" in sched_html_path.read_text(encoding="utf-8")
    assert sched_html_path.stat().st_size > 1000

    # 2. Render Building footprints map
    client = OvertureClient()
    fixtures = client.create_mock_santa_barbara_fixtures()
    cap_model = BuildingCapacityModel()
    classified_bldgs = cap_model.classify_and_assign_capacity(fixtures["buildings"])

    bldg_html_path = renderer.render_building_footprints_map(classified_bldgs, filename="test_bldgs.html")
    assert bldg_html_path.exists()
    assert bldg_html_path.stat().st_size > 1000
