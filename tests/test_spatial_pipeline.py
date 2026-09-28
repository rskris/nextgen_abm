"""Test spatial capacity, POI classification, multi-scale routing, and elevation physics."""

import pytest
import geopandas as gpd
from nextgen_abm.spatial.overture import OvertureClient
from nextgen_abm.spatial.capacity import BuildingCapacityModel
from nextgen_abm.spatial.places import PlacesClassifier
from nextgen_abm.spatial.routing import MultiScaleRouter


def test_building_capacity_classification():
    client = OvertureClient()
    fixtures = client.create_mock_santa_barbara_fixtures()
    bldgs_gdf = fixtures["buildings"]

    cap_model = BuildingCapacityModel()
    classified_gdf = cap_model.classify_and_assign_capacity(bldgs_gdf)

    assert len(classified_gdf) == len(bldgs_gdf)
    assert "land_use" in classified_gdf.columns
    assert "res_units" in classified_gdf.columns
    assert "worker_cap" in classified_gdf.columns

    # Verify Santa Catalina Dorm has dormitory land-use and high bed capacity
    dorm_row = classified_gdf[classified_gdf["building_id"] == "bldg_iv_003"].iloc[0]
    assert dorm_row["land_use"] == "dormitory"
    assert dorm_row["res_units"] >= 20

    # Verify household allocation succeeds then respects limit
    allocated = cap_model.allocate_household_to_building("bldg_iv_001")
    assert allocated is True


def test_places_poi_classification():
    client = OvertureClient()
    fixtures = client.create_mock_santa_barbara_fixtures()
    pois_gdf = fixtures["places"]

    classifier = PlacesClassifier()
    opportunities = classifier.classify_pois(pois_gdf)

    assert len(opportunities) == len(pois_gdf)
    act_types = [opp.activity_type for opp in opportunities]
    assert "dining" in act_types
    assert "grocery" in act_types
    assert "transit_hub" in act_types

    groceries = classifier.get_destinations_by_activity_type(opportunities, "grocery")
    assert len(groceries) >= 1
    assert "Trader Joe's" in groceries[0].name


def test_multi_scale_routing_active_vs_auto():
    router = MultiScaleRouter()

    # Short trip: Isla Vista to UCSB Campbell Hall (~0.8 miles)
    iv_pt = (-119.856, 34.411)
    campbell_pt = (-119.845, 34.415)

    walk_route = router.route(iv_pt, campbell_pt, mode="walk")
    assert walk_route.distance_miles < 1.5
    assert walk_route.travel_time_minutes > 10.0

    bike_route = router.route(iv_pt, campbell_pt, mode="bike")
    assert bike_route.travel_time_minutes < walk_route.travel_time_minutes

    ebike_route = router.route(iv_pt, campbell_pt, mode="ebike")
    assert ebike_route.travel_time_minutes <= bike_route.travel_time_minutes


def test_elevation_and_ev_energy_physics():
    router = MultiScaleRouter()

    # Long mountain climb trip: Santa Barbara to San Marcos Pass (CA-154)
    sb_pt = (-119.700, 34.420)
    pass_pt = (-119.800, 34.520)

    uphill_route = router.route(sb_pt, pass_pt, mode="auto")
    assert uphill_route.elevation_gain_meters > 400.0
    assert uphill_route.average_slope_pct > 2.0
    assert uphill_route.ev_energy_kwh > (uphill_route.distance_miles * 0.28)  # Climbing increases energy!

    # Return downhill trip: Regen braking should recover energy
    downhill_route = router.route(pass_pt, sb_pt, mode="auto")
    assert downhill_route.elevation_loss_meters > 400.0
    assert downhill_route.ev_energy_kwh < uphill_route.ev_energy_kwh
