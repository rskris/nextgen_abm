"""Tests for Hierarchical Multi-Modal Network and Native LTM Meso-Simulator."""

import pytest
from nextgen_abm.traffic.network import HierarchicalMultiModalNetwork, LTMNode, LTMLink
from nextgen_abm.traffic.ltm import LTMMesoSimulator, TripVehicle
from nextgen_abm.config import get_config


def test_hierarchical_network_multimodal_routing():
    net = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network()

    assert len(net.nodes) >= 15
    assert len(net.links) >= 20

    # 1. Car routing from Goleta (Fairview) to Santa Barbara Downtown (Carrillo)
    car_path = net.find_shortest_path("node_us101_goleta_fairview", "node_us101_sb_carrillo", mode="car")
    assert len(car_path) > 0
    # Car path must use US-101 links
    assert any("link_us101" in l for l in car_path)

    # 2. Bike routing in UCSB / Isla Vista
    bike_path = net.find_shortest_path("node_iv_pardall_tunnel", "node_ucsb_north_hall", mode="bike")
    assert len(bike_path) > 0
    assert any("link_bike" in l for l in bike_path)

    # 3. Ensure cars cannot use bike paths
    car_on_bike_path = net.find_shortest_path("node_iv_pardall_tunnel", "node_ucsb_north_hall", mode="car")
    # There is no car road through Pardall Tunnel
    assert len(car_on_bike_path) == 0

    # 4. Centroid snapping test
    snapped_node = net.snap_building("bldg_test", (-119.85, 34.41), allowed_mode="bike")
    assert snapped_node in ["node_iv_pardall_tunnel", "node_ucsb_library_circle", "node_ucsb_ocean_rd"]


def test_ltm_simulation_execution_and_ev_energy():
    net = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network()
    sim = LTMMesoSimulator(network=net)

    # Create trips on US-101 (Carpinteria to Santa Barbara Downtown)
    path = net.find_shortest_path("node_us101_carpinteria", "node_us101_sb_carrillo", mode="car")
    assert len(path) > 0

    trips = [
        TripVehicle(
            trip_id=f"trip_car_{i}",
            person_id=f"person_{i}",
            mode="car",
            path=path,
            departure_time_sec=7.5 * 3600.0 + (i * 30.0),  # Morning peak departures around 07:30
            is_ev=(i % 2 == 0),  # 50% EV fleet
            battery_kwh=75.0,
        )
        for i in range(20)
    ]

    result = sim.run_simulation(trips)

    assert result.total_trips_completed == 20
    assert result.avg_travel_time_min > 0.0
    assert result.total_vkt_km > 0.0
    assert result.total_vht_hours > 0.0
    assert result.total_ev_energy_kwh > 0.0
    assert result.step_duration_ms > 0.0
