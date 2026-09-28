"""Test GTFS transit feed loading and transit skims."""

import pytest
import geopandas as gpd
from nextgen_abm.spatial.gtfs import GTFSDataLoader, TransitSkimResult


def test_gtfs_tables_generation():
    loader = GTFSDataLoader()
    gtfs_tables = loader.generate_synthetic_sb_gtfs()

    assert "routes" in gtfs_tables
    assert "stops" in gtfs_tables
    assert "trips" in gtfs_tables

    routes = gtfs_tables["routes"]
    assert "MTD_24x" in routes["route_id"].values
    assert "CAE_SM_GOLETA" in routes["route_id"].values
    assert "SURFLINER" in routes["route_id"].values

    stops = gtfs_tables["stops"]
    assert "stop_ucsb_bus_loop" in stops["stop_id"].values
    assert "stop_goleta_amtrak" in stops["stop_id"].values

    stops_gdf = loader.get_stops_geodataframe(stops)
    assert isinstance(stops_gdf, gpd.GeoDataFrame)
    assert stops_gdf.crs.to_string() == "EPSG:4326"


def test_transit_skim_student_free_fare():
    loader = GTFSDataLoader()

    # From Isla Vista to Downtown Santa Barbara
    iv_coords = (-119.855, 34.412)
    sb_coords = (-119.702, 34.422)

    # Student should ride fare-free on MTD 24x
    student_skim = loader.lookup_transit_skim(iv_coords, sb_coords, departure_hour=9.0, is_student=True)
    assert student_skim is not None
    assert student_skim.fare_dollars == 0.0
    assert student_skim.route_id == "MTD_24x"
    assert student_skim.total_time_min > 0

    # Non-student should pay standard fare
    adult_skim = loader.lookup_transit_skim(iv_coords, sb_coords, departure_hour=9.0, is_student=False)
    assert adult_skim is not None
    assert adult_skim.fare_dollars == 1.75


def test_clean_air_express_commute():
    loader = GTFSDataLoader()

    # Santa Maria to Goleta Tech
    sm_coords = (-120.435, 34.952)
    goleta_coords = (-119.812, 34.434)

    commute_skim = loader.lookup_transit_skim(sm_coords, goleta_coords, departure_hour=7.0, is_student=False)
    assert commute_skim is not None
    assert commute_skim.route_id == "CAE_SM_GOLETA"
    assert commute_skim.total_time_min >= 60.0  # ~65-mile corridor
    assert commute_skim.fare_dollars == 7.00
