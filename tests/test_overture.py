"""Test Overture Maps GeoParquet querying and geometry processing."""

import pytest
import geopandas as gpd
from nextgen_abm.spatial.overture import OvertureClient, SANTA_BARBARA_BBOX, UCSB_ISLA_VISTA_BBOX


def test_overture_query_generation():
    client = OvertureClient()
    sql = client.build_building_query(bbox=UCSB_ISLA_VISTA_BBOX, limit=100)
    assert "theme=buildings" in sql
    assert f"bbox.xmin >= {UCSB_ISLA_VISTA_BBOX.min_x}" in sql
    assert "ST_AsText(geometry)" in sql
    assert "LIMIT 100" in sql


def test_overture_places_query_generation():
    client = OvertureClient()
    sql = client.build_places_query(bbox=SANTA_BARBARA_BBOX, limit=50)
    assert "theme=places" in sql
    assert "categories.primary AS primary_category" in sql
    assert "LIMIT 50" in sql


def test_mock_fixtures_generation():
    client = OvertureClient()
    fixtures = client.create_mock_santa_barbara_fixtures()
    assert "buildings" in fixtures
    assert "places" in fixtures

    bldgs = fixtures["buildings"]
    assert isinstance(bldgs, gpd.GeoDataFrame)
    assert len(bldgs) > 10
    assert "bldg_ucsb_campbell" in bldgs["id"].values
    assert "bldg_iv_001" in bldgs["id"].values
    assert bldgs.crs.to_string() == "EPSG:4326"

    places = fixtures["places"]
    assert isinstance(places, gpd.GeoDataFrame)
    assert "Goleta Amtrak Station" in places["name"].values
