"""DuckDB-powered Overture Maps GeoParquet ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt, box, Polygon


from nextgen_abm.config import get_config


@dataclass
class BoundingBox:
    """Geographic bounding box in EPSG:4326 (lon/lat)."""
    min_x: float  # West longitude
    min_y: float  # South latitude
    max_x: float  # East longitude
    max_y: float  # North latitude

    def as_tuple(self) -> tuple:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def to_shapely_polygon(self) -> Polygon:
        return box(self.min_x, self.min_y, self.max_x, self.max_y)


def get_county_bbox() -> BoundingBox:
    """Get county bounding box from active configuration."""
    cfg = get_config().spatial
    return BoundingBox(min_x=cfg.min_lon, min_y=cfg.min_lat, max_x=cfg.max_lon, max_y=cfg.max_lat)


# Santa Barbara County bounding box (including South Coast, Santa Ynez, and Santa Maria)
SANTA_BARBARA_BBOX = get_county_bbox()


# Focused UCSB & Isla Vista sub-bounding box for rapid prototyping
UCSB_ISLA_VISTA_BBOX = BoundingBox(
    min_x=-119.89,
    min_y=34.40,
    max_x=-119.83,
    max_y=34.43,
)


class OvertureClient:
    """Client for querying Overture Maps Foundation GeoParquet using DuckDB."""

    DEFAULT_RELEASE = "2024-08-20.0"
    OVERTURE_S3_BASE = "s3://overturemaps-us-west-2/release"

    def __init__(
        self,
        db_path: str = ":memory:",
        cache_dir: Optional[Path] = None,
        release: str = DEFAULT_RELEASE,
        s3_bucket: Optional[str] = None,
    ):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/overture_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.release = release
        self.s3_bucket = s3_bucket or "overturemaps-us-west-2"
        self.con = duckdb.connect(database=self.db_path)
        self._init_duckdb_extensions()


    def _init_duckdb_extensions(self):
        """Install and load required DuckDB extensions for remote GeoParquet."""
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except Exception:
            # Already installed or loaded
            pass

        try:
            self.con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass

        # Configure S3 anonymous access for public Overture buckets
        try:
            self.con.execute("SET s3_region='us-west-2';")
            self.con.execute("SET s3_url_style='path';")
        except Exception:
            pass

    def build_building_query(
        self,
        bbox: BoundingBox = SANTA_BARBARA_BBOX,
        limit: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> str:
        """Construct SQL to extract building footprints and attributes."""
        source = source_path or f"{self.OVERTURE_S3_BASE}/{self.release}/theme=buildings/type=building/*"
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
        SELECT 
            id,
            names.primary AS name,
            height,
            num_floors,
            class,
            subtype,
            bbox.xmin AS min_lon,
            bbox.ymin AS min_lat,
            bbox.xmax AS max_lon,
            bbox.ymax AS max_lat,
            ST_AsText(geometry) AS wkt_geom
        FROM read_parquet('{source}', hive_partitioning=1)
        WHERE bbox.xmin >= {bbox.min_x} AND bbox.xmax <= {bbox.max_x}
          AND bbox.ymin >= {bbox.min_y} AND bbox.ymax <= {bbox.max_y}
        {limit_clause}
        """
        return sql

    def build_places_query(
        self,
        bbox: BoundingBox = SANTA_BARBARA_BBOX,
        limit: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> str:
        """Construct SQL to extract places/POIs for activity destination modeling."""
        source = source_path or f"{self.OVERTURE_S3_BASE}/{self.release}/theme=places/type=place/*"
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
        SELECT 
            id,
            names.primary AS name,
            categories.primary AS primary_category,
            categories.alternate AS alt_categories,
            confidence,
            addresses[1].freeform AS address,
            bbox.xmin AS min_lon,
            bbox.ymin AS min_lat,
            ST_AsText(geometry) AS wkt_geom
        FROM read_parquet('{source}', hive_partitioning=1)
        WHERE bbox.xmin >= {bbox.min_x} AND bbox.xmax <= {bbox.max_x}
          AND bbox.ymin >= {bbox.min_y} AND bbox.ymax <= {bbox.max_y}
        {limit_clause}
        """
        return sql

    def query_to_geodataframe(self, sql: str) -> gpd.GeoDataFrame:
        """Execute query and convert resulting WKT geometry into a GeoPandas GeoDataFrame."""
        df = self.con.execute(sql).df()
        if df.empty or "wkt_geom" not in df.columns:
            return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

        geometries = [wkt.loads(geom) if geom else None for geom in df["wkt_geom"]]
        gdf = gpd.GeoDataFrame(df.drop(columns=["wkt_geom"]), geometry=geometries, crs="EPSG:4326")
        return gdf

    def query_buildings(self, bbox: BoundingBox = SANTA_BARBARA_BBOX, limit: Optional[int] = None) -> gpd.GeoDataFrame:
        """Query and return building footprints as a GeoDataFrame."""
        sql = self.build_building_query(bbox=bbox, limit=limit)
        return self.query_to_geodataframe(sql)

    def query_places(self, bbox: BoundingBox = SANTA_BARBARA_BBOX, limit: Optional[int] = None) -> gpd.GeoDataFrame:
        """Query and return places/POIs as a GeoDataFrame."""
        sql = self.build_places_query(bbox=bbox, limit=limit)
        return self.query_to_geodataframe(sql)


    def create_mock_santa_barbara_fixtures(self) -> Dict[str, gpd.GeoDataFrame]:
        """Generate high-fidelity building and POI fixtures for local offline development/testing.
        
        Includes representative buildings for:
        - Isla Vista student housing
        - UCSB Campus (Campbell Hall, Library, BioEng, Dining)
        - Goleta Tech & Commercial Center (Camino Real Marketplace, Deckers, Yardi)
        - Santa Barbara Downtown (State St retail, Courthouse, Cottage Hospital)
        - Santa Maria Residential and Agricultural centers
        """
        buildings_data = [
            # Isla Vista Student Housing
            {"id": "bldg_iv_001", "name": "6624 Del Playa Dr", "height": 7.5, "num_floors": 2, "class": "residential", "subtype": "multi_family", "lon": -119.856, "lat": 34.411},
            {"id": "bldg_iv_002", "name": "6510 Embarcadero del Norte", "height": 11.0, "num_floors": 3, "class": "residential", "subtype": "multi_family", "lon": -119.853, "lat": 34.413},
            {"id": "bldg_iv_003", "name": "Santa Catalina Residence Hall", "height": 38.0, "num_floors": 11, "class": "residential", "subtype": "dormitory", "lon": -119.866, "lat": 34.417},
            # UCSB Campus Key Buildings
            {"id": "bldg_ucsb_campbell", "name": "Campbell Hall", "height": 14.0, "num_floors": 2, "class": "institutional", "subtype": "lecture_hall", "lon": -119.845, "lat": 34.415},
            {"id": "bldg_ucsb_library", "name": "Davidson Library", "height": 32.0, "num_floors": 8, "class": "institutional", "subtype": "library", "lon": -119.846, "lat": 34.414},
            {"id": "bldg_ucsb_bioeng", "name": "BioEngineering Building", "height": 18.0, "num_floors": 4, "class": "institutional", "subtype": "laboratory", "lon": -119.842, "lat": 34.414},
            {"id": "bldg_ucsb_reccen", "name": "UCSB Recreation Center", "height": 12.0, "num_floors": 2, "class": "recreation", "subtype": "gym", "lon": -119.851, "lat": 34.416},
            # Goleta Commercial & Tech
            {"id": "bldg_goleta_costco", "name": "Costco Camino Real", "height": 9.0, "num_floors": 1, "class": "commercial", "subtype": "retail_warehouse", "lon": -119.873, "lat": 34.428},
            {"id": "bldg_goleta_yardi", "name": "Yardi Systems HQ", "height": 15.0, "num_floors": 3, "class": "commercial", "subtype": "office", "lon": -119.833, "lat": 34.435},
            # Santa Barbara South Coast
            {"id": "bldg_sb_cottage", "name": "Santa Barbara Cottage Hospital", "height": 22.0, "num_floors": 5, "class": "institutional", "subtype": "hospital", "lon": -119.719, "lat": 34.436},
            {"id": "bldg_sb_state_st", "name": "Paseo Nuevo Shops", "height": 12.0, "num_floors": 2, "class": "commercial", "subtype": "retail", "lon": -119.699, "lat": 34.419},
            # Santa Maria (North County Commute Anchor)
            {"id": "bldg_sm_res_001", "name": "Santa Maria Southland Estates", "height": 6.5, "num_floors": 2, "class": "residential", "subtype": "single_family", "lon": -120.435, "lat": 34.935},
            {"id": "bldg_sm_res_002", "name": "Orcutt Hills Residences", "height": 5.5, "num_floors": 1, "class": "residential", "subtype": "single_family", "lon": -120.442, "lat": 34.865},
        ]

        pois_data = [
            {"id": "poi_001", "name": "Freebirds World Burrito", "primary_category": "fast_food_restaurant", "lon": -119.855, "lat": 34.412},
            {"id": "poi_002", "name": "Trader Joe's Goleta", "primary_category": "supermarket", "lon": -119.818, "lat": 34.439},
            {"id": "poi_003", "name": "Target Goleta", "primary_category": "department_store", "lon": -119.824, "lat": 34.437},
            {"id": "poi_004", "name": "Goleta Amtrak Station", "primary_category": "train_station", "lon": -119.835, "lat": 34.439},
            {"id": "poi_005", "name": "Santa Barbara Amtrak Station", "primary_category": "train_station", "lon": -119.692, "lat": 34.414},
        ]

        # Convert to GeoDataFrames
        bldg_geoms = [
            box(b["lon"] - 0.0003, b["lat"] - 0.0003, b["lon"] + 0.0003, b["lat"] + 0.0003)
            for b in buildings_data
        ]
        bldg_df = pd.DataFrame(buildings_data).drop(columns=["lon", "lat"])
        bldg_gdf = gpd.GeoDataFrame(bldg_df, geometry=bldg_geoms, crs="EPSG:4326")

        poi_geoms = [
            box(p["lon"] - 0.00005, p["lat"] - 0.00005, p["lon"] + 0.00005, p["lat"] + 0.00005)
            for p in pois_data
        ]
        poi_df = pd.DataFrame(pois_data).drop(columns=["lon", "lat"])
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry=poi_geoms, crs="EPSG:4326")

        return {"buildings": bldg_gdf, "places": poi_gdf}
