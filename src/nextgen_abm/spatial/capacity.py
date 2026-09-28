"""Building capacity modeling and land-use parcel classification for Santa Barbara County."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import geopandas as gpd


@dataclass
class BuildingCapacity:
    building_id: str
    building_name: str
    land_use: str  # "residential_sfr", "residential_mfr", "dormitory", "commercial_office", "retail", "dining", "institutional_edu", "hospital"
    residential_units: int
    worker_capacity: int
    visitor_capacity: int
    height_m: float
    num_floors: int
    area_sqm: float
    current_households: int = 0
    current_workers: int = 0


class BuildingCapacityModel:
    """Assigns and tracks capacity for residential, employment, and activity destinations."""

    # Default square footage multipliers (sq meters per person)
    SQM_PER_OFFICE_WORKER = 25.0
    SQM_PER_RETAIL_WORKER = 45.0
    SQM_PER_RESIDENTIAL_UNIT = 80.0

    def __init__(self):
        self.capacities: Dict[str, BuildingCapacity] = {}

    def classify_and_assign_capacity(self, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process building GeoDataFrame and compute capacity attributes."""
        records = []
        for _, bldg in buildings_gdf.iterrows():
            bldg_id = str(bldg.get("id"))
            name = str(bldg.get("name", "Unknown Building"))
            b_class = str(bldg.get("class", "unclassified")).lower()
            subtype = str(bldg.get("subtype", "")).lower()
            floors = int(bldg.get("num_floors", 1)) if pd.notna(bldg.get("num_floors")) else 1
            height = float(bldg.get("height", 6.0)) if pd.notna(bldg.get("height")) else 6.0

            # Approximate footprint area in square meters (approx 1 deg lon/lat ~ 111km)
            geom = bldg.geometry
            if geom and hasattr(geom, "area"):
                footprint_sqm = geom.area * (111000 * 92000)  # scale lat/lon to meters in SB
            else:
                footprint_sqm = 150.0  # default footprint

            total_usable_sqm = footprint_sqm * floors

            # Determine land-use and capacity
            if "dorm" in subtype or "residence hall" in name.lower():
                land_use = "dormitory"
                res_units = max(20, int(total_usable_sqm / 30.0))  # student beds
                worker_cap = max(2, int(res_units * 0.05))
                visitor_cap = 50
            elif "multi_family" in subtype or b_class == "residential" and floors > 2:
                land_use = "residential_mfr"
                res_units = max(4, int(total_usable_sqm / self.SQM_PER_RESIDENTIAL_UNIT))
                worker_cap = 1
                visitor_cap = 10
            elif b_class == "residential":
                land_use = "residential_sfr"
                res_units = 1
                worker_cap = 0
                visitor_cap = 5
            elif "hospital" in subtype or "hospital" in name.lower():
                land_use = "hospital"
                res_units = 0
                worker_cap = max(50, int(total_usable_sqm / 35.0))
                visitor_cap = 500
            elif "lecture" in subtype or "school" in subtype or "library" in subtype or b_class == "institutional":
                land_use = "institutional_edu"
                res_units = 0
                worker_cap = max(10, int(total_usable_sqm / 40.0))
                visitor_cap = max(100, int(total_usable_sqm / 5.0))  # lecture halls accommodate high visitor volume
            elif "retail" in subtype or "supermarket" in subtype:
                land_use = "retail"
                res_units = 0
                worker_cap = max(5, int(total_usable_sqm / self.SQM_PER_RETAIL_WORKER))
                visitor_cap = max(50, int(total_usable_sqm / 8.0))
            else:
                land_use = "commercial_office"
                res_units = 0
                worker_cap = max(4, int(total_usable_sqm / self.SQM_PER_OFFICE_WORKER))
                visitor_cap = 20

            cap = BuildingCapacity(
                building_id=bldg_id,
                building_name=name,
                land_use=land_use,
                residential_units=res_units,
                worker_capacity=worker_cap,
                visitor_capacity=visitor_cap,
                height_m=height,
                num_floors=floors,
                area_sqm=total_usable_sqm
            )
            self.capacities[bldg_id] = cap

            records.append({
                "building_id": bldg_id,
                "name": name,
                "land_use": land_use,
                "res_units": res_units,
                "worker_cap": worker_cap,
                "visitor_cap": visitor_cap,
                "geometry": geom
            })

        return gpd.GeoDataFrame(records, crs=buildings_gdf.crs)

    def allocate_household_to_building(self, building_id: str) -> bool:
        """Attempt to place a household into a building; returns True if space available."""
        cap = self.capacities.get(building_id)
        if not cap or cap.residential_units <= 0:
            return False
        if cap.current_households < cap.residential_units:
            cap.current_households += 1
            return True
        return False
