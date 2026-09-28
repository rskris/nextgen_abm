"""Overture Places (POIs) classification into ABM activity opportunities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import geopandas as gpd


@dataclass
class ActivityOpportunity:
    poi_id: str
    name: str
    activity_type: str  # "grocery", "dining", "retail", "leisure", "medical", "education_k12", "education_higher", "transit_hub"
    earliest_open_hr: float
    latest_close_hr: float
    typical_duration_min: float
    lon: float
    lat: float


class PlacesClassifier:
    """Classifies Overture POIs into standardized activity types and operating windows."""

    # Activity category mappings from Overture taxonomy
    CATEGORY_MAP = {
        "supermarket": ("grocery", 7.0, 22.0, 45.0),
        "grocery_store": ("grocery", 7.0, 22.0, 35.0),
        "convenience_store": ("grocery", 6.0, 24.0, 15.0),
        "restaurant": ("dining", 11.0, 22.0, 60.0),
        "fast_food_restaurant": ("dining", 6.0, 24.0, 25.0),
        "cafe": ("dining", 6.5, 18.0, 30.0),
        "bar": ("dining", 16.0, 26.0, 90.0),  # open until 2am (26.0)
        "department_store": ("retail", 10.0, 21.0, 60.0),
        "clothing_store": ("retail", 10.0, 20.0, 45.0),
        "park": ("leisure", 6.0, 21.0, 60.0),
        "fitness_centre": ("leisure", 5.5, 23.0, 60.0),
        "gym": ("leisure", 5.5, 23.0, 60.0),
        "beach": ("leisure", 6.0, 22.0, 90.0),
        "hospital": ("medical", 0.0, 24.0, 120.0),
        "clinic": ("medical", 8.0, 17.0, 45.0),
        "school": ("education_k12", 7.75, 15.5, 360.0),
        "college": ("education_higher", 8.0, 22.0, 120.0),
        "university": ("education_higher", 8.0, 22.0, 120.0),
        "train_station": ("transit_hub", 5.0, 23.0, 10.0),
        "bus_station": ("transit_hub", 5.0, 23.0, 10.0),
    }

    def classify_pois(self, pois_gdf: gpd.GeoDataFrame) -> List[ActivityOpportunity]:
        """Classify a GeoDataFrame of Overture Places into ActivityOpportunities."""
        opportunities = []
        for _, poi in pois_gdf.iterrows():
            poi_id = str(poi.get("id"))
            name = str(poi.get("name", "Unknown Location"))
            cat = str(poi.get("primary_category", "unclassified")).lower()

            # Find matching activity profile or fallback
            act_type, open_hr, close_hr, dur = self.CATEGORY_MAP.get(
                cat, ("retail", 9.0, 20.0, 40.0)
            )

            geom = poi.geometry
            lon = geom.x if hasattr(geom, "x") else -119.85
            lat = geom.y if hasattr(geom, "y") else 34.42

            opp = ActivityOpportunity(
                poi_id=poi_id,
                name=name,
                activity_type=act_type,
                earliest_open_hr=open_hr,
                latest_close_hr=close_hr,
                typical_duration_min=dur,
                lon=lon,
                lat=lat
            )
            opportunities.append(opp)

        return opportunities

    def get_destinations_by_activity_type(
        self,
        opportunities: List[ActivityOpportunity],
        activity_type: str
    ) -> List[ActivityOpportunity]:
        """Filter opportunities for a specific activity type (e.g. 'grocery' or 'dining')."""
        return [opp for opp in opportunities if opp.activity_type == activity_type]
