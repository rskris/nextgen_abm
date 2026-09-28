"""Interactive PyDeck visualization dashboard for building-level activity flows and agent schedules."""

from pathlib import Path
from typing import Dict, List, Optional
import pydeck as pdk
import pandas as pd
import geopandas as gpd


class DashboardRenderer:
    """Renders interactive 3D geospatial dashboards for Santa Barbara County NextGen ABM."""

    SANTA_BARBARA_CENTER = {"longitude": -119.80, "latitude": 34.43, "zoom": 10.5, "pitch": 45}

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports/dashboard")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_od_flow_map(
        self,
        od_flows: Optional[pd.DataFrame] = None,
        filename: str = "regional_commute_flows.html"
    ) -> Path:
        """Render 3D ArcLayer showing inter-regional commute flows (e.g. Santa Maria to Goleta/SB)."""
        if od_flows is None:
            # Default representative flows
            od_flows = pd.DataFrame([
                {"name": "Santa Maria -> Goleta Tech", "from_lon": -120.435, "from_lat": 34.935, "to_lon": -119.833, "to_lat": 34.435, "volume": 6200, "color": [240, 80, 50]},
                {"name": "Santa Maria -> Downtown SB", "from_lon": -120.435, "from_lat": 34.935, "to_lon": -119.699, "to_lat": 34.419, "volume": 5800, "color": [240, 80, 50]},
                {"name": "Lompoc -> Goleta Tech", "from_lon": -120.455, "from_lat": 34.640, "to_lon": -119.833, "to_lat": 34.435, "volume": 2400, "color": [255, 150, 0]},
                {"name": "Goleta -> Downtown SB", "from_lon": -119.833, "from_lat": 34.435, "to_lon": -119.699, "to_lat": 34.419, "volume": 4200, "color": [40, 160, 220]},
                {"name": "Isla Vista -> UCSB Campus", "from_lon": -119.856, "from_lat": 34.411, "to_lon": -119.845, "to_lat": 34.415, "volume": 12000, "color": [0, 200, 100]},
            ])

        # PyDeck ArcLayer
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=od_flows,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="volume / 500",
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(**self.SANTA_BARBARA_CENTER)
        deck = pdk.Deck(
            layers=[arc_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nDaily Volume: {volume:,} travelers"},
            map_style="light"
        )

        out_path = self.output_dir / filename
        deck.to_html(str(out_path))
        return out_path

    def render_building_footprints_map(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        filename: str = "building_landuse.html"
    ) -> Path:
        """Render building footprints 3D PolygonLayer extruded by height."""
        # Convert GeoDataFrame to centroid/bounds format for PyDeck GeoJsonLayer
        out_path = self.output_dir / filename
        geojson_data = buildings_gdf.__geo_interface__

        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.8,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="properties.height_m * 2",
            get_fill_color="[200, 100, 100, 180]",
            get_line_color="[255, 255, 255]",
            pickable=True
        )

        view_state = pdk.ViewState(longitude=-119.85, latitude=34.415, zoom=14.0, pitch=45)
        deck = pdk.Deck(
            layers=[geojson_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nLand Use: {land_use}"},
            map_style="light"
        )
        deck.to_html(str(out_path))
        return out_path
