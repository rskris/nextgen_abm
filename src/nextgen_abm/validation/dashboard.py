"""Interactive PyDeck visualization dashboard for building-level activity flows and agent schedules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import pydeck as pdk


class DashboardRenderer:
    """Renders interactive 3D geospatial dashboards for Santa Barbara County NextGen ABM."""

    SANTA_BARBARA_CENTER = {"longitude": -119.78, "latitude": 34.425, "zoom": 11.2, "pitch": 45}

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports/dashboard")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _convert_schedules_to_od_flows(
        sched_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert raw agent activity schedules into OD trip flows and activity hub points.

        Args:
            sched_df: DataFrame of activity schedules with person_id, location_id, lon, lat, start_hour, end_hour.

        Returns:
            Tuple of (od_flows DataFrame, hubs DataFrame)
        """
        trips = []
        for (_, _), group in sched_df.groupby(["household_id", "person_id"]):
            group = group.sort_values("start_hour")
            records = group.to_dict("records")
            for i in range(len(records) - 1):
                orig = records[i]
                dest = records[i + 1]
                orig_lon, orig_lat = float(orig["lon"]), float(orig["lat"])
                dest_lon, dest_lat = float(dest["lon"]), float(dest["lat"])

                # Only create trip if location or coordinates represent a movement
                if orig_lon != dest_lon or orig_lat != dest_lat or orig.get("location_id") != dest.get("location_id"):
                    trips.append({
                        "from_name": str(orig.get("location_id", "Origin")),
                        "to_name": str(dest.get("location_id", "Destination")),
                        "from_lon": orig_lon,
                        "from_lat": orig_lat,
                        "to_lon": dest_lon,
                        "to_lat": dest_lat,
                        "purpose": str(dest.get("act_type", "travel")),
                        "dep_hour": float(orig.get("end_hour", 0.0)),
                    })

        if not trips:
            return pd.DataFrame(), pd.DataFrame()

        trips_df = pd.DataFrame(trips)
        od_flows = (
            trips_df.groupby(["from_name", "to_name", "from_lon", "from_lat", "to_lon", "to_lat", "purpose"])
            .size()
            .reset_index(name="volume")
        )

        def _get_purpose_color(purpose: str) -> List[int]:
            p = str(purpose).lower()
            if "work" in p:
                return [239, 68, 68, 220]      # Red/Coral
            elif "school" in p or "class" in p:
                return [16, 185, 129, 220]     # Emerald Green
            elif "home" in p:
                return [59, 130, 246, 220]     # Azure/Blue
            elif "shop" in p or "grocery" in p:
                return [245, 158, 11, 220]     # Amber
            return [168, 85, 247, 220]         # Violet

        od_flows["color"] = od_flows["purpose"].apply(_get_purpose_color)
        od_flows["stroke_width"] = od_flows["volume"].apply(
            lambda v: float(max(3.0, min(14.0, v * 1.5)))
        )
        od_flows["name"] = od_flows.apply(
            lambda r: f"{r['from_name']} ➔ {r['to_name']} ({r['purpose']})", axis=1
        )

        # Extract activity hubs (origins & destinations)
        hubs = []
        for loc_id, group in sched_df.groupby("location_id"):
            first = group.iloc[0]
            visits = len(group)
            hubs.append({
                "location_id": str(loc_id),
                "lon": float(first["lon"]),
                "lat": float(first["lat"]),
                "visits": visits,
                "radius": float(max(120, min(500, visits * 20))),
            })
        hubs_df = pd.DataFrame(hubs)

        return od_flows, hubs_df

    def render_od_flow_map(
        self,
        od_flows: Optional[pd.DataFrame] = None,
        filename: str = "regional_commute_flows.html",
        metrics_summary: Optional[Dict[str, Any]] = None,
        geh_scores: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Render 3D ArcLayer & ScatterplotLayer showing activity travel flows and hubs.

        Supports both aggregated OD flows (from_lon, to_lon) and raw agent schedules
        (converting them on-the-fly to trip arcs and anchor hubs).
        """
        hubs_df = pd.DataFrame()

        if od_flows is None:
            # Default representative regional flows (e.g. Santa Maria -> Goleta/SB)
            od_flows = pd.DataFrame([
                {"name": "Santa Maria -> Goleta Tech", "from_name": "Santa Maria", "to_name": "Goleta Tech", "from_lon": -120.435, "from_lat": 34.935, "to_lon": -119.833, "to_lat": 34.435, "volume": 6200, "stroke_width": 12.0, "color": [240, 80, 50, 220], "purpose": "work"},
                {"name": "Santa Maria -> Downtown SB", "from_name": "Santa Maria", "to_name": "Downtown SB", "from_lon": -120.435, "from_lat": 34.935, "to_lon": -119.699, "to_lat": 34.419, "volume": 5800, "stroke_width": 11.0, "color": [240, 80, 50, 220], "purpose": "work"},
                {"name": "Lompoc -> Goleta Tech", "from_name": "Lompoc", "to_name": "Goleta Tech", "from_lon": -120.455, "from_lat": 34.640, "to_lon": -119.833, "to_lat": 34.435, "volume": 2400, "stroke_width": 6.0, "color": [255, 150, 0, 220], "purpose": "work"},
                {"name": "Goleta -> Downtown SB", "from_name": "Goleta", "to_name": "Downtown SB", "from_lon": -119.833, "from_lat": 34.435, "to_lon": -119.699, "to_lat": 34.419, "volume": 4200, "stroke_width": 9.0, "color": [40, 160, 220, 220], "purpose": "work"},
                {"name": "Isla Vista -> UCSB Campus", "from_name": "Isla Vista", "to_name": "UCSB Campus", "from_lon": -119.856, "from_lat": 34.411, "to_lon": -119.845, "to_lat": 34.415, "volume": 12000, "stroke_width": 14.0, "color": [16, 185, 129, 220], "purpose": "school"},
            ])
        elif "from_lon" not in od_flows.columns and ("person_id" in od_flows.columns or "act_id" in od_flows.columns):
            # Input is raw schedule data; convert to OD flows & hubs
            od_flows, hubs_df = self._convert_schedules_to_od_flows(od_flows)
            if od_flows.empty:
                return self.render_od_flow_map(od_flows=None, filename=filename)

        # Ensure stroke_width column exists to avoid dynamic JS expressions
        if "stroke_width" not in od_flows.columns:
            if "volume" in od_flows.columns:
                max_vol = float(od_flows["volume"].max())
                od_flows["stroke_width"] = od_flows["volume"].apply(
                    lambda v: float(max(3.0, min(14.0, v * 1.5 if max_vol <= 50.0 else v / 400.0)))
                )
            else:
                od_flows["stroke_width"] = 3.0

        layers = []

        # 1. Activity Hubs Layer (Scatterplot)
        if not hubs_df.empty:
            hub_layer = pdk.Layer(
                "ScatterplotLayer",
                data=hubs_df,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[15, 23, 42, 210],
                get_line_color=[56, 189, 248, 255],
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
            )
            layers.append(hub_layer)

        # 2. 3D Trip Flow Arcs Layer
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=od_flows,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="stroke_width",
            pickable=True,
            auto_highlight=True,
        )
        layers.append(arc_layer)

        view_state = pdk.ViewState(**self.SANTA_BARBARA_CENTER)
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{name}</b><br/>"
                        "Volume: <b>{volume}</b> trips<br/>"
                        "Activity Hub: <b>{location_id}</b> ({visits} visits)",
                "style": {"backgroundColor": "#0f172a", "color": "#f8fafc", "fontSize": "12px", "borderRadius": "6px"}
            },
            map_style="light",
        )

        out_path = self.output_dir / filename
        deck.to_html(str(out_path))

        # Inject sleek modern floating HUD card into the HTML
        self._inject_hud_overlay(
            html_path=out_path,
            od_flows=od_flows,
            hubs_df=hubs_df,
            metrics_summary=metrics_summary,
            geh_scores=geh_scores,
        )

        return out_path

    def render_flow_deck(
        self,
        flows_df: Optional[pd.DataFrame] = None,
        filename: str = "dashboard.html",
        metrics_summary: Optional[Dict[str, Any]] = None,
        geh_scores: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Alias for render_od_flow_map to render flow decks with optional metrics overlay."""
        return self.render_od_flow_map(
            od_flows=flows_df,
            filename=filename,
            metrics_summary=metrics_summary,
            geh_scores=geh_scores,
        )

    def render_building_footprints_map(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        filename: str = "building_landuse.html",
    ) -> Path:
        """Render building footprints 3D PolygonLayer extruded by height."""
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
            pickable=True,
        )

        view_state = pdk.ViewState(longitude=-119.85, latitude=34.415, zoom=14.0, pitch=45)
        deck = pdk.Deck(
            layers=[geojson_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nLand Use: {land_use}"},
            map_style="light",
        )
        deck.to_html(str(out_path))
        return out_path

    @staticmethod
    def _inject_hud_overlay(
        html_path: Path,
        od_flows: pd.DataFrame,
        hubs_df: pd.DataFrame,
        metrics_summary: Optional[Dict[str, Any]] = None,
        geh_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Inject a high-contrast floating analytics HUD overlay into the generated PyDeck HTML."""
        try:
            content = html_path.read_text(encoding="utf-8")
        except Exception:
            return

        total_trips = int(od_flows["volume"].sum()) if not od_flows.empty and "volume" in od_flows.columns else 0
        total_corridors = len(od_flows)
        hub_count = len(hubs_df) if not hubs_df.empty else 0

        # Metrics display values
        vht_text = ""
        corridor_flow_text = ""
        time_text = ""
        if metrics_summary:
            if "total_vht_hours" in metrics_summary:
                vht_text = f"{metrics_summary['total_vht_hours']:.1f} hrs"
            if "simulated_corridor_flow" in metrics_summary:
                corridor_flow_text = f"{metrics_summary['simulated_corridor_flow']:,.0f} veh/h"
            if "avg_travel_time_min" in metrics_summary:
                time_text = f"{metrics_summary['avg_travel_time_min']:.1f} min"

        # Validation status
        geh_badge = ""
        if geh_scores:
            passing = sum(1 for g in geh_scores.values() if g < 5.0)
            total = len(geh_scores)
            pct = (passing / max(1, total)) * 100.0
            color = "#10b981" if pct >= 60 else "#f59e0b"
            geh_badge = f"""
            <div style="margin-top: 10px; padding: 6px 10px; background: rgba(16, 185, 129, 0.15); border-left: 3px solid {color}; border-radius: 4px; font-size: 11px;">
              <strong>Caltrans PeMS GEH:</strong> {passing}/{total} Sensors Passing (&lt; 5.0)
            </div>
            """

        hud_html = f"""
        <div id="deck-hud-overlay" style="
          position: absolute;
          top: 20px;
          left: 20px;
          z-index: 1000;
          background: rgba(15, 23, 42, 0.92);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 12px;
          padding: 18px 22px;
          color: #f8fafc;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
          max-width: 360px;
          pointer-events: auto;
        ">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #38bdf8;">Santa Barbara County</span>
            <span style="font-size: 10px; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 6px; border-radius: 4px;">FIPS 06083</span>
          </div>
          <div style="font-size: 15px; font-weight: 700; color: #ffffff; margin-bottom: 12px;">NextGen ABM 3D Mobility Deck</div>

          <!-- KPI Cards Grid -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
            <div style="background: rgba(30, 41, 59, 0.8); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
              <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">Simulated Trips</div>
              <div style="font-size: 18px; font-weight: 700; color: #38bdf8;">{total_trips:,}</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
              <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">OD Corridors</div>
              <div style="font-size: 18px; font-weight: 700; color: #a78bfa;">{total_corridors}</div>
            </div>
            {f'''
            <div style="background: rgba(30, 41, 59, 0.8); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
              <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">Avg Travel Time</div>
              <div style="font-size: 16px; font-weight: 700; color: #34d399;">{time_text}</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
              <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">Corridor Flow</div>
              <div style="font-size: 16px; font-weight: 700; color: #f472b6;">{corridor_flow_text}</div>
            </div>
            ''' if corridor_flow_text else ''}
          </div>

          <!-- Color Legend -->
          <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px; font-size: 11px;">
            <div style="font-size: 10px; font-weight: 600; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase;">Activity Trajectories</div>
            <div style="display: flex; flex-direction: column; gap: 4px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgb(239, 68, 68); display: inline-block;"></span>
                <span>Work Commutes (Downtown SB / Tech Park)</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgb(16, 185, 129); display: inline-block;"></span>
                <span>Education (UCSB Campus &amp; School Escort)</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgb(59, 130, 246); display: inline-block;"></span>
                <span>Return Home (Evening Joint Activities)</span>
              </div>
              {f'''
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; border: 2px solid #38bdf8; background: #0f172a; display: inline-block;"></span>
                <span>Activity Hubs ({hub_count} anchor locations)</span>
              </div>
              ''' if hub_count > 0 else ''}
            </div>
          </div>

          {geh_badge}
        </div>
        """

        if "</head>" in content and "mapbox-gl.css" not in content:
            content = content.replace(
                "</head>",
                '    <link rel="stylesheet" href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" />\n  </head>',
            )
        if "</body>" in content:
            content = content.replace("</body>", f"{hud_html}\n</body>")
        html_path.write_text(content, encoding="utf-8")
