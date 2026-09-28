"""GTFS transit feed loader and transit skim provider for Santa Barbara County."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


@dataclass
class TransitStop:
    stop_id: str
    agency_id: str
    stop_name: str
    lat: float
    lon: float
    routes_served: List[str]


@dataclass
class TransitSkimResult:
    origin_stop_id: str
    dest_stop_id: str
    route_id: str
    departure_time: str
    arrival_time: str
    in_vehicle_time_min: float
    wait_time_min: float
    total_time_min: float
    fare_dollars: float


class GTFSDataLoader:
    """Loads and indexes GTFS transit schedules for Santa Barbara MTD, Clean Air Express, and Amtrak Surfliner."""

    AGENCY_MTD = "SB_MTD"
    AGENCY_CLEAN_AIR = "CLEAN_AIR_EXPRESS"
    AGENCY_AMTRAK = "AMTRAK_PACIFIC_SURFLINER"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/gtfs")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_sb_gtfs(self) -> Dict[str, pd.DataFrame]:
        """Generate representative GTFS tables for Santa Barbara County key transit routes."""
        # 1. Routes
        routes = [
            {"agency_id": self.AGENCY_MTD, "route_id": "MTD_24x", "route_short_name": "24x", "route_long_name": "UCSB Express", "route_type": 3},
            {"agency_id": self.AGENCY_MTD, "route_id": "MTD_11", "route_short_name": "11", "route_long_name": "State - Hollister - UCSB", "route_type": 3},
            {"agency_id": self.AGENCY_MTD, "route_id": "MTD_27", "route_short_name": "27", "route_long_name": "Isla Vista Shuttle", "route_type": 3},
            {"agency_id": self.AGENCY_CLEAN_AIR, "route_id": "CAE_SM_GOLETA", "route_short_name": "CAE-1", "route_long_name": "Santa Maria to Goleta Express", "route_type": 3},
            {"agency_id": self.AGENCY_AMTRAK, "route_id": "SURFLINER", "route_short_name": "Surfliner", "route_long_name": "Pacific Surfliner Regional Rail", "route_type": 2},
        ]

        # 2. Stops
        stops = [
            # UCSB & Isla Vista
            {"stop_id": "stop_ucsb_bus_loop", "stop_name": "UCSB Bus Loop (North Hall)", "stop_lat": 34.4150, "stop_lon": -119.8455, "agency_id": self.AGENCY_MTD},
            {"stop_id": "stop_iv_pardall", "stop_name": "Isla Vista (Pardall & Embarcadero)", "stop_lat": 34.4128, "stop_lon": -119.8550, "agency_id": self.AGENCY_MTD},
            {"stop_id": "stop_camino_real", "stop_name": "Camino Real Marketplace", "stop_lat": 34.4285, "stop_lon": -119.8735, "agency_id": self.AGENCY_MTD},
            # Goleta & Tech
            {"stop_id": "stop_goleta_amtrak", "stop_name": "Goleta Amtrak Station (La Patera)", "stop_lat": 34.4390, "stop_lon": -119.8350, "agency_id": self.AGENCY_AMTRAK},
            {"stop_id": "stop_goleta_tech", "stop_name": "Hollister & Patterson (Tech Corridor)", "stop_lat": 34.4340, "stop_lon": -119.8120, "agency_id": self.AGENCY_MTD},
            # Santa Barbara Core
            {"stop_id": "stop_sb_transit_ctr", "stop_name": "Santa Barbara Transit Center (Chapala)", "stop_lat": 34.4225, "stop_lon": -119.7020, "agency_id": self.AGENCY_MTD},
            {"stop_id": "stop_sb_amtrak", "stop_name": "Santa Barbara Amtrak Station (State St)", "stop_lat": 34.4140, "stop_lon": -119.6925, "agency_id": self.AGENCY_AMTRAK},
            # Santa Maria (North County)
            {"stop_id": "stop_sm_transit_ctr", "stop_name": "Santa Maria Transit Center", "stop_lat": 34.9520, "stop_lon": -120.4350, "agency_id": self.AGENCY_CLEAN_AIR},
        ]

        # 3. Scheduled Trips / Timetables
        # Representative travel times between key pairs
        trips = [
            {"trip_id": "trip_24x_am", "route_id": "MTD_24x", "direction_id": 0, "headway_min": 15},
            {"trip_id": "trip_11_am", "route_id": "MTD_11", "direction_id": 0, "headway_min": 20},
            {"trip_id": "trip_cae_sm_am", "route_id": "CAE_SM_GOLETA", "direction_id": 0, "headway_min": 30},
            {"trip_id": "trip_surfliner_am", "route_id": "SURFLINER", "direction_id": 0, "headway_min": 120},
        ]

        return {
            "routes": pd.DataFrame(routes),
            "stops": pd.DataFrame(stops),
            "trips": pd.DataFrame(trips)
        }

    def get_stops_geodataframe(self, stops_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert GTFS stops dataframe to GeoDataFrame."""
        geoms = [Point(lon, lat) for lon, lat in zip(stops_df["stop_lon"], stops_df["stop_lat"])]
        return gpd.GeoDataFrame(stops_df, geometry=geoms, crs="EPSG:4326")

    def lookup_transit_skim(
        self,
        origin_coords: Tuple[float, float],  # (lon, lat)
        dest_coords: Tuple[float, float],    # (lon, lat)
        departure_hour: float = 8.0,
        is_student: bool = False
    ) -> Optional[TransitSkimResult]:
        """Estimate transit travel time, wait time, and fare between two points."""
        orig_lon, orig_lat = origin_coords
        dest_lon, dest_lat = dest_coords

        # Approximate distance in miles
        approx_dist_mi = (((orig_lon - dest_lon) * 55)**2 + ((orig_lat - dest_lat) * 69)**2)**0.5

        if approx_dist_mi < 0.2:
            # Too short for transit, purely walking
            return None

        # Regional commute from Santa Maria (lat > 34.8) to South Coast (lat < 34.5)
        if orig_lat > 34.8 and dest_lat < 34.5:
            # Clean Air Express Commuter Bus or Pacific Surfliner Rail
            in_veh_time = 75.0  # ~75 minutes along US-101 Gaviota corridor
            wait_time = 15.0
            fare = 7.00
            return TransitSkimResult(
                origin_stop_id="stop_sm_transit_ctr",
                dest_stop_id="stop_goleta_tech",
                route_id="CAE_SM_GOLETA",
                departure_time=f"{int(departure_hour):02d}:00",
                arrival_time=f"{int(departure_hour+1):02d}:30",
                in_vehicle_time_min=in_veh_time,
                wait_time_min=wait_time,
                total_time_min=in_veh_time + wait_time,
                fare_dollars=fare
            )

        # Isla Vista / UCSB to Downtown Santa Barbara (MTD Line 24x Express)
        if -119.87 <= orig_lon <= -119.83 and -119.72 <= dest_lon <= -119.68:
            in_veh_time = 25.0
            wait_time = 7.5
            fare = 0.0 if is_student else 1.75  # UCSB students ride free with student ID!
            return TransitSkimResult(
                origin_stop_id="stop_ucsb_bus_loop",
                dest_stop_id="stop_sb_transit_ctr",
                route_id="MTD_24x",
                departure_time=f"{int(departure_hour):02d}:00",
                arrival_time=f"{int(departure_hour):02d}:32",
                in_vehicle_time_min=in_veh_time,
                wait_time_min=wait_time,
                total_time_min=in_veh_time + wait_time,
                fare_dollars=fare
            )

        # Local Goleta / Isla Vista campus shuttle (MTD Line 27 / 28)
        if -119.88 <= orig_lon <= -119.83 and -119.88 <= dest_lon <= -119.83:
            in_veh_time = 10.0
            wait_time = 5.0
            fare = 0.0 if is_student else 1.75
            return TransitSkimResult(
                origin_stop_id="stop_iv_pardall",
                dest_stop_id="stop_ucsb_bus_loop",
                route_id="MTD_27",
                departure_time=f"{int(departure_hour):02d}:00",
                arrival_time=f"{int(departure_hour):02d}:15",
                in_vehicle_time_min=in_veh_time,
                wait_time_min=wait_time,
                total_time_min=in_veh_time + wait_time,
                fare_dollars=fare
            )

        # Default standard MTD bus travel profile
        in_veh_time = max(12.0, approx_dist_mi * 3.5)
        wait_time = 10.0
        fare = 0.0 if is_student else 1.75
        return TransitSkimResult(
            origin_stop_id="generic_origin_stop",
            dest_stop_id="generic_dest_stop",
            route_id="MTD_11",
            departure_time=f"{int(departure_hour):02d}:00",
            arrival_time=f"{int(departure_hour):02d}:{int(in_veh_time):02d}",
            in_vehicle_time_min=in_veh_time,
            wait_time_min=wait_time,
            total_time_min=in_veh_time + wait_time,
            fare_dollars=fare
        )
