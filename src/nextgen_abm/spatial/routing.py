"""Multi-scale network router and elevation gradient energy modeling for Santa Barbara County."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass
class RouteProfile:
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    mode: str  # "walk", "bike", "ebike", "auto", "transit"
    distance_miles: float
    travel_time_minutes: float
    elevation_gain_meters: float
    elevation_loss_meters: float
    average_slope_pct: float
    ev_energy_kwh: float  # Energy consumption if driven by EV


class MultiScaleRouter:
    """Multi-scale router handling active transport, road corridors, and elevation impedance."""

    # Earth radius in miles
    EARTH_RADIUS_MILES = 3958.8

    # Network circuity factors (actual network distance / straight-line distance)
    CIRCUITY_ACTIVE = 1.25
    CIRCUITY_ROAD = 1.22

    # Free-flow speeds (mph)
    SPEED_WALK_MPH = 3.1
    SPEED_BIKE_MPH = 11.0
    SPEED_EBIKE_MPH = 18.0
    SPEED_AUTO_LOCAL_MPH = 25.0
    SPEED_AUTO_HIGHWAY_MPH = 65.0

    # EV physics parameters
    EV_BASE_KWH_PER_MILE = 0.28  # Tesla Model 3 / Chevy Bolt baseline
    EV_CURB_WEIGHT_KG = 1750.0   # kg
    GRAVITY = 9.81               # m/s^2
    REGEN_EFFICIENCY = 0.65      # 65% regen braking energy recovery

    def calculate_haversine_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two (lon, lat) points in miles."""
        lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
        lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return self.EARTH_RADIUS_MILES * c

    def estimate_elevation_profile(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
        """Estimate elevation at points and compute elevation gain, loss, and avg slope.
        
        Santa Barbara regional topographic profile:
        - Sea level / coast (Isla Vista, Santa Barbara Waterfront): ~5-15m
        - Goleta / Foothills: ~30-80m
        - San Marcos Pass (CA-154 mountain summit): ~685m (2,247 ft)
        - Santa Ynez Valley (Solvang): ~150m
        - Santa Maria Valley: ~70m
        """
        elev1 = self._point_elevation_meters(p1)
        elev2 = self._point_elevation_meters(p2)

        elev_diff = elev2 - elev1
        gain = max(0.0, elev_diff)
        loss = max(0.0, -elev_diff)

        dist_m = self.calculate_haversine_distance(p1, p2) * 1609.34
        avg_slope_pct = (abs(elev_diff) / max(10.0, dist_m)) * 100.0

        return gain, loss, avg_slope_pct

    def _point_elevation_meters(self, pt: Tuple[float, float]) -> float:
        """Approximates Santa Barbara elevation based on latitude/longitude topography."""
        lon, lat = pt
        # Mountain ridge around CA-154 San Marcos Pass (lat ~34.50 to 34.54, lon ~-119.80)
        if 34.48 <= lat <= 34.55 and -119.85 <= lon <= -119.75:
            return 650.0  # mountain summit
        if lat > 34.85:
            return 75.0   # Santa Maria plain
        if -119.88 <= lon <= -119.82 and 34.40 <= lat <= 34.43:
            return 12.0   # Isla Vista / UCSB coastal mesa
        if -119.75 <= lon <= -119.65 and 34.40 <= lat <= 34.45:
            return 25.0   # Santa Barbara downtown
        return 45.0       # General rolling terrain

    def route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        mode: str,
        departure_hour: float = 8.0,
        congestion_factor: float = 1.0
    ) -> RouteProfile:
        """Compute distance, travel time, and EV energy profile between two points."""
        straight_dist = self.calculate_haversine_distance(origin, destination)
        gain_m, loss_m, slope_pct = self.estimate_elevation_profile(origin, destination)

        circuity = self.CIRCUITY_ACTIVE if mode in ["walk", "bike", "ebike"] else self.CIRCUITY_ROAD
        net_dist_miles = straight_dist * circuity

        # Adjust bicycle speed by slope (Tobler hill-climbing penalty)
        if mode == "walk":
            speed = self.SPEED_WALK_MPH
            if slope_pct > 5.0:
                speed *= 0.80
        elif mode == "bike":
            # Uphill penalty, downhill speed boost
            speed = self.SPEED_BIKE_MPH
            if gain_m > loss_m and slope_pct > 3.0:
                speed = max(4.5, speed * (1.0 - slope_pct * 0.08))
            elif loss_m > gain_m:
                speed = min(22.0, speed * (1.0 + slope_pct * 0.05))
        elif mode == "ebike":
            # E-bikes maintain high speeds on hills due to motor assist!
            speed = self.SPEED_EBIKE_MPH
            if slope_pct > 5.0:
                speed = max(12.0, speed * 0.85)
        elif mode == "auto":
            # Multi-scale speed: if distance > 3 miles, use highway corridor
            if net_dist_miles > 3.0:
                # Blend local streets with US-101 highway speed
                base_speed = (self.SPEED_AUTO_LOCAL_MPH * 0.25) + (self.SPEED_AUTO_HIGHWAY_MPH * 0.75)
            else:
                base_speed = self.SPEED_AUTO_LOCAL_MPH

            # Peak hour congestion penalty on US-101 (07:30 - 09:00 and 16:30 - 18:30)
            is_peak = (7.5 <= departure_hour <= 9.0) or (16.5 <= departure_hour <= 18.5)
            corridor_congestion = 1.6 if (is_peak and net_dist_miles > 5.0) else congestion_factor
            speed = base_speed / corridor_congestion
        elif mode == "transit":
            # Transit average effective speed including stops and acceleration
            speed = 18.0 / congestion_factor
        else:
            speed = self.SPEED_AUTO_LOCAL_MPH

        travel_time_hours = net_dist_miles / max(1.0, speed)
        travel_time_min = travel_time_hours * 60.0

        # Calculate EV energy consumption
        # Base energy + potential energy climb - regenerative braking recovery
        base_energy = net_dist_miles * self.EV_BASE_KWH_PER_MILE
        # Delta Potential Energy (Joules to kWh: 1 kWh = 3.6e6 J)
        climb_energy_kwh = (self.EV_CURB_WEIGHT_KG * self.GRAVITY * gain_m) / 3.6e6
        regen_recovery_kwh = ((self.EV_CURB_WEIGHT_KG * self.GRAVITY * loss_m) / 3.6e6) * self.REGEN_EFFICIENCY
        total_ev_kwh = max(0.02, base_energy + climb_energy_kwh - regen_recovery_kwh)

        return RouteProfile(
            origin=origin,
            destination=destination,
            mode=mode,
            distance_miles=round(net_dist_miles, 2),
            travel_time_minutes=round(travel_time_min, 1),
            elevation_gain_meters=round(gain_m, 1),
            elevation_loss_meters=round(loss_m, 1),
            average_slope_pct=round(slope_pct, 1),
            ev_energy_kwh=round(total_ev_kwh, 3)
        )
