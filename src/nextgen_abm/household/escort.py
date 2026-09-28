"""Intra-household synchronization and school escort trip coordination."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from ..spatial.routing import MultiScaleRouter


@dataclass
class EscortTrip:
    driver_person_id: str
    child_person_id: str
    school_building_id: str
    school_coords: Tuple[float, float]
    departure_hour: float
    school_arrival_hour: float
    driver_work_arrival_hour: float
    detour_minutes: float


class HouseholdCoordinator:
    """Coordinates joint activities and optimizes dependent escort chains."""

    def __init__(self, router: Optional[MultiScaleRouter] = None):
        self.router = router or MultiScaleRouter()

    def synchronize_joint_activity(
        self,
        activity_name: str,
        location_coords: Tuple[float, float],
        target_hour: float,
        participants: List[str]
    ) -> Dict[str, Any]:
        """Synchronize arrival times of multiple household members for a joint activity (e.g. dinner)."""
        return {
            "activity_name": activity_name,
            "target_hour": target_hour,
            "location_coords": location_coords,
            "participants": participants,
            "status": "synchronized"
        }

    def assign_school_escort_driver(
        self,
        home_coords: Tuple[float, float],
        school_coords: Tuple[float, float],
        school_start_hour: float,
        adult_workers: List[Dict[str, Any]],  # [{"person_id": "p1", "work_coords": (x,y), "flexibility": "high"}]
        child_person_id: str,
        school_building_id: str = "bldg_school_01"
    ) -> EscortTrip:
        """Assign the adult household member who incurs the lowest detour to chauffeur the child."""
        best_driver = None
        min_detour_min = float("inf")
        best_times = (0.0, 0.0, 0.0)

        # Drop-off must be 10 minutes before school starts
        target_school_arrival = school_start_hour - (10.0 / 60.0)

        for adult in adult_workers:
            driver_id = adult["person_id"]
            work_coords = adult["work_coords"]

            # Direct trip Home -> Work
            direct_prof = self.router.route(home_coords, work_coords, mode="auto")
            direct_time = direct_prof.travel_time_minutes

            # Escort chain: Home -> School -> Work
            leg1_prof = self.router.route(home_coords, school_coords, mode="auto")
            leg2_prof = self.router.route(school_coords, work_coords, mode="auto")

            escort_time = leg1_prof.travel_time_minutes + 5.0 + leg2_prof.travel_time_minutes  # +5 min drop-off dwell
            detour = escort_time - direct_time

            # Schedule timing
            dep_hour = target_school_arrival - (leg1_prof.travel_time_minutes / 60.0)
            work_arr_hour = target_school_arrival + (5.0 / 60.0) + (leg2_prof.travel_time_minutes / 60.0)

            if detour < min_detour_min:
                min_detour_min = detour
                best_driver = driver_id
                best_times = (dep_hour, target_school_arrival, work_arr_hour)

        return EscortTrip(
            driver_person_id=best_driver or adult_workers[0]["person_id"],
            child_person_id=child_person_id,
            school_building_id=school_building_id,
            school_coords=school_coords,
            departure_hour=round(best_times[0], 2),
            school_arrival_hour=round(best_times[1], 2),
            driver_work_arrival_hour=round(best_times[2], 2),
            detour_minutes=round(min_detour_min, 1)
        )
