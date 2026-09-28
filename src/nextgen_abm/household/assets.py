"""Space-time vehicle asset tracking and household mutual exclusion constraints."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VehicleLegRecord:
    driver_id: str
    from_building: str
    to_building: str
    departure_hour: float
    arrival_hour: float
    passenger_ids: List[str] = field(default_factory=list)


@dataclass
class VehicleAsset:
    vehicle_id: str
    household_id: str
    vehicle_type: str        # "ice", "ev", "hybrid"
    initial_location: str
    current_location: str
    legs_history: List[VehicleLegRecord] = field(default_factory=list)


class HouseholdVehicleManager:
    """Manages physical space-time conservation and conflict resolution for household vehicles."""

    def __init__(self, household_id: str, vehicles: List[VehicleAsset]):
        self.household_id = household_id
        self.vehicles = {v.vehicle_id: v for v in vehicles}

    def can_reserve_vehicle(
        self,
        vehicle_id: str,
        driver_id: str,
        start_location: str,
        departure_hour: float,
        arrival_hour: float
    ) -> Tuple[bool, str]:
        """Check if vehicle is physically at start_location and unoccupied during [departure, arrival]."""
        veh = self.vehicles.get(vehicle_id)
        if not veh:
            return False, f"Vehicle {vehicle_id} does not exist in household {self.household_id}"

        # 1. Location check: Vehicle must be physically present at departure building
        if veh.current_location != start_location:
            return False, f"Vehicle {vehicle_id} is located at {veh.current_location}, not {start_location}"

        # 2. Temporal overlap check: Single driver at a time
        for leg in veh.legs_history:
            # Overlap condition: not (arrival <= leg.departure or departure >= leg.arrival)
            if not (arrival_hour <= leg.departure_hour or departure_hour >= leg.arrival_hour):
                return False, f"Vehicle {vehicle_id} is already in use by {leg.driver_id} from {leg.departure_hour:.2f} to {leg.arrival_hour:.2f}"

        return True, "Vehicle available"

    def record_trip_leg(
        self,
        vehicle_id: str,
        driver_id: str,
        from_building: str,
        to_building: str,
        departure_hour: float,
        arrival_hour: float,
        passenger_ids: Optional[List[str]] = None
    ) -> bool:
        """Commit a vehicle trip leg and update physical location."""
        can_res, reason = self.can_reserve_vehicle(vehicle_id, driver_id, from_building, departure_hour, arrival_hour)
        if not can_res:
            return False

        veh = self.vehicles[vehicle_id]
        leg = VehicleLegRecord(
            driver_id=driver_id,
            from_building=from_building,
            to_building=to_building,
            departure_hour=departure_hour,
            arrival_hour=arrival_hour,
            passenger_ids=passenger_ids or []
        )
        veh.legs_history.append(leg)
        veh.current_location = to_building
        return True
