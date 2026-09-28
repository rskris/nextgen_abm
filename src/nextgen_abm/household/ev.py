"""Electric Vehicle (EV) battery State-of-Charge tracking and charging session manager."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ChargingSession:
    charger_type: str        # "home_l2", "workplace_l2", "dc_fast"
    location_id: str
    start_hour: float
    end_hour: float
    power_kw: float
    energy_added_kwh: float
    cost_dollars: float


class EVBatteryTracker:
    """Tracks continuous battery State-of-Charge (SoC) and manages charging activities."""

    MIN_RESERVE_SOC_PCT = 15.0  # 15% reserve to prevent stranding

    CHARGER_SPECS = {
        "home_l2": {"power_kw": 6.6, "rate_per_kwh": 0.22},       # CA residential TOU average
        "workplace_l2": {"power_kw": 6.6, "rate_per_kwh": 0.15},  # Subsidized employer charging
        "dc_fast": {"power_kw": 100.0, "rate_per_kwh": 0.45},     # Commercial fast charging (Supercharger/EA)
    }

    def __init__(self, vehicle_id: str, battery_capacity_kwh: float = 65.0, initial_soc_pct: float = 85.0):
        self.vehicle_id = vehicle_id
        self.capacity_kwh = battery_capacity_kwh
        self.current_energy_kwh = (initial_soc_pct / 100.0) * battery_capacity_kwh
        self.charging_sessions: List[ChargingSession] = []

    @property
    def current_soc_pct(self) -> float:
        return (self.current_energy_kwh / self.capacity_kwh) * 100.0

    def consume_energy(self, trip_energy_kwh: float) -> Tuple[bool, float]:
        """Consume energy for a trip leg. Returns (feasible, remaining_soc_pct)."""
        new_energy = self.current_energy_kwh - trip_energy_kwh
        new_soc = (new_energy / self.capacity_kwh) * 100.0

        if new_soc < self.MIN_RESERVE_SOC_PCT:
            return False, new_soc

        self.current_energy_kwh = new_energy
        return True, new_soc

    def plan_charging_session(
        self,
        charger_type: str,
        location_id: str,
        start_hour: float,
        duration_hours: float,
        target_soc_pct: float = 90.0
    ) -> ChargingSession:
        """Schedule a charging session at Home, Workplace, or DC Fast Charger."""
        spec = self.CHARGER_SPECS.get(charger_type, self.CHARGER_SPECS["home_l2"])
        power_kw = spec["power_kw"]
        rate = spec["rate_per_kwh"]

        # Maximum energy battery can accept
        max_target_kwh = (target_soc_pct / 100.0) * self.capacity_kwh
        deficit_kwh = max(0.0, max_target_kwh - self.current_energy_kwh)

        # Potential energy delivered by charger
        potential_delivered_kwh = power_kw * duration_hours
        actual_energy_kwh = min(deficit_kwh, potential_delivered_kwh)

        self.current_energy_kwh += actual_energy_kwh
        cost = actual_energy_kwh * rate

        session = ChargingSession(
            charger_type=charger_type,
            location_id=location_id,
            start_hour=start_hour,
            end_hour=start_hour + duration_hours,
            power_kw=power_kw,
            energy_added_kwh=round(actual_energy_kwh, 2),
            cost_dollars=round(cost, 2)
        )
        self.charging_sessions.append(session)
        return session
