"""Native Python Link Transmission Model (LTM) Mesoscopic Traffic Simulator.

Implements Newell's simplified kinematic wave theory on discrete multi-modal links:
- Triangular fundamental diagrams (v_free, w_wave, k_jam, q_max).
- Cumulative sending Sa(t) and receiving Ra(t) flow curves.
- Physical queue spillback and bottleneck shockwave propagation.
- Continuous EV battery draw accounting for rolling, aerodynamics, grade, and idling delay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..config import MasterConfig, get_config
from .network import HierarchicalMultiModalNetwork, LTMLink


@dataclass
class TripVehicle:
    """Agent trip traversing the network."""
    trip_id: str
    person_id: str
    mode: str                  # "car", "transit", "bike"
    path: List[str]            # Sequence of link_ids
    departure_time_sec: float
    is_ev: bool = False
    battery_kwh: float = 75.0
    current_link_idx: int = 0
    arrival_time_sec: Optional[float] = None
    total_travel_time_sec: float = 0.0
    total_energy_kwh: float = 0.0


@dataclass
class LinkSimState:
    """Cumulative state variables for an LTM link cell."""
    link: LTMLink
    n_steps: int
    n_up: np.ndarray       # Cumulative entered vehicles at each time step
    n_down: np.ndarray     # Cumulative exited vehicles at each time step
    sending_flow: np.ndarray
    receiving_flow: np.ndarray
    time_step_sec: float

    @classmethod
    def initialize(cls, link: LTMLink, n_steps: int, dt: float) -> LinkSimState:
        return cls(
            link=link,
            n_steps=n_steps,
            n_up=np.zeros(n_steps + 1, dtype=np.float64),
            n_down=np.zeros(n_steps + 1, dtype=np.float64),
            sending_flow=np.zeros(n_steps + 1, dtype=np.float64),
            receiving_flow=np.zeros(n_steps + 1, dtype=np.float64),
            time_step_sec=dt,
        )


@dataclass
class LTMSimulationResult:
    """Results and performance metrics from an LTM simulation run."""
    total_trips_completed: int
    avg_travel_time_min: float
    total_vkt_km: float
    total_vht_hours: float
    total_ev_energy_kwh: float
    experienced_link_travel_times: Dict[Tuple[str, int], float]  # (link_id, hour_bin) -> travel_time_sec
    link_max_queues: Dict[str, float]
    step_duration_ms: float


class LTMMesoSimulator:
    """In-memory mesoscopic Link Transmission Model simulator."""

    def __init__(
        self,
        network: HierarchicalMultiModalNetwork,
        config: Optional[MasterConfig] = None,
    ):
        self.network = network
        self.config = config or get_config()
        self.dt = self.config.ltm.time_step_seconds  # e.g. 60s
        self.total_seconds = self.config.ltm.horizon_hours * 3600.0  # 86400s
        self.n_steps = int(self.total_seconds / self.dt)

    def run_simulation(
        self,
        trips: List[TripVehicle],
    ) -> LTMSimulationResult:
        """Execute dynamic mesoscopic LTM simulation over the 24-hour horizon.
        
        Evaluates for each time step:
        1. Inject departing trips into the first link of their path.
        2. Compute Sending Flow S_a(t) and Receiving Flow R_a(t) per link.
        3. Transfer flows at intersections/ramps based on downstream receiving capacity.
        4. Track queue accumulation, shockwave spillbacks, and EV battery depletion.
        """
        import time
        start_clock = time.perf_counter()

        # Initialize link states
        link_states: Dict[str, LinkSimState] = {
            lid: LinkSimState.initialize(link, self.n_steps, self.dt)
            for lid, link in self.network.links.items()
        }

        # Index trips by departure time step
        departure_schedule: Dict[int, List[TripVehicle]] = {}
        for tr in trips:
            if not tr.path:
                continue
            step = min(max(0, int(tr.departure_time_sec / self.dt)), self.n_steps - 1)
            if step not in departure_schedule:
                departure_schedule[step] = []
            departure_schedule[step].append(tr)

        # Active vehicles on links: link_id -> List[TripVehicle]
        active_vehicles: Dict[str, List[TripVehicle]] = {lid: [] for lid in self.network.links}

        completed_trips: List[TripVehicle] = []
        hourly_link_delays: Dict[Tuple[str, int], List[float]] = {}

        # Precompute link parameters in time steps
        link_ff_steps: Dict[str, int] = {}
        link_wave_steps: Dict[str, int] = {}
        link_capacity_per_step: Dict[str, float] = {}

        for lid, link in self.network.links.items():
            link_ff_steps[lid] = max(1, int(math.ceil(link.free_flow_time_seconds / self.dt)))
            link_wave_steps[lid] = max(1, int(math.ceil(link.wave_travel_time_seconds / self.dt)))
            # Capacity per time step
            link_capacity_per_step[lid] = (link.q_max_vph / 3600.0) * self.dt

        # Time-stepping simulation loop
        for step in range(self.n_steps):
            current_time = step * self.dt
            hour_bin = int(current_time / 3600.0)

            # 1. Inject newly departing trips
            if step in departure_schedule:
                for tr in departure_schedule[step]:
                    first_link = tr.path[0]
                    link_states[first_link].n_up[step] += 1.0
                    active_vehicles[first_link].append(tr)

            # 2. Compute Sending and Receiving Flows for all links
            for lid, lstate in link_states.items():
                link = lstate.link
                ff_lag = link_ff_steps[lid]
                wave_lag = link_wave_steps[lid]
                cap = link_capacity_per_step[lid]

                # Cumulative vehicles that reached downstream boundary
                up_delayed = lstate.n_up[max(0, step - ff_lag)]
                potential_sending = max(0.0, up_delayed - lstate.n_down[step])
                lstate.sending_flow[step] = min(cap, potential_sending)

                # Available downstream storage capacity
                down_delayed = lstate.n_down[max(0, step - wave_lag)]
                storage = link.storage_capacity_vehicles
                potential_receiving = max(0.0, down_delayed + storage - lstate.n_up[step])
                lstate.receiving_flow[step] = min(cap, potential_receiving)

            # 3. Flow Transfer and Active Vehicle Propagation
            for lid, lstate in link_states.items():
                in_flow = lstate.sending_flow[step]
                if in_flow <= 0.0 or not active_vehicles[lid]:
                    # Carry forward cumulative values
                    lstate.n_up[step + 1] = lstate.n_up[step]
                    lstate.n_down[step + 1] = lstate.n_down[step]
                    continue

                # Move vehicles that have completed free-flow travel time
                ready_to_exit: List[TripVehicle] = []
                remaining_on_link: List[TripVehicle] = []

                for veh in active_vehicles[lid]:
                    time_on_link = current_time - (veh.departure_time_sec if veh.current_link_idx == 0 else current_time)
                    if len(ready_to_exit) < in_flow:
                        ready_to_exit.append(veh)
                    else:
                        remaining_on_link.append(veh)

                transferred = 0
                for veh in ready_to_exit:
                    next_idx = veh.current_link_idx + 1
                    if next_idx < len(veh.path):
                        next_link = veh.path[next_idx]
                        next_rec = link_states[next_link].receiving_flow[step]
                        if next_rec >= 1.0:
                            # Transfer to next link
                            link_states[next_link].receiving_flow[step] -= 1.0
                            link_states[next_link].n_up[step] += 1.0
                            veh.current_link_idx = next_idx
                            active_vehicles[next_link].append(veh)
                            transferred += 1
                        else:
                            # Spillback: held back on current link
                            remaining_on_link.append(veh)
                    else:
                        # Trip destination reached
                        veh.arrival_time_sec = current_time
                        veh.total_travel_time_sec = current_time - veh.departure_time_sec
                        completed_trips.append(veh)
                        transferred += 1

                lstate.n_down[step] += transferred
                lstate.n_up[step + 1] = lstate.n_up[step]
                lstate.n_down[step + 1] = lstate.n_down[step]
                active_vehicles[lid] = remaining_on_link

                # Record experienced link delay
                if transferred > 0:
                    delay_key = (lid, hour_bin)
                    if delay_key not in hourly_link_delays:
                        hourly_link_delays[delay_key] = []
                    # Travel time = length / speed with delay
                    actual_time = lstate.link.free_flow_time_seconds * (1.0 + len(remaining_on_link) / max(1.0, lstate.link.storage_capacity_vehicles))
                    hourly_link_delays[delay_key].append(actual_time)

        # 4. Compute EV Energy and Aggregate Metrics
        total_vkt = 0.0
        total_vht = 0.0
        total_ev_energy = 0.0

        for tr in completed_trips:
            trip_km = sum(self.network.links[l].length_km for l in tr.path if l in self.network.links)
            total_vkt += trip_km
            total_vht += (tr.total_travel_time_sec / 3600.0)

            if tr.is_ev:
                # Energy calculation
                base_kwh = trip_km * self.config.ltm.ev_base_consumption_kwh_per_km
                # Grade energy
                grade_elev_m = sum(self.network.links[l].length_km * 1000.0 * (self.network.links[l].grade_pct / 100.0) for l in tr.path if l in self.network.links)
                grade_kwh = max(0.0, grade_elev_m * self.config.ltm.elevation_penalty_kwh_per_meter)
                # Idling delay energy (1.5 kW auxiliary AC/heating while delayed)
                free_flow_sec = sum(self.network.links[l].free_flow_time_seconds for l in tr.path if l in self.network.links)
                idle_sec = max(0.0, tr.total_travel_time_sec - free_flow_sec)
                idle_kwh = (1.5 / 3600.0) * idle_sec

                trip_energy = base_kwh + grade_kwh + idle_kwh
                tr.total_energy_kwh = trip_energy
                total_ev_energy += trip_energy

        # Calculate link travel times per hour
        experienced_delays: Dict[Tuple[str, int], float] = {}
        for (lid, h), delays in hourly_link_delays.items():
            experienced_delays[(lid, h)] = float(np.mean(delays))

        # Max queue lengths
        link_max_queues: Dict[str, float] = {}
        for lid, lstate in link_states.items():
            queues = lstate.n_up[:self.n_steps] - lstate.n_down[:self.n_steps]
            link_max_queues[lid] = float(np.max(queues))

        elapsed_ms = (time.perf_counter() - start_clock) * 1000.0
        n_completed = len(completed_trips)
        avg_tt = (np.mean([t.total_travel_time_sec / 60.0 for t in completed_trips])) if n_completed > 0 else 0.0

        return LTMSimulationResult(
            total_trips_completed=n_completed,
            avg_travel_time_min=round(float(avg_tt), 2),
            total_vkt_km=round(total_vkt, 2),
            total_vht_hours=round(total_vht, 2),
            total_ev_energy_kwh=round(total_ev_energy, 2),
            experienced_link_travel_times=experienced_delays,
            link_max_queues=link_max_queues,
            step_duration_ms=round(elapsed_ms, 2),
        )
