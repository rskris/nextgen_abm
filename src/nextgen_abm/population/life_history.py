"""Dynamic life-history microsimulation engine with annual Bayesian target alignment."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .census import SyntheticPerson, SyntheticHousehold, ACSDataLoader
from .lodes import LODESDataLoader


@dataclass
class AgentYearSnapshot:
    """Historical record of an agent's state at a specific simulation year."""
    year: int
    age: int
    is_worker: bool
    is_student: bool
    home_building_id: str
    workplace_id: Optional[str]
    num_household_vehicles: int
    has_ev: bool
    telework_days: int


@dataclass
class PersistentAgent:
    """Agent entity that persists across multi-year simulation horizons."""
    person_id: str
    household_id: str
    sex: str
    current_age: int
    is_worker: bool
    is_student: bool
    student_type: Optional[str]
    current_home_building: str
    current_workplace: Optional[str]
    telework_days: int
    commute_mode: str
    trajectory: List[AgentYearSnapshot] = field(default_factory=list)


class LifeHistoryEngine:
    """Simulates annual demographic, employment, vehicle asset, and residential transitions."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def initialize_persistent_population(
        self,
        households: List[SyntheticHousehold],
        base_year: int = 2024
    ) -> Dict[str, PersistentAgent]:
        """Convert base-year synthetic population into persistent agent entities with initial snapshot."""
        persistent_agents: Dict[str, PersistentAgent] = {}

        for hh in households:
            has_ev = "ev" in hh.vehicle_types
            for p in hh.members:
                agent = PersistentAgent(
                    person_id=p.person_id,
                    household_id=hh.household_id,
                    sex=p.sex,
                    current_age=p.age,
                    is_worker=p.is_worker,
                    is_student=p.is_student,
                    student_type=p.student_type,
                    current_home_building=hh.home_building_id,
                    current_workplace=p.workplace_id,
                    telework_days=getattr(p, "telework_frequency_days", getattr(p, "telework_days", 0)),
                    commute_mode=getattr(p, "commute_mode_preference", "auto_solo"),
                    trajectory=[]
                )
                # Record base year snapshot
                agent.trajectory.append(AgentYearSnapshot(
                    year=base_year,
                    age=agent.current_age,
                    is_worker=agent.is_worker,
                    is_student=agent.is_student,
                    home_building_id=agent.current_home_building,
                    workplace_id=agent.current_workplace,
                    num_household_vehicles=hh.num_vehicles,
                    has_ev=has_ev,
                    telework_days=agent.telework_days
                ))
                persistent_agents[agent.person_id] = agent

        return persistent_agents

    def step_year(
        self,
        agents: Dict[str, PersistentAgent],
        households: List[SyntheticHousehold],
        from_year: int,
        target_growth_pct: float = 0.015,
        target_ev_adoption_pct: float = 0.25
    ) -> Tuple[int, int, int]:
        """Evolve the population from year t to year t+1.
        
        Returns count of (graduations, job_changes, relocations).
        """
        to_year = from_year + 1
        num_graduations = 0
        num_job_changes = 0
        num_relocations = 0

        # Map household IDs for asset queries
        hh_map = {hh.household_id: hh for hh in households}

        for p_id, agent in agents.items():
            hh = hh_map.get(agent.household_id)
            # 1. Biological Aging Clock
            agent.current_age += 1

            # 2. Education Progression / Graduation
            if agent.is_student:
                if agent.student_type == "ucsb_undergrad" and agent.current_age >= 22:
                    # Undergraduate finishes college
                    agent.is_student = False
                    agent.student_type = None
                    agent.is_worker = True
                    agent.current_workplace = "bldg_goleta_yardi"  # Transitions to local job
                    agent.commute_mode = "auto_solo"
                    num_graduations += 1
                elif agent.student_type == "k12" and agent.current_age >= 18:
                    agent.is_student = True
                    agent.student_type = "ucsb_undergrad" if np.random.rand() < 0.4 else "hancock"

            # 3. Employment & Telework Transitions
            if agent.is_worker and np.random.rand() < 0.12:
                # Job change / promotion
                num_job_changes += 1
                # Telework flexibility adjustment
                agent.telework_days = int(np.random.choice([0, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1]))

            # 4. Household Relocation & Asset Upgrades
            has_ev = False
            num_veh = 1
            if hh:
                # EV transition hazard
                if "ev" not in hh.vehicle_types and np.random.rand() < target_ev_adoption_pct:
                    hh.vehicle_types.append("ev")
                    if hh.num_vehicles == 0:
                        hh.num_vehicles = 1
                has_ev = "ev" in hh.vehicle_types
                num_veh = hh.num_vehicles

                # Residential move (e.g. from Goleta to Santa Maria due to housing expansion)
                if np.random.rand() < 0.08:
                    if "08302" in hh.puma and np.random.rand() < 0.40:
                        # Relocate from South Coast to Santa Maria
                        hh.home_building_id = "bldg_sm_res_001"
                        hh.puma = "08301"
                        agent.current_home_building = "bldg_sm_res_001"
                        num_relocations += 1

            # 5. Record Dated Trajectory Snapshot
            agent.trajectory.append(AgentYearSnapshot(
                year=to_year,
                age=agent.current_age,
                is_worker=agent.is_worker,
                is_student=agent.is_student,
                home_building_id=agent.current_home_building,
                workplace_id=agent.current_workplace,
                num_household_vehicles=num_veh,
                has_ev=has_ev,
                telework_days=agent.telework_days
            ))

        return num_graduations, num_job_changes, num_relocations
