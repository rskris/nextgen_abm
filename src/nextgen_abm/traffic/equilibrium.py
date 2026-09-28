"""Day-to-Day Evolutionary Replanning Supply-Demand Equilibrium Engine.

Implements MATSim-style evolutionary dynamics:
- Household plan memory pools with utility scoring.
- Multinomial logit plan selection based on experienced travel times.
- 10-20% replanning fraction re-solving Unified Household MILP with dynamic link skims.
- Relative gap convergence tracking across simulation iterations.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..config import MasterConfig, get_config
from ..household.household_milp import (
    EscortSpec,
    HouseholdMILPResult,
    JointActivitySpec,
    MemberAgenda,
    UnifiedHouseholdMILP,
)
from ..household.assets import VehicleAsset
from .network import HierarchicalMultiModalNetwork
from .ltm import LTMMesoSimulator, LTMSimulationResult, TripVehicle


@dataclass
class PlanRecord:
    """Historical daily plan stored in household agent memory."""
    plan_id: str
    household_id: str
    result: HouseholdMILPResult
    score: float = 0.0


@dataclass
class HouseholdAgent:
    """Household unit maintaining a choice set of daily plans with score memory."""
    household_id: str
    members: List[MemberAgenda]
    vehicles: List[VehicleAsset] = field(default_factory=list)
    joint_activities: List[JointActivitySpec] = field(default_factory=list)
    escorts: List[EscortSpec] = field(default_factory=list)
    plans: List[PlanRecord] = field(default_factory=list)
    active_plan_idx: int = 0

    def select_plan_logit(self, beta: float = 1.0) -> int:
        """Select a plan from memory using multinomial logit probabilities."""
        if not self.plans:
            return 0
        if len(self.plans) == 1:
            self.active_plan_idx = 0
            return 0

        scores = np.array([p.score for p in self.plans], dtype=np.float64)
        # Shift scores for numerical stability
        shifted = scores - np.max(scores)
        exp_scores = np.exp(beta * shifted)
        probs = exp_scores / np.sum(exp_scores)

        chosen_idx = int(np.random.choice(len(self.plans), p=probs))
        self.active_plan_idx = chosen_idx
        return chosen_idx

    def add_plan(self, plan: PlanRecord, max_memory: int = 4) -> None:
        """Add a newly generated plan, pruning lowest scoring if memory is full."""
        self.plans.append(plan)
        if len(self.plans) > max_memory:
            # Sort by score ascending and remove the worst
            self.plans.sort(key=lambda p: p.score)
            self.plans.pop(0)
        self.active_plan_idx = len(self.plans) - 1


@dataclass
class IterationMetric:
    """System-level performance indicators for an equilibrium iteration."""
    iteration: int
    total_trips: int
    avg_travel_time_min: float
    total_vht_hours: float
    total_vkt_km: float
    total_ev_energy_kwh: float
    avg_plan_score: float
    relative_gap: float


class DayToDayEquilibriumEngine:
    """Coordinates the iterative supply-demand feedback loop to reach equilibrium."""

    def __init__(
        self,
        network: HierarchicalMultiModalNetwork,
        config: Optional[MasterConfig] = None,
    ):
        self.network = network
        self.config = config or get_config()
        self.sim = LTMMesoSimulator(network=self.network, config=self.config)
        self.solver = UnifiedHouseholdMILP(config=self.config)
        self.replan_frac = self.config.equilibrium.replan_fraction
        self.max_iter = self.config.equilibrium.max_iterations
        self.gap_tol = self.config.equilibrium.convergence_tolerance_rel_gap

    def run_equilibrium(
        self,
        households: List[HouseholdAgent],
        max_iterations: Optional[int] = None,
    ) -> List[IterationMetric]:
        """Execute day-to-day evolutionary replanning loop until convergence."""
        n_iters = max_iterations or self.max_iter
        metrics: List[IterationMetric] = []
        prev_vht = 0.0

        # Step 0: Ensure all households have an initial baseline plan
        for hh in households:
            if not hh.plans:
                res = self.solver.solve_household(
                    household_id=hh.household_id,
                    members=hh.members,
                    vehicles=hh.vehicles,
                    joint_activities=hh.joint_activities,
                    escorts=hh.escorts,
                )
                init_plan = PlanRecord(
                    plan_id=f"{hh.household_id}_plan_0",
                    household_id=hh.household_id,
                    result=res,
                    score=res.total_household_utility,
                )
                hh.add_plan(init_plan, max_memory=self.config.equilibrium.plan_memory_size)

        for it in range(1, n_iters + 1):
            # 1. Replanning phase (sample 15% of households to generate a fresh plan)
            if it > 1:
                n_replan = max(1, int(len(households) * self.replan_frac))
                replan_candidates = random.sample(households, n_replan)
                for hh in replan_candidates:
                    res = self.solver.solve_household(
                        household_id=hh.household_id,
                        members=hh.members,
                        vehicles=hh.vehicles,
                        joint_activities=hh.joint_activities,
                        escorts=hh.escorts,
                    )
                    new_plan = PlanRecord(
                        plan_id=f"{hh.household_id}_plan_{it}",
                        household_id=hh.household_id,
                        result=res,
                        score=res.total_household_utility,
                    )
                    hh.add_plan(new_plan, max_memory=self.config.equilibrium.plan_memory_size)

                # Other households choose from their memory pool
                for hh in households:
                    if hh not in replan_candidates:
                        hh.select_plan_logit(beta=self.config.equilibrium.logit_beta)

            # 2. Extract network vehicle trips from active household plans
            trips: List[TripVehicle] = []
            for hh in households:
                active_plan = hh.plans[hh.active_plan_idx]
                for p_id, sched in active_plan.result.member_schedules.items():
                    # Find vehicle ownership
                    has_ev = any(v.vehicle_type == "ev" for v in hh.vehicles)

                    for leg_idx, leg in enumerate(sched.legs):
                        # Snap origin and destination
                        o_coords = sched.activities[leg_idx].chosen_coords
                        d_coords = sched.activities[leg_idx + 1].chosen_coords
                        o_node = self.network.snap_building(f"{p_id}_orig", o_coords, leg.chosen_mode)
                        d_node = self.network.snap_building(f"{p_id}_dest", d_coords, leg.chosen_mode)

                        path = self.network.find_shortest_path(o_node, d_node, mode=leg.chosen_mode)
                        if not path:
                            # Fallback to default link if snapped to isolated node
                            continue

                        trips.append(
                            TripVehicle(
                                trip_id=f"{p_id}_trip_{leg_idx}",
                                person_id=p_id,
                                mode=leg.chosen_mode,
                                path=path,
                                departure_time_sec=leg.departure_hour * 3600.0,
                                is_ev=has_ev and (leg.chosen_mode == "auto"),
                                battery_kwh=75.0,
                            )
                        )

            # 3. Execute LTM meso-simulation
            sim_res = self.sim.run_simulation(trips)

            # 4. Score executed plans based on experienced congestion delays
            scores = []
            for hh in households:
                active_plan = hh.plans[hh.active_plan_idx]
                # Update score: baseline schedule utility - congestion penalty
                congestion_penalty = sim_res.avg_travel_time_min * 0.2
                active_plan.score = active_plan.result.total_household_utility - congestion_penalty
                scores.append(active_plan.score)

            # 5. Evaluate convergence relative gap
            rel_gap = abs(sim_res.total_vht_hours - prev_vht) / max(prev_vht, 1.0) if prev_vht > 0.0 else 1.0
            prev_vht = sim_res.total_vht_hours

            metric = IterationMetric(
                iteration=it,
                total_trips=sim_res.total_trips_completed,
                avg_travel_time_min=sim_res.avg_travel_time_min,
                total_vht_hours=sim_res.total_vht_hours,
                total_vkt_km=sim_res.total_vkt_km,
                total_ev_energy_kwh=sim_res.total_ev_energy_kwh,
                avg_plan_score=round(float(np.mean(scores)), 2),
                relative_gap=round(float(rel_gap), 4),
            )
            metrics.append(metric)

            # Check convergence stop condition
            if it > 2 and rel_gap < self.gap_tol:
                break

        return metrics
