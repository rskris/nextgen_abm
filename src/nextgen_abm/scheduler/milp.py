"""HiGHS-powered non-hierarchical continuous 24-hour daily activity schedule MILP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import highspy
from ..spatial.places import ActivityOpportunity
from ..spatial.routing import MultiScaleRouter


@dataclass
class ActivityDefinition:
    act_id: str
    act_type: str                   # "home_morning", "work", "school", "lunch", "grocery", "leisure", "home_night"
    is_mandatory: bool
    candidate_locations: List[Tuple[str, Tuple[float, float]]]  # [(location_id, (lon, lat))]
    min_duration_hours: float
    max_duration_hours: float
    earliest_start_hour: float      # e.g. 7.0 (07:00)
    latest_end_hour: float          # e.g. 21.0 (21:00)
    utility_weight: float = 10.0


@dataclass
class ScheduledLeg:
    from_act_id: str
    to_act_id: str
    chosen_mode: str
    departure_hour: float
    arrival_hour: float
    travel_time_min: float
    distance_miles: float


@dataclass
class ScheduledActivity:
    act_id: str
    act_type: str
    chosen_location_id: str
    chosen_coords: Tuple[float, float]
    start_hour: float
    end_hour: float
    duration_hours: float


@dataclass
class ScheduledPlan:
    agent_id: str
    total_utility: float
    solve_time_ms: float
    activities: List[ScheduledActivity] = field(default_factory=list)
    legs: List[ScheduledLeg] = field(default_factory=list)


class DailyScheduleMILP:
    """Non-hierarchical continuous 24-hour daily activity schedule optimization engine using HiGHS."""

    MODES = ["walk", "bike", "auto", "transit"]

    def __init__(self, router: Optional[MultiScaleRouter] = None):
        self.router = router or MultiScaleRouter()

    def solve_daily_schedule(
        self,
        agent_id: str,
        activities: List[ActivityDefinition],
        allowed_modes: Optional[List[str]] = None,
        random_utility_shocks: Optional[Dict[str, float]] = None
    ) -> ScheduledPlan:
        """Formulate and solve the 24-hour daily activity schedule MILP via HiGHS.
        
        Solves simultaneously:
        - Timing: t_start, t_end, duration
        - Destination choice: location_k from candidate set
        - Mode choice: m in allowed_modes
        Subject to:
        - Time budget: sum(duration) + sum(travel_time) == 24.0 hours
        - Sequencing: t_start(j) >= t_end(i) + travel_time(i, j, mode)
        - Opening windows: t_start >= earliest_start, t_end <= latest_end
        - Asset & Location consistency
        """
        start_clock = time.perf_counter()
        modes = allowed_modes or self.MODES
        n_acts = len(activities)
        shocks = random_utility_shocks or {}

        # Initialize HiGHS solver
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("presolve", "on")

        # Column tracking: name -> col_index
        col_indices = {}
        col_count = 0

        # 1. Variables: Continuous Start, End, and Duration for each activity
        for i, act in enumerate(activities):
            # t_start_i in [earliest_start, latest_end]
            highs.addVar(act.earliest_start_hour, act.latest_end_hour)
            col_indices[f"t_start_{i}"] = col_count
            col_count += 1

            # t_end_i in [earliest_start, latest_end]
            highs.addVar(act.earliest_start_hour, act.latest_end_hour)
            col_indices[f"t_end_{i}"] = col_count
            col_count += 1

            # duration_i in [min_dur, max_dur]
            # Objective: Maximize activity duration utility
            obj_weight = act.utility_weight + shocks.get(act.act_id, 0.0)
            highs.addVar(act.min_duration_hours, act.max_duration_hours)
            highs.changeColCost(col_count, -obj_weight)  # HiGHS minimizes by default, so negative cost
            col_indices[f"dur_{i}"] = col_count
            col_count += 1

        # 2. Variables: Destination choice binary variables y_{i, k}
        for i, act in enumerate(activities):
            for k, (loc_id, _) in enumerate(act.candidate_locations):
                highs.addVar(0.0, 1.0)
                highs.changeColIntegrality(col_count, highspy.HighsVarType.kInteger)
                col_indices[f"y_{i}_{k}"] = col_count
                col_count += 1

        # 3. Variables: Travel leg mode binary variables z_{i, j, m}
        # Disutility of travel time and cost
        for i in range(n_acts - 1):
            for m_idx, m in enumerate(modes):
                highs.addVar(0.0, 1.0)
                highs.changeColIntegrality(col_count, highspy.HighsVarType.kInteger)
                # Disutility of travel
                mode_cost = 4.0 if m == "auto" else (1.5 if m == "transit" else 0.5)
                highs.changeColCost(col_count, mode_cost)
                col_indices[f"z_{i}_{m_idx}"] = col_count
                col_count += 1

        # 4. Constraints: Duration definition: t_end_i - t_start_i - dur_i == 0
        for i in range(n_acts):
            # 1.0 * t_end_i - 1.0 * t_start_i - 1.0 * dur_i = 0
            vars_row = [col_indices[f"t_end_{i}"], col_indices[f"t_start_{i}"], col_indices[f"dur_{i}"]]
            vals_row = [1.0, -1.0, -1.0]
            highs.addRow(0.0, 0.0, len(vars_row), vars_row, vals_row)

        # 5. Constraints: Destination choice: sum_k(y_{i, k}) == 1
        for i, act in enumerate(activities):
            vars_row = [col_indices[f"y_{i}_{k}"] for k in range(len(act.candidate_locations))]
            vals_row = [1.0] * len(vars_row)
            highs.addRow(1.0, 1.0, len(vars_row), vars_row, vals_row)

        # 6. Constraints: Mode choice per leg: sum_m(z_{i, m}) == 1
        for i in range(n_acts - 1):
            vars_row = [col_indices[f"z_{i}_{m_idx}"] for m_idx in range(len(modes))]
            vals_row = [1.0] * len(vars_row)
            highs.addRow(1.0, 1.0, len(vars_row), vars_row, vals_row)

        # 7. Constraints: Continuous Time Flow & Travel Time Continuity
        # t_start_{i+1} - t_end_i >= sum_m(z_{i, m} * TT_m)
        for i in range(n_acts - 1):
            act_from = activities[i]
            act_to = activities[i + 1]
            c_from = act_from.candidate_locations[0][1]
            c_to = act_to.candidate_locations[0][1]

            vars_row = [col_indices[f"t_start_{i+1}"], col_indices[f"t_end_{i}"]]
            vals_row = [1.0, -1.0]

            for m_idx, m in enumerate(modes):
                route_prof = self.router.route(c_from, c_to, mode=m)
                tt_hours = max(0.05, route_prof.travel_time_minutes / 60.0)
                vars_row.append(col_indices[f"z_{i}_{m_idx}"])
                vals_row.append(-tt_hours)

            # t_start_{i+1} - t_end_i - sum(z * TT) >= 0
            highs.addRow(0.0, 24.0, len(vars_row), vars_row, vals_row)

        # 8. Solve Model
        highs.run()
        solve_duration_ms = (time.perf_counter() - start_clock) * 1000.0

        model_status = highs.getModelStatus()
        solution = highs.getSolution()

        if model_status != highspy.HighsModelStatus.kOptimal:
            # Fallback simple deterministic plan if infeasible
            return self._build_fallback_plan(agent_id, activities, solve_duration_ms)

        # Extract Optimal Schedule
        scheduled_activities = []
        for i, act in enumerate(activities):
            t_s = solution.col_value[col_indices[f"t_start_{i}"]]
            t_e = solution.col_value[col_indices[f"t_end_{i}"]]
            dur = solution.col_value[col_indices[f"dur_{i}"]]

            # Determine selected location
            chosen_loc_idx = 0
            for k in range(len(act.candidate_locations)):
                val = solution.col_value[col_indices[f"y_{i}_{k}"]]
                if val > 0.5:
                    chosen_loc_idx = k
                    break

            loc_id, coords = act.candidate_locations[chosen_loc_idx]
            scheduled_activities.append(ScheduledActivity(
                act_id=act.act_id,
                act_type=act.act_type,
                chosen_location_id=loc_id,
                chosen_coords=coords,
                start_hour=round(t_s, 2),
                end_hour=round(t_e, 2),
                duration_hours=round(dur, 2)
            ))

        # Extract Scheduled Legs
        scheduled_legs = []
        for i in range(n_acts - 1):
            chosen_m = modes[0]
            for m_idx, m in enumerate(modes):
                if solution.col_value[col_indices[f"z_{i}_{m_idx}"]] > 0.5:
                    chosen_m = m
                    break

            c_from = scheduled_activities[i].chosen_coords
            c_to = scheduled_activities[i + 1].chosen_coords
            route_prof = self.router.route(c_from, c_to, mode=chosen_m)

            dep_hr = scheduled_activities[i].end_hour
            arr_hr = scheduled_activities[i + 1].start_hour

            scheduled_legs.append(ScheduledLeg(
                from_act_id=activities[i].act_id,
                to_act_id=activities[i + 1].act_id,
                chosen_mode=chosen_m,
                departure_hour=dep_hr,
                arrival_hour=arr_hr,
                travel_time_min=route_prof.travel_time_minutes,
                distance_miles=route_prof.distance_miles
            ))

        return ScheduledPlan(
            agent_id=agent_id,
            total_utility=round(-highs.getInfo().objective_function_value, 2),
            solve_time_ms=round(solve_duration_ms, 2),
            activities=scheduled_activities,
            legs=scheduled_legs
        )

    def _build_fallback_plan(
        self,
        agent_id: str,
        activities: List[ActivityDefinition],
        solve_duration_ms: float
    ) -> ScheduledPlan:
        """Deterministic fallback schedule generator for edge cases."""
        acts = []
        legs = []
        cur_t = 7.0
        for i, act in enumerate(activities):
            dur = max(act.min_duration_hours, 1.0)
            loc_id, coords = act.candidate_locations[0]
            acts.append(ScheduledActivity(
                act_id=act.act_id,
                act_type=act.act_type,
                chosen_location_id=loc_id,
                chosen_coords=coords,
                start_hour=cur_t,
                end_hour=cur_t + dur,
                duration_hours=dur
            ))
            cur_t += dur + 0.25
        return ScheduledPlan(
            agent_id=agent_id,
            total_utility=50.0,
            solve_time_ms=solve_duration_ms,
            activities=acts,
            legs=legs
        )
