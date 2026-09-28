"""Test HiGHS combinatorial scheduler, space-time pruning, and Metropolis-Hastings MCMC."""

import pytest
from nextgen_abm.spatial.places import ActivityOpportunity
from nextgen_abm.scheduler.pruning import SpaceTimePruner
from nextgen_abm.scheduler.milp import DailyScheduleMILP, ActivityDefinition
from nextgen_abm.scheduler.sampling import MetropolisHastingsSampler


def test_space_time_pruner():
    pruner = SpaceTimePruner()
    home = (-119.855, 34.412)   # Isla Vista
    work = (-119.845, 34.415)   # UCSB Campbell Hall

    candidates = [
        ActivityOpportunity("poi_close_1", "Freebirds IV", "dining", 11.0, 24.0, 30.0, -119.854, 34.413),
        ActivityOpportunity("poi_close_2", "Silvergreens IV", "dining", 8.0, 21.0, 30.0, -119.852, 34.413),
        ActivityOpportunity("poi_far_sm", "Santa Maria Burger", "dining", 10.0, 22.0, 30.0, -120.435, 34.952),
    ]

    pruned = pruner.prune_candidates(home, work, candidates, max_detour_miles=1.5, max_results=5)
    pruned_ids = [p.poi_id for p in pruned]

    # The close Isla Vista dining options must be included, while Santa Maria is pruned!
    assert "poi_close_1" in pruned_ids
    assert "poi_close_2" in pruned_ids
    assert "poi_far_sm" not in pruned_ids


def test_highs_daily_schedule_solve_speed_and_optimality():
    milp = DailyScheduleMILP()

    home_coords = (-119.856, 34.411)
    work_coords = (-119.833, 34.435)
    grocery_coords = (-119.818, 34.439)

    activities = [
        ActivityDefinition(
            act_id="act_home_morning",
            act_type="home_morning",
            is_mandatory=True,
            candidate_locations=[("home", home_coords)],
            min_duration_hours=1.0,
            max_duration_hours=3.0,
            earliest_start_hour=6.0,
            latest_end_hour=9.0,
            utility_weight=5.0
        ),
        ActivityDefinition(
            act_id="act_work",
            act_type="work",
            is_mandatory=True,
            candidate_locations=[("work", work_coords)],
            min_duration_hours=6.0,
            max_duration_hours=8.5,
            earliest_start_hour=8.0,
            latest_end_hour=18.0,
            utility_weight=15.0
        ),
        ActivityDefinition(
            act_id="act_grocery",
            act_type="grocery",
            is_mandatory=False,
            candidate_locations=[("grocery", grocery_coords)],
            min_duration_hours=0.5,
            max_duration_hours=1.5,
            earliest_start_hour=16.0,
            latest_end_hour=21.0,
            utility_weight=8.0
        ),
        ActivityDefinition(
            act_id="act_home_night",
            act_type="home_night",
            is_mandatory=True,
            candidate_locations=[("home", home_coords)],
            min_duration_hours=4.0,
            max_duration_hours=8.0,
            earliest_start_hour=18.0,
            latest_end_hour=24.0,
            utility_weight=10.0
        ),
    ]

    plan = milp.solve_daily_schedule("agent_test_01", activities)

    # Performance Target: Solve time must be under 50 ms!
    assert plan.solve_time_ms < 50.0
    assert len(plan.activities) == 4
    assert len(plan.legs) == 3

    # Check temporal ordering
    for i in range(len(plan.activities) - 1):
        assert plan.activities[i].end_hour <= plan.activities[i + 1].start_hour

    # Verify travel legs have valid mode and positive travel time
    for leg in plan.legs:
        assert leg.chosen_mode in ["walk", "bike", "auto", "transit"]
        assert leg.travel_time_min > 0.0


def test_metropolis_hastings_sampling():
    sampler = MetropolisHastingsSampler()

    home_coords = (-119.856, 34.411)
    work_coords = (-119.845, 34.415)

    activities = [
        ActivityDefinition("act_home", "home_morning", True, [("home", home_coords)], 1.0, 2.0, 7.0, 9.0),
        ActivityDefinition("act_study", "study", True, [("lib", work_coords)], 3.0, 5.0, 9.0, 16.0),
        ActivityDefinition("act_return", "home_night", True, [("home", home_coords)], 2.0, 6.0, 16.0, 22.0),
    ]

    samples = sampler.sample_choice_set("agent_sample_01", activities, n_samples=3, burn_in=1)

    assert len(samples) == 3
    for s in samples:
        assert s.acceptance_probability >= 0.0
        assert s.importance_weight > 0.0
        assert len(s.plan.activities) == 3
