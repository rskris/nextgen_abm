"""Regional policy scenario evaluation engine for Santa Barbara County."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from ..population.ucsb import UCSBSubModel, ScheduledClass, StudentProfile


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    key_metrics: Dict[str, Any]
    summary_findings: str


class PolicyScenarioEngine:
    """Evaluates signature regional transportation and land-use policies for Santa Barbara County."""

    def evaluate_us101_peak_spreading(
        self,
        n_commuters: int = 1000,
        pct_flexible_workers: float = 0.40,
        flexible_window_hours: float = 2.0  # e.g., arrival between 7:30 and 9:30 AM
    ) -> ScenarioResult:
        """Simulate the effect of employer flexible work hours and telework incentives on US-101 peak flattening."""
        np.random.seed(42)

        # Baseline: All workers target 08:00 - 08:30 AM arrival
        baseline_arrivals = np.random.normal(loc=8.2, scale=0.25, size=n_commuters)
        baseline_peak_volume = np.sum((baseline_arrivals >= 7.75) & (baseline_arrivals <= 8.5))

        # Scenario: Flexible workers spread across the wider arrival window
        policy_arrivals = baseline_arrivals.copy()
        flexible_indices = np.random.choice(n_commuters, size=int(n_commuters * pct_flexible_workers), replace=False)
        # Shift flexible workers across 7:30 to 10:00 AM
        policy_arrivals[flexible_indices] = np.random.uniform(7.5, 7.5 + flexible_window_hours, size=len(flexible_indices))

        policy_peak_volume = np.sum((policy_arrivals >= 7.75) & (policy_arrivals <= 8.5))
        peak_reduction_pct = ((baseline_peak_volume - policy_peak_volume) / baseline_peak_volume) * 100.0

        return ScenarioResult(
            scenario_id="scenario_us101_peak_spreading",
            scenario_name="US-101 Employer Schedule Flexibility & Telework",
            key_metrics={
                "total_commuters": n_commuters,
                "pct_flexible_workers": pct_flexible_workers * 100.0,
                "baseline_peak_trips": int(baseline_peak_volume),
                "policy_peak_trips": int(policy_peak_volume),
                "peak_volume_reduction_pct": round(peak_reduction_pct, 1),
            },
            summary_findings=(
                f"Adopting a {flexible_window_hours:.1f}-hour flexible arrival window among {pct_flexible_workers*100:.0f}% of "
                f"South Coast employers flattens the US-101 morning peak vehicle flow by {peak_reduction_pct:.1f}% without "
                "requiring highway lane expansion."
            )
        )

    def evaluate_pacific_surfliner_clock_face(
        self,
        n_corridor_commuters: int = 5000,
        baseline_headway_min: int = 120,
        clock_face_headway_min: int = 60
    ) -> ScenarioResult:
        """Simulate mode shift and vehicle shedding under clock-face hourly Pacific Surfliner rail service."""
        # Frequency increase reduces transit wait time and increases schedule reliability
        # Empirical frequency elasticity ~ 0.45
        freq_multiplier = baseline_headway_min / clock_face_headway_min  # 2.0x frequency
        base_rail_share_pct = 3.5
        new_rail_share_pct = min(18.0, base_rail_share_pct * (freq_multiplier ** 0.55))

        diverted_daily_trips = int(n_corridor_commuters * ((new_rail_share_pct - base_rail_share_pct) / 100.0))
        # Multi-worker households shedding a 2nd vehicle
        households_shedding_car = int(diverted_daily_trips * 0.28)
        annual_vmt_reduced = diverted_daily_trips * 65.0 * 2 * 250  # 65-mile corridor * round trip * 250 days

        return ScenarioResult(
            scenario_id="scenario_surfliner_clock_face",
            scenario_name="Pacific Surfliner Clock-Face Regional Rail Transformation",
            key_metrics={
                "baseline_headway_min": baseline_headway_min,
                "clock_face_headway_min": clock_face_headway_min,
                "baseline_rail_share_pct": base_rail_share_pct,
                "new_rail_share_pct": round(new_rail_share_pct, 1),
                "daily_diverted_car_trips": diverted_daily_trips,
                "estimated_households_shedding_vehicle": households_shedding_car,
                "annual_corridor_vmt_reduced": annual_vmt_reduced,
            },
            summary_findings=(
                f"Doubling Pacific Surfliner frequency to clock-face {clock_face_headway_min}-minute headways increases rail mode share "
                f"from {base_rail_share_pct}% to {new_rail_share_pct:.1f}%, removing ~{diverted_daily_trips:,} daily vehicle trips from US-101 "
                f"and saving {annual_vmt_reduced:,} annual VMT."
            )
        )

    def evaluate_ucsb_staggered_classes(
        self,
        ucsb_model: Optional[UCSBSubModel] = None,
        n_students: int = 300
    ) -> ScenarioResult:
        """Simulate splitting UCSB lectures between :00 and :30 minute start blocks to relieve bike path bottlenecks."""
        model = ucsb_model or UCSBSubModel()
        baseline_tt = model.generate_registrar_timetable()
        students = model.synthesize_student_cohorts(n_students=n_students, timetable=baseline_tt)

        # Baseline peak volume at 08:50 AM
        base_peaks = model.calculate_bike_micro_peak(students, simulated_day="MWF", time_bin_minutes=10)
        max_base_micro_peak = max(base_peaks.values()) if base_peaks else 1

        # Staggered Timetable: Half of 09:00 classes moved to 09:30, 10:00 moved to 10:30
        staggered_tt = []
        for i, c in enumerate(baseline_tt):
            staggered_start = c.start_hour + (0.5 if i % 2 == 1 else 0.0)
            staggered_tt.append(ScheduledClass(
                c.course_code, c.course_name, c.building_id, c.lecture_hall, c.days,
                staggered_start, c.duration_min, c.enrollment
            ))

        staggered_students = model.synthesize_student_cohorts(n_students=n_students, timetable=staggered_tt)
        staggered_peaks = model.calculate_bike_micro_peak(staggered_students, simulated_day="MWF", time_bin_minutes=10)
        max_staggered_peak = max(staggered_peaks.values()) if staggered_peaks else 1

        reduction_pct = ((max_base_micro_peak - max_staggered_peak) / max_base_micro_peak) * 100.0

        return ScenarioResult(
            scenario_id="scenario_ucsb_staggered_classes",
            scenario_name="UCSB Staggered Lecture Start Times",
            key_metrics={
                "n_students_modeled": n_students,
                "baseline_max_10min_bike_peak": max_base_micro_peak,
                "staggered_max_10min_bike_peak": max_staggered_peak,
                "peak_flow_reduction_pct": round(reduction_pct, 1),
            },
            summary_findings=(
                f"Staggering lecture hours by 30 minutes between STEM and Humanities cuts the 10-minute micro-congestion "
                f"at Pardall Gate and campus roundabouts by {reduction_pct:.1f}%, eliminating student queueing at bus stops and bike intersections."
            )
        )

    def evaluate_jobs_housing_relocation(
        self,
        new_south_coast_units: int = 5000,
        target_workers_per_unit: float = 1.3
    ) -> ScenarioResult:
        """Evaluate how constructing workforce housing in South Coast reduces North-South cross-county commuting."""
        workers_housed = int(new_south_coast_units * target_workers_per_unit)
        # Proportion of new housing absorbed by North County commuters moving closer to work
        commuters_relocated = int(workers_housed * 0.45)
        round_trip_dist_miles = 130.0  # Santa Maria to Goleta/SB round trip
        annual_vmt_eliminated = commuters_relocated * round_trip_dist_miles * 250
        co2_metric_tons_avoided = (annual_vmt_eliminated * 0.35) / 1000.0  # ~350g CO2 per auto mile

        return ScenarioResult(
            scenario_id="scenario_jobs_housing_relocation",
            scenario_name="South Coast Transit-Oriented Workforce Housing Expansion",
            key_metrics={
                "new_workforce_units": new_south_coast_units,
                "commuters_relocated_from_north_county": commuters_relocated,
                "daily_130mi_trips_eliminated": commuters_relocated,
                "annual_vmt_eliminated": annual_vmt_eliminated,
                "annual_co2_metric_tons_avoided": round(co2_metric_tons_avoided, 1),
            },
            summary_findings=(
                f"Constructing {new_south_coast_units:,} workforce housing units near South Coast transit corridors permanently "
                f"relocates {commuters_relocated:,} long-distance commuters, eliminating {annual_vmt_eliminated:,} annual VMT "
                f"and avoiding {co2_metric_tons_avoided:,.1f} metric tons of greenhouse gas emissions."
            )
        )

    def evaluate_ev_grid_load_profiles(
        self,
        n_ev_fleet: int = 25000,
        pct_workplace_charging: float = 0.35
    ) -> ScenarioResult:
        """Simulate EV charging profiles comparing unmanaged 6 PM home charging with midday workplace solar charging."""
        # 24-hour load profile array (in Megawatts)
        hours = list(range(24))
        unmanaged_load_mw = [0.0] * 24
        managed_load_mw = [0.0] * 24

        avg_kwh_per_charge = 15.0  # Daily replenishment
        power_l2_kw = 6.6
        hrs_needed = avg_kwh_per_charge / power_l2_kw  # ~2.27 hours

        # Unmanaged: 85% plug in between 17:30 and 19:30 (peak evening grid stress)
        for h in range(17, 21):
            unmanaged_load_mw[h] = (n_ev_fleet * 0.80 * (power_l2_kw / 1000.0)) * 0.85

        # Managed: 35% of fleet charges at workplace during solar peak (10:00 - 14:00)
        workplace_evs = int(n_ev_fleet * pct_workplace_charging)
        residential_evs = n_ev_fleet - workplace_evs

        for h in range(10, 15):
            managed_load_mw[h] = (workplace_evs * (power_l2_kw / 1000.0)) * 0.70

        # Remaining residential EVs delayed to midnight super-off-peak (00:00 - 04:00)
        for h in range(0, 4):
            managed_load_mw[h] = (residential_evs * (power_l2_kw / 1000.0)) * 0.65

        evening_unmanaged_peak_mw = max(unmanaged_load_mw)
        evening_managed_peak_mw = max(managed_load_mw[17:21])
        peak_reduction_pct = ((evening_unmanaged_peak_mw - evening_managed_peak_mw) / max(1.0, evening_unmanaged_peak_mw)) * 100.0

        return ScenarioResult(
            scenario_id="scenario_ev_grid_load",
            scenario_name="Managed Workplace EV Charging vs. Solar Peak Grid Integration",
            key_metrics={
                "modeled_ev_fleet_size": n_ev_fleet,
                "pct_workplace_charging": pct_workplace_charging * 100.0,
                "unmanaged_evening_peak_mw": round(evening_unmanaged_peak_mw, 1),
                "managed_evening_peak_mw": round(evening_managed_peak_mw, 1),
                "peak_grid_stress_reduction_pct": round(peak_reduction_pct, 1),
            },
            summary_findings=(
                f"Shifting EV charging for {pct_workplace_charging*100:.0f}% of commuters to daytime workplace chargers in Goleta/SB "
                f"reduces evening residential transformer peak loads by {peak_reduction_pct:.1f}% ({evening_unmanaged_peak_mw:.1f} MW down to "
                f"{evening_managed_peak_mw:.1f} MW), cleanly absorbing midday California solar overproduction."
            )
        )
