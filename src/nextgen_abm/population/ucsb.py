"""Dedicated UCSB and Isla Vista student sub-model with course schedule micro-peaks and bike physics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ScheduledClass:
    course_code: str
    course_name: str
    building_id: str
    lecture_hall: str
    days: str          # "MWF" or "TR"
    start_hour: float  # e.g. 9.0 (09:00 AM), 9.5 (09:30 AM)
    duration_min: float
    enrollment: int


@dataclass
class StudentProfile:
    student_id: str
    cohort: str        # "freshman_dorm", "iv_undergrad", "univ_apartment", "grad_commuter"
    residence_building_id: str
    roommate_ids: List[str] = field(default_factory=list)
    has_bicycle: bool = True
    has_skateboard: bool = False
    has_car: bool = False
    enrolled_classes: List[ScheduledClass] = field(default_factory=list)
    transit_pass_active: bool = True  # UCSB students ride Santa Barbara MTD free


class UCSBSubModel:
    """Simulates UCSB student scheduling, Isla Vista housing, and campus micro-mobility."""

    # Major lecture halls and campus anchors
    CAMPUS_ANCHORS = {
        "campbell_hall": {"building_id": "bldg_ucsb_campbell", "name": "Campbell Hall", "capacity": 860, "lon": -119.845, "lat": 34.415},
        "chem_1179": {"building_id": "bldg_ucsb_chem", "name": "Chemistry 1179", "capacity": 320, "lon": -119.844, "lat": 34.414},
        "davidson_library": {"building_id": "bldg_ucsb_library", "name": "Davidson Library", "capacity": 2500, "lon": -119.846, "lat": 34.414},
        "bioeng_hall": {"building_id": "bldg_ucsb_bioeng", "name": "BioEngineering Bldg", "capacity": 250, "lon": -119.842, "lat": 34.414},
        "reccen": {"building_id": "bldg_ucsb_reccen", "name": "Recreation Center", "capacity": 800, "lon": -119.851, "lat": 34.416},
    }

    def generate_registrar_timetable(self) -> List[ScheduledClass]:
        """Generate typical UCSB quarter timetable representing MWF and TR lecture blocks."""
        classes = [
            # MWF Morning & Midday Lectures (50-min blocks)
            ScheduledClass("CHEM_109A", "Organic Chemistry", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 9.0, 50.0, 800),
            ScheduledClass("MATH_4A", "Linear Algebra", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 10.0, 50.0, 750),
            ScheduledClass("COMM_1", "Intro to Communication", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 11.0, 50.0, 850),
            ScheduledClass("ECON_1", "Principles of Economics", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 13.0, 50.0, 800),
            ScheduledClass("BIO_1A", "Intro Biology", "bldg_ucsb_chem", "Chemistry 1179", "MWF", 9.0, 50.0, 300),
            ScheduledClass("PHYS_1", "Physics for Engineers", "bldg_ucsb_chem", "Chemistry 1179", "MWF", 11.0, 50.0, 310),

            # TR Lectures (75-min blocks)
            ScheduledClass("CMPSC_16", "Problem Solving C++", "bldg_ucsb_bioeng", "BioEng 1001", "TR", 9.5, 75.0, 220),
            ScheduledClass("HIST_17B", "American History", "bldg_ucsb_campbell", "Campbell Hall", "TR", 11.0, 75.0, 700),
            ScheduledClass("PSYCH_1", "Intro Psychology", "bldg_ucsb_campbell", "Campbell Hall", "TR", 14.0, 75.0, 820),
        ]
        return classes

    def synthesize_student_cohorts(
        self,
        n_students: int = 200,
        timetable: Optional[List[ScheduledClass]] = None
    ) -> List[StudentProfile]:
        """Synthesize students across Isla Vista houses, dorms, and university apartments."""
        timetable = timetable or self.generate_registrar_timetable()
        students: List[StudentProfile] = []

        # Cohort distribution: 25% Dorm Freshmen, 50% IV Undergrads, 15% Univ Apartments, 10% Commuters
        cohort_types = ["freshman_dorm", "iv_undergrad", "univ_apartment", "grad_commuter"]
        cohort_weights = [0.25, 0.50, 0.15, 0.10]

        for i in range(n_students):
            s_id = f"ucsb_{i:05d}"
            cohort = np.random.choice(cohort_types, p=cohort_weights)

            if cohort == "freshman_dorm":
                res_bldg = "bldg_iv_003"  # Santa Catalina / Dorms
                has_car = False           # Dorm residents cannot buy campus parking permits!
                has_bike = True
                has_skate = np.random.rand() < 0.25
            elif cohort == "iv_undergrad":
                res_bldg = np.random.choice(["bldg_iv_001", "bldg_iv_002"])
                has_car = np.random.rand() < 0.20   # Street parking in IV is severely limited
                has_bike = True
                has_skate = np.random.rand() < 0.15
            elif cohort == "univ_apartment":
                res_bldg = "bldg_iv_002"
                has_car = np.random.rand() < 0.35
                has_bike = True
                has_skate = False
            else:
                res_bldg = "bldg_sm_res_001"  # Commuter living in Santa Maria / Goleta
                has_car = True
                has_bike = np.random.rand() < 0.20
                has_skate = False

            # Assign 2 to 3 enrolled classes
            enrolled = list(np.random.choice(timetable, size=min(3, len(timetable)), replace=False))

            student = StudentProfile(
                student_id=s_id,
                cohort=cohort,
                residence_building_id=res_bldg,
                has_bicycle=has_bike,
                has_skateboard=has_skate,
                has_car=has_car,
                enrolled_classes=enrolled,
                transit_pass_active=True
            )
            students.append(student)

        return students

    def calculate_bike_micro_peak(
        self,
        students: List[StudentProfile],
        simulated_day: str = "MWF",
        time_bin_minutes: int = 10
    ) -> Dict[str, int]:
        """Compute the arrival micro-peak volumes on the Pardall / UCen bike path network.
        
        Shows the pronounced peak 10 minutes before the top of the hour!
        """
        bins = {}
        for hour in range(8, 18):
            for minute in range(0, 60, time_bin_minutes):
                time_key = f"{hour:02d}:{minute:02d}"
                bins[time_key] = 0

        for s in students:
            if not s.has_bicycle:
                continue
            for c in s.enrolled_classes:
                if simulated_day in c.days:
                    # Departs ~10 to 15 mins before class start
                    start_min = int(c.start_hour * 60)
                    dep_min = start_min - 10
                    h = (dep_min // 60)
                    m = (dep_min % 60) // time_bin_minutes * time_bin_minutes
                    bin_key = f"{h:02d}:{m:02d}"
                    if bin_key in bins:
                        bins[bin_key] += 1

        return bins

    def evaluate_mode_choice(
        self,
        student: StudentProfile,
        origin_id: str,
        destination_id: str,
        dist_miles: float
    ) -> str:
        """Determines student mode choice based on strict campus constraints."""
        # Freshmen in dorms traveling to campus
        if student.cohort == "freshman_dorm":
            if dist_miles < 0.5:
                return "walk"
            return "bike" if student.has_bicycle else "walk"

        # Isla Vista undergrads
        if student.cohort == "iv_undergrad":
            if dist_miles > 5.0 and student.has_car:
                return "auto"
            elif dist_miles > 3.0:
                return "bus"  # Free MTD Line 24x
            elif dist_miles < 0.4:
                return "walk"
            elif student.has_bicycle:
                return "bike"
            elif student.has_skateboard:
                return "skateboard"
            return "walk"

        # Commuter students from North County
        if student.cohort == "grad_commuter":
            return "auto" if student.has_car else "bus"

        return "bike" if student.has_bicycle else "walk"
