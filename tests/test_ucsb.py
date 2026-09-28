"""Test UCSB student sub-model, timetable micro-peaks, and active transport rules."""

import pytest
from nextgen_abm.population.ucsb import UCSBSubModel, ScheduledClass, StudentProfile


def test_ucsb_timetable_generation():
    model = UCSBSubModel()
    classes = model.generate_registrar_timetable()

    assert len(classes) >= 8
    course_codes = [c.course_code for c in classes]
    assert "CHEM_109A" in course_codes
    assert "CMPSC_16" in course_codes

    # Verify Campbell Hall capacity and start hours
    campbell_classes = [c for c in classes if c.lecture_hall == "Campbell Hall"]
    assert len(campbell_classes) >= 4
    for c in campbell_classes:
        assert c.enrollment >= 700


def test_student_cohort_synthesis():
    model = UCSBSubModel()
    students = model.synthesize_student_cohorts(n_students=100)

    assert len(students) == 100
    cohorts = set(s.cohort for s in students)
    assert "freshman_dorm" in cohorts
    assert "iv_undergrad" in cohorts

    # Check dorm freshmen rules: no car allowed
    dorm_students = [s for s in students if s.cohort == "freshman_dorm"]
    assert len(dorm_students) > 0
    for s in dorm_students:
        assert s.has_car is False
        assert s.transit_pass_active is True
        assert s.residence_building_id == "bldg_iv_003"


def test_bike_micro_peak_timing():
    model = UCSBSubModel()
    timetable = [
        ScheduledClass("CHEM_109A", "Chem", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 9.0, 50.0, 500),
        ScheduledClass("MATH_4A", "Math", "bldg_ucsb_campbell", "Campbell Hall", "MWF", 10.0, 50.0, 500),
    ]
    students = model.synthesize_student_cohorts(n_students=80, timetable=timetable)

    peaks = model.calculate_bike_micro_peak(students, simulated_day="MWF", time_bin_minutes=10)

    # 10 minutes before 9:00 AM is 08:50 AM
    assert "08:50" in peaks
    assert "09:50" in peaks

    # There should be concentrated trips in the 08:50 and 09:50 bins
    assert peaks["08:50"] > 0 or peaks["09:50"] > 0


def test_student_mode_choice_rules():
    model = UCSBSubModel()
    freshman = StudentProfile(
        student_id="s1",
        cohort="freshman_dorm",
        residence_building_id="bldg_iv_003",
        has_bicycle=True,
        has_car=False
    )
    # Freshman traveling 0.8 miles to Campbell Hall must walk or bike, cannot drive
    mode = model.evaluate_mode_choice(freshman, "bldg_iv_003", "bldg_ucsb_campbell", dist_miles=0.8)
    assert mode in ["bike", "walk"]
    assert mode != "auto"

    # IV undergrad traveling to downtown Santa Barbara (10 miles) takes free MTD bus
    iv_student = StudentProfile(
        student_id="s2",
        cohort="iv_undergrad",
        residence_building_id="bldg_iv_001",
        has_bicycle=True,
        has_car=False,
        transit_pass_active=True
    )
    mode_far = model.evaluate_mode_choice(iv_student, "bldg_iv_001", "bldg_sb_state_st", dist_miles=10.5)
    assert mode_far == "bus"
