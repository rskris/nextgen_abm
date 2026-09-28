"""Test dynamic life-history evolution and persistent agent trajectories."""

import pytest
from nextgen_abm.population.census import ACSDataLoader, SyntheticHousehold, SyntheticPerson
from nextgen_abm.population.life_history import LifeHistoryEngine, PersistentAgent


def test_persistent_population_initialization():
    engine = LifeHistoryEngine(random_seed=42)

    # Create dummy households
    p1 = SyntheticPerson(person_id="p1", household_id="hh1", age=21, sex="1", is_worker=False, is_student=True, student_type="ucsb_undergrad")
    hh1 = SyntheticHousehold(household_id="hh1", puma="08302", block_group_id="060830029011", home_building_id="bldg_iv_001", household_size=1, num_vehicles=0, members=[p1])

    agents = engine.initialize_persistent_population([hh1], base_year=2024)
    assert "p1" in agents
    agent = agents["p1"]
    assert agent.person_id == "p1"
    assert agent.current_age == 21
    assert len(agent.trajectory) == 1
    assert agent.trajectory[0].year == 2024
    assert agent.trajectory[0].is_student is True


def test_multi_year_simulation_progression():
    engine = LifeHistoryEngine(random_seed=42)

    # Create cohort of 20 college students and 20 working families
    loader = ACSDataLoader()
    sample = loader.generate_synthetic_pums_sample(n_households=50)
    hhs_df = sample["households"]
    persons_df = sample["persons"]

    # Assemble SyntheticHousehold objects
    households = []
    for _, h_row in hhs_df.iterrows():
        hh_id = h_row["SERIALNO"]
        p_rows = persons_df[persons_df["SERIALNO"] == hh_id]
        members = [
            SyntheticPerson(
                person_id=pr["PERSON_ID"],
                household_id=hh_id,
                age=pr["AGEP"],
                sex=str(pr["SEX"]),
                is_worker=pr["ESR"] == 1,
                is_student=pr["IS_STUDENT"],
                student_type=pr["STUDENT_TYPE"],
                telework_frequency_days=pr["TELEWORK_DAYS"]
            )
            for _, pr in p_rows.iterrows()
        ]
        hh = SyntheticHousehold(
            household_id=hh_id,
            puma=h_row["PUMA"],
            block_group_id=h_row["BLOCK_GROUP"],
            home_building_id="bldg_iv_001" if "08302" in h_row["PUMA"] else "bldg_sm_res_001",
            household_size=h_row["NP"],
            num_vehicles=h_row["VEH"],
            vehicle_types=h_row["VEH_TYPES"].split(",") if h_row["VEH_TYPES"] else [],
            members=members
        )
        households.append(hh)

    agents = engine.initialize_persistent_population(households, base_year=2024)
    assert len(agents) > 50

    # Step simulation 3 years forward (2024 -> 2025 -> 2026 -> 2027)
    engine.step_year(agents, households, from_year=2024)
    engine.step_year(agents, households, from_year=2025)
    grads, jobs, relocs = engine.step_year(agents, households, from_year=2026)

    # Check agent persistence
    sample_agent = next(iter(agents.values()))
    assert len(sample_agent.trajectory) == 4  # 2024, 2025, 2026, 2027
    years = [s.year for s in sample_agent.trajectory]
    assert years == [2024, 2025, 2026, 2027]
    assert sample_agent.trajectory[3].age == sample_agent.trajectory[0].age + 3
