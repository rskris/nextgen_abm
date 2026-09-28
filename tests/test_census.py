"""Test Census ACS PUMS ingestion and marginal targets."""

import pytest
import pandas as pd
from nextgen_abm.population.census import ACSDataLoader


def test_synthetic_pums_sample_generation():
    loader = ACSDataLoader()
    sample = loader.generate_synthetic_pums_sample(n_households=100)
    assert "households" in sample
    assert "persons" in sample

    hhs = sample["households"]
    persons = sample["persons"]

    assert len(hhs) == 100
    assert len(persons) >= 100

    # Verify key PUMS columns exist
    assert "SERIALNO" in hhs.columns
    assert "PUMA" in hhs.columns
    assert "VEH" in hhs.columns
    assert "HINCP" in hhs.columns
    assert "VEH_TYPES" in hhs.columns

    assert "PERSON_ID" in persons.columns
    assert "AGEP" in persons.columns
    assert "ESR" in persons.columns
    assert "STUDENT_TYPE" in persons.columns
    assert "TELEWORK_DAYS" in persons.columns

    # Verify both PUMAs are represented (North County 08301 and South Coast 08302)
    pumas = set(hhs["PUMA"].unique())
    assert "08301" in pumas
    assert "08302" in pumas


def test_block_group_marginals():
    loader = ACSDataLoader()
    marginals = loader.generate_block_group_marginals()
    assert isinstance(marginals, pd.DataFrame)
    assert len(marginals) >= 4
    assert "target_pop" in marginals.columns
    assert "avg_veh_per_hh" in marginals.columns
    # Check Isla Vista has high student percentage and low auto ownership
    iv_row = marginals[marginals["bg_id"] == "060830029011"].iloc[0]
    assert iv_row["pct_student"] > 0.70
    assert iv_row["avg_veh_per_hh"] < 1.0
