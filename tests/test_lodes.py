"""Test LEHD LODES ingestion and workplace assignment."""

import pytest
import pandas as pd
from nextgen_abm.population.lodes import LODESDataLoader


def test_lodes_od_generation():
    loader = LODESDataLoader()
    od_df = loader.generate_synthetic_lodes_od()
    assert isinstance(od_df, pd.DataFrame)
    assert len(od_df) >= 8

    # Verify column presence
    assert "h_geocode" in od_df.columns
    assert "w_geocode" in od_df.columns
    assert "S000" in od_df.columns  # Total jobs
    assert "SE01" in od_df.columns  # Low wage
    assert "SE02" in od_df.columns  # Mid wage
    assert "SE03" in od_df.columns  # High wage

    # Verify Santa Maria to Goleta/SB flow is present
    sm_to_goleta = od_df[
        (od_df["h_geocode"] == "060830024011") & (od_df["w_geocode"] == "060830017011")
    ]
    assert not sm_to_goleta.empty
    assert sm_to_goleta.iloc[0]["S000"] > 5000


def test_workplace_sampling():
    loader = LODESDataLoader()
    od_df = loader.generate_synthetic_lodes_od()

    # Worker living in Santa Maria
    home_geocode = "060830024011"
    workplace = loader.sample_workplace_for_residence(
        home_geocode=home_geocode,
        od_df=od_df,
        income_bracket="high"
    )
    assert isinstance(workplace, str)
    # The sampled workplace must be one of the known destinations
    assert workplace in ["060830017011", "060830010011", "060830024011"]
