"""LEHD LODES (v8) Origin-Destination and Workplace Ingestion for Santa Barbara County."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class CommuteFlow:
    """Aggregated commute flow between origin and destination zones."""
    home_zone: str
    work_zone: str
    total_jobs: int
    low_wage_jobs: int    # earnings <= $1,250/mo
    mid_wage_jobs: int    # $1,251 - $3,333/mo
    high_wage_jobs: int   # > $3,333/mo


class LODESDataLoader:
    """Loads and processes LEHD LODES employment statistics for Santa Barbara County."""

    SANTA_BARBARA_FIPS = "06083"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/lodes")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_lodes_od(self) -> pd.DataFrame:
        """Generate representative LODES Origin-Destination flows for Santa Barbara County.
        
        Empirical Santa Barbara dynamics:
        - Major job centers:
            - Goleta / Tech Corridor / UCSB: ~40,000 jobs
            - City of Santa Barbara (Downtown, Waterfront, Cottage Hospital): ~60,000 jobs
            - Santa Maria / Orcutt (Retail, Agriculture, Manufacturing): ~45,000 jobs
            - Lompoc & Vandenberg Space Force Base: ~18,000 jobs
        - The Gaviota commute mismatch:
            - ~12,000 - 15,000 workers reside in Santa Maria / Lompoc but commute to Goleta / SB!
        """
        flows = [
            # Intra-South Coast Commutes
            {"h_geocode": "060830010011", "w_geocode": "060830010011", "h_name": "Santa Barbara City", "w_name": "Downtown SB", "S000": 8500, "SE01": 1500, "SE02": 2500, "SE03": 4500},
            {"h_geocode": "060830017011", "w_geocode": "060830010011", "h_name": "Goleta", "w_name": "Downtown SB", "S000": 4200, "SE01": 600, "SE02": 1200, "SE03": 2400},
            {"h_geocode": "060830010011", "w_geocode": "060830017011", "h_name": "Santa Barbara City", "w_name": "Goleta Tech / UCSB", "S000": 5100, "SE01": 800, "SE02": 1500, "SE03": 2800},
            {"h_geocode": "060830029011", "w_geocode": "060830017011", "h_name": "Isla Vista", "w_name": "UCSB Campus", "S000": 3400, "SE01": 1800, "SE02": 1100, "SE03": 500},

            # The Inter-Regional Jobs-Housing Mismatch (North County -> South Coast Commuters)
            {"h_geocode": "060830024011", "w_geocode": "060830017011", "h_name": "Santa Maria", "w_name": "Goleta Tech", "S000": 6200, "SE01": 900, "SE02": 2300, "SE03": 3000},
            {"h_geocode": "060830024011", "w_geocode": "060830010011", "h_name": "Santa Maria", "w_name": "Downtown SB / Cottage Hosp", "S000": 5800, "SE01": 1200, "SE02": 2600, "SE03": 2000},
            {"h_geocode": "060830027011", "w_geocode": "060830017011", "h_name": "Lompoc", "w_name": "Goleta Tech", "S000": 2400, "SE01": 400, "SE02": 1000, "SE03": 1000},

            # Local North County Commutes
            {"h_geocode": "060830024011", "w_geocode": "060830024011", "h_name": "Santa Maria", "w_name": "Santa Maria Local", "S000": 16500, "SE01": 5500, "SE02": 6500, "SE03": 4500},
            {"h_geocode": "060830027011", "w_geocode": "060830028011", "h_name": "Lompoc", "w_name": "Vandenberg Space Force Base", "S000": 4800, "SE01": 600, "SE02": 1800, "SE03": 2400},
        ]
        return pd.DataFrame(flows)

    def sample_workplace_for_residence(
        self,
        home_geocode: str,
        od_df: pd.DataFrame,
        income_bracket: str = "high"  # "low", "mid", "high"
    ) -> str:
        """Sample a workplace geocode for a resident worker based on LODES probabilities."""
        matching_flows = od_df[od_df["h_geocode"] == home_geocode]
        if matching_flows.empty:
            # Fallback to general destination distribution
            matching_flows = od_df

        weight_col = "SE03" if income_bracket == "high" else ("SE02" if income_bracket == "mid" else "SE01")
        weights = matching_flows[weight_col].astype(float)
        total_w = weights.sum()

        if total_w <= 0:
            probs = [1.0 / len(matching_flows)] * len(matching_flows)
        else:
            probs = (weights / total_w).tolist()

        selected_row = matching_flows.sample(n=1, weights=probs).iloc[0]
        return selected_row["w_geocode"]
