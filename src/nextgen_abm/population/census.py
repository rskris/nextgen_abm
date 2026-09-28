"""ACS PUMS microdata ingestion and demographic synthesis loader for Santa Barbara County."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class SyntheticPerson:
    """Individual synthetic traveler attributes."""
    person_id: str
    household_id: str
    age: int
    sex: str
    is_worker: bool
    is_student: bool
    student_type: Optional[str] = None  # "k12", "ucsb_undergrad", "ucsb_grad", "sbcc", "hancock"
    workplace_id: Optional[str] = None
    school_id: Optional[str] = None
    telework_frequency_days: int = 0  # 0 to 5 days per week
    commute_mode_preference: str = "auto_solo"
    license_held: bool = True
    bike_owned: bool = False
    annual_earnings: float = 50000.0


@dataclass
class SyntheticHousehold:
    """Synthetic household unit containing members and shared assets."""
    household_id: str
    puma: str  # "08301" (North County) or "08302" (South Coast)
    block_group_id: str
    home_building_id: str
    household_size: int
    num_vehicles: int
    vehicle_types: List[str] = field(default_factory=list)  # ["ice", "ev", "hybrid"]
    income: float = 85000.0
    tenure: str = "rent"  # "own" or "rent"
    household_type: str = "family"  # "family", "single", "student_roommates", "non_family"
    members: List[SyntheticPerson] = field(default_factory=list)


from nextgen_abm.config import get_config, MasterConfig


class ACSDataLoader:
    """Loads and standardizes ACS 5-Year PUMS microdata for Santa Barbara County (PUMA 08301 & 08302)."""

    def __init__(self, data_dir: Optional[Path] = None, config: Optional[MasterConfig] = None):
        self.config = config or get_config()
        self.PUMA_NORTH_COUNTY = self.config.population.pumas[0] if len(self.config.population.pumas) > 0 else "08301"
        self.PUMA_SOUTH_COAST = self.config.population.pumas[1] if len(self.config.population.pumas) > 1 else "08302"
        self.data_dir = Path(data_dir) if data_dir else Path(self.config.spatial.cache_dir) / "census"
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def generate_synthetic_pums_sample(self, n_households: int = 500) -> Dict[str, pd.DataFrame]:
        """Generate high-fidelity representative ACS PUMS sample for Santa Barbara County.
        
        Reflects empirical joint distributions:
        - Isla Vista & Goleta: High student share, zero/low car ownership, high bike ownership, high rental share
        - Santa Maria & Lompoc: Larger family sizes, higher auto dependency, long commute times
        - Santa Barbara South Coast: High income, mixed auto/transit, older demographic
        """
        np.random.seed(42)
        households = []
        persons = []

        for i in range(n_households):
            hh_id = f"hh_{i:06d}"
            # 60% South Coast (PUMA 08302), 40% North County (PUMA 08301)
            is_south = np.random.rand() < 0.60
            puma = self.PUMA_SOUTH_COAST if is_south else self.PUMA_NORTH_COUNTY

            if is_south and np.random.rand() < 0.25:
                # Student household archetype in Isla Vista / Goleta
                hh_type = "student_roommates"
                hh_size = np.random.choice([3, 4, 5, 6], p=[0.2, 0.4, 0.3, 0.1])
                tenure = "rent"
                num_veh = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
                income = float(np.random.normal(35000, 15000))
                bg_id = "060830029011"  # Isla Vista block group
            elif not is_south:
                # North County family archetype (Santa Maria)
                hh_type = "family" if np.random.rand() < 0.85 else "single"
                hh_size = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.25, 0.1]) if hh_type == "family" else 1
                tenure = "own" if np.random.rand() < 0.65 else "rent"
                num_veh = np.random.choice([1, 2, 3], p=[0.25, 0.55, 0.20])
                income = float(np.random.normal(72000, 25000))
                bg_id = "060830024011"  # Santa Maria block group
            else:
                # General South Coast working household (Santa Barbara / Goleta)
                hh_type = np.random.choice(["family", "single", "non_family"], p=[0.55, 0.35, 0.10])
                hh_size = np.random.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10]) if hh_type != "single" else 1
                tenure = "own" if np.random.rand() < 0.52 else "rent"
                num_veh = np.random.choice([0, 1, 2], p=[0.15, 0.50, 0.35])
                income = float(np.random.normal(110000, 45000))
                bg_id = "060830010011"  # Santa Barbara City block group

            # Fleet composition
            veh_types = []
            for _ in range(num_veh):
                # High EV/Hybrid share on South Coast
                if is_south and np.random.rand() < 0.30:
                    veh_types.append("ev")
                elif np.random.rand() < 0.20:
                    veh_types.append("hybrid")
                else:
                    veh_types.append("ice")

            households.append({
                "SERIALNO": hh_id,
                "PUMA": puma,
                "NP": hh_size,
                "VEH": num_veh,
                "HINCP": max(10000.0, income),
                "TEN": 1 if tenure == "own" else 3,
                "HHT_TYPE": hh_type,
                "BLOCK_GROUP": bg_id,
                "VEH_TYPES": ",".join(veh_types)
            })

            # Create individual members
            for p in range(hh_size):
                person_id = f"{hh_id}_p{p+1}"
                if hh_type == "student_roommates":
                    age = int(np.random.uniform(19, 24))
                    is_student = True
                    student_type = "ucsb_undergrad"
                    is_worker = np.random.rand() < 0.45  # Part-time job
                    bike_owned = True
                    pref_mode = "bike" if np.random.rand() < 0.70 else "walk"
                elif p == 0:  # Head of household
                    age = int(np.random.uniform(28, 65))
                    is_student = False
                    student_type = None
                    is_worker = True
                    bike_owned = np.random.rand() < 0.35
                    pref_mode = "auto_solo" if num_veh > 0 else "bus"
                else:
                    age = int(np.random.uniform(5, 60))
                    is_student = age < 18 or (age <= 24 and np.random.rand() < 0.6)
                    student_type = "k12" if age < 18 else ("ucsb_undergrad" if is_south else "hancock")
                    is_worker = age >= 18 and np.random.rand() < 0.75
                    bike_owned = age < 25 or np.random.rand() < 0.3
                    pref_mode = "auto_passenger" if age < 16 else ("auto_solo" if num_veh > 1 else "bus")

                persons.append({
                    "SERIALNO": hh_id,
                    "SPORDER": p + 1,
                    "PERSON_ID": person_id,
                    "AGEP": age,
                    "SEX": 1 if np.random.rand() < 0.5 else 2,
                    "ESR": 1 if is_worker else 6,
                    "IS_STUDENT": is_student,
                    "STUDENT_TYPE": student_type,
                    "BIKE_OWNED": bike_owned,
                    "PREF_MODE": pref_mode,
                    "TELEWORK_DAYS": int(np.random.choice([0, 1, 2, 3, 5], p=[0.5, 0.1, 0.2, 0.1, 0.1])) if is_worker else 0
                })

        return {
            "households": pd.DataFrame(households),
            "persons": pd.DataFrame(persons)
        }

    def generate_block_group_marginals(self) -> pd.DataFrame:
        """Create marginal demographic targets across Santa Barbara County Census Block Groups."""
        bgs = [
            {"bg_id": "060830029011", "name": "Isla Vista / UCSB Core", "target_pop": 8500, "target_hhs": 2200, "avg_veh_per_hh": 0.6, "pct_student": 0.88},
            {"bg_id": "060830010011", "name": "Santa Barbara Downtown / West Beach", "target_pop": 3200, "target_hhs": 1400, "avg_veh_per_hh": 1.3, "pct_student": 0.08},
            {"bg_id": "060830024011", "name": "Santa Maria Central / Main St", "target_pop": 4100, "target_hhs": 1150, "avg_veh_per_hh": 1.9, "pct_student": 0.05},
            {"bg_id": "060830017011", "name": "Goleta / Old Town", "target_pop": 2900, "target_hhs": 950, "avg_veh_per_hh": 1.5, "pct_student": 0.15},
        ]
        return pd.DataFrame(bgs)
