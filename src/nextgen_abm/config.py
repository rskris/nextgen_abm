"""Master configuration system for NextGen ABM Santa Barbara.

Provides a unified, declarative configuration loaded from a top-level YAML file
and percolated downstream across all modules (spatial, population, household,
traffic LTM, equilibrium, policy, and calibration).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional
import yaml
from pydantic import BaseModel, Field


class SpatialConfig(BaseModel):
    """Bounding box, S3 bucket sources, and cache directories."""
    min_lon: float = Field(default=-120.65, description="Santa Barbara County West bounding longitude")
    min_lat: float = Field(default=34.35, description="Santa Barbara County South bounding latitude")
    max_lon: float = Field(default=-119.45, description="Santa Barbara County East bounding longitude")
    max_lat: float = Field(default=35.15, description="Santa Barbara County North bounding latitude")
    s3_bucket: str = Field(default="overturemaps-us-west-2", description="Overture Maps GeoParquet bucket")
    overture_release: str = Field(default="2024-08-20.0", description="Overture release version")
    cache_dir: str = Field(default="data/cache", description="Local DuckDB / GeoParquet cache directory")
    elevation_source: str = Field(default="usgs_3dep", description="Elevation gradient data provider")


class PopulationConfig(BaseModel):
    """Synthetic population, Census ACS PUMS, LEHD LODES, and UCSB student cohort settings."""
    county_fips: str = Field(default="06083", description="Santa Barbara County FIPS code")
    pumas: List[str] = Field(default_factory=lambda: ["08301", "08302"], description="PUMA codes")
    base_year: int = Field(default=2024, description="Base simulation year")
    simulation_years: int = Field(default=5, description="Multi-year dynamic life-history horizon")
    ucsb_target_enrollment: int = Field(default=26500, description="UCSB student enrollment target")
    sample_fraction: float = Field(
        default=0.10,
        description="Key driver scaling fraction for population runs (0.01=smoke test, 0.10=default sample, 1.0=full regional)"
    )


class HouseholdConfig(BaseModel):
    """Vehicle asset ownership, EV battery constraints, and intra-household parameters."""
    ev_battery_capacity_kwh: float = Field(default=75.0, description="Standard EV battery capacity in kWh")
    min_soc_reserve: float = Field(default=0.15, description="Minimum state-of-charge reserve threshold")
    home_charger_kw: float = Field(default=7.2, description="Level-2 home charging power in kW")
    work_charger_kw: float = Field(default=11.0, description="Workplace Level-2 charging power in kW")
    dcfc_charger_kw: float = Field(default=150.0, description="Public DC Fast Charger power in kW")
    joint_meal_reward: float = Field(default=5.0, description="Household utility bonus for synchronized dinner")
    escort_penalty: float = Field(default=1.5, description="Disutility multiplier for escort detours")


class SolverConfig(BaseModel):
    """HiGHS MILP mathematical programming scheduler settings."""
    solver_name: str = Field(default="highs", description="Optimization solver (highs / ortools)")
    time_step_minutes: int = Field(default=15, description="Scheduler discretization interval")
    time_horizon_hours: int = Field(default=24, description="Planning horizon length")
    time_limit_seconds: float = Field(default=10.0, description="Per-schedule solver timeout")
    mip_gap: float = Field(default=0.01, description="HiGHS relative MIP optimality gap")
    vot_car: float = Field(default=20.0, description="Value of travel time by car ($/hour)")
    vot_transit: float = Field(default=12.0, description="Value of travel time by transit ($/hour)")
    vot_bike: float = Field(default=15.0, description="Value of travel time by bike ($/hour)")
    vot_walk: float = Field(default=18.0, description="Value of travel time by walking ($/hour)")
    late_penalty_hourly: float = Field(default=40.0, description="Schedule late arrival penalty ($/hour)")


class LTMConfig(BaseModel):
    """Native Link Transmission Model (LTM) kinematic wave traffic meso-simulator settings."""
    time_step_seconds: float = Field(default=60.0, description="LTM time step for cumulative flow curves")
    horizon_hours: float = Field(default=24.0, description="Traffic simulation duration in hours")
    v_free_freeway_kmh: float = Field(default=105.0, description="Free flow speed on US-101 / SR-217 (km/h)")
    v_free_arterial_kmh: float = Field(default=55.0, description="Free flow speed on arterials (km/h)")
    v_free_bike_kmh: float = Field(default=20.0, description="Free flow speed on UCSB bike paths (km/h)")
    wave_speed_kmh: float = Field(default=20.0, description="Backward shockwave speed w (km/h)")
    jam_density_veh_km: float = Field(default=120.0, description="Jam density k_jam (veh/lane-km)")
    corridor_capacity_freeway_vph: float = Field(default=2000.0, description="Capacity Q_max freeway (veh/lane/h)")
    corridor_capacity_arterial_vph: float = Field(default=800.0, description="Capacity Q_max arterial (veh/lane/h)")
    bike_path_capacity_bph: float = Field(default=1500.0, description="Capacity Q_max UCSB bike paths (bikes/h)")
    ev_base_consumption_kwh_per_km: float = Field(default=0.18, description="Base EV energy consumption (kWh/km)")
    elevation_penalty_kwh_per_meter: float = Field(default=0.0003, description="Elevation energy penalty (kWh/m)")


class EquilibriumConfig(BaseModel):
    """Day-to-Day evolutionary replanning supply-demand equilibrium settings."""
    max_iterations: int = Field(default=5, description="Maximum day-to-day equilibrium iterations")
    replan_fraction: float = Field(default=0.15, description="Fraction of agents replanning each iteration (15%)")
    plan_memory_size: int = Field(default=4, description="Maximum scored plans retained in agent memory")
    logit_beta: float = Field(default=1.0, description="Multinomial logit plan selection sensitivity")
    convergence_tolerance_rel_gap: float = Field(default=0.02, description="Relative convergence gap target")


class CalibrationConfig(BaseModel):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) calibration settings."""
    max_iterations: int = Field(default=10, description="Max SPSA optimization iterations")
    spsa_a: float = Field(default=0.1, description="SPSA step size parameter a")
    spsa_c: float = Field(default=0.05, description="SPSA perturbation parameter c")
    pems_stations: List[str] = Field(
        default_factory=lambda: [
            "US101_Salinas",
            "US101_Carrillo",
            "US101_Fairview",
            "US101_Storke",
            "US101_Carpinteria",
        ],
        description="Target Caltrans PeMS detector stations",
    )
    target_geh: float = Field(default=5.0, description="Maximum GEH statistic threshold for validation")


class ScenarioConfig(BaseModel):
    """Regional policy scenario settings."""
    active_scenario: str = Field(default="baseline", description="Active scenario identifier")
    us101_toll_dollars: float = Field(default=0.0, description="US-101 peak-hour congestion pricing toll")
    surfliner_hourly: bool = Field(default=False, description="Clock-face Pacific Surfliner rail expansion")
    ucsb_staggered_classes: bool = Field(default=False, description="UCSB staggered 15-minute course blocks")
    north_south_remote_pct: float = Field(default=0.0, description="Incentivized remote work fraction")
    ev_tou_load_shift: bool = Field(default=False, description="EV off-peak charging tariff alignment")


class MasterConfig(BaseModel):
    """Single master configuration file percolated downstream across all components."""
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    household: HouseholdConfig = Field(default_factory=HouseholdConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    ltm: LTMConfig = Field(default_factory=LTMConfig)
    equilibrium: EquilibriumConfig = Field(default_factory=EquilibriumConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MasterConfig:
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def default(cls) -> MasterConfig:
        """Return default configuration instance."""
        return cls()


# Global config singleton management
_CURRENT_CONFIG: Optional[MasterConfig] = None


def get_config() -> MasterConfig:
    """Get the active global configuration, initializing from default if not set."""
    global _CURRENT_CONFIG
    if _CURRENT_CONFIG is None:
        default_yaml = Path("config.yaml")
        if default_yaml.exists():
            _CURRENT_CONFIG = MasterConfig.from_yaml(default_yaml)
        else:
            _CURRENT_CONFIG = MasterConfig.default()
    return _CURRENT_CONFIG


def set_config(config: MasterConfig) -> None:
    """Set the active global configuration."""
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = config
