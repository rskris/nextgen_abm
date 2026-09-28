# NextGen ABM: Complete End-to-End Running Guide

Welcome to **NextGen ABM (Santa Barbara County)**. This guide explains how to configure, execute, calibrate, and inspect the travel demand simulation end-to-end on your local machine.

---

## 1. Quickstart: One-Command Execution

A complete, self-contained demonstration script is provided at [`sample_run.py`](sample_run.py). It exercises the full architectural lifecycle (household synthesis, parallel multi-core MILP solving, Link Transmission Model traffic simulation, Caltrans PeMS sensor validation, and 3D dashboard export).

Run it directly from the project root:

```bash
# Make sure your virtual environment is active
source .venv/bin/activate

# Execute the end-to-end sample run
python sample_run.py
```

### What `sample_run.py` does:
1. **Loads Master Config**: Configures Santa Barbara County bounding box, HiGHS solver parameters, and LTM traffic simulation settings.
2. **Builds Multi-Modal Network**: Creates a 42-directional-link regional graph covering US-101, SR-217, Hollister Ave, State St, Carrillo St, and UCSB cycleways.
3. **Synthesizes Multi-Agent Households**: Instantiates realistic archetypes including family escort chains (parent dropping student at school en route to Goleta tech corridor) and Isla Vista student roommates.
4. **Solves Joint Household MILP in Parallel**: Dispatches 20 households across all **12 CPU cores** simultaneously using `highspy`, completing in **~1.4 seconds** (~70 ms/household).
5. **Simulates Dynamic Traffic (LTM)**: Executes Day-to-Day evolutionary replanning using Kinematic Wave Theory to propagate shockwaves, physical queue spillbacks, and vehicle-hours traveled (VHT).
6. **Validates Against Caltrans PeMS**: Computes the GEH statistic against District 5 highway loop detectors on US-101.
7. **Generates Output Artifacts**: Exports Parquet schedules, CSV link flows, JSON validation reports, and a 3D PyDeck dashboard in `outputs/`.

To inspect the 3D visualization in your browser:
```bash
open outputs/dashboard.html
```

---

## 2. Command-Line Interface (CLI)

The package provides a registered CLI command: `nextgen-abm`.

### CLI Help & Overview
```bash
nextgen-abm --help
```

Available subcommands:
- `sync-data`: Download and cache Santa Barbara open geospatial and sensor data.
- `run`: Run the parallel multi-agent simulation with Day-to-Day replanning.
- `calibrate`: Run automated SPSA parameter calibration against Caltrans PeMS freeway detectors.
- `dashboard`: Generate and open the interactive 3D PyDeck visualization.

---

### A. Data Ingestion & Caching (`sync-data`)

Syncs Santa Barbara County building footprints (~140,000 parcels) from Overture Maps GeoParquet, Caltrans District 5 PeMS sensor configurations, Census PUMS, and GTFS schedules:

```bash
nextgen-abm sync-data
```
*Note: If offline or without AWS S3 credentials, the system automatically falls back to bundled Santa Barbara benchmark fixtures.*

---

### B. Running Full Simulation (`run`)

Run the complete pipeline with customizable sample rates, iterations, and CPU concurrency:

```bash
# Light test run (5% sample, 3 equilibrium iterations, 12 CPU workers)
nextgen-abm run --sample 0.05 --iterations 3 --workers 12 --output-dir outputs

# Higher-fidelity run (20% sample, 5 iterations)
nextgen-abm run --sample 0.20 --iterations 5 --workers 12 --output-dir outputs
```

#### CLI Options for `run`:
| Option | Default | Description |
| :--- | :--- | :--- |
| `--config` | `config.yaml` | Path to master YAML configuration file |
| `--sample` | `0.10` | Population sampling rate (0.01 to 1.0) |
| `--iterations` | `3` | Number of Day-to-Day supply-demand iterations |
| `--workers` | `12` | Number of parallel worker processes for HiGHS MILP solving |
| `--output-dir` | `outputs` | Target directory for generated tables and visualizations |

---

### C. Automated Calibration (`calibrate`)

Calibrates behavioral utility parameters (travel time impedance, escort penalties) and highway capacities using Simultaneous Perturbation Stochastic Approximation (SPSA) against Caltrans PeMS detectors:

```bash
nextgen-abm calibrate --iterations 5
```

Outputs the final loss, parameter updates, and corridor GEH metrics. Target: **GEH < 5.0 on >85% of freeway detector stations**.

---

### D. 3D Visualization Dashboard (`dashboard`)

Renders a standalone interactive HTML dashboard visualizing network link congestion, travel speeds, and EV charging trajectories:

```bash
nextgen-abm dashboard --output-dir outputs
```

The CLI will automatically launch `outputs/dashboard.html` in your default browser.

---

## 3. Output Files & Artifacts

All simulation outputs are saved to `outputs/`:

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `daily_schedules.parquet` | Apache Parquet | Comprehensive agent diaries: activity start/end times, chosen modes (`car`, `transit`, `bike`, `walk`), and EV state-of-charge. |
| `network_corridor_metrics.csv` | CSV | Time-varying link metrics (hourly flow, density, average speed, physical queue length) on US-101 and arterial corridors. |
| `pems_geh_validation.json` | JSON | Caltrans PeMS comparison metrics: simulated volume vs. ground truth detector volume, GEH scores, and pass/fail indicators. |
| `dashboard.html` | HTML / PyDeck | Interactive 3D GPU-accelerated map with layer toggles for traffic volume, congestion heatmaps, and UCSB cycle corridors. |

---

## 4. Python API Example

If you want to use NextGen ABM programmatically in Python or Jupyter notebooks:

```python
from nextgen_abm.config import get_config
from nextgen_abm.traffic.network import HierarchicalMultiModalNetwork
from nextgen_abm.household.household_milp import UnifiedHouseholdMILP, MemberAgenda
from nextgen_abm.household.assets import VehicleAsset
from nextgen_abm.scheduler.milp import ActivityDefinition
from nextgen_abm.traffic.equilibrium import DayToDayEquilibriumEngine, HouseholdAgent

# 1. Load config and network
config = get_config()
network = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network(config)

# 2. Define locations (coordinates in WGS84)
home_coords = [("bldg_home", (-119.82, 34.43))]
work_coords = [("bldg_work", (-119.70, 34.42))]

# 3. Create household agenda with mandatory work tour
agenda = MemberAgenda(
    person_id="commuter_01",
    is_adult=True,
    has_license=True,
    activities=[
        ActivityDefinition("h1", "home_morning", True, home_coords, 0.5, 3.0, 6.0, 9.0),
        ActivityDefinition("w",  "work",          True, work_coords, 6.0, 9.0, 8.0, 18.0),
        ActivityDefinition("h2", "home_evening",  True, home_coords, 4.0, 10.0, 17.0, 24.0),
    ],
)
vehicle = VehicleAsset("car_01", "hh_01", "ev", "bldg_home", "bldg_home")
hh = HouseholdAgent("hh_01", members=[agenda], vehicles=[vehicle])

# 4. Run Day-to-Day supply-demand equilibrium
engine = DayToDayEquilibriumEngine(network=network, config=config)
metrics = engine.run_equilibrium([hh], max_iterations=3)

for m in metrics:
    print(f"Iteration {m.iteration}: VHT = {m.total_vht_hours:.2f} hrs, Relative Gap = {m.relative_gap:.4f}")
```

---

## 5. Running the Test Suite

The test suite validates the spatial queries, household MILP math, LTM conservation equations, and CLI subcommands:

```bash
# Run all tests
pytest -v

# Run with test coverage
pytest --cov=src/nextgen_abm
```
*Current test suite: **43 passing tests** running in ~2.5 seconds.*

---

## 6. Hardware Optimization & Performance

This machine is equipped with **12 CPU cores and 64 GB of RAM**. NextGen ABM is designed to scale across all cores:

- **Parallel Household Solver**: `nextgen_abm.household.parallel_milp.solve_households_parallel` automatically partitions the synthetic population across all 12 cores via Python `multiprocessing` / `ProcessPoolExecutor`.
- **HiGHS Solvers**: Each worker runs an independent, lightweight C++ solver instance (`highspy`) with zero Python GIL contention.
- **In-Memory Kinematic Wave LTM**: LTM network simulation operates purely in vectorized NumPy/Python arrays with zero disk I/O bottlenecks.

---

## 7. Troubleshooting & FAQ

### Q: `ModuleNotFoundError: No module named 'pyarrow'`
**Fix:** Run `uv pip install pyarrow>=14.0.0` or `pip install -e ".[dev]"`. PyArrow is required by GeoPandas to write Parquet files.

### Q: Does running without internet access cause Overture or Census queries to crash?
**No.** `OvertureClient` and `DataSyncManager` have built-in fallback fixtures for Santa Barbara County (PUMA 08301/08302 and US-101 corridor links). The model runs seamlessly fully offline.

### Q: How do I change the simulation date or corridor capacity?
Edit [`config.yaml`](config.yaml) in the project root. Parameters such as `simulation_date`, `time_step_min`, `sample_rate`, and `freeway_capacity_per_lane` can be modified declaratively.
