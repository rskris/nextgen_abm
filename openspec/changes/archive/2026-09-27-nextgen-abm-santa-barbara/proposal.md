## Why

Traditional four-step and legacy activity-based travel demand models rely on rigid, artificial decision hierarchies (mandatory tours before secondary stops, fixed mode choice before departure time) and static population snapshots. In complex urban regions like Santa Barbara County—characterized by severe North-South jobs-housing spatial mismatch across US-101, major institutional anchors like UCSB with non-standard student travel rhythms, and high EV adoption—these legacy models cannot evaluate critical policies such as peak-spreading, clock-face regional rail service, flexible work/university schedules, or electric grid charging stress.

This project implements a next-generation Activity-Based Model (ABM) grounded in the paradigm of Jinhua Zhao and Michel Bierlaire ("From Trips to Lives"). Built entirely on 100% open data and open-source software, it represents daily activity scheduling as a non-hierarchical combinatorial optimization problem (Mixed Integer Linear Programming) at the physical building footprint level, driven by dynamic multi-year synthetic life histories and coordinated household asset/vehicle dynamics.

## What Changes

- **Building-Level Spatial Representation**: Transition from aggregate Traffic Analysis Zones (TAZs) or Census Block Groups to physical building footprints sourced from Overture Maps GeoParquet via DuckDB.
- **Dynamic Synthetic Life Histories (Shift 4)**: Replace static demographic slices with multi-year synthetic life-event trajectories (aging, household formation, employment transitions, vehicle acquisitions) calibrated with Census PUMS and LODES.
- **Dedicated UCSB & Student Sub-Model**: Model 26,000+ UCSB students and Isla Vista high-occupancy households with lecture timetable anchors, campus bike path networks, micro-mobility, and fare-free transit.
- **Non-Hierarchical Combinatorial Daily Scheduler (Shifts 1 & 2)**: Formulate the full 24-hour day simultaneously as a Mixed Integer Linear Program (MILP) solved using the open-source HiGHS solver (`highspy`), incorporating spatial candidate pruning and Metropolis-Hastings importance sampling.
- **Intra-Household Coordination & Vehicle/EV Constraints (Shift 3)**: Model space-time physical vehicle trajectories, battery State-of-Charge (SoC), charging activities (home TOU vs. DC fast charging), and intra-household synchronization (school runs, joint meals).
- **Regional Policy Analysis Suite**: Implement evaluators for US-101 peak-spreading, Pacific Surfliner clock-face rail transformation, UCSB staggered lecture schedules, and housing relocation dynamics.

## Capabilities

### New Capabilities
- `building-level-spatial-data`: Pipeline for extracting and processing Overture Maps (Buildings, Places/POIs, Transportation) via DuckDB, deriving parcel capacities, and multi-scale routing.
- `synthetic-population-life-history`: Engine synthesizing households and tracking dated life-history transitions (demographic aging, housing moves, vehicle transactions) using Census PUMS and LEHD LODES.
- `ucsb-student-submodel`: Specialized module for UCSB/Isla Vista student housing typologies, class schedule micro-anchors, grade-separated bike paths, and active mobility.
- `combinatorial-schedule-optimizer`: Mathematical formulation and open-source solver (HiGHS via `highspy`) solving continuous 24-hour daily schedules under physical, temporal, and spatial constraints without decision hierarchies.
- `intra-household-asset-dynamics`: Module tracking household vehicle conservation, EV battery SoC and charging schedules, and synchronized joint activities (escort chains, shared meals).
- `policy-scenario-evaluator`: Scenario testing interface for Santa Barbara County policy interventions (US-101 employer flexibility, Surfliner clock-face rail, staggered university classes).

### Modified Capabilities
*(None - this is a greenfield initiative.)*

## Impact

- **Software Dependencies**: 100% open-source stack including Python (`duckdb`, `duckdb-spatial`, `geopandas`, `shapely`, `highspy`, `ortools`, `valhalla`/`osmnx`) and BEAM (LBNL / MATSim) for dynamic multi-modal traffic, EV charging, and grid simulation.
- **Data Footprint**: Cloud-native GeoParquet queries against Overture Maps Foundation S3 storage, US Census ACS PUMS APIs, LEHD LODES, and local GTFS feeds (Santa Barbara MTD, Clean Air Express, Amtrak Pacific Surfliner).
- **Regional Application**: Santa Barbara County Association of Governments (SBCAG) planning area, with initial focus on the South Coast / Goleta / UCSB corridor and the US-101 commute pipeline from North County (Santa Maria / Lompoc).
