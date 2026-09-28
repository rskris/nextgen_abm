## 1. Environment & Open Data Ingestion Pipeline

- [x] 1.1 Configure Python project environment with 100% open-source dependencies (`duckdb`, `duckdb-spatial`, `geopandas`, `highspy`, `ortools`, `shapely`, `pydeck`)
- [x] 1.2 Implement DuckDB S3 pipeline to query Overture Maps Foundation GeoParquet for Santa Barbara County (Buildings, Places/POIs, Transportation)
- [x] 1.3 Ingest US Census ACS 5-Year PUMS microdata for PUMAs 08301 and 08302 and Census Block Group marginal tables
- [x] 1.4 Ingest California LEHD LODES (v8) Origin-Destination (OD), Workplace Area (WAC), and Residence Area (RAC) data at Census block level
- [x] 1.5 Ingest regional GTFS transit schedules (Santa Barbara MTD, Clean Air Express, Amtrak Pacific Surfliner)

## 2. Building-Level Spatial Foundation & Multi-Scale Routing

- [x] 2.1 Process Overture building footprints and join Santa Barbara County Assessor parcel data to determine residential unit and commercial capacities
- [x] 2.2 Geocode and classify Overture Places (POIs) into activity opportunity types (grocery, dining, medical, leisure, school, higher-ed)
- [x] 2.3 Build multi-scale network router: point-to-point active travel routing for local trips (<3 miles) and highway/transit corridor skims for regional trips (>3 miles)
- [x] 2.4 Incorporate USGS 3DEP elevation data to assign road and cycleway slope gradients for active travel impedance and EV energy calculations

## 3. Synthetic Population & Dynamic Life History Engine

- [x] 3.1 Implement base-year synthetic population generator matching ACS PUMS joint distributions and Block Group marginal targets
- [x] 3.2 Anchor employed synthetic agents to workplace buildings using LEHD LODES block-level OD flows
- [x] 3.3 Implement dynamic annual life-event transition models (aging, marriage/partnership, childbirth, job turnover, vehicle transactions)
- [x] 3.4 Implement Bayesian target alignment (reweighting candidate evolved population against annual CA Department of Finance and ACS margins)
- [x] 3.5 Verify persistent agent identification and dated trajectory logging across multi-year simulation runs

## 4. Dedicated UCSB & Student Sub-Model

- [x] 4.1 Build student population synthesizer for 26,000+ UCSB students and Isla Vista group-quarters housing (4-8 roommates, shared assets)
- [x] 4.2 Ingest UCSB Registrar course schedule to anchor mandatory lecture, lab, and discussion start times to specific campus lecture halls
- [x] 4.3 Construct Isla Vista and UCSB campus dedicated Class-I grade-separated bike path network with roundabout impedance
- [x] 4.4 Implement student-specific mode choice rules (walk, bike, e-bike, skateboard, fare-free MTD Lines 24x, 11, 27, 28)

## 5. Combinatorial Schedule Optimization Engine (HiGHS)

- [x] 5.1 Formulate the continuous 24-hour daily schedule as a Mixed Integer Linear Program (MILP) in Python
- [x] 5.2 Implement spatial candidate opportunity pruning using space-time ellipses around fixed anchors
- [x] 5.3 Integrate the open-source HiGHS solver (`highspy`) to solve daily agent MILPs within 50ms performance targets
- [x] 5.4 Implement Metropolis-Hastings MCMC importance sampling to draw competitive alternative daily schedules for discrete choice simulation

## 6. Intra-Household Vehicle & EV Dynamics

- [x] 6.1 Implement multi-commodity vehicle flow constraints enforcing physical space-time vehicle conservation within households
- [x] 6.2 Implement EV battery State-of-Charge (SoC) tracking, minimum reserve constraints, and charging activity windows (Home TOU, Workplace L2, DC Fast)
- [x] 6.3 Implement intra-household temporal synchronization constraints for joint activities (shared meals, joint recreation)
- [x] 6.4 Implement dependent escort chains for child school drop-offs and pick-ups integrated into adult work commutes
- [x] 6.5 Build integration pipeline to export household plans and EV fleet profiles to BEAM (LBNL) for dynamic queue-based charging and traffic simulation

## 7. Policy Scenario Evaluator & Regional Case Studies

- [x] 7.1 Implement US-101 peak-spreading scenario evaluator (testing flexible arrival hours and telework incentives)
- [x] 7.2 Implement Pacific Surfliner clock-face regional rail scenario evaluator (hourly service and vehicle shedding analysis)
- [x] 7.3 Implement UCSB staggered class start-time scenario evaluator (evaluating bike path and bus queue relief)
- [x] 7.4 Implement North-South jobs-housing mismatch and residential relocation scenario evaluator
- [x] 7.5 Implement BEAM-driven EV grid charging and transformer load curve evaluator across South Coast and North County corridors

## 8. Calibration, Validation & Visual Dashboards

- [x] 8.1 Calibrate activity duration distributions against California Household Travel Survey (CHTS) and American Time Use Survey (ATUS)
- [x] 8.2 Validate simulated traffic flows against Caltrans PeMS detector counts along the US-101 corridor
- [x] 8.3 Build interactive visualization dashboard (using DuckDB + Lonboard / PyDeck) displaying building-level activity flows and agent schedules

## 9. Master Configuration & Downstream Pipeline Integration

- [x] 9.1 Implement master configuration system (`src/nextgen_abm/config.py` & `config.yaml`) validating spatial bounds, solver settings, and scenario parameters
- [x] 9.2 Percolate master configuration downstream through spatial, population, household, and traffic modules


## 10. Unified Joint Household MILP Formulation

- [x] 10.1 Implement Unified Joint Household MILP in HiGHS (`src/nextgen_abm/household/household_milp.py`) simultaneously solving all members with exact physical vehicle conservation
- [x] 10.2 Integrate school escort coordination and joint meal synchronization directly into the joint household MILP formulation

## 11. Multi-Modal LTM Network & Native Meso-Simulator

- [x] 11.1 Build hierarchical multi-modal network representation (`src/nextgen_abm/traffic/network.py`) modeling freeways, arterials, MTD transit corridors, and UCSB Class-I bike cells with building connectors
- [x] 11.2 Implement native Python Link Transmission Model (LTM) meso-simulator (`src/nextgen_abm/traffic/ltm.py`) with triangular fundamental diagrams, sending/receiving flows, cumulative curves, and link EV energy consumption

## 12. Day-to-Day Evolutionary Replanning & Automated Calibration

- [x] 12.1 Implement Day-to-Day evolutionary replanning engine (`src/nextgen_abm/traffic/equilibrium.py`) with plan memory, scoring, logit selection, and 15% replanning loop
- [x] 12.2 Implement automated SPSA calibration loop (`src/nextgen_abm/validation/spsa_calibration.py`) tuning utility parameters and LTM link capacities against Caltrans PeMS and CHTS

