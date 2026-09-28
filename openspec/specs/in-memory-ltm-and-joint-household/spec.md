## Requirements

### Requirement: Master Configuration System with Downstream Percolation
The system SHALL provide a centralized, validated configuration mechanism (`config.yaml` / `SimulationConfig`) defining geographical boundaries (Santa Barbara County FIPS 06083), solver options, network capacities, behavioral parameters, and scenario flags, percolating these settings immutably to all downstream components.

#### Scenario: Running simulation from master configuration
- **WHEN** the user provides or modifies a single `config.yaml`
- **THEN** all downstream modules (spatial routing, population synthesis, household MILP, LTM meso-simulator, and calibration) execute using the parameters defined in that configuration.

### Requirement: Unified Joint Household MILP Formulation
The system SHALL formulate and solve a single coupled multi-agent Mixed Integer Linear Program (MILP) in HiGHS for multi-person households, simultaneously optimizing all members' schedules while strictly enforcing physical vehicle space-time conservation and escort synchronization without heuristic post-hoc reconciliation.

#### Scenario: Simultaneous vehicle allocation and escorting
- **WHEN** a 2-adult 1-child household solves its daily schedule
- **THEN** HiGHS determines departure times, vehicle assignments, and school drop-off coordination simultaneously in under 50ms, guaranteeing no vehicle is double-booked.

### Requirement: Native Python Link Transmission Model (LTM) Meso-Simulator
The system SHALL provide a native in-memory Python mesoscopic traffic simulator based on Kinematic Wave Theory (triangular fundamental diagrams) that computes cumulative flow curves, bottleneck queues, backward shockwave spillback, and link-level EV energy consumption without requiring external Java/BEAM runtimes.

#### Scenario: Simulating US-101 and SR-217 corridor congestion
- **WHEN** peak morning commuter trips traverse US-101 and SR-217
- **THEN** the LTM meso-simulator calculates physical queue spillback, dynamic link travel times, and vehicle battery consumption across 15-minute time steps.

### Requirement: Hierarchical Multi-Modal Network Representation
The system SHALL represent freeways, arterials, MTD transit lines, and UCSB Class-I bicycle facilities as discrete LTM kinematic wave cells, connecting building footprints to the network via centroid access connectors.

#### Scenario: Simulating UCSB student bike rush
- **WHEN** morning class periods start at UCSB
- **THEN** student bicycle trips flow through dedicated Class-I bike path cells and roundabout nodes, calculating delay and queueing segregated from vehicular traffic.

### Requirement: Day-to-Day Evolutionary Replanning Loop
The system SHALL execute iterative supply-demand equilibrium using Day-to-Day evolutionary replanning (10–20% replanning fraction), maintaining a plan score memory pool for synthetic agents and converging network flows and utilities.

#### Scenario: Convergence of daily schedules to network equilibrium
- **WHEN** iterative day-to-day simulation runs
- **THEN** sampled agents re-optimize schedules against experienced dynamic link travel times until relative travel time and utility gaps fall below tolerance.

### Requirement: Automated Multi-Tier SPSA Calibration
The system SHALL provide an automated parameter calibration loop using Simultaneous Perturbation Stochastic Approximation (SPSA) to optimize behavioral utility coefficients and LTM link capacities against Caltrans PeMS detector counts (targeting GEH < 5) and CHTS time-use distributions.

#### Scenario: Calibrating against PeMS sensor counts
- **WHEN** the automated calibration routine is triggered
- **THEN** SPSA iteratively perturbs model parameters and evaluates simulated versus observed flows on US-101 stations until GEH statistics meet validation thresholds.
