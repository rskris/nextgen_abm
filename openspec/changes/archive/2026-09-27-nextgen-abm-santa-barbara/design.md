## Context

Traditional travel demand modeling relies either on four-step models (Trip Generation, Trip Distribution, Mode Choice, Traffic Assignment) or legacy Activity-Based Models (ABMs). Both paradigms suffer from fundamental structural limitations:
1. **Prescribed Decision Hierarchy**: Imposing artificial sequences (e.g., Mandatory Tour -> Destination -> Mode -> Departure Time) that fail under modern telework, hybrid schedules, and flexible leisure-dominated days.
2. **Static Snapshots**: Treating populations as disconnected cross-sectional samples without accounting for life-course trajectories (moving homes, changing jobs, vehicle purchases).
3. **Isolated Travelers**: Ignoring intra-household bargaining, physical vehicle asset conservation (a car driven to work cannot be used by a spouse at home), and escorting chains (school drop-offs).
4. **Aggregate Zones**: Traffic Analysis Zones (TAZs) blur micro-mobility, bike networks, and pedestrian access.

In Santa Barbara County (SBCAG), these issues are acute. Severe jobs-housing imbalance forces thousands to commute 60+ miles along the US-101 corridor between North County (Santa Maria/Lompoc) and South Coast job centers (Goleta/Santa Barbara). Concurrently, UCSB concentrates 26,000 students and 11,000 staff into Isla Vista with extreme active transport mode shares and lecture-driven micro-peaks.

Based on the paradigm articulated by Jinhua Zhao and Michel Bierlaire ("From Trips to Lives"), this design outlines a next-generation Activity-Based Model using 100% open data (Overture Maps, Census PUMS, LEHD LODES, GTFS) and open-source solvers (HiGHS, Google OR-Tools).

## Goals / Non-Goals

**Goals:**
- Provide a non-hierarchical, continuous 24-hour daily activity scheduler formulated as a Mixed Integer Linear Program (MILP) solved via HiGHS (`highspy`).
- Represent space at the building footprint level using Overture Maps GeoParquet queried via DuckDB with parcel-based capacity constraints.
- Implement a multi-year dynamic synthetic population engine that evolves dated life histories (aging, education, employment, car ownership, residential moves).
- Provide a dedicated UCSB student sub-model capturing course timetable micro-peaks, student housing typologies, and bicycle path networks.
- Enforce space-time vehicle conservation, EV battery State-of-Charge (SoC) dynamics, and intra-household synchronization (joint meals, school escorting).
- Enable evaluation of regional planning policies: US-101 peak-spreading, clock-face Pacific Surfliner rail expansion, and staggered university class times.

**Non-Goals:**
- Commercial software or proprietary solver dependencies (no Gurobi, CPLEX, or Esri licenses required).
- Full micro-traffic dynamic car-following simulation inside the MILP (network congestion, transit operations, and EV fleet charging are handled via iterative coupling with open-source BEAM / MATSim).
- Long-distance inter-state commercial aviation modeling.
- Individual psychological reasoning for life-history events (the model reproduces empirical transition rates, not cognitive causes).

## Decisions

### 1. Spatial Foundation: Overture Maps GeoParquet via DuckDB vs. Raw OSM XML
- **Decision**: Ingest building footprints, places (POIs), and transportation networks directly from Overture Maps Foundation S3 GeoParquet buckets using DuckDB with the `spatial` extension.
- **Rationale**: Santa Barbara County contains ~140,000 buildings. Overture provides globally consistent schemas, unique GERS identifiers, 3D heights, and taxonomy-standardized POIs without requiring multi-gigabyte raw XML parsing.
- **Alternative Considered**: Raw OpenStreetMap PBF extracts via `osmnx`. Rejected due to heavy memory overhead and inconsistent attribute coverage across parcel boundaries.

### 2. Multi-Scale Graph Routing
- **Decision**: Implement a two-tier routing architecture:
  - *Local active transport (<3 miles)*: Point-to-point contraction hierarchies on Overture/OSM pedestrian and cycleway networks via Valhalla.
  - *Regional vehicular/transit (>3 miles)*: Building centroid to nearest network link, routed across corridor highway/transit skims.
- **Rationale**: Avoids the $140,000 \times 140,000 \approx 19.6 \text{ billion cell}$ dense matrix explosion while preserving door-to-door fidelity in high-density active travel areas (Isla Vista, UCSB, Downtown SB).

### 3. Solver Selection: HiGHS (`highspy`) & Google OR-Tools CP-SAT
- **Decision**: Formulate the single-day household scheduling MILP in Python using `highspy` (HiGHS C++ solver) as primary, with Google OR-Tools CP-SAT as a constraint-programming alternative.
- **Rationale**: HiGHS is the fastest open-source simplex/MIP solver available (MIT licensed) and solves typical agent schedule formulations in 10–50 ms.
- **Alternative Considered**: Commercial solvers (Gurobi/CPLEX). Rejected to fulfill the strict 100% open-source requirement.

### 4. Dynamic Life Histories: Prior + Update Bayesian Framework
- **Decision**: Model life-event trajectories through stochastic transition hazard models (trained on PSID and retrospective ACS variables) as the "Prior", aligned annually against county-level California Department of Finance (DOF) and ACS 1-year margins as the "Update".
- **Rationale**: Avoids generating synthetic strangers each year; preserves persistent agent identities, enabling path-dependent vehicle lease and residential relocation tracking.

### 5. Multi-Commodity Space-Time Household Vehicle Flow
- **Decision**: Treat every household vehicle as an explicit physical entity with location conservation constraints:
  $$\sum_{p \in H} \sum_{j} x_{kjm}^{p, v} \le \sum_{p \in H} \sum_{i} x_{ikm}^{p, v} \quad (\forall k \neq \text{Home})$$
  Coupled with continuous battery State-of-Charge (SoC) tracking for electric vehicles.
- **Rationale**: Completely eliminates unrealistic mode switches where vehicles appear or disappear at arbitrary locations.

### 6. Native In-Memory Python Link Transmission Model (LTM) & BEAM Dual Execution
- **Decision**: Implement a native Python mesoscopic traffic simulator based on the **Link Transmission Model (LTM)** and Kinematic Wave Theory directly within the repository, while preserving the MATSim/BEAM plans export capability for external scaling.
- **Rationale**: 
  - Zero external runtime or JVM dependencies: Users can run full iterative supply-demand simulations on macOS/Linux using pure Python and NumPy.
  - Newell's simplified kinematic wave theory provides physical queue storage, bottleneck delays, backward shockwave propagation, and link-level EV battery draw without particle-tracking overhead.
  - Paired with an automated Day-to-Day evolutionary replanning loop (10-20% replanning fraction, plan memory and scoring).

### 7. Unified Joint Household MILP
- **Decision**: Formulate and solve a coupled multi-agent MILP per household in HiGHS that simultaneously optimizes all members' daily schedules.
- **Rationale**: Replaces heuristic post-hoc vehicle allocation with exact physical vehicle space-time conservation and synchronized escort chains in under 50ms per household.

### 8. Master Configuration File Percolated Downstream
- **Decision**: Provide a single declarative `config.yaml` / `SimulationConfig` at the root level that strongly defines geographic, solver, network, behavioral, and scenario parameters and percolates immutably downstream to all components.

### 9. Automated Multi-Tier SPSA Calibration
- **Decision**: Integrate Simultaneous Perturbation Stochastic Approximation (SPSA) to actively calibrate utility and capacity parameters against Caltrans PeMS loop detectors (targeting GEH < 5) and CHTS time-use distributions.

## Risks / Trade-offs

- **[Computational Runtime for Countywide Scale]** → *Mitigation*: Spatial opportunity pruning (restricting candidate destinations to space-time ellipses defined by home/work anchors) and multi-core parallelization over independent households.
- **[Overture Maps POI Opening Hours Completeness]** → *Mitigation*: Fallback to standard archetype time-window priors (e.g. retail 09:00-21:00) when explicit opening hour tags are absent.
- **[Student Population Sampling Bias in PUMS]** → *Mitigation*: Synthetic student cohort generator constrained against UCSB Registrar published enrollments and Isla Vista building unit counts.
- **[Equilibrium Congestion & EV Charging Feedback]** → *Mitigation*: Native Python LTM meso-simulator coupled with Day-to-Day evolutionary replanning, returning updated time-dependent link delays and energy skims to the HiGHS scheduler across iterations.

