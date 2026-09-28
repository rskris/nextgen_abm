## Requirements

### Requirement: Baseline Synthetic Population Generation
The system SHALL generate an initial synthetic population of individuals and households for Santa Barbara County matching Census ACS PUMS microdata distributions and Census Block Group marginal controls.

#### Scenario: Base year synthesis validation
- **WHEN** the population synthesis is executed for the base simulation year
- **THEN** aggregate marginals for age, household size, worker status, income brackets, and vehicle counts match ACS target margins within a 5% margin of error across all Census Block Groups.

### Requirement: Dynamic Multi-Year Life History Evolution
The system SHALL evolve individual and household attributes annually over a multi-year simulation horizon using a Prior (stochastic demographic and economic hazard models) and Update (Bayesian alignment against annual target margins) framework, maintaining persistent agent identifiers.

#### Scenario: Annual agent lifecycle progression
- **WHEN** the simulation clock advances from year $t$ to year $t+1$
- **THEN** each agent increments in age, evaluates hazard models for marriage, childbirth, education completion, job changes, vehicle acquisition, and residential relocation, retaining their persistent ID across the time series.

### Requirement: LEHD LODES Workplace Anchoring
The system SHALL anchor employed synthetic agents to discrete work building locations according to Census LEHD LODES Origin-Destination employment statistics at the block level.

#### Scenario: Commute corridor flow verification
- **WHEN** worker agents residing in North County (Santa Maria / Lompoc) are assigned workplaces
- **THEN** the proportion commuting to South Coast employment centers matches LEHD LODES observed inter-regional commuter distributions.
