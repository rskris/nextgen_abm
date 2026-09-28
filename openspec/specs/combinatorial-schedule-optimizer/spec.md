## Requirements

### Requirement: Mixed Integer Linear Program (MILP) Daily Schedule Formulation
The system SHALL formulate an agent's 24-hour day simultaneously as a Mixed Integer Linear Program that maximizes daily random utility subject to continuous time flow, duration bounds, spatial travel times, and activity window constraints without an assumed decision hierarchy.

#### Scenario: Solving daily schedule under flexible work window
- **WHEN** an agent has a 4-hour flexible work obligation and a grocery shopping need
- **THEN** the MILP solver simultaneously determines optimal departure times, durations, sequence, destination buildings, and travel modes without pre-determining the sequence.

### Requirement: Open-Source Solver Integration via HiGHS
The system SHALL execute schedule optimization using the open-source HiGHS solver (`highspy`) or Google OR-Tools CP-SAT, completing single-household schedule solutions within 50 milliseconds.

#### Scenario: Solver execution performance benchmark
- **WHEN** a 2-person household schedule optimization problem is submitted to HiGHS
- **THEN** the solver converges to integer optimality within 50 milliseconds on standard CPU hardware.

### Requirement: Spatial Opportunity Pruning
The system SHALL prune non-mandatory secondary activity candidate locations to space-time ellipses defined by the agent's fixed home and work anchors and daily travel time budget.

#### Scenario: Grocery store destination candidate filtering
- **WHEN** selecting candidate buildings for a grocery errand
- **THEN** the candidate destination set is filtered to retail buildings reachable within the agent's spatial ellipse, reducing integer decision variables by at least 95%.

### Requirement: Metropolis-Hastings Stochastic Choice Sampling
The system SHALL use Markov Chain Monte Carlo (Metropolis-Hastings) importance sampling to draw competitive alternative daily schedules from the combinatorial space for discrete choice simulation and parameter estimation.

#### Scenario: Generating stochastic alternative schedules
- **WHEN** random utility draws are applied to the base schedule
- **THEN** the sampling engine generates a distribution of near-optimal schedules weighted by importance sampling probabilities.
