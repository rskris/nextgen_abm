## ADDED Requirements

### Requirement: Space-Time Physical Vehicle Conservation
The system SHALL track each household vehicle as an explicit physical asset in space and time, preventing any household member from operating a vehicle unless it is physically present at their current building location and not occupied by another driver.

#### Scenario: Blocking vehicle use when parked away from home
- **WHEN** Person 1 drives Household Car A to their workplace at 08:00
- **THEN** Household Car A is unavailable to Person 2 at home until Person 1 returns the vehicle or Person 2 travels to Person 1's location.

### Requirement: Electric Vehicle Battery State-of-Charge Dynamics
The system SHALL model continuous battery State-of-Charge (SoC) for electric vehicles, accounting for terrain-adjusted energy consumption and explicit charging activity windows (Home Level-2, Workplace Level-2, and DC Fast Charging).

#### Scenario: Mandatory EV charging detour
- **WHEN** an EV's projected State-of-Charge falls below the minimum 15% reserve threshold during a planned round trip
- **THEN** the optimizer forces an explicit charging stop at a commercial fast charger or increases dwell time at an equipped workplace parking building.

### Requirement: Household Escort Chains and Joint Activity Synchronization
The system SHALL synchronize arrival and departure times for joint household activities (shared meals, joint recreation) and model dependent escort trips where an adult driver chauffeurs a child to school before continuing to work.

#### Scenario: School run escort coordination
- **WHEN** a child has a mandatory 08:15 school start time
- **THEN** the household optimizer assigns one adult to chauffeur the child from Home to School and synchronizes departure time to arrive prior to 08:15.

### Requirement: BEAM EV Fleet and Charging Infrastructure Integration
The system SHALL export synthetic household daily plans and vehicle battery profiles to BEAM (LBNL) to simulate dynamic charging sessions, charger plug queuing, and electrical load curves across Santa Barbara County.

#### Scenario: Exporting EV plans to BEAM
- **WHEN** household daily schedules with assigned EV assets are finalized
- **THEN** the system generates BEAM-compatible vehicle and plan definitions specifying battery capacities, initial State-of-Charge, and preferred charging windows.
