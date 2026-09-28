## Requirements

### Requirement: Student Housing and Cohort Segmentation
The system SHALL segment the UCSB student population into distinct cohorts—On-Campus Dorm Residents, Isla Vista Off-Campus House/Apartment Residents, University Apartments, and Regional Commuters—allocating them to verified residential buildings.

#### Scenario: Isla Vista high-occupancy household synthesis
- **WHEN** synthetic student households are placed in Isla Vista residential buildings
- **THEN** the synthesizer forms multi-student group-quarter households with 4 to 8 roommates, zero or one shared vehicle, and high bicycle asset counts.

### Requirement: Course Schedule Timetable Micro-Anchoring
The system SHALL assign synthetic students mandatory lecture, lab, and discussion activities tied to published UCSB Registrar course start times and building room locations.

#### Scenario: Monday-Wednesday-Friday 9:00 AM lecture peak
- **WHEN** an enrolled student has a 9:00 AM class at Campbell Hall
- **THEN** a mandatory activity is scheduled from 09:00 to 09:50 at Campbell Hall building coordinates, prompting a travel departure from their residence between 08:35 and 08:50.

### Requirement: Active Transport and Micro-Mobility Priority
The system SHALL enforce active transportation modes (walking, standard bicycle, e-bike, skateboard) and Santa Barbara MTD bus transit as the primary mode choice set for on-campus and Isla Vista student agents.

#### Scenario: Mode restriction for dorm residents
- **WHEN** an on-campus freshman resident plans travel between dorms and lecture halls
- **THEN** auto-driver mode is barred from their choice set, restricting available modes to walking, cycling, skateboarding, or campus shuttles.
