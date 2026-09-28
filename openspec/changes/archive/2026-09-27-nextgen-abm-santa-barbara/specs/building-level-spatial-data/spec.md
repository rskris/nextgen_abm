## ADDED Requirements

### Requirement: Spatial Data Ingestion via Overture Maps and DuckDB
The system SHALL ingest building footprints, places (POIs), and transportation networks for Santa Barbara County directly from Overture Maps Foundation GeoParquet data using DuckDB spatial queries without requiring proprietary GIS software.

#### Scenario: Query buildings within Santa Barbara County bounding box
- **WHEN** the spatial data pipeline is executed with county bounding box `[-120.9, 34.3, -119.4, 35.1]`
- **THEN** DuckDB queries the Overture S3 building dataset and returns valid GeoDataFrames containing building footprints, heights, and global entity reference IDs (GERS).

### Requirement: Building Capacity Assignment
The system SHALL assign residential dwelling unit capacity and employment floor area capacity to each physical building footprint by joining spatial parcel attributes and land-use classifications.

#### Scenario: Capacity derivation for multi-family residential building
- **WHEN** an apartment building parcel is processed
- **THEN** the system sets the residential capacity to the verified unit count and flags it as eligible to house synthetic households.

### Requirement: Multi-Scale Network Graph and Routing
The system SHALL construct a multi-scale network graph providing point-to-point active travel routing on pedestrian/cycle links for short trips (<3 miles) and highway/transit corridor skims for long-distance regional trips (>3 miles).

#### Scenario: Route query between local buildings
- **WHEN** an agent requests travel time and distance between an Isla Vista apartment and a campus lecture hall
- **THEN** the routing engine calculates shortest-path distance and duration along the dedicated pedestrian/bike path network.
