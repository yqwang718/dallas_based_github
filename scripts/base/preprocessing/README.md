# Preprocessing Pipeline

This folder documents and implements the full data preparation pipeline that
converts raw Dallas Police Department records and census data into the JSONL
feature files consumed by the base DCM model.

## Pipeline Overview

```
Step 1 (optional)       Step 2                  Step 3
─────────────────       ──────                  ──────
Raw address CSVs   →  Census .dta / TIGER  →  Geocoded CSVs
       │                      │                      │
  geocode_addresses.py   prepare_blocks.py     prepare_agents.py
       │                      │                      │
       ▼                      ▼                      ▼
offenders_geocoded.csv   census_merged.gpkg    offenders.jsonl
incidents_geocoded.csv   blocks.jsonl          victims.jsonl
                                               *_post_covid.jsonl
```

## Steps

### Step 1: Geocoding (`geocode_addresses.py`)

Batch-geocodes raw arrest and incident addresses using ArcGIS Online.

**This step is optional.** If you already have geocoded CSVs with latitude and
longitude columns, skip directly to Step 3.

- **Requires:** `arcgis` Python package and an ArcGIS Online account
- **Input:** Raw Dallas PD CSVs with address columns
- **Output:** `offenders_geocoded.csv`, `incidents_geocoded.csv`
- **Credentials:** Pass via `--client-id` argument or `ARCGIS_CLIENT_ID` env var

```bash
python geocode_addresses.py \
    --arrests-csv  data/raw/Offender_Arrests_Dallas.csv \
    --incidents-csv data/raw/incidents_original.csv \
    --output-dir    data/raw \
    --client-id     "$ARCGIS_CLIENT_ID"
```

### Step 2: Census Block Group Features (`prepare_blocks.py`)

Merges census demographics, income, land area, transit stops, and POI counts
into block group features. Outputs the GeoPackage used by Step 3 and optionally
writes `blocks.jsonl` directly.

- **Input:** Census GeoPackage (pre-merged from Stata + TIGER sources),
  optional transit and POI layers
- **Output:** `blocks.jsonl` (BlockFeatures, one JSON object per line)

```bash
python prepare_blocks.py \
    --census-gpkg    data/census/census_merged.gpkg \
    --transit-layer  data/census/dart_stops.gpkg \
    --poi-layer      data/census/dallas_pois.gpkg \
    --output-jsonl   data/features/blocks.jsonl
```

Census source: American Community Survey 5-year estimates and Census 2010 SF1
from the U.S. Census Bureau. See the manuscript for exact table IDs.

### Step 3: Agent Feature Extraction (`prepare_agents.py`)

Joins geocoded agent records to block groups via point-in-polygon, applies date
filtering (pre/post-COVID split at 2020-03-23), distance filtering (>10 m),
race and crime-type remapping, then writes agent JSONL files.

- **Input:** Geocoded CSVs from Step 1, census GeoPackage from Step 2, mappers
- **Output:** `offenders.jsonl`, `victims.jsonl`, `offenders_post_covid.jsonl`,
  `victims_post_covid.jsonl`

```bash
python prepare_agents.py \
    --offenders-csv  data/raw/offenders_geocoded.csv \
    --incidents-csv  data/raw/incidents_geocoded.csv \
    --census-gpkg    data/census/census_merged.gpkg \
    --mappers-dir    scripts/base/preprocessing/mappers \
    --output-dir     data/features
```

## Mappers

The `mappers/` directory contains two lookup tables:

| File | Format | Description |
|------|--------|-------------|
| `race_remap.json` | JSON dict | Maps raw race strings (e.g. "Hispanic or Latino") to standardized categories (BLACK, WHITE, HISPANIC, ASIAN, OTHER) |
| `recategorize_crime.csv` | CSV | Maps raw Dallas PD incident types to NIBRS crime categories |

## Testing with Synthetic Data

`generate_sample_raw_data.py` creates small synthetic versions of all
intermediate inputs so the full pipeline can be tested end-to-end without real
data:

```bash
python generate_sample_raw_data.py --output-dir data/raw_sample

python prepare_blocks.py \
    --census-gpkg data/raw_sample/census_merged_sample.gpkg \
    --output-jsonl data/raw_sample/blocks_test.jsonl

python prepare_agents.py \
    --offenders-csv data/raw_sample/offenders_geocoded_sample.csv \
    --incidents-csv data/raw_sample/incidents_geocoded_sample.csv \
    --census-gpkg   data/raw_sample/census_merged_sample.gpkg \
    --mappers-dir   mappers \
    --output-dir    data/raw_sample/features
```

## Column Schemas

### Geocoded Offenders CSV

| Column | Type | Description |
|--------|------|-------------|
| `IncidentNum` | str | Incident identifier for merge |
| `ArArrestDate` | str | Arrest date |
| `Ar_LAT`, `Ar_LON` | float | Arrest location coordinates (WGS84) |
| `H_LAT`, `H_LON` | float | Home address coordinates (WGS84) |
| `Race` | str | Raw race string (remapped via `race_remap.json`) |
| `Type of Incident` | str | Raw incident type (remapped via `recategorize_crime.csv`) |

### Geocoded Incidents CSV

| Column | Type | Description |
|--------|------|-------------|
| `IncidentNum` | str | Incident identifier for merge |
| `Type of Incident` | str | Raw incident type |
| `Date of Incident` | str | Incident date |
| `I_LAT`, `I_LON` | float | Incident location coordinates (WGS84) |
| `V_LAT`, `V_LON` | float | Victim home coordinates (WGS84) |
| `Victim Race` | str | Victim's raw race string |

### Census GeoPackage Attributes

| Column | Type | Description |
|--------|------|-------------|
| `WHITE`, `BLACK`, `ASIAN`, `HISPANIC`, `OTHER` | int | Race counts |
| `POPULATION` | int | Total population |
| `INCOME` | int | Median household income |
| `EMPLOYEES` | int | Total employees |
| `arealand` | int | Land area |
| `avg_household_size` | float | Average household size |
| `home_owners_perc` | float | Homeownership rate |
| `underage_perc` | float | Fraction under 18 |
| `pois` | int | Count of points of interest |
| `transit_stops` | int | Count of transit stops |

## Data Sources

- **Police microdata:** Publicly available from the Dallas Police Department.
  The download URL is cited in the manuscript.
- **Census data:** U.S. Census Bureau ACS 5-year estimates and 2010 SF1.
- **TIGER/Line geometries:** U.S. Census Bureau.
- **Transit stops:** Dallas Area Rapid Transit (DART) open data.
