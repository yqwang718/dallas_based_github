#!/usr/bin/env python3
"""
Step 3: Build agent JSONL files from geocoded CSVs and census block groups.

Performs point-in-polygon assignment, date filtering, distance filtering,
race/crime-type remapping, and writes the final JSONL outputs consumed by
the base Dallas DCM pipeline.

Usage:
    python prepare_agents.py \
        --offenders-csv    data/raw/offenders_geocoded.csv \
        --incidents-csv    data/raw/incidents_geocoded.csv \
        --census-gpkg      data/census/census_merged.gpkg \
        --mappers-dir      scripts/base/preprocessing/mappers \
        --output-dir       data/features \
        --crs              EPSG:2276

Outputs:
    offenders.jsonl, victims.jsonl,
    offenders_post_covid.jsonl, victims_post_covid.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dcm.protocols import AgentFeatures

logger = logging.getLogger(__name__)

PRE_COVID_START = datetime(2014, 6, 1)
PRE_COVID_END = datetime(2020, 3, 23)
POST_COVID_START = PRE_COVID_END

OFFENDER_CRIME_TYPES = [
    "burglary_breaking_entering",
    "motor_vehicle_theft",
    "larceny_theft_offenses",
    "assault_offenses",
    "robbery",
    "drug_narcotic_violations",
]

VICTIM_CRIME_TYPES = [
    "motor_vehicle_theft",
    "larceny_theft_offenses",
    "assault_offenses",
    "robbery",
]

MIN_DISTANCE_METERS = 10.0


def load_race_remap(mappers_dir):
    """Load the race category mapping from JSON."""
    path = Path(mappers_dir) / "race_remap.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_crime_remap(mappers_dir):
    """Load crime-type remapping from CSV. Returns dict: raw_type -> NIBRS category."""
    path = Path(mappers_dir) / "recategorize_crime.csv"
    df = pd.read_csv(path)
    return dict(zip(df["Type of Incident"], df["rNIBRS Crime Category"]))


NIBRS_TO_SLUG = {
    "BURGLARY/ BREAKING & ENTERING": "burglary_breaking_entering",
    "MOTOR VEHICLE THEFT": "motor_vehicle_theft",
    "LARCENY/ THEFT OFFENSES": "larceny_theft_offenses",
    "ASSAULT OFFENSES": "assault_offenses",
    "ROBBERY": "robbery",
    "DRUG/ NARCOTIC VIOLATIONS": "drug_narcotic_violations",
}


def map_crime_type(raw_type, crime_remap):
    """Map a raw incident type string to a standardized slug."""
    nibrs = crime_remap.get(raw_type)
    if nibrs is None:
        return "others"
    return NIBRS_TO_SLUG.get(nibrs, "others")


def append_block_ids(df, census_gdf, lon_col, lat_col, id_col):
    """Assign block group IDs via point-in-polygon."""
    points = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df[lon_col], df[lat_col])],
        crs=census_gdf.crs,
    )
    joined = gpd.sjoin(points, census_gdf[["geometry"]], how="left", predicate="within")
    df[id_col] = joined["index_right"].fillna(-1).astype(int).values
    return df


def compute_distance(row, x1_col, y1_col, x2_col, y2_col):
    """Euclidean distance in projected coordinates (meters)."""
    dx = row[x1_col] - row[x2_col]
    dy = row[y1_col] - row[y2_col]
    return np.sqrt(dx**2 + dy**2)


def load_and_merge(offenders_csv, incidents_csv, crs):
    """Load geocoded CSVs and merge on incident number."""
    off_cols = [
        "IncidentNum", "ArArrestDate", "ArLAddress", "ArLCity",
        "Ar_LAT", "Ar_LON", "H_LAT", "H_LON", "Race",
    ]
    inc_cols = [
        "IncidentNum", "Type of Incident", "Date of Incident",
        "I_LAT", "I_LON", "V_LAT", "V_LON", "Victim Race",
    ]

    logger.info("Loading offenders: %s", offenders_csv)
    df_off = pd.read_csv(offenders_csv, usecols=[c for c in off_cols if c != "IncidentNum"] + ["IncidentNum"])

    logger.info("Loading incidents: %s", incidents_csv)
    all_cols = pd.read_csv(incidents_csv, nrows=0).columns.tolist()
    use_cols = [c for c in inc_cols if c in all_cols]
    df_inc = pd.read_csv(incidents_csv, usecols=use_cols)

    if "IncidentNum" not in df_inc.columns and "Incident Number w/year" in all_cols:
        df_inc = pd.read_csv(incidents_csv, usecols=use_cols + ["Incident Number w/year"])
        df_inc = df_inc.rename(columns={"Incident Number w/year": "IncidentNum"})

    df = pd.merge(df_off, df_inc, on="IncidentNum", how="outer")
    logger.info("Merged rows: %d", len(df))
    return df


def project_coords(df, src_crs, dst_crs, lon_col, lat_col, x_col, y_col):
    """Project WGS84 lat/lon to a local CRS and store as new columns."""
    valid = df[lon_col].notna() & df[lat_col].notna()
    gdf = gpd.GeoDataFrame(
        df.loc[valid],
        geometry=gpd.points_from_xy(df.loc[valid, lon_col], df.loc[valid, lat_col]),
        crs=src_crs,
    ).to_crs(dst_crs)
    df.loc[valid, x_col] = gdf.geometry.x.values
    df.loc[valid, y_col] = gdf.geometry.y.values
    return df


def extract_agents(df, census_gdf, role, crime_types, race_remap, crime_remap,
                   period_start, period_end, crs):
    """Extract agent features for one role (offender or victim) and one period."""
    if role == "offender":
        home_lon, home_lat = "H_LON", "H_LAT"
        inc_lon, inc_lat = "Ar_LON", "Ar_LAT"
        race_col = "Race"
        date_col = "ArArrestDate"
    else:
        home_lon, home_lat = "V_LON", "V_LAT"
        inc_lon, inc_lat = "I_LON", "I_LAT"
        race_col = "Victim Race"
        date_col = "Date of Incident"

    df_role = df.dropna(subset=[home_lon, home_lat, inc_lon, inc_lat, date_col]).copy()

    df_role["_date"] = pd.to_datetime(df_role[date_col], errors="coerce")
    df_role = df_role[
        (df_role["_date"] >= period_start) & (df_role["_date"] < period_end)
    ].copy()
    logger.info("  %s in period: %d rows", role, len(df_role))

    df_role = project_coords(df_role, "EPSG:4326", crs, home_lon, home_lat, "H_X", "H_Y")
    df_role = project_coords(df_role, "EPSG:4326", crs, inc_lon, inc_lat, "I_X", "I_Y")

    df_role = df_role.dropna(subset=["H_X", "H_Y", "I_X", "I_Y"])

    df_role = append_block_ids(df_role, census_gdf, "I_X", "I_Y", "I_CensusID")
    df_role = df_role[df_role["I_CensusID"] != -1]
    logger.info("  After incident-in-Dallas filter: %d", len(df_role))

    df_role["_dist"] = df_role.apply(
        lambda r: compute_distance(r, "H_X", "H_Y", "I_X", "I_Y"), axis=1
    )
    df_role = df_role[df_role["_dist"] > MIN_DISTANCE_METERS]
    logger.info("  After distance filter (>%dm): %d", MIN_DISTANCE_METERS, len(df_role))

    df_role = append_block_ids(df_role, census_gdf, "H_X", "H_Y", "H_CensusID")
    df_role = df_role[df_role["H_CensusID"] != -1]
    logger.info("  After home-in-study-area filter: %d", len(df_role))

    type_col = "Type of Incident"
    if type_col not in df_role.columns:
        type_col = [c for c in df_role.columns if "type" in c.lower() and "incident" in c.lower()]
        type_col = type_col[0] if type_col else None

    agents = []
    for _, row in df_role.iterrows():
        race = race_remap.get(row.get(race_col, ""), "OTHER")

        raw_crime = row.get(type_col, "") if type_col else ""
        crime = map_crime_type(raw_crime, crime_remap)
        if crime not in crime_types:
            continue

        af = AgentFeatures(
            home_block_id=int(row["H_CensusID"]),
            home_coord=(float(row["H_X"]), float(row["H_Y"])),
            race=race,
            crime_type=crime,
            incident_block_id=int(row["I_CensusID"]),
            incident_block_coord=(float(row["I_X"]), float(row["I_Y"])),
        )
        agents.append(af)

    logger.info("  Agent features created: %d", len(agents))
    return agents


def save_agents_jsonl(agents, output_path):
    """Write agent list to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for agent in agents:
            f.write(agent.model_dump_json() + "\n")
    logger.info("Wrote %d agents to %s", len(agents), output_path)


def main():
    parser = argparse.ArgumentParser(description="Build agent JSONL from geocoded data.")
    parser.add_argument("--offenders-csv", required=True)
    parser.add_argument("--incidents-csv", required=True)
    parser.add_argument("--census-gpkg", required=True, help="census_merged.gpkg")
    parser.add_argument("--mappers-dir", default="scripts/base/preprocessing/mappers")
    parser.add_argument("--output-dir", default="data/features")
    parser.add_argument("--crs", default="EPSG:2276", help="Projected CRS (Texas N Central)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    race_remap = load_race_remap(args.mappers_dir)
    crime_remap = load_crime_remap(args.mappers_dir)

    census_gdf = gpd.read_file(args.census_gpkg).to_crs(args.crs)
    census_gdf = census_gdf.reset_index(drop=True)
    logger.info("Census block groups: %d", len(census_gdf))

    df = load_and_merge(args.offenders_csv, args.incidents_csv, args.crs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    periods = [
        ("pre_covid", PRE_COVID_START, PRE_COVID_END, ""),
        ("post_covid", POST_COVID_START, datetime(2024, 1, 1), "_post_covid"),
    ]

    for period_name, start, end, suffix in periods:
        logger.info("=== Period: %s ===", period_name)

        for role, crime_types, file_prefix in [
            ("offender", OFFENDER_CRIME_TYPES, "offenders"),
            ("victim", VICTIM_CRIME_TYPES, "victims"),
        ]:
            logger.info("Processing %s...", role)
            agents = extract_agents(
                df, census_gdf, role, crime_types, race_remap, crime_remap,
                start, end, args.crs,
            )
            save_agents_jsonl(agents, output_dir / f"{file_prefix}{suffix}.jsonl")

    logger.info("All agent JSONL files written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
