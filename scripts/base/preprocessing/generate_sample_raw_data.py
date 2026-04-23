#!/usr/bin/env python3
"""
Generate synthetic raw data for end-to-end pipeline testing.

Produces:
    data/raw_sample/offenders_geocoded_sample.csv
    data/raw_sample/incidents_geocoded_sample.csv
    data/raw_sample/census_merged_sample.gpkg

The synthetic data has ~50 agent rows and 16 block group polygons arranged
in a 4x4 grid matching the existing sample blocks.jsonl. Coordinates use
EPSG:2276 (Texas N Central, feet). The geocoded lat/lon columns are already
filled so the geocode_addresses.py step can be skipped.

Usage:
    python generate_sample_raw_data.py [--output-dir data/raw_sample]
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

logger = logging.getLogger(__name__)

GRID_ORIGIN_X = 759500.0
GRID_ORIGIN_Y = 2119500.0
CELL_SIZE = 950.0
GRID_COLS = 4
GRID_ROWS = 4
NUM_BLOCKS = GRID_COLS * GRID_ROWS
CRS = "EPSG:2276"

RACES = ["Black", "White", "Hispanic or Latino", "Asian", "Middle Eastern"]
RACE_WEIGHTS = [0.40, 0.25, 0.25, 0.05, 0.05]

CRIME_TYPES = [
    "BURGLARY OF HABITATION - FORCED ENTRY",
    "UNAUTHORIZED USE OF MOTOR VEH - AUTOMOBILE",
    "THEFT OF PROP > OR EQUAL $100 BUT <$750- NOT SHOPLIFT",
    "ASSAULT -BODILY INJURY ONLY",
    "ROBBERY OF INDIVIDUAL (AGG)",
    "POSS CONT SUB PEN GRP 1 <1G",
]


def make_block_polygons():
    """Create 16 block group polygons on a 4x4 grid."""
    block_data = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            bid = row * GRID_COLS + col
            x0 = GRID_ORIGIN_X + col * CELL_SIZE
            y0 = GRID_ORIGIN_Y + row * CELL_SIZE
            geom = box(x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE)

            total_pop = np.random.randint(300, 800)
            dist = np.random.dirichlet([2, 2, 2, 1, 1])

            block_data.append({
                "geometry": geom,
                "block_id": bid,
                "WHITE": int(dist[0] * total_pop),
                "BLACK": int(dist[1] * total_pop),
                "HISPANIC": int(dist[2] * total_pop),
                "ASIAN": int(dist[3] * total_pop),
                "OTHER": int(dist[4] * total_pop),
                "POPULATION": total_pop,
                "INCOME": np.random.randint(30000, 90000),
                "EMPLOYEES": np.random.randint(50, 400),
                "arealand": np.random.randint(100000, 500000),
                "avg_household_size": round(np.random.uniform(1.5, 3.0), 1),
                "home_owners_perc": round(np.random.uniform(0.2, 0.6), 2),
                "underage_perc": round(np.random.uniform(0.1, 0.25), 3),
                "pois": np.random.randint(0, 10),
                "transit_stops": np.random.randint(0, 5),
            })

    return gpd.GeoDataFrame(block_data, crs=CRS)


def random_point_in_block(gdf, block_id):
    """Generate a random point inside a block group polygon."""
    poly = gdf.iloc[block_id].geometry
    minx, miny, maxx, maxy = poly.bounds
    return (
        np.random.uniform(minx, maxx),
        np.random.uniform(miny, maxy),
    )


def make_offenders_csv(gdf, n=50, seed=42):
    """Build synthetic geocoded offenders CSV."""
    rng = np.random.default_rng(seed)
    rows = []

    dates_pre = pd.date_range("2016-01-01", "2020-03-01", periods=n)
    for i in range(n):
        home_block = rng.integers(0, NUM_BLOCKS)
        hx, hy = random_point_in_block(gdf, home_block)
        arrest_block = rng.integers(0, NUM_BLOCKS)
        ax, ay = random_point_in_block(gdf, arrest_block)

        race = rng.choice(RACES, p=RACE_WEIGHTS)
        crime = rng.choice(CRIME_TYPES)
        date = dates_pre[i]

        rows.append({
            "IncidentNum": f"{i+1:06d}-{date.year}",
            "ArrestYr": date.year,
            "ArrestNumber": f"{date.year % 100:02d}-{i+1:06d}",
            "ArArrestDate": date.strftime("%m/%d/%Y 12:00:00 AM"),
            "ArArrestTime": f"{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}",
            "ArLAddress": f"{rng.integers(100, 9999)} SAMPLE ST",
            "ArLCity": "DALLAS",
            "ArState": "TX",
            "ArLZip": str(rng.choice([75201, 75204, 75207, 75210, 75215, 75223, 75226, 75227])),
            "HAddress": f"{rng.integers(100, 9999)} HOME AVE",
            "HCity": "DALLAS",
            "HState": "TX",
            "HZIP": str(rng.choice([75201, 75204, 75207, 75210, 75215, 75223, 75226, 75227])),
            "Race": race,
            "Sex": rng.choice(["Male", "Female"]),
            "Age": rng.integers(18, 65),
            "Ar_LAT": ay,
            "Ar_LON": ax,
            "H_LAT": hy,
            "H_LON": hx,
            "Type of Incident": crime,
        })

    return pd.DataFrame(rows)


def make_incidents_csv(offenders_df, gdf, seed=43):
    """Build synthetic geocoded incidents CSV matching offender records."""
    rng = np.random.default_rng(seed)
    rows = []

    for _, off_row in offenders_df.iterrows():
        inc_block = rng.integers(0, NUM_BLOCKS)
        ix, iy = random_point_in_block(gdf, inc_block)

        vic_home_block = rng.integers(0, NUM_BLOCKS)
        vx, vy = random_point_in_block(gdf, vic_home_block)

        vic_race = rng.choice(RACES, p=RACE_WEIGHTS)

        rows.append({
            "IncidentNum": off_row["IncidentNum"],
            "Type of Incident": off_row["Type of Incident"],
            "Date of Incident": off_row["ArArrestDate"],
            "City": "DALLAS",
            "State": "TX",
            "Victim Home Address": f"{rng.integers(100, 9999)} VICTIM LN",
            "Victim City": "DALLAS",
            "Victim State": "TX",
            "Victim Zip Code": str(rng.choice([75201, 75204, 75207, 75210, 75215])),
            "Victim Race": vic_race,
            "I_LAT": iy,
            "I_LON": ix,
            "V_LAT": vy,
            "V_LON": vx,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic raw data for pipeline testing.")
    parser.add_argument("--output-dir", default="data/raw_sample", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Generating block group polygons (%d blocks on %dx%d grid)...",
                NUM_BLOCKS, GRID_COLS, GRID_ROWS)
    gdf = make_block_polygons()
    gpkg_path = output / "census_merged_sample.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info("Saved: %s", gpkg_path)

    logger.info("Generating offenders CSV (%d rows)...", 50)
    offenders = make_offenders_csv(gdf, n=50, seed=args.seed)
    offenders_path = output / "offenders_geocoded_sample.csv"
    offenders.to_csv(offenders_path, index=False)
    logger.info("Saved: %s", offenders_path)

    logger.info("Generating incidents CSV...")
    incidents = make_incidents_csv(offenders, gdf, seed=args.seed + 1)
    incidents_path = output / "incidents_geocoded_sample.csv"
    incidents.to_csv(incidents_path, index=False)
    logger.info("Saved: %s", incidents_path)

    logger.info("Sample raw data generation complete. Files in: %s", output)
    logger.info("To test the pipeline:\n"
                "  python prepare_blocks.py --census-gpkg %s --output-jsonl out/blocks.jsonl\n"
                "  python prepare_agents.py --offenders-csv %s --incidents-csv %s "
                "--census-gpkg %s --output-dir out/features",
                gpkg_path, offenders_path, incidents_path, gpkg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
