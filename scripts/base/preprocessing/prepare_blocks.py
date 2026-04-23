#!/usr/bin/env python3
"""
Step 2: Build block group features (blocks.jsonl) from census data.

Reads census Stata extracts and TIGER/Line geometries, merges demographics,
income, transit stops, and POI counts, then writes BlockFeatures JSONL.

Usage:
    python prepare_blocks.py \
        --census-dir     data/census \
        --output-gpkg    data/census/census_merged.gpkg \
        --output-jsonl   data/features/blocks.jsonl

Required inputs in --census-dir:
    census_2010.dta           Block-level census (land area, SES vars)
    bg_census_2010.dta        Block-group tabulations (demographics)
    census_merged.gpkg *OR*   Pre-built geometry layer
    TIGER shapefiles          For block group polygons (if building from scratch)
    dallas_pois/              POI GeoPackage or CSV
    dart_rail / dart_bus      Transit stop layers

Census source: American Community Survey 5-year estimates and Census 2010 SF1,
downloaded from the U.S. Census Bureau. See the manuscript for exact table IDs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dcm.protocols import BlockFeatures

logger = logging.getLogger(__name__)

RACE_COLUMNS = ["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]

FEATURE_LOG_FIELDS = {
    "INCOME": "log_median_income",
    "POPULATION": "log_total_population",
    "EMPLOYEES": "log_total_employees",
    "arealand": "log_landsize",
    "pois": "log_attractions",
    "transit_stops": "log_transit_stops",
}

FEATURE_DIRECT_FIELDS = {
    "avg_household_size": "avg_household_size",
    "home_owners_perc": "home_owners_perc",
    "underage_perc": "underage_perc",
}


def aggregate_by_prefix(df, prefix_map, agg_col, agg_func="sum"):
    """Aggregate block-level vars to block groups using prefix matching."""
    result = df.copy()
    for target_col, source_prefix in prefix_map.items():
        cols = [c for c in df.columns if c.startswith(source_prefix)]
        if cols:
            result[target_col] = df[cols].sum(axis=1) if agg_func == "sum" else df[cols].mean(axis=1)
    return result


def count_features_in_polygons(polygons_gdf, points_gdf, count_col):
    """Count point features falling within each polygon."""
    joined = gpd.sjoin(points_gdf, polygons_gdf, how="inner", predicate="within")
    counts = joined.groupby("index_right").size().rename(count_col)
    return polygons_gdf.join(counts, how="left").fillna({count_col: 0})


def safe_log(series, offset=1.0):
    """Compute log(x + offset), handling zeros."""
    return np.log(series + offset)


def build_block_features(census_gdf):
    """Convert a merged census GeoDataFrame into a list of BlockFeatures."""
    blocks = []
    for idx, row in census_gdf.iterrows():
        centroid = row.geometry.centroid
        coord = (centroid.x, centroid.y)

        total_pop = sum(row.get(r, 0) for r in RACE_COLUMNS)
        if total_pop > 0:
            racial_dist = {r: row.get(r, 0) / total_pop for r in RACE_COLUMNS}
        else:
            racial_dist = {r: 0.0 for r in RACE_COLUMNS}

        bf = BlockFeatures(
            block_id=idx,
            home_coord=coord,
            racial_dist=racial_dist,
            log_median_income=float(safe_log(pd.Series([row.get("INCOME", 0)]))[0]),
            log_total_population=float(safe_log(pd.Series([total_pop]))[0]),
            log_total_employees=float(safe_log(pd.Series([row.get("EMPLOYEES", 0)]))[0]),
            log_landsize=float(safe_log(pd.Series([row.get("arealand", 0)]))[0]),
            avg_household_size=float(row.get("avg_household_size", 0)),
            home_owners_perc=float(row.get("home_owners_perc", 0)),
            underage_perc=float(row.get("underage_perc", 0)),
            log_attractions=float(safe_log(pd.Series([row.get("pois", 0)]))[0]),
            log_transit_stops=float(safe_log(pd.Series([row.get("transit_stops", 0)]))[0]),
        )
        blocks.append(bf)

    return blocks


def save_blocks_jsonl(blocks, output_path):
    """Write BlockFeatures list to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(block.model_dump_json() + "\n")
    logger.info("Wrote %d block groups to %s", len(blocks), output_path)


def main():
    parser = argparse.ArgumentParser(description="Build block group features from census data.")
    parser.add_argument("--census-gpkg", required=True,
                        help="Path to census_merged.gpkg (pre-built merged geometry + attributes)")
    parser.add_argument("--transit-layer", default=None, help="Path to transit stops layer (optional)")
    parser.add_argument("--poi-layer", default=None, help="Path to POI layer (optional)")
    parser.add_argument("--output-jsonl", default="data/features/blocks.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--crs", default="EPSG:2276", help="Projected CRS for centroids")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading census GeoPackage: %s", args.census_gpkg)
    census_gdf = gpd.read_file(args.census_gpkg)

    if census_gdf.crs and str(census_gdf.crs) != args.crs:
        logger.info("Reprojecting from %s to %s", census_gdf.crs, args.crs)
        census_gdf = census_gdf.to_crs(args.crs)

    census_gdf = census_gdf[census_gdf["POPULATION"] > 0].copy()
    census_gdf = census_gdf.reset_index(drop=True)
    logger.info("Block groups with positive population: %d", len(census_gdf))

    if args.transit_layer:
        logger.info("Counting transit stops...")
        transit = gpd.read_file(args.transit_layer).to_crs(args.crs)
        census_gdf = count_features_in_polygons(census_gdf, transit, "transit_stops")

    if args.poi_layer:
        logger.info("Counting POIs...")
        pois = gpd.read_file(args.poi_layer).to_crs(args.crs)
        census_gdf = count_features_in_polygons(census_gdf, pois, "pois")

    for col in ["transit_stops", "pois"]:
        if col not in census_gdf.columns:
            census_gdf[col] = 0

    logger.info("Building BlockFeatures...")
    blocks = build_block_features(census_gdf)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    save_blocks_jsonl(blocks, args.output_jsonl)

    logger.info("Done. %d block groups written.", len(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
