#!/usr/bin/env python3
"""
Step 1: Geocode raw Dallas PD arrest and incident addresses.

Requires the ArcGIS Python API and an ArcGIS Online account.
Install: pip install arcgis

Usage:
    python geocode_addresses.py \
        --arrests-csv  data/raw/Offender_Arrests_Dallas.csv \
        --incidents-csv data/raw/incidents_original.csv \
        --output-dir    data/raw \
        --arcgis-url    https://ucirvine.maps.arcgis.com/ \
        --client-id     $ARCGIS_CLIENT_ID

If you already have geocoded CSVs of arrest and incident data from city of dallas,
skip this step and proceed directly to prepare_agents.py.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from arcgis.geocoding import batch_geocode, get_geocoders
from arcgis.gis import GIS

logger = logging.getLogger(__name__)

DALLAS_CITY_TYPOS = {
    "DSLLAS": "DALLAS", "Dallas": "DALLAS", "DDALLAS": "DALLAS",
    "DALLSS": "DALLAS", "DALLAS TEXAS": "DALLAS", "DALLAS,TX": "DALLAS",
    "DALALAS": "DALLAS", "DALLAS1": "DALLAS", "DALLASQ": "DALLAS",
    "ALLAS": "DALLAS", "DALLASTT": "DALLAS", "DALLASTX": "DALLAS",
    "DALLAS TX": "DALLAS", "DLLAS": "DALLAS", "DAQLLAS": "DALLAS",
    "DALLAST": "DALLAS", "DALLSA": "DALLAS", "DALLASS": "DALLAS",
    "SALLAS": "DALLAS", "DALLA": "DALLAS", "DA": "DALLAS",
    "DALALS": "DALLAS", "DALLS": "DALLAS", "DALAS": "DALLAS",
    "DALLLAS": "DALLAS", "DLS": "DALLAS", "DAL": "DALLAS",
    "DALLAAS": "DALLAS",
}

VICTIM_CITY_TYPOS = {
    "DALLLAS": "DALLAS", "DDALLAS": "DALLAS", "DSLLAS": "DALLAS",
    "DALLSS": "DALLAS", "Dallas": "DALLAS", "DALLA": "DALLAS",
    "DALLASS": "DALLAS",
}


def store_results(df, results, lat_col, lon_col, start_idx):
    """Store ArcGIS batch_geocode results back into the DataFrame."""
    for i, result in enumerate(results):
        row_idx = start_idx + i
        if row_idx >= len(df):
            break
        if result and "location" in result:
            loc = result["location"]
            df.at[row_idx, lat_col] = loc.get("y")
            df.at[row_idx, lon_col] = loc.get("x")


def geocode_column(df, address_col, lat_col, lon_col, geocoder, batch_size=500,
                   checkpoint_path=None, checkpoint_interval=2000):
    """Batch-geocode an address column using ArcGIS Online."""
    total = len(df)
    start_index = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        geocoded_count = existing[lat_col].notna().sum()
        start_index = int(geocoded_count)
        logger.info("Resuming from row %d (checkpoint: %s)", start_index, checkpoint_path)
        df[lat_col] = existing[lat_col]
        df[lon_col] = existing[lon_col]

    for i in range(start_index, total, batch_size):
        end = min(i + batch_size, total)
        addresses = df[address_col].iloc[i:end].tolist()

        addresses = [a if isinstance(a, str) else "" for a in addresses]

        logger.info("Geocoding rows %d–%d of %d", i, end - 1, total)
        results = batch_geocode(addresses, geocoder=geocoder)
        store_results(df, results, lat_col, lon_col, i)

        if checkpoint_path and (end % checkpoint_interval == 0 or end == total):
            df.to_csv(checkpoint_path, index=False)
            logger.info("Checkpoint saved at row %d", end)

    return df


def prepare_arrests(csv_path):
    """Load and clean the raw arrests CSV."""
    df = pd.read_csv(csv_path)

    df["ArLCity"] = df["ArLCity"].replace(DALLAS_CITY_TYPOS)
    df["ArLZip"] = df["ArLZip"].astype(str)
    df["HZIP"] = df["HZIP"].astype(str)

    df["ArLAddress_new"] = (
        df["ArLAddress"].fillna("") + "," +
        df["ArLCity"].fillna("") + "," +
        df["ArState"].fillna("") + "," +
        df["ArLZip"]
    )
    df["HAddress_new"] = (
        df["HAddress"].fillna("") + "," +
        df["HCity"].fillna("") + "," +
        df["HState"].fillna("") + "," +
        df["HZIP"]
    )

    df["Ar_LAT"] = None
    df["Ar_LON"] = None
    df["H_LAT"] = None
    df["H_LON"] = None

    return df


def prepare_incidents(csv_path):
    """Load and clean the raw incidents CSV."""
    df = pd.read_csv(csv_path)

    df["City"] = df["City"].replace(VICTIM_CITY_TYPOS)
    df["State"] = df["State"].fillna("TX")

    df["Victim Zip Code"] = df["Victim Zip Code"].astype(str)
    df["V_Address_new"] = (
        df["Victim Home Address"].fillna("") + "," +
        df["Victim City"].fillna("") + "," +
        df["Victim State"].fillna("") + "," +
        df["Victim Zip Code"]
    )

    df["V_LAT"] = None
    df["V_LON"] = None

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Geocode raw Dallas PD arrest and incident addresses (ArcGIS Online)."
    )
    parser.add_argument("--arrests-csv", required=True, help="Path to Offender_Arrests_Dallas.csv")
    parser.add_argument("--incidents-csv", required=True, help="Path to incidents_original.csv")
    parser.add_argument("--output-dir", default="data/raw", help="Directory for geocoded CSVs")
    parser.add_argument("--arcgis-url", default="https://ucirvine.maps.arcgis.com/")
    parser.add_argument(
        "--client-id",
        default=os.environ.get("ARCGIS_CLIENT_ID", ""),
        help="ArcGIS OAuth client_id (or set ARCGIS_CLIENT_ID env var)",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.client_id:
        logger.error("ArcGIS client_id required. Use --client-id or set ARCGIS_CLIENT_ID.")
        return 1

    logger.info("Authenticating with ArcGIS Online...")
    gis = GIS(args.arcgis_url, client_id=args.client_id)
    geocoder = get_geocoders(gis)[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading arrests CSV: %s", args.arrests_csv)
    df_arrests = prepare_arrests(args.arrests_csv)

    arrest_output = output_dir / "offenders_geocoded.csv"
    logger.info("Geocoding arrest locations...")
    geocode_column(
        df_arrests, "ArLAddress_new", "Ar_LAT", "Ar_LON", geocoder,
        batch_size=args.batch_size, checkpoint_path=str(arrest_output),
    )
    logger.info("Geocoding home locations...")
    geocode_column(
        df_arrests, "HAddress_new", "H_LAT", "H_LON", geocoder,
        batch_size=args.batch_size, checkpoint_path=str(arrest_output),
    )
    df_arrests.to_csv(arrest_output, index=False)
    logger.info("Saved: %s", arrest_output)

    logger.info("Loading incidents CSV: %s", args.incidents_csv)
    df_incidents = prepare_incidents(args.incidents_csv)

    incident_output = output_dir / "incidents_geocoded.csv"
    logger.info("Geocoding victim home locations...")
    geocode_column(
        df_incidents, "V_Address_new", "V_LAT", "V_LON", geocoder,
        batch_size=args.batch_size, checkpoint_path=str(incident_output),
    )
    df_incidents.to_csv(incident_output, index=False)
    logger.info("Saved: %s", incident_output)

    logger.info("Geocoding complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
