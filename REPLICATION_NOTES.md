# Replication Notes

This package keeps the base Dallas DCM workflow close to the original project
layout so that the same scripts and relative paths continue to work.

## Included Runtime Surface

The replication folder contains the minimum code needed to execute the base
pipeline:

- `dcm/` — discrete choice model engine (models, interactions, MLE utilities,
  pydantic schemas, tests)
- `main.py` — base DCM pipeline entry point
- `pyproject.toml` / `setup.py` — package metadata and installer (setup.py is a
  thin shim; all metadata is in pyproject.toml)
- `requirements.txt` — pinned top-level deps for users who prefer `pip install
  -r` over `pip install -e .`
- `scripts/base/` — all runner scripts, utility scripts (plotter, saver,
  summarizer, distance decay), configuration, and synthetic data generation
- `scripts/base/preprocessing/` — data preparation pipeline that converts raw
  police records and census data into the JSONL features consumed by the model
- `.github/workflows/tests.yml` — CI that runs `pytest dcm/tests.py` and the
  synthetic smoke test on every push / pull request

## Synthetic Files and Real Counterparts

The synthetic files in `data/features/` mirror the names and schema of the real
analytic inputs expected by the base runner:

- `blocks.jsonl`
- `victims.jsonl`
- `offenders.jsonl`
- `victims_post_covid.jsonl`
- `offenders_post_covid.jsonl`

These files preserve the same top-level fields used by the base pipeline, such
as:

- block group identifiers and centroids
- agent home block IDs and incident block IDs
- projected coordinates
- crime-type labels
- block feature columns referenced by `scripts/base/config_base.yaml`

## Regenerating the Synthetic Data

The repository includes:

```bash
python scripts/base/make_synthetic_base_data.py
```

That script rewrites the synthetic JSONL files in `data/features/`.

## Swapping in Real Inputs

The primary data source (Dallas Police Department incident and arrest records)
is publicly available; the download URL is cited in the manuscript. Those files
are too large to host on GitHub, so this repository ships synthetic stand-ins
instead.

To replicate the published estimates, replace the synthetic files in
`data/features/` with the real JSONL files while preserving the same file
names. The key expectations are:

- `block_id`, `home_block_id`, and `incident_block_id` must be aligned
  (these refer to census block group IDs)
- `home_coord` and `incident_block_coord` must remain projected coordinates
- block feature columns must still include the names listed in
  `scripts/base/config_base.yaml`

## Preprocessing (Raw Data → JSONL Features)

The `scripts/base/preprocessing/` folder contains three scripts that document
and implement the full data cleaning pipeline:

1. **`geocode_addresses.py`** (optional) — batch-geocodes raw address CSVs
   using ArcGIS Online. Skip this if you already have geocoded CSV files.
2. **`prepare_blocks.py`** — builds census block group features from a merged
   census GeoPackage. Writes `blocks.jsonl`.
3. **`prepare_agents.py`** — joins geocoded agent records to block groups,
   applies temporal and distance filters, remaps race/crime categories, and
   writes the agent JSONL files.

A sample data generator (`generate_sample_raw_data.py`) creates synthetic
intermediate inputs so the full pipeline can be tested end-to-end without real
data. See `scripts/base/preprocessing/README.md` for detailed usage.

## Recommended Execution Order

For the synthetic package:

```bash
python scripts/base/make_synthetic_base_data.py
bash scripts/base/run_base_sample.sh
```

For the full synthetic base suite:

```bash
bash scripts/base/run_base.sh
```

To test the preprocessing pipeline with sample data:

```bash
python scripts/base/preprocessing/generate_sample_raw_data.py
python scripts/base/preprocessing/prepare_blocks.py \
    --census-gpkg data/raw_sample/census_merged_sample.gpkg \
    --output-jsonl data/raw_sample/blocks_test.jsonl
python scripts/base/preprocessing/prepare_agents.py \
    --offenders-csv data/raw_sample/offenders_geocoded_sample.csv \
    --incidents-csv data/raw_sample/incidents_geocoded_sample.csv \
    --census-gpkg   data/raw_sample/census_merged_sample.gpkg \
    --mappers-dir   scripts/base/preprocessing/mappers \
    --output-dir    data/raw_sample/features
```

`run_base_sample.sh` is intended as the quickest smoke test. `run_base.sh`
runs the full set of base model variants including robustness specifications.
