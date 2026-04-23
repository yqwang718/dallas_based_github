#!/usr/bin/env python3
"""
Generate synthetic base-pipeline feature files for the replication package.

The generated files preserve the schema and cross-file key structure of the
real analytic JSONL inputs, but all values are synthetic.
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "features"

RNG_SEED = 20260407
GRID_SIDE = 4
SAMPLES_PER_CRIME = 18
OTHER_SAMPLES = 6
RACES = ["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]
ANALYZED_CRIME_TYPES = [
    "burglary_breaking_entering",
    "motor_vehicle_theft",
    "larceny_theft_offenses",
    "assault_offenses",
    "robbery",
    "drug_narcotic_violations",
]
ALL_CRIME_TYPES = ANALYZED_CRIME_TYPES + ["others"]


def round_float(value: float) -> float:
    """Round to a stable precision for compact JSON output."""
    return round(float(value), 6)


def jitter(coord: list[float], rng: np.random.Generator, scale: float) -> list[float]:
    """Add small random noise to a projected coordinate."""
    base = np.asarray(coord, dtype=float)
    noise = rng.normal(loc=0.0, scale=scale, size=2)
    return [round_float(base[0] + noise[0]), round_float(base[1] + noise[1])]


def make_racial_dist(block_id: int) -> dict[str, float]:
    """Create a synthetic racial composition that sums to one."""
    dominant = block_id % len(RACES)
    secondary = (block_id + 2) % len(RACES)

    values = np.full(len(RACES), 0.06, dtype=float)
    values[dominant] += 0.56
    values[secondary] += 0.14
    values += 0.01 * (block_id % 3)
    values = np.clip(values, 0.01, None)
    values /= values.sum()

    return {
        race: round_float(value)
        for race, value in zip(RACES, values, strict=True)
    }


def build_blocks() -> list[dict[str, object]]:
    """Build a small synthetic block inventory with realistic-looking features."""
    blocks: list[dict[str, object]] = []
    origin_x = 760000.0
    origin_y = 2120000.0
    spacing = 950.0

    for block_id in range(GRID_SIDE * GRID_SIDE):
        row, col = divmod(block_id, GRID_SIDE)
        home_coord = [
            round_float(origin_x + col * spacing + (row % 2) * 75.0),
            round_float(origin_y + row * spacing + (col % 2) * 55.0),
        ]
        blocks.append(
            {
                "block_id": block_id,
                "home_coord": home_coord,
                "racial_dist": make_racial_dist(block_id),
                "log_median_income": round_float(10.8 + 0.10 * row + 0.06 * col),
                "log_total_population": round_float(6.1 + 0.11 * row + 0.07 * col),
                "log_total_employees": round_float(4.0 + 0.14 * col + 0.05 * row),
                "log_landsize": round_float(12.1 + 0.04 * block_id),
                "avg_household_size": round_float(1.8 + 0.10 * ((row + col) % 5)),
                "home_owners_perc": round_float(0.28 + 0.04 * col + 0.03 * row),
                "underage_perc": round_float(0.12 + 0.015 * ((block_id + 1) % 6)),
                "log_attractions": round_float(np.log1p((block_id * 3) % 11)),
                "log_transit_stops": round_float(np.log1p((row + 2 * col) % 8)),
                "extra_features": None,
            }
        )

    return blocks


def select_race(block: dict[str, object], sample_idx: int, crime_idx: int) -> str:
    """Select a plausible agent race based on the home block composition."""
    race_order = sorted(
        block["racial_dist"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if (sample_idx + crime_idx) % 5 == 0:
        return race_order[1][0]
    if (sample_idx + crime_idx) % 11 == 0:
        return race_order[2][0]
    return race_order[0][0]


def build_agents(
    blocks: list[dict[str, object]],
    role: str,
    phase_shift: int,
) -> list[dict[str, object]]:
    """Build synthetic agent records for one role/dataset."""
    rng = np.random.default_rng(RNG_SEED + phase_shift)
    records: list[dict[str, object]] = []
    role_is_victim = role.startswith("victims")
    n_blocks = len(blocks)
    role_shift = 1 if role_is_victim else 5
    victim_offsets = [0, 1, 4, 5, 1, 4, 0]
    offender_offsets = [0, 2, 3, 6, 7, 9, 4]

    for crime_idx, crime_type in enumerate(ALL_CRIME_TYPES):
        samples = OTHER_SAMPLES if crime_type == "others" else SAMPLES_PER_CRIME
        for sample_idx in range(samples):
            home_idx = (
                crime_idx * 3 + sample_idx * 2 + phase_shift + role_shift
            ) % n_blocks
            offsets = victim_offsets if role_is_victim else offender_offsets
            incident_idx = (home_idx + offsets[(crime_idx + sample_idx) % len(offsets)]) % n_blocks
            if sample_idx % 7 == 0:
                incident_idx = home_idx

            home_block = blocks[home_idx]
            incident_block = blocks[incident_idx]
            race = select_race(home_block, sample_idx, crime_idx)

            records.append(
                {
                    "agent_id": len(records),
                    "home_block_id": home_idx,
                    "home_coord": jitter(home_block["home_coord"], rng, scale=120.0),
                    "race": race,
                    "crime_type": crime_type,
                    "incident_block_id": incident_idx,
                    "incident_block_coord": jitter(
                        incident_block["home_coord"],
                        rng,
                        scale=90.0,
                    ),
                }
            )

    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks()

    outputs = {
        "blocks.jsonl": blocks,
        "victims.jsonl": build_agents(blocks, role="victims", phase_shift=0),
        "offenders.jsonl": build_agents(blocks, role="offenders", phase_shift=11),
        "victims_post_covid.jsonl": build_agents(
            blocks, role="victims_post_covid", phase_shift=23
        ),
        "offenders_post_covid.jsonl": build_agents(
            blocks, role="offenders_post_covid", phase_shift=37
        ),
    }

    for filename, records in outputs.items():
        write_jsonl(OUTPUT_DIR / filename, records)
        print(f"Wrote {filename}: {len(records):,} records")


if __name__ == "__main__":
    main()
