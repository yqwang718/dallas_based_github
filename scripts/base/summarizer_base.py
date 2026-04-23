#!/usr/bin/env python3
"""Compute summary statistics for the base Dallas DCM pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dcm.protocols import (
    AgentFeatures,
    BlockFeatures,
    Config,
    load_data,
)
from table_utils import save_table

logger = logging.getLogger(__name__)


def _extract_field_values(
    objects: List[Any], field: str, skip_dicts: bool = False
) -> List[float]:
    values = []
    for obj in objects:
        if field.startswith("extra_features."):
            extra_field_name = field.replace("extra_features.", "")
            if hasattr(obj, "extra_features") and obj.extra_features:
                value = obj.extra_features.get(extra_field_name, None)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
        else:
            value = getattr(obj, field, None)
            if value is not None:
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, tuple) and all(
                    isinstance(x, (int, float)) for x in value
                ):
                    values.append(np.sqrt(sum(x**2 for x in value)))
                elif isinstance(value, dict) and skip_dicts:
                    continue
    return values


def _compute_field_statistics(
    objects: List[Any],
    fields: Optional[List[str]],
    result_key: str,
    skip_dicts: bool = False,
) -> Dict[str, float]:
    stats = {}
    if fields:
        for field in fields:
            values = _extract_field_values(objects, field, skip_dicts)

            if values:
                stats[f"{field}_mean"] = np.mean(values)
                stats[f"{field}_std"] = np.std(values)
                stats[f"{field}_min"] = np.min(values)
                stats[f"{field}_max"] = np.max(values)
            else:
                stats[f"{field}_mean"] = np.nan
                stats[f"{field}_std"] = np.nan
                stats[f"{field}_min"] = np.nan
                stats[f"{field}_max"] = np.nan

    return stats


def get_extra_feature_names(blocks: List[BlockFeatures]) -> List[str]:
    """Extract extra feature names from blocks."""
    if not blocks or not blocks[0].extra_features:
        return []

    extra_feature_names = sorted(blocks[0].extra_features.keys())
    return [f"extra_features.{name}" for name in extra_feature_names]


def compute_summary_statistics(
    agents: List[AgentFeatures],
    blocks: List[BlockFeatures],
    agent_fields: Optional[List[str]] = None,
    block_fields: Optional[List[str]] = None,
    interactions: Optional[
        List[Tuple[str, str, str, Callable[[Any, Any], float]]]
    ] = None,
) -> Dict[str, Dict[str, float]]:
    results = {"agent_stats": {}, "block_stats": {}, "interaction_stats": {}}

    block_map = {
        block.block_id: block for block in blocks if block.block_id is not None
    }

    results["agent_stats"] = _compute_field_statistics(
        agents, agent_fields, "agent_stats", skip_dicts=False
    )

    incident_blocks = []
    for agent in agents:
        if agent.incident_block_id is None:
            raise ValueError("Agent missing incident_block_id")
        incident_block = block_map.get(agent.incident_block_id)
        if not incident_block:
            raise ValueError(
                f"Block not found for incident_block_id: {agent.incident_block_id}"
            )
        incident_blocks.append(incident_block)

    results["block_stats"] = _compute_field_statistics(
        incident_blocks, block_fields, "block_stats", skip_dicts=True
    )

    if not interactions:
        return results

    for interaction_tuple in interactions:
        if len(interaction_tuple) == 5:
            interaction_name, agent_field, block_field, lambda_func, use_home_block = (
                interaction_tuple
            )
        else:
            interaction_name, agent_field, block_field, lambda_func = interaction_tuple
            use_home_block = False

        values = []
        for agent in agents:
            if use_home_block and agent_field == "home_block":
                if agent.home_block_id is None:
                    continue
                home_block = block_map.get(agent.home_block_id)
                if not home_block:
                    continue
                agent_value = home_block
            else:
                agent_value = getattr(agent, agent_field, None)
                if agent_value is None:
                    continue

            if agent.incident_block_id is None:
                continue
            incident_block = block_map.get(agent.incident_block_id)
            if not incident_block:
                continue

            block_value = getattr(incident_block, block_field, None)
            if block_value is None:
                continue

            try:
                if use_home_block and agent_field == "home_block":
                    home_block_value = getattr(agent_value, block_field, None)
                    if home_block_value is None:
                        continue
                    interaction_value = lambda_func(home_block_value, block_value)
                else:
                    interaction_value = lambda_func(agent_value, block_value)

                if not isinstance(interaction_value, (int, float)):
                    continue

                if np.isnan(interaction_value):
                    continue

                values.append(interaction_value)
            except Exception:
                continue

        if values:
            results["interaction_stats"][f"{interaction_name}_mean"] = np.mean(values)
            results["interaction_stats"][f"{interaction_name}_std"] = np.std(values)
            results["interaction_stats"][f"{interaction_name}_min"] = np.min(values)
            results["interaction_stats"][f"{interaction_name}_max"] = np.max(values)

            if "distance" in interaction_name.lower():
                percentiles = [25, 50, 75]
                percentile_values = np.percentile(values, percentiles)

                results["interaction_stats"][f"{interaction_name}_p25"] = (
                    percentile_values[0]
                )
                results["interaction_stats"][f"{interaction_name}_p50"] = (
                    percentile_values[1]
                )
                results["interaction_stats"][f"{interaction_name}_p75"] = (
                    percentile_values[2]
                )

                full_percentiles = [1, 10, 25, 50, 75, 99]
                full_percentile_values = np.percentile(values, full_percentiles)
                logger.info(f"\nPercentiles for interaction_stats.{interaction_name}:")
                logger.info(f"  Min: {np.min(values):.3f}")
                for p, val in zip(full_percentiles, full_percentile_values):
                    logger.info(f"  {p}th percentile: {val:.3f}")
                logger.info(f"  Max: {np.max(values):.3f}")
        else:
            results["interaction_stats"][f"{interaction_name}_mean"] = np.nan
            results["interaction_stats"][f"{interaction_name}_std"] = np.nan
            results["interaction_stats"][f"{interaction_name}_min"] = np.nan
            results["interaction_stats"][f"{interaction_name}_max"] = np.nan

    return results


def compute_distance(coord1: Tuple[float, ...], coord2: Tuple[float, ...]) -> float:
    if len(coord1) != len(coord2):
        raise ValueError("Coordinates must have the same dimension")

    distance_meters = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))
    return distance_meters / 1000


def compute_log_distance(coord1: Tuple[float, ...], coord2: Tuple[float, ...]) -> float:
    distance = compute_distance(coord1, coord2)
    return np.log(1e-3 + distance)


def compute_racial_dissimilarity(
    racial_dist1: Dict[str, float], racial_dist2: Dict[str, float]
) -> float:
    """Compute racial dissimilarity between two racial distributions.

    Uses the L1 norm (Manhattan distance): 0.5 * sum(|p1_i - p2_i|)
    The 0.5 factor normalizes the result to [0, 1] range.
    """
    if not racial_dist1 or not racial_dist2:
        return np.nan

    all_races = set(racial_dist1.keys()) | set(racial_dist2.keys())

    l1_distance = sum(
        abs(racial_dist1.get(race, 0.0) - racial_dist2.get(race, 0.0))
        for race in all_races
    )

    return 0.5 * l1_distance


def compute_income_difference(income1: float, income2: float) -> float:
    """Compute absolute difference between two log incomes."""
    if income1 is None or income2 is None:
        return np.nan

    return abs(income2 - income1)


def format_summary_table(summary_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []

    categories = {
        "agent_stats": "Agent",
        "block_stats": "Block",
        "interaction_stats": "Interaction",
    }

    for category_key, category_name in categories.items():
        for stat_name, value in summary_stats.get(category_key, {}).items():
            field_name = stat_name.replace("_mean", "").replace("_std", "")
            stat_type = "mean" if "_mean" in stat_name else "std"
            rows.append(
                {
                    "Category": category_name,
                    "Field": field_name,
                    "Statistic": stat_type,
                    "Value": value,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df_pivot = df.pivot_table(
            index=["Category", "Field"], columns="Statistic", values="Value"
        ).reset_index()
        return df_pivot

    return df


def format_combined_summary_table(
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    rows_data = []

    all_fields = set()
    for crime_type, stats in all_stats.items():
        for category, cat_stats in stats.items():
            for stat_name in cat_stats.keys():
                if "_mean" in stat_name:
                    field_name = stat_name.replace("_mean", "")
                    all_fields.add((category, field_name))

    def sort_key(x):
        category, field = x
        if category == "interaction_stats":
            if field == "racial_dissimilarity":
                return (3, 0, field)
            elif field == "income_difference":
                return (3, 1, field)
            elif field == "distance":
                return (1, 0, field)
            elif field == "log_distance":
                return (1, 1, field)
            else:
                return (1, 2, field)
        elif field.startswith("extra_features."):
            return (2, 0, field)
        else:
            return (0, 0, field)

    sorted_fields = sorted(all_fields, key=sort_key)

    for category, field in sorted_fields:
        display_field = (
            field.replace("extra_features.", "")
            if field.startswith("extra_features.")
            else field
        )

        if "distance" in field.lower() and category == "interaction_stats":
            stat_types = [
                ("mean", "mean"),
                ("std", "std"),
                ("25th percentile", "p25"),
                ("median", "p50"),
                ("75th percentile", "p75"),
            ]

            if field == "log_distance":
                base_name = "Log Distance"
            elif field == "distance":
                base_name = "Distance"
            else:
                base_name = display_field

            for stat_display, stat_suffix in stat_types:
                row = {
                    "Category": category.replace("_stats", "").title(),
                    "Field": f"{base_name} ({stat_display})",
                }

                for crime_type, stats in all_stats.items():
                    category_key = (
                        f"{category}"
                        if category.endswith("_stats")
                        else f"{category}_stats"
                    )
                    stat_key = f"{field}_{stat_suffix}"
                    stat_val = stats.get(category_key, {}).get(stat_key, np.nan)

                    if not np.isnan(stat_val):
                        row[crime_type] = f"{stat_val:.3f}"
                    else:
                        row[crime_type] = "N/A"

                rows_data.append(row)
                continue

        if field == "racial_dissimilarity":
            display_name = "Racial Dissimilarity"
        elif field == "income_difference":
            display_name = "Income Difference"
        else:
            display_name = display_field

        row = {
            "Category": category.replace("_stats", "").title(),
            "Field": display_name,
        }

        for crime_type, stats in all_stats.items():
            category_key = (
                f"{category}" if category.endswith("_stats") else f"{category}_stats"
            )
            mean_key = f"{field}_mean"
            std_key = f"{field}_std"

            mean_val = stats.get(category_key, {}).get(mean_key, np.nan)
            std_val = stats.get(category_key, {}).get(std_key, np.nan)

            if not np.isnan(mean_val) and not np.isnan(std_val):
                row[crime_type] = f"{mean_val:.3f} ± {std_val:.3f}"
            else:
                row[crime_type] = "N/A"

        rows_data.append(row)

    return pd.DataFrame(rows_data)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file and create Config object."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def main():
    """Main function to run summary statistics analysis."""
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for crime data"
    )
    parser.add_argument(
        "--config",
        default="config_base.yaml",
        help="Path to configuration file (default: config_base.yaml)",
    )
    parser.add_argument(
        "--output",
        default="summary_table",
        help="Output file path without extension (default: summary_table)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "latex"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--caption",
        default="Summary Statistics by Crime Type",
        help="Caption for LaTeX table (ignored for CSV)",
    )
    parser.add_argument(
        "--label",
        default="tab:summary_stats",
        help="Label for LaTeX table (ignored for CSV)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_path = args.config
    if not Path(config_path).exists():
        logger.error(f"Config file {config_path} not found!")
        return 1

    config = load_config(config_path)

    crime_types = [
        "burglary_breaking_entering",
        "motor_vehicle_theft",
        "larceny_theft_offenses",
        "assault_offenses",
        "robbery",
        "drug_narcotic_violations",
    ]

    analyses = [(crime_type, crime_type) for crime_type in crime_types]

    agent_name = config.data.agent
    if "victim" in agent_name.lower():
        all_combined_types = [
            "motor_vehicle_theft",
            "larceny_theft_offenses",
            "assault_offenses",
            "robbery",
        ]
        logger.info(
            f"Using victim-specific crime types for all_combined: {all_combined_types}"
        )
    else:
        all_combined_types = crime_types
        logger.info(f"Using all crime types for all_combined: {all_combined_types}")

    analyses.append(("all_combined", all_combined_types))

    all_summary_stats = {}

    agent_file = f"{config.data.data_root}/{config.data.agent}.jsonl"
    block_file = f"{config.data.data_root}/{config.data.block}.jsonl"

    agent_fields = []
    block_fields = config.model.feature_names.copy()

    include_extra_features = getattr(config.model, "include_extra_features", False)

    interactions = [
        (
            "log_distance",
            "home_coord",
            "home_coord",
            lambda agent_coord, block_coord: compute_log_distance(
                agent_coord, block_coord
            ),
        ),
        (
            "distance",
            "home_coord",
            "home_coord",
            lambda agent_coord, block_coord: compute_distance(agent_coord, block_coord),
        ),
        (
            "racial_dissimilarity",
            "home_block",
            "racial_dist",
            lambda home_racial_dist, incident_racial_dist: compute_racial_dissimilarity(
                home_racial_dist, incident_racial_dist
            ),
            True,
        ),
        (
            "income_difference",
            "home_block",
            "log_median_income",
            lambda home_income, incident_income: compute_income_difference(
                home_income, incident_income
            ),
            True,
        ),
    ]

    for label, crime_filter in analyses:
        logger.info(f"Processing {label}...")

        agent_filter_dict = config.data.agent_filter_dict or {}
        agent_filter_dict["crime_type"] = crime_filter

        try:
            agents = load_data(agent_file, AgentFeatures, agent_filter_dict)

            if len(agents) == 0:
                logger.warning(f"No agents found for {label}, skipping...")
                continue

            blocks = load_data(block_file, BlockFeatures, config.data.block_filter_dict)

            current_block_fields = block_fields.copy()
            if include_extra_features:
                extra_feature_names = get_extra_feature_names(blocks)
                if extra_feature_names:
                    current_block_fields.extend(extra_feature_names)
                    logger.info(
                        f"Added {len(extra_feature_names)} extra features for {label}"
                    )

            stats = compute_summary_statistics(
                agents=agents,
                blocks=blocks,
                agent_fields=agent_fields,
                block_fields=current_block_fields,
                interactions=interactions,
            )

            all_summary_stats[label] = stats

            logger.info(f"Completed {label}: {len(agents)} agents analyzed")

        except Exception as e:
            logger.error(f"Error processing {label}: {str(e)}")
            continue

    if all_summary_stats:
        logger.info("\nCreating combined summary table...")

        combined_df = format_combined_summary_table(all_summary_stats)

        if args.verbose:
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS BY CRIME TYPE")
            print("=" * 80)
            print(combined_df.to_string(index=False))

        output_extension = ".tex" if args.format == "latex" else ".csv"
        output_path = args.output + output_extension

        save_table(
            combined_df,
            output_path,
            format=args.format,
            caption=args.caption,
            label=args.label,
            adjustbox=True,
        )

        logger.info(f"\nSummary table saved to {output_path} (format: {args.format})")
        return 0
    else:
        logger.error("No summary statistics were generated.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
