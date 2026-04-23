#!/usr/bin/env python3
"""
Convert JSON estimator files to CSV or LaTeX format with significance stars.

Usage:
    python saver_base.py input_dir output_dir [--latex]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from table_utils import dataframe_to_latex, format_coefficient


def extract_estimators_from_json(
    json_data: Dict,
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, float], Dict[str, int]]:
    results = json_data.get("results", {})

    estimator_order = []
    estimator_set = set()
    formatted_data = {}
    bic_data = {}
    num_agents_data = {}

    non_null_estimators = set()

    for crime_type, crime_data in results.items():
        if not isinstance(crime_data, dict) or "estimators" not in crime_data:
            continue

        estimators = crime_data["estimators"]
        std_errors = crime_data["standard_errors"]

        formatted_data[crime_type] = {}

        if "bic" in crime_data:
            bic_data[crime_type] = crime_data["bic"]
        if "num_agents" in crime_data:
            num_agents_data[crime_type] = crime_data["num_agents"]

        for key in ["distance", "race", "income"]:
            if key in estimators and estimators[key] is not None:
                if key not in estimator_set:
                    estimator_order.append(key)
                    estimator_set.add(key)
                non_null_estimators.add(key)
                formatted_data[crime_type][key] = format_coefficient(
                    estimators[key], std_errors.get(key)
                )

        if "features" in estimators and isinstance(estimators["features"], dict):
            for feature_name, feature_value in estimators["features"].items():
                if feature_value is not None:
                    if feature_name not in estimator_set:
                        estimator_order.append(feature_name)
                        estimator_set.add(feature_name)
                    non_null_estimators.add(feature_name)
                    feature_se = std_errors.get("features", {}).get(feature_name)
                    formatted_data[crime_type][feature_name] = format_coefficient(
                        feature_value, feature_se
                    )

    estimator_names = [name for name in estimator_order if name in non_null_estimators]

    return estimator_names, formatted_data, bic_data, num_agents_data


def create_dataframe(
    estimator_names: List[str],
    formatted_data: Dict[str, Dict[str, str]],
    bic_data: Dict[str, float],
    num_agents_data: Dict[str, int],
) -> pd.DataFrame:
    """Create a pandas DataFrame from the formatted data, including BIC and num_agents."""
    crime_types = list(formatted_data.keys())

    data_dict = {}
    for crime_type in crime_types:
        column_values = []
        for estimator in estimator_names:
            value = formatted_data[crime_type].get(estimator, "")
            column_values.append(value)
        data_dict[crime_type] = column_values

    df = pd.DataFrame(data_dict, index=estimator_names)

    separator_row = pd.Series([""] * len(crime_types), index=crime_types, name="")

    bic_row_data = []
    for crime_type in crime_types:
        if crime_type in bic_data:
            bic_row_data.append(f"{bic_data[crime_type]:.2f}")
        else:
            bic_row_data.append("")
    bic_row = pd.Series(bic_row_data, index=crime_types, name="BIC")

    num_agents_row_data = []
    for crime_type in crime_types:
        if crime_type in num_agents_data:
            num_agents_row_data.append(f"{num_agents_data[crime_type]:,}")
        else:
            num_agents_row_data.append("")
    num_agents_row = pd.Series(num_agents_row_data, index=crime_types, name="N")

    df = pd.concat(
        [
            df,
            separator_row.to_frame().T,
            bic_row.to_frame().T,
            num_agents_row.to_frame().T,
        ]
    )

    return df


def process_json_file(
    input_file: Path, output_dir: Path, is_latex: bool, caption: str, label_prefix: str
) -> None:
    """Process a single JSON file and save the output."""
    with open(input_file, "r") as f:
        json_data = json.load(f)

    estimator_names, formatted_data, bic_data, num_agents_data = (
        extract_estimators_from_json(json_data)
    )

    if not estimator_names:
        print(f"Warning: No valid estimators found in {input_file}")
        return

    df = create_dataframe(estimator_names, formatted_data, bic_data, num_agents_data)

    base_name = input_file.stem
    output_extension = ".tex" if is_latex else ".csv"
    output_file = output_dir / (base_name + output_extension)

    if is_latex:
        label = f"{label_prefix}:{base_name}"
        caption_text = caption.replace("{filename}", base_name)
        latex_output = dataframe_to_latex(
            df,
            caption_text,
            label,
            adjustbox=True,
            add_midrule_before_row="",
        )
        with open(output_file, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table saved to {output_file}")
    else:
        df.to_csv(output_file)
        print(f"CSV file saved to {output_file}")

    num_estimators = len(df) - 3
    print(
        f"  - Table dimensions: {num_estimators} estimators + BIC/N × {len(df.columns)} crime types"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert all JSON estimator files in a directory to CSV or LaTeX format"
    )
    parser.add_argument("input_dir", help="Input directory containing JSON files")
    parser.add_argument("output_dir", help="Output directory for CSV/LaTeX files")
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output as LaTeX tables (default: CSV)",
    )
    parser.add_argument(
        "--caption",
        default="Estimation Results for {filename}",
        help="Caption for LaTeX tables (use {filename} as placeholder)",
    )
    parser.add_argument(
        "--label-prefix",
        default="tab",
        help="Label prefix for LaTeX tables (will be formatted as prefix:filename)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(json_files)} JSON file(s) in '{input_dir}'")
    print(f"Output format: {'LaTeX' if args.latex else 'CSV'}")
    print(f"Output directory: '{output_dir}'")
    print()

    for json_file in sorted(json_files):
        print(f"Processing {json_file.name}...")
        try:
            process_json_file(
                json_file, output_dir, args.latex, args.caption, args.label_prefix
            )
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
