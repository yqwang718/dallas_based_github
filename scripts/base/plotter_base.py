#!/usr/bin/env python3
"""
Generate bar plots for crime model estimators.
Produces separate figures for distance, race, and income parameters,
comparing offenders and victims across different crime types.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return None


def extract_estimator_data(data, estimator_name):
    """
    Extract estimator values and standard errors for all crime types.

    Args:
        data: JSON data from estimator file
        estimator_name: Name of estimator ('distance', 'race', or 'income')

    Returns:
        crime_types: List of crime type names
        values: List of estimator values
        errors: List of standard errors
    """
    crime_types = []
    values = []
    errors = []

    crime_type_order = [
        "all_crime_types",
        "burglary_breaking_entering",
        "motor_vehicle_theft",
        "larceny_theft_offenses",
        "assault_offenses",
        "robbery",
        "drug_narcotic_violations",
    ]

    crime_type_labels = {
        "all_crime_types": "All",
        "burglary_breaking_entering": "Burglary",
        "motor_vehicle_theft": "Vehicle Theft",
        "larceny_theft_offenses": "Larceny",
        "assault_offenses": "Assault",
        "robbery": "Robbery",
        "drug_narcotic_violations": "Drug",
    }

    results = data.get("results", {})

    for crime_type in crime_type_order:
        if crime_type in results:
            crime_data = results[crime_type]

            if (
                "estimators" in crime_data
                and estimator_name in crime_data["estimators"]
            ):
                crime_types.append(crime_type_labels[crime_type])
                values.append(crime_data["estimators"][estimator_name])

                if (
                    "standard_errors" in crime_data
                    and estimator_name in crime_data["standard_errors"]
                ):
                    errors.append(crime_data["standard_errors"][estimator_name])
                else:
                    errors.append(0)

    return crime_types, values, errors


def create_bar_plot(
    crime_types,
    offender_values,
    offender_errors,
    victim_values,
    victim_errors,
    estimator_name,
    output_path,
    suffix="",
):
    """
    Create a bar plot comparing offenders and victims.

    Args:
        crime_types: List of crime type names
        offender_values: List of offender estimator values
        offender_errors: List of offender standard errors
        victim_values: List of victim estimator values
        victim_errors: List of victim standard errors
        estimator_name: Name of the estimator for the title
        output_path: Path to save the figure
        suffix: Optional suffix to append to the title
    """
    victim_values = list(victim_values)
    victim_errors = list(victim_errors)

    for i, crime_type in enumerate(crime_types):
        if crime_type in ["Burglary", "Drug"]:
            victim_values[i] = 0
            victim_errors[i] = 0

    plt.figure(figsize=(14, 6))

    offender_color = "darkgray"
    victim_color = "dimgray"

    x = np.arange(len(crime_types))
    width = 0.35

    plt.bar(
        x - width / 2,
        offender_values,
        width,
        yerr=offender_errors,
        capsize=4,
        color=offender_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
        hatch="//",
        label="Offenders",
    )

    plt.bar(
        x + width / 2,
        victim_values,
        width,
        yerr=victim_errors,
        capsize=4,
        color=victim_color,
        alpha=0.9,
        edgecolor="black",
        linewidth=1,
        label="Victims",
    )

    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.xlabel("Crime Type", fontsize=14, fontweight="bold")
    plt.ylabel("Coefficient Value", fontsize=14, fontweight="bold")

    title_map = {
        "distance": "Distance Coefficient (log)",
        "race": "Racial Dissimilarity",
        "income": "Income Difference",
    }

    title = title_map.get(estimator_name, estimator_name.title())

    if suffix == "_race_disagg":
        if estimator_name == "distance":
            title = "Log Distance (using disaggregated racial dissimilarity)"
        elif estimator_name == "race":
            title = "Racial Dissimilarity (Disaggregated)"
        elif estimator_name == "income":
            title = "Income Difference (using disaggregated racial dissimilarity)"
    elif suffix == "_race_bernasco":
        if estimator_name == "distance":
            title = "Log Distance (using Bernasco's racial dissimilarity)"
        elif estimator_name == "race":
            title = "Racial Dissimilarity (Bernasco)"
        elif estimator_name == "income":
            title = "Income Difference (using Bernasco's racial dissimilarity)"
    elif suffix == "_distance_l2":
        if estimator_name == "distance":
            title = "Distance Coefficient (L2 norm)"
        elif estimator_name == "race":
            title = "Racial Dissimilarity (using L2 norm distance)"
        elif estimator_name == "income":
            title = "Income Difference (using L2 norm distance)"
    elif suffix == "_ses_dummy" and estimator_name == "income":
        title = "Income level of Incident BG"

    plt.title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    plt.xticks(x, crime_types, fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12, loc="upper right")

    plt.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bar plots for crime model estimators"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing offenders{suffix}.json and victims{suffix}.json files",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help='Suffix for the JSON files (e.g., "_race_l1" for offenders_race_l1.json)',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plots/estimators",
        help="Output directory for saving plots (default: data/plots/estimators)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="Output format for plots (default: pdf)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    offender_file = input_dir / f"offenders{args.suffix}.json"
    victim_file = input_dir / f"victims{args.suffix}.json"

    print(f"Loading offender data from: {offender_file}")
    offender_data = load_json_data(offender_file)
    if offender_data is None:
        sys.exit(1)

    print(f"Loading victim data from: {victim_file}")
    victim_data = load_json_data(victim_file)
    if victim_data is None:
        sys.exit(1)

    estimators = ["distance", "race", "income"]

    for estimator in estimators:
        print(f"\nProcessing {estimator} estimator...")

        crime_types_off, values_off, errors_off = extract_estimator_data(
            offender_data, estimator
        )

        crime_types_vic, values_vic, errors_vic = extract_estimator_data(
            victim_data, estimator
        )

        if crime_types_off != crime_types_vic:
            print(
                f"Info: Crime types differ between offenders and victims for {estimator}. Merging data..."
            )
            victim_dict = {ct: (val, err) for ct, val, err in zip(crime_types_vic, values_vic, errors_vic)}

            values_vic_merged = []
            errors_vic_merged = []
            for crime_type in crime_types_off:
                if crime_type in victim_dict:
                    values_vic_merged.append(victim_dict[crime_type][0])
                    errors_vic_merged.append(victim_dict[crime_type][1])
                else:
                    values_vic_merged.append(0)
                    errors_vic_merged.append(0)

            values_vic = values_vic_merged
            errors_vic = errors_vic_merged

        if not crime_types_off:
            print(f"Warning: No data found for {estimator}")
            continue

        output_filename = f"{estimator}{args.suffix}.{args.format}"
        output_path = output_dir / output_filename

        create_bar_plot(
            crime_types_off,
            values_off,
            errors_off,
            values_vic,
            errors_vic,
            estimator,
            output_path,
            args.suffix,
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
