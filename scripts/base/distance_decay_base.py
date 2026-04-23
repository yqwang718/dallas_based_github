#!/usr/bin/env python3
"""Plot distance decay curves from estimator JSON files."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_estimator_data(filepath):
    """Load estimator data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def plot_distance_decay_curves(
    base_path, suffix, save_dir="plots/distance_decay_curves"
):
    """
    Plot distance decay curves for offenders and victims.

    Args:
        base_path: Base directory path (e.g., "data/estimators/base")
        suffix: Suffix for files (e.g., "", "_distance_l2", "_race_bernasco")
        save_dir: Directory to save plots (default: "plots/distance_decay_curves")
    """
    crime_types = ["assault_offenses", "robbery", "motor_vehicle_theft", "larceny_theft_offenses"]
    crime_labels = {
        "assault_offenses": "Assault",
        "robbery": "Robbery",
        "motor_vehicle_theft": "Motor Vehicle Theft",
        "larceny_theft_offenses": "Larceny",
    }

    line_styles = {
        "assault_offenses": "-",
        "robbery": "--",
        "motor_vehicle_theft": "-.",
        "larceny_theft_offenses": ":",
    }

    colors = {"offenders": "black", "victims": "gray"}

    plt.figure(figsize=(10, 6))

    distances_km = np.geomspace(0.1, 10, 1000)

    for agent_type in ["offenders", "victims"]:
        filepath = Path(base_path) / f"{agent_type}{suffix}.json"

        if not filepath.exists():
            print(f"Warning: File {filepath} not found. Skipping {agent_type}.")
            continue

        data = load_estimator_data(filepath)

        for crime_type in crime_types:
            if crime_type not in data["results"]:
                print(
                    f"Warning: {crime_type} not found in {agent_type} data. Skipping."
                )
                continue

            beta = data["results"][crime_type]["estimators"]["distance"]

            normalization_factor = 0.1**beta
            decay = distances_km**beta / normalization_factor

            label = f"{agent_type.capitalize()} - {crime_labels[crime_type]}"
            color = colors[agent_type]
            linestyle = line_styles[crime_type]

            plt.plot(
                distances_km,
                decay,
                color=color,
                linestyle=linestyle,
                linewidth=1,
                label=label,
                alpha=0.9,
            )

    plt.axhline(y=1, color="gray", linestyle="-.", alpha=0.3, linewidth=1)

    plt.xscale("log")
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Relative Probability (normalized at 0.1 km)", fontsize=12)
    plt.title("Distance Decay Functions by Crime Type", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    plt.xlim(0.1, 10)

    marker_distances_km = [0.1, 0.2, 0.5, 1, 2, 5]
    for dist_km in marker_distances_km:
        plt.axvline(x=dist_km, color="gray", linestyle=":", alpha=0.2)
        plt.text(
            dist_km,
            plt.ylim()[0] + 0.02 * (plt.ylim()[1] - plt.ylim()[0]),
            f"{dist_km}km",
            ha="center",
            va="bottom",
            fontsize=8,
            alpha=0.7,
        )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    output_filename = os.path.join(save_dir, f"distance_decay_curves{suffix}.pdf")
    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    print(f"Plot saved as {output_filename}")

    output_filename_png = os.path.join(save_dir, f"distance_decay_curves{suffix}.png")
    plt.savefig(output_filename_png, dpi=600, bbox_inches="tight")
    print(f"Plot also saved as {output_filename_png}")

    plt.close()


def print_distance_values(base_path, suffix):
    """Print distance estimator values for reference."""
    print("\nDistance Estimator Values:")
    print("-" * 60)

    crime_types = ["assault_offenses", "robbery", "motor_vehicle_theft", "larceny_theft_offenses"]

    for agent_type in ["offenders", "victims"]:
        filepath = Path(base_path) / f"{agent_type}{suffix}.json"

        if not filepath.exists():
            continue

        data = load_estimator_data(filepath)
        print(f"\n{agent_type.upper()}:")

        for crime_type in crime_types:
            if crime_type in data["results"]:
                beta = data["results"][crime_type]["estimators"]["distance"]
                se = data["results"][crime_type]["standard_errors"]["distance"]
                print(f"  {crime_type}: β = {beta:.4f} (SE = {se:.4f})")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot distance decay curves from estimator JSON files."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="data/estimators/base",
        help="Base path to estimator files",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for estimator files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/plots/distance_decay_curves/base",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    print_distance_values(args.base_path, args.suffix)

    plot_distance_decay_curves(args.base_path, args.suffix, args.save_dir)


if __name__ == "__main__":
    main()
