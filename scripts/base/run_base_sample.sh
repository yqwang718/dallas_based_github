#!/bin/bash

# Run a quick synthetic end-to-end check for the base Dallas pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${REPO_ROOT}"

BASE_CONFIG="scripts/base/config_base.yaml"
TEMP_CONFIG="scripts/base/config_base_sample.yaml"

cleanup() {
    rm -f "${TEMP_CONFIG}"
}
trap cleanup EXIT

run_sample_model() {
    local agent="$1"
    local crime_type="$2"
    local output_file="$3"

    echo "Running sample model for ${agent} (${crime_type})"

    "${PYTHON_BIN}" <<EOF
import yaml

with open("${BASE_CONFIG}", "r", encoding="utf-8") as file_obj:
    config = yaml.safe_load(file_obj)

config["data"]["agent"] = "${agent}"
config["data"]["agent_filter_dict"] = {"crime_type": "${crime_type}"}
config["optimizer"]["max_iter"] = 120
config["optimizer"]["chunk_size"] = 1024
config["output_file"] = "${output_file}"

with open("${TEMP_CONFIG}", "w", encoding="utf-8") as file_obj:
    yaml.safe_dump(config, file_obj, default_flow_style=False, sort_keys=False)
EOF

    "${PYTHON_BIN}" main.py --config "${TEMP_CONFIG}"
    echo "[OK] ${output_file}"
    echo ""
}

mkdir -p data/estimators/base data/tables/base

echo "Generating synthetic feature files..."
"${PYTHON_BIN}" scripts/base/make_synthetic_base_data.py

rm -f data/estimators/base/*_sample.json

run_sample_model "victims" "assault_offenses" "data/estimators/base/victims_sample.json"
run_sample_model "offenders" "assault_offenses" "data/estimators/base/offenders_sample.json"

echo "Exporting estimator tables..."
"${PYTHON_BIN}" scripts/base/saver_base.py data/estimators/base data/tables/base

echo "Synthetic sample run completed."
echo "Estimator JSON: data/estimators/base/"
echo "Tables: data/tables/base/"
