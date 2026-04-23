#!/bin/bash

# Run all base DCM model variants for both offenders and victims.
# This script generates JSON files in data/estimators/base/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${REPO_ROOT}"

BASE_CONFIG="scripts/base/config_base.yaml"
TEMP_CONFIG="scripts/base/config_base_temp.yaml"

cleanup() {
    rm -f "${TEMP_CONFIG}"
}
trap cleanup EXIT

run_model() {
    local agent="$1"
    local distance_interaction="$2"
    local race_interaction="$3"
    local income_interaction="$4"
    local output_suffix="$5"

    local output_file="data/estimators/base/${agent}${output_suffix}.json"

    echo "Running: ${agent}${output_suffix}"
    echo "  distance_interaction: ${distance_interaction}"
    echo "  race_interaction: ${race_interaction}"
    echo "  income_interaction: ${income_interaction}"
    echo "  output: ${output_file}"

    "${PYTHON_BIN}" <<EOF
import yaml

with open("${BASE_CONFIG}", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["data"]["agent"] = "${agent}"
config["model"]["distance_interaction"] = "${distance_interaction}"
config["model"]["race_interaction"] = "${race_interaction}"
config["model"]["income_interaction"] = "${income_interaction}"
config["output_file"] = "${output_file}"

with open("${TEMP_CONFIG}", "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
EOF

    "${PYTHON_BIN}" main.py --config "${TEMP_CONFIG}"

    echo "[OK] Completed: ${agent}${output_suffix}"
    echo ""
}

mkdir -p data/estimators/base

echo "=========================================="
echo "OFFENDERS MODELS"
echo "=========================================="
run_model "offenders" "l2_log" "l1" "abs_diff" ""
run_model "offenders" "l2" "l1" "abs_diff" "_distance_l2"
run_model "offenders" "l2_log" "dissimilarity" "abs_diff" "_race_bernasco"
run_model "offenders" "l2_log" "threshold" "abs_diff" "_race_disagg"
run_model "offenders" "l2_log" "l1" "dummy" "_ses_dummy"
run_model "offenders_post_covid" "l2_log" "l1" "abs_diff" ""

echo "=========================================="
echo "VICTIMS MODELS"
echo "=========================================="
run_model "victims" "l2_log" "l1" "abs_diff" ""
run_model "victims" "l2" "l1" "abs_diff" "_distance_l2"
run_model "victims" "l2_log" "dissimilarity" "abs_diff" "_race_bernasco"
run_model "victims" "l2_log" "threshold" "abs_diff" "_race_disagg"
run_model "victims" "l2_log" "l1" "dummy" "_ses_dummy"
run_model "victims_post_covid" "l2_log" "l1" "abs_diff" ""

echo "=========================================="
echo "All base estimator runs completed"
echo "=========================================="
echo "Output files in data/estimators/base/:"
ls -la data/estimators/base/*.json
