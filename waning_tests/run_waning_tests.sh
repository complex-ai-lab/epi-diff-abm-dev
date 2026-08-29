#!/bin/bash
# ---------------------------------------------------------------------------
# Immunity-waning validation runs (see plans/Task_ Add Configurable Immunity
# Waning to the ABM.md).
#
# For each configuration this script:
#   1. patches ONLY the waning keys in covid_abm/yamls/config.yaml
#   2. runs the existing pipeline:  python3 main.py <FIPS>
#   3. copies the resulting training_proportions.csv into waning_tests/data/
#
# config.yaml and the pre-existing baseline training_proportions.csv are
# backed up before anything runs and restored at the end, so no original
# result files are overwritten.
#
# Usage:
#   bash waning_tests/run_waning_tests.sh [FIPS] [none|fixed|stochastic|all]
#   FIPS defaults to 01031
#   second arg selects which test group to run (default: all)
#
# Reproducibility: SEED defaults to 42 (same as run_all_sims.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

FIPS="${1:-01031}"
GROUP="${2:-all}"
export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export GENERATING_COUNTERFACTUAL=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
CONFIG="covid_abm/yamls/config.yaml"

# result_graphs path produced by abm_nets.py for this county (waning mode is
# NOT part of the path, so every run writes the same file -> we copy it out).
DATE="202010-202104"
RG_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_True_False_metro_0"
PROP_SRC="${RG_DIR}/training_proportions.csv"

DATA_DIR="waning_tests/data"
BAK_DIR="waning_tests/_backup"
mkdir -p "$DATA_DIR" "$BAK_DIR"

# ---- back up things we must not clobber ------------------------------------
cp "$CONFIG" "$BAK_DIR/config.yaml.orig"
if [[ -f "$PROP_SRC" ]]; then
    cp "$PROP_SRC" "$BAK_DIR/training_proportions.baseline.csv"
    # This is the "already-generated result from the same seed" used as the
    # Test 1 reference. Keep a copy under data/ too.
    cp "$PROP_SRC" "$DATA_DIR/none_baseline_reference.csv"
    echo "[info] saved existing baseline -> $DATA_DIR/none_baseline_reference.csv"
else
    echo "[warn] no pre-existing baseline at $PROP_SRC (Test 1 reference will be missing)"
fi

restore() {
    echo "[info] restoring $CONFIG and baseline training_proportions.csv"
    cp "$BAK_DIR/config.yaml.orig" "$CONFIG"
    if [[ -f "$BAK_DIR/training_proportions.baseline.csv" ]]; then
        mkdir -p "$RG_DIR"
        cp "$BAK_DIR/training_proportions.baseline.csv" "$PROP_SRC"
    fi
}
trap restore EXIT

run_one() {
    local label="$1"; shift
    echo ""
    echo "=========================================================="
    echo " waning test run: ${label}   (FIPS ${FIPS}, SEED ${SEED})"
    echo "=========================================================="
    $PY waning_tests/set_waning_config.py "$@"
    $PY main.py "$FIPS"
    cp "$PROP_SRC" "${DATA_DIR}/${label}.csv"
    echo "[info] -> ${DATA_DIR}/${label}.csv"
}

if [[ "$GROUP" == "none" || "$GROUP" == "all" ]]; then
    run_one "none_verify" --mode none
fi

if [[ "$GROUP" == "fixed" || "$GROUP" == "all" ]]; then
    run_one "fixed_60"  --mode fixed --recovered-to-susceptible-time 60
    run_one "fixed_80"  --mode fixed --recovered-to-susceptible-time 80
    run_one "fixed_100" --mode fixed --recovered-to-susceptible-time 100
fi

if [[ "$GROUP" == "stochastic" || "$GROUP" == "all" ]]; then
    run_one "stochastic_0.010" --mode stochastic --waning-rate 0.01
    run_one "stochastic_0.020" --mode stochastic --waning-rate 0.02
    run_one "stochastic_0.040" --mode stochastic --waning-rate 0.04
fi

echo ""
echo "[done] all requested runs finished. Now generate the plots:"
echo "    $PY waning_tests/plot_waning_tests.py"
