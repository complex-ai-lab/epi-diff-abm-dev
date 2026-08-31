#!/bin/bash
# ---------------------------------------------------------------------------
# Validation runs for the VACCINATED state with independent waning
# (see plans/Task_ Add a VACCINATED State with Independent Waning.md).
#
# For each configuration this script:
#   1. patches ONLY the waning keys in covid_abm/yamls/config.yaml
#      (vaccinated_test/set_vacc_config.py)
#   2. runs the existing pipeline:  python3 main.py <FIPS>   (full 251-epoch
#      recalibration -- accepted for this task)
#   3. copies the run's training_proportions.csv (now incl. a `vaccinated`
#      column) to vaccinated_test/data/<label>.csv
#
# config.yaml and the pre-existing baseline result files are backed up before
# anything runs and restored on exit.
#
# Usage:
#   bash vaccinated_test/run_vacc_tests.sh [FIPS] [none|fixed|stochastic|all]
#   FIPS defaults to 01031 ; group defaults to all
#   SEED defaults to 42 (same as run_all_sims.sh)
# ---------------------------------------------------------------------------
set -euo pipefail

FIPS="${1:-01031}"
GROUP="${2:-all}"
export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export GENERATING_COUNTERFACTUAL=false
# abm_nets.py imports pyplot without forcing a backend; pin a headless one so
# the run does not depend on $DISPLAY (interactive Tk aborts under torch.compile).
export MPLBACKEND="${MPLBACKEND:-Agg}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
CONFIG="covid_abm/yamls/config.yaml"

DATE="202010-202104"
RG_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_True_False_metro_0"
PROP_SRC="${RG_DIR}/training_proportions.csv"
PARAMS_SRC="${RG_DIR}/calibrated_params.txt"
# aligned simulated-vs-actual daily cases written by abm_nets.py on the last epoch
GEN_SRC="results/${FIPS}/0.0005_3_5_metro_0/generated_factual.csv"

DATA_DIR="vaccinated_test/data/${FIPS}"
BAK_DIR="vaccinated_test/_backup"
mkdir -p "$DATA_DIR" "$BAK_DIR"

# ---- back up things we must not clobber ----------------------------------
cp "$CONFIG" "$BAK_DIR/config.yaml.orig"
for f in training_proportions.csv calibrated_params.txt training_loss.csv; do
    [[ -f "${RG_DIR}/${f}" ]] && cp "${RG_DIR}/${f}" "${BAK_DIR}/${f}.orig"
done

restore() {
    echo "[info] restoring $CONFIG and the metro_0 baseline result files"
    cp "$BAK_DIR/config.yaml.orig" "$CONFIG"
    for f in training_proportions.csv calibrated_params.txt training_loss.csv; do
        [[ -f "${BAK_DIR}/${f}.orig" ]] && { mkdir -p "$RG_DIR"; cp "${BAK_DIR}/${f}.orig" "${RG_DIR}/${f}"; }
    done
}
trap restore EXIT

run_one() {
    local label="$1"; shift
    echo ""
    echo "=========================================================="
    echo " vaccinated test run: ${label}   (FIPS ${FIPS}, SEED ${SEED})"
    echo "=========================================================="
    $PY vaccinated_test/set_vacc_config.py "$@"
    $PY main.py "$FIPS"
    cp "$PROP_SRC" "${DATA_DIR}/${label}.csv"
    [[ -f "$GEN_SRC" ]] && cp "$GEN_SRC" "${DATA_DIR}/${label}_cases.csv"
    [[ -f "$PARAMS_SRC" ]] && cp "$PARAMS_SRC" "${DATA_DIR}/${label}_params.txt"
    echo "[info] -> ${DATA_DIR}/${label}.csv (+ ${label}_cases.csv, ${label}_params.txt)"
}

if [[ "$GROUP" == "none" || "$GROUP" == "all" ]]; then
    # Test A: none mode -> infected fraction must match the pre-change baseline
    run_one "none" --mode none
fi

if [[ "$GROUP" == "fixed" || "$GROUP" == "all" ]]; then
    # Test B: natural immunity R in {40,60,80} d, vaccine immunity V = 1.5x R.
    # (Shifted down from the waning_tests 60/80/100 so that some VACCINATED -> S
    #  transitions land inside the 182-day window; vaccine rollout starts day 86.)
    run_one "fixed_40" --mode fixed --recovered-to-susceptible-time 40 --vaccinated-to-susceptible-time 60
    run_one "fixed_60" --mode fixed --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90
    run_one "fixed_80" --mode fixed --recovered-to-susceptible-time 80 --vaccinated-to-susceptible-time 120
fi

if [[ "$GROUP" == "stochastic" || "$GROUP" == "all" ]]; then
    # Test C: natural-immunity rates reuse the waning_tests stochastic sweep,
    #         vaccine-immunity rate = rate / 1.5  (=> ~1.5x longer mean immunity)
    run_one "stochastic_0.005" --mode stochastic --recovered-waning-rate 0.005 --vaccinated-waning-rate 0.003333
    run_one "stochastic_0.010" --mode stochastic --recovered-waning-rate 0.010 --vaccinated-waning-rate 0.006667
    run_one "stochastic_0.020" --mode stochastic --recovered-waning-rate 0.020 --vaccinated-waning-rate 0.013333
    run_one "stochastic_0.040" --mode stochastic --recovered-waning-rate 0.040 --vaccinated-waning-rate 0.026667
fi

echo ""
echo "[done] all requested runs finished. Now generate the plots:"
echo "    $PY vaccinated_test/plot_vacc_tests.py"
