#!/bin/bash
# ---------------------------------------------------------------------------
# One comparable set of counterfactual runs for a county using its
# vaccinated_test `fixed_60` calibration as the frozen model.
#
#   FIPS is the first positional arg (default 01031). The frozen params
#   vaccinated_test/data/<FIPS>/fixed_60_params.txt must already exist
#   (produced by vaccinated_test/run_cal_fixed_60.sh <FIPS>).
#
#   * config = the committed 0.0005_3_5_True_True_False_metro_0 setup
#     (WITH_K=true, WITH_VACC=true, USE_7DAY_AVG=false, SEED 42) PLUS
#     IMMUNITY_WANING_MODE=fixed, RECOVERED_TO_SUSCEPTIBLE_TIME=60,
#     VACCINATED_TO_SUSCEPTIBLE_TIME=90
#   * frozen params = vaccinated_test/data/01031/fixed_60_params.txt  (via CF_PARAM_FILE)
#   * all 11 CF types x 30 iterations, common random numbers across CF types
#     (abm_nets sets _cf_iter per iteration; transition.py / action.py key every
#      stochastic draw on (seed, _cf_iter, step, call-site), NOT on cf_type)
#   * every artifact -> vaccinated_test/results/01031/counterfactuals/fixed_60/
#     (via CF_OUTPUT_DIR); result_graphs/ and results/ are never touched
#
# Usage:
#   bash vaccinated_test/run_cf_fixed_60.sh              # 01031, full run (11 x 30)
#   bash vaccinated_test/run_cf_fixed_60.sh 36003        # another county
#   CF_TYPES=1,3 CF_ITERS=2 bash vaccinated_test/run_cf_fixed_60.sh 36003  # smoke test
# ---------------------------------------------------------------------------
set -euo pipefail

FIPS="${1:-01031}"
DATE="202010-202104"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
CONFIG="covid_abm/yamls/config.yaml"
PARAMS="$REPO_ROOT/vaccinated_test/data/${FIPS}/fixed_60_params.txt"
OUTDIR="$REPO_ROOT/vaccinated_test/results/${FIPS}/counterfactuals/fixed_60"

[[ -f "$PARAMS" ]] || { echo "[error] frozen params not found: $PARAMS"; exit 1; }

export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$PARAMS"
export CF_OUTPUT_DIR="$OUTDIR"

BAK="$(mktemp -d)"
cp "$CONFIG" "$BAK/config.yaml.orig"
restore() {
    echo "[info] restoring $CONFIG"
    cp "$BAK/config.yaml.orig" "$CONFIG"
    rm -rf "$BAK"
}
trap restore EXIT

# fresh output dir (save_proportions_to_disk2 and cf_raw_*.csv both accumulate;
# a stale dir would merge/append into old columns)
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# patch ONLY the waning keys (targeted line replacement, no YAML re-dump)
$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

echo "=========================================================="
echo " CF set: fixed_60   FIPS ${FIPS}   SEED ${SEED}"
echo "   params : $PARAMS"
echo "   output : $OUTDIR"
echo "   types  : ${CF_TYPES:-1..11}   iters: ${CF_ITERS:-30}"
echo "=========================================================="

$PY main.py "$FIPS"

echo ""
echo "[done] CF artifacts in: $OUTDIR"
ls -1 "$OUTDIR" | sed 's/^/   /'
