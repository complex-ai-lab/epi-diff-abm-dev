#!/bin/bash
# ---------------------------------------------------------------------------
# Run the calibrated ABM: factual run (CF 7) + counterfactuals, default county 01031.
# Settings come from covid_abm/yamls/config.yaml; CF_TYPES / CF_ITERS /
# LAYER_ATTRIBUTION are in `run_settings`. Same as step 3 of run_p0_county.sh.
#
#   bash run.sh                       # needs calibrate.sh output
#   CF_TYPES=7 CF_ITERS=5 bash run.sh # factual only, 5 stochastic iterations
#   OVERWRITE=1 bash run.sh           # replace an existing run
#
# Output: results/<FIPS>/run/
#   beta_daily.csv              day, date, phase, week, beta
#   seirv_daily.csv             day, date, S, E, I, R, V  (mean over iterations)
#   cases_sim_vs_observed.csv   day, date, simulated_cases, simulated_cases_sd, observed_cases
#   cf_raw_<type>.csv, counterfactual_data<type>*.csv, plots  (one set per CF type)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
FIPS="${FIPS:-01031}"
PY="${PY:-python}"
PARAMS="$(ls result_graphs/$FIPS/*/*_metro_0/calibrated_params.txt 2>/dev/null | head -1 || true)"
[[ -n "$PARAMS" ]] || { echo "[run] no calibrated_params.txt under result_graphs/$FIPS -- run calibrate.sh first" >&2; exit 1; }
OUT="${CF_OUTPUT_DIR:-results/$FIPS/run}"

# cf_raw_<type>.csv is appended to, so a run always starts from an empty folder
if [[ -d "$OUT" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[run] $OUT exists; set OVERWRITE=1 to redo"; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[run] FIPS $FIPS params $PARAMS -> $OUT  $(date)"
GENERATING_COUNTERFACTUAL=true R_SPLINE=0 CF_PARAM_FILE="$PWD/$PARAMS" CF_OUTPUT_DIR="$PWD/$OUT" \
    $PY main.py "$FIPS"
echo "[run] done -> $OUT  $(date)"
