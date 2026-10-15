#!/bin/bash
# ---------------------------------------------------------------------------
# Run the ABM (default: factual run = CF 7, 30 stochastic iterations, county 01031).
# Settings come from config.yaml. Beta / initial infected / k are the calibrated
# values unless BETA / INITIAL_INFECTED / K are set in config.yaml.
#
#   bash run.sh                           # -> results/01031/run/
#   RUN_NAME=beta_1.2 bash run.sh         # -> results/01031/beta_1.2/  (keep runs side by side)
#   CALIBRATION=path/calibrated_params.txt bash run.sh   # use a specific calibration
#   CF_TYPES=1,7 CF_ITERS=5 bash run.sh   # other counterfactual types / iterations
#   OVERWRITE=1 bash run.sh               # replace an existing output folder
#
# Output (results/<FIPS>/<RUN_NAME>/):
#   beta_daily.csv              day, date, phase, week, beta
#   seirv_daily.csv             day, date, S, E, I, R, V  (mean over iterations)
#   cases_sim_vs_observed.csv   day, date, simulated_cases, simulated_cases_sd, observed_cases
#   cf_raw_7.csv, counterfactual_data7*.csv, counterfactual_results7.png
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
FIPS="${FIPS:-01031}"
PY="${PY:-python}"
export ABM_CONFIG="$(realpath "${CONFIG:-config.yaml}")"

# Calibration that matches the E->I / I->R in the config (same folder calibrate.sh writes)
CAL_DIR="result_graphs/$FIPS/$($PY - <<'PY'
import os, yaml
cfg = yaml.safe_load(open(os.environ["ABM_CONFIG"]))
md, epi = cfg["simulation_metadata"], cfg["epidemic"]
print(f"{md['DATE']}/{md['INITIAL_INFECTION_RATE']}_{epi['EXPOSED_TO_INFECTED_TIME']}_{epi['INFECTED_TO_RECOVERED_TIME']}"
      f"_{md['WITH_K']}_{md['WITH_VACC']}_{md['USE_7DAY_AVG']}_metro_0")
PY
)"
PARAMS="${CALIBRATION:-$CAL_DIR/calibrated_params.txt}"
[[ -f "$PARAMS" ]] || echo "[run] no calibration at $PARAMS -- only OK if BETA, INITIAL_INFECTED and K are all set in config.yaml" \
                         "(otherwise run calibrate.sh, or point CALIBRATION= at an existing calibrated_params.txt)"
OUT="results/$FIPS/${RUN_NAME:-run}"

# cf_raw_<type>.csv is appended to, so a run always starts from an empty folder
if [[ -d "$OUT" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[run] $OUT exists; set OVERWRITE=1 or choose another RUN_NAME"; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"
cp "$ABM_CONFIG" "$OUT/config_used.yaml"

echo "[run] FIPS $FIPS params $PARAMS -> $OUT  $(date)"
GENERATING_COUNTERFACTUAL=true R_SPLINE=0 CF_PARAM_FILE="$(realpath -m "$PARAMS")" CF_OUTPUT_DIR="$PWD/$OUT" \
    $PY main.py "$FIPS"
echo "[run] done -> $OUT  $(date)"
