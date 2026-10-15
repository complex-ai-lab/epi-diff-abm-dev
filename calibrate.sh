#!/bin/bash
# ---------------------------------------------------------------------------
# Calibrate the ABM for one county (default 01031).
# Every setting comes from covid_abm/yamls/config.yaml (`epidemic` + `run_settings`),
# which main.py exports as environment variables. Same recipe as step 2 of
# run_p0_county.sh: Poisson spline calibration, 251 epochs, lambda = endpoint = 25.
#
#   bash calibrate.sh                 # FIPS=01031, PY=python
#   OVERWRITE=1 bash calibrate.sh     # replace an existing calibration
#   EPOCHS=2 bash calibrate.sh        # smoke test (any run_settings key can be overridden this way)
#
# Output: result_graphs/<FIPS>/<DATE>/<...>_metro_0/
#   calibrated_params.txt  26 weekly beta, initial infected fraction, k   (read by run.sh)
#   r_daily.txt, spline_coef.txt, spline_meta.json, training_loss.csv, <epoch>/ plots
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
FIPS="${FIPS:-01031}"
PY="${PY:-python}"
OUT="$($PY - <<'PY'
import yaml
md = yaml.safe_load(open("covid_abm/yamls/config.yaml"))["simulation_metadata"]
epi = yaml.safe_load(open("covid_abm/yamls/config.yaml"))["epidemic"]
print(f"{md['DATE']}/{md['INITIAL_INFECTION_RATE']}_{epi['EXPOSED_TO_INFECTED_TIME']}_{epi['INFECTED_TO_RECOVERED_TIME']}"
      f"_{md['WITH_K']}_{md['WITH_VACC']}_{md['USE_7DAY_AVG']}_metro_0")
PY
)"
OUT="result_graphs/$FIPS/$OUT"

if [[ -f "$OUT/calibrated_params.txt" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[calibrate] $OUT/calibrated_params.txt exists; set OVERWRITE=1 to redo"; exit 1
fi
rm -rf "$OUT"

echo "[calibrate] FIPS $FIPS -> $OUT  $(date)"
GENERATING_COUNTERFACTUAL=false $PY main.py "$FIPS"

nrows=$(wc -l < "$OUT/calibrated_params.txt")
[[ "$nrows" == "28" ]] || { echo "[calibrate] calibrated_params.txt has $nrows rows, expected 28" >&2; exit 1; }
echo "[calibrate] done -> $OUT/calibrated_params.txt  $(date)"
