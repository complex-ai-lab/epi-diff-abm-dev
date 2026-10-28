#!/bin/bash
# ---------------------------------------------------------------------------
# Calibrate the ABM to observed daily cases (default county 01031).
# All settings come from config.yaml (`epidemic` + `run_settings`).
#
#   bash calibrate.sh                     # ~27 min on one A100
#   OVERWRITE=1 bash calibrate.sh         # replace an existing calibration
#   SKIP_EXISTING=1 bash calibrate.sh     # exit 0 (instead of 1) if the calibration exists
#   EPOCHS=2 bash calibrate.sh            # smoke test (any run_settings key can be overridden like this)
#   CONFIG=my_config.yaml bash calibrate.sh
#
# Output: result_graphs/<FIPS>/<DATE>/<INITIAL_INFECTION_RATE>_<E->I>_<I->R>_True_True_False_metro_0/
#   calibrated_params.txt  26 weekly R_t, initial infected fraction, k   (used by run.sh)
#   r_daily.txt, spline_coef.txt, spline_meta.json, training_loss.csv, per-epoch plots
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
FIPS="${FIPS:-01031}"
PY="${PY:-python}"
export ABM_CONFIG="$(realpath "${CONFIG:-config.yaml}")"

OUT="result_graphs/$FIPS/$($PY - <<'PY'
import os, yaml
cfg = yaml.safe_load(open(os.environ["ABM_CONFIG"]))
md, epi = cfg["simulation_metadata"], cfg["epidemic"]
mixing, durations = epi.get("MIXING", "network"), epi.get("STAGE_DURATIONS", "fixed")
variant = "" if (mixing, durations) == ("network", "fixed") else f"_{mixing}_{durations}"
print(f"{md['DATE']}/{md['INITIAL_INFECTION_RATE']}_{epi['EXPOSED_TO_INFECTED_TIME']}_{epi['INFECTED_TO_RECOVERED_TIME']}"
      f"_{md['WITH_K']}_{md['WITH_VACC']}_{md['USE_7DAY_AVG']}_metro_0{variant}")
PY
)"

if [[ -f "$OUT/calibrated_params.txt" && "${OVERWRITE:-0}" != "1" ]]; then
    [[ "${SKIP_EXISTING:-0}" == "1" ]] && { echo "[calibrate] $OUT/calibrated_params.txt exists; skipped"; exit 0; }
    echo "[calibrate] $OUT/calibrated_params.txt exists; set OVERWRITE=1 to redo"; exit 1
fi
rm -rf "$OUT"

echo "[calibrate] FIPS $FIPS config $ABM_CONFIG -> $OUT  $(date)"
GENERATING_COUNTERFACTUAL=false $PY main.py "$FIPS"

nrows=$(wc -l < "$OUT/calibrated_params.txt")
[[ "$nrows" == "28" ]] || { echo "[calibrate] calibrated_params.txt has $nrows rows, expected 28" >&2; exit 1; }
echo "[calibrate] done -> $OUT/calibrated_params.txt  $(date)"
