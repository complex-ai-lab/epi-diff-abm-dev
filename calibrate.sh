#!/bin/bash
# ---------------------------------------------------------------------------
# Calibrate the ABM to observed daily cases.
# All settings come from config.yaml (`epidemic` + `run_settings`), including which counties
# (run_settings.COUNTIES; FIPS=<code> does one county instead).
#
#   bash calibrate.sh                     # every county in COUNTIES, ~27 min each on one A100
#   FIPS=13013 bash calibrate.sh          # just this county
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
PY="${PY:-python}"
export ABM_CONFIG="$(realpath "${CONFIG:-config.yaml}")"

# No FIPS given: do every county in the config's run_settings.COUNTIES (and EXCLUDE_COUNTIES), one after another.
if [[ -z "${FIPS:-}" ]]; then
    COUNTY_LIST="$($PY -c 'import os, sys; sys.path.insert(0, os.getcwd()); from run_counties import select_counties
print(" ".join(select_counties(None, os.environ["ABM_CONFIG"])))' | tail -1)"
    echo "[calibrate] counties from $ABM_CONFIG: $COUNTY_LIST"
    failed=""
    for f in $COUNTY_LIST; do FIPS="$f" bash "$PWD/calibrate.sh" || failed="$failed $f"; done
    [[ -z "$failed" ]] || { echo "[calibrate] failed:$failed"; exit 1; }
    exit 0
fi
# Networks: data/networks/covid_output_p0/<FIPS>, else the scratch copy of the P0 networks
if [[ -z "${NETWORKS_DIR:-}" && ! -d "data/networks/covid_output_p0/$FIPS" ]]; then
    export NETWORKS_DIR="${NETWORKS_DIR_FALLBACK:-/scratch/alrodri_root/alrodri/pkbiswas/epi-diff-abm-dev/networks/covid_output_p0}"
fi

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
