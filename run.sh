#!/bin/bash
# ---------------------------------------------------------------------------
# Run the ABM (default: factual run = CF 7, 30 stochastic iterations, county 01031).
# Settings come from config.yaml: HORIZON_DAYS sets the number of simulated days;
# R_t / initial infected / k are the calibrated values unless R_T (or R_T_DAILY) /
# INITIAL_INFECTED / K are set there.
#
#   bash run.sh                           # -> results/01031/run/
#   RUN_NAME=r_t_1.2 bash run.sh          # -> results/01031/r_t_1.2/  (keep runs side by side)
#   CALIBRATION=path/calibrated_params.txt bash run.sh   # use a specific calibration
#   CF_TYPES=1,7 CF_ITERS=5 bash run.sh   # other counterfactual types / iterations
#   OVERWRITE=1 bash run.sh               # replace an existing output folder
#   REF_OUT=my_ref/01031 bash run.sh      # where the reference_results-format copy goes
#   CF_TYPES=99 CF_SPEC=counterfactuals/my_cf.yaml bash run.sh
#                                         # custom CF (counterfactuals/TEMPLATE.yaml) -> results/<FIPS>/<spec name>/,
#                                         # and with REF_OUT also copied to $REF_OUT/<spec name>/
#
# Output (results/<FIPS>/<RUN_NAME>/):
#   r_t_daily.csv               day, date, phase, r_t
#   seirv_daily.csv             day, date, S, E, I, R, V  (mean over iterations)
#   cases_sim_vs_observed.csv   day, date, simulated_cases, simulated_cases_sd, observed_cases
#   seir_rates_daily.csv        day, date, phase, r_t, beta, sigma, gamma, ... (SEIRV rates, seirv_fit.py)
#   cf_raw_7.csv, counterfactual_data7*.csv, counterfactual_results7.png
# and, for a factual run, the same files in the reference_results format in
# reference_results_<RUN_NAME>/<FIPS>/ (config_<FIPS>.yaml + the four CSVs above; export_reference.py).
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
mixing, durations = epi.get("MIXING", "network"), epi.get("STAGE_DURATIONS", "fixed")
variant = "" if (mixing, durations) == ("network", "fixed") else f"_{mixing}_{durations}"
print(f"{md['DATE']}/{md['INITIAL_INFECTION_RATE']}_{epi['EXPOSED_TO_INFECTED_TIME']}_{epi['INFECTED_TO_RECOVERED_TIME']}"
      f"_{md['WITH_K']}_{md['WITH_VACC']}_{md['USE_7DAY_AVG']}_metro_0{variant}")
PY
)"
PARAMS="${CALIBRATION:-$CAL_DIR/calibrated_params.txt}"
[[ -f "$PARAMS" ]] || echo "[run] no calibration at $PARAMS -- only OK if R_T (or R_T_DAILY), INITIAL_INFECTED and K are all set in config.yaml" \
                         "(otherwise run calibrate.sh, or point CALIBRATION= at an existing calibrated_params.txt)"
# Custom CF 99: the spec's `name:` names the run folder (unless RUN_NAME is given).
# CF_TYPES / CF_SPEC come from the shell, else from the config's run_settings.
CF_NAME="$($PY - <<'PY'
import os, sys, yaml
sys.path.insert(0, os.getcwd())
from covid_abm.custom_cf import spec_name
rs = yaml.safe_load(open(os.environ["ABM_CONFIG"])).get("run_settings") or {}
types = str(os.environ.get("CF_TYPES") or rs.get("CF_TYPES") or "")
spec = os.environ.get("CF_SPEC") or rs.get("CF_SPEC")
if "99" in [t.strip() for t in types.split(",")]:
    if not spec:
        sys.exit("[run] CF_TYPES has 99 but no CF_SPEC is set")
    print(spec_name(yaml.safe_load(open(spec)) or {}, spec))
PY
)"
RUN_NAME="${RUN_NAME:-${CF_NAME:-run}}"
OUT="results/$FIPS/$RUN_NAME"

# cf_raw_<type>.csv is appended to, so a run always starts from an empty folder
if [[ -d "$OUT" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[run] $OUT exists; set OVERWRITE=1 or choose another RUN_NAME"; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"
cp "$ABM_CONFIG" "$OUT/config_used.yaml"

echo "[run] FIPS $FIPS params $PARAMS -> $OUT  $(date)"
GENERATING_COUNTERFACTUAL=true R_SPLINE=0 CF_PARAM_FILE="$(realpath -m "$PARAMS")" CF_OUTPUT_DIR="$PWD/$OUT" \
    $PY main.py "$FIPS"
if [[ -f "$OUT/seir_rates_daily.csv" ]]; then
    $PY export_reference.py "$OUT" "${REF_OUT:-reference_results_${RUN_NAME:-run}/$FIPS}" "$PARAMS"
fi
if [[ -n "$CF_NAME" && -n "${REF_OUT:-}" ]]; then
    # custom CF: its outputs next to the county's reference files, in a folder named after the spec
    mkdir -p "$REF_OUT/$CF_NAME"
    cp "$OUT"/counterfactual_interventions.csv "$OUT"/cf_spec_used.yaml "$OUT"/cf_raw_99.csv \
       "$OUT"/counterfactual_data99*.csv "$OUT"/counterfactual_results99.png "$OUT"/config_used.yaml "$REF_OUT/$CF_NAME/"
    echo "[run] custom CF $CF_NAME -> $REF_OUT/$CF_NAME"
fi
echo "[run] done -> $OUT  $(date)"
