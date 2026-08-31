#!/bin/bash
# ---------------------------------------------------------------------------
# "Cleaner" immunity-waning validation: freeze ONE calibrated parameter set
# (from the `none` calibration) and only toggle the waning mechanism in a
# forward-only pass. Days 0..~first-recovered are identical across every run;
# later divergence is the pure waning effect.
#
# Contrast with waning_tests/, which re-calibrates (251 epochs) per config so
# each curve is a different fit and the early weeks are not comparable.
#
# Steps:
#   1. (once) reproduce the `none` calibration to recover its calibrated_params
#      -> frozen_cal_params_iso_wane/frozen_params_none.txt
#      result_graphs/.../metro_0/{calibrated_params,training_proportions,
#      training_loss} and config.yaml are backed up + restored, nothing
#      original is clobbered.
#   2. forward-only runs: none / fixed{60,80,100} / stochastic{0.005,0.01,0.02,0.04}
#      -> frozen_cal_params_iso_wane/data/<label>.csv
#
# Usage:
#   bash frozen_cal_params_iso_wane/run_iso_wane.sh [FIPS]
#   FIPS defaults to 01031.  SEED defaults to 42 (same as run_all_sims.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

FIPS="${1:-01031}"
export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export GENERATING_COUNTERFACTUAL=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
CONFIG="covid_abm/yamls/config.yaml"

DATE="202010-202104"
RG_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_True_False_metro_0"
PROP_SRC="${RG_DIR}/training_proportions.csv"
PARAM_SRC="${RG_DIR}/calibrated_params.txt"

OUT_DIR="frozen_cal_params_iso_wane"
DATA_DIR="${OUT_DIR}/data"
BAK_DIR="${OUT_DIR}/_backup"
FROZEN="${OUT_DIR}/frozen_params_none.txt"
mkdir -p "$DATA_DIR" "$BAK_DIR"

# ---- back up everything a calibration run would overwrite ------------------
for f in "$CONFIG" "$PROP_SRC" "$PARAM_SRC" "${RG_DIR}/training_loss.csv"; do
    [[ -f "$f" ]] && cp "$f" "${BAK_DIR}/$(basename "$f").orig"
done

restore() {
    echo "[info] restoring config.yaml + result_graphs/.../metro_0 originals"
    [[ -f "${BAK_DIR}/config.yaml.orig" ]]            && cp "${BAK_DIR}/config.yaml.orig" "$CONFIG"
    [[ -f "${BAK_DIR}/training_proportions.csv.orig" ]] && cp "${BAK_DIR}/training_proportions.csv.orig" "$PROP_SRC"
    [[ -f "${BAK_DIR}/calibrated_params.txt.orig" ]]  && cp "${BAK_DIR}/calibrated_params.txt.orig" "$PARAM_SRC"
    [[ -f "${BAK_DIR}/training_loss.csv.orig" ]]      && cp "${BAK_DIR}/training_loss.csv.orig" "${RG_DIR}/training_loss.csv"
}
trap restore EXIT

# ---- 1. recover the `none` calibrated parameters -------------------------
if [[ ! -f "$FROZEN" ]]; then
    echo "=========================================================="
    echo " reproducing the 'none' calibration to recover its params"
    echo "=========================================================="
    $PY "${OUT_DIR}/../waning_tests/set_waning_config.py" --mode none
    $PY main.py "$FIPS"
    cp "$PARAM_SRC" "$FROZEN"
    echo "[info] frozen params -> $FROZEN"

    # sanity: this run must reproduce the stored none baseline trajectory
    $PY - "$PROP_SRC" "waning_tests/data/none_baseline_reference.csv" <<'EOF'
import sys, pandas as pd, numpy as np
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2])
n = min(len(a), len(b))
d = float(np.max(np.abs(a["infected"].to_numpy()[:n] - b["infected"].to_numpy()[:n])))
print(f"[check] reproduced none calibration: max|Δ infected| vs stored baseline = {d:.3e}")
print("[check] OK" if d < 1e-6 else "[check] WARNING - not bit-identical (inspect)")
EOF
else
    echo "[info] reusing existing $FROZEN"
fi

# ---- 2. forward-only waning sweep with the frozen params ----------------
run_fwd() {
    local label="$1"; shift
    echo ""
    echo "---- forward run: ${label} ----"
    $PY "${OUT_DIR}/iso_wane_forward.py" --fips "$FIPS" --seed "$SEED" \
        --params "$FROZEN" --out "${DATA_DIR}/${label}.csv" "$@"
}

run_fwd "none"           --mode none
run_fwd "fixed_60"       --mode fixed --recovered-to-susceptible-time 60
run_fwd "fixed_80"       --mode fixed --recovered-to-susceptible-time 80
run_fwd "fixed_100"      --mode fixed --recovered-to-susceptible-time 100
run_fwd "stochastic_0.005" --mode stochastic --waning-rate 0.005
run_fwd "stochastic_0.010" --mode stochastic --waning-rate 0.010
run_fwd "stochastic_0.020" --mode stochastic --waning-rate 0.020
run_fwd "stochastic_0.040" --mode stochastic --waning-rate 0.040

echo ""
echo "[done] forward runs complete. Now:"
echo "    $PY ${OUT_DIR}/plot_iso_wane.py"
