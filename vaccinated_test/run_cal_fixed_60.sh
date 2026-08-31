#!/bin/bash
# ---------------------------------------------------------------------------
# fixed_60 vaccinated-state CALIBRATION for one county.
#
#   * config = the committed 0.0005_3_5_True_True_False_metro_0 setup
#     (WITH_K=true, WITH_VACC=true, USE_7DAY_AVG=false, SEED 42) PLUS
#     IMMUNITY_WANING_MODE=fixed, RECOVERED_TO_SUSCEPTIBLE_TIME=60,
#     VACCINATED_TO_SUSCEPTIBLE_TIME=90
#   * full 251-epoch recalibration:  python3 main.py <FIPS>
#   * collects the frozen artefacts into vaccinated_test/data/<FIPS>/:
#       fixed_60_params.txt   (26 weekly R, initial infection rate, k)
#       fixed_60.csv          (training_proportions.csv, incl. `vaccinated`)
#       fixed_60_cases.csv    (aligned simulated vs actual daily cases)
#   * the raw calibration output tree (result_graphs/<FIPS>/.../metro_0 with the
#     per-epoch simulation_results.png) is MOVED to
#       vaccinated_test/results/<FIPS>/calibration_fixed_60/
#     so it is not mistaken for a plain (no-waning) baseline calibration.
#   * config.yaml is restored on exit.
#
# Usage:
#   bash vaccinated_test/run_cal_fixed_60.sh 36003
# ---------------------------------------------------------------------------
set -euo pipefail

FIPS="${1:?usage: run_cal_fixed_60.sh <FIPS>}"
DATE="202010-202104"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
CONFIG="covid_abm/yamls/config.yaml"

RG_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_True_False_metro_0"
GEN_SRC="results/${FIPS}/0.0005_3_5_metro_0/generated_factual.csv"
DATA_DIR="$REPO_ROOT/vaccinated_test/data/${FIPS}"
CAL_KEEP="$REPO_ROOT/vaccinated_test/results/${FIPS}/calibration_fixed_60"

mkdir -p "$DATA_DIR"

export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export GENERATING_COUNTERFACTUAL=false

BAK="$(mktemp -d)"
cp "$CONFIG" "$BAK/config.yaml.orig"
restore() {
    echo "[info] restoring $CONFIG"
    cp "$BAK/config.yaml.orig" "$CONFIG"
    rm -rf "$BAK"
}
trap restore EXIT

if [[ -e "$RG_DIR" ]]; then
    echo "[error] $RG_DIR already exists - refusing to overwrite. Move it aside first."
    exit 1
fi

$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

echo "=========================================================="
echo " fixed_60 CALIBRATION   FIPS ${FIPS}   SEED ${SEED}"
echo "=========================================================="

$PY main.py "$FIPS"

# ---- collect frozen artefacts -------------------------------------------------
cp "${RG_DIR}/calibrated_params.txt"      "${DATA_DIR}/fixed_60_params.txt"
cp "${RG_DIR}/training_proportions.csv"   "${DATA_DIR}/fixed_60.csv"
[[ -f "$GEN_SRC" ]] && cp "$GEN_SRC"      "${DATA_DIR}/fixed_60_cases.csv"

# ---- move the raw calibration tree out of result_graphs/ --------------------
rm -rf "$CAL_KEEP"
mkdir -p "$(dirname "$CAL_KEEP")"
mv "$RG_DIR" "$CAL_KEEP"
rmdir -p "$(dirname "$RG_DIR")" 2>/dev/null || true

echo ""
echo "[done] frozen params -> ${DATA_DIR}/fixed_60_params.txt"
echo "[done] calibration tree kept at -> ${CAL_KEEP}"
$PY - "$DATA_DIR/fixed_60_params.txt" <<'EOF'
import sys, numpy as np
v = np.loadtxt(sys.argv[1]).ravel()
R = v[:26]
print(f"       mean R {R.mean():.3f}  range [{R.min():.2f}, {R.max():.2f}]  "
      f"init {v[-2]*1e5:.1f}/100k  k {v[-1]:.3f}")
EOF
