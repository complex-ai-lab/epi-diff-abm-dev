#!/bin/bash
# ---------------------------------------------------------------------------
# Work-essentiality counterfactuals (CF-B) for ONE county, on the ISOLATED
# ACS-C24030 occupation networks (covid_output_essnet). CF types:
#   7  = reference (uniform vaccine, occ+school factual)
#   18 = essential-first vacc, occ+school factual        (vaccine allocation alone)
#   19 = uniform vacc, NON-ESSENTIAL workplaces closed, school factual
#   20 = essential-first vacc, NON-ESSENTIAL closed, school factual
#   21 = essential-first vacc, all open
#   22 = uniform vacc, NON-ESSENTIAL closed, school closed
#   23 = essential-first vacc, NON-ESSENTIAL closed, school closed
#   24 = uniform vacc, ALL workplaces closed, school factual  (retained-benefit ref)
#
# Calibration: results/<FIPS>/TargetedCF/essentiality/calibration/calibrated_params.txt
#   if present (recalibrate_essnet.sh output), else the committed
#   log1p_rw2_l5_rmax5 params. Requires the Phase-2 essnet + patched population
#   (targeted_cf/phase2/build_essnet.sh <FIPS>).
#
# Every artifact -> results/<FIPS>/TargetedCF/<TAG>/. Nothing committed is
# touched; config.yaml restored on exit.
#
#   FIPS     (required)   TAG default essentiality   SEED 42
#   CF_TYPES default 7,18,19,20,21,22,23,24   CF_ITERS default 30
#   PYTHON   default .venv_ds
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
CONFIG="covid_abm/yamls/config.yaml"
FIPS="${FIPS:?set FIPS}"
TAG="${TAG:-essentiality}"
SEED="${SEED:-42}"
CF_TYPES="${CF_TYPES:-7,18,19,20,21,22,23,24}"
CF_ITERS="${CF_ITERS:-30}"

CAUSAL="$(grep '^NETWORKS_DIR' .env | cut -d= -f2)"
ESSNET="$(dirname "$CAUSAL")/covid_output_essnet"
POP_CSV="$REPO_ROOT/targeted_cf/phase2/populations/${FIPS}_population.csv"
OUT="results/${FIPS}/TargetedCF/${TAG}"

RECAL="results/${FIPS}/TargetedCF/essentiality/calibration/calibrated_params.txt"
RMAX5="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
if [[ -f "$RECAL" ]]; then CAL="$RECAL"; CAL_SRC="essnet_recalibrated"; else CAL="$RMAX5"; CAL_SRC="log1p_rw2_l5_rmax5_reused"; fi

for chk in "$CAL:calibration params" \
           "$POP_CSV:patched population (run targeted_cf/phase2/build_essnet.sh $FIPS)" \
           "$ESSNET/$FIPS/mobility_networks/occnets/0.pkl:essnet occnets (run targeted_cf/phase2/build_essnet.sh $FIPS)"; do
    p="${chk%%:*}"; msg="${chk#*:}"
    [[ -e "$p" ]] || { echo "[error] missing $msg -> $p" >&2; exit 1; }
done
[[ -d "populations/pop${FIPS}" ]] || { echo "[error] populations/pop${FIPS} missing" >&2; exit 1; }

export SEED
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

BAK="$(mktemp -d)"
cp "$CONFIG" "$BAK/config.yaml.orig"
trap 'echo "[info] restoring $CONFIG"; cp "$BAK/config.yaml.orig" "$CONFIG"; rm -rf "$BAK"' EXIT

$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

mkdir -p "$OUT/calibration"
DEST_CAL="$OUT/calibration/calibrated_params.txt"
if [[ "$(readlink -f "$CAL")" != "$(readlink -f "$DEST_CAL" 2>/dev/null || echo _)" ]]; then
    cp "$CAL" "$DEST_CAL"
fi
rm -rf "$OUT/counterfactuals"; mkdir -p "$OUT/counterfactuals"

echo "=========================================================="
echo " work-essentiality CF   TAG=${TAG}   FIPS ${FIPS}   SEED ${SEED}"
echo "   calibration : $CAL  ($CAL_SRC)"
echo "   networks    : $ESSNET  (ACS C24030 occnets, symlinked household/school/rand)"
echo "   agent occ   : $POP_CSV"
echo "   CF types    : ${CF_TYPES}   iters: ${CF_ITERS}"
echo "=========================================================="

export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$REPO_ROOT/$CAL"
export CF_OUTPUT_DIR="$REPO_ROOT/$OUT/counterfactuals"
export CF_TYPES CF_ITERS
export NETWORKS_DIR="$ESSNET"
export AGENT_OCC_CSV="$POP_CSV"

$PY main.py "$FIPS"

if [[ "${SKIP_ANALYZE:-0}" != "1" ]]; then
    FIPSES="$FIPS" TAG="$TAG" $PY targeted_cf/analyze_essentiality_cf.py
fi

cat > "$OUT/RUN_CONFIG.txt" <<EOF
TAG=${TAG}
FIPS=${FIPS}
SEED=${SEED}
family=work_essentiality_CF (CF-B)
calibration_source=${CAL}
calibration_kind=${CAL_SRC}
NETWORKS_DIR=${ESSNET}
AGENT_OCC_CSV=${POP_CSV}
waning_mode=fixed
RECOVERED_TO_SUSCEPTIBLE_TIME=60
VACCINATED_TO_SUSCEPTIBLE_TIME=90
cf_types=${CF_TYPES}
cf_iters=${CF_ITERS}
git_rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] $OUT"
