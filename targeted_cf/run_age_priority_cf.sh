#!/bin/bash
# ---------------------------------------------------------------------------
# Age-priority vaccination counterfactuals (CF-A) for ONE county.
#
# Reuses the committed `log1p_rw2_l5_rmax5` calibration (fixed_60 waning,
# vacc 90) as a frozen model -- NO recalibration -- and runs CF types
#   7  = reference (uniform vaccine, factual school + work)
#   12 = elderly-first vacc, school factual, work factual
#   13 = working-age-first vacc, school factual, work factual
#   14 = working-age-first vacc, school closed, work open
#   15 = working-age-first vacc, school open,   work open
#   16 = elderly-first vacc,     school open,   work closed
#   17 = elderly-first vacc,     school closed, work closed
# with common random numbers across CF types.
#
# Every artifact -> results/<FIPS>/TargetedCF/<TAG>/  (mirrors the
# results/<FIPS>/LossCompare/<TAG>/ layout). result_graphs/, the committed
# baselines, vaccinated_test/, waning_tests/ are never touched; config.yaml is
# restored on exit (trap).
#
# Env knobs:
#   FIPS      (required)   e.g. 01031
#   TAG       default age_priority
#   SEED      default 42
#   CF_TYPES  default 7,12,13,14,15,16,17
#   CF_ITERS  default 30
#   PYTHON    default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#
# Usage:
#   FIPS=01031 bash targeted_cf/run_age_priority_cf.sh
#   FIPS=01031 CF_ITERS=2 TAG=age_priority_smoke bash targeted_cf/run_age_priority_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
CONFIG="covid_abm/yamls/config.yaml"
FIPS="${FIPS:?set FIPS, e.g. 01031}"
TAG="${TAG:-age_priority}"
SEED="${SEED:-42}"
CF_TYPES="${CF_TYPES:-7,12,13,14,15,16,17}"
CF_ITERS="${CF_ITERS:-30}"
DATE="202010-202104"

CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
OUT="results/${FIPS}/TargetedCF/${TAG}"

if [[ ! -f "$CAL" ]]; then
    echo "[error] reused calibration not found: $CAL" >&2
    echo "[error] expected the log1p_rw2_l5_rmax5 fixed_60 / vacc 90 calibration for ${FIPS}." >&2
    exit 1
fi
if [[ ! -d "populations/pop${FIPS}" ]]; then
    echo "[error] populations/pop${FIPS} does not exist" >&2
    exit 1
fi

export SEED
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

BAK="$(mktemp -d)"
cp "$CONFIG" "$BAK/config.yaml.orig"
restore() { echo "[info] restoring $CONFIG"; cp "$BAK/config.yaml.orig" "$CONFIG"; rm -rf "$BAK"; }
trap restore EXIT

# Match the waning config the reused calibration assumed: fixed, R->S 60d,
# VACCINATED->S 90d. (Targeted line-replace, every other line untouched.)
$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

mkdir -p "$OUT/calibration"
cp "$CAL" "$OUT/calibration/calibrated_params.txt"
rm -rf "$OUT/counterfactuals"
mkdir -p "$OUT/counterfactuals"

echo "=========================================================="
echo " age-priority CF   TAG=${TAG}   FIPS ${FIPS}   SEED ${SEED}"
echo "   calibration : $CAL  (frozen, no recalibration)"
echo "   waning      : fixed  R->S 60d   VACCINATED->S 90d"
echo "   CF types    : ${CF_TYPES}   iters: ${CF_ITERS}"
echo "   output      : $OUT"
echo "=========================================================="

export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$REPO_ROOT/$CAL"
export CF_OUTPUT_DIR="$REPO_ROOT/$OUT/counterfactuals"
export CF_TYPES
export CF_ITERS

$PY main.py "$FIPS"

# Finalise the plots: the per-iteration counterfactual_results<ct>.png that
# main.py/execute() leaves behind is a single draw; the analyzer overwrites it
# with the n-iter mean +/- 10-90 band and produces the age breakdown + bars.
# (run_all_age_priority_cf.sh sets SKIP_ANALYZE=1 and runs the analyzer once for
#  all counties itself.)
if [[ "${SKIP_ANALYZE:-0}" != "1" ]]; then
    FIPSES="$FIPS" TAG="$TAG" $PY targeted_cf/analyze_age_priority_cf.py
fi

cat > "$OUT/RUN_CONFIG.txt" <<EOF
TAG=${TAG}
FIPS=${FIPS}
SEED=${SEED}
family=age_priority_vaccination_CF (CF-A)
calibration_source=${CAL}
recalibration=none
waning_mode=fixed
RECOVERED_TO_SUSCEPTIBLE_TIME=60
VACCINATED_TO_SUSCEPTIBLE_TIME=90
cf_types=${CF_TYPES}
cf_iters=${CF_ITERS}
git_rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] $OUT"
