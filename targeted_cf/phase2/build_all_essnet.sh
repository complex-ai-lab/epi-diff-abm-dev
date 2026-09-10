#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 2: build the isolated essnet for every county, write the industry-mix
# report, then run the CF7 check (essnet nets + reused log1p_rw2_l5_rmax5 params)
# to decide REUSE vs RECALIBRATE per county.
#
#   FIPSES  default "17117 01031 36003"
#   TAG     default essentiality
#   PYTHON  default .venv_ds
#   SEED    default 42
#
# Outputs:
#   targeted_cf/phase2/{data,populations}/ , /scratch/.../covid_output_essnet/
#   results/TargetedCF_run/<TAG>/industry_mix.txt
#   results/TargetedCF_run/<TAG>/cf7_check.txt
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"
PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
TAG="${TAG:-essentiality}"
SEED="${SEED:-42}"
FIPSES="${FIPSES:-17117 01031 36003}"
CONFIG="covid_abm/yamls/config.yaml"

CAUSAL="$(grep '^NETWORKS_DIR' .env | cut -d= -f2)"
ESSNET="$(dirname "$CAUSAL")/covid_output_essnet"
export NETWORKS_ESSNET="$ESSNET"

REPORT="results/TargetedCF_run/${TAG}/industry_mix.txt"
CHECK="results/TargetedCF_run/${TAG}/cf7_check.txt"
mkdir -p "results/TargetedCF_run/${TAG}/logs"
: > "$REPORT"; : > "$CHECK"

export SEED CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" MPLBACKEND=Agg

BAK="$(mktemp -d)"; cp "$CONFIG" "$BAK/config.yaml.orig"
trap 'echo "[info] restoring $CONFIG"; cp "$BAK/config.yaml.orig" "$CONFIG"; rm -rf "$BAK"' EXIT
$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

for FIPS in $FIPSES; do
    echo; echo ">>> $(date -u +%H:%M:%SZ)  phase-2 build ${FIPS}"
    FIPS="$FIPS" TAG="$TAG" PYTHON="$PY" bash targeted_cf/phase2/build_essnet.sh \
        2>&1 | tee "results/TargetedCF_run/${TAG}/logs/build_${FIPS}.log"

    echo ">>> $(date -u +%H:%M:%SZ)  CF7 check ${FIPS}  (essnet nets + rmax5 params)"
    CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
    [[ -f "$CAL" ]] || { echo "[error] $CAL missing"; exit 1; }
    OUT="results/${FIPS}/TargetedCF/${TAG}_p2check"
    rm -rf "$OUT"; mkdir -p "$OUT/counterfactuals"
    GENERATING_COUNTERFACTUAL=true \
      CF_PARAM_FILE="$REPO_ROOT/$CAL" \
      CF_OUTPUT_DIR="$REPO_ROOT/$OUT/counterfactuals" \
      CF_TYPES=7 CF_ITERS="${CF7_ITERS:-30}" \
      NETWORKS_DIR="$ESSNET" \
      $PY main.py "$FIPS" 2>&1 | tee "results/TargetedCF_run/${TAG}/logs/cf7check_${FIPS}.log"

    $PY targeted_cf/phase2/cf7_check.py "$FIPS" --tag "$TAG" \
        --cfdir "$OUT/counterfactuals"
done

echo; echo "=========================================================="
echo " industry mix : $REPORT"
echo " CF7 check    : $CHECK"
cat "$CHECK"
echo "=========================================================="
