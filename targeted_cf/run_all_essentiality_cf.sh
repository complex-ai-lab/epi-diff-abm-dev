#!/bin/bash
# ---------------------------------------------------------------------------
# Work-essentiality CF (CF-B), all counties:
#   for each FIPS:
#     * build the Phase-2 essnet if missing (census C24030 -> patch pop -> occnets)
#     * read cf7_check.txt; if RECALIBRATE -> recalibrate_essnet.sh
#     * run_essentiality_cf.sh (SKIP_ANALYZE=1)
#   then one analyzer call over ANALYZE_FIPSES.
#
# Env: FIPSES (default "17117 01031 36003"), ANALYZE_FIPSES (default = FIPSES),
#      TAG (default essentiality), SEED (42), CF_TYPES (default 3,7,18..23),
#      CF_ITERS (30), PYTHON (.venv_ds).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"
PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
export TAG="${TAG:-essentiality}"
export SEED="${SEED:-42}"
export CF_TYPES="${CF_TYPES:-7,18,19,20,21,22,23,24}"
export CF_ITERS="${CF_ITERS:-30}"
FIPSES="${FIPSES:-17117 01031 36003}"
ANALYZE_FIPSES="${ANALYZE_FIPSES:-$FIPSES}"

CAUSAL="$(grep '^NETWORKS_DIR' .env | cut -d= -f2)"
ESSNET="$(dirname "$CAUSAL")/covid_output_essnet"
LOGDIR="results/TargetedCF_run/${TAG}/logs"
CHECK="results/TargetedCF_run/${TAG}/cf7_check.txt"
mkdir -p "$LOGDIR"

echo "############################################################"
echo "# work-essentiality CF batch   TAG=${TAG}"
echo "#   sim counties: ${FIPSES}   roll-up: ${ANALYZE_FIPSES}"
echo "#   CF types: ${CF_TYPES}   iters: ${CF_ITERS}   SEED=${SEED}"
echo "############################################################"

for FIPS in $FIPSES; do
    echo; echo ">>> $(date -u +%H:%M:%SZ)  county ${FIPS}"

    if [[ ! -f "targeted_cf/phase2/populations/${FIPS}_population.csv" || \
          ! -e "$ESSNET/${FIPS}/mobility_networks/occnets/0.pkl" ]]; then
        echo ">>> phase-2 build ${FIPS}"
        FIPS="$FIPS" TAG="$TAG" PYTHON="$PY" NETWORKS_ESSNET="$ESSNET" \
            bash targeted_cf/phase2/build_essnet.sh 2>&1 | tee "$LOGDIR/build_${FIPS}.log"
    fi

    VERDICT="RECALIBRATE"
    if [[ -f "$CHECK" ]] && grep -q "^${FIPS} .*REUSE" "$CHECK"; then VERDICT="REUSE"; fi
    if [[ ! -f "results/${FIPS}/TargetedCF/essentiality/calibration/calibrated_params.txt" && "$VERDICT" == "RECALIBRATE" ]]; then
        echo ">>> $(date -u +%H:%M:%SZ)  recalibrate ${FIPS} (cf7_check: $VERDICT)"
        FIPS="$FIPS" SEED="$SEED" PYTHON="$PY" \
            bash targeted_cf/recalibrate_essnet.sh 2>&1 | tee "$LOGDIR/recal_${FIPS}.log"
    else
        echo ">>> ${FIPS} calibration: $VERDICT (using $([[ "$VERDICT" == REUSE ]] && echo log1p_rw2_l5_rmax5 || echo existing essnet recal))"
    fi

    echo ">>> $(date -u +%H:%M:%SZ)  CF run ${FIPS}"
    FIPS="$FIPS" TAG="$TAG" SEED="$SEED" CF_TYPES="$CF_TYPES" CF_ITERS="$CF_ITERS" \
        PYTHON="$PY" SKIP_ANALYZE=1 \
        bash targeted_cf/run_essentiality_cf.sh 2>&1 | tee "$LOGDIR/run_${FIPS}.log"
done

echo; echo ">>> $(date -u +%H:%M:%SZ)  analysis (roll-up: ${ANALYZE_FIPSES})"
FIPSES="$(echo "$ANALYZE_FIPSES" | tr ' ' ',')" TAG="$TAG" \
    $PY targeted_cf/analyze_essentiality_cf.py 2>&1 | tee "$LOGDIR/analyze.log"

echo; echo "[all done] TAG=${TAG}"
echo "  per county : results/<FIPS>/TargetedCF/${TAG}/"
echo "  roll-ups   : results/TargetedCF_run/${TAG}/"
