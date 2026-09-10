#!/bin/bash
# ---------------------------------------------------------------------------
# Drive run_household_alloc_cf.sh (CF-E, plan sec 10.2) for every county, then
# analyse: CF-A-style cases/age breakdown + the household-spillover hSAR pass +
# the cross-county roll-ups (incl. the gap-vs-total headline figure).
#
# Env knobs (all optional):
#   TAG            default household_alloc
#   FIPSES         default "01031"   counties to SIMULATE (space-separated)
#   ANALYZE_FIPSES default = FIPSES  counties in the roll-up / HEADLINE
#   SEED           default 42
#   CF_TYPES       default 7,25,26,27,28
#   CF_ITERS       default 30
#   PYTHON         default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#
# Usage:
#   FIPSES="17117 01031 36003" bash targeted_cf/run_all_household_alloc_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
export TAG="${TAG:-household_alloc}"
export SEED="${SEED:-42}"
export CF_TYPES="${CF_TYPES:-7,25,26,27,28}"
export CF_ITERS="${CF_ITERS:-30}"
FIPSES="${FIPSES:-01031}"
ANALYZE_FIPSES="${ANALYZE_FIPSES:-$FIPSES}"

LOGDIR="results/TargetedCF_run/${TAG}/logs"
SPILL_LOGDIR="results/TargetedCF_run/spillover/logs"
mkdir -p "$LOGDIR" "$SPILL_LOGDIR"

echo "############################################################"
echo "# household-alloc CF batch (CF-E)   TAG=${TAG}"
echo "#   counties: ${FIPSES}   SEED=${SEED}"
echo "#   CF types: ${CF_TYPES}   iters: ${CF_ITERS}"
echo "############################################################"

for FIPS in $FIPSES; do
    echo
    echo ">>> $(date -u +%H:%M:%SZ)  county ${FIPS}"
    FIPS="$FIPS" TAG="$TAG" SEED="$SEED" CF_TYPES="$CF_TYPES" CF_ITERS="$CF_ITERS" \
        PYTHON="$PY" SKIP_ANALYZE=1 \
        bash targeted_cf/run_household_alloc_cf.sh \
        2>&1 | tee "$LOGDIR/run_${FIPS}.log"
done

echo
echo ">>> $(date -u +%H:%M:%SZ)  CF-A-style analysis (roll-up: ${ANALYZE_FIPSES})"
FIPSES="$(echo "$ANALYZE_FIPSES" | tr ' ' ',')" TAG="$TAG" \
    $PY targeted_cf/analyze_household_alloc_cf.py \
    2>&1 | tee "$LOGDIR/analyze.log"

echo
echo ">>> $(date -u +%H:%M:%SZ)  household-spillover roll-up (family=household_alloc)"
$PY targeted_cf/analyze_spillover.py \
    --fipses $ANALYZE_FIPSES --families household_alloc \
    2>&1 | tee "$SPILL_LOGDIR/analyze_household_alloc.log"

echo
echo ">>> $(date -u +%H:%M:%SZ)  re-run CF-A analyzer (now with spillover_summary for gap-vs-total)"
FIPSES="$(echo "$ANALYZE_FIPSES" | tr ' ' ',')" TAG="$TAG" \
    $PY targeted_cf/analyze_household_alloc_cf.py \
    2>&1 | tee -a "$LOGDIR/analyze.log"

echo
echo "[all done] TAG=${TAG}"
echo "  per county : results/<FIPS>/TargetedCF/${TAG}/"
echo "  roll-ups   : results/TargetedCF_run/${TAG}/  +  results/TargetedCF_run/spillover/"
