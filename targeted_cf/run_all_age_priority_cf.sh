#!/bin/bash
# ---------------------------------------------------------------------------
# Drive run_age_priority_cf.sh for every county, then analyse.
#
# Env knobs (all optional):
#   TAG            default age_priority
#   FIPSES         default "01031"   counties to SIMULATE (space-separated)
#   ANALYZE_FIPSES default = FIPSES  counties to include in the roll-up / HEADLINE
#                  (set this to every county that has results on disk when you are
#                   adding counties incrementally, so the roll-up stays complete)
#   SEED           default 42
#   CF_TYPES       default 7,12,13,14,15,16,17
#   CF_ITERS       default 30
#   PYTHON         default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#
# Usage:
#   bash targeted_cf/run_all_age_priority_cf.sh
#   FIPSES="17117 01031 36003" bash targeted_cf/run_all_age_priority_cf.sh
#   # add two counties to an existing 01031 run, roll up all three:
#   FIPSES="17117 36003" ANALYZE_FIPSES="01031 17117 36003" \
#     bash targeted_cf/run_all_age_priority_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
export TAG="${TAG:-age_priority}"
export SEED="${SEED:-42}"
export CF_TYPES="${CF_TYPES:-7,12,13,14,15,16,17}"
export CF_ITERS="${CF_ITERS:-30}"
FIPSES="${FIPSES:-01031}"
ANALYZE_FIPSES="${ANALYZE_FIPSES:-$FIPSES}"

LOGDIR="results/TargetedCF_run/${TAG}/logs"
mkdir -p "$LOGDIR"

echo "############################################################"
echo "# age-priority CF batch   TAG=${TAG}"
echo "#   counties: ${FIPSES}   SEED=${SEED}"
echo "#   CF types: ${CF_TYPES}   iters: ${CF_ITERS}"
echo "############################################################"

for FIPS in $FIPSES; do
    echo
    echo ">>> $(date -u +%H:%M:%SZ)  county ${FIPS}"
    FIPS="$FIPS" TAG="$TAG" SEED="$SEED" CF_TYPES="$CF_TYPES" CF_ITERS="$CF_ITERS" \
        PYTHON="$PY" SKIP_ANALYZE=1 \
        bash targeted_cf/run_age_priority_cf.sh \
        2>&1 | tee "$LOGDIR/run_${FIPS}.log"
done

echo
echo ">>> $(date -u +%H:%M:%SZ)  analysis (roll-up counties: ${ANALYZE_FIPSES})"
FIPSES="$(echo "$ANALYZE_FIPSES" | tr ' ' ',')" TAG="$TAG" \
    $PY targeted_cf/analyze_age_priority_cf.py \
    2>&1 | tee "$LOGDIR/analyze.log"

echo
echo "[all done] TAG=${TAG}"
echo "  per county : results/<FIPS>/TargetedCF/${TAG}/"
echo "  roll-ups   : results/TargetedCF_run/${TAG}/"
