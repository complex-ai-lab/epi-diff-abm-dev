#!/bin/bash
# ---------------------------------------------------------------------------
# Household-spillover hSAR analysis for every county, then one cross-county
# roll-up per family. Output-only (no new CF types, no engine changes).
#
# For each FIPS:
#   1. run_spillover_cf.sh FAMILY=age_priority   (SKIP_ANALYZE=1)
#   2. if results/<FIPS>/TargetedCF/essentiality/counterfactuals/cf_raw_7.csv
#      exists -> run_spillover_cf.sh FAMILY=essentiality (SKIP_ANALYZE=1)
# then analyze_spillover.py over ANALYZE_FIPSES for every family that ran.
#
# Env: FIPSES (default "17117 01031 36003"), ANALYZE_FIPSES (default = FIPSES),
#      FAMILIES (default "age_priority"; add "essentiality" to also do CF-B),
#      SEED (42), CF_ITERS (30), PYTHON.
#
# Usage:
#   PYTHON=$PY FIPSES="17117 01031 36003" bash targeted_cf/run_all_spillover_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
FIPSES="${FIPSES:-17117 01031 36003}"
ANALYZE_FIPSES="${ANALYZE_FIPSES:-$FIPSES}"
FAMILIES="${FAMILIES:-age_priority}"
SEED="${SEED:-42}"
CF_ITERS="${CF_ITERS:-30}"

LOGDIR="results/TargetedCF_run/spillover/logs"
mkdir -p "$LOGDIR"

RAN_FAMILIES=""
for FAM in $FAMILIES; do
    for FIPS in $FIPSES; do
        OUT="results/${FIPS}/TargetedCF/${FAM}"
        if [[ ! -f "$OUT/counterfactuals/cf_raw_7.csv" ]]; then
            echo ">>> [skip] ${FIPS} ${FAM}: no committed CF run ($OUT/counterfactuals/cf_raw_7.csv)"
            continue
        fi
        echo ">>> $(date -u +%H:%M:%SZ)  spillover sim  FIPS=${FIPS}  FAMILY=${FAM}"
        FIPS="$FIPS" FAMILY="$FAM" SEED="$SEED" CF_ITERS="$CF_ITERS" \
            PYTHON="$PY" SKIP_ANALYZE=1 \
            bash targeted_cf/run_spillover_cf.sh 2>&1 | tee "$LOGDIR/run_${FAM}_${FIPS}.log"
        case " $RAN_FAMILIES " in *" $FAM "*) ;; *) RAN_FAMILIES="$RAN_FAMILIES $FAM";; esac
    done
done

if [[ -z "${RAN_FAMILIES// }" ]]; then
    echo "[all done] nothing ran (no committed CF runs found)"
    exit 0
fi

echo ">>> $(date -u +%H:%M:%SZ)  cross-county analysis (families:${RAN_FAMILIES}  counties: ${ANALYZE_FIPSES})"
$PY targeted_cf/analyze_spillover.py \
    --fipses $ANALYZE_FIPSES --families $RAN_FAMILIES \
    2>&1 | tee "$LOGDIR/analyze.log"

echo "[all done] families:${RAN_FAMILIES}"
