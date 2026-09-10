#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 2 build for ONE county: ACS C24030 -> patched population -> regenerated
# occ networks -> assembled isolated essnet dir. Nothing committed is touched.
#
#   FIPS   (required)
#   PYTHON default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#   TAG    default essentiality   (industry_mix.txt goes to results/TargetedCF_run/<TAG>/)
#
# Usage:  FIPS=01031 bash targeted_cf/phase2/build_essnet.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"
PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
FIPS="${FIPS:?set FIPS}"
TAG="${TAG:-essentiality}"
REPORT="results/TargetedCF_run/${TAG}/industry_mix.txt"

echo "=========================================================="
echo " Phase 2 build   FIPS ${FIPS}"
echo "=========================================================="

$PY targeted_cf/phase2/fetch_industry_c24030.py "$FIPS"
$PY targeted_cf/phase2/patch_population.py "$FIPS" --report "$REPORT"
$PY targeted_cf/phase2/regen_occnets.py "$FIPS"
$PY targeted_cf/phase2/assemble_essnet.py "$FIPS"

echo "[build_essnet] ${FIPS} done"
