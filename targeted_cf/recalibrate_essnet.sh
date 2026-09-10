#!/bin/bash
# ---------------------------------------------------------------------------
# 251-epoch recalibration of ONE county on the ISOLATED ACS-C24030 occ networks
# (covid_output_essnet), with the SAME knobs as log1p_rw2_l5_rmax5:
#   LOSS_MODE=log1p, RW2 lambda 5, endpoint anchor 5, PARAM_SNAPSHOT_K=20,
#   R_MAX=5, waning fixed R->S 60 / VACCINATED->S 90, SEED 42.
#
# Output -> results/<FIPS>/TargetedCF/essentiality/calibration/
#   (calibrated_params.txt + training diagnostics). run_essentiality_cf.sh picks
#   this up automatically if present.
#
# Nothing committed is touched; result_graphs/<FIPS> is cleaned up; config.yaml
# restored on exit.
#
#   FIPS (required)   SEED default 42   R_MAX default 5   PYTHON default .venv_ds
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"
PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
CONFIG="covid_abm/yamls/config.yaml"
FIPS="${FIPS:?set FIPS}"
DATE="202010-202104"

CAUSAL="$(grep '^NETWORKS_DIR' .env | cut -d= -f2)"
ESSNET="$(dirname "$CAUSAL")/covid_output_essnet"
[[ -e "$ESSNET/$FIPS/mobility_networks/occnets/0.pkl" ]] || {
    echo "[error] essnet occnets missing — run targeted_cf/phase2/build_essnet.sh $FIPS" >&2; exit 1; }

OUT="results/${FIPS}/TargetedCF/essentiality/calibration"
RG_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_True_False_metro_0"
CURR_DIR="result_graphs/${FIPS}/${DATE}/0.0005_3_5_True_False_False_metro_0"
GEN_SRC="results/${FIPS}/0.0005_3_5_metro_0/generated_factual.csv"

export SEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND=Agg
export LOSS_MODE=log1p R_SMOOTH_LAMBDA=5 R_SMOOTH_ORDER=2 R_SMOOTH_ENDPOINT_LAMBDA=5
export PARAM_SNAPSHOT_K=20
export R_MAX="${R_MAX:-5}"
export NETWORKS_DIR="$ESSNET"

BAK="$(mktemp -d)"; cp "$CONFIG" "$BAK/config.yaml.orig"
trap 'echo "[info] restoring $CONFIG"; cp "$BAK/config.yaml.orig" "$CONFIG"; rm -rf "$BAK"' EXIT

$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

mkdir -p "$OUT"
rm -rf "$RG_DIR" "$CURR_DIR"

echo "=========================================================="
echo " essnet RECALIBRATION  FIPS ${FIPS}  SEED ${SEED}  R_MAX ${R_MAX}"
echo "   networks : $ESSNET"
echo "   knobs    : log1p + RW2 l5 + endpoint l5 + K20 + fixed 60/90"
echo "=========================================================="

export GENERATING_COUNTERFACTUAL=false
$PY main.py "$FIPS"

cp -r "$RG_DIR"/. "$OUT/"
[[ -f "$GEN_SRC" ]] && cp "$GEN_SRC" "$OUT/generated_factual.csv"
[[ -e "$CURR_DIR" ]] && { rm -rf "$OUT/_curriculum_phase1_novacc"; mv "$CURR_DIR" "$OUT/_curriculum_phase1_novacc"; }
rm -rf "$RG_DIR"
find "result_graphs/${FIPS}" -type d -empty -delete 2>/dev/null || true
rmdir "result_graphs/${FIPS}/${DATE}" "result_graphs/${FIPS}" 2>/dev/null || true

cat > "$OUT/RECAL_CONFIG.txt" <<EOF
FIPS=${FIPS}
SEED=${SEED}
networks=covid_output_essnet (ACS C24030 occnets)
knobs=log1p_rw2_l5 + endpoint l5 + PARAM_SNAPSHOT_K=20 + R_MAX=${R_MAX} + fixed_60/90
epochs=251
git_rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] $OUT/calibrated_params.txt"
