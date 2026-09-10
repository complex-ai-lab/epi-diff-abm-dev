#!/bin/bash
# ---------------------------------------------------------------------------
# Household-spillover secondary-protection analysis (hSAR) for ONE county,
# ONE family. Output-only: re-runs the EXISTING CF simulation (same params,
# seed, networks, waning config -> CRN-paired with the committed CF-A / CF-B
# run) and measures the intra-household protection signal. No new CF types,
# no engine changes; config.yaml is restored on exit (trap).
#
# Requires the CF-A (run_age_priority_cf.sh) or CF-B (run_essentiality_cf.sh)
# run for this county to already exist -- the spillover run reuses its network
# and calibration and CRN-pairs against its CF7.
#
# Env knobs:
#   FIPS      (required)   e.g. 01031
#   FAMILY    default age_priority        (age_priority | essentiality)
#   TAG       default = FAMILY            (writes into results/<FIPS>/TargetedCF/<TAG>)
#   SEED      default 42
#   CF_TYPES  default 7,12,13,14,15,16,17 (age_priority) / 7,18,19,20,21,22,23 (essentiality)
#   CF_ITERS  default 30
#   SKIP_ANALYZE=1  -> run the sim + write raw csvs, skip analyze_spillover.py
#   PYTHON    default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#
# Usage:
#   FIPS=01031 CF_ITERS=2 bash targeted_cf/run_spillover_cf.sh          # smoke
#   FIPS=01031 bash targeted_cf/run_spillover_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
CONFIG="covid_abm/yamls/config.yaml"
FIPS="${FIPS:?set FIPS, e.g. 01031}"
FAMILY="${FAMILY:-age_priority}"
TAG="${TAG:-$FAMILY}"
SEED="${SEED:-42}"
CF_ITERS="${CF_ITERS:-30}"

CAUSAL_NETS="$(grep '^NETWORKS_DIR' .env | cut -d= -f2)"
if [[ "$FAMILY" == "age_priority" ]]; then
    CF_TYPES="${CF_TYPES:-7,12,13,14,15,16,17}"
    NETS_DIR="$CAUSAL_NETS"          # inherit .env (same as run_age_priority_cf.sh)
    CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
elif [[ "$FAMILY" == "household_alloc" ]]; then
    CF_TYPES="${CF_TYPES:-7,25,26,27,28}"
    NETS_DIR="$CAUSAL_NETS"          # same causal nets + reused calibration as CF-E
    CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
    # CF 28 essential-household labels: match run_household_alloc_cf.sh (auto C24030
    # patch when present, else committed Occupations).
    PHASE2_POP="$REPO_ROOT/targeted_cf/phase2/populations/${FIPS}_population.csv"
    if [[ -z "${AGENT_OCC_CSV:-}" && -f "$PHASE2_POP" ]]; then
        export AGENT_OCC_CSV="$PHASE2_POP"
    fi
    [[ -n "${AGENT_OCC_CSV:-}" ]] && echo "[info] CF28 labels: ${AGENT_OCC_CSV}"
elif [[ "$FAMILY" == "essentiality" ]]; then
    CF_TYPES="${CF_TYPES:-7,18,19,20,21,22,23}"
    NETS_DIR="$(dirname "$CAUSAL_NETS")/covid_output_essnet"
    CAL="results/${FIPS}/TargetedCF/essentiality/calibration/calibrated_params.txt"
    if [[ ! -f "$CAL" ]]; then
        echo "[warn] $CAL missing -> falling back to reused log1p_rw2_l5_rmax5 params" >&2
        CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
    fi
    export AGENT_OCC_CSV="${AGENT_OCC_CSV:-$REPO_ROOT/targeted_cf/phase2/populations/${FIPS}_population.csv}"
else
    echo "[error] FAMILY must be age_priority | household_alloc | essentiality (got '$FAMILY')" >&2
    exit 1
fi

OUT="results/${FIPS}/TargetedCF/${TAG}"

if [[ ! -f "$CAL" ]]; then
    echo "[error] calibration not found: $CAL" >&2
    exit 1
fi
if [[ ! -f "$OUT/counterfactuals/cf_raw_7.csv" ]]; then
    echo "[error] $OUT/counterfactuals/cf_raw_7.csv not found." >&2
    echo "[error] Run the CF-A/CF-B run for ${FIPS} first (run_age_priority_cf.sh /" >&2
    echo "[error] run_essentiality_cf.sh); the spillover analysis reuses its network +" >&2
    echo "[error] calibration and CRN-pairs against its CF7." >&2
    exit 1
fi
if [[ ! -d "populations/pop${FIPS}" ]]; then
    echo "[error] populations/pop${FIPS} does not exist" >&2
    exit 1
fi

export SEED
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$REPO_ROOT/$CAL"
# transition.py reads os.getenv("NETWORKS_DIR") after load_dotenv (which does NOT
# override an existing env var), so exporting here wins without editing .env.
export NETWORKS_DIR="$NETS_DIR"

if [[ ! -d "$NETS_DIR/$FIPS/mobility_networks" ]]; then
    echo "[error] networks not found: $NETS_DIR/$FIPS/mobility_networks" >&2
    exit 1
fi

BAK="$(mktemp -d)"
cp "$CONFIG" "$BAK/config.yaml.orig"
restore() { echo "[info] restoring $CONFIG"; cp "$BAK/config.yaml.orig" "$CONFIG"; rm -rf "$BAK"; }
trap restore EXIT

# match the waning config the reused calibration assumed (fixed, R->S 60d, VACC->S 90d)
$PY vaccinated_test/set_vacc_config.py --mode fixed \
    --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90

echo "=========================================================="
echo " household-spillover hSAR   FAMILY=${FAMILY}   FIPS ${FIPS}   SEED ${SEED}"
echo "   calibration : $CAL  (frozen, reused)"
echo "   networks    : $NETS_DIR"
echo "   CF types    : ${CF_TYPES}   iters: ${CF_ITERS}"
echo "   output      : $OUT   (spillover_* + counterfactuals/*_hh_spillover*)"
echo "=========================================================="

ANALYZE_FLAG=""
[[ "${SKIP_ANALYZE:-0}" == "1" ]] && ANALYZE_FLAG="--skip-analyze"

$PY targeted_cf/spillover_hook.py \
    --fips "$FIPS" --cf-types "$CF_TYPES" --n-iters "$CF_ITERS" \
    --output-dir "$OUT" --family "$FAMILY" $ANALYZE_FLAG

cat > "$OUT/SPILLOVER_RUN_CONFIG.txt" <<EOF
FAMILY=${FAMILY}
FIPS=${FIPS}
SEED=${SEED}
analysis=household_spillover_hSAR (output-only, no new CF types)
calibration_source=${CAL}
networks=${NETS_DIR}
waning_mode=fixed  RECOVERED_TO_SUSCEPTIBLE_TIME=60  VACCINATED_TO_SUSCEPTIBLE_TIME=90
cf_types=${CF_TYPES}
cf_iters=${CF_ITERS}
git_rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[done] $OUT"
