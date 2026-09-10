#!/bin/bash
# ---------------------------------------------------------------------------
# Household-level vs individual-level vaccine allocation (CF-E / plan §10.2)
# for ONE county.
#
# Reuses the committed `log1p_rw2_l5_rmax5` calibration (fixed_60 waning,
# vacc 90) as a frozen model -- NO recalibration, NO essnet networks -- and
# runs CF types
#   7  = reference (individual, uniform-over-susceptibles allocation)
#   25 = whole-household bolus, RANDOM household order
#   26 = whole-household bolus, elderly-home first
#   27 = whole-household bolus, largest households first
#   28 = whole-household bolus, essential-worker homes first
# NPIs (school + work) are factual in every CF; only *which* susceptibles get
# the dose (and how correlated they are within homes) changes. CRN across CF.
#
# CF 28 needs essential-worker household labels (transition._require_agent_essential
# -> the `Occupations` column). By default this script points AGENT_OCC_CSV at
# the row-aligned ACS-C24030 patch (targeted_cf/phase2/populations/) when it
# exists, so "essential household" is a real cut and not ~everyone; otherwise it
# falls back to the committed (skewed) Occupations and prints a warning
# (CF28 then ~= CF25). Set AGENT_OCC_CSV yourself to override.
#
# Every artifact -> results/<FIPS>/TargetedCF/<TAG>/  (mirrors the LossCompare
# layout). result_graphs/, committed baselines, vaccinated_test/, waning_tests/,
# covid_output_essnet, targeted_cf/phase2/ are never touched; config.yaml is
# restored on exit (trap).
#
# Env knobs:
#   FIPS         (required)   e.g. 01031
#   TAG          default household_alloc
#   SEED         default 42
#   CF_TYPES     default 7,25,26,27,28
#   CF_ITERS     default 30
#   AGENT_OCC_CSV  CF28 essential-household labels (auto-set, see above)
#   SKIP_ANALYZE=1  -> run the sim, skip both analyzers (driver does them once)
#   SKIP_SPILLOVER=1 -> run the CF sim + CF-A analyzer, skip the hSAR pass
#   PYTHON       default /home/pkbiswas/.conda/envs/.venv_ds/bin/python
#
# Usage:
#   FIPS=01031 bash targeted_cf/run_household_alloc_cf.sh
#   FIPS=01031 CF_ITERS=2 TAG=household_alloc_smoke bash targeted_cf/run_household_alloc_cf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="/home/pkbiswas/COVID-ABM/epi-diff-abm-dev"
cd "$REPO_ROOT"

PY="${PYTHON:-/home/pkbiswas/.conda/envs/.venv_ds/bin/python}"
CONFIG="covid_abm/yamls/config.yaml"
FIPS="${FIPS:?set FIPS, e.g. 01031}"
TAG="${TAG:-household_alloc}"
SEED="${SEED:-42}"
CF_TYPES="${CF_TYPES:-7,25,26,27,28}"
CF_ITERS="${CF_ITERS:-30}"

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

# --- CF 28 essential-household labels ------------------------------------------
PHASE2_POP="$REPO_ROOT/targeted_cf/phase2/populations/${FIPS}_population.csv"
if [[ -z "${AGENT_OCC_CSV:-}" ]]; then
    if [[ -f "$PHASE2_POP" ]]; then
        export AGENT_OCC_CSV="$PHASE2_POP"
        echo "[info] CF28 essential-household labels: ACS-C24030 patch ($AGENT_OCC_CSV)"
    else
        echo "[warn] CF28 using committed Occupations (skewed; CF28 ~= CF25)." >&2
        echo "[warn] run 'python targeted_cf/phase2/patch_population.py ${FIPS}' for a real split." >&2
    fi
else
    echo "[info] CF28 essential-household labels: AGENT_OCC_CSV=$AGENT_OCC_CSV (caller-set)"
fi
OCC_LABEL_SRC="${AGENT_OCC_CSV:-committed}"

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
echo " household-alloc CF (CF-E)   TAG=${TAG}   FIPS ${FIPS}   SEED ${SEED}"
echo "   calibration : $CAL  (frozen, no recalibration)"
echo "   networks    : covid_output_causal (inherits .env; NO essnet)"
echo "   waning      : fixed  R->S 60d   VACCINATED->S 90d"
echo "   CF types    : ${CF_TYPES}   iters: ${CF_ITERS}"
echo "   CF28 labels : ${OCC_LABEL_SRC}"
echo "   output      : $OUT"
echo "=========================================================="

export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$REPO_ROOT/$CAL"
export CF_OUTPUT_DIR="$REPO_ROOT/$OUT/counterfactuals"
export CF_TYPES
export CF_ITERS

$PY main.py "$FIPS"

cat > "$OUT/RUN_CONFIG.txt" <<EOF
TAG=${TAG}
FIPS=${FIPS}
SEED=${SEED}
family=household_level_allocation_CF (CF-E, plan sec 10.2)
calibration_source=${CAL}
recalibration=none
networks=covid_output_causal
cf28_essential_household_labels=${OCC_LABEL_SRC}
waning_mode=fixed
RECOVERED_TO_SUSCEPTIBLE_TIME=60
VACCINATED_TO_SUSCEPTIBLE_TIME=90
cf_types=${CF_TYPES}
cf_iters=${CF_ITERS}
git_rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# --- household-spillover hSAR pass (re-runs the sim, CRN-paired to CF7 above) --
# Runs FIRST so spillover_summary_<FIPS>.csv exists for the CF-A analyzer's
# combined gap-vs-total panel. Its own analyze_spillover pass is gated by
# SKIP_ANALYZE (the all-county driver sets 1 and analyses once at the end).
if [[ "${SKIP_SPILLOVER:-0}" != "1" ]]; then
    echo ">>> household-spillover hSAR pass"
    FIPS="$FIPS" FAMILY=household_alloc TAG="$TAG" SEED="$SEED" \
        CF_TYPES="$CF_TYPES" CF_ITERS="$CF_ITERS" PYTHON="$PY" \
        SKIP_ANALYZE="${SKIP_ANALYZE:-0}" \
        bash targeted_cf/run_spillover_cf.sh
fi

# --- CF-A-style analyzer (cases, totals, age bars) + combined gap-vs-total ----
if [[ "${SKIP_ANALYZE:-0}" != "1" ]]; then
    FIPSES="$FIPS" TAG="$TAG" $PY targeted_cf/analyze_household_alloc_cf.py
fi

echo "[done] $OUT"
