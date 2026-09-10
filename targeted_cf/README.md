# `targeted_cf/` — targeted-allocation counterfactual families

Shared scripts dir for the CF families in
`plans/Task_ Targeted-Allocation Counterfactuals.md`. Each family: a
`run_<fam>_cf.sh` + `run_all_<fam>_cf.sh` + `analyze_<fam>_cf.py` trio, output
under `results/<FIPS>/TargetedCF/<TAG>/` (mirrors the LossCompare layout) with
per-iteration uncertainty (mean + 10–90 band on line plots, bar = mean ± 1 SD,
CRN-paired Δ-vs-CF7). New CF types are append-only in `abm_nets.py` /
`transition.py`.

| family | CF types | runbook | calibration |
|--------|----------|---------|-------------|
| **CF-A age priority** | 12–17 | `Tasks/age_priority_cf_run.md` | reuse `log1p_rw2_l5_rmax5` |
| **CF-B work essentiality** | 18–23 (+3,7 refs) | `Tasks/work_essentiality_cf_run.md` | Phase-2: ACS C24030 occ nets in isolated `covid_output_essnet`, recalibrate `…_essnet` if CF7 drifts (`targeted_cf/phase2/`) |

---

## CF-A — age-priority vaccination

Six CF types crossing **vaccine allocation priority** with **school/work
closure**, using the committed `log1p_rw2_l5_rmax5` calibration
(fixed_60 waning, vacc 90) as a frozen model — **no recalibration**.

| CF | vaccine priority | school | work |
|----|------------------|--------|------|
| 7  | uniform (reference) | factual | factual |
| 12 | elderly-first     | factual | factual |
| 13 | working-age-first | factual | factual |
| 14 | working-age-first | closed  | open   |
| 15 | working-age-first | open    | open   |
| 16 | elderly-first     | open    | closed |
| 17 | elderly-first     | closed  | closed |

- elderly-first  = `65A → 50t64 → 40t49 → 30t39 → 20t29 → U19`
- working-age-first = `{20t29,30t39,40t49,50t64}` (equal), then `65A`, then `U19`
- daily dose count = factual `vaccines(t)` in every CF.

## Run

```bash
# smoke test (2 iters)
FIPS=01031 CF_ITERS=2 TAG=age_priority_smoke bash targeted_cf/run_age_priority_cf.sh

# one county, full
FIPS=01031 bash targeted_cf/run_age_priority_cf.sh

# all counties + analysis
FIPSES="17117 01031 36003" bash targeted_cf/run_all_age_priority_cf.sh
```

## Model changes (all reused calibrations stay valid)

| file | change |
|------|--------|
| `covid_abm/substeps/new_transmission/transition.py` | `AGE_PRIORITY` / `CF_VACC_PRIORITY` constants; `_age_group_masks` (correct child/adult/elderly split); `apply_vaccines(..., agents_ages, priority)` age-priority topk offset (susceptibles only); `logic_map` rows 12–17; per-age-group `age_incidence_history`; `save_age_incidence_to_disk` |
| `abm_nets.py` | `CF_POLICY_DESC` / `CF_FOLDER_MAP` rows 12–17; default CF range `1..17`; collect + average `age_incidence` over the 30 CF iters |

`get_mean_agent_interactions` (which feeds transmission) is **deliberately left
mis-labelled** — its split is baked into every committed calibration; re-labelling
it would change the factual dynamics. The corrected split is used only by the
diagnostic age outputs.

## Outputs — every number carries a spread

`results/<FIPS>/TargetedCF/<TAG>/` — mirrors the `results/<FIPS>/LossCompare/<TAG>/`
layout: `calibration/`, `counterfactuals/…`, `cf_raw_mean_<ct>.csv`,
`cf_mean_summary.csv` (`*_mean` / `*_sd`), `cf7_fit.png`, `cf_all_mean.png`,
`R_weekly.png`, plus `age_breakdown_<FIPS>.csv` and `age_bars_<FIPS>.png`.

- **line plots** (`cf7_fit.png`, `cf_all_mean.png`, and each
  `counterfactuals/counterfactual_results<ct>.png` — the analyzer overwrites the
  single-draw plot `execute()` leaves) = n-iter **mean + 10–90 percentile band**.
- **`age_bars_<FIPS>.png`** (no panel titles) left panel = totals by group,
  y-axis "new infections", bar = mean, whisker = ± 1 SD across iters; right panel
  = **per-iteration paired** Δ-vs-CF7 as a **percent**, y-axis
  "new infection vs CF7 (%)" (`pct_i = (total(CFk,i) − total(CF7,i)) /
  total(CF7,i) × 100`; CRN pairs iteration `i`, so this SD is ~5–15× tighter than
  differencing marginal means).
- **`age_breakdown_<FIPS>.csv`** — `{group}_cases_{mean,sd}`, paired
  `d_{group}_{mean,sd,p10,p90}` (absolute) and `d_{group}_pct_{mean,sd}`
  (percent).

`results/TargetedCF_run/<TAG>/` — `cf_mean_summary_ALL.csv`, `cf7_fit_ALL.csv`,
`age_breakdown_ALL.csv`, `age_bars_ALL.png` (paired Δ absolute ± 1 SD, faceted),
`HEADLINE.txt` (paired `d_i` as `mean ± SD [p10, p90]`; single seed — repeat
43/44 for load-bearing numbers), `logs/`.

The analyzer **fails loud** (non-zero exit) on any missing file, wrong-shape
series, iter-count mismatch between `cf_raw` and the per-iteration age file, or a
per-CF group-sum vs reported-cases ratio outside `[0.80, 1.03]`.

---

## CF-B — work essentiality  (spec: `Tasks/work_essentiality_cf_run.md`)

Six CF types (18–23, + CF 3/7 as references) crossing **vaccine allocation**
(uniform vs essential-worker-first) with **targeted vs blanket workplace
closure** (non-essential-only vs all workplaces).

**Phase 2 first** — the committed occupation data is broken (17117 has zero FOOD
workers, HEALTHCARE 57 %). `targeted_cf/phase2/` re-sources industry from ACS
5-year `C24030` into isolated folders, rebuilds only the occupational networks
into a sibling scratch dir `covid_output_essnet` (household/school/random nets
symlinked, unchanged), and re-checks the calibration. Nothing committed is
overwritten.

```bash
PY=/home/pkbiswas/.conda/envs/.venv_ds/bin/python
# Phase 2 build + industry report + CF7 check
PYTHON=$PY FIPSES="17117 01031 36003" TAG=essentiality \
  bash targeted_cf/phase2/build_all_essnet.sh
# full run (recalibrates flagged counties, then CF {3,7,18..23} x 30 x 3 + analysis)
PYTHON=$PY FIPSES="17117 01031 36003" TAG=essentiality \
  bash targeted_cf/run_all_essentiality_cf.sh
```

Outputs (in addition to the CF-A-style set): `industry_mix.txt` (committed vs
C24030 sector shares + essential/non-essential/not-in-workforce headcounts),
`cf7_check.txt` (REUSE/RECALIBRATE per county), `essential_breakdown_<FIPS>.csv`,
`essential_bars_<FIPS>.png`, `targeted_vs_full_closure_ALL.csv`
(`retained_benefit = (CF7−CF19)/(CF7−CF3)` per iteration ± SD).

`essential` vs `non-essential` crosswalk (CISA v4.0 → 19 NAICS sectors) is in
`transition.py::ESSENTIAL_SECTORS`. FOOD is essential by default — flipping it is
the sensitivity axis (`TAG=essentiality_food_ne`).

---

## Household-spillover secondary-protection analysis (hSAR)

**Output-only. No new CF types, no engine changes.** `abm_nets.py`,
`transition.py`, `main.py`, the calibration loop and CF types 1–24 are
untouched; the only new files are `targeted_cf/spillover_hook.py`,
`targeted_cf/analyze_spillover.py`, `run_spillover_cf.sh`,
`run_all_spillover_cf.sh`.

`spillover_hook.py` **re-runs the existing CF simulation** (same params, seed,
networks, waning config — CRN-paired with the committed CF-A / CF-B run) and,
from the per-step `disease_stage` trajectory the runner already keeps in
`runner.state_trajectory`, measures the **household secondary attack rate
(hSAR)** in a rolling 14-day window:

- **hSAR_vacc(t)** — of the susceptible cohabitants (at window start) of agents
  with ≥1 ever-vaccinated household member, the fraction infected during the
  window.
- **hSAR_unvacc(t)** — same, for households where no member was ever vaccinated.
- **protection gap** = `hSAR_unvacc − hSAR_vacc` (positive ⇒ spillover-protection
  signal). Headline = its **per-iteration PAIRED Δ vs CF7** (`d_i = gap_k(i) −
  gap_7(i)`), mean ± SD [p10, p90], in **percentage points**.

1-person households have no cohabitant → excluded from every hSAR (`hh1` = n/a).
Age cut = child/adult/elderly susceptible cohabitant (age.pickle); size cut =
`hh1`/`hh2`/`hh3p`. Essential-status cut is essentiality-family only.

This is a **correlational** output of the simulation (whose gets the dose vs the
infection rate of their cohabitants), **not** a mechanistic within-household
contact path. Say so in the paper.

```bash
PY=/home/pkbiswas/.conda/envs/.venv_ds/bin/python
# prerequisite: the CF-A run for the county must already exist
ls results/17117/TargetedCF/age_priority/counterfactuals/cf_raw_7.csv
# smoke (2 iters, one county)
PYTHON=$PY FIPS=01031 CF_ITERS=2 bash targeted_cf/run_spillover_cf.sh
# full: all counties, age-priority family, + cross-county roll-up
PYTHON=$PY FIPSES="17117 01031 36003" bash targeted_cf/run_all_spillover_cf.sh
# also do the essentiality family (needs its CF-B run to exist):
PYTHON=$PY FIPSES="17117 01031 36003" FAMILIES="age_priority essentiality" \
  bash targeted_cf/run_all_spillover_cf.sh
# re-analyse without re-running the sim:
PYTHON=$PY $PY targeted_cf/analyze_spillover.py --fipses 17117 01031 36003 --families age_priority
```

Outputs land in the **same** `results/<FIPS>/TargetedCF/<family>/` dir as the CF
run (existing artifacts untouched):

| path | contents |
|------|----------|
| `counterfactuals/counterfactual_data<ct>_hh_spillover_raw.csv` | wide: `t` + one column-set per iter, per metric (`hSAR_vacc_<i>`, `hSAR_unvacc_child_<i>`, …) |
| `counterfactuals/counterfactual_data<ct>_hh_spillover.csv` | per-day mean across iters |
| `spillover_summary_<FIPS>.csv` | CF × `hSAR_{vacc,unvacc}_{mean,sd,p10,p90}`, paired `dSAR_*`, `protection_gap_*`, `d_protection_gap_*`, and the age / hh-size cuts |
| `spillover_cf7_regression_<FIPS>.txt` | mandatory CF7 case-total check: hook total vs committed `cf_raw_7.csv`, ratio must be ∈ [0.97, 1.03] (else the run raises) |
| `spillover_hSAR_<FIPS>.png` | 14-day hSAR time series, mean + 10–90 band, one line per CF, vline at vaccine rollout (no titles) |
| `spillover_bars_<FIPS>.png` | left: absolute hSAR by CF (vacc vs unvacc); right: paired Δ protection-gap vs CF7 (pp) |
| `spillover_age_bars_<FIPS>.png` | child/adult/elderly: paired Δ protection-gap vs CF7 (pp), one bar per CF |

Cross-county roll-up — `results/TargetedCF_run/spillover/`:
`spillover_summary_ALL.csv`, `spillover_bars_ALL_<family>.png`,
`spillover_age_bars_ALL_<family>.png`, `HEADLINE.txt`, `logs/`.

Fails loud (RuntimeError, non-zero exit) on any missing/short raw file, a
Household-ID / `num_agents` mismatch, an hSAR outside [0, 1], an age/size
denominator group-sum mismatch, a NaN in the paired Δ of the overall
vacc/unvacc hSAR, or a CF7 case-total regression outside [0.97, 1.03]
(CRN pairing broken — check SEED / NETWORKS_DIR / CF_PARAM_FILE / vacc-config).
Single seed (42); for a load-bearing number re-run with SEED=43/44 (own TAGs)
and report the across-seed spread.
