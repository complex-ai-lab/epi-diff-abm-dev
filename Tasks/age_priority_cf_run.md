# Runbook / task: age-priority vaccination counterfactuals (CF-A only)

**Status:** IMPLEMENTED (2026-09-03). Code changes in
`covid_abm/substeps/new_transmission/transition.py`, `abm_nets.py`, and
`targeted_cf/`. Smoke test + full 01031 run done. The "Implementation" section
below is the spec that was built to; "Run" is repeatable.

**Scope — this task only does CF-A** (age-prioritised vaccination × school/work
NPI combinations); cases only. No essentiality / budget / SES CFs. Parent design
+ rationale + other families: `plans/Task_ Targeted-Allocation Counterfactuals.md`.

**What it produces.** For counties 17117 / 01031 / 36003, using the **already
calibrated** `log1p_rw2_l5_rmax5` params (fixed_60 waning, vacc 90, R_MAX 5 — no
recalibration), six counterfactuals that cross **vaccine allocation priority**
with **school/work closure**:

| CF | vaccine priority | school | work | one-liner |
|----|------------------|--------|------|-----------|
| 7  | uniform (factual) | factual | factual | **reference** — matches history, unchanged |
| 12 | elderly-first     | factual | factual | shield the old, NPIs as they were |
| 13 | working-age-first | factual | factual | protect 20–64, NPIs as they were |
| 14 | working-age-first | closed  | open   | protect workers, schools shut, work open |
| 15 | working-age-first | open    | open   | protect workers, everything open ("let it run, shield labour") |
| 16 | elderly-first     | open    | closed | shield the old, workplaces shut, schools open |
| 17 | elderly-first     | closed  | closed | shield the old + full closure |

"elderly-first" = 65A → 50t64 → 40t49 → 30t39 → 20t29 → U19.
"working-age-first" = {20t29,30t39,40t49,50t64} first (equal rank), then 65A,
then U19. Daily dose count is the factual `vaccines(t)` series in every CF.

Plus, per county: daily cases per CF, a **child / adult / elderly** breakdown of
cases (the age-mapping bug is fixed as part of this — §1.2), and grouped bar
charts.

---

## 0. Non-negotiables

- **No silent error handling.** No `try/except: pass`, no default-on-failure, no
  clamp-and-continue on bad data. Any missing/short/NaN file, HTTP non-200,
  empty candidate pool, a cumulative dose quota that exceeds the ever-eligible
  pool, a shape mismatch, an unknown CF type, or an unknown age code →
  `raise RuntimeError(...)` naming the file / county / step / expected-vs-actual,
  and exit non-zero. The run scripts use `set -euo pipefail` and tee the
  traceback to the batch log. (The `min(quota, n_susceptible)` line already in
  `apply_vaccines` is *physical* — keep it — but add a hard error if the
  cumulative daily quota over the whole run exceeds the eligible pool, which
  would mean the dose series or the population is wrong.)
- Do **not** touch: `main.py`, `LearnableParams`, the loss, the calibration loop,
  CF types 1–11, the committed baselines, `result_graphs/`, `vaccinated_test/`,
  `waning_tests/`, `results/*/LossCompare/*`.
- `config.yaml` is patched-then-restored per run via a `trap` (copy the pattern
  from `results/LossCompare_run/run_waning_variant.sh`).
- Env: `.venv_ds` (`/home/pkbiswas/.conda/envs/.venv_ds/bin/python`), A100,
  `SEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MPLBACKEND=Agg`.

---

## 1. Background the implementer needs

### 1.1 Age encoding (authoritative)

`scripts/process_counties.py::create_county_folder` writes
`populations/pop<FIPS>/age.pickle` (a pickled `pd.Series` of ints) with:

| age years | code | label (`populations/pop<FIPS>/mapping.json`) |
|-----------|------|--------|
| < 20  | **5** | U19 |
| 20–29 | **0** | 20t29 |
| 30–39 | **1** | 30t39 |
| 40–49 | **2** | 40t49 |
| 50–64 | **3** | 50t64 |
| 65+   | **4** | 65A |

`agents_ages` in `transition.forward` is exactly these codes (loaded via
`load_population_attribute`, values 0..5).

### 1.2 The mapping bug — fix the DIAGNOSTIC sites only

`transition.get_age_stage_proportions`, `transition.get_mean_agent_interactions`
and `covid_abm/substeps/utils.py::get_mean_agent_interactions` all currently do:

```python
child_agents   = (agents_ages < 1)      # -> code {0}  == 20t29  (WRONG)
adult_agents   = (1 <= agents_ages <= 4)# -> 30t39..65A          (WRONG: has elderly, missing 20s)
elderly_agents = (agents_ages > 4)      # -> code {5}  == U19     (WRONG)
```

Correct grouping for this model:

```python
CHILD_CODES   = (5,)            # U19
ADULT_CODES   = (0, 1, 2, 3)    # 20–64
ELDERLY_CODES = (4,)            # 65+
```

**Fix only `get_age_stage_proportions` + the new age-incidence counter** (both
pure diagnostic output). **Do NOT touch `get_mean_agent_interactions`** (either
copy): its per-agent interaction constant feeds `I_bar` in `_lam` and is baked
into every committed calibration — re-labelling it changes the factual dynamics
and invalidates the reused `log1p_rw2_l5_rmax5` params. Leave a comment there
saying so. `covid_abm/substeps/utils.py`'s copy populates
`environment/mean_interactions`, which `forward()` never actually reads, so it is
dead for dynamics — leave it too.

Every `*_age_proportions.csv` produced before this fix is void — say so in the
paper. Verify the fix: `age.pickle` code-5 (U19) count ≈ the scaled census
`AGE_0_9 + AGE_10_19`; code-4 (65A) count ≈ scaled census 65+.

### 1.3 SFSusceptibility is not age-graded (as wired) — interpretation caveat

`environment.SFSusceptibility` (shape 9) is indexed by the 6-value age code, so
65A (code 4) → 1.03, identical to everyone 40+. The 1.27/1.52 tail is dead code
from a 9-bracket model. So in this model the elderly are **not** more
susceptible per contact. Do **not** fix this here (it changes the factual →
needs recalibration; flag it, don't do it). Consequence: the elderly-first vs
working-age-first case contrast comes only from *contact structure* + *removing
susceptibles from circulation*, not from an age-susceptibility gradient — report
that plainly.

### 1.4 Calibration to reuse (per county)

```
results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt
```
26 weekly R + infection-rate scale + k, mean of last 20 epochs. Waning config
that calibration assumed: `fixed`, `RECOVERED_TO_SUSCEPTIBLE_TIME=60`,
`VACCINATED_TO_SUSCEPTIBLE_TIME=90`. `R_MAX` is irrelevant to a CF run (frozen
params bypass the `LearnableParams` box).

### 1.5 Why the calibration can be reused

CF 7 logic is byte-identical to the committed engine, and CFs 12–17 only differ
inside `if generating_counterfactual`. The factual case trajectory is unchanged,
so R(t) is still valid. (See the parent doc §1 for the general rule.)

---

## 2. Implementation

### Step 1 — fix the age-group masks (diagnostic sites only)  *(`transition.py`)*

Introduce module constants `CHILD_CODES / ADULT_CODES / ELDERLY_CODES` (§1.2) and
a helper `_age_group_masks(agents_ages) -> (child, adult, elderly)` (bool [N]
tensors; `RuntimeError` if `child | adult | elderly` is not all-true, listing the
offending codes). Use it in **`get_age_stage_proportions`** and the new
age-incidence counter (Step 4) only. **Leave both `get_mean_agent_interactions`
copies untouched** (§1.2) with a comment explaining the calibration dependency.

### Step 2 — age priority in `apply_vaccines`  *(`transition.py`)*

Thread `agents_ages` in (from `forward`, where it already exists). Extend the
signature: `apply_vaccines(self, current_stages, num_vaccines, tau=0.1, t=0,
agents_ages=None, priority=None)`.

```python
# priority is None (uniform, today's path) or a dict {age_code: rank}, higher rank = sooner
is_susceptible = self.soft_eq(current_stages, self.SUSCEPTIBLE_VAR, temperature=tau)
logits  = torch.log(is_susceptible + 1e-10).view(N, 1)
u       = torch.rand(logits.shape, generator=self._crn_gen(t, "vaccine"), ...)   # unchanged
gumbels = -torch.log(-torch.log(u + 1e-10) + 1e-10)
y = logits + gumbels
if priority is not None:
    if agents_ages is None:
        raise RuntimeError("apply_vaccines: priority set but agents_ages is None")
    susc_bool = (current_stages.view(-1) == self.SUSCEPTIBLE_VAR)
    rank = torch.zeros(N, device=device)
    for code, r in priority.items():
        rank[agents_ages.view(-1).long() == code] = float(r)
    BIG = 1.0e4                        # >> |logits| (~23) and gumbel (~O(1))
    y = (y.view(-1) + BIG * rank * susc_bool.float()).view(N, 1)   # offset ONLY susceptibles
```

`topk`, the straight-through mask, and the `S -> VACCINATED` move are unchanged.
Keep `num_vax_int = min(num_vax_int, int(is_susceptible.sum().item()))`. Track a
running `self._cum_quota` / `self._cum_eligible_ever`; on the final step, if
`self._cum_quota > self._cum_eligible_ever` → `RuntimeError` (dose series vs
population mismatch).

Priority dicts (module constant):
```python
AGE_PRIORITY = {
    "elderly_first":     {4: 5, 3: 4, 2: 3, 1: 2, 0: 1, 5: 0},
    "working_age_first": {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 0},
}
```

### Step 3 — register CF 12–17

`transition.forward`:

```python
# --- age-priority CF family (12-17): vaccine allocation x school/work NPI ---
CF_VACC_PRIORITY = {
    12: "elderly_first",     13: "working_age_first",
    14: "working_age_first", 15: "working_age_first",
    16: "elderly_first",     17: "elderly_first",
}
# logic_map additions  (school, work): "F" = factual, 0 = open, 1 = closed
#   12: ("F", "F")   13: ("F", "F")
#   14: (1,   0)      15: (0,   0)
#   16: (0,   1)      17: (1,   1)
```

After the existing `if cf_type == 11: num_vaccines = 0` block:

```python
vacc_priority = None
if generating_counterfactual and cf_type in CF_VACC_PRIORITY:
    vacc_priority = AGE_PRIORITY[CF_VACC_PRIORITY[cf_type]]
```

and pass `priority=vacc_priority, agents_ages=agents_ages` into the
`apply_vaccines` call (only reached when `with_vacc`).

`abm_nets.py`:
- `CF_POLICY_DESC` additions:
  ```
  12: "elderly-first vacc; school factual, work factual",
  13: "working-age-first vacc; school factual, work factual",
  14: "working-age-first vacc; school closed, work open",
  15: "working-age-first vacc; school open, work open",
  16: "elderly-first vacc; school open, work closed",
  17: "elderly-first vacc; school closed, work closed",
  ```
- add to **both** `cf_folder_map` dicts:
  ```
  12: "12_agepri_elderly_first_schF_workF",
  13: "13_agepri_workingage_first_schF_workF",
  14: "14_agepri_workingage_first_schC_workO",
  15: "15_agepri_workingage_first_schO_workO",
  16: "16_agepri_elderly_first_schO_workC",
  17: "17_agepri_elderly_first_schC_workC",
  ```
- default CF loop `range(1, 12)` → `range(1, 18)` (the `CF_TYPES` env override
  already lets the runbook pick a subset).
- analyzer `CF_NAMES` (Step 6).

### Step 4 — per-age-group daily new-infection counter  *(`transition.py`)*

In `__init__`: `self.age_incidence_history = []`; reset it in `forward` at
`t == 0` next to the other history resets.

In `forward`, right after `newly_exposed_today` gets its final value (after the
`just_vaccinated` masking, before it is unsqueezed):

```python
ne = newly_exposed_today.view(-1)
child_m, adult_m, elderly_m = self._age_group_masks(agents_ages)
self.age_incidence_history.append([
    t, float(ne[child_m].sum()), float(ne[adult_m].sum()), float(ne[elderly_m].sum()),
])
```

Add `save_age_incidence_to_disk(self, path, epoch_num)` mirroring
`save_proportions_to_disk2` (wide, one column set per iteration, outer-merge on
`t`). Columns: `t, child_new_<n>, adult_new_<n>, elderly_new_<n>`.

### Step 5 — collect the age-incidence file in the CF loop  *(`abm_nets.py::eval_net`)*

In the CF iteration loop where `all_age_proportions.append(...)` happens, also
`all_age_incidence.append(substep.age_incidence_history)` and call
`substep.save_age_incidence_to_disk(<path>, num)`. After the loop, average over
the 30 iterations exactly like `avg_age_proportions` and write
`counterfactual_data{cf_type}_age_incidence.csv` with columns
`t, child_new, adult_new, elderly_new` (means).

### Step 6 — `targeted_cf/` scripts

```
targeted_cf/
  README.md                     # one screen: what this validates, how to run
  run_age_priority_cf.sh        # ONE county: reuse calibration, run CF 7 + 12..17
  run_all_age_priority_cf.sh    # 3 counties + analysis
  analyze_age_priority_cf.py    # LossCompare-style plots + age-group case bars
```

`run_age_priority_cf.sh` (model on `results/LossCompare_run/run_waning_variant.sh`,
minus the calibration step):

```bash
set -euo pipefail
FIPS="${FIPS:?}"; TAG="${TAG:-age_priority}"; SEED="${SEED:-42}"
CAL="results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt"
[[ -f "$CAL" ]] || { echo "[error] missing calibration: $CAL"; exit 1; }
OUT="results/${FIPS}/TargetedCF/${TAG}"
BAK="$(mktemp -d)"; cp covid_abm/yamls/config.yaml "$BAK/"
trap 'cp "$BAK/config.yaml" covid_abm/yamls/config.yaml; rm -rf "$BAK"' EXIT
# match the waning config the reused calibration assumed
python vaccinated_test/set_vacc_config.py --mode fixed \
   --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90
mkdir -p "$OUT/calibration"; cp "$CAL" "$OUT/calibration/calibrated_params.txt"
rm -rf "$OUT/counterfactuals"; mkdir -p "$OUT/counterfactuals"
export SEED CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export GENERATING_COUNTERFACTUAL=true
export CF_PARAM_FILE="$PWD/$CAL"
export CF_OUTPUT_DIR="$PWD/$OUT/counterfactuals"
export CF_TYPES="${CF_TYPES:-7,12,13,14,15,16,17}"
export CF_ITERS="${CF_ITERS:-30}"
python main.py "$FIPS"
# write RUN_CONFIG.txt: TAG, FIPS, SEED, calibration source, CF_TYPES, CF_ITERS, git rev, timestamp
```

`run_all_age_priority_cf.sh`: loop `FIPSES="17117 01031 36003"`, tee each to
`results/TargetedCF_run/${TAG}/logs/run_<FIPS>.log`, then run the analyzer.

> Optional matched-NPI uniform baselines: add `1,2,3` to `CF_TYPES`. CF 1
> (uniform, all-open) is the uniform counterpart of CF 15; CF 3 (uniform,
> work-closed) of CF 16; CF 2 (uniform, school-closed/work-open) of CF 14.

### Step 7 — `analyze_age_priority_cf.py`

Copy `results/LossCompare_run/analyze_waning_variant.py`; set `CF_NAMES` to the
7 rows in the table above (folder names from Step 3); point `county_dir` at
`results/<FIPS>/TargetedCF/<TAG>/`.

**Every number carries a spread — no bare means.**
- daily-case line plots → mean **+ 10–90 percentile band** across iterations.
- age-group bar charts → bar = mean, **whisker = ± 1 SD** across iterations.
- Δ-vs-CF7 (bars + `HEADLINE.txt`) → **per-iteration PAIRED difference**
  `d_i = total(CFk, iter i) − total(CF7, iter i)` (CRN pairs iteration `i`
  across CF types, so this is ~5–15× tighter than differencing marginal means),
  reported as `mean ± SD [p10, p90]`.

Reproduce, in the same file layout: `cf_raw_mean_<ct>.csv`, `cf_mean_summary.csv`
(now `peak_mean/peak_sd/total_mean/total_sd/total_p10/total_p90`), `cf7_fit.png`,
`cf_all_mean.png` (mean + light 10–90 band per CF), `R_weekly.png`, and the
roll-ups `results/TargetedCF_run/<TAG>/{cf_mean_summary_ALL,cf7_fit_ALL}.csv`.

Also **overwrite** the per-CF `counterfactuals/counterfactual_results<ct>.png`
(and the `all_counterfactual_results/<folder>/graphs/<FIPS>_…` copy) — `execute()`
leaves a single-draw plot there; replace it with the n-iter mean + 10–90 band vs
actual. The per-county run script runs the analyzer for this reason unless
`SKIP_ANALYZE=1` (set by the all-county driver, which analyses once at the end).

Add on top:

1. **Age-group cases — per iteration.** Read
   `counterfactual_data<ct>_age_incidence_raw.csv` (wide: `t` + `child_new_<i>/
   adult_new_<i>/elderly_new_<i>` for each iteration). For each iteration sum the
   trailing eval window → per-iteration group totals. Fail loud if the iteration
   count ≠ `cf_raw_<ct>.csv` rows, if columns aren't `0..N` contiguous, or if
   there's a NaN. Sanity: `mean(Σ groups)` vs `cf_raw_mean_<ct>` should agree up
   to `k` (age counters are raw exposures, `cf_raw` is `× k`) — `RuntimeError`
   if the ratio is outside `[0.80, 1.03]`.
2. `age_breakdown_<FIPS>.csv`: rows = CF, cols =
   `{child,adult,elderly,total}_cases_{mean,sd}`,
   `d_{child,adult,elderly,total}_{mean,sd,p10,p90}` (paired vs CF7, absolute),
   `d_{child,adult,elderly,total}_pct_{mean,sd}` (paired vs CF7, percent:
   per-iteration `(CFk − CF7) / CF7 × 100`), `groupsum_vs_reported_ratio`.
3. **Bar charts** `age_bars_<FIPS>.png` (no titles): left panel = total new
   infections by group, y-axis "new infections", bar=mean whisker=±1 SD, one bar
   per CF; right panel = **paired** Δ-vs-CF7 as a **percent**, y-axis
   "new infection vs CF7 (%)", bar=mean whisker=±1 SD (per-iteration
   `(CFk − CF7) / CF7 × 100`). `age_bars_ALL.png` = the paired-Δ (absolute)
   panel faceted by county.
4. Roll-up `results/TargetedCF_run/<TAG>/age_breakdown_ALL.csv` +
   `HEADLINE.txt`: per county × CF × group, the paired `d_i` as
   `mean ± SD [p10, p90]` → **the headline: does elderly-first cut elderly
   infections, and at what cost to the other groups / the total.** Single seed
   (42); the file notes that a load-bearing number needs seeds 43/44 too.

---

## 3. Run

```bash
cd /home/pkbiswas/COVID-ABM/epi-diff-abm-dev

# smoke test first (2 iters, one county, all 7 CF types) — must exit 0 and produce every artifact
PYTHON=/home/pkbiswas/.conda/envs/.venv_ds/bin/python \
FIPS=01031 CF_ITERS=2 TAG=age_priority_smoke \
  bash targeted_cf/run_age_priority_cf.sh

# full run: 3 counties x CF {7,12,13,14,15,16,17} x 30 iters + analysis
PYTHON=/home/pkbiswas/.conda/envs/.venv_ds/bin/python \
FIPSES="17117 01031 36003" TAG=age_priority \
  bash targeted_cf/run_all_age_priority_cf.sh

# adding counties to an existing run — simulate the new ones, roll up ALL:
FIPSES="17117 36003" ANALYZE_FIPSES="01031 17117 36003" TAG=age_priority \
  bash targeted_cf/run_all_age_priority_cf.sh

# re-analyse only (no simulation), e.g. after an analyzer change:
FIPSES=01031,17117,36003 TAG=age_priority \
  /home/pkbiswas/.conda/envs/.venv_ds/bin/python targeted_cf/analyze_age_priority_cf.py
```

Runtime ≈ 7 CF types × 30 iters × ~30 s ≈ 105 min/county → ~5–6 h total on one
A100. Background it; watch `results/TargetedCF_run/age_priority/logs/`.

`run_all_age_priority_cf.sh` analyses the counties in `ANALYZE_FIPSES` (default =
`FIPSES`) — set it explicitly to every county with results on disk when adding
counties incrementally, or the roll-up drops the earlier ones.

---

## 4. Outputs

### Per county — `results/<FIPS>/TargetedCF/age_priority/`

| path | contents |
|------|----------|
| `RUN_CONFIG.txt` | TAG, FIPS, SEED, calibration source, CF_TYPES, CF_ITERS, git rev, timestamp |
| `calibration/calibrated_params.txt` | copy of the reused `log1p_rw2_l5_rmax5` params |
| `counterfactuals/cf_raw_<ct>.csv` | n_iter × 168 daily cases per CF (7, 12–17) |
| `counterfactuals/counterfactual_results<ct>.png` | n-iter **mean + 10–90 band** vs actual (analyzer overwrites execute()'s single draw); same for the `all_counterfactual_results/<CF>/graphs/` copy |
| `counterfactuals/counterfactual_data<ct>_age_incidence.csv` | child/adult/elderly new infections/day (n-iter mean) |
| `counterfactuals/counterfactual_data<ct>_age_incidence_raw.csv` | same, wide, **one column set per iteration** (the spread source) |
| `counterfactuals/counterfactual_data<ct>_age_proportions.csv` | S/E/I/R/D by group (now correctly labelled) |
| `cf_raw_mean_<ct>.csv`, `cf_mean_summary.csv` | case means + `peak_mean/peak_sd/total_mean/total_sd/total_p10/total_p90` |
| `cf7_fit.png`, `cf_all_mean.png`, `R_weekly.png` | line plots, mean + 10–90 band |
| `age_breakdown_<FIPS>.csv` | CF × group → `*_cases_{mean,sd}`, paired `d_*_{mean,sd,p10,p90}` (absolute) and `d_*_pct_{mean,sd}` (percent, per-iteration `(CFk − CF7)/CF7 × 100`) vs CF7 |
| `age_bars_<FIPS>.png` | no titles; left: totals by group, y "new infections" (bar=mean, ±1 SD); right: paired Δ-vs-CF7 as percent, y "new infection vs CF7 (%)" (bar=mean, ±1 SD) |

### Cross-county — `results/TargetedCF_run/age_priority/`

| path | contents |
|------|----------|
| `cf_mean_summary_ALL.csv`, `cf7_fit_ALL.csv` | per county × CF: peak/total mean+SD; CF7 fit metrics + `cf7_total_sd` |
| `age_breakdown_ALL.csv` | per county × CF × group: `*_cases_{mean,sd}`, paired `d_*_{mean,sd,p10,p90}` (absolute), `d_*_pct_{mean,sd}` (percent) |
| `age_bars_ALL.png` | paired Δ-vs-CF7 (absolute, ±1 SD), faceted by county |
| `HEADLINE.txt` | per county × CF × group: paired `d_i` as `mean ± SD [p10, p90]`; single seed 42 (note: repeat 43/44 for load-bearing numbers) |
| `logs/run_<FIPS>.log`, `logs/analyze.log` | full stdout incl. any traceback |

---

## 5. How to read it

1. **Factual sanity** — `cf7_fit.png` / `cf7_fit_ALL.csv` must match the existing
   `results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5/` CF7 fit **to within CRN
   noise** (same params, waning, seed; CF 7 logic is byte-identical). If it
   doesn't, something from Steps 1–5 leaked into the factual path — stop.
2. **Totals** — `cf_mean_summary.csv` (`total_mean ± total_sd`): CF 12 vs 13
   (same NPIs, different vaccine target) isolates the allocation effect; CF
   14/15 vs 16/17 shows how it changes as schools/work open or close. The
   *marginal* totals overlap heavily (SD ~25 % of the mean) — don't read the
   allocation effect off these.
3. **Age-group cases — the point.** Read the **paired** Δ in
   `age_breakdown_<FIPS>.csv` / `HEADLINE.txt` (`d_*_mean ± d_*_sd`), NOT the
   difference of marginal means. CRN pairing shrinks the SD ~5–15×, so a small
   allocation effect that is invisible in the marginal totals is clearly
   non-zero here. Under elderly-first (12/16/17): does `d_elderly` go negative,
   and at what `d_adult` / `d_child` / `d_total` cost? Working-age-first
   (13/14/15): the mirror.
4. **Bars** — `age_bars_<FIPS>.png`: right panel (paired Δ-vs-CF7 as a percent of
   the CF7 group total, ±1 SD) is the readable summary. A bar whose whisker
   clears 0 is a real effect at this seed.
   Watch whether untargeted groups go positive (displacement). The left panel's
   whiskers are the (much wider) marginal SD — expected, not a contradiction.

---

## 6. Gotchas

- **Age outputs are meaningless without Step 1.** Verify the fix with a one-off:
  print the code histogram per group for `pop01031` — the child group count
  should ≈ the U19 count from
  `data/state_data/AL/01031/agents_ages.csv` (`AGE_0_9 + AGE_10_19`), not the
  20s count.
- **CF 7 must stay byte-identical.** The `priority=None` default path through
  `apply_vaccines` must be the exact current code. Regression check: run CF 7
  alone, diff `cf_raw_7.csv` against the LossCompare `cf_raw_7.csv` for the same
  county — allow only CRN-scale differences (ideally identical).
- **`k` under-reporting.** `cf_raw_<ct>` daily cases are `× k`; the age-incidence
  counters are raw new exposures. Scale before comparing totals (the analyzer
  asserts consistency after scaling — if it fires, the scaling is wrong, don't
  loosen it).
- **Non-deterministic CUDA** (`Tasks/waning_variant_run.md` §6): the analyzer's
  per-iteration ± SD captures the Monte-Carlo spread at ONE seed. The GNN
  scatter has no deterministic CUDA kernel, so the *paired* Δ can still shift
  seed-to-seed. For a load-bearing CF12−CF7 / CF13−CF7 number, re-run with
  `SEED=43` and `SEED=44` (own TAGs) and report the across-seed spread of the
  paired Δ alongside the within-seed SD.
- **65+ is one bucket** — no within-65+ resolution.
- **Vaccine rollout starts ~day 86** of 182; only ~3 months of prioritisation is
  in-window. Longer horizon is out of scope (data ends 2021-04-25).
- **CF 14–17 change NPIs, so `apply_intervention_fast` fires the closed/open
  branch** — CRN keeps the closure fraction shared with any other CF at the same
  (iter, step, network), so 15's "all open" occ network is the same realisation
  as CF 1's. Good for comparability; note it.
- Restore-on-exit `trap` must fire even on `raise` paths — set it immediately
  after `mktemp`.

---

## 7. Report back

- files + functions changed;
- the CF 7 byte-identity check result;
- the age-mapping-fix verification histogram;
- `HEADLINE.txt` contents (per-group paired `d_i` mean ± SD [p10, p90]);
- any `RuntimeError` hit and the data that caused it.
