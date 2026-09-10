# Runbook / task: household-spillover secondary-protection analysis

**Status:** NOT STARTED. Parent design + rationale + other families:
`plans/Task_ Targeted-Allocation Counterfactuals.md` §2 (shared infrastructure)
and the companion idea log appended after the essentiality task.

**Scope — output-only analysis, no new CF types, no architecture change.**
For counties 17117 / 01031 / 36003, on the **committed `covid_output_causal`
networks** with the **reused `log1p_rw2_l5_rmax5` params**, measure the
intra-household secondary-protection signal that each existing CF already
generates but never reports: does having a vaccinated household member reduce a
susceptible cohabitant's probability of acquiring infection, and does the
allocation strategy (age-priority CF-A, essential-first CF-B) change the
magnitude or distribution of that protection?

**No new CF types are introduced.** Every CF in the existing `CF_TYPES` lists
for the age-priority (`12–17`) and essentiality (`18–24`) families is re-used
exactly. The only additions are:

- a per-iteration **household-vaccination-status tagging** step applied
  post-hoc to the simulation's own state trajectory;
- new per-CF output files (`counterfactual_data<ct>_hh_spillover_raw.csv`);
- a new analyzer `analyze_spillover.py` that reads those files alongside the
  existing `cf_raw_<ct>.csv` and produces spillover-specific plots and tables.

Calibration is **fully reused**: `log1p_rw2_l5_rmax5` for the age-priority
family; `log1p_rw2_l5_rmax5_essnet` (or reused, per `cf7_check.txt`) for the
essentiality family on the essnet networks. **No recalibration is triggered.**

**What it produces.** For each CF and each iteration, a daily time series of:

- new infections in **susceptible agents with ≥1 vaccinated household member**
  ("hh-protected susceptibles") vs. **susceptible agents with no vaccinated
  household member** ("hh-unprotected susceptibles");
- the **household secondary attack rate (hSAR)**: fraction of susceptible
  cohabitants of a vaccinated agent who become infected in a rolling 14-day
  window, compared to the same quantity in matched unvaccinated households.

Headline: **per-iteration paired ΔhSAR** (CFk − CF7), where CRN pairs
iteration `i` across CF types, reported `mean ± SD [p10, p90]`. Does
elderly-first vaccination (CF12) generate more household spillover protection
for non-elderly cohabitants than working-age-first (CF13)? Does essential-worker
vaccination (CF18) protect the non-essential and not-in-workforce cohabitants
who live with essential workers?

---

## 0. Non-negotiables

- **No silent error handling.** No `try/except: pass`, no default-on-failure,
  no clamp-and-continue. Any missing/short/NaN file, a `Household ID` mismatch,
  a household with zero members in the sim's agent tensor, a shape mismatch
  between `household_members` and `num_agents` → `raise RuntimeError(...)` naming
  the file / county / step / expected-vs-actual, exit non-zero. `set -euo
  pipefail`, tee the traceback.
- **Report every effect with a spread, and difference CFs pairwise** — same
  as CF-A and CF-B: line plots = n-iter mean + 10–90 band; bar charts = bar
  mean / whisker ± 1 SD; every "CFk vs CF7" number is the **per-iteration
  PAIRED** Δ (`d_i = metric(CFk, i) − metric(CF7, i)`; CRN pairs iteration
  `i`), reported `mean ± SD [p10, p90]`.
- Do **not** touch: `main.py`, `LearnableParams`, the loss, the calibration
  loop, CF types 1–24, the committed baselines, `result_graphs/`,
  `vaccinated_test/`, `waning_tests/`, `results/*/LossCompare/*`,
  `targeted_cf/phase2/`, `covid_output_essnet`.
- `config.yaml` patched-then-restored per run via a `trap` (copy the pattern
  from `targeted_cf/run_age_priority_cf.sh`).
- Env: `.venv_ds` (`/home/pkbiswas/.conda/envs/.venv_ds/bin/python`), A100,
  `SEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MPLBACKEND=Agg`.
- **`household_members` tensor is loaded once in `__init__` of the spillover
  hook**, from the committed `{fips}_population.csv` (`Household ID` column),
  and is **read-only** thereafter. The sim's own state tensors are never
  written to by the spillover hook.
- The spillover hook fires **only when `SPILLOVER_ANALYSIS=true`** is exported
  to the environment before the CF run. It is invisible to calibration, CF 1–11,
  and any run that does not set this flag. This guarantees zero regression risk
  to the base engine.

---

## 1. Background the implementer needs

### 1.1 Household network and agent index

`populations/pop<FIPS>/` contains:
- `age.pickle` — per-agent age code (already used by the sim).
- The household structure is in
  `data/population_data/<ST>_population_data/<FIPS>_population.csv`,
  column `Household ID`. **Row order == agent index** (same as `age.pickle`,
  the mobility nets, and `agent_essential`).
- `HOUSEHOLD_NETWORK.pkl` is a `networkx.Graph` whose edges are within-household
  contacts. Its node set is `{0 … num_agents-1}`. This is the same file the sim
  loads; the spillover hook reads it independently at init time to build the
  `household_members` lookup.

The spillover analysis does **not** depend on the household network edges
(transmission paths through that network are already handled by the sim). It
uses only the **household membership** relation: which agents share the same
`Household ID`. The transmission model already simulates household spread
correctly; this analysis measures the *downstream protective signal* — whether
a susceptible agent's cohabitants being vaccinated correlates with a lower
observed infection rate for that susceptible agent across the CF run.

### 1.2 What "household spillover protection" means in this model

When agent `v` (susceptible) lives in a household with agent `u` who is
vaccinated, two things are happening in the sim:

1. `u`'s own susceptibility is suppressed (the `VACCINATED` state has a
   near-zero `SFSusceptibility` weight), so `u` rarely gets infected.
2. Even if `u` gets infected, `u`'s `I_bar` (mean interactions, which
   mediates community transmission from `u`) is unchanged — the model does not
   distinguish between `u` being vaccinated and `u` being a
   carrier. **Household transmission still passes through `u`'s household edges
   to `v`.**

The "spillover protection" in this model is therefore **indirect** and operates
through two channels:
- **Contact shielding:** `u` being vaccinated means `u` is less likely to
  import infection from the community and carry it home to `v`.
- **Dose displacement:** a dose used on `u` is not available for `v`; under
  some allocation strategies (e.g. elderly-first) the vaccinated `u` may not
  be the household member most likely to expose `v`.

This is an *epidemiological signal*, not a mechanistic network path — it is
estimated by comparing the infection rate of `v`-type agents across CF pairs
that differ only in *who* received vaccines. It must be reported as a
**correlational output** of the simulation, not a causal mechanism, because
the same CRN-paired Δ logic used throughout CF-A / CF-B applies here.

State this clearly in the paper: "Household spillover protection is estimated
as the difference in infection rates for susceptible agents in vaccinated vs.
unvaccinated households, across CF pairs with identical CRN draws. It captures
the indirect epidemiological signal from vaccination, not a mechanistic
within-household contact path."

### 1.3 Household membership tensor

```python
# Built once in SpilloverHook.__init__
df = pd.read_csv(pop_csv)                          # committed population csv
hh_id = df["Household ID"].to_numpy(dtype=np.int64)   # shape [N]
# household_members[i] = 1D int64 tensor of agent indices in the same household as i
# (including i itself — subtract i in SAR numerator)
from collections import defaultdict
hh_groups = defaultdict(list)
for agent_idx, hid in enumerate(hh_id):
    hh_groups[hid].append(agent_idx)
# store as a list-of-tensors (ragged; household sizes vary 1–7)
household_members = [torch.tensor(v, dtype=torch.long, device=device)
                     for v in hh_groups.values()]
# also store per-agent lookup: agent -> household tensor index
agent_to_hh = torch.zeros(N, dtype=torch.long, device=device)
for hh_idx, members in enumerate(household_members):
    agent_to_hh[members] = hh_idx
```

Assertions at build time (fail loud):
- `len(df) == num_agents` — else raise naming the csv and both counts.
- `hh_id` has no NaN.
- Every agent index `0..N-1` appears exactly once in some `hh_groups` value.
- No household has size > 10 (sanity; `{fips}_population.csv` `Household Size`
  column is the reference; raise if any group exceeds it by more than 1).

### 1.4 Definition of hSAR

**Household secondary attack rate (hSAR):** in a rolling 14-day window ending
at day `t`, the fraction of initially-susceptible cohabitants of vaccinated
agents who became infected:

```
hSAR_vacc(t) = Σ_{u: vaccinated at any point in [t-13, t]}
                 Σ_{v ∈ household(u), v ≠ u, v susceptible at t-13}
                   1[v infected in (t-13, t]]
               ──────────────────────────────────────────────────────
               Σ_{u: vaccinated at any point in [t-13, t]}
                 Σ_{v ∈ household(u), v ≠ u, v susceptible at t-13}
                   1
```

```
hSAR_unvacc(t) = same, but for households where NO member was vaccinated
                 in [t-13, t]
```

Both are computed per iteration per CF and written to
`counterfactual_data<ct>_hh_spillover_raw.csv`. The primary headline is
`ΔhSAR(t) = hSAR_unvacc(t) − hSAR_vacc(t)` (positive = protection signal),
and its **per-iteration paired CRN-Δ** vs CF7.

**Edge cases — fail loud:**
- If the vaccinated-household denominator is 0 for any day and county, write
  `NaN` for that day (do not impute). The analyzer skips NaN days in the
  time-series plot but raises if more than 30 days are NaN for any CF × county.
- If `hSAR_vacc > 1.0` or `hSAR_unvacc > 1.0` for any day, raise.

### 1.5 Breakdown groups for the spillover analysis

Three analysis cuts, reported separately:

| cut | `v` agent type | question |
|-----|----------------|----------|
| **age** | child / adult / elderly (CF-A codes: `{5}` / `{0,1,2,3}` / `{4}`) | do children and elderly cohabitants benefit most from household spillover? |
| **essential status** | essential / non-essential / not-in-workforce (CF-B codes: `1`/`2`/`0` from `agent_essential`) | do not-in-workforce agents living with essential workers gain protection from essential-first vaccination? |
| **household size** | 1-person / 2-person / 3+-person | is the spillover signal size-dependent? |

Age breakdown is always computed (uses committed `age.pickle`).
Essential-status breakdown is computed **only for essentiality-family CF runs**
(`SPILLOVER_FAMILY=essentiality` flag, §2.4). It requires `agent_essential`
which is loaded lazily via `_require_agent_essential` — the same call used
by `transition.py` CF-B paths; the spillover hook calls it if and only if
the essential-status cut is requested.
Household-size breakdown uses the `household_members` list (sizes already
computed during §1.3 build).

### 1.6 Which CFs to run the spillover analysis over

| family | CFs | networks | calibration |
|--------|-----|----------|-------------|
| age-priority | 7, 12, 13, 14, 15, 16, 17 | `covid_output_causal` | `log1p_rw2_l5_rmax5` |
| essentiality | 7, 18, 19, 20, 21, 22, 23 | `covid_output_essnet` | `…_essnet` or reused |

CF 7 (uniform vaccine, factual NPI) is the reference in both families. The
**paired Δ** uses the same CRN iteration index `i` across CF types, which is
already guaranteed by the existing CRN discipline in `_crn_gen`.

CF 24 (all occ closed reference) is **not** included in the spillover analysis
— it is an NPI reference, not a vaccine allocation variant, and the spillover
signal is driven by vaccine placement, not closure policy.

---

## 2. Implementation

### Step 0 — `SpilloverHook` class  *(`targeted_cf/spillover_hook.py`, new file)*

A self-contained class that attaches to a running simulation instance. It reads
the sim's `state` dict each timestep (read-only), computes the hSAR
incrementally, and writes per-iteration output files. It is **not** imported by
`transition.py` or `abm_nets.py` in any existing code path — it is instantiated
only by the spillover run scripts.

```
targeted_cf/
  spillover_hook.py         # SpilloverHook class (new)
  run_spillover_cf.sh       # ONE county, one family
  run_all_spillover_cf.sh   # all counties, both families
  analyze_spillover.py      # new analyzer
```

`SpilloverHook.__init__(sim, config, device, output_dir, family)`:
- reads `{fips}_population.csv` (committed path; fail loud per §1.3).
- builds `household_members`, `agent_to_hh`, `hh_sizes` (int tensor `[num_hh]`).
- loads `age.pickle` from `populations/pop<FIPS>/age.pickle`; builds `age_mask`
  dict `{child, adult, elderly}` using `CHILD_CODES={5}`, `ADULT_CODES={0,1,2,3}`,
  `ELDERLY_CODES={4}` (the corrected codes from CF-A, §3 caveat 1 of the parent
  plan).
- if `family == "essentiality"`: calls `sim._require_agent_essential()` to get
  `agent_essential` tensor; else `self.agent_essential = None`.
- allocates rolling buffers: `vacc_status` (bool `[N]`), `susc_at_window_start`
  (bool `[N]`), `infected_in_window` (bool `[N]`).
- initialises `self.records = []` (list of per-day dicts, flushed per iteration).

`SpilloverHook.step(t, state)`:
- reads `current_stages = state['agents']['citizens']['current_disease_stages']`
  (shape `[N, 1]`, same tensor the sim uses); shape-check vs `N` → raise if
  mismatched.
- updates `vacc_status[i] = True` if agent `i` ever entered `VACCINATED` (cumulative).
- for the rolling 14-day window: on each day `t`, record the day-`t` hSAR using
  the window `[t-13, t]` (skip days 0–12, write NaN). See §1.4 for numerator /
  denominator logic.
- appends a dict to `self.records`:
  ```python
  {"t": t,
   "hSAR_vacc": ..., "hSAR_unvacc": ...,          # overall
   "hSAR_vacc_child": ..., "hSAR_unvacc_child": ...,
   "hSAR_vacc_adult": ..., "hSAR_unvacc_adult": ...,
   "hSAR_vacc_elderly": ..., "hSAR_unvacc_elderly": ...,
   "hSAR_vacc_hh1": ..., "hSAR_unvacc_hh1": ...,  # size-1 households
   "hSAR_vacc_hh2": ..., "hSAR_unvacc_hh2": ...,
   "hSAR_vacc_hh3p": ..., "hSAR_unvacc_hh3p": ...,
   # if family == "essentiality":
   "hSAR_vacc_ess": ..., "hSAR_unvacc_ess": ...,
   "hSAR_vacc_noness": ..., "hSAR_unvacc_noness": ...,
   "hSAR_vacc_notinwf": ..., "hSAR_unvacc_notinwf": ...,
  }
  ```

`SpilloverHook.flush(output_dir, epoch_num, cf_type)`:
- converts `self.records` to a `pd.DataFrame`.
- writes `counterfactual_data<cf_type>_hh_spillover_<epoch_num>.csv` to
  `output_dir/counterfactuals/` (one file per iteration per CF; wide format,
  one row per day).
- clears `self.records` and resets buffers.

### Step 1 — hook the sim's evaluation loop  *(`abm_nets.py`)*

**Do not modify the CF loop itself.** Instead, in
`run_spillover_cf.sh` (§2.4), export `SPILLOVER_ANALYSIS=true`. A thin
wrapper script `targeted_cf/spillover_hook.py::run_with_spillover(fips,
cf_types, n_iters, config_path, output_dir, family)` calls the standard
`abm_nets.eval_net` but patches in a per-timestep callback that
instantiates `SpilloverHook` once per iter and calls
`hook.step(t, state)` after each `forward()`:

```python
# In run_with_spillover (NOT in abm_nets.py itself):
for epoch in range(n_iters):
    hook = SpilloverHook(sim, config, device, output_dir, family)
    for t in range(T):
        sim.forward(t)
        hook.step(t, sim.state)
    hook.flush(output_dir, epoch, cf_type)
```

`sim.state` is the existing mutable state dict that `transition.forward`
already updates in-place. The hook reads it but **never writes to it**.

The `run_with_spillover` function is called directly from
`run_spillover_cf.sh` via a small Python entry-point
`targeted_cf/spillover_hook.py::main()` (using the same `config.yaml`
patching and `CF_PARAM_FILE` / `NETWORKS_DIR` env vars as the existing run
scripts).

**The existing `abm_nets.eval_net` is not modified.** Zero regression risk.

### Step 2 — aggregate per-iteration files  *(`targeted_cf/spillover_hook.py`)*

After all iterations of a single CF type, a helper
`aggregate_spillover_iters(output_dir, cf_type, n_iters)` reads the
`n_iters` per-iteration csv files and writes:

- `counterfactual_data<ct>_hh_spillover.csv` — mean per day (columns:
  `t`, all `hSAR_*` columns, mean across iters).
- `counterfactual_data<ct>_hh_spillover_raw.csv` — stacked (wide): `t` +
  one column per metric per iter (`hSAR_vacc_0 … hSAR_vacc_<n-1>`,
  `hSAR_unvacc_0 … hSAR_unvacc_<n-1>`, etc.) — same layout as
  `counterfactual_data<ct>_age_incidence_raw.csv` from CF-A, for
  consistency with the existing analyzer pattern.

Fail loud if any iter file is missing or has fewer than 182 rows.

### Step 3 — cross-CF paired-Δ table  *(`analyze_spillover.py`)*

`compute_spillover_summary(output_dir, cf_types, ref_cf=7)` reads all
`*_hh_spillover_raw.csv` files for the given CF types, then for each
`CFk ≠ ref_cf`:

```python
# per-iteration total (integrate over days 14..181 to skip NaN-window)
total_vacc_k[i]   = hSAR_vacc_k[:, i].nanmean()    # mean over non-NaN days
total_unvacc_k[i] = hSAR_unvacc_k[:, i].nanmean()
total_vacc_7[i]   = hSAR_vacc_7[:, i].nanmean()    # CF7 reference
total_unvacc_7[i] = hSAR_unvacc_7[:, i].nanmean()

d_vacc_i   = total_vacc_k   - total_vacc_7          # paired Δ (same iter index i)
d_unvacc_i = total_unvacc_k - total_unvacc_7
```

Writes `spillover_summary_<FIPS>.csv`:

| column | description |
|--------|-------------|
| `cf` | CF type |
| `hSAR_vacc_mean / _sd / _p10 / _p90` | hSAR for vaccinated-hh susceptibles, across iters |
| `hSAR_unvacc_mean / _sd / _p10 / _p90` | hSAR for unvaccinated-hh susceptibles |
| `dSAR_vacc_mean / _sd / _p10 / _p90` | paired Δ vs CF7: vaccinated-hh susceptibles |
| `dSAR_unvacc_mean / _sd / _p10 / _p90` | paired Δ vs CF7: unvaccinated-hh susceptibles |
| `protection_gap_mean / _sd` | `hSAR_unvacc − hSAR_vacc` (mean over iters) — the spillover signal |
| `d_protection_gap_mean / _sd` | paired Δ of protection_gap vs CF7 |

Repeat for each age-group, essentiality-group, and household-size sub-column
set. Fail loud if the paired `d_i` array has any NaN (upstream NaN propagation
would mean a whole-day window was missing).

### Step 4 — register outputs in output layout

No new CF types, no new calibration dirs. Spillover outputs land under the
**same per-county `TargetedCF/<TAG>/` directory** as the corresponding family:

```
results/<FIPS>/TargetedCF/age_priority/
  counterfactuals/
    counterfactual_data<ct>_hh_spillover_raw.csv   # NEW
    counterfactual_data<ct>_hh_spillover.csv        # NEW
  spillover_summary_<FIPS>.csv                      # NEW
  spillover_hSAR_<FIPS>.png                         # NEW (§3.2)
  spillover_bars_<FIPS>.png                         # NEW (§3.3)
  spillover_age_bars_<FIPS>.png                     # NEW (§3.4)

results/<FIPS>/TargetedCF/essentiality/
  counterfactuals/
    counterfactual_data<ct>_hh_spillover_raw.csv   # NEW
    counterfactual_data<ct>_hh_spillover.csv        # NEW
  spillover_summary_<FIPS>.csv                      # NEW
  spillover_hSAR_<FIPS>.png                         # NEW
  spillover_bars_<FIPS>.png                         # NEW
  spillover_ess_bars_<FIPS>.png                     # NEW (§3.5, essentiality family only)
```

Cross-county roll-ups under `results/TargetedCF_run/spillover/`:

```
results/TargetedCF_run/spillover/
  spillover_summary_ALL.csv
  spillover_bars_ALL.png
  spillover_age_bars_ALL.png
  spillover_ess_bars_ALL.png     # essentiality family only
  HEADLINE.txt
  logs/
```

All existing artifacts (`cf_raw_<ct>.csv`, `cf7_fit.png`, `cf_all_mean.png`,
`R_weekly.png`, `essential_bars_<FIPS>.png`, etc.) are **untouched** — the
spillover scripts never call the existing analyzers, and the existing analyzers
do not read spillover files.

### Step 5 — `targeted_cf/` scripts

```
targeted_cf/
  spillover_hook.py          # SpilloverHook + run_with_spillover + aggregate + main()
  run_spillover_cf.sh        # ONE county, ONE family
  run_all_spillover_cf.sh    # all counties, both families + cross-county analysis
  analyze_spillover.py       # plots + tables (§3)
```

**`run_spillover_cf.sh`** — copy `run_age_priority_cf.sh`, changing:
- `TAG` default `age_priority` (reuse the existing directory); `FAMILY`
  default `age_priority`; `CF_TYPES` default `7,12,13,14,15,16,17`.
- `OUT=results/${FIPS}/TargetedCF/${TAG}` — **same directory** as the
  corresponding CF-A or CF-B run; spillover csvs land alongside existing
  artifacts.
- `export SPILLOVER_ANALYSIS=true`; `export SPILLOVER_FAMILY=${FAMILY}`.
- `NETWORKS_DIR`: if `FAMILY=age_priority` → `covid_output_causal`;
  if `FAMILY=essentiality` → `covid_output_essnet`.
- `CAL`: if `FAMILY=age_priority` → committed `log1p_rw2_l5_rmax5` params;
  if `FAMILY=essentiality` → `results/${FIPS}/TargetedCF/essentiality/
  calibration/calibrated_params.txt` (falls back to `log1p_rw2_l5_rmax5`
  if the recalibrated file is absent, with a printed warning).
- Entry point: `$PY targeted_cf/spillover_hook.py \
    --fips $FIPS --cf-types $CF_TYPES --n-iters $CF_ITERS \
    --output-dir $OUT --family $FAMILY` (handles config patch + trap restore
  internally via the same pattern as the age-priority script).
- `SKIP_ANALYZE` variable; call `analyze_spillover.py` unless set.
- assert that the CF-A (or CF-B) `cf_raw_7.csv` already exists in `$OUT` →
  else raise with the message "run `run_age_priority_cf.sh` (or
  `run_essentiality_cf.sh`) first; spillover analysis reuses the existing CF
  run's network and calibration". The spillover hook re-runs the sim from
  scratch (same params, seed, networks) but **must match the existing CF7
  case total to CRN noise** — verified at the end of the script (§5.1).

**`run_all_spillover_cf.sh`** — for each FIPS:
1. Run `run_spillover_cf.sh FAMILY=age_priority` (`SKIP_ANALYZE=1`).
2. If `cf7_check.txt` exists for this FIPS (essentiality has been run):
   run `run_spillover_cf.sh FAMILY=essentiality` (`SKIP_ANALYZE=1`).
3. After all counties: one `analyze_spillover.py` call over `ANALYZE_FIPSES`
   for each family, then the cross-county roll-up.

Update `targeted_cf/README.md`.

---

## 3. Plots and tables (analyzer)

**All plots follow the same style as `analyze_age_priority_cf.py` and
`analyze_essentiality_cf.py`:** `MPLBACKEND=Agg`, no titles (captions in
the paper), consistent colour palette across CF types, `figsize=(10, 4)` for
time-series and `(10, 5)` for bar panels, `dpi=150`, `tight_layout()`,
saved as `.png`. The line-series plots use the same mean + 10–90 percentile
band style as `cf_all_mean.png`.

### 3.1 Regression check — CF7 spillover vs existing CF7 case total

Before producing any plot, verify:

```python
new_cf7_total  = spillover_raw_cf7["cf_total_i"].mean()   # summed cases from hook run
old_cf7_total  = pd.read_csv(f"{out}/cf_raw_7.csv").sum(axis=1).mean()
ratio = new_cf7_total / old_cf7_total
assert 0.97 <= ratio <= 1.03, (
    f"CF7 case total mismatch: hook run {new_cf7_total:.0f} vs "
    f"existing run {old_cf7_total:.0f} (ratio {ratio:.3f}). "
    f"Check SEED, NETWORKS_DIR, and CAL match."
)
```

If the assert fires, stop and report — a mismatch means the spillover hook
re-ran the sim under different params or networks than the committed CF run,
breaking the CRN pairing.

### 3.2 `spillover_hSAR_<FIPS>.png` — hSAR time series

Two-panel figure (left = vaccinated-hh susceptibles; right = unvaccinated-hh
susceptibles). Each panel:
- x-axis: simulation day (0–181), with a vertical dashed line at ~day 86
  (vaccine rollout start, read from the first nonzero day of
  `vaccines` in `intervention.csv`).
- y-axis: `hSAR` (14-day rolling, dimensionless 0–1), labelled
  `"14-day household SAR"`.
- one line per CF type; colour and linestyle from the shared palette
  (same as `cf_all_mean.png`); legend outside the right panel.
- shaded band: 10–90 percentile across iterations (same as the CF-A /
  CF-B mean + band style).
- Mean ± band only — **no individual iteration lines**.

This is the time-resolved version of the spillover signal; it shows when
during the rollout the hSAR divergence between vaccinated and unvaccinated
households peaks.

### 3.3 `spillover_bars_<FIPS>.png` — paired Δ summary bar chart

Two-panel bar chart, **same layout as `essential_bars_<FIPS>.png`**:

- **Left panel** — absolute hSAR by CF type, two bar groups side by side:
  `hSAR_vacc` (mean over days 14–181 and iters) and `hSAR_unvacc`.
  x-axis: CF label. y-axis: `"mean 14-day household SAR"`.
  Bar = mean; whisker = ± 1 SD across iterations.
- **Right panel** — paired ΔhSAR vs CF7, one bar per CF type
  (one bar = `d_protection_gap_mean = Δ(hSAR_unvacc − hSAR_vacc)` vs CF7).
  y-axis: `"Δ protection gap vs CF7 (percentage points)"`.
  Bar = mean; whisker = ± 1 SD; horizontal zero line.
  A bar whose whisker clears zero is a real shift in the spillover protection
  gap relative to uniform allocation.

Colour: CF7 = grey (reference); other CFs use the same colour map as
`cf_all_mean.png`. No titles.

### 3.4 `spillover_age_bars_<FIPS>.png` — age-group breakdown

Three-group bar chart (child / adult / elderly), **same layout as
`age_bars_<FIPS>.png` from CF-A**:

- One sub-figure per CF type (or, if many CFs, one figure with grouped bars
  per age group with one bar per CF). Use the same `figsize` and layout as
  `age_bars_<FIPS>.png`.
- y-axis: `"Δ hSAR vs CF7 (pp)"` (percentage points, per-iteration paired).
- Separately report vaccinated-hh-susceptibles and unvaccinated-hh-susceptibles
  sub-bars within each age group (two shades of the same hue, consistent with
  the CF-A two-category colour scheme).
- Bar = mean; whisker = ± 1 SD.

**Key question this plot answers:** does elderly-first vaccination (CF12)
generate larger spillover protection for *child* susceptible cohabitants than
working-age-first (CF13)? The age-group bars make this directly visible.

### 3.5 `spillover_ess_bars_<FIPS>.png` — essentiality-group breakdown  *(essentiality family only)*

Same layout as `essential_bars_<FIPS>.png` right panel, but the metric is
`Δ hSAR` rather than `Δ new infections`:

- Three bar groups: essential / non-essential / not-in-workforce.
- Each group has bars for CF18, CF19, CF20, CF21, CF22, CF23 vs CF7.
- y-axis: `"Δ hSAR vs CF7 (pp)"`.
- Bar = mean; whisker = ± 1 SD; horizontal zero line.

**Key question:** does essential-first vaccination (CF18) generate a detectable
increase in spillover protection for the *not-in-workforce* agents (children,
elderly) who live with essential workers — at the cost of reduced spillover
protection for non-essential workers' households?

### 3.6 `HEADLINE.txt`

Per county and in the cross-county roll-up:

```
=== Household Spillover Analysis — <FIPS> ===
Family: <age_priority | essentiality>
Calibration: <log1p_rw2_l5_rmax5 | log1p_rw2_l5_rmax5_essnet>
CF7 regression check: ratio = <ratio> (PASS / FAIL)

Overall hSAR (days 14–181, mean ± SD across iters):
  CF7   vacc-hh: <X.XXX ± X.XXX>   unvacc-hh: <X.XXX ± X.XXX>
  CF12  vacc-hh: <...>               unvacc-hh: <...>
  ...

Protection gap (unvacc − vacc hSAR), paired Δ vs CF7 [mean ± SD, p10, p90]:
  CF12 vs CF7: <mean ± SD [p10, p90]>
  CF13 vs CF7: <...>
  ...

Age breakdown — paired Δ protection gap vs CF7:
  child:   CF12 <mean ± SD>   CF13 <mean ± SD>
  adult:   CF12 <...>         CF13 <...>
  elderly: CF12 <...>         CF13 <...>

[Essentiality family only]
Essential-status breakdown — paired Δ protection gap vs CF7:
  essential:    CF18 <mean ± SD>   CF19 <...>
  non-essential: CF18 <...>        CF19 <...>
  not-in-workforce: CF18 <...>     CF19 <...>

Cross-family note: spillover results for age-priority use covid_output_causal +
log1p_rw2_l5_rmax5; essentiality results use covid_output_essnet +
log1p_rw2_l5_rmax5_essnet. These are not directly comparable (different networks,
possibly different calibrations). See plans/Task_ Targeted-Allocation
Counterfactuals.md §7.4.
```

---

## 4. Run

```bash
cd /home/pkbiswas/COVID-ABM/epi-diff-abm-dev
PY=/home/pkbiswas/.conda/envs/.venv_ds/bin/python

# --- prerequisite check: age-priority CF must already be done
ls results/17117/TargetedCF/age_priority/cf_raw_7.csv   # must exist

# --- smoke test (2 iters, one county, age-priority family)
PYTHON=$PY FIPS=01031 CF_ITERS=2 FAMILY=age_priority \
  bash targeted_cf/run_spillover_cf.sh

# --- full run: all counties, both families, + cross-county analysis
PYTHON=$PY FIPSES="17117 01031 36003" \
  bash targeted_cf/run_all_spillover_cf.sh

# --- re-run essentiality family only (e.g. after FOOD sensitivity re-run)
PYTHON=$PY FIPS=17117 FAMILY=essentiality CF_ITERS=30 TAG=essentiality \
  bash targeted_cf/run_spillover_cf.sh

# --- re-analyse without re-running sim (SKIP_SIM=1 if implemented):
PYTHON=$PY FIPSES="17117 01031 36003" \
  $PY targeted_cf/analyze_spillover.py \
    --fipses 17117 01031 36003 \
    --families age_priority essentiality \
    --output-dir results/TargetedCF_run/spillover/
```

Runtime: the spillover hook re-runs the sim from scratch with the same params
and seed, so runtime ≈ that of the original CF run — **8 CF types × 30 iters
× ~30 s ≈ 2 h/county × 3 counties ≈ 6 h** for each family, background it.
The hook adds negligible overhead per step (tensor slices + accumulation,
no gradient).

---

## 5. Consistency checks

### 5.1 CF7 case total regression (mandatory before any plot)

Described in §3.1. Ratio must be in `[0.97, 1.03]`. If it fails: check that
`SEED`, `NETWORKS_DIR`, `CF_PARAM_FILE`, and `set_vacc_config.py` invocation
are identical between the original CF run and the spillover run. A mismatch
here means the CRN pairing is broken and all paired Δ results are invalid.

### 5.2 hSAR sanity bounds

For each CF × county:
- `hSAR_vacc` and `hSAR_unvacc` must be in `[0.0, 1.0]` for all non-NaN days.
- `hSAR_vacc(t) ≤ hSAR_unvacc(t)` is expected (protection signal) but **not
  enforced** — the model may produce `hSAR_vacc > hSAR_unvacc` on some days
  (e.g., early in rollout when very few agents are vaccinated and the
  denominator is small). Report but do not raise.
- NaN count ≤ 30 days per CF × county → else raise naming the CF and county.

### 5.3 Paired-Δ consistency with `cf_mean_summary.csv`

For each CF: the mean total case count in the spillover run (summed from the
hook's own `vacc_status` bookkeeping — total newly infected per iter) must
match the existing `cf_mean_summary.csv` `total_mean` within 3 %. If it
doesn't, the spillover hook's state reading is off (possibly reading the wrong
`current_disease_stages` tensor). Fail loud.

### 5.4 Group sum check

`hSAR_child_denom + hSAR_adult_denom + hSAR_elderly_denom` must equal
`hSAR_overall_denom` for every day and iter (denominators = susceptible
cohabitant counts per group). Fail loud on any day-iter combination where this
does not hold within floating-point tolerance (`rtol=1e-4`).

---

## 6. Gotchas

- **The spillover hook re-runs the sim.** This is a design choice to avoid
  modifying `abm_nets.eval_net`. The re-run uses identical params, seed,
  networks, and waning config, so all CF draws are CRN-paired with the
  original run. The mandatory CF7 regression check (§5.1) guards against
  any accidental divergence. If the check fails, the most common causes are:
  (a) a different `set_vacc_config.py` invocation (check fixed/waning mode),
  (b) a different `NETWORKS_DIR`, (c) a stale `config.yaml` left by a failed
  prior run (the `trap` restore prevents this under normal operation).
- **`household_members` ragged tensor**: PyTorch does not have a native ragged
  tensor type. Store as a Python list of 1D tensors. The hSAR computation
  loops over households — this is acceptable because it runs post-step (not
  inside the GPU forward pass) and `num_households ≈ 20 k` for these counties.
  If profiling shows it is slow, vectorise with `torch.nn.utils.rnn.pad_sequence`
  and a mask.
- **Size-1 households.** An agent living alone has no cohabitants; their
  household never contributes to either the vaccinated-hh or unvaccinated-hh
  denominator. They are silently excluded from all hSAR computations (correct).
  The household-size breakdown `hSAR_hh1` columns will always be NaN →
  documented in `HEADLINE.txt`.
- **Vaccine rollout starts ~day 86.** Before that day no agent is in
  `VACCINATED`, so `hSAR_vacc` denominator is 0 and hSAR_vacc is NaN for
  days 0–99 (86 + the 14-day rolling window). The time-series plot handles
  this via the NaN guard; the summary table integrates only days 14–181
  (the window skip) and the vaccinated-hh hSAR is effectively a
  post-rollout signal.
- **`EDUCATION` exempt from non-essential occ closure (CF-B), but not
  from the spillover analysis.** Education-sector agents can be in either
  the essential or not-in-workforce groups of the essential-status breakdown.
  No special casing needed — the `agent_essential` tensor already assigns
  them code `1`.
- **Cross-family comparability.** The age-priority spillover runs on
  `covid_output_causal` + `log1p_rw2_l5_rmax5`; the essentiality spillover
  runs on `covid_output_essnet` + possibly `…_essnet`. The two families
  cannot be directly compared on hSAR magnitude — they differ in both networks
  and (possibly) calibration. Report them in separate panels; include the
  cross-family note in `HEADLINE.txt` (§3.6).
- **Non-deterministic CUDA** (`Tasks/waning_variant_run.md` §6): the CF7
  regression check uses a 3 % tolerance specifically to accommodate this.
  For a load-bearing spillover number, re-run with `SEED=43/44` (own TAGs)
  and report the across-seed spread.
- **`agent_essential` lazy load:** the spillover hook calls
  `sim._require_agent_essential()` only for the essentiality family. For the
  age-priority family `self.agent_essential = None` and the essential-status
  breakdown columns are omitted from the output csv (not NaN — simply not
  present). The analyzer checks for their presence before plotting
  `spillover_ess_bars`.
- **No mortality analysis.** `M` is a flat scalar; agents never enter a DEAD
  stage. All spillover results are on infections / cases only (consistent with
  the rest of the CF families).
- **Restore-on-exit `trap` must fire on `raise` paths too.** Copy the pattern
  from `run_age_priority_cf.sh` exactly.

---

## 7. Report back

- files + functions added (no existing file should change beyond `targeted_cf/README.md`);
- CF7 regression check ratio (should be in `[0.97, 1.03]`) per county per family;
- `spillover_summary_ALL.csv` — overall protection gap (mean ± SD [p10, p90])
  for each CF vs CF7, age-priority family;
- age breakdown (`spillover_age_bars_ALL.png`) — does CF12 (elderly-first) show
  a larger spillover Δ for child cohabitants than CF13 (working-age-first)?
- essentiality breakdown (`spillover_ess_bars_ALL.png`) — does CF18
  (essential-first vaccine) produce a detectable Δ hSAR for the
  not-in-workforce group?
- household-size breakdown — is there a monotone relationship between
  household size and spillover signal magnitude?
- `HEADLINE.txt` — key numbers with paired Δ mean ± SD [p10, p90];
- any `RuntimeError` hit and the data that caused it;
- FOOD-sensitivity cross-reference: does the FOOD-non-essential sensitivity
  run (from the essentiality task) produce a meaningfully different
  `spillover_ess_bars`?

> Cross-family comparison of spillover magnitude between age-priority and
> essentiality runs is not valid — different networks and possibly different
> calibrations. A fully consistent cross-family spillover comparison would
> require re-running CF-A on the essnet baseline (appendix item, parent plan
> §7.4).