# Runbook / task: household-level vs individual-level vaccine allocation (CF-E)

**Status:** NOT STARTED. This is the build spec for **CF-E / §10.2** of
`plans/Task_ Targeted-Allocation Counterfactuals.md`. Companion to the
already-implemented CF-A runbook (`Tasks/age_priority_cf_run.md`) and the
output-only household-spillover analysis (`Tasks/household_protection.md`). The
"Implementation" section is the spec to build to; "Run" is meant to be
repeatable once the code lands.

**Scope — this task only does CF-E** (whole-household vs individual vaccine
allocation), NPIs held factual throughout, cases only. No leakage sweep (CF-F),
no timing factorial (CF-G) — those are separate task files. Parent design +
rationale + the other extension families: `plans/Task_ Targeted-Allocation
Counterfactuals.md` §10.

**The question CF-E answers.** The household-spillover analysis
(`Tasks/household_protection.md`, `results/TargetedCF_run/spillover/HEADLINE.txt`)
established **Observation 2**: under *individual* uniform allocation (CF7) the
intra-household secondary-protection gap is only **+0.14 … +0.33 pp** across the
3 counties, and every age-priority CF moves it by **< ±0.35 pp** — individual
targeting produces almost no household spillover. CF-E asks the policy question
head-on:

> *Does concentrating doses **within** households — vaccinate everyone in a
> household before moving to the next — buy the spillover protection that
> spreading the same number of doses **across** households does not?*

Either answer is a result: "household-optimal allocation manufactures a
detectable within-home protection gap" (spillover is achievable, just not by
individual targeting), or "even whole-household allocation barely moves it"
(the model's household channel is genuinely weak — a strong robustness point for
Observation 2).

**What it produces.** For counties 17117 / 01031 / 36003, using the **already
calibrated** `log1p_rw2_l5_rmax5` params (fixed_60 waning, vacc 90 — **no
recalibration**), four counterfactuals that replace uniform-over-individuals
dose selection with **whole-household** selection, plus the CF7 individual-
allocation reference:

| CF | dose selection | household order | one-liner |
|----|----------------|----------------|-----------|
| 7  | **individual**, uniform over susceptibles | — | **reference** — factual, byte-identical to the committed engine |
| 25 | whole-household bolus | **random** household order | "same coverage, delivered household-by-household" |
| 26 | whole-household bolus | households with **≥1 elderly member first** | shield-the-vulnerable-home |
| 27 | whole-household bolus | **largest households first** | cut the most onward transmission per home |
| 28 | whole-household bolus | **essential-worker households first** | protect the homes most exposed to community import |

Daily dose count is the factual `vaccines(t)` series in every CF; NPIs
(school + work) are factual in every CF (`logic_map` row `("F", "F")` for
25–28). Coverage (Σ doses over the run) is equal to CF7 by construction — the
only thing that changes is *which* susceptibles get the slots and how correlated
they are within homes.

**All five CFs run on `covid_output_causal` + the frozen `log1p_rw2_l5_rmax5`
params — the SAME baseline as CF-A.** CF-E never changes contact structure (it
only reorders who gets a dose), so there is **no factual drift, no essnet
networks, and no recalibration**. The essnet nets / `…_essnet` re-fit exist only
because CF-B changes the occ-network topology; CF-E does not. CF 28 needs *labels*
for "which households contain an essential worker", not the essnet *networks* —
see §1.6.

Per county: daily cases per CF, a child/adult/elderly case breakdown, grouped
bar charts (CF-A analyzer artifacts), **and** the full household-spillover hSAR
set (reusing `spillover_hook.py` + `analyze_spillover.py` over CF {7,25,26,27,28}).

---

## 0. Non-negotiables

- **No silent error handling.** No `try/except: pass`, no default-on-failure, no
  clamp-and-continue on bad data. Any missing/short/NaN file, an empty candidate
  pool, a cumulative dose quota that exceeds the ever-eligible pool, a
  `Household ID` row-count ≠ `num_agents`, a NaN household id, a household with
  zero members in the agent tensor, a shape mismatch, an unknown CF type, or an
  unknown age/sector code → `raise RuntimeError(...)` naming the file / county /
  step / expected-vs-actual, and exit non-zero. Run scripts use `set -euo
  pipefail` and tee the traceback to the batch log. The `min(quota,
  n_susceptible)` clamp already in `apply_vaccines` is *physical* — keep it — but
  add a hard error if the cumulative daily quota over the whole run exceeds the
  ever-eligible pool (dose series / population mismatch).
- Do **not** touch: `main.py`, `LearnableParams`, the loss, the calibration
  loop, **CF types 1–24**, the committed baselines, `result_graphs/`,
  `vaccinated_test/`, `waning_tests/`, `results/*/LossCompare/*`,
  `targeted_cf/phase2/`, `covid_output_essnet`.
- `abm_nets.py` `CF_POLICY_DESC` / `CF_FOLDER_MAP` / the CF-loop range are
  **append-only** — add 25–28, never renumber 1–24.
- `config.yaml` is patched-then-restored per run via a `trap` (copy the pattern
  from `targeted_cf/run_age_priority_cf.sh`).
- Env: `.venv_ds` (`/home/pkbiswas/.conda/envs/.venv_ds/bin/python`), A100,
  `SEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MPLBACKEND=Agg`.
- **CRN discipline.** The new whole-household selection gets its own call-id
  (`_crn_call_ids["vaccine_hh"]`), seeded via `_crn_gen(t, "vaccine_hh")` so it
  is independent of `COUNTERFACTUAL_TYPE`. The per-agent Gumbel tiebreak stays on
  the existing `"vaccine"` call-id — shared byte-for-byte with CF7 — so
  within-household and within-tier order uses the *same* draw as the individual
  reference. CF 7's `priority=None`/`household_order=None` path must remain the
  exact current code.

---

## 1. Background the implementer needs

### 1.1 Where individual allocation happens today

`transition.apply_vaccines(current_stages, num_vaccines, tau, t, agents_ages,
priority, essential_first)` (`covid_abm/substeps/new_transmission/transition.py`):

- builds `is_susceptible` (soft), `logits = log(is_susceptible + 1e-10)`;
- adds Gumbel noise from `u ~ rand(generator=_crn_gen(t, "vaccine"))` in CF mode;
- CF-A adds `BIG * age_rank * susc` to the score (susceptibles only); CF-B adds
  `BIG * essential_flag * susc`;
- `topk(y, num_vax_int)` with `num_vax_int = min(quota, n_susceptible)`;
- straight-through mask, `S → VACCINATED`.

Selection is **uniform over susceptibles** for CF7 (no offset). CF-E adds a third
offset mode — a **per-household** score, broadcast to every member of the
household — so `topk` drains whole households in a fixed order.

### 1.2 Household membership (the tensor to add)

`data/population_data/<ST>_population_data/<FIPS>_population.csv`, column
`Household ID`. **Row order == agent index** (same as `age.pickle`, the mobility
nets, `agent_essential`). `HOUSEHOLD_NETWORK.pkl` is the within-household contact
graph the sim already loads for transmission; CF-E does **not** need its edges —
only the membership relation (which agents share a `Household ID`).

This is exactly what `SpilloverHook.__init__` already builds
(`targeted_cf/spillover_hook.py`; `Tasks/household_protection.md` §1.3):
`agent_to_hh` (int64 `[N]`),
`hh_size_per_agent` (int64 `[N]`), `n_hh`. CF-E promotes that to a shared
read-only loader in `transition.__init__`, lazily invoked only by CF 25–28
(mirrors `_require_agent_essential`).

Build-time asserts (fail loud), identical to the spillover hook:
`len(df) == num_agents`; no NaN in `Household ID`; every agent index `0..N-1`
appears exactly once; no household of size > 10; each group within ±1 of its
declared `Household Size`.

### 1.3 Why the calibration can be reused

CF 7 logic is byte-identical to the committed engine. CF 25–28 only differ inside
`if generating_counterfactual and cf_type in CF_HH_ALLOC`, and they change **only
which susceptibles are removed**, never the NPI series, the `vaccines(t)` totals,
`I_bar`, or any factual-path quantity. The factual case trajectory is unchanged →
R(t) is still valid (parent doc §1 rule; same argument as CF-A §1.5).

**Corollary for the paper (§1 of the parent doc):** any household-allocation CF
that beats CF7 is an **upper bound** — R(t) was fit assuming the factual ≈ uniform
individual allocation.

### 1.4 Coverage-equality invariant

Every CF (7 and 25–28) takes `topk(y, num_vax_int)` with the *same*
`num_vax_int = min(vaccines(t), n_susceptible)` each day. Σ doses over the run is
therefore equal across CFs up to the per-day susceptible-pool clamp. The analyzer
**asserts** `Σ doses(CFk) == Σ doses(CF7)` within 0.5 % per iteration (the hook
already reconstructs total-vaccinated from its `ever_vacc` bookkeeping — §5.3 of
`Tasks/household_protection.md`); a mismatch means a household bolus is spilling
past the daily quota or the clamp fires differently.

### 1.5 SFSusceptibility is not age-graded (interpretation caveat, unchanged)

As in CF-A §1.3: `environment.SFSusceptibility` indexed by the 6-value age code
gives 65A (code 4) the same 1.03 as everyone 40+. So CF26's "shield the elderly
home" benefit, if any, comes from **contact structure + removing susceptibles**,
not an age-susceptibility gradient. Report it plainly.

### 1.6 CF 28 — essential-worker household labels (NOT the essnet networks)

CF 28 orders households by "contains ≥1 essential-sector worker". That flag comes
from `_require_agent_essential()` → the `Occupations` column of a population csv,
mapped through `ESSENTIAL_SECTORS` / `NON_ESSENTIAL_SECTORS`. **It rebuilds no
network** — the sim still runs on `covid_output_causal`, and `log1p_rw2_l5_rmax5`
stays frozen. Two label sources:

| `Occupations` source | how | consequence |
|---|---|---|
| **committed `{fips}_population.csv`** (default) | `_require_agent_essential` globs it automatically | Phase-1 census skew (parent §4): essential ≈ 60–80 % of workers, so "household has an essential worker" is near-universal → **CF 28 ≈ CF 25 (random)**. Directional only. |
| **`targeted_cf/phase2/populations/<FIPS>_population.csv`** (recommended) | `export AGENT_OCC_CSV=<that path>` before the run | ACS C24030 sector mix. `patch_population.py` copies `ID / Age / Household Size / Household ID` through **unchanged** and reassigns only the sector, so the csv is **row- and household-aligned** with the committed `pop<FIPS>` — safe to use for the household tag while the sim runs on the causal networks. |

Use the C24030 override for CF 28 to be a meaningful contrast. It needs the
phase-2 population patch to exist for the county
(`python targeted_cf/phase2/patch_population.py <FIPS>`, part of
`build_all_essnet.sh` — you do **not** need `regen_occnets.py` /
`assemble_essnet.py` / the recalibration, only the patched csv). If the file is
absent, CF 28 falls back to the committed labels with the caveat above — the run
must print which source it used.

`_require_agent_essential` asserts `len(df) == num_agents` but not per-row agent
identity; the alignment guarantee comes from `patch_population.py`'s
copy-through, so **only** point `AGENT_OCC_CSV` at that phase-2 csv (or the
committed one), never at an independently regenerated population.

---

## 2. Implementation

### Step 1 — shared household loader  *(`transition.py`)*

Module constants near `CF_VACC_PRIORITY` / `ESSENTIALITY_CF_TYPES`:

```python
# --- Household-level allocation CF family (CF 25-28) -----------------------
# Whole-household dose selection: once a household is reached in the campaign
# order, every eligible (susceptible) member is vaccinated before the next
# household. NPIs factual for all four. Reference = CF 7 (individual, uniform).
HH_ALLOC_CF_TYPES = frozenset({25, 26, 27, 28})
CF_HH_ORDER = {25: "random", 26: "elderly_home_first",
               27: "largest_first", 28: "essential_home_first"}
```

`_crn_call_ids["vaccine_hh"] = 83` (unused id; 57 = `"vaccine"`, keep it).

`_require_household_members(self)` — lazy, cached on `self._hh_members`; fails
loud; returns `(agent_to_hh [N] int64, hh_size_per_agent [N] int64, n_hh int)`.
Copy the body of `SpilloverHook.__init__`'s household block verbatim (same csv
glob, same asserts). Never called by calibration or CF 1–24.

### Step 2 — whole-household mode in `apply_vaccines`  *(`transition.py`)*

Extend the signature: `apply_vaccines(..., priority=None, essential_first=False,
household_order=None)`. `household_order ∈ {None, "random",
"elderly_home_first", "largest_first", "essential_home_first"}`. Mutually
exclusive with `priority` and `essential_first` (raise if >1 set).

After the existing Gumbel `y = logits + gumbels`, before the `topk`:

```python
if household_order is not None:
    agent_to_hh, hh_size_pa, n_hh = self._require_household_members()
    susc = (current_stages.reshape(-1) == self.SUSCEPTIBLE_VAR).float()

    # per-household base key, drawn ONCE per iteration (t=0), CRN-keyed,
    # independent of COUNTERFACTUAL_TYPE -> the campaign walks households in a
    # fixed order across the whole run; cohabitants are always consecutive.
    g_hh = self._crn_gen(0, "vaccine_hh")
    hh_tiebreak = torch.rand(n_hh, generator=g_hh, device=device)      # (0,1)

    if household_order == "random":
        hh_key = hh_tiebreak
    elif household_order == "elderly_home_first":
        a = agents_ages.reshape(-1).long()
        elderly_agent = torch.zeros(len(a), device=device)
        for c in ELDERLY_CODES:
            elderly_agent += (a == c).float()
        hh_has_elderly = torch.zeros(n_hh, device=device)
        hh_has_elderly.scatter_add_(0, agent_to_hh, elderly_agent)
        hh_key = 2.0 * (hh_has_elderly > 0).float() + hh_tiebreak
    elif household_order == "largest_first":
        hh_size = torch.zeros(n_hh, device=device)
        hh_size.scatter_add_(0, agent_to_hh, torch.ones(len(agent_to_hh), device=device))
        hh_key = hh_size + 0.1 * hh_tiebreak
    elif household_order == "essential_home_first":
        e = self._require_agent_essential()                            # code 1 = essential
        hh_has_ess = torch.zeros(n_hh, device=device)
        hh_has_ess.scatter_add_(0, agent_to_hh, (e == 1).float())
        hh_key = 2.0 * (hh_has_ess > 0).float() + hh_tiebreak
    else:
        raise RuntimeError(f"apply_vaccines: unknown household_order {household_order!r}")

    per_agent_hh_key = hh_key[agent_to_hh]                             # broadcast
    BIG = 1.0e4      # >> |logits| (~23) and Gumbel (~O(1)); dominates ordering
    y = (y.reshape(-1) + BIG * per_agent_hh_key * susc).view(N, 1)
```

`topk`, the straight-through mask, and the `S → VACCINATED` move are unchanged.
Because `per_agent_hh_key` is identical for cohabitants, all susceptible members
of the top household outscore every agent in the next household → `topk` takes
the whole household before spilling (the household bolus). Ties within a
household are broken by the CF7-shared Gumbel. Keep
`num_vax_int = min(num_vax_int, n_susceptible)`; add the cumulative-quota check
(`self._cum_hh_quota` / `self._cum_eligible_ever` → `RuntimeError` on the final
step if quota > eligible).

> `hh_key` is continuous (the `hh_tiebreak` term), so no two households collide;
> `largest_first` uses `+ 0.1 * tiebreak` so size is the primary sort and the
> random term only orders same-size homes. `elderly_home_first` /
> `essential_home_first` are two-tier: flagged homes (key ≈ 2 + u) fully drained
> before unflagged (key ≈ u), random within tier.

### Step 3 — register CF 25–28

`transition.forward`, after the CF-B `essential_first` / `essential_edge_mask`
block:

```python
household_order = None
if generating_counterfactual and cf_type in HH_ALLOC_CF_TYPES:
    household_order = CF_HH_ORDER[cf_type]
```

`logic_map` additions (all NPIs factual):

```python
25: ("F", "F"),   26: ("F", "F"),   27: ("F", "F"),   28: ("F", "F"),
```

Pass `household_order=household_order` into the `apply_vaccines` call
(only reached when `with_vacc`). `agents_ages` is already threaded in (CF-A).

`abm_nets.py`:
- `CF_POLICY_DESC`:
  ```
  25: "whole-household vacc; random household order; NPI factual",
  26: "whole-household vacc; elderly-home first; NPI factual",
  27: "whole-household vacc; largest households first; NPI factual",
  28: "whole-household vacc; essential-worker homes first; NPI factual",
  ```
- **both** `CF_FOLDER_MAP` entries:
  ```
  25: "25_hhalloc_random",
  26: "26_hhalloc_elderly_home_first",
  27: "27_hhalloc_largest_first",
  28: "28_hhalloc_essential_home_first",
  ```
- default CF loop `range(1, 25)` → `range(1, 29)` (the `CF_TYPES` env override
  already lets the runbook pick a subset).
- CF 28 calls `_require_agent_essential` from inside `apply_vaccines` (§1.6 for
  the `AGENT_OCC_CSV` label source) — it is a **lazy load, no config flag
  needed**. Leave `_essentiality_run` keyed on CF 18–24 only, so a pure CF-E run
  never writes the per-essentiality incidence files (CF-E doesn't use them).
- analyzer `CF_NAMES` (Step 6).

### Step 4 — age-incidence counter: no change

`age_incidence_history` is already logged for **every** CF type in a CF run
(`transition.forward`, ~line 1060) and averaged in `abm_nets.eval_net`. CF 25–28
inherit it for free — the child/adult/elderly case breakdown drops out with no
new code.

### Step 5 — `targeted_cf/` scripts (CF-A trio, copied)

```
targeted_cf/
  run_household_alloc_cf.sh       # ONE county: reuse calibration, run CF 7 + 25..28
  run_all_household_alloc_cf.sh   # 3 counties + analysis + spillover pass
  analyze_household_alloc_cf.py   # CF-A-style: cases, totals, age bars
```

`run_household_alloc_cf.sh` — copy `run_age_priority_cf.sh`, change:
- `TAG` default `household_alloc`; `CF_TYPES` default `7,25,26,27,28`.
- `CAL` = committed `results/${FIPS}/LossCompare/log1p_rw2_l5_rmax5/calibration/
  calibrated_params.txt` (frozen, reused); `NETWORKS_DIR` inherits `.env`
  (`covid_output_causal`, same as CF-A).
- everything else (waning-config match via `set_vacc_config.py --mode fixed
  --recovered-to-susceptible-time 60 --vaccinated-to-susceptible-time 90`, the
  `trap` restore, `CF_PARAM_FILE` / `CF_OUTPUT_DIR` exports, `RUN_CONFIG.txt`)
  is identical.
- **CF 28 labels (§1.6):** if `AGENT_OCC_CSV` is not already set, default it to
  `targeted_cf/phase2/populations/${FIPS}_population.csv` when that file exists
  (and `export` it); else print a `[warn] CF28 using committed Occupations
  (CF28 ≈ CF25)` line and continue. Record the resolved path in `RUN_CONFIG.txt`.
- after `main.py`, if `SKIP_ANALYZE != 1`: run `analyze_household_alloc_cf.py`
  **and then** the spillover pass for this county (Step 7).

`run_all_household_alloc_cf.sh` — copy `run_all_age_priority_cf.sh`: loop
`FIPSES="17117 01031 36003"` with `SKIP_ANALYZE=1`, tee to
`results/TargetedCF_run/household_alloc/logs/run_<FIPS>.log`, then one
`analyze_household_alloc_cf.py` over `ANALYZE_FIPSES`, then the spillover roll-up
(Step 7). Update `targeted_cf/README.md` with a CF-E section.

### Step 6 — `analyze_household_alloc_cf.py`

Copy `analyze_age_priority_cf.py`; set:

```python
TAG = os.environ.get("TAG", "household_alloc")
CF_NAMES = {
    7:  ("07_timevar_factual_school_factual_work", "FACTUAL (individual)"),
    25: ("25_hhalloc_random",                      "hh-alloc random"),
    26: ("26_hhalloc_elderly_home_first",          "hh-alloc elderly-home"),
    27: ("27_hhalloc_largest_first",               "hh-alloc largest-first"),
    28: ("28_hhalloc_essential_home_first",        "hh-alloc ess-home"),
}
CF_ORDER = [7, 25, 26, 27, 28]
```

**Palette — matplotlib `tab10` in natural order, indexed by CF position** (same
as `analyze_age_priority_cf.py`'s intent): `PAL = {ct: list(plt.cm.tab10.colors)[i]
for i, ct in enumerate(CF_ORDER)}` → **CF7 = blue**, CF25 = orange, CF26 =
green, CF27 = red, CF28 = purple. CF25–28 land on the *same* colours
`analyze_spillover.py` gives them, so the two plot families agree; CF7 is blue
here (the FACTUAL reference), not grey. Do **not** use
`tab10(linspace(0,1,len(CF_ORDER)))` — a 5-point sample lands a grey on CF27.

Point `county_dir` at `results/<FIPS>/TargetedCF/<TAG>/`. **No other logic
changes** — the CF-A analyzer's plot/table code is CF-list-agnostic. It
reproduces, in the same file layout and style:

- `counterfactuals/counterfactual_results<ct>.png` (+ the
  `all_counterfactual_results/<folder>/graphs/` copy) — the analyzer **overwrites**
  `execute()`'s single-draw plot with the **n-iter mean + 10–90 percentile band
  vs actual**.
- `cf_raw_mean_<ct>.csv`, `cf_mean_summary.csv`
  (`peak_mean/peak_sd/total_mean/total_sd/total_p10/total_p90`).
- `cf7_fit.png`, `cf_all_mean.png` (mean + light 10–90 band per CF),
  `R_weekly.png`.
- `age_breakdown_<FIPS>.csv` — CF × group → `{group}_cases_{mean,sd}`, paired
  `d_{group}_{mean,sd,p10,p90}` (absolute), `d_{group}_pct_{mean,sd}`
  (per-iteration `(CFk − CF7)/CF7 × 100`), `groupsum_vs_reported_ratio`
  (asserted ∈ `[0.80, 1.03]`).
- `age_bars_<FIPS>.png` (no titles) — left: totals by group, y "new infections",
  bar = mean / whisker ± 1 SD; right: **per-iteration paired** Δ-vs-CF7 as a
  percent, y "new infection vs CF7 (%)".
- roll-ups `results/TargetedCF_run/household_alloc/`:
  `cf_mean_summary_ALL.csv`, `cf7_fit_ALL.csv`, `age_breakdown_ALL.csv`,
  `age_bars_ALL.png` (paired Δ absolute ± 1 SD, faceted by county),
  `HEADLINE.txt` (per county × CF × group paired `d_i` as `mean ± SD
  [p10, p90]`; single seed 42, note repeat 43/44).

**Every number carries a spread — no bare means.** Δ-vs-CF7 is always the
per-iteration **PAIRED** difference `d_i = total(CFk, i) − total(CF7, i)` (CRN
pairs iteration `i`; ~5–15× tighter than differencing marginal means).

### Step 7 — household-spillover pass (reuse `spillover_hook.py` unchanged in spirit)

CF-E is *the* scenario where hSAR should move if the household channel has any
force, so the spillover analysis is run over CF {7,25,26,27,28} exactly as for
CF-A. Minimal edits to the existing files:

- `spillover_hook.py`: add `"household_alloc"` to the `--family` choices; it
  needs **no** essential cut (`self.agent_essential = None`, age + hh-size cuts
  only — the default non-essentiality path). `md["_essentiality_run"] = False`
  stays. Everything else (CRN-paired re-run, CF7 regression check, raw-csv
  writers) is untouched.
- `analyze_spillover.py`: add to `CF_NAMES`
  ```
  25: "hh-alloc random",  26: "hh-alloc elderly-home",
  27: "hh-alloc largest",  28: "hh-alloc ess-home",
  ```
  and `fam_cf["household_alloc"] = [7, 25, 26, 27, 28]`.
- `run_spillover_cf.sh`: add a `household_alloc` branch — `CF_TYPES` default
  `7,25,26,27,28`, `NETS_DIR` = `covid_output_causal`, `CAL` = committed
  `log1p_rw2_l5_rmax5`, `TAG` = `household_alloc`, and the **same `AGENT_OCC_CSV`
  default as Step 5** (the spillover hook re-runs the sim, so CF 28 must resolve
  the identical essential-household labels or its CF-vs-CF7 pairing drifts). It
  re-runs the sim (same params/seed/networks) and CRN-pairs against the
  `cf_raw_7.csv` written in Step 5 — the mandatory CF7 case-total regression
  check (ratio ∈ [0.97, 1.03]) guards the pairing.

Spillover outputs land in the **same** `results/<FIPS>/TargetedCF/household_alloc/`
dir; roll-up under `results/TargetedCF_run/spillover/` with `_household_alloc`
suffixes (`spillover_bars_ALL_household_alloc.png`, etc.) alongside the existing
age-priority ones.

### Step 8 — combined gap-vs-total panel  *(add to `analyze_household_alloc_cf.py`)*

Per §10.2 "report the paired Δ protection-gap **and** the total-infection Δ side
by side". After the spillover summary exists, one extra figure per county
`household_alloc_gap_vs_total_<FIPS>.png` — **two panels, same bar style as
`age_bars_<FIPS>.png`** (bar = mean, whisker ± 1 SD, horizontal zero line, one
bar per CF 25–28 in the shared `_palette` — CF25 orange / 26 green / 27 red /
28 purple — no per-metric colouring, no twin axis):

- **Left** — paired Δ **total new infections** vs CF7 (absolute), y
  "Δ total infections vs CF7". Source: `age_breakdown_<FIPS>.csv`
  `d_total_{mean,sd}`.
- **Right** — paired Δ **household protection gap** vs CF7 (percentage points),
  y "Δ protection gap vs CF7 (pp)". Source: `spillover_summary_<FIPS>.csv`
  `d_protection_gap_{mean,sd}`.

Cross-county roll-up `household_alloc_gap_vs_total_ALL.png`: a 2-row grid
(row 0 = Δ total infections, row 1 = Δ protection gap), one column per county,
same per-CF palette.
This is the headline figure: does any household-order variant move **both** the
total *and* the within-home gap, or does concentrating doses within homes trade
total-infection efficiency for a gap that still doesn't open?

---

## 3. Run

```bash
cd /home/pkbiswas/COVID-ABM/epi-diff-abm-dev
PY=/home/pkbiswas/.conda/envs/.venv_ds/bin/python

# CF 28 label prep (once per county): the row-aligned C24030 Occupations patch,
# so "essential-worker household" is a real cut and not ~everyone (§1.6).
# Skippable — CF 28 then falls back to committed labels (CF28 ≈ CF25).
for F in 17117 01031 36003; do $PY targeted_cf/phase2/patch_population.py $F; done

# smoke test first (2 iters, one county, all 5 CF types) — must exit 0, every artifact
PYTHON=$PY FIPS=01031 CF_ITERS=2 TAG=household_alloc_smoke \
  bash targeted_cf/run_household_alloc_cf.sh

# full run: 3 counties x CF {7,25,26,27,28} x 30 iters + CF-A analysis + spillover pass
PYTHON=$PY FIPSES="17117 01031 36003" TAG=household_alloc \
  bash targeted_cf/run_all_household_alloc_cf.sh

# adding counties to an existing run:
PYTHON=$PY FIPSES="36003" ANALYZE_FIPSES="17117 01031 36003" TAG=household_alloc \
  bash targeted_cf/run_all_household_alloc_cf.sh

# re-analyse only (no simulation):
FIPSES=17117,01031,36003 TAG=household_alloc \
  $PY targeted_cf/analyze_household_alloc_cf.py
FIPSES=17117,01031,36003 \
  $PY targeted_cf/analyze_spillover.py --fipses 17117 01031 36003 --families household_alloc
```

Runtime ≈ 5 CF types × 30 iters × ~30 s ≈ 75 min/county for the CF run, plus the
same again for the spillover re-run → **~5 h/county, ~15 h total** on one A100.
Background it; watch `results/TargetedCF_run/household_alloc/logs/`. Single seed
(42); for a load-bearing CFk−CF7 number re-run with `SEED=43` / `SEED=44` (own
TAGs) and report the across-seed spread of the paired Δ.

---

## 4. Outputs

### Per county — `results/<FIPS>/TargetedCF/household_alloc/`

| path | contents |
|------|----------|
| `RUN_CONFIG.txt` / `SPILLOVER_RUN_CONFIG.txt` | TAG, FIPS, SEED, calibration source, CF_TYPES, CF_ITERS, git rev, timestamp |
| `calibration/calibrated_params.txt` | copy of the reused `log1p_rw2_l5_rmax5` params |
| `counterfactuals/cf_raw_<ct>.csv` | n_iter × daily cases per CF (7, 25–28) |
| `counterfactuals/counterfactual_results<ct>.png` | n-iter **mean + 10–90 band** vs actual (analyzer overwrites the single draw) |
| `counterfactuals/counterfactual_data<ct>_age_incidence{,_raw}.csv` | child/adult/elderly new infections/day (mean; raw = one col-set per iter) |
| `counterfactuals/counterfactual_data<ct>_hh_spillover{,_raw}.csv` | hSAR per day (mean; raw = one col-set per iter per metric) |
| `cf_raw_mean_<ct>.csv`, `cf_mean_summary.csv` | case means + `peak/total_{mean,sd,p10,p90}` |
| `cf7_fit.png`, `cf_all_mean.png`, `R_weekly.png` | line plots, mean + 10–90 band |
| `age_breakdown_<FIPS>.csv`, `age_bars_<FIPS>.png` | CF × group cases + paired Δ-vs-CF7 (absolute + percent) |
| `spillover_summary_<FIPS>.csv` | CF × `hSAR_{vacc,unvacc}_{mean,sd,p10,p90}`, paired `dSAR_*`, `protection_gap_*`, `d_protection_gap_*`, age + hh-size cuts |
| `spillover_cf7_regression_<FIPS>.txt` | CF7 case-total check (ratio ∈ [0.97, 1.03] or the run raises) |
| `spillover_hSAR_<FIPS>.png` | 2-panel 14-day hSAR time series (vacc-cohabitant / no-vacc-cohabitant), mean + 10–90 band, one line per CF, vline at rollout, no titles |
| `spillover_bars_<FIPS>.png` | left: absolute hSAR by CF (vacc vs unvacc, ± 1 SD); right: paired Δ protection-gap vs CF7 (pp) |
| `spillover_age_bars_<FIPS>.png` | child/adult/elderly: paired Δ protection-gap vs CF7 (pp), one bar per CF |
| `household_alloc_gap_vs_total_<FIPS>.png` | **headline** — left: paired Δ total infections vs CF7; right: paired Δ protection-gap vs CF7 (pp) |

### Cross-county — `results/TargetedCF_run/household_alloc/` and `.../spillover/`

| path | contents |
|------|----------|
| `household_alloc/cf_mean_summary_ALL.csv`, `cf7_fit_ALL.csv` | per county × CF: peak/total mean+SD; CF7 fit metrics |
| `household_alloc/age_breakdown_ALL.csv`, `age_bars_ALL.png` | per county × CF × group paired Δ (absolute ± 1 SD, faceted) |
| `household_alloc/household_alloc_gap_vs_total_ALL.png` | gap-vs-total headline, faceted by county |
| `household_alloc/HEADLINE.txt` | per county × CF × group paired `d_i` as `mean ± SD [p10, p90]`; single seed 42 |
| `spillover/spillover_summary_ALL.csv` | + `family = household_alloc` rows |
| `spillover/spillover_bars_ALL_household_alloc.png`, `spillover_age_bars_ALL_household_alloc.png` | faceted paired Δ protection-gap |
| `spillover/HEADLINE.txt` | household_alloc family section: CF7 hSAR + paired Δ gap per CF, by age, by hh-size |
| `*/logs/run_<FIPS>.log`, `logs/analyze.log` | full stdout incl. any traceback |

---

## 5. How to read it

1. **Factual sanity** — `cf7_fit.png` / `cf7_fit_ALL.csv` must match the existing
   `results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5/` CF7 fit to within CRN noise
   (CF7 logic byte-identical). The `spillover_cf7_regression_<FIPS>.txt` ratio
   must be ∈ [0.97, 1.03]. If either fails, something leaked into the factual
   path — stop.
2. **Coverage equality** — the analyzer's per-iteration `Σ doses(CFk) vs
   Σ doses(CF7)` check must pass (< 0.5 %). If a household bolus is spilling past
   the daily quota this fires first.
3. **Total infections** — `cf_mean_summary.csv` (`total_mean ± total_sd`) and the
   **paired** `d_total` in `age_breakdown_<FIPS>.csv`. Whole-household allocation
   removes susceptibles in correlated clumps; expect a **small** total effect
   (Observation 2 predicts the household channel is weak) — read it off the
   paired Δ, not the overlapping marginal totals. CF27 (largest-first) is the
   most likely to move the total (drains high-degree homes).
4. **The point — the protection gap.** `spillover_summary_<FIPS>.csv`
   `d_protection_gap_mean ± sd` and `spillover_bars_<FIPS>.png` right panel. Under
   individual allocation (CF7) the gap is +0.14 … +0.33 pp. Does **any**
   household-order variant push the paired Δ of that gap clearly positive
   (whisker clears 0)? CF26 (elderly-home-first) and CF28 (essential-home-first)
   are the designed-to-shield variants; CF25 (random household bolus) isolates
   the pure "concentrate within home" effect with no targeting.
5. **`household_alloc_gap_vs_total_<FIPS>.png`** is the decision figure: a
   variant in the **top-left / bottom-right** quadrant (gap opens, total drops)
   is a win for household-optimal allocation; one that only trades total for gap,
   or moves neither, is the Observation-2 robustness result.
6. **Age / hh-size cuts** — `spillover_age_bars_<FIPS>.png` and the `by hh-size`
   line in `HEADLINE.txt`. Does the spillover Δ concentrate in **child**
   cohabitants (the not-in-workforce members a household bolus is meant to
   shield) and in **hh3p** homes (more cohabitants per dose)? The `elderly` cut
   under CF26 is a tiny-N cell — treat large +pp values there as
   small-denominator noise, exactly as flagged for CF12/16/17 in the
   age-priority spillover HEADLINE.

---

## 6. Gotchas

- **CF 7 must stay byte-identical.** `household_order=None` (and `priority=None`,
  `essential_first=False`) must route through the exact current `apply_vaccines`
  code. Regression: run CF 7 alone, diff `cf_raw_7.csv` against the LossCompare
  `cf_raw_7.csv` for the same county — CRN-scale differences only.
- **Household key drawn once per iteration, not per step.** Seed it with
  `_crn_gen(0, "vaccine_hh")` (t=0) so the campaign order is fixed for the whole
  run and cohabitants are consecutive. Drawing it per step reshuffles the
  household order daily — still a "vaccinate cohabitants together within a day"
  policy, but no longer a coherent household-by-household campaign; don't.
- **CRN semantics vs CF7.** `topk` now selects a different agent set than CF7,
  so past the rollout this is **not** a matched-recipient comparison — only the
  RNG *streams* are shared (`"vaccine"` Gumbel + `"vaccine_hh"` household key are
  both `COUNTERFACTUAL_TYPE`-independent). State it, same wording as parent §5
  caveat 2.
- **Household bolus → sharper S-depletion kinks.** A large household cohort
  vaccinated on one step. CF mode uses plain `torch.bernoulli` (not ST-Bernoulli),
  numerically fine — but check `cf_all_mean.png` for spurious oscillation near
  the rollout start.
- **Size-1 households never contribute to hSAR** (no cohabitant) — `hh1` columns
  are structurally NaN, reported as n/a. A household bolus on a singleton is just
  an individual dose; CF25's effect is entirely from multi-person homes.
- **CF28 essential-household labels** — see §1.6. Default (committed
  `Occupations`) makes CF28 ≈ CF25; run it with
  `AGENT_OCC_CSV=targeted_cf/phase2/populations/<FIPS>_population.csv` (the
  row-aligned C24030 patch) for a real contrast — still on `covid_output_causal`,
  still `log1p_rw2_l5_rmax5`, no essnet networks. The run must log which label
  source it used.
- **Vaccine rollout starts ~day 86** of 182; only ~3 months of the campaign is
  in-window, and hSAR_vacc is undefined until ~day 100 (rollout + 14-day window).
  The spillover summary integrates over days ≥ the county rollout day only.
- **Non-deterministic CUDA** (`Tasks/waning_variant_run.md` §6): per-iteration ±
  SD is the Monte-Carlo spread at one seed; the *paired* Δ can still shift
  seed-to-seed. Load-bearing numbers → seeds 43/44, report the spread.
- **No mortality.** `M` is a flat scalar; all CF-E results are infections/cases
  only (consistent with every other CF family).
- Restore-on-exit `trap` must fire on `raise` paths too — set it immediately
  after `mktemp`.

---

## 7. Report back

- files + functions changed (`transition.py`: `HH_ALLOC_CF_TYPES` /
  `CF_HH_ORDER`, `_require_household_members`, `apply_vaccines(...,
  household_order)`, `logic_map` 25–28; `abm_nets.py`: `CF_POLICY_DESC` /
  `CF_FOLDER_MAP` 25–28, CF range → 29, `_hh_alloc_run` flag;
  `targeted_cf/{run_household_alloc_cf.sh, run_all_household_alloc_cf.sh,
  analyze_household_alloc_cf.py}`; `spillover_hook.py` + `analyze_spillover.py`
  `household_alloc` family; `targeted_cf/README.md`);
- the CF 7 byte-identity check result + the spillover CF7 regression ratio per
  county;
- the coverage-equality check (`Σ doses(CFk) vs CF7`) per CF per county;
- `HEADLINE.txt` (per-group paired `d_i` totals) + the spillover
  `d_protection_gap` per CF, by age and by hh-size;
- **the answer to the CF-E question**: does concentrating doses within
  households open a within-home protection gap that individual targeting
  (CF-A) does not — and at what total-infection cost? Point to
  `household_alloc_gap_vs_total_ALL.png`;
- any `RuntimeError` hit and the data that caused it.

> Cross-family note: CF-E runs on `covid_output_causal` + `log1p_rw2_l5_rmax5`,
> the same baseline as CF-A — so the CF-E vs CF-A spillover comparison **is**
> apples-to-apples (unlike the essentiality family, which is on `covid_output_essnet`).
