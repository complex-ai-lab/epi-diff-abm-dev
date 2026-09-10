# Runbook / task: work-essentiality counterfactuals (CF-B only)

**Status:** DONE (2026-09-03), all 3 counties. Phase 2 (ACS `C24030` → isolated
patched populations + `covid_output_essnet`) built; CF7 check → all 3
RECALIBRATE; `…_essnet` re-fits done; CF {7,18–24}×30 run; analysis in
`results/TargetedCF_run/essentiality/`. `retained_benefit` reliable for
17117 / 01031 (~0.06–0.07), ill-conditioned for 36003 (full workplace closure
only cuts 16 % there — community/household-driven epidemic). Nothing committed
was overwritten.

**Scope — CF-B only** (vaccinate essential workers first; close only
non-essential workplaces); cases only, no mortality. No age / budget / SES CFs.
Parent design + rationale + other families:
`plans/Task_ Targeted-Allocation Counterfactuals.md` §4.

**What it produces.** For counties 17117 / 01031 / 36003, on **isolated
ACS-C24030 occupation networks** with either the reused `log1p_rw2_l5_rmax5`
params or a re-fitted `…_essnet` baseline (decided by a CF7 check, §1.5), six
new counterfactuals crossing **vaccine allocation** with **targeted vs full
workplace closure**:

| CF | vaccine allocation | workplace closure | school | question |
|----|--------------------|-------------------|--------|----------|
| 7  | uniform (reference) | factual | factual | matches history, **unchanged** |
| 18 | **essential-first** | factual | factual | does prioritising essential workers for vaccine help? |
| 19 | uniform | **non-essential closed** (essential = factual) | factual | targeted closure — how much of the full-closure benefit is kept? |
| 20 | **essential-first** | **non-essential closed** | factual | both levers together |
| 21 | **essential-first** | open | open | protect essential workers, let everything run |
| 22 | uniform | **non-essential closed** | closed | targeted work closure + schools closed |
| 23 | **essential-first** | **non-essential closed** | closed | full targeted package |
| 24 | uniform (ref) | **ALL closed** | factual | full-closure reference for `retained_benefit` |

"essential-first" = every essential-sector worker vaccinated before any
non-essential worker or non-worker (ties Gumbel-random). "non-essential closed" =
non-essential-sector workplace edges always take the **closed** removal severity;
essential-sector workplace edges keep their **factual** severity (so essential
workers have exactly the occ contact they had in history).

Headline: `retained_benefit = (CF7 − CF19) / (CF7 − CF24)` per iteration
(mean ± SD) — the fraction of the *close-all-workplaces* transmission drop that a
*close-only-non-essential* policy still delivers. CF18 vs CF7 isolates the
vaccine-allocation effect.

Plus, per county: daily cases per CF and an **essential / non-essential /
not-in-workforce** breakdown of new infections, with per-iteration uncertainty.

---

## 0. Non-negotiables

- **No silent error handling.** No `try/except: pass`, no default-on-failure, no
  clamp-and-continue. Any missing/short/NaN file, HTTP non-200, empty candidate
  pool, a `{fips}_population.csv` row count ≠ `num_agents`, an `Occupations`
  value not in the crosswalk, a shape mismatch, an unknown CF type →
  `raise RuntimeError(...)` naming the file / county / step / expected-vs-actual,
  exit non-zero. `set -euo pipefail`, tee the traceback.
- **Report every CF effect with a spread, and difference CFs pairwise** — same
  as CF-A: line plots = n-iter mean + 10–90 band; bar charts = bar mean /
  whisker ± 1 SD; every "CFk vs CF7" number is the **per-iteration PAIRED**
  Δ (`d_i = total(CFk, i) − total(CF7, i)`; CRN pairs iteration `i`), reported
  `mean ± SD [p10, p90]`.
- Do **not** touch: `main.py`, `LearnableParams`, the loss, the calibration
  loop, CF types 1–17, the committed baselines, `result_graphs/`,
  `vaccinated_test/`, `waning_tests/`, `results/*/LossCompare/*`.
- `config.yaml` patched-then-restored per run via a `trap` (copy the pattern
  from `targeted_cf/run_age_priority_cf.sh`).
- Env: `.venv_ds` (`/home/pkbiswas/.conda/envs/.venv_ds/bin/python`), A100,
  `SEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `MPLBACKEND=Agg`.
- **`agent_essential` is loaded lazily** — only a work-essentiality run
  (`_essentiality_run`, i.e. any of CF 18–24 in `CF_TYPES`) touches it.
  Age-priority runs, CF 1–17 and calibration never load it, so this cannot
  regress the base engine.

---

## 1. Background the implementer needs

### 1.1 Where occupation lives

`data/population_data/<ST>_population_data/<FIPS>_population.csv` (glob
`data/population_data/*/<FIPS>_population.csv` — fail loud on 0 or >1 match).
Columns: `ID, Age, Household Size, Occupations, Household ID`. **Row order ==
agent index** (`ID` is `range(0, num_agents)`; `age.pickle`, the mobility nets
and the sim all key on the same index). `Occupations` is one of the 19 NAICS
sector names, `"Unemployed"`, or blank (`NaN`, for non-working-age).

The sim does **not** load occupation today — it is used only at network-gen
time (`networks/gen_mob_nw.py` builds a Watts–Strogatz subgraph per sector, then
**merges them into one `occnets/<t>.pkl`** per timestep). This task adds
`self.agent_essential` in `transition.py.__init__` (lazy).

### 1.2 Essentiality crosswalk (CISA v4.0 → this repo's sectors)

```python
ESSENTIAL_SECTORS = {
    "AGRICULTURE", "MINING", "UTILITIES", "TRANSPORTATION", "HEALTHCARE", "FOOD",
    "MANUFACTURING", "CONSTRUCTION", "WASTEMANAGEMENT", "OTHER", "EDUCATION",
}
NON_ESSENTIAL_SECTORS = {
    "WHOLESALETRADE", "RETAILTRADE", "INFORMATION", "FINANCEINSURANCE",
    "REALESTATERENTAL", "SCIENTIFICTECHNICAL", "ENTERPRISEMANAGEMENT", "ART",
}
# "Unemployed" / NaN -> group "other" (not in workforce; never in occ nets)
```

- `OTHER` = public administration (81/92) → essential.
- `EDUCATION` → essential for **vaccine priority**, and **exempt from the
  non-essential occ closure** (teacher–teacher occ contacts are otherwise
  double-controlled with the school lever).
- `FOOD` is one sector but CISA splits it (grocery/production essential,
  dine-in not). **Default essential.** The primary sensitivity axis is flipping
  `FOOD` (and secondarily `MANUFACTURING`) to non-essential — run it.
- Fail loud on any `Occupations` value not in
  `ESSENTIAL_SECTORS ∪ NON_ESSENTIAL_SECTORS ∪ {"Unemployed", NaN}`.

### 1.3 DATA — Phase 2 (re-sourced) is the path this task takes

`scripts/census.py::obtain_occupation_distribution` uses the **2017 `ecnbasic`**
endpoint, which is suppressed for small county × NAICS cells and excludes
government / agriculture / most self-employed. The committed synthetic
populations are badly skewed: 01031 has HEALTHCARE at 27 % of workers and **no**
retail / manufacturing / construction / transport / agriculture / education;
17117 has HEALTHCARE at **57 %** and **zero FOOD** workers. Unusable for an
essentiality analysis.

**This task re-sources industry from ACS 5-year `C24030` (Sex by Industry —
full 19-sector coverage, the same ACS the age / household data already use) and
rebuilds the occupation column + occupational contact networks — all into
ISOLATED folders (§1.7), leaving the committed data untouched.** ACS `C24030`
gives realistic mixes (verified): 01031 HEALTHCARE 12.5 %, RETAIL 11 %, MFG
9.9 %, EDU 9.6 %, TRANSPORT 9 %; 17117 HEALTHCARE 13.4 %, MFG 11.7 %; 36003
EDU 19.2 % (college county), HEALTHCARE 14.7 %, MFG 13.8 %.

**Expect recalibration.** The occ networks change substantially (19 sectors vs
~10, healthcare no longer dominant), so the factual (CF7) trajectory will drift
from the `log1p_rw2_l5_rmax5` fit. The Phase-2 build runs a CF7 check and, if it
fails the threshold (§2.0), recalibrates a `log1p_rw2_l5_rmax5_essnet` baseline
with identical knobs. All essentiality CFs then run off whichever baseline the
check settled on.

(The committed occupation data / networks are not used by this task — they are
documented in `industry_mix.txt` only as the "before" of the data-quality fix.)

### 1.4 Selective closure needs no per-edge tags and no re-netgen at CF time

Every occ edge in `occnets/<t>.pkl` is **within one sector** (`gen_mob_nw.py`
builds each sector's subgraph on that sector's agent IDs, then merges). So for
an occ edge `(u, v)`, `essential[u] == essential[v]` — the edge inherits its
endpoints' class. Selective closure = apply the "closed" edge removal only to
edges whose endpoints are **non-essential** and keep every essential-endpoint
edge. Computed at sim time from `self.agent_essential`; the (isolated Phase-2)
`occnets/<t>.pkl` is untouched by the CF.

Every occ edge in `occnets/<t>.pkl` is **within one sector** (`gen_mob_nw.py`
builds each sector's subgraph on that sector's agent IDs, then merges). So for
an occ edge `(u, v)`, `essential[u] == essential[v]` — the edge inherits its
endpoints' class. Selective closure = apply the "closed" edge removal only to
edges whose endpoints are **non-essential** and keep every essential-endpoint
edge. Computed at sim time from `self.agent_essential`; the factual
`occnets/<t>.pkl` is untouched. (This replaces the plan §4 "B2a" route —
simpler, no `networks/` regen.)

### 1.5 Calibration

The **CF-code** changes are still non-invasive — CF 7's trajectory is
byte-identical to the engine at a *fixed* network set, because the essentiality
code fires only for `cf_type in {18..23}` (`essential_first=False`,
`essential_edge_mask=None` defaults = exact current `apply_vaccines` /
`apply_intervention_fast`). But the **Phase-2 network set is different**, so CF 7
on the essnet nets ≠ CF 7 on the committed nets. Procedure (§2.0):

1. Build the isolated essnet (census → patch population → regen occnets).
2. CF7 check: run `CF_TYPES=7 CF_ITERS=30` on the essnet nets with the
   `log1p_rw2_l5_rmax5` params. Compare `cf7_total` and the regime RMSEs to the
   actual series.
3. If `|cf7_total − actual_total| / actual_total ≤ 0.15` **and** `rmse_all_14`
   within ~30 % of the committed run → **reuse** `log1p_rw2_l5_rmax5` params.
   Else → **recalibrate** `log1p_rw2_l5_rmax5_essnet` (identical knobs:
   `LOSS_MODE=log1p`, `R_SMOOTH_ORDER=2`, `R_SMOOTH_LAMBDA=5`,
   `R_SMOOTH_ENDPOINT_LAMBDA=5`, `PARAM_SNAPSHOT_K=20`, `R_MAX=5`, fixed_60/90,
   251 epochs) on the essnet nets.
4. Run CF 18–23 with `CF_PARAM_FILE` = whichever params, `NETWORKS_DIR` =
   essnet, `AGENT_OCC_CSV` = the patched population.

The essnet CF7 is the reference for the per-iteration paired Δ — regression
check = it must match the essnet calibration's own CF7 to CRN noise (§5.1).

### 1.6 Breakdown groups

essential-sector worker / non-essential-sector worker / not-in-workforce
(children, elderly, unemployed). Same 3-way machinery as CF-A's
`_age_group_masks`, keyed on `agent_essential` instead of age code. (CF-A's age
outputs still get written by the unchanged `get_age_stage_proportions` — leave
them, the essentiality analyzer ignores them.)

### 1.7 Phase-2 isolated folder layout — NOTHING committed is overwritten

```
targeted_cf/phase2/
  fetch_industry_c24030.py     # ACS C24030 -> isolated agents_occupations csv
  patch_population.py          # reassign the Occupations column of the committed
                               #   {FIPS}_population.csv from the new sector mix
                               #   (ages / households / IDs unchanged), + industry_mix report
  regen_occnets.py             # rebuild ONLY occnets/<t>.pkl for the patched pop
  assemble_essnet.py           # symlink HOUSEHOLD/SCHOOL/randnets from covid_output_causal,
                               #   place the new occnets, copy residence_commute_data.txt
  build_essnet.sh              # orchestrate the 4 above for one FIPS
  build_all_essnet.sh          # all 3 counties + industry_mix.txt + CF7 check
  data/agents_occupations_<FIPS>.csv            # C24030, 19-sector
  populations/<FIPS>_population.csv             # patched (Occupations only)

# scratch (off /home, sibling of the committed covid_output_causal):
/scratch/alrodri_root/alrodri/pkbiswas/epi-diff-abm-dev/networks/covid_output_essnet/<FIPS>/mobility_networks/
  HOUSEHOLD_NETWORK.pkl -> ../../covid_output_causal/<FIPS>/mobility_networks/HOUSEHOLD_NETWORK.pkl   (symlink)
  SCHOOL_NETWORK.pkl    -> …/SCHOOL_NETWORK.pkl                                                        (symlink)
  randnets              -> …/randnets                                                                 (symlink dir)
  occnets/{0..181}.pkl                                                                                (NEW, regenerated)
  occnets/<FIPS>_residence_commute_data.txt   (copied from causal — unused at phase 0, harmless)
```

Runs point at it with `NETWORKS_DIR=/scratch/.../covid_output_essnet` and
`AGENT_OCC_CSV=targeted_cf/phase2/populations/<FIPS>_population.csv`. Committed
`populations/pop<FIPS>/`, `data/population_data/`, `data/state_data/` and the
`covid_output_causal` scratch dir are **read-only** to this task.

---

## 2. Implementation

### Step 0 — Phase-2 build (`targeted_cf/phase2/`, run once per county)

Everything writes to `targeted_cf/phase2/…` or the `covid_output_essnet` scratch
dir — nothing committed is touched. **Fail loud** at every step.

**0a `fetch_industry_c24030.py <FIPS>`** — query ACS 5-year `C24030` (Sex by
Industry) for the county, sum male+female per industry, map to the repo's 19
sectors (crosswalk below), write
`targeted_cf/phase2/data/agents_occupations_<FIPS>.csv` (`Occupation,Number` —
same shape as the committed `agents_occupations.csv`). Raise on HTTP non-200,
missing variable, or a sector total of 0 for > 3 sectors.

```
C24030 (M idx, F idx) -> sector   (F idx = M idx + 27)
 004 AGRICULTURE   005 MINING      006 CONSTRUCTION   007 MANUFACTURING
 008 WHOLESALETRADE 009 RETAILTRADE 011 TRANSPORTATION 012 UTILITIES
 013 INFORMATION   015 FINANCEINSURANCE 016 REALESTATERENTAL
 018 SCIENTIFICTECHNICAL 019 ENTERPRISEMANAGEMENT 020 WASTEMANAGEMENT
 022 EDUCATION     023 HEALTHCARE  025 ART          026 FOOD
 027 + 028 -> OTHER   (other services + public administration)
```

**0b `patch_population.py <FIPS>`** — read the committed
`data/population_data/<ST>_population_data/<FIPS>_population.csv`; keep `ID`,
`Age`, `Household Size`, `Household ID` **exactly**; re-draw the `Occupations`
column for the working-age rows (`18 ≤ Age ≤ 65`) from the new sector fractions
using the same logic as `custom_population.py` (7 % unemployment, sector count =
`round(frac * num_employed)`, `np.random.seed(<FIPS as int>)` for
reproducibility, leftover eligible → `"Unemployed"`). Write
`targeted_cf/phase2/populations/<FIPS>_population.csv`. Assert row count and
`ID`/`Age`/`Household ID` columns are identical to the source → else raise.
Append this county's block to `results/TargetedCF_run/<TAG>/industry_mix.txt`
(§ industry report below).

**0c `regen_occnets.py <FIPS>`** — replicate the occupation-network loop from
`networks/gen_mob_nw.py` (only that loop): read the patched population, group by
sector, for each `t in 0..181` compute `occ_factor = 1 + workplaces_pct_change/100`
from the Google mobility report (`networks/isolate_county_data.get_dataframes_dict`,
start date `2020-10-12`), build `custom_watts_strogatz_graph(n, k=min(round(mu *
occ_factor), n-1), p=[rewire, ids])` per sector, merge, `pickle.dump` the
`nx.Graph` to `<essnet>/<FIPS>/mobility_networks/occnets/<t>.pkl`. `random.seed`
and `np.random.seed` to `<FIPS as int>` before the loop. Assert 182 files
written.

**0d `assemble_essnet.py <FIPS>`** — in
`<essnet>/<FIPS>/mobility_networks/`: symlink `HOUSEHOLD_NETWORK.pkl`,
`SCHOOL_NETWORK.pkl`, `randnets/` from the sibling `covid_output_causal/<FIPS>/…`
(raise if a source is missing); copy `occnets/<FIPS>_residence_commute_data.txt`
from causal if present. `occnets/*.pkl` are the real files from 0c.

**0e industry report** — `results/TargetedCF_run/<TAG>/industry_mix.txt`, per
county:
- table: sector → agent count → **% of employed workers** → **% of total
  population**, sorted desc, for BOTH the committed and the C24030-patched
  population (side by side so the improvement is visible);
- summary line: `essential = X workers (Y% of employed, Z% of total pop)`,
  `non-essential = …`, `not-in-workforce = …`, for both.

**0f CF7 check** (`build_all_essnet.sh` tail) — for each county run
`CF_TYPES=7 CF_ITERS=30 NETWORKS_DIR=<essnet> CF_PARAM_FILE=<rmax5 params>` and
compare to the actual series (§1.5 step 3). Print `REUSE` or `RECALIBRATE` per
county to `results/TargetedCF_run/<TAG>/cf7_check.txt`.

### Essentiality crosswalk — one row per repo sector

```python
ESSENTIAL_SECTORS = {
    "AGRICULTURE", "MINING", "UTILITIES", "TRANSPORTATION", "HEALTHCARE", "FOOD",
    "MANUFACTURING", "CONSTRUCTION", "WASTEMANAGEMENT", "OTHER", "EDUCATION",
}
NON_ESSENTIAL_SECTORS = {
    "WHOLESALETRADE", "RETAILTRADE", "INFORMATION", "FINANCEINSURANCE",
    "REALESTATERENTAL", "SCIENTIFICTECHNICAL", "ENTERPRISEMANAGEMENT", "ART",
}
```

### Step 1 — essentiality constants + lazy loader  *(`transition.py`)*

Module constants: `ESSENTIAL_SECTORS`, `NON_ESSENTIAL_SECTORS` (§1.2), and

```python
ESSENTIALITY_CF_TYPES   = frozenset({18, 19, 20, 21, 22, 23})
CF_ESSENTIAL_VACC       = frozenset({18, 20, 21, 23})   # essential-first vaccine
CF_SELECTIVE_OCC        = frozenset({19, 20, 22, 23})   # non-essential-only closure
```

In `__init__`: `self._agent_essential = None` (do **not** load here).

```python
def _require_agent_essential(self):
    """int8 [N]: 0 = not in workforce, 1 = essential sector, 2 = non-essential.
    Loaded once, on first use by a CF 18-23 path. Fails loud.
    AGENT_OCC_CSV env (set by the Phase-2 run scripts) overrides the default
    committed population csv."""
    if self._agent_essential is not None:
        return self._agent_essential
    fips = self.config["simulation_metadata"]["POPULATION"]
    override = os.environ.get("AGENT_OCC_CSV", "").strip()
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"AGENT_OCC_CSV={override!r} does not exist")
        hits = [override]
    else:
        hits = glob.glob(f"data/population_data/*/{fips}_population.csv")
    if len(hits) != 1:
        raise RuntimeError(f"_require_agent_essential: expected exactly one "
                           f"population csv for {fips}, found {hits}")
    df = pd.read_csv(hits[0])
    n = int(self.config["simulation_metadata"]["num_agents"])
    if len(df) != n:
        raise RuntimeError(f"{hits[0]}: {len(df)} rows != num_agents {n}")
    occ = df["Occupations"]
    code = np.zeros(n, dtype=np.int64)
    known = ESSENTIAL_SECTORS | NON_ESSENTIAL_SECTORS | {"Unemployed"}
    bad = set(occ.dropna().unique()) - known
    if bad:
        raise RuntimeError(f"{hits[0]}: unknown Occupations values {sorted(bad)}")
    code[occ.isin(ESSENTIAL_SECTORS).to_numpy()] = 1
    code[occ.isin(NON_ESSENTIAL_SECTORS).to_numpy()] = 2
    self._agent_essential = torch.tensor(code, device=self.device)
    return self._agent_essential

def _essential_group_masks(self):
    e = self._require_agent_essential()
    return (e == 1), (e == 2), (e == 0)      # essential, non-essential, other
```

### Step 2 — essential-first in `apply_vaccines`  *(`transition.py`)*

Add `essential_first=False`. When set, build the rank on **susceptibles only**
exactly like CF-A's age priority:

```python
if essential_first:
    e = self._require_agent_essential()
    susc = (current_stages.reshape(-1) == self.SUSCEPTIBLE_VAR).float()
    rank = (e == 1).float()                    # essential = 1, everyone else = 0
    BIG = 1.0e4
    y = (y.reshape(-1) + BIG * rank * susc).view(N, 1)
```

`priority` (CF-A age dict) and `essential_first` are mutually exclusive — raise
if both are set. Everything downstream (`topk`, straight-through mask,
`S → VACCINATED`) unchanged; keep the `min(quota, n_susceptible)` clamp.

### Step 3 — selective occ closure in `apply_intervention_fast`  *(`transition.py`)*

Add `essential_edge_mask=None` (bool [E], True where BOTH edge endpoints are
essential-sector). When set, split the occ edge list: **non-essential** edges
always take the CLOSED removal severity; **essential** edges take the removal
severity implied by `intervention` (the *factual* occ value that day), so
essential workers keep exactly their factual occ contact:

```python
if essential_edge_mask is not None:
    open_rm   = torch.clamp(OPEN_RM_MEAN   + RM_SD * eps, 0.0, 0.25)
    closed_rm = torch.clamp(CLOSED_RM_MEAN + RM_SD * eps, 0.75, 1.0)
    ess_rm = open_rm if intervention == 0 else closed_rm
    e_idx  = essential_edge_mask.nonzero(as_tuple=True)[0]
    ne_idx = (~essential_edge_mask).nonzero(as_tuple=True)[0]
    g_sel = self._crn_gen(t, "interv_occ_selective") if cf else None
    ke  = int(e_idx.numel()  * (1.0 - ess_rm.item()))
    kne = int(ne_idx.numel() * (1.0 - closed_rm.item()))
    kept_e  = e_idx[torch.randperm(e_idx.numel(),  device=self.device, generator=g_sel)[:ke]]
    kept_ne = ne_idx[torch.randperm(ne_idx.numel(), device=self.device, generator=g_sel)[:kne]]
    return edges[torch.cat([kept_e, kept_ne])]
```

Put this block **before** the existing `if intervention == 0: … else: …` /
`randperm` code so the non-selective path stays byte-for-byte. New CRN call-id
`"interv_occ_selective": 47`. `essential_edge_mask=None` default = current code.

### Step 4 — register CF 18–24

`transition.forward`, extend `logic_map` (occ lever stays **factual** for the
selective types — the split is done in `apply_intervention_fast`):
```python
    18: ("F", "F"),   # essential-first vacc; occ factual;              school factual
    19: ("F", "F"),   # uniform vacc;         non-ess occ closed (sel); school factual
    20: ("F", "F"),   # essential-first;      non-ess occ closed (sel); school factual
    21: (0,   0),     # essential-first;      occ open;                 school open
    22: (1,   "F"),   # uniform;              non-ess occ closed (sel); school closed
    23: (1,   "F"),   # essential-first;      non-ess occ closed (sel); school closed
    24: ("F", 1),     # uniform;              ALL occ closed;           school factual  (NOT selective)
```
After the `cf_type == 11` block:
```python
essential_first = generating_counterfactual and cf_type in CF_ESSENTIAL_VACC
essential_edge_mask = None
if generating_counterfactual and cf_type in CF_SELECTIVE_OCC:   # {19,20,22,23}
    e = self._require_agent_essential()
    occ_t = self.networks['occ'][t]
    essential_edge_mask = (e[occ_t[:, 0]] == 1) & (e[occ_t[:, 1]] == 1)
```
Pass `essential_first` into `apply_vaccines(...)` and `essential_edge_mask` into
the **occ** `apply_intervention_fast(...)` call only.

`abm_nets.py`: append to `CF_POLICY_DESC` / `CF_FOLDER_MAP` (folder names in the
committed code); `ESSENTIALITY_CF_TYPES = {18..24}`;
default CF loop range `1..25`. The essentiality run's `CF_TYPES` default is
`7,18,19,20,21,22,23,24` — CF 24 (uniform vacc, ALL occ closed, factual school)
is the full-closure denominator for `retained_benefit`, run on the essnet nets.

### Step 5 — per-group daily new-infection counter  *(`transition.py`)*

`self.essential_incidence_history = []`; reset at `t == 0`. After
`newly_exposed_today` is finalised (same spot as CF-A's age counter), gate on
**"is this a work-essentiality run"** — `abm_nets.py` sets
`sim.config['simulation_metadata']['_essentiality_run'] = any(ct in 18..24 for
ct in cf_types_to_run)` before the CF loop, so CF 7 (and any CF) in an
essentiality run logs the breakdown, and age-priority / calibration never touch
`_require_agent_essential`:
```python
if generating_counterfactual and self.config['simulation_metadata'].get('_essentiality_run', False):
    ne = newly_exposed_today.reshape(-1)
    em, nm, om = self._essential_group_masks()
    self.essential_incidence_history.append(
        [t, float(ne[em].sum().item()), float(ne[nm].sum().item()), float(ne[om].sum().item())])
```
`save_essential_incidence_to_disk(path, epoch_num)` mirroring
`save_age_incidence_to_disk` (columns `t, essential_new_<i>, nonessential_new_<i>,
other_new_<i>`).

`abm_nets.py eval_net` CF loop: alongside `all_age_incidence`, collect
`all_essential_incidence` for **every CF in an essentiality run** (`_ess_run`),
write per-iter `counterfactual_data<ct>_essential_incidence_raw.csv` and the
averaged `counterfactual_data<ct>_essential_incidence.csv` (`t, essential_new,
nonessential_new, other_new`).

### Step 6 — `targeted_cf/` scripts

Add:
```
targeted_cf/
  recalibrate_essnet.sh          # 251-epoch log1p_rw2_l5 recal on the essnet nets, one FIPS
  run_essentiality_cf.sh         # ONE county: CF 7 + 18..23 on essnet nets
  run_all_essentiality_cf.sh     # Phase-2 build (if missing) + per-county + analysis
  analyze_essentiality_cf.py
```
`run_essentiality_cf.sh` — copy `run_age_priority_cf.sh`, changing:
- `TAG` default `essentiality`, `CF_TYPES` default `7,18,19,20,21,22,23`,
  `OUT=results/${FIPS}/TargetedCF/${TAG}`, analyzer name, `RUN_CONFIG.txt`
  `family=work_essentiality_CF`;
- `export NETWORKS_DIR=/scratch/.../covid_output_essnet` (essnet, not causal);
- `export AGENT_OCC_CSV="$PWD/targeted_cf/phase2/populations/${FIPS}_population.csv"`;
- `CAL` = `results/${FIPS}/TargetedCF/essentiality/calibration/calibrated_params.txt`
  if it exists (recalibrated), else the committed `log1p_rw2_l5_rmax5` params —
  and record which in `RUN_CONFIG.txt`;
- assert the essnet dir + patched population exist → else raise with the
  `targeted_cf/phase2/build_essnet.sh` command to run.
Keep: `set_vacc_config.py --mode fixed --recovered-to-susceptible-time 60
--vaccinated-to-susceptible-time 90`, the `trap` restore, `SKIP_ANALYZE`.

`recalibrate_essnet.sh` — copy the calibration half of
`results/LossCompare_run/run_waning_variant.sh`: `GENERATING_COUNTERFACTUAL=false
NETWORKS_DIR=<essnet> LOSS_MODE=log1p R_SMOOTH_ORDER=2 R_SMOOTH_LAMBDA=5
R_SMOOTH_ENDPOINT_LAMBDA=5 PARAM_SNAPSHOT_K=20 R_MAX=5` + `set_vacc_config.py
--mode fixed 60/90` + `main.py <FIPS>`, stage `calibrated_params.txt` +
diagnostics to `results/<FIPS>/TargetedCF/essentiality/calibration/`, `trap`
restore, `result_graphs/<FIPS>` cleaned up after.

`run_all_essentiality_cf.sh` — for each FIPS: if
`targeted_cf/phase2/populations/<FIPS>_population.csv` or the essnet occnets are
missing → run `targeted_cf/phase2/build_essnet.sh <FIPS>`; read
`cf7_check.txt`; if `RECALIBRATE` → `recalibrate_essnet.sh <FIPS>`; then
`run_essentiality_cf.sh` (`SKIP_ANALYZE=1`). Finally one analyzer call over
`ANALYZE_FIPSES`.

Update `targeted_cf/README.md`.

### Step 7 — `analyze_essentiality_cf.py`

Copy `analyze_age_priority_cf.py` and swap:
- `CF_NAMES` → the 7 rows above (folders from Step 4).
- `GROUPS = ["essential", "nonessential", "other"]`; read
  `counterfactual_data<ct>_essential_incidence_raw.csv` (cols
  `essential_new_<i>` etc.) in `load_..._per_iter`.
- keep everything else: per-CF `counterfactual_results<ct>.png` overwritten with
  n-iter mean + 10–90 band; `cf7_fit.png`, `cf_all_mean.png`, `R_weekly.png`;
  `cf_mean_summary.csv` with `*_mean/*_sd/*_p10/*_p90`; `age_bars`-equivalent
  `essential_bars_<FIPS>.png` (no titles; left = totals by group ± 1 SD, y-axis
  "new infections"; right = **paired** Δ-vs-CF7 as a **percent**, y-axis
  "new infection vs CF7 (%)", per-iteration `(CFk − CF7)/CF7 × 100`, ± 1 SD);
  `essential_breakdown_<FIPS>.csv` (carries both `d_*_{mean,sd,p10,p90}` absolute
  and `d_*_pct_{mean,sd}` percent); roll-ups + `HEADLINE.txt`.
- **fail-loud** consistency check unchanged: `mean(Σ groups)` vs
  `cf_raw_mean_<ct>.sum() / k` ratio must be in `[0.80, 1.03]`.
- extra roll-up `targeted_vs_full_closure_ALL.csv`:
  `retained_benefit = (CF7_total − CF19_total) / (CF7_total − CF24_total)` per
  iteration (paired → mean ± SD [p10, p90]) — the fraction of the
  *close-all-workplaces* transmission drop that a *close-only-non-essential*
  policy still delivers. CF 24 is in the default `CF_TYPES`, run on the essnet
  nets. Fail loud if `CF7_total ≤ CF24_total` for any iteration (the full-closure
  reference is broken).
- also read `industry_mix.txt` and echo the essential / non-essential /
  not-in-workforce headcount + shares into `HEADLINE.txt` (the cost side of the
  targeted-closure trade-off).

---

## 3. Run

```bash
cd /home/pkbiswas/COVID-ABM/epi-diff-abm-dev
PY=/home/pkbiswas/.conda/envs/.venv_ds/bin/python

# --- Phase 2: build the isolated essnet for all 3 counties + industry report + CF7 check
PYTHON=$PY FIPSES="17117 01031 36003" TAG=essentiality \
  bash targeted_cf/phase2/build_all_essnet.sh
#   -> targeted_cf/phase2/{data,populations}/ , /scratch/.../covid_output_essnet/ ,
#      results/TargetedCF_run/essentiality/{industry_mix.txt,cf7_check.txt}

# --- (recalibrate any county the CF7 check flags — the all-driver does this for you)
PYTHON=$PY FIPS=01031 bash targeted_cf/recalibrate_essnet.sh

# --- smoke test (2 iters, one county) — must exit 0, all artifacts
PYTHON=$PY FIPS=01031 CF_ITERS=2 TAG=essentiality_smoke \
  bash targeted_cf/run_essentiality_cf.sh

# --- full run: build (if needed) + recal (if flagged) + CF {3,7,18..23} x 30 x 3 + analysis
PYTHON=$PY FIPSES="17117 01031 36003" TAG=essentiality \
  bash targeted_cf/run_all_essentiality_cf.sh

# --- FOOD-non-essential sensitivity (edit ESSENTIAL_SECTORS, own TAG; reuses the essnet nets)
PYTHON=$PY FIPSES="17117 01031 36003" TAG=essentiality_food_ne \
  bash targeted_cf/run_all_essentiality_cf.sh

# --- add counties / re-analyse: same ANALYZE_FIPSES pattern as age-priority
```

Runtime: Phase-2 build ≈ 15 min/county (occnet regen) + 45 min CF7 check;
recalibration (if flagged) ≈ 30 min/county; CF run 8 types × 30 iters ×
~30 s ≈ 2 h/county. **~10–12 h total for three counties** — background it in
stages, watch `results/TargetedCF_run/essentiality/logs/`.

---

## 4. Outputs

### Per county — `results/<FIPS>/TargetedCF/essentiality/`

| path | contents |
|------|----------|
| `RUN_CONFIG.txt` | TAG, FIPS, SEED, **calibration source** (`log1p_rw2_l5_rmax5` reused / `…_essnet` recalibrated), **NETWORKS_DIR=essnet**, `AGENT_OCC_CSV`, CF_TYPES, CF_ITERS, git rev, timestamp, `family=work_essentiality_CF` |
| `calibration/calibrated_params.txt` | the essnet-recalibrated params, or a copy of the reused `log1p_rw2_l5_rmax5` params |
| `counterfactuals/cf_raw_<ct>.csv` | n_iter × 168 daily cases per CF (3, 7, 18–23) |
| `counterfactuals/counterfactual_results<ct>.png` | n-iter **mean + 10–90 band** vs actual (analyzer overwrites execute()'s single draw) + the `all_counterfactual_results/<CF>/graphs/` copy |
| `counterfactuals/counterfactual_data<ct>_essential_incidence{,_raw}.csv` | essential / non-essential / other new infections per day (mean; and wide per-iteration) |
| `cf_raw_mean_<ct>.csv`, `cf_mean_summary.csv` | case means + `peak_mean/sd`, `total_mean/sd/p10/p90` |
| `cf7_fit.png`, `cf_all_mean.png`, `R_weekly.png` | line plots, mean + 10–90 band |
| `essential_breakdown_<FIPS>.csv` | CF × group → `*_cases_{mean,sd}`, paired `d_*_{mean,sd,p10,p90}` (absolute) and `d_*_pct_{mean,sd}` (percent, per-iteration `(CFk − CF7)/CF7 × 100`) vs CF7 |
| `essential_bars_<FIPS>.png` | no titles; left: totals by group, y "new infections" (±1 SD); right: paired Δ-vs-CF7 as percent, y "new infection vs CF7 (%)" (±1 SD) |

### Cross-county — `results/TargetedCF_run/essentiality/`

`industry_mix.txt` (committed vs C24030, sector %/essential shares — the §"slight
update"), `cf7_check.txt` (REUSE / RECALIBRATE per county),
`cf_mean_summary_ALL.csv`, `cf7_fit_ALL.csv`, `essential_breakdown_ALL.csv`,
`essential_bars_ALL.png`, `targeted_vs_full_closure_ALL.csv`
(`retained_benefit = (CF7−CF19)/(CF7−CF24)` ± SD, + essential/non-essential/
not-in-workforce headcount), `HEADLINE.txt` (paired `d_i` as
`mean ± SD [p10, p90]` + the essential/non-essential/not-in-workforce headcount),
`logs/`.

---

## 5. How to read it

1. **Factual sanity / regression** — `cf_raw_7.csv` (essnet run) must match the
   **essnet calibration's own CF7** to CRN noise (identical params, waning,
   seed, nets; CF 7 hits no essentiality code). If the run reused
   `log1p_rw2_l5_rmax5`, that means `cf7_check.txt` said REUSE and the essnet
   CF7 already tracked the actual series. If they differ, an `essential_first` /
   `essential_edge_mask` default leaked into the factual path — stop.
   (`cf_raw_7` will NOT match the committed `covid_output_causal` CF7 — different
   networks — and that's expected.)
2. **Vaccine allocation** — CF18 vs CF7 (`essential_breakdown`, paired Δ): does
   prioritising essential workers cut *their* infections, and what happens to
   non-essential / not-in-workforce and the total? With the C24030 mix essential
   is still a majority (~65–75 %) but not overwhelming — read the **paired** Δ,
   not the marginal totals.
3. **Targeted vs full closure — the headline.**
   `targeted_vs_full_closure_ALL.csv`: `retained_benefit =
   (CF7−CF19)/(CF7−CF24)` per iteration (mean ± SD) — the fraction of the
   *close-all-workplaces* transmission drop that *close-only-non-essential*
   (CF19) still delivers. Pair it with the workforce kept in person (essential
   share, from `industry_mix.txt`) for the cost side: a small `retained_benefit`
   with a large essential share means "closing the closable half buys little".
4. **Bars** — `essential_bars_<FIPS>.png` right panel (paired Δ-vs-CF7 as a
   percent of the CF7 group total, ± 1 SD): a bar whose whisker clears 0 is a
   real effect at this seed. Watch non-essential /
   other going positive under CF18/21 (displacement).
5. **`industry_mix.txt`** — the committed-vs-C24030 side-by-side is itself a
   result (the data-quality fix); the essential/non-essential/not-in-workforce
   shares are the denominator for every claim.

---

## 6. Gotchas

- **Phase-2 build is isolated (§1.7)** — everything under `targeted_cf/phase2/`
  or `covid_output_essnet` scratch. If any Phase-2 script would write to
  `populations/pop<FIPS>/`, `data/`, or `covid_output_causal`, that's a bug —
  fail loud instead.
- **`patch_population.py` must not change ages / households / IDs** — only the
  `Occupations` column of working-age rows. Assert column equality vs the source.
- **occnet regen** replicates only `gen_mob_nw.py`'s occ loop. Match its Google
  mobility handling (`workplaces_percent_change_from_baseline`, start
  `2020-10-12`, year rollover), `custom_watts_strogatz_graph` signature, and the
  per-timestep `occ_factor`. Seed `random` + `np.random` to `int(FIPS)`.
- **Recalibration is expected.** `cf7_check.txt` will likely say RECALIBRATE for
  most/all counties (19 sectors, healthcare no longer dominant → different occ
  edge count → different factual R). Budget ~30 min/county.
- **`retained_benefit` uses CF 24** (all occ closed, factual school) on the
  essnet nets — `CF_TYPES` default includes `24`. The committed base CF3/CF4 are
  on the old nets and not comparable.
- **essential ≈ 65–75 % even with C24030** — a large majority of workers were
  genuinely deemed essential in 2020–21, so "vaccinate essential first" is a
  broad lever; the sharper contrast is CF19's *selective closure*. FOOD flip is
  still the main robustness axis.
- **CF 7 code path unchanged.** `essential_first=False` /
  `essential_edge_mask=None` defaults must be the exact current
  `apply_vaccines` / `apply_intervention_fast`. Regression check = §5.1.
- **`_require_agent_essential` is lazy and fail-loud** — never called by
  calibration or CF 1–17. If it were eager in `__init__` a missing population
  csv would break the base engine.
- **No work-from-home state** — the model only removes edges, so "non-essential
  closure" deletes those contacts entirely; it overstates the abrupt drop for
  telework sectors (household + community contacts of those workers are still
  modelled, so it is not a total overstatement).
- **`EDUCATION` is exempt from the non-essential occ closure** (double-controlled
  with the school lever) and counted essential for vaccine priority. If a county
  has meaningful EDUCATION employment (17117 has 33 — negligible) revisit.
- **`FOOD` essentiality is a modelling choice**, not in any census variable —
  it's the primary sensitivity axis.
- **Unemployment fixed at 7 %**, non-working-age never assigned a sector,
  children/elderly not in occ nets — all inherited from `custom_population.py`.
- **Occ-net `μ`/rewire are fixed literature values** (Abueg et al. 2021), not
  calibrated — the essential/non-essential contact contrast inherits that.
- **Within-sector-edge assumption** (§1.4): `essential[u] == essential[v]` for
  every occ edge. Assert it once per run (`(e[occ_t[:,0]] == e[occ_t[:,1]]) |
  (e[occ_t[:,0]]==0) | ...`) → `RuntimeError` if an occ edge ever crosses the
  essential boundary (would mean netgen changed).
- **CRN**: new call-id `interv_occ_selective` keyed independent of
  `COUNTERFACTUAL_TYPE`; `essential_first` reuses the existing `vaccine` draw.
- **Non-deterministic CUDA** (`Tasks/waning_variant_run.md` §6): per-iteration
  ± SD is one seed; re-run `SEED=43/44` (own TAGs) for a load-bearing number.
- Restore-on-exit `trap` must fire on `raise` paths too.

---

## 7. Report back

- files + functions changed;
- `industry_mix.txt` — committed vs C24030 sector shares + essential /
  non-essential / not-in-workforce headcounts, all 3 counties (the "slight
  update");
- `cf7_check.txt` — REUSE / RECALIBRATE per county, with the essnet CF7 fit
  numbers;
- CF 7 regression check (essnet CF7 vs the essnet calibration's own CF7);
- `targeted_vs_full_closure_ALL.csv` — the retained-benefit fraction ± SD;
- `HEADLINE.txt` — per-group paired `d_i` mean ± SD [p10, p90];
- FOOD-non-essential sensitivity delta;
- any `RuntimeError` hit and the data that caused it.

> If the recalibration on essnet is used, cross-family comparisons with CF-A
> (which ran on `covid_output_causal` + `log1p_rw2_l5_rmax5`) are **not
> apples-to-apples** — note it; a fully consistent set would re-run CF-A on the
> essnet baseline in an appendix (parent doc §7.4 / §6 caveat).
