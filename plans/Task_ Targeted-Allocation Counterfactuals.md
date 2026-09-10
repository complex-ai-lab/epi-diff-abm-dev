# Task: Targeted-Allocation Counterfactuals (age / work-essentiality / budget-cadence / SES; + §10 extensions: household-level, leakage, timing×targeting, booster-hold, network-hub, adaptive)

Feasibility + step-by-step design for four **new** counterfactual families on top of
the existing school/occ closure CF engine. Companion to
`plans/Task_ Add a VACCINATED State with Independent Waning.md` and the runbook
`Tasks/waning_variant_run.md`. This document is the plan / gotcha catalogue for the
paper; a `Tasks/targeted_cf_run.md` runbook gets written once the code lands.

---

## 0. TL;DR

| # | CF family | Difficulty | New agent attribute | Factual changes? | Calibration |
|---|---|---|---|---|---|
| A | Age-specific vaccination priority | Easy | none (age already loaded) | no | **reuse** `log1p_rw2_l5_rmax5` |
| B1 | Vaccinate essential workers first | Medium | occupation sector | no | **reuse** |
| B2 | Close only non-essential workplaces | Medium | occupation sector (edge tags) | no *if factual graph kept byte-identical* | **reuse** (see §3) |
| C | Vaccine cadence under a budget ceiling | Medium | none | no | **reuse** |
| D0 | SES-priority vaccination (tag only) | Medium | SES quintile | no | **reuse** (but result ≈ null, see §6) |
| D1 | SES-priority + SES-coupled behaviour | Hard | SES quintile | **yes** | **recalibrate** a new baseline |
| E | Household-level (vs individual) allocation | Easy | household id (spillover hook has it) | no | **reuse** — see §10.2 |
| F | Targeting-fidelity / leakage sweep | Trivial | none | no | **reuse** — see §10.3 |
| G | Timing × targeting factorial ("who" vs "when") | Easy | none | no | **reuse** — see §10.4 |
| H | Waning-aware allocation / booster-hold | Easy | vaccination day (spillover hook has it) | no | **reuse** — see §10.5 |
| I | Network-hub (degree-targeted) vaccination | Medium | mean_interactions (in state) | no | **reuse** — see §10.6 |
| J | Reactive / state-switching allocation | Medium | none | no | **reuse** — see §10.7 |

**Extension families E–J** (§10) are the low-cost second wave motivated by the
A / B / spillover results: allocation value is direct not spillover (E), degrades
under leakage (F), and is second-order to timing (G). Recommended first-paper
subset: **E + F + G**.

**Baseline calibration to reuse** (your "fixed_60, vacc 90, R_MAX 5" run, already
in `results/`):

```
results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5/calibration/calibrated_params.txt
FIPS ∈ {17117, 01031, 36003}
waning = fixed, RECOVERED_TO_SUSCEPTIBLE_TIME=60, VACCINATED_TO_SUSCEPTIBLE_TIME=90, R_MAX=5
```

Feed it to a CF run with `CF_PARAM_FILE=…/calibrated_params.txt` exactly as
`results/LossCompare_run/run_waning_variant.sh` does.

---

## 1. How the CF engine works today (the parts we touch)

- `covid_abm/substeps/new_transmission/transition.py :: forward()` reads
  `populations/pop<FIPS>/intervention.csv` **every timestep** and pulls three
  scalars for day `t`: `school_intervention`, `occ_intervention`, `vaccines`.
- `GENERATING_COUNTERFACTUAL=true` → `logic_map[cf_type]` overrides the two
  closure scalars (`0/1/"F"/"CF"`), and `cf_type == 11` sets `num_vaccines = 0`.
  Everything else is inherited from the active config (waning mode, etc.).
- `apply_intervention_fast(value, edges, t, net_id)` removes a fraction of edges
  from one network. Open → remove ~N(0.125, 0.0625) clamped [0,0.25]; closed →
  ~N(0.875, 0.0625) clamped [0.75,1]. In CF mode the fraction + kept-edge subset
  come from `_crn_gen(t, "interv_<net>")` — **independent of `COUNTERFACTUAL_TYPE`**
  (common random numbers across policies).
- `apply_vaccines(current_stages, num_vaccines, tau, t)` builds
  `is_susceptible` (soft), adds Gumbel noise (CRN via `_crn_gen(t,"vaccine")` in
  CF mode), takes `topk(y, num_vaccines)`, moves those `S → VACCINATED`.
  **Selection is currently uniform over susceptibles — this is the single hook
  for A, B1, D.**
- Occupation nets: `networks/gen_mob_nw.py` builds a Watts–Strogatz subgraph
  **per NAICS sector** then `G_all.add_edges_from(...)` **merges them into one
  graph per timestep** (`occnets/<t>.pkl`). The sim never sees sector identity.
- Agent attributes that reach the sim: **only `age`** (`populations/pop<FIPS>/age.pickle`,
  a pickled `pd.Series` loaded by `load_population_attribute`). `age.pickle` values:
  `0=20t29, 1=30t39, 2=40t49, 3=50t64, 4=65A, 5=U19`.
- `populations/pop<FIPS>/*` row order == agent index == `{fips}_population.csv`
  row order. `{fips}_population.csv` columns: `ID, Age, Household Size,
  Occupations, Household ID` (`Occupations` = one of 19 NAICS sector names,
  `"Unemployed"`, or blank for non-working-age).

### The calibration-reuse rule (state this in the paper)

The calibrated weekly-R vector absorbs **everything not explicitly in the model**
(behaviour, seasonality, variants, reporting). Therefore:

> A CF may reuse `log1p_rw2_l5_rmax5` **iff the factual path (CF7-equivalent)
> produces a byte-identical case trajectory after the change** — i.e. the new
> mechanism only fires inside `if generating_counterfactual` / for `cf_type >= 12`.
> If the factual dynamics move, R(t) was fit to the wrong generative model and
> the CF contrast is contaminated → recalibrate a fresh baseline and run *all*
> CFs you want to compare off that same baseline.

Corollary for the paper: any allocation CF that beats factual is an **upper
bound** on the real-world benefit, because R(t) was estimated assuming the
factual (≈ uniform) allocation.

---

## 2. Shared infrastructure (do once, before A–D)

1. **New agent-attribute loader in `transition.py.__init__`** (mirrors how the
   networks are preloaded — no `config.yaml` state-schema surgery):
   ```python
   pop_csv = f"data/population_data/<ST>_population_data/{POPULATION}_population.csv"
   self.agent_occ   = <int64 tensor [N]>   # sector id
   self.agent_ses   = <int64 tensor [N]>   # SES quintile 0..4
   ```
   Age is already in `state['agents']['citizens']['age']` — thread it into
   `apply_vaccines` (currently not passed). **Fail loud** (see item 6): a missing
   file, a row count ≠ `num_agents`, an unmappable sector, a NaN, or an
   out-of-range code is a `RuntimeError` with the offending path/row, not a
   silent `-1` sentinel.
2. **CRN discipline.** Every new stochastic choice gets a new call-id in
   `self._crn_call_ids` (e.g. `"vaccine_age": 101`), seeded via `_crn_gen(t, id)`
   so it is independent of `COUNTERFACTUAL_TYPE`.
3. **Register new CF types (12+).** Append — never renumber 1–11 — in *all* of:
   `abm_nets.py` `CF_POLICY_DESC`, both copies of `cf_folder_map`, the CF loop
   `range(1, 12)` → `range(1, N)`; `transition.py` `logic_map`; the analyzer
   `CF_NAMES`.
4. **Per-age-group daily new-infection counters** in the substep (see §3), so
   the corrected child/adult/elderly split has a case series to aggregate.
   (Mortality is out of scope — `M` is a flat scalar and agents never enter a
   DEAD stage, so the model has no age-resolved deaths; don't build that.)
5. **New scripts dir** `targeted_cf/` (mirrors `vaccinated_test/` +
   `results/LossCompare_run/`): `set_alloc_config.py` (patches only the new
   keys, targeted line-replace like `set_vacc_config.py`), a per-county
   `run_*.sh` that reuses `CF_PARAM_FILE` + `CF_OUTPUT_DIR` and restores
   `config.yaml` on a trap, an all-county driver, and an `analyze_*.py`.
6. **No silent error handling (hard requirement).** New code must not `try/except:
   pass`, must not substitute a default on a failed fetch/read, must not clamp a
   bad value into range. On any deviation — missing/short/NaN data file, HTTP
   non-200, empty candidate pool, dose count > eligible pool, shape mismatch,
   unknown CF type, unknown age/sector/SES code — raise with a message that names
   the file, the county, the step, and the expected vs actual, and exit non-zero.
   The runbook prints the traceback to the batch log. (The existing `try/except`
   in `abm_nets.execute()`, the commuter loader, `process_counties` vaccine fetch
   etc. are pre-existing and out of scope, but do not copy that pattern.)
7. **Outputs** land under `results/<FIPS>/TargetedCF/<TAG>/`, **mirroring the
   `results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5/` layout** — same subfolders
   (`calibration/`, `counterfactuals/…/all_counterfactual_results/<CF>/…`) and
   the same analyzer artifacts (`cf_mean_summary.csv`, `cf7_fit.png`,
   `cf_all_mean.png`, `R_weekly.png`, `cf_raw_mean_<ct>.csv`) — plus the new
   age artifacts. Cross-county roll-ups under `results/TargetedCF_run/<TAG>/`.
   Never touch `result_graphs/`, the committed baselines, `vaccinated_test/`,
   `waning_tests/`.

---

## 3. CF-A — Age-priority vaccination × school/work NPI  *(Easy, reuse calibration)*

**Runbook: `Tasks/age_priority_cf_run.md` — this is the fully specified first
task. Cases only, no mortality analysis.**

**Idea.** Same daily dose count as factual (`vaccines(t)`), but fill from a
priority age order instead of uniformly, crossed with the school/work closure
levers. `apply_vaccines` gains `agents_ages`; a large priority offset is added
to the candidate logits **for susceptibles only** so `topk` drains the top band
before spilling. Ties within a band stay Gumbel-random (CRN preserved).

**Six new CF types (12–17):**

| CF | vaccine priority | school | work |
|----|------------------|--------|------|
| 12 | elderly-first     | factual | factual |
| 13 | working-age-first | factual | factual |
| 14 | working-age-first | closed  | open   |
| 15 | working-age-first | open    | open   |
| 16 | elderly-first     | open    | closed |
| 17 | elderly-first     | closed  | closed |

- elderly-first = `65A → 50t64 → 40t49 → 30t39 → 20t29 → U19`
- working-age-first = `{20t29,30t39,40t49,50t64}` equal rank, then `65A`, then `U19`
- reference = CF 7 (uniform vaccine, factual school + work).
- optional matched-NPI uniform baselines: CF 1 ↔ CF 15, CF 3 ↔ CF 16, CF 2 ↔ CF 14.

**Factual (CF7):** untouched → **reuse** `log1p_rw2_l5_rmax5`.

**Deliverables (see the runbook) — every number carries a spread:**
- daily cases per CF (`cf_raw_<ct>.csv`); per-CF line plot = n-iter **mean +
  10–90 percentile band** vs actual (`counterfactual_results<ct>.png`, analyzer
  overwrites `execute()`'s single draw).
- per-age-group (child / adult / elderly) **daily new infections**, logged
  **per iteration** (`counterfactual_data<ct>_age_incidence_raw.csv`) → mean +
  SD; corrected `*_age_proportions.csv`.
- grouped bar plots per county (`age_bars_<FIPS>.png`): total cases per group
  per CF (bar = mean, whisker ± 1 SD); + a **per-iteration paired** Δ-vs-CF7
  panel (CRN pairs iteration `i`, so the paired SD is ~5–15× tighter than
  differencing marginal means).
- `age_breakdown_<FIPS>.csv` (CF × group → `*_cases_{mean,sd}`, paired
  `d_*_{mean,sd,p10,p90}`) + cross-county roll-up + `HEADLINE.txt`.
- the LossCompare analyzer set (`cf7_fit.png`, `cf_all_mean.png`, `R_weekly.png`,
  `cf_mean_summary.csv` with `*_mean/*_sd`) reproduced in the same layout.

**Data.** None required beyond the reused calibration. *Optional validation:* CDC
"COVID-19 Vaccinations in the US, County"
(`https://data.cdc.gov/resource/8xkx-amqh.csv`, already used in
`scripts/process_counties.py`) carries
`administered_dose1_recip_5plus / _12plus / _18plus / _65plus` — real
age-structured uptake to score CF-12's age split against.

**Gotchas / caveats.**
1. **BLOCKING BUG — the child/adult/elderly split is mis-wired.**
   `process_counties.create_county_folder` encodes `age.pickle` as
   `20t29=0, 30t39=1, 40t49=2, 50t64=3, 65A=4, U19=5` (matches
   `populations/pop<FIPS>/mapping.json`). But
   `transition.get_age_stage_proportions` / `get_mean_agent_interactions` /
   `substeps/utils.py` use `child = age<1` (→ code 0 = **people in their 20s**),
   `elderly = age>4` (→ code 5 = **under-20s**), `adult = 1..4` (30+ incl.
   elderly). Every existing `*_age_proportions.csv` is **mislabeled**. Fix to
   `child = {5}`, `adult = {0,1,2,3}`, `elderly = {4}` **in the diagnostic paths
   only** (`get_age_stage_proportions` + the new age-incidence counter). LEAVE
   `get_mean_agent_interactions` as-is — its per-agent constant feeds `I_bar` in
   transmission and is baked into every committed calibration; re-labelling it
   changes the factual → invalidates the reused params. Note in the paper that
   pre-fix age outputs are void. (Done — CF-A, `_age_group_masks` in
   transition.py.)
2. **`SFSusceptibility` does NOT rise with age as wired.** Shape-9 vector
   `[0.35, 0.69, 1.03, 1.03, 1.03, 1.03, 1.27, 1.52, 1.52]` is indexed by the
   6-value age code, so 65A (code 4) → 1.03, same as everyone 40+; the 1.27/1.52
   tail is dead code from a 9-bracket model. The elderly are **not**
   intrinsically more susceptible per contact in this model — the age-priority
   case contrast comes only from contact structure + removing susceptibles.
   Leave it and report the limitation (fixing it changes the factual → needs
   recalibration; out of scope).
3. **`65A` is one bucket** (65+), `U19` is one bucket — no within-65+ or
   child/teen resolution.
4. **Contact-rate wiring is inconsistent** across
   `transition.get_mean_agent_interactions` (2/3/4), `mobility_mapping.json`
   (μ 1/3/3), `random_network_parameters.csv` (2/4/3). Confirm which path is
   live and document it — it drives the transmission side of the CF.
5. Priority spill must be deterministic + CRN-safe; if the **cumulative** dose
   quota over the run exceeds the ever-eligible pool, fail loud (not a silent
   clamp).
6. CF 14–17 move NPIs, so `apply_intervention_fast` fires the open/closed
   branch — CRN keeps the closure realisation shared with any base CF at the
   same (iter, step, network) (CF 15's open occ net == CF 1's). Good for
   comparability.
7. The real Dec-2020→spring-2021 US rollout **was** phased oldest-first, so
   "uniform" (CF7) is a synthetic reference, not history.

---

## 4. CF-B — Work essentiality  *(Medium)* — **runbook: `Tasks/work_essentiality_cf_run.md`**

Six new CF types **18–23** crossing vaccine allocation with targeted vs blanket
workplace closure; cases only; reuse `log1p_rw2_l5_rmax5` **iff** CF7 stays
byte-identical (essentiality code fires only for `cf_type ≥ 18`). Per-iteration
uncertainty + CRN-paired Δ, same as CF-A. Uses the **existing** `Occupations`
column in `{fips}_population.csv` — no `config.yaml` state-schema change, loaded
lazily in `transition.py.__init__` and only touched by CF 18–23.

### B1 — vaccinate essential workers first  *(reuse calibration)*

`apply_vaccines(..., essential_first=True)` — the CF-A priority-offset trick
keyed on an `agent_essential` flag (essential = rank 1, everyone else 0),
susceptibles only. Factual (uniform) unchanged → **reuse**.

### B2 — close only non-essential workplaces  *(reuse calibration — no netgen)*

**Every occ edge is within one sector** (`gen_mob_nw.py` builds each sector's WS
subgraph on that sector's agent IDs, then merges), so `essential[u] ==
essential[v]` for every occ edge. Selective closure = at sim time, build
`essential_edge_mask = (agent_essential[occ_t[:,0]]==1) & (…[:,1]==1)` and, when
the factual `occ_intervention` says "closed", apply the edge removal only to the
non-essential subset, keeping every essential-endpoint edge. The factual
`occnets/<t>.pkl` is untouched and `essential_edge_mask=None` (default) is the
exact current code → **reuse**. (Simpler than the old "B2a" per-edge-tag route —
no `networks/` regen.)

### CF types

| CF | vaccine | occ closure | school |
|----|---------|-------------|--------|
| 7 (ref) | uniform | blanket, factual sched | factual |
| 18 | essential-first | blanket, factual sched | factual |
| 19 | uniform | **non-essential only**, factual sched | factual |
| 20 | essential-first | **non-essential only**, factual sched | factual |
| 21 | essential-first | open | open |
| 22 | uniform | **non-essential only**, factual sched | closed |
| 23 | essential-first | **non-essential only**, factual sched | closed |

Headline: **CF19 vs CF7** (targeted vs blanket) and **CF19 vs base CF3**
(targeted vs full closure) → fraction of the full-closure transmission benefit
retained by exempting essential workers.

### Essentiality crosswalk (CISA v4.0 "Essential Critical Infrastructure Workforce" → this repo's 19 NAICS sectors)

| Sector (repo name / NAICS) | Class | Note |
|---|---|---|
| AGRICULTURE 11, UTILITIES 22, TRANSPORTATION 48-49, HEALTHCARE 62, FOOD 72 (grocery/production) | essential | |
| MANUFACTURING 31-33 | mixed → essential | CISA leaned essential for critical mfg/food/pharma |
| CONSTRUCTION 23, WASTEMANAGEMENT 56, OTHER 81/92 (public admin) | essential | |
| WHOLESALETRADE 42, RETAILTRADE 44-45 | mixed | grocery essential, rest not; repo has no sub-split |
| INFORMATION 51, FINANCEINSURANCE 52, REALESTATERENTAL 53, SCIENTIFICTECHNICAL 54, ENTERPRISEMANAGEMENT 55 | non-essential (telework) | |
| ART 71, FOOD 72 (dine-in) | non-essential | closable |
| EDUCATION 61 | handled by the **school** lever, exclude from occ closure | |

**Data — Phase 1 uses the EXISTING `Occupations` column; the sector mix is
skewed but usable with a headline caveat.**
`scripts/census.py::obtain_occupation_distribution` (2017 `ecnbasic`, suppressed
for small county × NAICS cells, excludes government/agriculture/self-employed)
gave the 3 counties **no retail / manufacturing / construction / transport /
agriculture** workers and 17117 **no FOOD** workers, with HEALTHCARE at 27–57 %
of the workforce. So "essential" ends up 60–80 % of workers — the CF18 vaccine
lever is muted by construction. Phase-1 results are **directional**; the FOOD
flip is the robustness check. **Phase 2** (own task section
`Tasks/work_essentiality_cf_run.md` §2.9): re-source via ACS 5-year `C24030`
(Sex by Industry — full NAICS coverage), regenerate the 3 populations + occ
nets, re-check CF7, recalibrate `log1p_rw2_l5_rmax5_essnet` only if it drifted,
re-run 18–23 (and A/C for comparability if recalibrated).

**Gotchas / caveats.**
1. Sector-mix skew (above) is the dominant caveat, not a hard blocker for
   Phase 1 — B1/B2 run off the existing `Occupations` and the reused
   calibration. Only Phase 2 (ACS re-source + netgen) risks a factual drift →
   recalibration.
2. "Essential" was **defined differently by every state and shifted over time**
   (esp. construction, manufacturing, cannabis, places of worship). Cite CISA
   v1→v4 and pick a fixed mapping; run the classification as a sensitivity axis.
3. No **work-from-home** state: the model can only *remove edges*. Professional
   services "closing" = deleting contacts, same as a shuttered restaurant — it
   overstates the transmission drop from telework sectors (those workers still
   have household + community contacts, which is captured, but the abruptness is
   not).
4. Unemployment hard-coded at 7% (`custom_population.py`); non-working-age never
   get a sector; children/elderly are not in occ nets at all.
5. Occupation `μ`/rewire in `networks/network_parameters/occupation_network_parameters.csv`
   are **fixed literature values (Abueg et al. 2021), not calibrated** — the
   essential/non-essential contrast inherits their uncertainty.

---

## 5. CF-C — Vaccine rollout cadence under a budget ceiling  *(Medium, reuse calibration)*

**Idea.** Hold the **cumulative** dose count fixed (or under a ceiling) but
change *when* doses arrive: daily (factual) → weekly / biweekly / monthly
boluses, or a fixed doses-per-week ceiling with a carried-forward backlog, or a
delayed start.

**Model change.** Reshape the `vaccines` column in the CF branch only. Cleanest:
a pure function `reshape_doses(series, cadence, cap, start_delay) -> series` with
`sum(out) == sum(in)` (or `<=` for a hard cap), selected by `cf_type`. Apply
where `num_vaccines = curr_intervention.iloc[0]['vaccines']` is read, or
precompute `populations/pop<FIPS>/intervention_cf<NN>.csv` and switch the read
path on `cf_type`.

**New CF types:** `18` weekly boluses, `19` monthly boluses, `20` 50%-throughput
cap w/ backlog, `21` rollout delayed 4 weeks, …

**Factual:** `vaccines(t)` untouched → **reuse** `log1p_rw2_l5_rmax5`. Cheap
(no gradient) → sweep many cadence values.

**Gotchas / caveats.**
1. **Dose scaling.** `vaccines` is a **raw county count**; `num_agents ≈ 48 k`
   but the county populations are larger (01031 ≈ 58 k). `apply_vaccines` clamps
   `num_vax_int` to the current susceptible count, so under daily cadence the
   truncation is small but under a **monthly bolus** a large count can be
   silently clipped → the reshaped scenarios lose doses relative to factual.
   Rescale by `num_agents / county_pop` (from `data/processed_data` or ACS
   `B01003`) and assert `Σ reshaped ≈ Σ factual` after clamping.
2. **CRN semantics.** `_crn_gen(t,"vaccine")` is keyed on `t`; moving the
   non-zero days changes which agents are picked across cadences. That is the
   *intended* scenario difference, but it is **not** a matched-recipient
   comparison — only the RNG *stream* is shared. State it.
3. A weekly/monthly bolus vaccinates a big cohort on one step → sharper
   S-depletion kinks; check the ST-Bernoulli path isn't destabilised (CF mode
   uses plain `torch.bernoulli`, so fine).
4. If you add a real budget model (doses cost X, weekly budget B), keep it in a
   sidecar config — the task note says it's optional and not required.
5. Waning interacts: with `VACCINATED_TO_SUSCEPTIBLE_TIME = 90`, a **delayed**
   rollout means fewer agents wane back to S inside the 182-day window — the
   cadence effect and the waning effect are partly confounded. Report both the
   `vaccinated` and `susceptible` compartment traces (already in
   `counterfactual_dataXX_proportions.csv`).

---

## 6. CF-D — SES-based vaccine rollout  *(Medium → Hard)*

**Constraint.** No SES attribute, no sub-county geography, one county per sim.

### D0 — SES tag only  *(reuse calibration, but weak result)*

**Model change.** Assign each **household** an income bracket sampled from ACS
`B19001` (16 brackets) → map to SES quintile; every member inherits it. Write
`populations/pop<FIPS>/ses.pickle` (per-agent quintile). `apply_vaccines`
priority-offset keyed on quintile (`22` low-SES-first, `23` high-SES-first).

**Factual:** random → reuse. **But:** if SES does not touch contacts,
susceptibility, or compliance, CF-22/23 differ from factual **only by the label
of who gets an immune slot** — the aggregate case curve barely moves (≈ null
result). Only worth publishing as a "targeting an orthogonal axis does nothing —
you need the coupling" point.

### D1 — SES coupled to behaviour  *(recalibrate a fresh baseline)*

Add ≥1 coupling so SES targeting has epidemiological content:
- SES → **isolation compliance**: scale `quarantine_start_prob` /
  `make_isolation_decision` inputs by quintile (low SES → lower ability to
  isolate: less paid leave, denser housing).
- SES → **occupation mix**: bias low-SES agents toward in-person essential
  sectors (needs B's occupation fix first; ACS `C24010`/PUMS gives
  occupation × income).
- SES → **household size**: already weakly present via `B11016` but not
  SES-conditioned; ACS `B25010`/PUMS by income.

Any of these changes the **factual** dynamics → **recalibrate**
`log1p_rw2_l5_rmax5`-style with the SES-coupled model to get `…_ses` params, and
run the SES CFs (and, for a clean comparison, re-run A/B/C) off that baseline.

### D2 — curated SES-stratified networks  *(recalibrate; largest lift)*

Assortative-by-SES community/occupation mixing in `networks/`. Highest fidelity,
most work, most degrees of freedom to defend.

**Data.**
- Census (county cross-section): `B19001` household income, `B19013` median hh
  income, `B17001`/`S1701` poverty, `B15003` education, `C24010` occ×… ,
  `B25010`/`B25014` household size / crowding, PUMS for joint distributions.
- **CDC/ATSDR Social Vulnerability Index (SVI)** — county *and* tract, four
  themes (socioeconomic status, household composition, minority/language,
  housing/transport). A ready composite; `data.cdc.gov` resource, or the ATSDR
  download. Good for parameterising "which county is more vulnerable" and for a
  tract-level extension.
- **Vaccination equity signal:** CDC "COVID-19 Vaccination and ... by SVI"
  datasets give national vaccination coverage by SVI quartile over time — use
  these curves to set the *factual* differential uptake by quintile (the real
  rollout already favoured higher-SES areas early).
- **CovidCast:** no SES-stratified county time series. `fb-survey` has
  `smoothed_wcovid_vaccinated`, `…_vaccine_likely_*`, hesitancy — county level,
  noisy, not by SES. Not a substitute for ACS/SVI here.

**Gotchas / caveats.**
1. **Ecological fallacy / no spatial sorting.** Assigning the county income
   distribution to households with no residential clustering **understates**
   how concentrated exposure is by SES. Tract-level networks (D2) or at least
   SES-assortative mixing is needed for a strong claim.
2. SES quintile is **synthetic** → a coupling-strength sensitivity sweep is
   mandatory, not optional.
3. **Confounding:** SES correlates with age (retirees), household size, and
   occupation (already essential-heavy in this model). An SES CF and an age CF
   and a B1 CF are **not independent levers** — say so, and consider a factorial
   design.
4. **Baseline fragmentation:** D1/D2 force a new calibration. If some CFs run off
   `rmax5` and D runs off `…_ses`, cross-family effect sizes are not directly
   comparable. Cleanest: once any family needs a recalibration, run the whole CF
   matrix off that one baseline. Recommend treating **D as its own sub-study**
   with A/B/C replicated on the `…_ses` baseline in an appendix.

---

## 7. Cross-cutting caveats for the paper

1. **R(t) absorbs unmodelled heterogeneity** → allocation CFs give an *upper
   bound* on real benefit (see §1 corollary).
2. **Flat mortality** (`M` scalar; agents never enter a DEAD stage): the model
   has no age-resolved deaths. All allocation CFs are judged on **infections /
   cases only**. Do not attempt a deaths result from this model.
3. **CRN is keyed independent of `COUNTERFACTUAL_TYPE`** — preserve that for
   every new stochastic call (new call-id in `_crn_call_ids`), or the
   policy-vs-RNG attribution the benchmark relies on breaks.
4. **Report every CF effect with a spread, and difference CFs pairwise.** The
   analyzer keeps all `n_iter` draws: line plots get a mean + 10–90 band, bar
   charts get bar = mean / whisker = ± 1 SD, and any "CFk vs CF7" number is the
   **per-iteration PAIRED** Δ (`d_i = total(CFk, i) − total(CF7, i)` — CRN pairs
   iteration `i` across CF types, so the paired SD is ~5–15× tighter than
   differencing marginal means and a small allocation effect becomes visible).
   Separately, the **non-deterministic CUDA scatter** (`Tasks/waning_variant_run.md`
   §6) means the paired Δ itself moves seed-to-seed → for a load-bearing number
   re-run with seeds 43/44 and report the across-seed spread too.
5. **3 counties, 182 days, window 2020-10-26 → 2021-04-25.** The vaccine rollout
   in `intervention.csv` starts ~day 86; only ~3 months of rollout is in-window,
   so cadence/priority effects are **timing-limited** — a monthly bolus only
   fits ~3 boluses.
6. **Data quality is county-specific.** The essentiality and SES CFs are only
   defensible on counties where good industry (ACS `C24030`) and income
   (`B19001`) data exist and the regenerated networks still reproduce the
   factual fit.
7. **Synthetic population RNG:** `custom_population.py` shuffles ages/households
   with the global NumPy seed; regenerating populations to add occ/SES columns
   **must reuse the same seed** or every downstream network + calibration
   shifts. Snapshot the current `populations/pop<FIPS>/` before regenerating.
8. **No silent error handling** (see §2.6). All new scripts fail loud and report
   the failing file / county / step. A CF number that produced fewer than the
   expected 30 iterations, or an all-CF roll-up missing a county, is an error,
   not a skipped row.
9. **Age outputs before the mapping fix are void** (§3 caveat 1) — do not cite
   any pre-fix `*_age_proportions.csv`.

---

## 8. Suggested build / run order

1. **CF-A** — age-mask fix, CF 12–17, per-age case incidence + bars.
   `Tasks/age_priority_cf_run.md`. No recalibration. **DONE** (2026-09-03; all 3
   counties in `results/*/TargetedCF/age_priority/`).
2. **CF-B Phase 1** — CF 18–23, essential-first vaccine + non-essential-only
   closure off the **existing** `Occupations` column.
   `Tasks/work_essentiality_cf_run.md`. No recalibration (CF7 byte-identity
   check). Ships with the sector-mix-skew caveat (§4).
3. **CF-C** (cadence). No recalibration. Cheap sweep.
4. **CF-B Phase 2** — ACS `C24030` re-source → regen populations + occ nets →
   check CF7 → recalibrate `…_essnet` only if drifted → re-run 18–23 (+ A/C if
   recalibrated). `Tasks/work_essentiality_cf_run.md` §2.9.
5. **CF-E + CF-F + CF-G** (§10) — household-level allocation, targeting-fidelity
   sweep, timing×targeting factorial. All reuse `log1p_rw2_l5_rmax5`, all reuse
   the CF-A + `analyze_spillover.py` code. This is the recommended first-paper
   extension set — do it **before** CF-D.
6. **CF-D0** (SES tag only) — quick null-ish result.
7. **CF-D1** (SES ↔ compliance) — recalibrate `…_ses`, re-run A/B/C(/E-G) on it
   for the appendix. Optional **D2**.
8. **CF-H / CF-I / CF-J** (§10) — booster-hold, network-hub, adaptive switch.
   Appendix / second-paper / reviewer-response material.

Each family: per-county artifacts under `results/<FIPS>/TargetedCF/<TAG>/`,
cross-county roll-ups under `results/TargetedCF_run/<TAG>/`, driven by a
`run_all_<fam>_cf.sh` + `analyze_<fam>_cf.py` pair in `targeted_cf/` copied from
the age-priority ones. Each with its own `Tasks/<fam>_cf_run.md`.

---

## 9. Files that change, per family

**Shared invariants (every family):** `main.py`, `LearnableParams`, the loss, the
calibration loop, **CF types 1–11**, `result_graphs/`, and the committed
baselines stay untouched. `abm_nets.py` `CF_POLICY_DESC` / `CF_FOLDER_MAP` / CF
loop range are **append-only** (12–17 age, 18–23 essentiality, 24+ next). Each
family gets a `targeted_cf/run_<fam>_cf.sh` + `run_all_<fam>_cf.sh` +
`analyze_<fam>_cf.py` (copied from the age-priority trio) and a
`Tasks/<fam>_cf_run.md`. `config.yaml` is patched-and-restored per run via a
`trap`. All new code fails loud (§2.6). Per-iteration uncertainty + CRN-paired
Δ in every analyzer (§7.4).

### CF-A — age priority (DONE)
| File | Change |
|---|---|
| `covid_abm/substeps/new_transmission/transition.py` | `CHILD/ADULT/ELDERLY_CODES`, `_age_group_masks` (diagnostic sites only — **not** `get_mean_agent_interactions`); `AGE_PRIORITY` / `CF_VACC_PRIORITY`; `apply_vaccines(..., agents_ages, priority)`; `logic_map` 12–17; `age_incidence_history` + `save_age_incidence_to_disk` |
| `abm_nets.py` | `CF_POLICY_DESC` + shared `CF_FOLDER_MAP` 12–17; CF loop `range(1, 18)`; collect + average `age_incidence` over the CF iters |
| `targeted_cf/` | `run_age_priority_cf.sh`, `run_all_age_priority_cf.sh`, `analyze_age_priority_cf.py`, `README.md` |
| `Tasks/age_priority_cf_run.md` | runbook |

### CF-B — work essentiality (`Tasks/work_essentiality_cf_run.md`; DONE — build in progress)
| File | Change |
|---|---|
| `covid_abm/substeps/new_transmission/transition.py` | `ESSENTIAL_SECTORS` / `NON_ESSENTIAL_SECTORS`, `CF_ESSENTIAL_VACC` / `CF_SELECTIVE_OCC` / `ESSENTIALITY_CF_TYPES`; lazy `_require_agent_essential` (`AGENT_OCC_CSV` env override, else glob `data/population_data/*/<FIPS>_population.csv`; fail loud) + `_essential_group_masks`; `apply_vaccines(..., essential_first)`; `apply_intervention_fast(..., essential_edge_mask)` selective closure (essential-endpoint edges kept; new `_crn_call_ids["interv_occ_selective"]`); `logic_map` 18–23; `essential_incidence_history` + `save_essential_incidence_to_disk` (only for `cf_type in {18..23}`) |
| `abm_nets.py` | `CF_POLICY_DESC` / `CF_FOLDER_MAP` 18–23; CF loop `range(1, 24)`; collect + average `essential_incidence` for `cf_type in {18..23}` |
| `targeted_cf/phase2/` (new) | `fetch_industry_c24030.py`, `patch_population.py`, `regen_occnets.py`, `assemble_essnet.py`, `cf7_check.py`, `build_essnet.sh`, `build_all_essnet.sh` — ACS C24030 → isolated patched population + `covid_output_essnet` scratch dir (household/school/rand symlinked); NOTHING committed overwritten |
| `targeted_cf/` | `run_essentiality_cf.sh`, `recalibrate_essnet.sh`, `run_all_essentiality_cf.sh`, `analyze_essentiality_cf.py` (`industry_mix.txt`, `essential_bars`, `targeted_vs_full_closure_ALL.csv` = `(CF7−CF19)/(CF7−CF3)` on the essnet nets), README |
| `Tasks/work_essentiality_cf_run.md` | runbook |

CF7 check (essnet nets): **all 3 counties RECALIBRATE** — the C24030 occ nets are
much less connected (healthcare no longer dominant), CF7 on the frozen
`log1p_rw2_l5_rmax5` params undershoots actual by 38–64 %. So each county gets a
`log1p_rw2_l5_rmax5_essnet` re-fit (`recalibrate_essnet.sh`, same knobs) →
cross-family comparison with CF-A is not apples-to-apples (note in the paper).

### CF-C / CF-D — see §5 / §6; not yet specced as own task files.

### CF-E … CF-J — see §10; not yet specced as own task files.

---

## 10. Candidate extension CFs for the paper *(low-cost, calibration-reuse, motivated by the A / B / spillover results)*

Added 2026-09-07 after the age-priority (§3) and household-spillover
(`targeted_cf/spillover_hook.py`, `Tasks/household_protection.md`) runs landed.
Everything here is **output-cheap**: factual `vaccines(t)` and NPI series are
untouched, the new mechanism fires only for `cf_type >= 25`, so all reuse
`log1p_rw2_l5_rmax5` (age-priority causal nets) and all are sweepable with no
gradient step.

### 10.1 What A / B / spillover already established (the paper's spine)

1. **Allocation is a real *transmission* lever, not just a mortality-shifter.**
   CF-A paired Δ: working-age-first + open NPI (CF14/15) adds **+14–16 k**
   infections vs uniform in 36003; elderly-first + closed NPI (CF16/17) cuts the
   total. (`results/TargetedCF_run/age_priority/HEADLINE.txt`.)
2. **But the *indirect* household-spillover benefit of vaccination is small and
   allocation-insensitive.** hSAR protection gap under uniform allocation (CF7)
   is only **+0.14 … +0.33 pp** across the 3 counties, and every age-priority
   CF's per-iteration paired Δ of that gap is **within ±0.35 pp** (mostly ±0.10).
   Elderly-first does **not** manufacture more household protection for
   non-elderly cohabitants than working-age-first. The value of targeting in
   this model is **direct** (the vaccinee), essentially not spillover.
   (`results/TargetedCF_run/spillover/HEADLINE.txt`.)
3. **Rollout *timing* relative to the wave dominates the allocation choice for
   fast-epidemic scenarios.** CF14/15 push the epidemic to *before* the day-~85
   rollout, so doses land after the peak and the allocation label barely matters.

The extension CFs below stress-test **when** and **how precisely** doses are
delivered, and whether **explicit household / network targeting** can produce the
spillover that individual-level targeting doesn't.

### 10.2 CF-E — Household-level vs individual-level allocation  *(Easy — the direct spillover follow-up)*

**Idea.** Replace uniform-over-individuals dose selection with **whole-household**
allocation: once a household is selected, vaccinate all its eligible members
before moving on. Variants:

| CF | household selection order |
|----|---------------------------|
| 25 | random household order (household bolus, coverage held equal) |
| 26 | **shield-the-vulnerable** — households with ≥1 elderly member first |
| 27 | **largest households first** (max onward transmission) |
| 28 | essential-worker households first (needs CF-B `_require_agent_essential`) |

**Why it's interesting (Nature framing).** Observation 2 says individual
allocation produces almost no household protection. CF-E asks the policy question
head-on: *does concentrating doses within households buy the spillover that
spreading them across households does not?* "Individual-optimal vs
household-optimal vaccine allocation" is a clean, decision-relevant contrast, and
either answer is a result.

**Model change.** New `apply_vaccines` priority mode keyed on the
household-membership tensor **already built for the spillover hook**
(`SpilloverHook.agent_to_hh` / `household_members`) — promote it to a shared
read-only loader in `transition.__init__` per §2.1. New
`_crn_call_ids["vaccine_hh"]`.

**Analysis.** Re-use `analyze_spillover.py` unchanged — CF-E is exactly the
scenario where `hSAR_vacc` should fall if the mechanism has any force. Report the
paired Δ protection-gap **and** the total-infection Δ side by side.

**Caveat.** Household bolus vaccinates correlated agents on one step → sharper
S-depletion kinks (CF mode uses plain `torch.bernoulli`, numerically fine).
Coverage is equal by construction; assert `Σ doses` matches factual after the
susceptible-pool clamp.

### 10.3 CF-F — Targeting-fidelity / leakage sweep  *(Trivial — turns the "upper-bound" caveat into a result)*

**Idea.** Real targeted rollouts are leaky. Blend the priority score with a
uniform draw: `score = θ·priority + (1−θ)·uniform`; sweep
`θ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` for elderly-first and essential-first.
Encode policy × θ in `cf_type`, or pass θ by env for a pure sweep.

**Why it's interesting.** §1's corollary — every allocation CF is an *upper
bound* because R(t) was fit to ≈uniform allocation. CF-F **quantifies how much
theoretical benefit survives realistic leakage**: a headline number ("elderly-
first retains X % of its infection reduction at θ = 0.7") instead of a
hand-wave, and directly comparable to the observed rollout-equity gradient (CDC
vaccination-by-SVI curves, §6).

**Model change.** One line in `apply_vaccines` (convex-combine the
Gumbel-perturbed priority score with a uniform score). Reuse calibration. This
produces a *curve*, not a point.

### 10.4 CF-G — Timing × targeting factorial ("who" vs "when")  *(Easy — motivated by CF14/15)*

**Idea.** 3 × 3: rollout start {−14 d faster, factual, +28 d delayed} × priority
{uniform, elderly-first, working-age-first}, **cumulative dose count held fixed**
(CF-C `reshape_doses` machinery, §5).

**Why it's interesting.** Observation 3. The factorial cleanly separates the
**timing main effect, the allocation main effect, and their interaction** — e.g.
"a 2-week-faster uniform rollout beats a factual-speed elderly-first rollout"
(or not) is concrete and actionable.

**Model change.** `reshape_doses(series, start_delay)` (§5) × the CF-A priority
hook. Reuse calibration. Dose-conservation assert per §5 caveat 1.

**Caveat.** Only ~3 months of rollout is in-window (§7.5); the +28 d arm leaves
very little rollout — report the truncation and treat it as the realistic
delay ceiling.

### 10.5 CF-H — Waning-aware allocation / booster-hold  *(Easy — exploits the VACCINATED→S feature)*

**Idea.** `VACCINATED_TO_SUSCEPTIBLE_TIME = 90`, so a dose given on day 86 has
fully waned by day 176. CF: hold back a fraction `f` of daily doses as a reserve,
then at day (first-dose + 90) deliver them as **boosters to the earliest
vaccinees** (reset their waning clock) rather than as first doses to new people.
Sweep `f ∈ {0, 0.15, 0.30}`.

**Why it's interesting.** "First doses first vs hold second doses" was a real
UK-vs-US fight in early 2021. The model has exactly the mechanism (independent
vaccine waning) to adjudicate it in a calibrated setting; almost no ABM paper
exploits a waning-state for booster timing.

**Model change.** Track per-agent vaccination day (the spillover hook already
does this via `cummax` over the `VACCINATED` stage) → a re-vaccination branch in
`apply_vaccines` that targets near-expiry `VACCINATED` agents and resets
`agents_next_stage_times`. Reuse calibration (factual = all first doses). New
`_crn_call_ids["booster"]`.

**Caveat.** Waning mode / times are a *config* choice, not calibrated — run the
booster CF at `VACCINATED_TO_SUSCEPTIBLE_TIME ∈ {60, 90, 150}` as a sensitivity
band; the sign of the result can flip with the waning speed.

### 10.6 CF-I — Network-hub vaccination  *(Medium — classic result, calibrated-ABM novelty)*

**Idea.** Vaccinate the **highest-contact agents first** — top-k by
`agents_mean_interactions` (already in `state['agents']['citizens']`). Compare
"vaccinate the hubs" against age- and essential-targeting on total infections
and on the epidemic peak.

**Why it's interesting.** Degree-targeted immunization is textbook network
epidemiology but is rarely tested inside a *mobility-calibrated* county ABM with
empirical interaction distributions, competing head-to-head with deployable
demographic policies. Hub-targeting winning big ⇒ "the deployable policies leave
X on the table"; hub-targeting *not* winning ⇒ "scale-free intuitions don't
transfer to realistic contact structure". Both publishable.

**Model change.** Thread `agents_mean_interactions` into `apply_vaccines`
(currently not passed) + a `priority="degree"` mode. Reuse calibration. New
`_crn_call_ids["vaccine_deg"]`.

**Caveat.** `mean_interactions` is a static per-agent scalar from network
construction, not a dynamic contact count — say the "hub" definition is
structural, and frame CF-I as a **non-deployable benchmark ceiling** (you can't
observe contact rate at rollout time), the way CF-24 is the full-closure ref for
CF-B.

### 10.7 CF-J — Reactive / state-switching allocation  *(Medium — novel policy question)*

**Idea.** Switch strategy mid-rollout on an observable trigger: vaccinate
**working-age while the epidemic is growing** (blunt transmission), then switch
to **elderly once cases turn over** (protect the vulnerable through the tail).
Trigger = sign of the smoothed 7-day `daily_infected` slope, or R̂(t) crossing 1.

**Why it's interesting.** All of CF-A is *static*. One adaptive switch is a small
change that maps onto how rollouts were actually re-prioritised in 2021;
"adaptive beats the best static policy by X" is a genuinely new claim for this
model class.

**Model change.** State-dependent branch in `apply_vaccines` reading a smoothed
`daily_infected` slope from state. Reuse calibration. **CRN caveat:** the *switch
day* becomes scenario-dependent → past the switch this is no longer a
matched-recipient comparison, only a shared RNG stream (same wording as §5
caveat 2).

### 10.8 Recommended subset for a first submission

**CF-E + CF-F + CF-G.** One coherent story off the reused calibration, all
sweepable, all reusing the CF-A + spillover analysis code:

> *The epidemiological value of targeted allocation is **direct, not spillover**
> (E); it **degrades under realistic targeting leakage** (F); and it is
> **second-order to rollout timing** (G).*

CF-H and CF-I are strong appendix / second-paper material. CF-J is the adaptive
comparator to add if a reviewer asks for one. None of E–J needs a recalibration;
if any later variant does (e.g. an SES-coupled version), fold it into the CF-D
sub-study (§6 caveat 4) rather than fragmenting the main baseline.
