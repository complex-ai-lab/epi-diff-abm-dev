# Task: Add a VACCINATED State with Independent Waning

Repository: `https://github.com/complex-ai-lab/epi-diff-abm-dev`

Inspect the existing implementation before editing. Focus on:

- `covid_abm/substeps/new_transmission/transition.py`
- `covid_abm/yamls/config.yaml`

Make minimal changes and preserve existing behavior.

> **Status:** revised 2026-08-29 after a code read. The revision adds the
> `SFInfector` extension (§2b), the `update_stages` fixed-point term (§2a), the
> same-step vaccinate/expose tie-break (§3b), the `proportion_history` handling
> that keeps `abm_nets.py` untouched (§2c), and a concrete validation protocol
> (§9). It also notes this task **extends the immunity-waning feature already in
> the repo** (`plans/Task_ Add Configurable Immunity Waning to the ABM.md`,
> `waning_tests/`), it does not start from a clean slate.

## Goal

Add a distinct `VACCINATED` state, initially treating vaccinated individuals
exactly like `RECOVERED` for infection susceptibility:

```text
S → VACCINATED → S          (vaccine immunity, its own duration/rate)
I → RECOVERED  → S          (natural immunity, its own duration/rate)
```

Vaccination counts still come from the existing intervention/vaccination
mechanism. The two waning mechanisms have **independent** durations/rates.

## 1. Configuration (`covid_abm/yamls/config.yaml`, `simulation_metadata`)

Current keys (from the waning feature already merged): `IMMUNITY_WANING_MODE`,
`RECOVERED_TO_SUSCEPTIBLE_TIME: 100`, `WANING_RATE: 0.01`.

Target state:

```yaml
IMMUNITY_WANING_MODE: "none"        # "none" | "fixed" | "stochastic"

RECOVERED_TO_SUSCEPTIBLE_TIME: 100
VACCINATED_TO_SUSCEPTIBLE_TIME: 150

RECOVERED_WANING_RATE: 0.01
VACCINATED_WANING_RATE: 0.00667

VACCINATED_VAR: 5                   # new disease-stage id (S=0,E=1,I=2,R=3,D=4)

# WANING_RATE: 0.01                 # keep as a deprecated alias, see below
```

- `RECOVERED_WANING_RATE` replaces `WANING_RATE`. **Keep `WANING_RATE` in the
  file** and have `transition.py` read
  `RECOVERED_WANING_RATE`, falling back to `WANING_RATE`, so
  `waning_tests/set_waning_config.py` and the existing `waning_tests/` scripts
  keep working unchanged.
- Do **not** use the existing `k` parameter for any of this.

### 1b. `SFInfector` must be extended (required — otherwise a hard crash)

`environment.SFInfector` currently has `shape: [5]` and is indexed by disease
stage inside `NewTransmission._lam()` (`SFInfector[x_j[:, 1].long()]`). A
vaccinated agent carrying stage `5` would index out of bounds.

Change it to `shape: [6]` and append a trailing `0.0` (vaccinated agents are
not infectious):

```yaml
SFInfector:
  shape: [6]
  value: [0.0, 0.33, 0.72, 0.0, 0.0, 0.0]
```

`SFSusceptibility` is indexed by **age group**, not stage — leave it unchanged.

## 2. Add the VACCINATED state in `transition.py`

### 2a. Constant + stage progression

- Read `self.VACCINATED_VAR` in `__init__` alongside the other `*_VAR` reads
  (`self.config["simulation_metadata"].get("VACCINATED_VAR", 5)`).
- In `update_stages()`, `VACCINATED` must be a **fixed point** of
  `stage_progression`, exactly like `SUSCEPTIBLE`, `RECOVERED`, `MORTALITY`:

  ```python
  stage_progression = ... \
      + (current_stages == self.VACCINATED_VAR) * self.VACCINATED_VAR \
      + ...
  ```

  Without this term a vaccinated agent matches no branch and is silently reset
  to `S` every timestep.
- `newly_exposed_today` already gates on `current_stages == SUSCEPTIBLE_VAR`, so
  `VACCINATED` agents cannot be exposed. `agents_infected_index`
  (`> S and < R`) already excludes stage 5. No change needed in either.

### 2b. Do not repurpose `RECOVERED_VAR`

Keep `RECOVERED_VAR` as is. `VACCINATED_VAR` is a new, separate id.

### 2c. Counting / proportions without touching `abm_nets.py`

`abm_nets.py:~193` reads `NewTransmission.proportion_history` as a fixed 5-wide
list (`t, S, E, I, R, D`) and `abm_nets.py:~356` reads `age_proportion_history`
as a fixed 16-wide list. Point 8 forbids modifying `abm_nets.py`. Therefore:

- **Keep `proportion_history` 5-wide.** Add a parallel
  `self.vaccinated_proportion_history` list of `[t, vaccinated_fraction]`,
  reset in the same place `proportion_history` is reset (`t == 0`).
- In `save_proportions_to_disk()`, after building the existing dataframe, merge
  the vaccinated fraction in as an extra `vaccinated` column (join on `t`) and
  write the single CSV. Nothing reads this CSV back with a strict schema
  (`plot_waning_tests.py` accesses columns by name), so this is safe.
- `save_proportions_to_disk2()` (counterfactual path, `num_iterations > 1`):
  add the `vaccinated_{epoch}` column the same way.
- `get_age_stage_proportions()` / `age_proportion_history`: leave structurally
  unchanged for now (vaccinated agents are simply absent from the per-age
  S/E/I/R/D buckets). Only affects counterfactual age-proportion outputs, which
  this task must not restructure. Note it in the code comment.

Because `S + E + I + R + D` will no longer sum to 1 once vaccination starts
(the remainder is the vaccinated fraction), add a one-line comment saying so.

## 3. Vaccination destination: S → VACCINATED

### 3a. `apply_vaccines()`

Change only the destination:

```python
stage_delta = (self.VACCINATED_VAR - self.SUSCEPTIBLE_VAR) * vax_mask
```

Leave the selection mechanism, the vaccine counts from intervention data, the
straight-through mask, and everything else untouched. Do **not** add a
vaccination-rate parameter.

### 3b. Same-step tie-break (required)

`newly_exposed_today` is computed from **pre-vaccination** stages, and
`apply_vaccines()` runs afterwards. An agent selected for both vaccination and
exposure on the same timestep currently becomes stage `E + R = 4` (DEAD) under
the old S→R code — a latent bug — and would become `E + VACCINATED = 6`
(undefined) under S→VACCINATED.

Vaccination takes precedence. Immediately after `apply_vaccines()`:

```python
just_vaccinated = (pre_vaccination_stages == self.SUSCEPTIBLE_VAR) & \
                  (current_stages == self.VACCINATED_VAR)
newly_exposed_today = newly_exposed_today * (~just_vaccinated)
```

where `pre_vaccination_stages` is captured just before the `if with_vacc:`
block. This also keeps the `IMMUNITY_WANING_MODE="none"` infected trajectory
bit-identical to the pre-change baseline (see §9.1).

## 4. VACCINATED susceptibility

`VACCINATED` behaves exactly like `RECOVERED` w.r.t. infection: zero
susceptibility, only `SUSCEPTIBLE → EXPOSED` is possible. No reduced-efficacy
model yet. (Already satisfied by the existing `newly_exposed_today` gate once
§2a and §3b are in.)

## 5. Fixed waning (`IMMUNITY_WANING_MODE == "fixed"`)

Independent deterministic timers, each set **once** when the agent enters the
state, never reset on later timesteps:

```text
RECOVERED  --RECOVERED_TO_SUSCEPTIBLE_TIME--> SUSCEPTIBLE
VACCINATED --VACCINATED_TO_SUSCEPTIBLE_TIME--> SUSCEPTIBLE
```

Implementation mirrors the existing fixed-mode `R → S` block:

- `newly_recovered  = (pre_vaccination_stages != R) & (updated_stages == R)`
  → set `updated_next_stage_times` to `t + RECOVERED_TO_SUSCEPTIBLE_TIME`.
- `newly_vaccinated = (pre_vaccination_stages != V) & (updated_stages == V)`
  → set `updated_next_stage_times` to `t + VACCINATED_TO_SUSCEPTIBLE_TIME`.
- Fire: `waned = ((updated_stages == R) | (updated_stages == V)) &
  (updated_next_stage_times <= t)` → set those agents to `S` and their timer to
  the existing `WANING_NO_TRANSITION_TIME` sentinel.

A just-recovered / just-vaccinated agent has timer `> t`, so it cannot wane on
the same timestep.

## 6. Stochastic waning (`IMMUNITY_WANING_MODE == "stochastic"`)

No deterministic timers. Each timestep, independent constant-hazard draws:

```python
recovered_probability  = 1.0 - math.exp(-self.RECOVERED_WANING_RATE)
vaccinated_probability = 1.0 - math.exp(-self.VACCINATED_WANING_RATE)
```

(Use `math.exp` to match the existing code; the rates are Python floats.)

- `already_recovered  = (pre_vaccination_stages == R) & (updated_stages == R)`
- `already_vaccinated = (pre_vaccination_stages == V) & (updated_stages == V)`
- One `draw = torch.rand_like(updated_stages.float())`;
  `waning_mask = (already_recovered & (draw < recovered_probability)) |
  (already_vaccinated & (draw < vaccinated_probability))`
  → set masked agents to `S`.

Using `pre_vaccination_stages` for the `already_*` checks guarantees an agent
that became `R`/`V` this timestep is not eligible until the next one.

> Known limitation (carried over from the waning feature): the per-step
> `torch.rand_like` draw pulls from the global RNG stream, so `stochastic` mode
> desyncs the stream from `none`/`fixed` from `t = 0`. Acceptable here — the
> validation in §9 uses recalibration, not a frozen-parameter isolation.

## 7. None mode (`IMMUNITY_WANING_MODE == "none"`)

Existing behavior preserved, except vaccination now yields `VACCINATED` instead
of `RECOVERED`. No waning of either state.

## 8. Do not modify

`abm_nets.py`, counterfactual definitions/types, calibration architecture,
`LearnableParams`, calibration loss, existing intervention/vaccination input,
the `k` parameter, `RECOVERED` transmission susceptibility — **unless inspection
shows it is strictly necessary**. Known-necessary exceptions already identified:
`config.yaml` `SFInfector` shape (§1b) and the new keys (§1). `abm_nets.py`
stays untouched via §2c.

## 9. Validation

All artifacts go in **`vaccinated_test/`** (sibling of `waning_tests/`). County
`01031`, `SEED=42`, env `.venv_ds`, `.env` `NETWORKS_DIR` as configured. Reuse
`waning_tests/`-style tooling:

- `vaccinated_test/set_vacc_config.py` — targeted in-place patch of
  `IMMUNITY_WANING_MODE`, `RECOVERED_TO_SUSCEPTIBLE_TIME`,
  `VACCINATED_TO_SUSCEPTIBLE_TIME`, `RECOVERED_WANING_RATE`,
  `VACCINATED_WANING_RATE` (no YAML re-dump).
- `vaccinated_test/run_vacc_tests.sh` — backs up `config.yaml` and the
  `result_graphs/01031/.../metro_0/{training_proportions.csv,
  calibrated_params.txt,training_loss.csv}` baseline, patches only the waning
  keys per run, runs `python3 main.py 01031` (full 251-epoch recalibration —
  **recalibration is expected/accepted here**), copies each run's
  `training_proportions.csv` (now with the `vaccinated` column) to
  `vaccinated_test/data/<label>.csv`, restores the originals on exit.
- `vaccinated_test/plot_vacc_tests.py` — reads `data/*.csv`, writes the PNGs
  below and prints a PASS/FAIL line per test.
- `vaccinated_test/wane_rate_check.py` — 3 short **forward** sims (frozen `none`
  params, no recalibration) that dump the per-step
  `NewTransmission.waning_events_history` (eligible R / VACCINATED pool and how
  many of each waned) and check the empirical daily hazard `Σ waned / Σ pool`
  against the configured `1 − exp(−rate)` for each state independently.

### 9.1 Test A — `none` mode vs. the pre-change baseline

Run: `IMMUNITY_WANING_MODE=none`, new code, full calibration → `data/none.csv`.

Reference: `waning_tests/data/none_baseline_reference.csv` (the seed-42 `none`
baseline generated *before* the VACCINATED state existed — already on disk).

**Pass:** `max |Δ infected fraction|` between the two ≈ 0
(`< 1e-3`; expect `0` up to the documented GNN-scatter nondeterminism). Also
assert the run's `vaccinated` column is `> 0` after the vaccine rollout day and
that its `recovered` column no longer contains the vaccine-driven jump seen
before. Rationale: vaccinated agents are immune identically to recovered and
`apply_vaccines` consumes the same RNG regardless of target stage, so the
infection trajectory must not move — only the *labelling* of protected agents
changes.

Artifact: `vaccinated_test/none_vs_baseline.png` (infected overlay + the Δ),
plus `vaccinated_test/none_compartments.png` (S/E/I/R/D/vaccinated stacked or
line, showing vaccinated is now its own compartment).

### 9.2 Test B — fixed mode, independent durations

Runs (natural immunity `R ∈ {40, 60, 80}` d — shifted down from the
`waning_tests/` 60/80/100 sweep so some vaccine waning lands in-window;
vaccine immunity = 1.5× longer):

| label | `RECOVERED_TO_SUSCEPTIBLE_TIME` | `VACCINATED_TO_SUSCEPTIBLE_TIME` |
|-------|------|------|
| `fixed_40` | 40 | 60  |
| `fixed_60` | 60 | 90  |
| `fixed_80` | 80 | 120 |

**Pass:**
1. The `infected` curves are ordered — shorter natural immunity → larger
   late-window (last 40 d) mean infected (`R40 ≥ R60 ≥ R80`).
2. `recovered` visibly wanes in-window in every run (half-fall day exists).
3. **The `VACCINATED → S` onset day (first day the run's `vaccinated` fraction
   drops measurably below the `none` run's) matches `≈ 86 + V`** — i.e. the
   vaccine timer is on its own independent schedule. `V=60 → ~day 146`,
   `V=90 → ~day 176`, `V=120 → not in window`.
4. No agent waned on the same timestep it entered R/V (`pre_vaccination_stages`
   guard).

> **Window limitation:** the vaccine rollout in
> `populations/pop01031/intervention.csv` starts at **day 86** of a 182-day
> run, so only `V=60` shows a substantial `VACCINATED → S` decay before the
> horizon. That is enough for the onset-day check (criterion 3); the full
> vaccine-decay curve and the rate check live in test C.

Artifacts: `vaccinated_test/fixed_infected.png`,
`vaccinated_test/fixed_compartments.png` (two panels: `recovered` and
`vaccinated` fraction, each vs the no-waning `none` run).

### 9.3 Test C — stochastic mode, independent rates

Runs (natural-immunity rates reuse the `waning_tests/` stochastic sweep;
vaccine-immunity rate = rate ÷ 1.5, i.e. ~1.5× longer mean immunity):

| label | `RECOVERED_WANING_RATE` | `VACCINATED_WANING_RATE` |
|-------|------|------|
| `stochastic_0.005` | 0.005 | 0.003333 |
| `stochastic_0.010` | 0.010 | 0.006667 |
| `stochastic_0.020` | 0.020 | 0.013333 |
| `stochastic_0.040` | 0.040 | 0.026667 |

**Pass:**
1. `infected`: the top natural rate (0.04) separates clearly; the retained
   `vaccinated` fraction (vs the `none` run) is monotonic in the rate (higher
   rate → less retained). (The lower `infected` curves bunch within calibration
   noise — same behaviour as `waning_tests/`.)
2. **`wane_rate_check.py`:** for each rate the empirical daily hazard
   `Σ waned / Σ pool` matches `1 − exp(−rate)` for `R` and for `V`
   **independently** (< a few % off), and `hazard_R / hazard_V ≈ 1.5`.
   Measured: R0.02 → emp_R 0.0196 / emp_V 0.0132 / ratio 1.49; R0.04 →
   emp_R 0.0396 / emp_V 0.0264 / ratio 1.50.
3. Same-timestep-waning guard holds (`pre_vaccination_stages`).

Artifacts: `vaccinated_test/stochastic_infected.png`,
`vaccinated_test/stochastic_compartments.png`,
`vaccinated_test/wane_rate_check.png`.

### 9.4 No isolated / frozen-parameter test

Not required for this task (unlike `frozen_cal_params_iso_wane/`).
Recalibration per configuration is acceptable.

### Deliverable

At the end, report the exact files changed and briefly describe the
implementation, and summarize the PASS/FAIL of Tests A–C with the generated
PNGs.
