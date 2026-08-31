# VACCINATED-state validation (`vaccinated_test/`)

Validates the distinct `VACCINATED` disease stage with waning **independent** of
natural (`RECOVERED`) immunity. Task spec:
`plans/Task_ Add a VACCINATED State with Independent Waning.md`.

## What changed in the model

| File | Change |
|------|--------|
| `covid_abm/yamls/config.yaml` | `simulation_metadata`: added `VACCINATED_VAR: 5`, `VACCINATED_TO_SUSCEPTIBLE_TIME: 150`, `RECOVERED_WANING_RATE: 0.01`, `VACCINATED_WANING_RATE: 0.00667` (`WANING_RATE` kept as a deprecated alias). `environment.SFInfector` resized `[5] -> [6]` (trailing `0.0`) so stage 5 can be indexed. |
| `covid_abm/substeps/new_transmission/transition.py` | new `VACCINATED_VAR`; `update_stages` gets a `VACCINATED` fixed-point term; `apply_vaccines` moves `S -> VACCINATED` (was `S -> RECOVERED`); a same-step vaccinate/expose tie-break (vaccination wins); fixed & stochastic waning blocks handle `R -> S` and `VACCINATED -> S` on independent timers/rates; `vaccinated` fraction tracked in a parallel list and merged as an extra column in `save_proportions_to_disk` (keeps `proportion_history` 5-wide, `abm_nets.py` untouched). |

`abm_nets.py`, the calibration architecture, `LearnableParams`, the loss, the
counterfactual definitions, and `RECOVERED` transmission susceptibility are
**unchanged**.

## Run it

```bash
# .venv_ds env; SEED=42; county 01031
PYTHON=~/.conda/envs/.venv_ds/bin/python \
  bash vaccinated_test/run_vacc_tests.sh 01031 all
python3 vaccinated_test/plot_vacc_tests.py
```

`run_vacc_tests.sh` patches only the waning keys per run, runs the full
251-epoch recalibration (`main.py 01031`), collects
`training_proportions.csv` (now with a `vaccinated` column) into
`vaccinated_test/data/<FIPS>/<label>.csv` plus the aligned
simulated-vs-actual daily cases (`generated_factual.csv`) into
`vaccinated_test/data/<FIPS>/<label>_cases.csv`, and restores `config.yaml` +
the `result_graphs/<FIPS>/.../metro_0` baseline files on exit.

All per-county artifacts live under a `<FIPS>` subfolder: input trajectories in
`vaccinated_test/data/<FIPS>/`, plots in `vaccinated_test/results/<FIPS>/`.

8 runs (~15-20 min GPU each): `none` + `fixed_{60,80,100}` +
`stochastic_{0.005,0.010,0.020,0.040}`.

## Tests

### A - `none` mode vs. the pre-change baseline

`IMMUNITY_WANING_MODE=none`, new code. Compared against
`waning_tests/data/<FIPS>/none_baseline_reference.csv` (seed-42 `none` run from
*before* the VACCINATED state existed).

**Pass:** `max |Δ infected fraction| < 1e-3` (expected ≈ 0 — vaccinated agents
are immune identically to recovered and `apply_vaccines` consumes the same RNG
regardless of target stage, so only the *label* of protected agents changes),
and the `vaccinated` column is `> 0` after the rollout day.

Artifacts: `none_vs_baseline.png`, `none_compartments.png`.

### B - fixed mode, independent durations

| label | `RECOVERED_TO_SUSCEPTIBLE_TIME` | `VACCINATED_TO_SUSCEPTIBLE_TIME` |
|-------|------|------|
| `fixed_40` | 40 | 60  |
| `fixed_60` | 60 | 90  |
| `fixed_80` | 80 | 120 |

Vaccine immunity is 1.5× the natural value. (Shifted down from the
`waning_tests/` 60/80/100 sweep so that some `VACCINATED → S` transitions land
inside the 182-day window — the vaccine rollout in
`populations/pop01031/intervention.csv` starts at day 86.)

**Pass:** the three `infected` curves are distinct and ordered
(`60 ≥ 80 ≥ 100` late-window mean infected); within each run the `vaccinated`
compartment decays on a visibly longer timescale than `recovered`
(half-fall-day gap `> 0`).

Artifacts: `fixed_infected.png`, `fixed_compartments.png`.

### C - stochastic mode, independent rates

| label | `RECOVERED_WANING_RATE` | `VACCINATED_WANING_RATE` |
|-------|------|------|
| `stochastic_0.005` | 0.005 | 0.003333 |
| `stochastic_0.010` | 0.010 | 0.006667 |
| `stochastic_0.020` | 0.020 | 0.013333 |
| `stochastic_0.040` | 0.040 | 0.026667 |

Natural-immunity rates reuse the `waning_tests/` stochastic sweep; vaccine rate
= rate ÷ 1.5.

**Pass:** the `vaccinated` fraction retained vs. the `none` run is monotonic in
the rate (higher V rate → less retained); the top natural rate (0.04) separates
the `infected` curve clearly (the lower rates bunch within calibration noise,
same as `waning_tests/`).

Artifacts: `stochastic_infected.png`, `stochastic_compartments.png`.

### Rate check - `wane_rate_check.py`

The compartment fractions can't be used to fit a decay rate (vaccination inflow
runs the whole window, recovered inflow tracks the epidemic). Instead,
`NewTransmission.waning_events_history` logs, per step, the eligible R /
VACCINATED pool and how many of each waned. `wane_rate_check.py` does 3 short
**forward** sims (frozen `none` params, no recalibration) and checks the
empirical daily hazard `Σ waned / Σ pool` against the configured
`1 − exp(−rate)` for each state independently, and that the R/V ratio ≈ 1.5.

```bash
python3 vaccinated_test/wane_rate_check.py   # -> results/<FIPS>/wane_rate_check.png
```

## No isolated / frozen-parameter test

Unlike `waning_tests/frozen_cal_params_iso_wane/`, recalibration per config is
accepted here.

## Files

| File | Purpose |
|------|---------|
| `set_vacc_config.py` | targeted in-place patch of the 5 waning keys in `config.yaml`. |
| `run_vacc_tests.sh` | orchestrates the 8 recalibration runs, backs up/restores originals. |
| `plot_vacc_tests.py` | reads `data/<FIPS>/*.csv`, writes the infected + compartment PNGs + `sim_vs_actual` plots to `results/<FIPS>/`, prints PASS/FAIL. |
| `wane_rate_check.py` | 3 short forward sims + per-state empirical-hazard check (`results/<FIPS>/wane_rate_check.png`). |
| `data/<FIPS>/` | collected `training_proportions.csv` (`<label>.csv`) + `generated_factual.csv` (`<label>_cases.csv`) per run + `waning_events_*.csv`. |
| `results/<FIPS>/` | generated plots. |
| `_backup/` | pristine `config.yaml` + metro_0 baseline result files (restored automatically). |
