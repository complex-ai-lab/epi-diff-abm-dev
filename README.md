# Epidemic Differentiable ABM — single-county test package (`test-ABM` branch)

This branch is a minimal copy of the COVID-19 ABM for **one county, 01031 (Coffee County, AL)**,
2020-10-26 → 2021-04-25. It has everything needed to **calibrate** the model and **run** it, including the
counterfactuals. It uses the repo layout of `b5a6372` (Facundo Yan's last commit) and the model code from
`preetom-dev`. The recipe is the P0 / `cft_poisson_lam25` setup:
- P0 population and networks
- per-layer contact weights
- 14-day burn-in
- Poisson loss with a spline on log beta
- fixed immunity waning (R→S 60 days, V→S 90 days)
- a separate vaccinated (V) stage with the observed age-phased (65+ first) vaccine allocation
- common random numbers across counterfactuals

## Quick start

```bash
pip install -r requirements.txt   # Python 3.11, CUDA GPU recommended

bash calibrate.sh                 # ~25 min on one A100 -> result_graphs/01031/.../calibrated_params.txt
bash run.sh                       # factual run (CF 7)                 -> results/01031/run/
```

`EPOCHS=2 bash calibrate.sh` and `CF_TYPES=7 CF_ITERS=2 bash run.sh` are quick smoke tests. Any key in
`run_settings` can be overridden from the shell this way. Use `PY=/path/to/python` to choose the interpreter.

## Config — `covid_abm/yamls/config.yaml`

The file starts with the **epidemic parameters**:

| key | meaning |
|---|---|
| `EXPOSED_TO_INFECTED_TIME` | E → I, days (3) |
| `INFECTED_TO_RECOVERED_TIME` | I → R, days (5) |
| `BETA_MIN`, `BETA_MAX` | box for the calibrated weekly transmission rate beta (0.1, 5) |
| `BETA_INIT` | beta at the start of calibration (2.5) |
| `BETA_BURN_IN` | fixed beta during the 14 burn-in days (1.0) |
| `IMMUNITY_WANING_MODE`, `RECOVERED_TO_SUSCEPTIBLE_TIME`, `VACCINATED_TO_SUSCEPTIBLE_TIME`, `*_WANING_RATE` | waning of natural and vaccine immunity |

Next comes `run_settings`. `main.py` exports these as the environment variables that `abm_nets.py` and
`transition.py` read, under the same names as in `run_p0_county.sh`. They cover:
- the networks directory, seed and layer weights
- `I0_MAX`, the loss and the spline settings, and the number of epochs
- the counterfactual types, iterations and layer attribution

The `BETA_*` keys map to `R_SPLINE_R_MIN`, `R_MAX`, `R_SPLINE_INIT_R` and `BURN_IN_R`. After that come the
usual `simulation_metadata`, `state` and `substeps` sections. Nothing rewrites the config file any more (see
the change list at the end).

## Layout

```
main.py, abm_nets.py, constants.py   entry point, calibration loop, counterfactual loop
calibrate.sh, run.sh                 the two launchers
covid_abm/                           model (transition.py = disease dynamics, spline_basis.py) + yamls/config.yaml
agent_torch/                         AgentTorch framework
populations/pop01031/                age.pickle, disease_stages.csv, intervention.csv (196 days incl. 14-day
                                     burn-in, school/occ flags, vaccines, vaccines_65plus), burnin_days.txt
data/processed_data/01031/202010-202104/daily_data.csv      observed cases (fit uses cases_singular)
data/population_data/AL_population_data/01031_population.csv   P0 agents (occupation, household) for CF 18-28
data/networks/covid_output_p0/01031/mobility_networks/      household, school, daily occnets / randnets
```

The networks are compressed `.npz` edge lists. They hold the same edges, in the same order, as the P0 networkx
`.pkl` files; `transition.py` loads the `.npz` when it exists and falls back to `.pkl` otherwise.

## Outputs

`calibrate.sh` writes to `result_graphs/01031/202010-202104/0.0005_3_5_True_True_False_metro_0/`:
- `calibrated_params.txt`: 28 rows — 26 weekly beta values, then the initial infected fraction, then k
- `r_daily.txt`, `spline_coef.txt`, `spline_meta.json` and `training_loss.csv`

`run.sh` writes to `results/01031/run/`:

| file | contents |
|---|---|
| `beta_daily.csv` | `day, date, phase, week, beta` — beta each simulated day, including the burn-in |
| `seirv_daily.csv` | `day, date, S, E, I, R, V` — daily agent counts in the factual run (mean over iterations) |
| `cases_sim_vs_observed.csv` | `day, date, simulated_cases, simulated_cases_sd, observed_cases` — reported cases on the 168 scored days |
| `cf_raw_<type>.csv`, `counterfactual_data<type>*.csv`, `*.png` | per counterfactual type, as before |

`day` is the simulation step, so days 0–13 are the burn-in and day 14 is 2020-10-26. The three summary CSVs
come from CF 7 (the factual run), so `CF_TYPES` must include 7.

## Changes relative to `preetom-dev`

* `config.yaml` now begins with the `epidemic` and `run_settings` blocks. `main.py` exports them as environment
  variables, so no shell wrapper is needed for the recipe.
* `agent_torch/core/dataloader.py` keeps the population path and agent count in memory. Before, it rewrote
  `config.yaml` (sorted, comments lost) on every run.
* `abm_nets.py` writes the 3 summary CSVs at the end of CF 7.
* `transition.py` reads `.npz` networks.
* `main.py` is the last committed version (`a2ff0f5`). The uncommitted `MODEL_PKG` / `POP_SUFFIX` horizon
  switches are not included.
* Data preparation, network generation, the horizon/targeted-CF/analysis code, other counties and the Slurm
  scripts are not on this branch.
