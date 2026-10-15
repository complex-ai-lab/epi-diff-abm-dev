# Epidemic Differentiable ABM — single-county test package (`test-ABM` branch)

This branch is a minimal, self-contained copy of the COVID-19 agent-based model for one county,
**01031 (Coffee County, AL)**, from 2020-10-26 to 2021-04-25. It contains the county's data and contact
networks, one config file, and two scripts: one to calibrate the model and one to run it.

A run with the shipped config reproduces the reference results for 01031 byte for byte (`calibrated_params.txt`
and `cf_raw_7.csv`).

## 1. Setup

```bash
git clone -b test-ABM https://github.com/complex-ai-lab/epi-diff-abm-dev.git
cd epi-diff-abm-dev
pip install -r requirements.txt        # Python 3.11; an NVIDIA GPU is strongly recommended
```

Both scripts use `python` by default. Use `PY=/path/to/python bash ...` to choose a different interpreter.

## 2. The config file — `config.yaml` (repo root)

Everything is set in **`config.yaml`**. The block at the top is the one to edit:

```yaml
epidemic:
  EXPOSED_TO_INFECTED_TIME: 3        # E -> I, days
  INFECTED_TO_RECOVERED_TIME: 5      # I -> R, days

  HORIZON_DAYS: 182                  # run.sh: days simulated after the 14-day burn-in (1..182)
  BETA: null                         # transmission rate used by run.sh: null = calibrated weekly beta,
                                     #   a number = that beta every week, a list of 26 = one per week
  BETA_DAILY: null                   # daily beta (replaces BETA): a list from 2020-10-26, or a CSV
  INITIAL_INFECTED: null             # fraction infected at day 0; null = calibrated
  K: null                            # reporting fraction (cases = k x new infections); null = calibrated

  BETA_MIN: 0.1                      # calibration: bounds and starting value of beta
  BETA_MAX: 5.0
  BETA_INIT: 2.5
  BETA_BURN_IN: 1.0                  # fixed beta during the 14 burn-in days

  IMMUNITY_WANING_MODE: fixed        # none | fixed | stochastic
  RECOVERED_TO_SUSCEPTIBLE_TIME: 60  # R -> S, days
  VACCINATED_TO_SUSCEPTIBLE_TIME: 90 # V -> S, days
```

Below it are:
- **`run_settings`**: the calibration recipe (seed, contact-layer weights, Poisson loss, spline on log beta,
  251 epochs) and which counterfactual types to run. `CF_TYPES: "7"` is the factual run.
- **`simulation_metadata` / `state` / `substeps`**: the AgentTorch model definition, which you normally leave
  alone.

Any `run_settings` key can be overridden for one command from the shell, e.g. `EPOCHS=2 bash calibrate.sh`.
To use a different config file, run `CONFIG=my_config.yaml bash run.sh`.

## 3. Calibrate

```bash
bash calibrate.sh                  # ~27 min on one A100
OVERWRITE=1 bash calibrate.sh      # redo an existing calibration
EPOCHS=2 bash calibrate.sh         # 1-minute smoke test
```

Calibration fits **26 weekly beta values, the initial infected fraction and k** to the observed daily cases.
It uses the current E→I, I→R and waning settings. The result is written to

```
result_graphs/01031/202010-202104/0.0005_<E->I>_<I->R>_True_True_False_metro_0/
    calibrated_params.txt   28 rows: beta week 0..25, initial infected fraction, k
    r_daily.txt, spline_coef.txt, spline_meta.json, training_loss.csv, per-epoch plots
```

The folder name contains E→I and I→R. A calibration with 3/5 and one with 4/5 are therefore stored side by
side, and `run.sh` picks the one that matches the config.

## 4. Run

```bash
bash run.sh                        # factual run, 30 stochastic iterations, ~2 min -> results/01031/run/
```

Output in `results/01031/<RUN_NAME>/` (default `RUN_NAME=run`):

| file | contents |
|---|---|
| `beta_daily.csv` | `day, date, phase, beta` — the beta the simulator used on every simulated day |
| `seirv_daily.csv` | `day, date, S, E, I, R, V` — agent counts per day, mean over the iterations |
| `cases_sim_vs_observed.csv` | `day, date, simulated_cases, simulated_cases_sd, observed_cases` — every day after the burn-in (`observed_cases` is blank after 2021-04-11, where the case data ends) |
| `config_used.yaml` | copy of the config the run used |
| `cf_raw_7.csv`, `counterfactual_data7*.csv`, `counterfactual_results7.png` | raw per-iteration output |

`day` is the simulation step. Days 0–13 (2020-10-12 to 2020-10-25) are an unscored burn-in, and day 14 is
2020-10-26, the first observed day. The model has 48,113 agents, and S+E+I+R+V adds up to that on every day.

## 5. Running with modified inputs

Edit `config.yaml`, then run again. Use `RUN_NAME` so earlier results are not overwritten.

**Choose the horizon.** `HORIZON_DAYS: 60` simulates 2020-10-26 to 2020-12-24, after the 14-day burn-in. The
maximum is 182, because interventions, vaccines and networks exist for 182 days. Because the randomness is
fixed per day, a 60-day run is identical to the first 60 days of the full run. The horizon applies to
`run.sh` only; calibration always uses the full window.

**Change beta** (keep the calibrated initial infected fraction and k):

```yaml
  BETA: 1.2                                  # or a list of 26 weekly values
```
```bash
RUN_NAME=beta_1.2 bash run.sh
```

**Daily beta.** `BETA_DAILY` replaces `BETA` and takes one value per day.
- A list in the config starts on 2020-10-26 and needs at least `HORIZON_DAYS` values; the burn-in keeps
  `BETA_BURN_IN`:
  ```yaml
    BETA_DAILY: [1.30, 1.28, 1.25, 1.21, ...]
  ```
- A CSV with a `beta` column is read the same way. If the CSV also has a `day` column, rows are placed by
  simulation day, burn-in included. The easiest way to make one is to edit the `beta` column of a previous
  run's `beta_daily.csv`:
  ```bash
  cp results/01031/run/beta_daily.csv my_beta.csv     # edit the beta column
  ```
  ```yaml
    BETA_DAILY: my_beta.csv                          # relative to the repo root
  ```
  ```bash
  RUN_NAME=my_beta bash run.sh
  ```
  Feeding an unedited `beta_daily.csv` back in reproduces that run exactly.

**Change E→I or I→R.** The calibrated beta was fitted with the old durations, so normally you recalibrate:

```yaml
  EXPOSED_TO_INFECTED_TIME: 4
```
```bash
bash calibrate.sh                            # new calibration in ..._4_5_..._metro_0/
RUN_NAME=ei_4 bash run.sh                    # uses that calibration
```

To keep the old fit and only change the durations:

1. Set the new duration in `config.yaml`. Durations are only ever read from there:
   ```yaml
     EXPOSED_TO_INFECTED_TIME: 4      # was 3
   ```
2. Point `run.sh` at the calibration fitted with the old durations (E→I 3, I→R 5). `CALIBRATION` only chooses
   which beta, initial infected fraction and k to load; it does not set any duration.
   ```bash
   CALIBRATION=result_graphs/01031/202010-202104/0.0005_3_5_True_True_False_metro_0/calibrated_params.txt \
   RUN_NAME=ei_4_oldfit bash run.sh
   ```

Check `results/01031/ei_4_oldfit/config_used.yaml` to confirm which durations the run used.

**Run without any calibration.** Set `BETA` (or `BETA_DAILY`), `INITIAL_INFECTED` and `K`, then run
`RUN_NAME=my_params bash run.sh`.

**Run counterfactuals.** For example, `CF_TYPES=1,7,11 RUN_NAME=cf bash run.sh` runs all-open, factual, and
no vaccines. The full list is in `CF_POLICY_DESC` in `abm_nets.py`. The three summary CSVs always come from
CF 7, so include 7 in the list.

Other inputs (vaccines, school or workplace closures, observed cases) are the data files listed below. You can
edit them directly.

## 6. Where the data is

| path | contents |
|---|---|
| `populations/pop01031/age.pickle` | age group of each agent (0=20–29, 1=30–39, 2=40–49, 3=50–64, 4=65+, 5=<20) |
| `populations/pop01031/disease_stages.csv` | initial stage of each agent (all susceptible; seeding is by the initial infected fraction) |
| `populations/pop01031/intervention.csv` | one row per simulated day `t` (196 = 14 burn-in + 182): `date, school_intervention, occ_intervention` (0 open / 1 closed), `vaccines`, `vaccines_65plus` (first doses) |
| `populations/pop01031/burnin_days.txt`, `mapping.json` | burn-in length; age-code labels |
| `data/processed_data/01031/202010-202104/daily_data.csv` | observed daily cases (Delphi COVIDcast); the fit uses `cases_singular` |
| `data/population_data/AL_population_data/01031_population.csv` | synthetic population (age, household, occupation), used by the vaccine-priority and household counterfactuals |
| `data/networks/covid_output_p0/01031/mobility_networks/` | contact networks: `HOUSEHOLD_NETWORK.npz`, `SCHOOL_NETWORK.npz`, `occnets/<t>.npz` and `randnets/<t>.npz` (workplace and community, one per observed day) |
| `result_graphs/`, `results/` | outputs written by the scripts (git-ignored) |

The networks are compressed edge lists, readable with `np.load(path)["edges"]` (one row per contact pair).

## 7. Code map

| file | role |
|---|---|
| `main.py` | entry point; reads `config.yaml`, seeds, builds the simulator, calls `abm_nets.eval_net` |
| `abm_nets.py` | calibration loop (spline on log beta, Poisson loss) and counterfactual / run loop; writes the 3 CSVs |
| `covid_abm/substeps/new_transmission/transition.py` | disease dynamics: S→E→I→R, S→V vaccination, waning, interventions |
| `covid_abm/spline_basis.py` | B-spline basis for log beta |
| `agent_torch/` | AgentTorch simulation framework |
| `calibrate.sh`, `run.sh` | the two launchers |

The model code comes from the `preetom-dev` branch; the repo layout follows `b5a6372`. Data preparation,
network generation and multi-county runs are on `main` / `preetom-dev`.
