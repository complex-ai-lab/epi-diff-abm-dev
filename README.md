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
python3.11 -m venv .venv               # Python 3.11 (check `python --version`; clusters often default to another)
.venv/bin/pip install -r requirements.txt   # an NVIDIA GPU is strongly recommended
```

Both scripts use `python` by default. Either activate the venv (`source .venv/bin/activate`) or pass it
explicitly, e.g. `PY=.venv/bin/python bash calibrate.sh`.

## 2. The config file — `config.yaml` (repo root)

Everything is set in **`config.yaml`**. The block at the top is the one to edit:

```yaml
epidemic:
  EXPOSED_TO_INFECTED_TIME: 3        # E -> I, days
  INFECTED_TO_RECOVERED_TIME: 5      # I -> R, days

  HORIZON_DAYS: 182                  # run.sh: days simulated after the 14-day burn-in (1..182)
  R_T: null                          # transmission scale used by run.sh: null = calibrated weekly R_t,
                                     #   a number = that R_t every week, a list of 26 = one per week
  R_T_DAILY: null                    # daily R_t (replaces R_T): a list from 2020-10-26, or a CSV
  INITIAL_INFECTED: null             # fraction infected at day 0; null = calibrated
  K: null                            # reporting fraction (cases = k x new infections); null = calibrated

  R_T_MIN: 0.1                       # calibration: bounds and starting value of R_t
  R_T_MAX: 5.0
  R_T_INIT: 2.5
  R_T_BURN_IN: 1.0                   # fixed R_t during the 14 burn-in days

  IMMUNITY_WANING_MODE: fixed        # none | fixed | stochastic
  RECOVERED_TO_SUSCEPTIBLE_TIME: 60  # R -> S, days
  VACCINATED_TO_SUSCEPTIBLE_TIME: 90 # V -> S, days
```

`R_T` is the transmission scale called R_t in the paper (it was `BETA` in earlier versions; old `BETA*` keys
now raise an error). It is **not** the beta of an SIR/SEIR model, and not the effective reproduction number
of the simulated epidemic. Each day, a susceptible agent's infection probability is `1 - exp(-λ)` with
λ = R_t × susceptibility(age) / mean interactions(age) × the sum over its infectious contacts of
layer weight × infectiousness(E or I) × gamma infectiousness profile(days since infection); see `_lam` in
`covid_abm/substeps/new_transmission/transition.py`.

Below it are:
- **`run_settings`**: the calibration recipe (seed, contact-layer weights, Poisson loss, spline on log R_t,
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

Calibration fits **26 weekly R_t values, the initial infected fraction and k** to the observed daily cases.
It uses the current E→I, I→R and waning settings. The result is written to

```
result_graphs/01031/202010-202104/0.0005_<E->I>_<I->R>_True_True_False_metro_0/
    calibrated_params.txt   28 rows: R_t week 0..25, initial infected fraction, k
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
| `r_t_daily.csv` | `day, date, phase, r_t` — the R_t the simulator used on every simulated day |
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

**Change R_t** (keep the calibrated initial infected fraction and k):

```yaml
  R_T: 1.2                                   # or a list of 26 weekly values
```
```bash
RUN_NAME=r_t_1.2 bash run.sh
```

**Daily R_t.** `R_T_DAILY` replaces `R_T` and takes one value per day.
- A list in the config starts on 2020-10-26 and needs at least `HORIZON_DAYS` values; the burn-in keeps
  `R_T_BURN_IN`:
  ```yaml
    R_T_DAILY: [1.30, 1.28, 1.25, 1.21, ...]
  ```
- A CSV with an `r_t` column is read the same way. If the CSV also has a `day` column, rows are placed by
  simulation day, burn-in included. The easiest way to make one is to edit the `r_t` column of a previous
  run's `r_t_daily.csv`:
  ```bash
  cp results/01031/run/r_t_daily.csv my_r_t.csv      # edit the r_t column
  ```
  ```yaml
    R_T_DAILY: my_r_t.csv                           # relative to the repo root
  ```
  ```bash
  RUN_NAME=my_r_t bash run.sh
  ```
  Feeding an unedited `r_t_daily.csv` back in reproduces that run exactly.

**Change E→I or I→R.** The calibrated R_t was fitted with the old durations, so normally you recalibrate:

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
   which R_t, initial infected fraction and k to load; it does not set any duration.
   ```bash
   CALIBRATION=result_graphs/01031/202010-202104/0.0005_3_5_True_True_False_metro_0/calibrated_params.txt \
   RUN_NAME=ei_4_oldfit bash run.sh
   ```

Check `results/01031/ei_4_oldfit/config_used.yaml` to confirm which durations the run used.

**Run without any calibration.** Set `R_T` (or `R_T_DAILY`), `INITIAL_INFECTED` and `K`, then run
`RUN_NAME=my_params bash run.sh`.

**Run counterfactuals.** For example, `CF_TYPES=1,7,11 RUN_NAME=cf bash run.sh` runs all-open, factual, and
no vaccines. The full list is in `CF_POLICY_DESC` in `abm_nets.py`. The three summary CSVs always come from
CF 7, so include 7 in the list.

Other inputs (vaccines, school or workplace closures, observed cases) are the data files listed below. You can
edit them directly.

## 6. SEIR-equivalent rates

`seir_rates.py` turns a run's output into the rates of a plain SEIR model with the same daily incidence. New
exposures each day are `simulated_cases / K`. They are divided by S × I / N from the end of the previous day,
which gives `beta_seir`; `r_eff = beta_seir × S / N × INFECTED_TO_RECOVERED_TIME`. Sigma and gamma are
1/E→I and 1/I→R.

```bash
.venv/bin/python seir_rates.py results/01031/run --k 0.285242      # --k only if the config had K: null
.venv/bin/python seir_rates.py reference_results/13013 --out seir_13013.csv
```

The output is `seir_rates.csv` with `day, date, r_t, new_exposed, S_prev, E_prev, I_prev, beta_seir, r_eff,
beta_seir_over_r_t`. In the 11 reference counties `beta_seir` is about 0.17–0.36 × R_t (median per county).
The ratio is not a constant: it drifts over time and differs between counties. The two models also differ
in structure. The ABM's stage durations are fixed rather than exponential, exposed agents are infectious, and
S also changes through vaccination and waning. So the SEIR rates reproduce the ABM's incidence, not every
compartment.

## 7. SEIRV mode (`MIXING: mean_field`) and `seirv.py`

Two switches in the `epidemic` block turn the ABM into a stochastic SEIRV model:

```yaml
  MIXING: mean_field                 # network (default) | mean_field
  STAGE_DURATIONS: exponential       # fixed (default) | exponential
  IMMUNITY_WANING_MODE: stochastic   # or none; fixed has no SEIRV equivalent
```

- **`mean_field`**: every agent's daily infection probability is `1 - exp(-R_t × I / N)`, with I the number of
  agents in I. Contact networks, closures, layer weights, age susceptibility, E infectiousness and the gamma
  infectiousness profile are not used, so R_t is exactly the SEIRV beta. Layer attribution is switched off.
- **`exponential`**: each day an agent leaves E with probability `1 - exp(-1/EXPOSED_TO_INFECTED_TIME)` and I with
  `1 - exp(-1/INFECTED_TO_RECOVERED_TIME)`, instead of after a fixed number of days.
- Vaccination (doses from `intervention.csv`, S → V) works as before.

Defaults (`network`, `fixed`) are the paper model, unchanged. Calibration works in either mode. A non-default
structure is calibrated into its own folder, `..._metro_0_<MIXING>_<STAGE_DURATIONS>/`, so `run.sh` never mixes
up the two. Calibrating in mean-field mode fits an SEIRV model to the cases, not the network ABM, and the
school and work counterfactuals have no effect in it because closures only remove network edges.

`seirv.py` runs the same model without agents. It uses the same parameters from the config, the same vaccine
doses and the same order of steps within a day. It writes the expected-value trajectory and, with `--iters`,
the mean and SD of chain-binomial runs:

```bash
.venv/bin/python seirv.py --config my_mf_config.yaml --iters 200 --out seirv_model.csv
.venv/bin/python seirv.py --config my_mf_config.yaml --r-t-csv results/01031/mf/r_t_daily.csv \
    --i0 0.002238 --compare results/01031/mf        # compare with a mean_field run.sh output
```

Columns: `day, date, phase, r_t, S, E, I, R, V, new_E, cases` (+ `<col>_mean`, `<col>_sd`). Check on 01031
(R_t 0.3, burn-in 0.25, stochastic waning, observed vaccines, 30 ABM iterations against 200 SEIRV runs): every
compartment on every day is within about 2 standard errors of the SEIRV mean, and the largest gap is 0.26% of N.

## 8. Where the data is

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
| `reference_results/<FIPS>/` | stored factual (CF 7) results for 01031 and 10 other counties (see below) |

### Reference results for other counties

`reference_results/` holds `r_t_daily.csv`, `seirv_daily.csv` and `cases_sim_vs_observed.csv` for 11 counties,
built from the stored calibrated P0 runs (CF 7, 30 iterations) on `preetom-dev`:

| FIPS | county | FIPS | county |
|---|---|---|---|
| 01031 | Coffee, AL | 30029 | Flathead, MT |
| 13013 | Barrow, GA | 36051 | Livingston, NY |
| 17037 | DeKalb, IL | 42059 | Greene, PA |
| 19027 | Carroll, IA | 47005 | Benton, TN |
| 21071 | Floyd, KY | 51149 | Prince George, VA |
| 27005 | Becker, MN | | |

The columns are the same as in a `run.sh` output, with one difference: `cases_sim_vs_observed.csv` covers
only the 168 observed days (2020-10-26 to 2021-04-11), because the stored runs did not keep simulated cases
after the last observed day. `r_t_daily.csv` and `seirv_daily.csv` cover all 196 days. For 01031 the files
match `bash run.sh` (SD to within float rounding). Only 01031 has data and networks in this branch, so the
other counties cannot be rerun here.

Each folder also has `config_<FIPS>.yaml`, the config of that run in the same format as `config.yaml`, with the
county's population fields and its calibrated `R_T` (26 weekly values), `INITIAL_INFECTED` and `K` filled in.
`CONFIG=reference_results/01031/config_01031.yaml bash run.sh` reproduces the stored 01031 run byte for byte
without a calibration file.

The networks are compressed edge lists, readable with `np.load(path)["edges"]` (one row per contact pair).

## 9. Code map

| file | role |
|---|---|
| `main.py` | entry point; reads `config.yaml`, seeds, builds the simulator, calls `abm_nets.eval_net` |
| `abm_nets.py` | calibration loop (spline on log R_t, Poisson loss) and counterfactual / run loop; writes the 3 CSVs |
| `covid_abm/substeps/new_transmission/transition.py` | disease dynamics: S→E→I→R, S→V vaccination, waning, interventions |
| `covid_abm/spline_basis.py` | B-spline basis for log R_t |
| `agent_torch/` | AgentTorch simulation framework |
| `calibrate.sh`, `run.sh` | the two launchers |
| `seir_rates.py` | SEIR-equivalent beta and R_eff of a run (section 6) |
| `seirv.py` | the SEIRV model that `MIXING: mean_field` + `STAGE_DURATIONS: exponential` simulate (section 7) |
| `seirv_fit.py`, `seirv_plot.py` | SEIRV rates inferred from a network run, and the SEIRV-vs-ABM check (section 10) |

The model code comes from the `preetom-dev` branch; the repo layout follows `b5a6372`. Data preparation,
network generation and multi-county runs are on `main` / `preetom-dev`.

## 10. From a network run to SEIRV parameters (`seirv_fit.py`)

The calibrated R_t of the network ABM is not an SEIRV beta, and beta / R_t is not constant over time (in the 11
reference counties a single ratio misses the ABM's cases by 23–83%). Instead, beta is inferred day by day
from the run's output (`seirv_daily.csv`, `cases_sim_vs_observed.csv`, `config_<FIPS>.yaml`):

```
new_E(t) = simulated_cases(t) / K                         (burn-in: S(t-1) - S(t), no vaccines or waning yet)
beta(t)  = new_E(t) * N / (S(t-1) * I(t-1))
sigma    = 1 / EXPOSED_TO_INFECTED_TIME,  gamma = 1 / INFECTED_TO_RECOVERED_TIME
doses(t) = V(t) - V(t-1) + doses(t - VACCINATED_TO_SUSCEPTIBLE_TIME)        (matches intervention.csv for 01031)
```

These go into a closed discrete-time SEIRV model that starts from the ABM's day-0 state and computes S, E, I,
R, V itself, one step per day:

```
new_E = beta(t) * S * I / N             S -> E
vac   = min(doses(t), S - new_E)        S -> V
E -> I: sigma * E,   I -> R: gamma * I
R -> S, V -> S: the cohorts that entered R / V RECOVERED_TO_SUSCEPTIBLE_TIME / VACCINATED_TO_SUSCEPTIBLE_TIME days ago
cases = K * new_E
```

```bash
.venv/bin/python seirv_fit.py reference_results/01031 results/seirv_fit/01031.csv
.venv/bin/python seirv_plot.py results/seirv_fit/01031.csv     # 01031_cases.png, 01031_seirv.png
```

The CSV has, per day, `beta, sigma, gamma`, the E->I and I->R rates measured in the ABM (`sigma_eff`,
`gamma_eff`), `R_eff = beta * S / N / gamma`, `doses`, the five compartments of both models (`S_abm`, ...,
`S_seirv`, ...) and `cases_abm, cases_seirv, cases_observed`. The plots compare daily cases (ABM, SEIRV,
observed) and the five compartments (ABM vs SEIRV).

Results for the 11 reference counties (days 14–181): SEIRV cases match the ABM with correlation 0.95–0.996 and
8–18% absolute error, no compartment is more than 3.5% of N off, and V matches exactly. gamma_eff is about
0.20 (= 1/5), but sigma_eff is about 0.25 instead of 1/3: agents stay about a day longer in E in the ABM, so
the SEIRV model has a lower E (and R) peak. The script needs `IMMUNITY_WANING_MODE: fixed`.
