# Immunity-waning validation (`waning_tests/`)

Implements + validates configurable immunity waning (`S → E → I → R → S`) added
to the core ABM. See `plans/Task_ Add Configurable Immunity Waning to the ABM.md`.

## What changed in the model

| File | Change |
|------|--------|
| `covid_abm/yamls/config.yaml` | added `IMMUNITY_WANING_MODE: "none"` and `WANING_RATE: 0.01` under `simulation_metadata` (`RECOVERED_TO_SUSCEPTIBLE_TIME: 100` was already present). |
| `covid_abm/substeps/new_transmission/transition.py` | `NewTransmission.__init__` reads the 3 keys (`.get()` + defaults). `NewTransmission.forward` runs a waning block **after** the normal `S→E→I→R` progression. |

Waning modes (fixed before the run, not learnable, not a counterfactual):

* **`none`** – unchanged behaviour, `S → E → I → R`.
* **`fixed`** – when an agent enters `R` at step `t`, its next transition is
  scheduled for `t + RECOVERED_TO_SUSCEPTIBLE_TIME`; at that step it goes
  `R → S`. Timer is set once and not reset.
* **`stochastic`** – each day, every agent **already** in `R` at the start of
  the step returns to `S` with probability `p = 1 - exp(-WANING_RATE)`.
  An agent that becomes `R` during step `t` is not eligible until step `t+1`.

Vaccinated agents use `RECOVERED_VAR` (see `apply_vaccines`), so vaccine-derived
immunity wanes on the same rule. This is intentional and commented in the code.

To use it, edit only `config.yaml`, then run the normal pipeline:

```yaml
IMMUNITY_WANING_MODE: "none"
# or
IMMUNITY_WANING_MODE: "fixed"
RECOVERED_TO_SUSCEPTIBLE_TIME: 100
# or
IMMUNITY_WANING_MODE: "stochastic"
WANING_RATE: 0.01
```

```bash
python3 main.py 01031
```

## Running the 3 validation tests

Everything is driven from the repo root. Use the `.venv_ds` env
(`conda activate .venv_ds`, or `export PYTHON=~/.conda/envs/.venv_ds/bin/python`).

`run_waning_tests.sh`:
1. backs up `config.yaml` **and** the existing
   `result_graphs/01031/202010-202104/0.0005_3_5_True_True_False_metro_0/training_proportions.csv`
   into `waning_tests/_backup/`,
2. for each configuration: patches only the waning keys
   (`set_waning_config.py`), runs `python3 main.py 01031`, copies the run's
   `training_proportions.csv` to `waning_tests/data/<label>.csv`,
3. restores `config.yaml` and the original baseline file on exit.

The already-generated `none` result from seed 42 is reused as the **Test 1
reference** (`waning_tests/data/none_baseline_reference.csv`); it is *not*
re-run.

```bash
# all 7 runs (1 none-verify + 3 fixed + 3 stochastic) on 01031, SEED=42
bash waning_tests/run_waning_tests.sh 01031 all

# or a single group
bash waning_tests/run_waning_tests.sh 01031 none
bash waning_tests/run_waning_tests.sh 01031 fixed
bash waning_tests/run_waning_tests.sh 01031 stochastic
```

Each `main.py 01031` run is a full 251-epoch calibration (GPU, ~same cost as
one county in `run_all_sims.sh`).

Then build the plots + PASS/FAIL summary:

```bash
python3 waning_tests/plot_waning_tests.py
```

Outputs:

```
waning_tests/
├── no_waning_same_seed_comparison.png     # Test 1
├── fixed_waning_60_80_100_days.png        # Test 2
└── stochastic_waning_005_010_020.png      # Test 3
```

### Pass criteria

* **Test 1** – `none` run overlays the stored seed-42 baseline
  (`max |Δ infected fraction|` ≈ 0; tiny residuals only from the
  non-deterministic GNN scatter ops noted in `main.py`).
* **Test 2** – the three curves are distinct; shorter immunity (60d) gives a
  larger late-window infected fraction than 80d ≥ 100d.
* **Test 3** – the three curves are distinct; higher `WANING_RATE` gives a
  larger late-window infected fraction (0.005 ≤ 0.01 ≤ 0.02).

## Files

| File | Purpose |
|------|---------|
| `set_waning_config.py` | targeted in-place patch of the 3 waning keys in `config.yaml` (no YAML re-dump). |
| `run_waning_tests.sh` | orchestrates the runs, backs up/restores originals, collects trajectories. |
| `plot_waning_tests.py` | reads `data/*.csv`, writes the 3 PNGs, prints PASS/FAIL. |
| `data/` | collected `training_proportions.csv` per run + the reused baseline. |
| `_backup/` | pristine `config.yaml` and baseline trajectory (restored automatically). |
