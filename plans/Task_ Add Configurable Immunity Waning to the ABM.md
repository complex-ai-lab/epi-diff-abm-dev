# Task: Add Configurable Immunity Waning to the ABM

## Codebase

First inspect the existing code, especially:

- `covid_abm/substeps/new_transmission/transition.py`
- `covid_abm/yamls/config.yaml`

Make minimal changes. Preserve all existing epidemic, vaccination, calibration, and counterfactual logic.

---

## Goal

Add optional immunity waning directly to the **main ABM dynamics**:

```text
S → E → I → R → S
```

with three modes:

```yaml
IMMUNITY_WANING_MODE: "none"
IMMUNITY_WANING_MODE: "fixed"
IMMUNITY_WANING_MODE: "stochastic"
```

Do **not** add new counterfactual types.

Do **not** modify `abm_nets.py` unless absolutely required.

Do **not** modify the calibration machinery or `LearnableParams`.

The selected waning mode and its parameter are fixed before running the model.

---

# 1. Configuration

Modify:

`covid_abm/yamls/config.yaml`

Add under `simulation_metadata`:

```yaml
IMMUNITY_WANING_MODE: "none"
RECOVERED_TO_SUSCEPTIBLE_TIME: 100
WANING_RATE: 0.01
```

Interpretation:

### `none`

Existing behavior:

```text
S → E → I → R
```

### `fixed`

Every recovered agent becomes susceptible after the configured number of days:

```text
R --RECOVERED_TO_SUSCEPTIBLE_TIME days--> S
```

For example:

```yaml
IMMUNITY_WANING_MODE: "fixed"
RECOVERED_TO_SUSCEPTIBLE_TIME: 100
```

### `stochastic`

Every recovered agent has a daily probability of losing immunity:

```text
p = 1 - exp(-WANING_RATE)
```

For example:

```yaml
IMMUNITY_WANING_MODE: "stochastic"
WANING_RATE: 0.01
```

The waning rate is constant throughout the run.

Do not use the existing `k` parameter for this.

---

# 2. Core ABM implementation

Modify:

`covid_abm/substeps/new_transmission/transition.py`

## Initialization

Read:

```python
self.IMMUNITY_WANING_MODE
self.RECOVERED_TO_SUSCEPTIBLE_TIME
self.WANING_RATE
```

from `simulation_metadata`.

Use `.get()` with sensible defaults if consistent with the existing code.

---

## Fixed mode

When:

```python
self.IMMUNITY_WANING_MODE == "fixed"
```

schedule an R → S transition when an agent enters R.

If an agent becomes recovered at timestep `t`, schedule:

```python
t + self.RECOVERED_TO_SUSCEPTIBLE_TIME
```

as its next transition.

When that time is reached:

```text
R → S
```

Do not reset the timer on subsequent timesteps.

Do not change the existing:

```text
S → E
E → I
I → R
```

logic.

---

## Stochastic mode

When:

```python
self.IMMUNITY_WANING_MODE == "stochastic"
```

do not schedule a deterministic R → S transition.

Instead, on each one-day timestep, for agents that were already recovered:

```python
waning_probability = 1.0 - torch.exp(-self.WANING_RATE)

waning_draw = torch.rand_like(disease_stage.float())

waning_mask = (
    (disease_stage == self.RECOVERED_VAR)
    & (waning_draw < waning_probability)
)

disease_stage = torch.where(
    waning_mask,
    torch.full_like(disease_stage, self.SUSCEPTIBLE_VAR),
    disease_stage,
)
```

Adapt variable names to the existing implementation.

Use tensor/device-safe operations.

---

## Newly recovered agents

An agent that becomes:

```text
I → R
```

during timestep `t` must not immediately become:

```text
R → S
```

during the same timestep under stochastic waning.

Only agents that were already recovered at the beginning of the timestep should be eligible for stochastic waning.

---

## None mode

When:

```yaml
IMMUNITY_WANING_MODE: "none"
```

the model must behave exactly as it currently does.

---

# 3. Do not modify counterfactuals

Do NOT:

- add counterfactual types;
- modify `COUNTERFACTUAL_TYPE`;
- modify `cf_types_to_run`;
- modify `cf_folder_map`;
- add waning-specific counterfactual branches.

Existing counterfactual runs should simply inherit whichever core ABM waning configuration is active.

---

# 4. Do not modify calibration

Do NOT:

- add waning to `LearnableParams`;
- make `WANING_RATE` learnable;
- change the calibration loss;
- change the existing calibrated parameters;
- change the existing `k` parameter.

Waning is a fixed model configuration selected before a run.

---

# 5. Vaccination

Inspect how vaccination is represented.

If vaccinated individuals use the same `RECOVERED_VAR` state as naturally recovered individuals, the new R → S mechanism will apply to them as well.

Do not redesign vaccination for this task.

Add a short code comment if necessary to make this behavior explicit.

---

# 6. Three validation plots

Create a separate output folder for these tests, with clear naming, for example:

```text
waning_tests/
```

or another appropriate folder following the repository's conventions.

The tests should generate plots, not alter the existing result files.

---

## Test 1 — No waning baseline

Use:

```yaml
IMMUNITY_WANING_MODE: "none"
```

Use an **already-generated result from the same seed** as the baseline/reference.

Plot the infection trajectory from the existing result and the newly run `none` configuration together.

The purpose is to verify that adding the waning code with:

```text
mode = none
```

does not change the existing ABM dynamics.

Save with an understandable name such as:

```text
waning_tests/no_waning_same_seed_comparison.png
```

---

## Test 2 — Fixed immunity duration

Run the model three times using the same seed and otherwise identical configuration:

```yaml
IMMUNITY_WANING_MODE: "fixed"
```

with:

```text
RECOVERED_TO_SUSCEPTIBLE_TIME = 60
RECOVERED_TO_SUSCEPTIBLE_TIME = 80
RECOVERED_TO_SUSCEPTIBLE_TIME = 100
```

Plot the three infection trajectories together.

Save as:

```text
waning_tests/fixed_waning_60_80_100_days.png
```

The plot legend must clearly identify:

```text
Fixed 60 days
Fixed 80 days
Fixed 100 days
```

---

## Test 3 — Stochastic waning

Run the model three times using the same seed and otherwise identical configuration with:

```yaml
IMMUNITY_WANING_MODE: "stochastic"
```

Use these three reasonable daily rates:

```text
WANING_RATE = 0.005
WANING_RATE = 0.01
WANING_RATE = 0.02
```

These correspond approximately to mean immunity durations of:

```text
200 days
100 days
50 days
```

Plot the three infection trajectories together.

Save as:

```text
waning_tests/stochastic_waning_005_010_020.png
```

Clearly label the three curves with their waning rates.

---

# 7. Test requirements

Use the repository's existing simulation/result/plotting infrastructure where possible.

Do not build a large testing framework.

The three required outputs are simply:

```text
waning_tests/
├── no_waning_same_seed_comparison.png
├── fixed_waning_60_80_100_days.png
└── stochastic_waning_005_010_020.png
```

If the repository already has a standard results/plots directory, place `waning_tests` inside that structure instead.

Do not overwrite the original generated results.

---

# 8. Final requirements

The final implementation should let the user change only the YAML configuration before running the existing pipeline:

```yaml
IMMUNITY_WANING_MODE: "none"
```

or:

```yaml
IMMUNITY_WANING_MODE: "fixed"
RECOVERED_TO_SUSCEPTIBLE_TIME: 100
```

or:

```yaml
IMMUNITY_WANING_MODE: "stochastic"
WANING_RATE: 0.01
```

Everything else in the ABM pipeline should remain unchanged.

At the end, report:

- files changed;
- functions changed;
- the three generated plot filenames;
- whether each validation passed.