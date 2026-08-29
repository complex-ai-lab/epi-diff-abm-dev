# Isolated immunity-waning test (`frozen_cal_params_iso_wane/`)

A cleaner companion to `waning_tests/`.

## Why this exists

`waning_tests/` runs a **full 251-epoch recalibration** for every waning
configuration (`run_waning_tests.sh` → `main.py 01031` each time). So every
curve there is a *different fit*: the optimiser refits the weekly transmission
schedule `R_t`, the initial infected proportion and `k` for each setting. That
makes the first ~4 months of those plots not directly comparable — the curves
differ because of the calibration, not because of waning. Waning is
mechanically negligible until a recovered pool exists, yet the `waning_tests/`
curves already diverge from `none` around day 30.

This folder isolates the waning effect:

1. Recover the calibrated parameter vector from the `none` calibration once
   (`main.py 01031` with `IMMUNITY_WANING_MODE=none`) →
   `frozen_params_none.txt` (26 weekly `R` values, then `infected_proportion`,
   then `k`).
2. Run a **single forward pass** of the ABM (`iso_wane_forward.py`, no
   optimiser, no gradient, `torch.no_grad()`), holding those parameters and the
   RNG seed fixed, toggling **only** `IMMUNITY_WANING_MODE` /
   `RECOVERED_TO_SUSCEPTIBLE_TIME` / `WANING_RATE`.

Because parameters + seed are identical, every curve overlays exactly until the
first recovered agents appear; all later spread is the pure waning mechanism.

## Run it

```bash
# uses .venv_ds; SEED defaults to 42 (per-county seed 1073, same as main.py 01031)
PYTHON=~/.conda/envs/.venv_ds/bin/python \
  bash frozen_cal_params_iso_wane/run_iso_wane.sh 01031

python3 frozen_cal_params_iso_wane/plot_iso_wane.py
```

`run_iso_wane.sh` backs up and restores `covid_abm/yamls/config.yaml` and
`result_graphs/01031/202010-202104/0.0005_3_5_True_True_False_metro_0/{calibrated_params.txt,training_proportions.csv,training_loss.csv}`
on exit — no original result files are left modified. It skips step 1 if
`frozen_params_none.txt` already exists.

## Configurations

| label | mode | knob |
|-------|------|------|
| `none` | none | — (sanity: overlays every other curve for the whole run) |
| `fixed_60/80/100` | fixed | `R→S` exactly N days after entering R |
| `stochastic_0.005/0.010/0.020/0.040` | stochastic | daily `p = 1 - exp(-rate)` |

## Outputs

```
frozen_cal_params_iso_wane/
├── frozen_params_none.txt        # the frozen calibrated vector
├── data/<label>.csv              # t,susceptible,exposed,infected,recovered,dead
├── fixed_iso.png                 # none + fixed 60/80/100
└── stochastic_iso.png            # none + stochastic 0.005/0.010/0.020/0.040
```

## Caveats

* **`fixed` vs `none` is the cleanest contrast** — `fixed` mode consumes no
  extra randomness, so those runs share the global RNG stream with `none` for
  the entire simulation.
* **`stochastic` mode** draws a per-step `torch.rand_like` for the waning
  coin-flip from the same global RNG. Once waning activates, the RNG stream is
  offset relative to `none`, so a small part of the `stochastic − none`
  difference is stream shift rather than mechanism. The effect is second-order
  next to the waning signal but it is not zero. (Fixing it properly means
  giving the waning draw its own `torch.Generator` in
  `covid_abm/substeps/new_transmission/transition.py`.)
* The frozen forward run is **not** bit-identical to
  `waning_tests/data/none_baseline_reference.csv`: the stored baseline is
  epoch 250 of the calibration, whose RNG state has advanced through 250 prior
  episodes. `run_iso_wane.sh` still checks that re-running the calibration
  reproduces that baseline (confirming `frozen_params_none.txt` is really the
  `none` fit); the isolation runs then start from a clean seed.
