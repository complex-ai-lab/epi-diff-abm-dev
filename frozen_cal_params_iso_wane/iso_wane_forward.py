#!/usr/bin/env python3
"""Isolated immunity-waning test: FROZEN calibrated params, forward sim only.

Unlike waning_tests/ (which re-runs the full 251-epoch calibration for every
waning configuration, so every curve is a different fit), this driver:

  1. loads ONE fixed parameter vector (the calibrated weekly R schedule +
     initial infected proportion + k from the `none` calibration),
  2. runs a single forward pass of the ABM (no optimiser, no gradient),
  3. only toggles IMMUNITY_WANING_MODE / WANING_RATE / RECOVERED_TO_SUSCEPTIBLE_TIME.

Because the parameters and the RNG seed are identical across runs, the
trajectories are bit-identical until the first recovered agents appear; any
later divergence is the pure effect of the waning mechanism.

Caveat: in `stochastic` mode the per-step `torch.rand_like` waning draw pulls
from the same global RNG stream, so once waning activates the RNG stream is
shifted relative to `none`/`fixed`. `fixed` vs `none` is the cleanest contrast
(it consumes no extra randomness).

Usage (one config per process, mirrors waning_tests/run_waning_tests.sh):
    python3 frozen_cal_params_iso_wane/iso_wane_forward.py \
        --params frozen_cal_params_iso_wane/frozen_params_none.txt \
        --mode none  --out frozen_cal_params_iso_wane/data/none.csv
    ... --mode fixed --recovered-to-susceptible-time 80 --out .../fixed_80.csv
    ... --mode stochastic --waning-rate 0.02 --out .../stochastic_0.020.csv
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

import abm_nets
import covid_abm
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from main import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="01031")
    ap.add_argument("--mode", required=True,
                    choices=["none", "fixed", "stochastic"])
    ap.add_argument("--waning-rate", type=float, default=None)
    ap.add_argument("--recovered-to-susceptible-time", type=int, default=None)
    ap.add_argument("--params", required=True,
                    help="calibrated_params.txt: NUM_WEEKS R values, then "
                         "infected_proportion, then k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Same per-(county, phase) seed derivation as `python3 main.py <fips>`.
    applied_seed = seed_everything(args.seed, args.fips, phase=0,
                                   deterministic=True)

    importlib.import_module(f"populations.pop{args.fips}")
    pop_module = importlib.import_module(f"populations.pop{args.fips}")
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop_module))

    md = sim.config["simulation_metadata"]
    md["metro_calibration_phase"] = 0
    md["POPULATION"] = args.fips
    md["random_seed"] = applied_seed
    md["GENERATING_COUNTERFACTUAL"] = False

    # the only thing that varies between runs
    md["IMMUNITY_WANING_MODE"] = args.mode
    if args.waning_rate is not None:
        md["WANING_RATE"] = args.waning_rate
    if args.recovered_to_susceptible_time is not None:
        md["RECOVERED_TO_SUSCEPTIBLE_TIME"] = args.recovered_to_susceptible_time

    orig_pop_dir = md["population_dir"]
    md["population_dir"] = os.path.join(os.path.dirname(orig_pop_dir),
                                        f"pop{args.fips}")

    num_steps = md["num_steps_per_episode"]
    num_weeks = md["NUM_WEEKS"]

    # NewTransmission.__init__ reads the waning keys from config, so the runner
    # must be built AFTER the config edits above.
    runner = sim._get_runner(sim.config)
    runner.init()

    # ---- freeze the calibrated parameters --------------------------------
    param_array = np.loadtxt(args.params)
    param_tensor = torch.tensor(param_array, dtype=torch.float,
                                device=abm_nets.DEVICE)
    param_tensor = param_tensor[:, None] if param_tensor.ndim == 1 else param_tensor

    learnable_params = [(name, p) for (name, p) in runner.named_parameters()]
    # indices 1/3/4 == R2 / infected_proportion / k (same mapping abm_nets.py
    # uses for both calibration and the counterfactual replay path)
    assert learnable_params[1][0].split(".")[5] == "R2", learnable_params[1][0]
    assert learnable_params[3][0].split(".")[5] == "infected_proportion"
    assert learnable_params[4][0].split(".")[5] == "k"

    abm_nets.map_and_replace_tensor(learnable_params[1][0])(
        runner, True, param_tensor[:num_weeks], mode_calibrate=True)
    abm_nets.map_and_replace_tensor(learnable_params[3][0])(
        runner, True, param_tensor[-2], mode_calibrate=True)
    abm_nets.map_and_replace_tensor(learnable_params[4][0])(
        runner, True, param_tensor[-1], mode_calibrate=True)

    print(f"[iso_wane] fips={args.fips} mode={args.mode} "
          f"waning_rate={md['WANING_RATE']} "
          f"R_to_S_time={md['RECOVERED_TO_SUSCEPTIBLE_TIME']} "
          f"seed={applied_seed}", flush=True)
    print(f"[iso_wane] frozen R (wk0..2)={param_array[:3]} "
          f"infected_proportion={param_array[-2]:.6g} k={param_array[-1]:.6g}",
          flush=True)

    with torch.no_grad():
        runner.step(num_steps)

    substep = runner.initializer.transition_function["0"].new_transmission
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    substep.save_proportions_to_disk(args.out)
    print(f"[iso_wane] wrote {args.out} ({len(substep.proportion_history)} rows)",
          flush=True)


if __name__ == "__main__":
    main()
