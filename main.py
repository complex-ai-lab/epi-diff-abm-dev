#!/usr/bin/env python3
import os

# NOTE: must be set before any CUDA / cuBLAS context is created, otherwise
# torch.use_deterministic_algorithms(True) will raise for cuBLAS-backed ops
# (matmul, addmm, ...). Setting it here covers both `python3 main.py <fips>`
# runs and the SLURM array jobs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import random
import argparse
import importlib

import numpy as np
import torch

import abm_nets
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import covid_abm


# Base seed for reproducibility. Resolution order (first hit wins):
#   1. --seed on the command line
#   2. the SEED environment variable
#   3. DEFAULT_BASE_SEED below
# The actual RNG seed is derived per (county, phase) from this base so that
# running a single county on its own is reproducible, and every county in a
# multi-county run gets an independent-but-deterministic RNG stream.
DEFAULT_BASE_SEED = 42

_UINT31 = 2**31 - 1


def resolve_base_seed(cli_seed=None):
    if cli_seed is not None:
        return int(cli_seed)
    env_seed = os.environ.get("SEED", "").strip()
    if env_seed:
        return int(env_seed)
    return DEFAULT_BASE_SEED


def seed_everything(base_seed, fips_code=None, phase=0, deterministic=True):
    """Seed Python / NumPy / Torch (CPU + CUDA) RNGs.

    The per-county, per-phase offset keeps `main.py <fips>` bit-for-bit
    consistent with the same county+phase inside a larger batch run.
    """
    seed = base_seed
    if fips_code is not None:
        seed = (base_seed + int(fips_code) + 1_000_003 * int(phase)) % _UINT31

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # warn_only=True: the GNN message-passing scatter/index ops have no
        # deterministic CUDA implementation; we still want the rest pinned.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def run_county_phase(fips_code, phase, base_seed=DEFAULT_BASE_SEED, deterministic=True):
    applied_seed = seed_everything(base_seed, fips_code, phase, deterministic)

    try:
        # Dynamically import the populations.pop<FIPS> module
        pop_module = importlib.import_module(f"populations.pop{fips_code}")
    except ModuleNotFoundError:
        print(f"Error: Population module populations.pop{fips_code} not found.")
        sys.exit(1)

    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop_module))

    # Configure simulation metadata in-memory for this FIPS and phase
    if "GENERATING_COUNTERFACTUAL" in os.environ:
        is_cf_env = os.environ["GENERATING_COUNTERFACTUAL"].lower() in ["true", "1"]
        sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL'] = is_cf_env

    sim.config['simulation_metadata']['metro_calibration_phase'] = phase
    sim.config['simulation_metadata']['POPULATION'] = fips_code
    sim.config['simulation_metadata']['random_seed'] = applied_seed

    is_cf = str(sim.config['simulation_metadata'].get('GENERATING_COUNTERFACTUAL', False)).lower() in ['true', '1']
    mode_str = "Counterfactual Generation" if is_cf else f"Metro Calibration Phase {phase}"

    print(f"\n=======================================================", flush=True)
    print(f"Running FIPS {fips_code} | Mode: {mode_str} | seed: {applied_seed}"
          f" (base {base_seed}, deterministic={deterministic})", flush=True)
    print(f"=======================================================", flush=True)

    # Update the population directory to point to the current FIPS code folder
    original_pop_dir = sim.config['simulation_metadata']['population_dir']
    parent_dir = os.path.dirname(original_pop_dir)
    sim.config['simulation_metadata']['population_dir'] = os.path.join(parent_dir, f"pop{fips_code}")

    runner = sim._get_runner(sim.config)
    runner.init()

    abm_nets.eval_net(sim, runner)

def main():
    parser = argparse.ArgumentParser(description="Run COVID ABM calibration / counterfactuals for one or more counties.")
    parser.add_argument("fips", help="Comma-separated FIPS code(s), e.g. '01009' or '01009,01031'.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (overrides the SEED env var; default %d)." % DEFAULT_BASE_SEED)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                        help="Disable torch deterministic algorithms / cudnn determinism.")
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    base_seed = resolve_base_seed(args.seed)

    fips_list = [f.strip() for f in args.fips.split(',')]
    fips_list = [f for f in fips_list if f]  # Remove empty entries

    if len(fips_list) == 1:
        run_county_phase(fips_list[0], phase=0, base_seed=base_seed, deterministic=args.deterministic)
    elif len(fips_list) > 1:
        # Phase 1: Factoring/Calibration Phase
        for fips in fips_list:
            run_county_phase(fips, phase=1, base_seed=base_seed, deterministic=args.deterministic)

        # Phase 2: Calibration with Commuter Interaction
        for fips in fips_list:
            run_county_phase(fips, phase=2, base_seed=base_seed, deterministic=args.deterministic)
    else:
        print("Error: No valid FIPS codes provided.")
        sys.exit(1)

if __name__ == '__main__':
    main()
