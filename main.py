#!/usr/bin/env python3
import os

# NOTE: must be set before any CUDA / cuBLAS context is created, otherwise
# torch.use_deterministic_algorithms(True) will raise for cuBLAS-backed ops
# (matmul, addmm, ...). Setting it here covers both `python3 main.py <fips>`
# runs and the SLURM array jobs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import yaml

import json

# The model config: config.yaml at the repo root, or the file named by ABM_CONFIG.
# AgentTorch's DataLoader reads the same ABM_CONFIG.
CONFIG_PATH = os.path.abspath(os.environ.get("ABM_CONFIG") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
os.environ["ABM_CONFIG"] = CONFIG_PATH


def beta_daily_spec(beta_daily):
    """BETA_DAILY from config.yaml -> {"start_day", "values"} for transition.py.
      list              one beta per day starting on the first observed day (2020-10-26)
      path to a .csv    column `beta`; if it also has a `day` column (simulation step, as in
                        run.sh's beta_daily.csv) rows are placed by day, burn-in included
    Relative paths are relative to the repo root."""
    if beta_daily is None:
        return None
    burn_in = int(yaml.safe_load(open(CONFIG_PATH))["simulation_metadata"].get("BURN_IN_DAYS", 0) or 0)
    if isinstance(beta_daily, str):
        import pandas as pd
        path = beta_daily if os.path.isabs(beta_daily) else os.path.join(os.path.dirname(os.path.abspath(__file__)), beta_daily)
        df = pd.read_csv(path)
        if "day" in df.columns:
            df = df.sort_values("day")
            days = df["day"].astype(int).tolist()
            if days != list(range(days[0], days[0] + len(days))):
                raise ValueError(f"BETA_DAILY csv {path}: `day` must be consecutive")
            return {"start_day": days[0], "values": df["beta"].astype(float).tolist()}
        return {"start_day": burn_in, "values": df["beta"].astype(float).tolist()}
    return {"start_day": burn_in, "values": [float(v) for v in beta_daily]}


def export_run_settings(config_path=CONFIG_PATH):
    """Export config.yaml's `run_settings` and the beta keys of `epidemic` as the
    environment variables abm_nets.py / transition.py read. Variables already set
    in the shell win, so e.g. `CF_TYPES=7 bash run.sh` still works."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    epi = cfg.get("epidemic", {})
    env = dict(cfg.get("run_settings") or {})
    env.update({
        "R_SPLINE_R_MIN": epi["BETA_MIN"],
        "R_MAX": epi["BETA_MAX"],
        "R_SPLINE_INIT_R": epi["BETA_INIT"],
        "BURN_IN_R": epi["BETA_BURN_IN"],
        # run.sh only: fixed values instead of the calibrated ones (null = calibrated)
        "BETA_OVERRIDE": json.dumps(epi.get("BETA")),
        "I0_OVERRIDE": json.dumps(epi.get("INITIAL_INFECTED")),
        "K_OVERRIDE": json.dumps(epi.get("K")),
        "BETA_DAILY_OVERRIDE": json.dumps(beta_daily_spec(epi.get("BETA_DAILY"))),
        "HORIZON_DAYS": epi.get("HORIZON_DAYS") or "",
    })
    for key, value in env.items():
        os.environ.setdefault(key, str(value))


export_run_settings()

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

    # HORIZON_DAYS (config.yaml, runs only): number of days simulated after the burn-in
    horizon = os.environ.get("HORIZON_DAYS", "").strip()
    if horizon and os.environ.get("GENERATING_COUNTERFACTUAL", "").lower() in ("true", "1"):
        md = sim.config['simulation_metadata']
        burn_in = int(md.get('BURN_IN_DAYS', 0) or 0)
        max_days = md['num_steps_per_episode'] - burn_in
        if not 1 <= int(horizon) <= max_days:
            print(f"Error: HORIZON_DAYS={horizon} must be between 1 and {max_days} "
                  f"(interventions and networks exist for {max_days} days).")
            sys.exit(1)
        sim.data_loader.set_config_attribute('num_steps_per_episode', burn_in + int(horizon))
        sim.config = sim.data_loader.get_config()

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
