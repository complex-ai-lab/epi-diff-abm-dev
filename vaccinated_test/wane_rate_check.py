#!/usr/bin/env python3
"""Direct check that R->S and VACCINATED->S use their configured rates,
independently.

The compartment fractions in vaccinated_test/data/*.csv cannot be used to fit a
decay rate: vaccination inflow runs the whole 182-day window (intervention.csv)
and recovered inflow tracks the epidemic, so neither compartment is a clean
exponential. Instead this script uses the per-step waning instrumentation
(NewTransmission.waning_events_history): for each step it logs the eligible
R / VACCINATED pool and how many of each waned that step. The empirical daily
hazard  sum(waned) / sum(pool)  must match  1 - exp(-rate)  for each state.

Forward sims only (frozen `none` calibrated params) - no recalibration, ~1 min
each. Writes vaccinated_test/wane_rate_check.png and prints a table.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import importlib
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import abm_nets
import covid_abm
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from main import seed_everything

FIPS = "01031"
PARAMS = os.path.join(
    REPO, "waning_tests", "frozen_cal_params_iso_wane", "frozen_params_none.txt"
)
HERE = os.path.join(REPO, "vaccinated_test")
OUTDIR = os.path.join(HERE, "data", FIPS)
RESULTS = os.path.join(HERE, "results", FIPS)

# Stochastic only: for a fixed timer "waned / pool" per step is not a hazard
# (it depends on the age structure of the pool), so an empirical-vs-configured
# comparison is only meaningful in stochastic mode. Fixed-mode independence is
# checked in plot_vacc_tests.py via the VACCINATED->S onset day (= 86 + V).
# (label, mode, R_to_S, V_to_S, R_rate, V_rate)
CONFIGS = [
    ("stoch_R0.01_V0.00667", "stochastic", None, None, 0.010, 0.006667),
    ("stoch_R0.02_V0.01333", "stochastic", None, None, 0.020, 0.013333),
    ("stoch_R0.04_V0.02667", "stochastic", None, None, 0.040, 0.026667),
]


def run_one(label, mode, r_to_s, v_to_s, r_rate, v_rate):
    ev_path = os.path.join(OUTDIR, f"waning_events_{label}.csv")
    if os.path.exists(ev_path) and "--force" not in sys.argv:
        print(f"[reuse] {ev_path}")
        return pd.read_csv(ev_path)

    seed_everything(42, FIPS, 0, True)
    pop = importlib.import_module(f"populations.pop{FIPS}")
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop))
    md = sim.config["simulation_metadata"]
    md["metro_calibration_phase"] = 0
    md["POPULATION"] = FIPS
    md["GENERATING_COUNTERFACTUAL"] = False
    md["IMMUNITY_WANING_MODE"] = mode
    if r_to_s is not None:
        md["RECOVERED_TO_SUSCEPTIBLE_TIME"] = r_to_s
    if v_to_s is not None:
        md["VACCINATED_TO_SUSCEPTIBLE_TIME"] = v_to_s
    if r_rate is not None:
        md["RECOVERED_WANING_RATE"] = r_rate
    if v_rate is not None:
        md["VACCINATED_WANING_RATE"] = v_rate
    md["population_dir"] = os.path.join(
        os.path.dirname(md["population_dir"]), f"pop{FIPS}"
    )

    runner = sim._get_runner(sim.config)
    runner.init()
    nsteps, nw = md["num_steps_per_episode"], md["NUM_WEEKS"]
    p = torch.tensor(np.loadtxt(PARAMS), dtype=torch.float,
                     device=abm_nets.DEVICE)[:, None]
    lp = [(n, x) for (n, x) in runner.named_parameters()]
    abm_nets.map_and_replace_tensor(lp[1][0])(runner, True, p[:nw], mode_calibrate=True)
    abm_nets.map_and_replace_tensor(lp[3][0])(runner, True, p[-2], mode_calibrate=True)
    abm_nets.map_and_replace_tensor(lp[4][0])(runner, True, p[-1], mode_calibrate=True)

    with torch.no_grad():
        runner.step(nsteps)

    sub = runner.initializer.transition_function["0"].new_transmission
    sub.save_waning_events_to_disk(ev_path)
    return pd.read_csv(ev_path)


def hazard(pool, waned):
    pool, waned = np.asarray(pool), np.asarray(waned)
    m = pool > 0
    return float(waned[m].sum() / pool[m].sum()) if m.any() else float("nan")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 4.5),
                             squeeze=False)
    for ax, (label, mode, r_to_s, v_to_s, r_rate, v_rate) in zip(axes[0], CONFIGS):
        df = run_one(label, mode, r_to_s, v_to_s, r_rate, v_rate)
        # expected daily probability
        if mode == "stochastic":
            exp_r, exp_v = 1 - math.exp(-r_rate), 1 - math.exp(-v_rate)
        else:
            exp_r, exp_v = 1.0 / r_to_s, 1.0 / v_to_s  # rough: 1 cohort/day out
        emp_r = hazard(df["recovered_pool"], df["recovered_waned"])
        emp_v = hazard(df["vaccinated_pool"], df["vaccinated_waned"])
        rows.append((label, mode, exp_r, emp_r, exp_v, emp_v,
                     emp_r / emp_v if emp_v else float("nan")))

        # daily hazard time series (where pool > 0)
        for col, lbl, c in [("recovered", "R->S", "tab:green"),
                            ("vaccinated", "VACCINATED->S", "tab:purple")]:
            pool = df[f"{col}_pool"].to_numpy()
            wan = df[f"{col}_waned"].to_numpy()
            m = pool > 20
            ax.plot(df["t"][m], wan[m] / pool[m], ".", ms=4, color=c, alpha=0.5,
                    label=f"{lbl} daily")
        if mode == "stochastic":
            ax.axhline(exp_r, color="tab:green", ls="--", lw=1)
            ax.axhline(exp_v, color="tab:purple", ls="--", lw=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Day"); ax.set_ylabel("waned / pool")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("Per-step waning hazard: dashed = configured 1-exp(-rate)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "wane_rate_check.png"), dpi=130)
    plt.close(fig)

    print(f"\n{'config':22s} {'mode':11s} "
          f"{'exp_R':>8s} {'emp_R':>8s} {'exp_V':>8s} {'emp_V':>8s} {'R/V':>6s}")
    for label, mode, er, mr, ev, mv, ratio in rows:
        print(f"{label:22s} {mode:11s} {er:8.4f} {mr:8.4f} {ev:8.4f} {mv:8.4f} "
              f"{ratio:6.2f}")
    print("\nstochastic: emp_R ~ exp_R and emp_V ~ exp_V independently; "
          "R/V ~ configured rate ratio (1.5).")


if __name__ == "__main__":
    main()
