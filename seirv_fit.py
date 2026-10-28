"""SEIRV rates of an ABM run, inferred from its output: writes <dir>/seir_rates_daily.csv.

    .venv/bin/python seirv_fit.py reference_results/01031
    .venv/bin/python seirv_fit.py results/01031/run --k 0.285242      # --k only if the config has K: null

run.sh does this automatically (abm_nets.write_factual_run_csvs). <dir> needs seirv_daily.csv,
cases_sim_vs_observed.csv, r_t_daily.csv and the run's config (config_used.yaml or config_<FIPS>.yaml).

  new_E(t)  cases/K after the burn-in; S(t-1) - S(t) - doses(t) in the burn-in (no cases are reported there)
  beta(t)   new_E * N / (S(t-1) * I(t-1))
  sigma, gamma  1 / EXPOSED_TO_INFECTED_TIME, 1 / INFECTED_TO_RECOVERED_TIME
  doses(t)  V(t) - V(t-1) + V -> S waning on day t (none: 0; fixed: doses VACCINATED_TO_SUSCEPTIBLE_TIME days
            earlier; stochastic: V(t-1) * (1 - exp(-VACCINATED_WANING_RATE)))
  sigma_eff, gamma_eff  E->I and I->R flows divided by E(t-1), I(t-1)
  R_eff     beta * S / N / gamma

Columns: day, date, phase, r_t, beta, sigma, gamma, sigma_eff, gamma_eff, R_eff, doses, new_E, k.
seirv_check.py runs a SEIRV model with these rates and compares it with the ABM.
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
import yaml


def infer_rates(seirv, cases, r_t, k, epi):
    """Per-day SEIRV rates of a run. seirv: day, date, S, E, I, R, V; cases: day, simulated_cases (from the
    first reported day); r_t: day, phase, r_t; epi: the config's epidemic block. Days with cases only."""
    seirv = seirv.set_index("day").loc[:cases["day"].iloc[-1]]
    sim_cases = cases.set_index("day")["simulated_cases"]
    r_t = r_t.set_index("day")
    T, burn = len(seirv), int(cases["day"].iloc[0])
    N = seirv[list("SEIRV")].iloc[0].sum()
    S, E, I, V = (seirv[c].to_numpy() for c in "SEIV")
    mode = epi.get("IMMUNITY_WANING_MODE", "none")
    v_wane, v_rate = epi.get("VACCINATED_TO_SUSCEPTIBLE_TIME"), epi.get("VACCINATED_WANING_RATE")

    newE = np.full(T, np.nan); doses = np.zeros(T)
    for t in range(1, T):
        waned = (doses[t - v_wane] if mode == "fixed" and t >= v_wane else
                 V[t - 1] * (1 - math.exp(-v_rate)) if mode == "stochastic" else 0.0)
        doses[t] = V[t] - V[t - 1] + waned
        newE[t] = sim_cases.loc[t] / k if t >= burn else S[t - 1] - S[t] - doses[t]
    beta = np.full(T, np.nan); beta[1:] = newE[1:] * N / (S[:-1] * I[:-1])
    f_EI = np.full(T, np.nan); f_EI[1:] = E[:-1] + newE[1:] - E[1:]
    f_IR = np.full(T, np.nan); f_IR[1:] = I[:-1] + f_EI[1:] - I[1:]
    sigma_eff = np.full(T, np.nan); sigma_eff[1:] = f_EI[1:] / E[:-1]
    gamma_eff = np.full(T, np.nan); gamma_eff[1:] = f_IR[1:] / I[:-1]
    gamma = 1 / epi["INFECTED_TO_RECOVERED_TIME"]
    return pd.DataFrame({
        "day": seirv.index, "date": seirv["date"].values, "phase": r_t["phase"].reindex(seirv.index).values,
        "r_t": r_t["r_t"].reindex(seirv.index).values,
        "beta": beta, "sigma": 1 / epi["EXPOSED_TO_INFECTED_TIME"], "gamma": gamma,
        "sigma_eff": sigma_eff, "gamma_eff": gamma_eff, "R_eff": beta * S / N / gamma,
        "doses": doses, "new_E": newE, "k": k,
    })


def run_config(run_dir):
    """The epidemic block of a run folder's config (config_used.yaml, else config_<FIPS>.yaml)."""
    configs = [os.path.join(run_dir, "config_used.yaml")] + sorted(glob.glob(os.path.join(run_dir, "config_*.yaml")))
    config = next((c for c in configs if os.path.exists(c)), None)
    if config is None:
        raise SystemExit(f"{run_dir}: no config_used.yaml or config_<FIPS>.yaml")
    return config, yaml.safe_load(open(config))["epidemic"]


def main(run_dir, k=None):
    config, epi = run_config(run_dir)
    k = k if k is not None else epi.get("K")
    if k is None:
        raise SystemExit(f"{config}: K is null (the run used the calibrated K); pass it with --k")
    read = lambda name: pd.read_csv(os.path.join(run_dir, name))
    rates = infer_rates(read("seirv_daily.csv"), read("cases_sim_vs_observed.csv"), read("r_t_daily.csv"), k, epi)
    path = os.path.join(run_dir, "seir_rates_daily.csv")
    rates.to_csv(path, index=False)
    obs = rates[rates.phase == "observed"]
    print(f"{path}: beta median {obs.beta.median():.3f} ({obs.beta.min():.3f}..{obs.beta.max():.3f}), "
          f"sigma {obs.sigma.iloc[0]:.3f}, gamma {obs.gamma.iloc[0]:.3f}, k {k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--k", type=float, help="reporting fraction, if the config has K: null")
    a = ap.parse_args()
    main(a.run_dir, a.k)
