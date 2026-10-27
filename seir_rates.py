"""SEIR-equivalent rates of an ABM run, backed out from its output CSVs.

    .venv/bin/python seir_rates.py results/01031/run --k 0.285242
    .venv/bin/python seir_rates.py reference_results/13013 --out /tmp/seir_13013.csv

Reads r_t_daily.csv, seirv_daily.csv, cases_sim_vs_observed.csv and the run's config
(config_used.yaml, or config_<FIPS>.yaml in reference_results/<FIPS>/) and writes
seir_rates.csv (default: next to them), one row per simulated day after the burn-in.
K comes from the config; a run that used the calibrated K (K: null) needs --k, the last
row of its calibrated_params.txt. Columns:

  day, date, r_t        the ABM's R_t on that day (config R_T)
  new_exposed           agents newly exposed that day = simulated_cases / K
  S_prev, E_prev, I_prev
                        agent counts at the end of the previous day (the state the day's
                        infections were drawn from)
  beta_seir             new_exposed * N / (S_prev * I_prev): the beta of a classic SEIR
                        dS/dt = -beta S I / N that gives the same daily incidence
  r_eff                 beta_seir * S_prev / N * INFECTED_TO_RECOVERED_TIME: the SEIR effective
                        reproduction number
  beta_seir_over_r_t    beta_seir / r_t

sigma = 1 / EXPOSED_TO_INFECTED_TIME and gamma = 1 / INFECTED_TO_RECOVERED_TIME are printed.
The ABM's stage durations are fixed rather than exponential, exposed agents are also
infectious, and S also changes through vaccination and waning, so a SEIR model with these
rates matches the incidence day by day but not every compartment.
"""
import argparse
import glob
import os
import sys

import pandas as pd
import yaml


def main(run_dir, k=None, out_path=None):
    configs = ([os.path.join(run_dir, "config_used.yaml")]
               + sorted(glob.glob(os.path.join(run_dir, "config_*.yaml"))))
    config = next((c for c in configs if os.path.exists(c)), None)
    if config is None:
        sys.exit(f"{run_dir}: no config_used.yaml or config_<FIPS>.yaml")
    epi = yaml.safe_load(open(config))["epidemic"]
    k = k if k is not None else epi.get("K")
    if k is None:
        sys.exit(f"{config}: K is null (the run used the calibrated K); pass it with --k")
    e_i, i_r = epi["EXPOSED_TO_INFECTED_TIME"], epi["INFECTED_TO_RECOVERED_TIME"]

    r_t = pd.read_csv(os.path.join(run_dir, "r_t_daily.csv")).set_index("day")["r_t"]
    seirv = pd.read_csv(os.path.join(run_dir, "seirv_daily.csv")).set_index("day")
    cases = pd.read_csv(os.path.join(run_dir, "cases_sim_vs_observed.csv")).set_index("day")
    n = float(seirv[["S", "E", "I", "R", "V"]].iloc[0].sum())

    days = cases.index
    prev = seirv.reindex(days - 1)
    out = pd.DataFrame({
        "day": days,
        "date": cases["date"].values,
        "r_t": r_t.reindex(days).values,
        "new_exposed": cases["simulated_cases"].values / k,
        "S_prev": prev["S"].values,
        "E_prev": prev["E"].values,
        "I_prev": prev["I"].values,
    })
    out["beta_seir"] = out["new_exposed"] * n / (out["S_prev"] * out["I_prev"])
    out["r_eff"] = out["beta_seir"] * out["S_prev"] / n * i_r
    out["beta_seir_over_r_t"] = out["beta_seir"] / out["r_t"]
    path = out_path or os.path.join(run_dir, "seir_rates.csv")
    out.to_csv(path, index=False)

    q = out["beta_seir_over_r_t"].quantile([0.1, 0.5, 0.9]).values
    print(f"{run_dir}: N {n:.0f}, K {k:.4f}, sigma 1/{e_i} = {1 / e_i:.3f}/day, gamma 1/{i_r} = {1 / i_r:.3f}/day")
    print(f"  beta_seir: median {out['beta_seir'].median():.3f}/day "
          f"(range {out['beta_seir'].min():.3f}..{out['beta_seir'].max():.3f})")
    print(f"  r_eff:     median {out['r_eff'].median():.3f} (range {out['r_eff'].min():.3f}..{out['r_eff'].max():.3f})")
    print(f"  beta_seir / r_t: 10/50/90% {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--k", type=float, help="reporting fraction, if the config has K: null")
    ap.add_argument("--out", help="output CSV (default <run_dir>/seir_rates.csv)")
    a = ap.parse_args()
    main(a.run_dir, a.k, a.out)
