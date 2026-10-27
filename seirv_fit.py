"""Infer SEIRV rates from an ABM run and check a closed SEIRV model with them reproduces the run.

    .venv/bin/python seirv_fit.py reference_results/01031 results/seirv_fit/01031.csv

Inferred from the run (seirv_daily.csv, cases_sim_vs_observed.csv, config_<FIPS>.yaml):
  new_E(t)  cases/K after the burn-in; S(t-1) - S(t) in the burn-in (no vaccines or waning yet)
  beta(t)   new_E * N / (S(t-1) * I(t-1))
  doses(t)  V(t) - V(t-1) + doses that waned (vaccinated VACCINATED_TO_SUSCEPTIBLE_TIME days earlier)
  sigma_eff, gamma_eff  E->I and I->R flows divided by E(t-1), I(t-1)
The SEIRV model starts from the ABM's day 0 and only gets beta(t), sigma = 1/E->I, gamma = 1/I->R,
doses(t) and the fixed waning times (R->S, V->S) as inputs; it computes S, E, I, R, V itself.
"""
import glob, os, sys
import numpy as np, pandas as pd, yaml

run_dir, out_path = sys.argv[1], sys.argv[2]
epi = yaml.safe_load(open(glob.glob(os.path.join(run_dir, "config_*.yaml"))[0]))["epidemic"]
assert epi["IMMUNITY_WANING_MODE"] == "fixed"
K, ei, ir = epi["K"], epi["EXPOSED_TO_INFECTED_TIME"], epi["INFECTED_TO_RECOVERED_TIME"]
r_wane, v_wane = epi["RECOVERED_TO_SUSCEPTIBLE_TIME"], epi["VACCINATED_TO_SUSCEPTIBLE_TIME"]
sigma, gamma = 1 / ei, 1 / ir

cases = pd.read_csv(f"{run_dir}/cases_sim_vs_observed.csv").set_index("day")
abm = pd.read_csv(f"{run_dir}/seirv_daily.csv").set_index("day").loc[:cases.index[-1]]   # days with cases
r_t = pd.read_csv(f"{run_dir}/r_t_daily.csv").set_index("day")
T, burn = len(abm), cases.index[0]
N = abm[list("SEIRV")].iloc[0].sum()
S, E, I, R, V = (abm[c].to_numpy() for c in "SEIRV")

# --- infer inputs from the run -----------------------------------------------
newE = np.full(T, np.nan); doses = np.zeros(T)
for t in range(1, T):
    doses[t] = V[t] - V[t - 1] + (doses[t - v_wane] if t >= v_wane else 0)
    newE[t] = cases.loc[t, "simulated_cases"] / K if t >= burn else S[t - 1] - S[t] - doses[t]
beta = np.full(T, np.nan); beta[1:] = newE[1:] * N / (S[:-1] * I[:-1])
f_EI = np.full(T, np.nan); f_EI[1:] = E[:-1] + newE[1:] - E[1:]
f_IR = np.full(T, np.nan); f_IR[1:] = I[:-1] + f_EI[1:] - I[1:]
sigma_eff = np.full(T, np.nan); sigma_eff[1:] = f_EI[1:] / E[:-1]
gamma_eff = np.full(T, np.nan); gamma_eff[1:] = f_IR[1:] / I[:-1]

# --- closed SEIRV, exponential E and I, fixed-delay waning ----------------------
m = np.zeros((T, 5)); m[0] = S[0], E[0], I[0], R[0], V[0]
into_R = np.zeros(T); into_V = np.zeros(T); m_newE = np.full(T, np.nan)
for t in range(1, T):
    s, e, i, r, v = m[t - 1]
    ne = beta[t] * s * i / N
    vac = min(doses[t], s - ne)
    ei_, ir_ = sigma * e, gamma * i
    wr = into_R[t - r_wane] if t >= r_wane else 0
    wv = into_V[t - v_wane] if t >= v_wane else 0
    into_R[t], into_V[t], m_newE[t] = ir_, vac, ne
    m[t] = s - ne - vac + wr + wv, e + ne - ei_, i + ei_ - ir_, r + ir_ - wr, v + vac - wv

out = pd.DataFrame({
    "day": abm.index, "date": abm["date"], "phase": r_t["phase"].reindex(abm.index).values,
    "r_t": r_t["r_t"].reindex(abm.index).values,
    "beta": beta, "sigma": sigma, "gamma": gamma, "sigma_eff": sigma_eff, "gamma_eff": gamma_eff,
    "R_eff": beta * S / N / gamma, "doses": doses,
    **{f"{c}_abm": abm[c].values for c in "SEIRV"},
    **{f"{c}_seirv": m[:, j] for j, c in enumerate("SEIRV")},
    "cases_abm": cases["simulated_cases"].reindex(abm.index).values,
    "cases_seirv": K * m_newE,
    "cases_observed": cases["observed_cases"].reindex(abm.index).values,
})
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
out.to_csv(out_path, index=False)

w = out[out.day >= burn]
err = (w.cases_seirv - w.cases_abm).abs().sum() / w.cases_abm.sum()
gap = max((out[f"{c}_seirv"] - out[f"{c}_abm"]).abs().max() for c in "SEIRV") / N
print(f"{os.path.basename(run_dir)}  beta med {np.nanmedian(w.beta):.3f}  sigma_eff {np.nanmedian(w.sigma_eff):.3f}  "
      f"gamma_eff {np.nanmedian(w.gamma_eff):.3f}  cases err {err:.1%} corr {np.corrcoef(w.cases_seirv, w.cases_abm)[0,1]:.3f}  "
      f"total {w.cases_seirv.sum():.0f}/{w.cases_abm.sum():.0f}  max compartment gap {gap:.2%} of N")
