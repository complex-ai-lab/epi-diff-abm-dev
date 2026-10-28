"""Run a SEIRV model with the rates in a seir_rates_daily.csv and compare it with the ABM run they came from.

    .venv/bin/python seirv_check.py --rates reference_results/01031/seir_rates_daily.csv
    .venv/bin/python seirv_check.py --rates my_rates.csv --abm reference_results/01031 --out results/my_check

--rates  a seir_rates_daily.csv (seirv_fit.py / run.sh): per day beta, sigma, gamma, doses, and k.
--abm    the ABM run to compare with, in the reference_results format (seirv_daily.csv,
         cases_sim_vs_observed.csv, config_<FIPS>.yaml or config_used.yaml); default: the folder of --rates.
--out    output folder; default results/seirv_check/<folder of --abm's parent>/<folder of --abm>.

The SEIRV model starts from the ABM's day-0 S, E, I, R, V and takes only beta(t), sigma, gamma, doses(t) and k
from the rates file, plus the waning setting from the ABM's config; it computes S, E, I, R, V itself:

    new_E = beta(t) * S * I / N             S -> E
    vac   = min(doses(t), S - new_E)        S -> V
    E -> I: sigma * E,   I -> R: gamma * I
    R -> S, V -> S: none: no waning; fixed: the cohorts that entered R / V RECOVERED_TO_SUSCEPTIBLE_TIME /
                    VACCINATED_TO_SUSCEPTIBLE_TIME days ago; stochastic: R, V x (1 - exp(-*_WANING_RATE))
    cases = k * new_E

Writes seirv_check.csv (rates, both models' compartments and cases, observed cases), cases.png (daily cases:
ABM, SEIRV, observed) and seirv.png (S, E, I, R, V: ABM vs SEIRV, and beta).
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seirv_fit import run_config

ABM, SEIRV, OBS = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def simulate_seirv(rates, start, N, epi):
    """Closed SEIRV model driven by the rates, exponential E and I. start: S, E, I, R, V on the first day."""
    T = len(rates)
    beta, doses = rates["beta"].to_numpy(), rates["doses"].to_numpy()
    sigma, gamma = rates["sigma"].iloc[0], rates["gamma"].iloc[0]
    mode = epi.get("IMMUNITY_WANING_MODE", "none")
    r_wane, v_wane = epi.get("RECOVERED_TO_SUSCEPTIBLE_TIME"), epi.get("VACCINATED_TO_SUSCEPTIBLE_TIME")
    p_r = 1 - math.exp(-epi.get("RECOVERED_WANING_RATE", 0.0))
    p_v = 1 - math.exp(-epi.get("VACCINATED_WANING_RATE", 0.0))

    m = np.zeros((T, 5)); m[0] = start
    into_R = np.zeros(T); into_V = np.zeros(T); new_E = np.full(T, np.nan)
    for t in range(1, T):
        s, e, i, r, v = m[t - 1]
        ne = beta[t] * s * i / N
        vac = min(doses[t], s - ne)
        ei_, ir_ = sigma * e, gamma * i
        if mode == "fixed":
            wr = into_R[t - r_wane] if t >= r_wane else 0
            wv = into_V[t - v_wane] if t >= v_wane else 0
        elif mode == "stochastic":
            wr, wv = r * p_r, v * p_v
        else:
            wr = wv = 0
        into_R[t], into_V[t], new_E[t] = ir_, vac, ne
        m[t] = s - ne - vac + wr + wv, e + ne - ei_, i + ei_ - ir_, r + ir_ - wr, v + vac - wv
    return m, new_E


def plot(out, name, N, out_dir):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    out = out.assign(date=pd.to_datetime(out["date"]))
    w = out[out.cases_abm.notna()]

    def fmt(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(w.date, w.cases_observed, color=OBS, lw=1.2, label="Observed cases", zorder=1)
    ax.plot(w.date, w.cases_abm, color=ABM, lw=2, label="ABM (network)", zorder=2)
    ax.plot(w.date, w.cases_seirv, color=SEIRV, lw=2, ls=(0, (4, 2)), label="SEIRV (inferred β)", zorder=3)
    ax.set_ylabel("Daily cases"); ax.set_ylim(bottom=0); fmt(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(f"{name}: daily cases, ABM vs SEIRV vs observed", x=0.07, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cases.png"), dpi=150); plt.close(fig)

    names = {"S": "Susceptible", "E": "Exposed", "I": "Infected", "R": "Recovered", "V": "Vaccinated"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, c in zip(axes.flat, "SEIRV"):
        ax.plot(out.date, 100 * out[c + "_abm"] / N, color=ABM, lw=2, label="ABM")
        ax.plot(out.date, 100 * out[c + "_seirv"] / N, color=SEIRV, lw=2, ls=(0, (4, 2)), label="SEIRV")
        gap = 100 * (out[c + "_seirv"] - out[c + "_abm"]).abs().max() / N
        ax.set_title(f"{names[c]} ({c}), max gap {gap:.2f}% of N", loc="left", color=INK, fontsize=10)
        ax.set_ylabel("% of population"); fmt(ax)
    axes[0, 0].legend(frameon=False)
    last = axes.flat[5]
    last.plot(w.date, w.beta, color=INK, lw=1.5)
    last.set_title(f"Inferred β per day (σ = {w.sigma.iloc[0]:.3f}, γ = {w.gamma.iloc[0]:.3f})",
                   loc="left", color=INK, fontsize=10); fmt(last)
    fig.suptitle(f"{name}: compartments, ABM vs SEIRV (N = {N:,.0f})", x=0.04, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "seirv.png"), dpi=150); plt.close(fig)


def main(rates_path, abm_dir=None, out_dir=None):
    abm_dir = os.path.normpath(abm_dir or os.path.dirname(rates_path) or ".")
    out_dir = out_dir or os.path.join("results", "seirv_check", os.path.basename(os.path.dirname(abm_dir)) or ".",
                                      os.path.basename(abm_dir))
    _, epi = run_config(abm_dir)
    rates = pd.read_csv(rates_path)
    k = float(rates["k"].iloc[0])
    seirv = pd.read_csv(os.path.join(abm_dir, "seirv_daily.csv")).set_index("day")
    cases = pd.read_csv(os.path.join(abm_dir, "cases_sim_vs_observed.csv")).set_index("day")
    abm = seirv.reindex(rates["day"])
    N = abm[list("SEIRV")].iloc[0].sum()
    m, new_E = simulate_seirv(rates, abm[list("SEIRV")].iloc[0].to_numpy(), N, epi)

    out = rates.drop(columns=["new_E", "k"]).assign(
        **{f"{c}_abm": abm[c].values for c in "SEIRV"},
        **{f"{c}_seirv": m[:, j] for j, c in enumerate("SEIRV")},
        cases_abm=cases["simulated_cases"].reindex(rates["day"]).values,
        cases_seirv=k * new_E,
        cases_observed=cases["observed_cases"].reindex(rates["day"]).values,
    )
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "seirv_check.csv"), index=False)
    plot(out, f"{os.path.basename(os.path.dirname(abm_dir))}/{os.path.basename(abm_dir)}", N, out_dir)

    w = out[out.cases_abm.notna()]
    err = (w.cases_seirv - w.cases_abm).abs().sum() / w.cases_abm.sum()
    gap = max((out[f"{c}_seirv"] - out[f"{c}_abm"]).abs().max() for c in "SEIRV") / N
    print(f"{abm_dir}: cases err {err:.1%} corr {w.cases_seirv.corr(w.cases_abm):.3f}  "
          f"total {w.cases_seirv.sum():.0f}/{w.cases_abm.sum():.0f}  max compartment gap {gap:.2%} of N  -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", required=True, help="seir_rates_daily.csv to take beta, sigma, gamma, doses, k from")
    ap.add_argument("--abm", help="ABM run folder (reference_results format); default: the folder of --rates")
    ap.add_argument("--out", help="output folder; default results/seirv_check/<parent>/<folder>")
    a = ap.parse_args()
    main(a.rates, a.abm, a.out)
