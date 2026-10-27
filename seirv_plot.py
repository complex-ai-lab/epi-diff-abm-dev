"""Plots for a seirv_fit.py CSV: cases (ABM vs SEIRV vs observed) and S/E/I/R/V (ABM vs SEIRV).

    .venv/bin/python seirv_plot.py results/seirv_fit/01031.csv     # -> results/seirv_fit/01031_cases.png, 01031_seirv.png
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import pandas as pd

ABM, SEIRV, OBS = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})

path = sys.argv[1]
fips = os.path.basename(path)[:-4]
a = pd.read_csv(path, parse_dates=["date"])
N = a[[c + "_abm" for c in "SEIRV"]].iloc[0].sum()
w = a[a.cases_abm.notna()]

def fmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# --- cases --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(w.date, w.cases_observed, color=OBS, lw=1.2, label="Observed cases", zorder=1)
ax.plot(w.date, w.cases_abm, color=ABM, lw=2, label="ABM (network)", zorder=2)
ax.plot(w.date, w.cases_seirv, color=SEIRV, lw=2, ls=(0, (4, 2)), label="SEIRV (inferred β)", zorder=3)
ax.set_ylabel("Daily cases"); ax.set_ylim(bottom=0); fmt(ax)
ax.legend(frameon=False, loc="upper right")
fig.suptitle(f"{fips}: daily cases, ABM vs SEIRV vs observed", x=0.07, ha="left", color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(path[:-4] + "_cases.png", dpi=150); plt.close(fig)

# --- compartments (small multiples, % of N, each panel its own scale) ----------
names = {"S": "Susceptible", "E": "Exposed", "I": "Infected", "R": "Recovered", "V": "Vaccinated"}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
for ax, c in zip(axes.flat, "SEIRV"):
    ax.plot(a.date, 100 * a[c + "_abm"] / N, color=ABM, lw=2, label="ABM")
    ax.plot(a.date, 100 * a[c + "_seirv"] / N, color=SEIRV, lw=2, ls=(0, (4, 2)), label="SEIRV")
    gap = 100 * (a[c + "_seirv"] - a[c + "_abm"]).abs().max() / N
    ax.set_title(f"{names[c]} ({c}), max gap {gap:.2f}% of N", loc="left", color=INK, fontsize=10)
    ax.set_ylabel("% of population"); fmt(ax)
axes[0, 0].legend(frameon=False)
last = axes.flat[5]
last.plot(w.date, w.beta, color=INK, lw=1.5)
last.set_title(f"Inferred β per day (σ = {w.sigma.iloc[0]:.3f}, γ = {w.gamma.iloc[0]:.3f})",
               loc="left", color=INK, fontsize=10); fmt(last)
fig.suptitle(f"{fips}: compartments, ABM vs SEIRV (N = {N:,.0f})", x=0.04, ha="left", color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(path[:-4] + "_seirv.png", dpi=150); plt.close(fig)
print("wrote", path[:-4] + "_cases.png", path[:-4] + "_seirv.png")
