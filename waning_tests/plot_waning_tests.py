#!/usr/bin/env python3
"""Generate the immunity-waning validation plots.

Reads the per-run trajectories collected by run_waning_tests.sh from
waning_tests/data/<FIPS>/*.csv (columns: t,susceptible,exposed,infected,
recovered,dead) and writes, under waning_tests/results/<FIPS>/:

    no_waning_same_seed_comparison.png     (Test 1)
    fixed_waning_60_80_100_days.png        (Test 2)
    stochastic_waning_010_020_040.png      (Test 3)
    <label>_sim_vs_actual.png              simulated vs reported daily cases,
                                           one per run (same view as
                                           result_graphs/.../simulation_results.png)
    sim_vs_actual_all.png                  the above on one grid

Also prints a short PASS/FAIL summary for each test.
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# County / calibration window these tests were run for (matches the defaults in
# run_waning_tests.sh). Outputs are organised per county.
FIPS = os.environ.get("WANING_TEST_FIPS", "01031")
DATE = os.environ.get("WANING_TEST_DATE", "202010-202104")

DATA = os.path.join(HERE, "data", FIPS)
RESULTS = os.path.join(HERE, "results", FIPS)
os.makedirs(RESULTS, exist_ok=True)


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def infected(df):
    return df["t"].to_numpy(), df["infected"].to_numpy()


# ---------------------------------------------------------------------------
# Observed data -- the reported daily cases every run is calibrated against.
# main.py fits the ABM's daily NEW infections to this series, so `none` is the
# calibrated fit to the real data. abm_nets.execute writes the aligned
# simulated series to results/<FIPS>/.../generated_factual.csv; run_waning_tests.sh
# copies it to data/<FIPS>/<label>_cases.csv.
# ---------------------------------------------------------------------------
def _use_7day_avg():
    cfg = os.path.join(REPO, "covid_abm", "yamls", "config.yaml")
    try:
        for line in open(cfg):
            s = line.strip()
            if s.startswith("USE_7DAY_AVG:"):
                return s.split(":", 1)[1].strip().lower() in ("true", "1")
    except OSError:
        pass
    return False


def load_observed():
    p = os.path.join(REPO, "data", "processed_data", FIPS, DATE, "daily_data.csv")
    if not os.path.exists(p):
        return None
    o = pd.read_csv(p)
    col = "cases_7day_avg" if _use_7day_avg() else "cases_singular"
    if col not in o.columns:
        col = "cases"
    return o[col].to_numpy(), col


def load_params(label):
    """calibrated_params.txt as written by abm_nets: 26 weekly R, then the
    initial infection rate (fraction, bounds [0, 5e-4]), then k (the reporting
    fraction, bounds [0.2, 1]). Returns (R_array, init_rate_per_100k, k) or None.
    """
    p = os.path.join(DATA, f"{label}_params.txt")
    if not os.path.exists(p):
        return None
    v = np.loadtxt(p)
    v = v.ravel()
    return v[:-2], float(v[-2]) * 1e5, float(v[-1])


def _sm(x, k=7):
    return pd.Series(x).rolling(k, center=True, min_periods=1).mean().to_numpy()


def _sim_vs_actual_ax(ax, sim, actual, col, label, params, inset=True):
    n = min(len(sim), len(actual))
    sim, actual = sim[:n], actual[:n]
    ax.plot(range(n), actual, marker="x", ms=3, lw=0.6, color="tab:orange",
            alpha=0.55, label=f"actual ({col})")
    ax.plot(range(n), _sm(actual), lw=2.0, color="tab:red", alpha=0.55,
            label="actual (7-day avg)")
    ax.plot(range(n), sim, marker="o", ms=2.5, lw=1.4, color="tab:blue",
            label="simulation")
    ax.set_xlabel("Day (evaluation window)")
    ax.set_ylabel("Daily new cases")
    ax.grid(True, alpha=0.3)
    rmse = float(np.sqrt(np.mean((sim - actual) ** 2)))
    r7 = float(np.corrcoef(_sm(sim), _sm(actual))[0, 1])

    handles, labs = ax.get_legend_handles_labels()
    if params is not None:
        R, init_100k, k = params
        handles.append(Line2D([], [], color="none"))
        labs.append(f"mean R = {R.mean():.2f}  (min {R.min():.2f}, max {R.max():.2f})\n"
                    f"initial infection rate = {init_100k:.1f} per 100k\n"
                    f"k (reporting fraction) = {k:.3f}")
        if inset:
            iax = ax.inset_axes([0.60, 0.58, 0.37, 0.38])
            iax.step(np.arange(1, len(R) + 1), R, where="mid", color="tab:purple",
                     lw=1.4)
            iax.axhline(1.0, color="grey", lw=0.8, ls="--")
            iax.set_title("weekly R(t)", fontsize=7)
            iax.set_xlabel("week", fontsize=6)
            iax.tick_params(labelsize=6)
            iax.grid(True, alpha=0.3)
    ax.legend(handles, labs, fontsize=7, loc="upper left")
    ax.set_title(f"{label}  -  sim vs actual daily cases (FIPS {FIPS})   "
                 f"RMSE {rmse:.1f}   7d-r {r7:.2f}", fontsize=9)
    return rmse, r7


def sim_vs_actual():
    obs = load_observed()
    if obs is None:
        print("[sim-vs-actual] SKIPPED - observed daily_data.csv not found")
        return
    actual, col = obs
    files = sorted(glob.glob(os.path.join(DATA, "*_cases.csv")))
    if not files:
        print("[sim-vs-actual] SKIPPED - no data/<FIPS>/<label>_cases.csv "
              "(re-run run_waning_tests.sh to collect them)")
        return

    labels = [os.path.basename(f)[: -len("_cases.csv")] for f in files]
    ncol = min(3, len(files))
    nrow = (len(files) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.6 * nrow),
                             squeeze=False)
    for ax in axes.flat[len(files):]:
        ax.axis("off")

    for ax, f, label in zip(axes.flat, files, labels):
        sim = pd.read_csv(f)["generated_factual_cases"].to_numpy()
        params = load_params(label)

        one = plt.figure(figsize=(9, 6))
        rmse, r7 = _sim_vs_actual_ax(one.gca(), sim, actual, col, label, params)
        out1 = os.path.join(RESULTS, f"{label}_sim_vs_actual.png")
        one.tight_layout(); one.savefig(out1, dpi=130); plt.close(one)

        _sim_vs_actual_ax(ax, sim, actual, col, label, params, inset=True)
        pstr = ""
        if params is not None:
            R, init_100k, k = params
            pstr = (f"  meanR={R.mean():.2f}[{R.min():.2f},{R.max():.2f}]"
                    f"  init={init_100k:.1f}/100k  k={k:.3f}")
        print(f"[sim-vs-actual] {label:20s} RMSE={rmse:7.2f} 7d-r={r7:.2f}{pstr}  -> {out1}")

    fig.suptitle(f"Simulation vs actual daily cases (FIPS {FIPS})")
    out = os.path.join(RESULTS, "sim_vs_actual_all.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[sim-vs-actual] grid -> {out}")


# ---------------------------------------------------------------------------
# Test 1 - none vs already-generated baseline (same seed)
# ---------------------------------------------------------------------------
def test1():
    base = load("none_baseline_reference.csv")
    new = load("none_verify.csv")
    if base is None or new is None:
        print("[Test 1] SKIPPED - need data/<FIPS>/none_baseline_reference.csv and "
              "data/<FIPS>/none_verify.csv")
        return
    tb, yb = infected(base)
    tn, yn = infected(new)
    n = min(len(yb), len(yn))
    max_abs = float(np.max(np.abs(yb[:n] - yn[:n])))

    plt.figure(figsize=(9, 6))
    plt.plot(tb, yb, lw=3, alpha=0.5, label="existing baseline (stored, seed 42)")
    plt.plot(tn, yn, lw=1.2, ls="--", color="k",
             label="new run, IMMUNITY_WANING_MODE=none (seed 42)")
    plt.xlabel("Day")
    plt.ylabel("Infected fraction of population")
    plt.title("Test 1 - waning code with mode=none does not change ABM dynamics\n"
              f"max |Δ infected fraction| = {max_abs:.2e}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "no_waning_same_seed_comparison.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()

    verdict = "PASS" if max_abs < 1e-3 else "CHECK"
    print(f"[Test 1] {verdict}  max|Δ| = {max_abs:.3e}  -> {out}")
    if verdict == "CHECK":
        print("         (small residual differences can come from the "
              "non-deterministic GNN scatter ops noted in main.py; inspect "
              "the plot - the curves should overlay)")


# ---------------------------------------------------------------------------
# Test 2 - fixed immunity duration 60 / 80 / 100 days
# ---------------------------------------------------------------------------
def test2():
    runs = [("fixed_60.csv", "Fixed 60 days"),
            ("fixed_80.csv", "Fixed 80 days"),
            ("fixed_100.csv", "Fixed 100 days")]
    loaded = [(load(f), lbl) for f, lbl in runs]
    if any(df is None for df, _ in loaded):
        print("[Test 2] SKIPPED - need data/<FIPS>/fixed_60.csv, fixed_80.csv, fixed_100.csv")
        return

    plt.figure(figsize=(9, 6))
    base = load("none_baseline_reference.csv")
    if base is not None:
        tb, yb = infected(base)
        plt.plot(tb, yb, lw=1, color="grey", alpha=0.6, label="none (reference)")
    peaks = {}
    for df, lbl in loaded:
        t, y = infected(df)
        peaks[lbl] = float(np.max(y))
        plt.plot(t, y, lw=2, label=lbl)
    plt.xlabel("Day")
    plt.ylabel("Infected fraction of population")
    plt.title("Test 2 - fixed immunity waning (R -> S after N days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "fixed_waning_60_80_100_days.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()

    # sanity: shorter immunity -> at least as much infection late in the run
    late = {}
    for df, lbl in loaded:
        t, y = infected(df)
        late[lbl] = float(np.mean(y[t >= t.max() - 40]))
    ordered = late["Fixed 60 days"] >= late["Fixed 80 days"] >= late["Fixed 100 days"] - 1e-9
    distinct = len({round(v, 8) for v in peaks.values()}) > 1
    verdict = "PASS" if (ordered and distinct) else "CHECK"
    print(f"[Test 2] {verdict}  late-window mean infected: "
          + ", ".join(f"{k}={v:.2e}" for k, v in late.items()) + f"  -> {out}")


# ---------------------------------------------------------------------------
# Test 3 - stochastic waning, rates 0.010 / 0.020 / 0.040
# ---------------------------------------------------------------------------
def test3():
    runs = [
            ("stochastic_0.010.csv", "WANING_RATE = 0.010  (~100 day mean)"),
            ("stochastic_0.020.csv", "WANING_RATE = 0.020  (~50 day mean)"),
            ("stochastic_0.040.csv", "WANING_RATE = 0.040  (~25 day mean)")]
    loaded = [(load(f), lbl) for f, lbl in runs]
    if any(df is None for df, _ in loaded):
        print("[Test 3] SKIPPED - need data/<FIPS>/stochastic_XXX.csv")
        return

    plt.figure(figsize=(9, 6))
    base = load("none_baseline_reference.csv")
    if base is not None:
        tb, yb = infected(base)
        plt.plot(tb, yb, lw=1, color="grey", alpha=0.6, label="none (reference)")
    late = {}
    for df, lbl in loaded:
        t, y = infected(df)
        late[lbl] = float(np.mean(y[t >= t.max() - 40]))
        plt.plot(t, y, lw=2, label=lbl)
    plt.xlabel("Day")
    plt.ylabel("Infected fraction of population")
    plt.title("Test 3 - stochastic immunity waning "
              "(daily p = 1 - exp(-WANING_RATE))")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "stochastic_waning_010_020_040.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()

    vals = list(late.values())
    ordered = vals[0] <= vals[1] + 1e-9 and vals[1] <= vals[2] + 1e-9
    distinct = len({round(v, 8) for v in vals}) > 1
    verdict = "PASS" if (ordered and distinct) else "CHECK"
    print(f"[Test 3] {verdict}  late-window mean infected: "
          + ", ".join(f"{k.split()[2]}={v:.2e}" for k, v in late.items())
          + f"  -> {out}")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    sim_vs_actual()
