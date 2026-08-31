#!/usr/bin/env python3
"""Plots + PASS/FAIL summary for the VACCINATED-state validation.

Reads vaccinated_test/data/<FIPS>/*.csv (columns:
t,susceptible,exposed,infected,recovered,dead,vaccinated) and writes, under
vaccinated_test/results/<FIPS>/:

    none_vs_baseline.png
    none_compartments.png
    fixed_infected.png
    fixed_compartments.png
    stochastic_infected.png
    stochastic_compartments.png
    <label>_sim_vs_actual.png    simulated vs reported daily cases, one per run
                                 (same view as result_graphs/.../simulation_results.png)
    sim_vs_actual_all.png        the above on one grid

Baseline for Test A is waning_tests/data/<FIPS>/none_baseline_reference.csv (the
seed-42 `none` run generated before the VACCINATED state existed).
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
# run_vacc_tests.sh). Outputs are organised per county.
FIPS = os.environ.get("VACC_TEST_FIPS", "01031")
DATE = os.environ.get("VACC_TEST_DATE", "202010-202104")

DATA = os.path.join(HERE, "data", FIPS)
RESULTS = os.path.join(HERE, "results", FIPS)
os.makedirs(RESULTS, exist_ok=True)
BASELINE = os.path.join(REPO, "waning_tests", "data", FIPS,
                        "none_baseline_reference.csv")


def load(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def late_mean(df, col="infected", window=40):
    t = df["t"].to_numpy()
    return float(np.mean(df[col].to_numpy()[t >= t.max() - window]))


def half_fall_day(df, col):
    """Day `col` first drops below half its running peak, after that peak."""
    t = df["t"].to_numpy()
    y = df[col].to_numpy()
    if y.max() <= 0:
        return None
    pk = int(np.argmax(y))
    thr = y[pk] / 2.0
    after = np.where(y[pk:] < thr)[0]
    return int(t[pk + after[0]]) if len(after) else None


def decay_lambda(df, col):
    """LS fit of log(frac) ~ -lambda * t on the post-peak positive tail."""
    t = df["t"].to_numpy().astype(float)
    y = df[col].to_numpy().astype(float)
    if y.max() <= 0:
        return None
    pk = int(np.argmax(y))
    tt, yy = t[pk:], y[pk:]
    m = yy > 1e-5
    if m.sum() < 5:
        return None
    A = np.polyfit(tt[m] - tt[m][0], np.log(yy[m]), 1)
    return -float(A[0])


# ---------------------------------------------------------------------------
# Observed data -- the reported daily cases every run is calibrated against.
# main.py fits the ABM's daily NEW infections to this series, so `none` is the
# calibrated fit to the real data. abm_nets.execute writes the aligned
# simulated series to results/<FIPS>/.../generated_factual.csv; run_vacc_tests.sh
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
              "(re-run run_vacc_tests.sh to collect them)")
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
def test_a():
    new = load("none.csv")
    if new is None or not os.path.exists(BASELINE):
        print("[Test A] SKIPPED - need data/<FIPS>/none.csv and the baseline")
        return
    base = pd.read_csv(BASELINE)
    n = min(len(new), len(base))
    d = np.abs(new["infected"].to_numpy()[:n] - base["infected"].to_numpy()[:n])
    max_abs = float(d.max())
    vacc_max = float(new["vaccinated"].max()) if "vaccinated" in new else 0.0

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(base["t"], base["infected"], lw=3, alpha=0.5,
               label="pre-change baseline (S->R vaccination, seed 42)")
    ax[0].plot(new["t"], new["infected"], lw=1.2, ls="--", color="k",
               label="new run, VACCINATED state, IMMUNITY_WANING_MODE=none")
    ax[0].set_ylabel("Infected fraction")
    ax[0].set_title("Test A - VACCINATED state must not change infection dynamics\n"
                    f"max |Δ infected fraction| = {max_abs:.2e}   "
                    f"(vaccinated fraction reaches {vacc_max:.3f})")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(new["t"][:n], d, color="crimson", lw=1)
    ax[1].set_ylabel("|Δ|")
    ax[1].set_xlabel("Day")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "none_vs_baseline.png"), dpi=130)
    plt.close(fig)

    # compartments of the new run (shows vaccinated is now its own bucket)
    plt.figure(figsize=(9, 6))
    for col, c in [("susceptible", "tab:blue"), ("exposed", "tab:orange"),
                   ("infected", "tab:red"), ("recovered", "tab:green"),
                   ("dead", "black"), ("vaccinated", "tab:purple")]:
        if col in new:
            plt.plot(new["t"], new[col], label=col, color=c)
    plt.xlabel("Day"); plt.ylabel("Fraction of population")
    plt.title("Test A - compartments, none mode (VACCINATED tracked separately)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "none_compartments.png"), dpi=130)
    plt.close()

    verdict = "PASS" if (max_abs < 1e-3 and vacc_max > 0) else "CHECK"
    print(f"[Test A] {verdict}  max|Δ infected| = {max_abs:.3e}  "
          f"vaccinated_max = {vacc_max:.4f}")


# ---------------------------------------------------------------------------
def _infected_fig(runs, title, out):
    plt.figure(figsize=(9, 6))
    b = load("none.csv")
    if b is not None:
        plt.plot(b["t"], b["infected"], lw=1, color="grey", alpha=0.6,
                 label="none")
    lm = {}
    for f, lbl in runs:
        df = load(f)
        if df is None:
            continue
        lm[lbl] = late_mean(df)
        plt.plot(df["t"], df["infected"], lw=2, label=lbl)
    plt.xlabel("Day"); plt.ylabel("Infected fraction")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    return lm


def _compartment_fig(runs, title, out, note=None):
    """Two panels: recovered fraction and vaccinated fraction, each with the
    no-waning `none` run as reference so waning shows as the gap below it."""
    none = load("none.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for j, col in enumerate(["recovered", "vaccinated"]):
        if none is not None:
            ax[j].plot(none["t"], none[col], lw=2.5, color="grey", alpha=0.6,
                       label="none (no waning)")
        for f, lbl in runs:
            df = load(f)
            if df is not None:
                ax[j].plot(df["t"], df[col], lw=1.8, label=lbl)
        ax[j].set_title(f"{col} fraction")
        ax[j].set_xlabel("Day"); ax[j].set_ylabel("fraction of population")
        ax[j].grid(True, alpha=0.3)
        ax[j].legend(fontsize=8)
    sup = title + (f"\n{note}" if note else "")
    fig.suptitle(sup)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def test_b():
    runs = [("fixed_40.csv", "R=40 / V=60 d"),
            ("fixed_60.csv", "R=60 / V=90 d"),
            ("fixed_80.csv", "R=80 / V=120 d")]
    if any(load(f) is None for f, _ in runs):
        print("[Test B] SKIPPED - need data/<FIPS>/fixed_{40,60,80}.csv")
        return
    lm = _infected_fig(runs, "Test B - fixed waning, independent R / V durations",
                       os.path.join(RESULTS, "fixed_infected.png"))
    _compartment_fig(
        runs, "Test B - fixed mode",
        os.path.join(RESULTS, "fixed_compartments.png"),
        note="vaccine waning starts ~ rollout(day86)+V; only V=60 is well inside "
             "the 182-day window",
    )

    none = load("none.csv")
    # vaccine inflow is identical across runs and none never wanes, so
    # none.vaccinated - run.vaccinated = cumulative vaccine immunity lost to
    # waning. It should turn positive around day 86 + V and grow.
    onsets = {}
    for f, lbl, vtime in [("fixed_40.csv", "R=40 / V=60 d", 60),
                          ("fixed_60.csv", "R=60 / V=90 d", 90),
                          ("fixed_80.csv", "R=80 / V=120 d", 120)]:
        df = load(f)
        n = min(len(df), len(none))
        deficit = none["vaccinated"].to_numpy()[:n] - df["vaccinated"].to_numpy()[:n]
        t = df["t"].to_numpy()[:n]
        on = t[deficit > 0.002]
        onsets[lbl] = (int(on[0]) if len(on) else None, float(deficit.max()), vtime)

    ordered = lm["R=40 / V=60 d"] >= lm["R=60 / V=90 d"] >= lm["R=80 / V=120 d"] - 1e-9
    rec_wanes = all(half_fall_day(load(f), "recovered") is not None for f, _ in runs)
    # V=60 must show clear in-window vaccine waning; the onset must track day86+V
    v60_on = onsets["R=40 / V=60 d"][0]
    onset_ok = v60_on is not None and 130 <= v60_on <= 165
    verdict = "PASS" if (ordered and rec_wanes and onset_ok) else "CHECK"
    print(f"[Test B] {verdict}  late-window infected: "
          + ", ".join(f"{k}={v:.2e}" for k, v in lm.items()))
    for k, (on, mx, vt) in onsets.items():
        print(f"         {k}: vaccine-waning onset (vs none) @ day {on}  "
              f"[expect ~ 86+{vt}={86+vt}], max deficit {mx:.4f}")


def test_c():
    runs = [("stochastic_0.005.csv", "R=0.005 / V=0.00333"),
            ("stochastic_0.010.csv", "R=0.010 / V=0.00667"),
            ("stochastic_0.020.csv", "R=0.020 / V=0.01333"),
            ("stochastic_0.040.csv", "R=0.040 / V=0.02667")]
    if any(load(f) is None for f, _ in runs):
        print("[Test C] SKIPPED - need data/<FIPS>/stochastic_*.csv")
        return
    lm = _infected_fig(runs, "Test C - stochastic waning, independent R / V rates",
                       os.path.join(RESULTS, "stochastic_infected.png"))
    _compartment_fig(runs, "Test C - stochastic mode",
                     os.path.join(RESULTS, "stochastic_compartments.png"))

    none = load("none.csv")
    # vaccine immunity retained relative to the no-waning run: higher V rate
    # -> lower retained fraction. Must be monotonic in the rate.
    retained = {}
    for f, lbl in runs:
        df = load(f)
        n = min(len(df), len(none))
        vn = none["vaccinated"].to_numpy()[:n]
        vr = df["vaccinated"].to_numpy()[:n]
        m = vn > 0.02
        retained[lbl] = float(np.mean(vr[m] / vn[m]))

    rv = list(retained.values())
    mono_ret = all(rv[i] >= rv[i + 1] - 1e-6 for i in range(len(rv) - 1))
    infl = list(lm.values())
    top_separates = infl[-1] > 1.3 * max(infl[:-1])
    verdict = "PASS" if (mono_ret and top_separates) else "CHECK"
    print(f"[Test C] {verdict}  late-window infected: "
          + ", ".join(f"{k.split()[0]}={v:.2e}" for k, v in lm.items())
          + "  (only the top rate separates cleanly - same as waning_tests)")
    for k, v in retained.items():
        print(f"         {k}: mean vaccinated fraction retained vs none = {v:.3f}")
    print("         -> exact per-state hazards: run wane_rate_check.py")


if __name__ == "__main__":
    test_a()
    test_b()
    test_c()
    sim_vs_actual()
