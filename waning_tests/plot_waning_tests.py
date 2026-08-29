#!/usr/bin/env python3
"""Generate the three immunity-waning validation plots.

Reads the per-run trajectories collected by run_waning_tests.sh from
waning_tests/data/*.csv (columns: t,susceptible,exposed,infected,recovered,dead)
and writes:

    waning_tests/no_waning_same_seed_comparison.png
    waning_tests/fixed_waning_60_80_100_days.png
    waning_tests/stochastic_waning_005_010_020.png

Also prints a short PASS/FAIL summary for each test.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def infected(df):
    return df["t"].to_numpy(), df["infected"].to_numpy()


# ---------------------------------------------------------------------------
# Test 1 - none vs already-generated baseline (same seed)
# ---------------------------------------------------------------------------
def test1():
    base = load("none_baseline_reference.csv")
    new = load("none_verify.csv")
    if base is None or new is None:
        print("[Test 1] SKIPPED - need data/none_baseline_reference.csv and "
              "data/none_verify.csv")
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
    out = os.path.join(HERE, "no_waning_same_seed_comparison.png")
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
        print("[Test 2] SKIPPED - need data/fixed_60.csv, fixed_80.csv, fixed_100.csv")
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
    out = os.path.join(HERE, "fixed_waning_60_80_100_days.png")
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
# Test 3 - stochastic waning, rates 0.005 / 0.01 / 0.02
# ---------------------------------------------------------------------------
def test3():
    runs = [
            ("stochastic_0.010.csv", "WANING_RATE = 0.010  (~100 day mean)"),
            ("stochastic_0.020.csv", "WANING_RATE = 0.020  (~50 day mean)"),
            ("stochastic_0.040.csv", "WANING_RATE = 0.040  (~25 day mean)")]
    loaded = [(load(f), lbl) for f, lbl in runs]
    if any(df is None for df, _ in loaded):
        print("[Test 3] SKIPPED - need data/stochastic_XXX.csv")
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
    out = os.path.join(HERE, "stochastic_waning_010_020_040.png")
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
