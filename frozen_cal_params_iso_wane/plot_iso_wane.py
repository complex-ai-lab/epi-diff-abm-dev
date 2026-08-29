#!/usr/bin/env python3
"""Plot the frozen-param isolated-waning sweep.

Reads frozen_cal_params_iso_wane/data/*.csv (columns t,susceptible,exposed,
infected,recovered,dead) and writes:

    frozen_cal_params_iso_wane/fixed_iso.png
    frozen_cal_params_iso_wane/stochastic_iso.png

All curves in a figure share one calibrated parameter vector and one RNG seed,
so they overlay exactly until the first recovered agents appear; later spread
is the pure waning effect. A vertical marker shows the day the curves first
separate.
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
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def infected(df):
    return df["t"].to_numpy(), df["infected"].to_numpy()


def first_divergence_day(t, y_ref, y_other, tol=1e-9):
    d = np.abs(y_ref - y_other)
    idx = np.where(d > tol)[0]
    return int(t[idx[0]]) if len(idx) else None


def sustained_divergence_day(t, y_ref, y_other, tol=5e-4):
    """First day |Δ| exceeds `tol` (~24 agents) and stays above it.

    Ignores the 1-2 agent RNG-stream-shift blips that stochastic mode causes
    from t=0 (it calls torch.rand_like every step); marks where a real,
    mechanism-driven gap opens.
    """
    d = np.abs(y_ref - y_other)
    above = d > tol
    for i in range(len(above)):
        if above[i] and above[i:].mean() > 0.8:
            return int(t[i])
    return None


def late_mean(t, y, window=40):
    return float(np.mean(y[t >= t.max() - window]))


def make_fig(runs, none_df, title, out, ref_label):
    plt.figure(figsize=(9, 6))
    tn, yn = infected(none_df)
    plt.plot(tn, yn, lw=1.4, color="grey", alpha=0.7, label="none (frozen params)")

    div_days = []
    print(f"\n=== {title} ===")
    print(f"  {'config':26s} {'peak':>9s} {'peak_day':>9s} "
          f"{'late40':>9s}  {'exact≠@':>8s} {'visible≠@':>10s}")
    print(f"  {'none':26s} {yn.max():9.5f} {tn[yn.argmax()]:9.0f} "
          f"{late_mean(tn, yn):9.5f}  {'-':>8s} {'-':>10s}")
    for fname, lbl in runs:
        df = load(fname)
        if df is None:
            print(f"  [missing] {fname}")
            continue
        t, y = infected(df)
        n = min(len(t), len(tn))
        exact = first_divergence_day(t[:n], yn[:n], y[:n])
        vis = sustained_divergence_day(t[:n], yn[:n], y[:n])
        if vis is not None:
            div_days.append(vis)
        plt.plot(t, y, lw=2, label=lbl)
        print(f"  {lbl:26s} {y.max():9.5f} {t[y.argmax()]:9.0f} "
              f"{late_mean(t, y):9.5f}  {str(exact):>8s} {str(vis):>10s}")

    if div_days:
        dd = min(div_days)
        plt.axvline(dd, color="k", ls=":", lw=1, alpha=0.6)
        plt.text(dd + 2, plt.ylim()[1] * 0.95,
                 f"curves separate ~day {dd}\n(|Δ| > ~24 agents)",
                 fontsize=9, va="top")

    plt.xlabel("Day")
    plt.ylabel("Infected fraction of population")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  -> {out}")


def main():
    none_df = load("none.csv")
    if none_df is None:
        raise SystemExit("missing data/none.csv - run run_iso_wane.sh first")

    make_fig(
        [("fixed_60.csv", "fixed  R->S after 60 d"),
         ("fixed_80.csv", "fixed  R->S after 80 d"),
         ("fixed_100.csv", "fixed  R->S after 100 d")],
        none_df,
        "Isolated waning (frozen calibrated params) - fixed R->S delay",
        os.path.join(HERE, "fixed_iso.png"),
        "none",
    )

    make_fig(
        [("stochastic_0.005.csv", "stochastic  rate 0.005 (~200 d)"),
         ("stochastic_0.010.csv", "stochastic  rate 0.010 (~100 d)"),
         ("stochastic_0.020.csv", "stochastic  rate 0.020 (~50 d)"),
         ("stochastic_0.040.csv", "stochastic  rate 0.040 (~25 d)")],
        none_df,
        "Isolated waning (frozen calibrated params) - stochastic daily p = 1-exp(-rate)",
        os.path.join(HERE, "stochastic_iso.png"),
        "none",
    )


if __name__ == "__main__":
    main()
