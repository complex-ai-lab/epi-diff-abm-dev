#!/usr/bin/env python3
"""Weekly scaling-factor R(t) trend for the vaccinated_test calibrations.

Same view as the state-level "Scaling Factor (R) Trend Over Time" figure, but
for a single county and the vaccinated_test frozen calibrations. Reads
vaccinated_test/data/<FIPS>/<label>_params.txt (26 weekly R, then initial
infection rate, then k) and writes:

    vaccinated_test/results/<FIPS>/<label>_R_trend.png

For context the committed baseline calibration this CF set is kept comparable to
(result_graphs/<FIPS>/<DATE>/0.0005_3_5_True_True_False_metro_0) is drawn as a
faint reference line.

Usage:
    python vaccinated_test/plot_R_trend.py                 # 01031, label=fixed_60
    python vaccinated_test/plot_R_trend.py 36003           # another county
    R_TREND_LABEL=fixed_80 python vaccinated_test/plot_R_trend.py
"""
import os
import sys
import datetime as dt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

FIPS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VACC_TEST_FIPS", "01031")
DATE = os.environ.get("VACC_TEST_DATE", "202010-202104")
LABEL = os.environ.get("R_TREND_LABEL", "fixed_60")
NUM_WEEKS = 26
WINDOW_START = dt.date(2020, 10, 26)          # day 0 of the calibration window

DATA = os.path.join(HERE, "data", FIPS)
RESULTS = os.path.join(HERE, "results", FIPS)
os.makedirs(RESULTS, exist_ok=True)

# colour-blind-safe: bold blue for the run, muted grey for the reference
C_RUN = "#1b6ec2"
C_REF = "#8a8f98"


def load_R(path):
    v = np.loadtxt(path).ravel()
    return v[:NUM_WEEKS], float(v[-2]) * 1e5, float(v[-1])


def main():
    run_path = os.path.join(DATA, f"{LABEL}_params.txt")
    if not os.path.exists(run_path):
        raise SystemExit(f"[error] not found: {run_path}")
    R, init100k, k = load_R(run_path)

    ref_path = os.path.join(
        REPO, "result_graphs", FIPS, DATE,
        "0.0005_3_5_True_True_False_metro_0", "calibrated_params.txt",
    )
    ref = load_R(ref_path) if os.path.exists(ref_path) else None

    week_dates = [WINDOW_START + dt.timedelta(days=7 * i) for i in range(NUM_WEEKS)]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    if ref is not None:
        ax.step(week_dates, ref[0], where="mid", color=C_REF, lw=1.8,
                label=f"baseline metro_0  (mean R {ref[0].mean():.2f})")
        ax.plot(week_dates, ref[0], "o", ms=4, color=C_REF, alpha=0.5)

    ax.step(week_dates, R, where="mid", color=C_RUN, lw=2.4,
            label=f"{LABEL} calibration  (mean R {R.mean():.2f})")
    ax.plot(week_dates, R, "o", ms=6, color=C_RUN)

    ax.axhline(1.0, color="#c04040", lw=1.0, ls="--", alpha=0.8)
    ax.annotate("R = 1", (week_dates[-1], 1.0), xytext=(4, 2),
                textcoords="offset points", fontsize=8, color="#c04040",
                va="bottom", ha="left")

    imax = int(np.argmax(R))
    ax.annotate(f"peak {R[imax]:.2f}", (week_dates[imax], R[imax]),
                xytext=(10, -12), textcoords="offset points", ha="left",
                fontsize=8, color=C_RUN)

    ax.set_title(f"Weekly scaling factor R(t) — FIPS {FIPS}, {LABEL} "
                 f"(vaccinated-state calibration)", fontsize=11)
    ax.set_xlabel("Calibration week")
    ax.set_ylabel("Scaling factor R")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, max(4.0, R.max() * 1.15))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    txt = (f"initial infection rate = {init100k:.1f} per 100k\n"
           f"k (reporting fraction) = {k:.3f}\n"
           f"R range = [{R.min():.2f}, {R.max():.2f}]")
    ax.text(0.30, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))

    fig.tight_layout()
    out = os.path.join(RESULTS, f"{LABEL}_R_trend.png")
    fig.savefig(out, dpi=140)
    print(f"[R-trend] {LABEL}: mean R {R.mean():.3f}  range [{R.min():.2f}, "
          f"{R.max():.2f}]  init {init100k:.1f}/100k  k {k:.3f}")
    print(f"[R-trend] wrote {out}")


if __name__ == "__main__":
    main()
