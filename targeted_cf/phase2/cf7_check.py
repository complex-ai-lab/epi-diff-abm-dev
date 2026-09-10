#!/usr/bin/env python3
"""Phase 2 / step 0f — decide REUSE vs RECALIBRATE per county.

Reads the CF7 run done on the essnet nets with the reused log1p_rw2_l5_rmax5
params and compares it to the actual case series. Appends a line per county to
results/TargetedCF_run/<TAG>/cf7_check.txt and prints REUSE / RECALIBRATE.

Threshold (either fails -> RECALIBRATE):
  * |cf7_total - actual_total| / actual_total  <= 0.15
  * rmse_all_14  <=  1.6 x  the committed log1p_rw2_l5_rmax5 rmse_all_14

Usage:  python targeted_cf/phase2/cf7_check.py 01031 --tag essentiality \
            --cfdir results/<FIPS>/TargetedCF/essentiality_p2check/counterfactuals
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# committed rmax5 CF7 rmse_all_14 (results/<FIPS>/LossCompare/log1p_rw2_l5_rmax5)
COMMITTED_RMSE = {"01031": 15.35, "17117": 13.69, "36003": 11.16}


def die(m):
    sys.exit(f"[cf7_check] {m}")


def load_cf7(cfdir):
    for c in (os.path.join(cfdir, "cf_raw_7.csv"),
              os.path.join(cfdir, "all_counterfactual_results",
                           "07_timevar_factual_school_factual_work", "cf_raw_7.csv")):
        if os.path.isfile(c):
            r = np.loadtxt(c, delimiter=",")
            return r if r.ndim == 2 else r[None, :]
    die(f"cf_raw_7.csv not found under {cfdir}")


def load_actual(cfdir):
    d = os.path.join(cfdir, "all_counterfactual_results",
                     "07_timevar_factual_school_factual_work", "data")
    hits = glob.glob(os.path.join(d, "*_counterfactual_data7.csv"))
    if not hits:
        die(f"actual-cases file not found under {d}")
    return pd.read_csv(hits[0])["actual_cases"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fips")
    ap.add_argument("--tag", default="essentiality")
    ap.add_argument("--cfdir", required=True)
    a = ap.parse_args()

    r = load_cf7(a.cfdir)
    actual = load_actual(a.cfdir)
    m = r.mean(0)
    if m.shape != actual.shape:
        die(f"{a.fips}: cf7 mean {m.shape} != actual {actual.shape}")

    cf7_total = float(m.sum())
    act_total = float(actual.sum())
    total_rel = abs(cf7_total - act_total) / act_total
    rmse14 = float(np.sqrt(np.mean((actual[14:] - m[14:]) ** 2)))
    ref = COMMITTED_RMSE.get(a.fips, rmse14)

    verdict = "REUSE"
    if total_rel > 0.15 or rmse14 > 1.6 * ref:
        verdict = "RECALIBRATE"

    line = (f"{a.fips}  {verdict}  cf7_total={cf7_total:.0f} actual_total={act_total:.0f} "
            f"rel={total_rel:.3f}  rmse_all_14={rmse14:.2f} (committed ref {ref:.2f})")
    out = os.path.join(REPO_ROOT, "results", "TargetedCF_run", a.tag, "cf7_check.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write(line + "\n")
    print(f"[cf7_check] {line}")
    print(verdict)   # last stdout line -> shell captures it


if __name__ == "__main__":
    main()
