#!/usr/bin/env python3
"""Phase 2 / step 0b — reassign the `Occupations` column of the committed
`<FIPS>_population.csv` from the ACS-C24030 sector mix.

ID / Age / Household Size / Household ID are copied through **unchanged** — the
only difference from the committed population is which working-age agents get
which sector, so the household / school / random contact networks are identical
and only the occupational networks need regenerating (step 0c).

Reassignment logic mirrors `networks/custom_population.py` exactly (7 %
unemployment, per-sector count = int(frac * num_employed), leftover eligible ->
"Unemployed"), with `np.random.seed(int(FIPS))` for reproducibility.

Also appends this county's block to the industry-mix report (committed vs
C24030, sector share of employed and of total population, + essential /
non-essential / not-in-workforce headcounts).

Usage:  python targeted_cf/phase2/patch_population.py 01031 [--report <path>]
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ESSENTIAL_SECTORS = {
    "AGRICULTURE", "MINING", "UTILITIES", "TRANSPORTATION", "HEALTHCARE", "FOOD",
    "MANUFACTURING", "CONSTRUCTION", "WASTEMANAGEMENT", "OTHER", "EDUCATION",
}
NON_ESSENTIAL_SECTORS = {
    "WHOLESALETRADE", "RETAILTRADE", "INFORMATION", "FINANCEINSURANCE",
    "REALESTATERENTAL", "SCIENTIFICTECHNICAL", "ENTERPRISEMANAGEMENT", "ART",
}


def die(m):
    sys.exit(f"[patch_population] {m}")


def committed_pop_csv(fips):
    hits = glob.glob(os.path.join(REPO_ROOT, "data", "population_data", "*",
                                  f"{fips}_population.csv"))
    if len(hits) != 1:
        die(f"{fips}: expected one committed population csv, found {hits}")
    return hits[0]


def essentiality(sector):
    if sector in ESSENTIAL_SECTORS:
        return "essential"
    if sector in NON_ESSENTIAL_SECTORS:
        return "non-essential"
    return "not-in-workforce"   # NaN / Unemployed


def reassign(df, occ_counts, fips):
    df = df.reset_index(drop=True).copy()
    elig = df.index[(df["Age"] >= 18) & (df["Age"] <= 65)].to_numpy()
    total_jobs = sum(occ_counts.values())
    if total_jobs <= 0:
        die(f"{fips}: C24030 total jobs = {total_jobs}")
    num_employed = int(len(elig) * (1 - 0.07))
    occ_list = []
    for sector, cnt in occ_counts.items():
        occ_list.extend([sector] * int((cnt / total_jobs) * num_employed))
    if len(occ_list) >= len(elig):
        die(f"{fips}: assigned occ list ({len(occ_list)}) >= eligible ({len(elig)})")
    np.random.seed(int(fips))
    elig_shuf = elig.copy()
    np.random.shuffle(elig_shuf)
    new_occ = np.full(len(df), None, dtype=object)   # NaN -> None for non-working-age
    for pos, sector in zip(elig_shuf[:len(occ_list)], occ_list):
        new_occ[pos] = sector
    for pos in elig_shuf[len(occ_list):]:
        new_occ[pos] = "Unemployed"
    df["Occupations"] = new_occ
    return df


def mix_block(fips, df_old, df_new):
    n = len(df_new)
    lines = [f"================ {fips} ================",
             f"  total agents = {n}",
             ""]
    for label, df in (("COMMITTED (2017 ecnbasic)", df_old),
                      ("C24030 (ACS 5yr, phase 2)", df_new)):
        vc = df["Occupations"].value_counts(dropna=False)
        employed = int(df["Occupations"].notna().sum() - vc.get("Unemployed", 0))
        lines.append(f"  --- {label} ---")
        lines.append(f"      {'sector':22s} {'count':>8s} {'% employed':>11s} {'% total pop':>12s}")
        for sec, c in vc.items():
            if pd.isna(sec) or sec == "Unemployed":
                continue
            pe = 100 * c / employed if employed else 0.0
            lines.append(f"      {sec:22s} {c:8d} {pe:10.1f}% {100 * c / n:11.1f}%")
        # essential / non-essential / not-in-workforce
        grp = df["Occupations"].map(lambda s: essentiality(s if isinstance(s, str) else np.nan))
        gc = grp.value_counts()
        for g in ("essential", "non-essential", "not-in-workforce"):
            c = int(gc.get(g, 0))
            pe = 100 * c / employed if (employed and g != "not-in-workforce") else float("nan")
            pe_s = f"{pe:10.1f}%" if pe == pe else " " * 11
            lines.append(f"      {g:22s} {c:8d} {pe_s} {100 * c / n:11.1f}%")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fips")
    ap.add_argument("--report", default=None,
                    help="industry_mix.txt to append to (default: "
                         "results/TargetedCF_run/essentiality/industry_mix.txt)")
    a = ap.parse_args()
    fips = a.fips

    src = committed_pop_csv(fips)
    occ_csv = os.path.join(REPO_ROOT, "targeted_cf", "phase2", "data",
                           f"agents_occupations_{fips}.csv")
    if not os.path.isfile(occ_csv):
        die(f"{occ_csv} missing — run fetch_industry_c24030.py {fips} first")

    df_old = pd.read_csv(src)
    occ_df = pd.read_csv(occ_csv)
    occ_counts = dict(zip(occ_df["Occupation"], occ_df["Number"].astype(int)))

    df_new = reassign(df_old, occ_counts, fips)

    # invariants: everything except Occupations must be untouched
    for col in ("ID", "Age", "Household Size", "Household ID"):
        if not df_old[col].equals(df_new[col]):
            die(f"{fips}: column {col!r} changed — reassignment bug")
    if len(df_old) != len(df_new):
        die(f"{fips}: row count changed {len(df_old)} -> {len(df_new)}")

    out_dir = os.path.join(REPO_ROOT, "targeted_cf", "phase2", "populations")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{fips}_population.csv")
    df_new.to_csv(out, index=False)
    print(f"[patch_population] {fips}: {src}\n  -> {out}  ({len(df_new)} rows, "
          f"Occupations reassigned from C24030)")

    report = a.report or os.path.join(
        REPO_ROOT, "results", "TargetedCF_run", "essentiality", "industry_mix.txt")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "a") as f:
        f.write(mix_block(fips, df_old, df_new))
    print(f"[patch_population] industry mix appended to {report}")


if __name__ == "__main__":
    main()
