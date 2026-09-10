#!/usr/bin/env python3
"""Phase 2 / step 0a — re-source county industry counts from ACS 5-year C24030
("Sex by Industry for the civilian employed population 16 years and over").

Writes  targeted_cf/phase2/data/agents_occupations_<FIPS>.csv  in the same shape
as the committed data/state_data/<ST>/<FIPS>/agents_occupations.csv
("Occupation,Number", one row per the repo's 19 NAICS sectors).

Full 19-sector coverage (unlike the committed 2017 `ecnbasic` pull). Fails loud
on HTTP != 200, a missing variable, or > 3 sectors coming back 0.

Usage:  python targeted_cf/phase2/fetch_industry_c24030.py 01031 [--year 2021]
"""
import argparse
import os
import sys
import requests
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

# C24030 MALE index -> repo sector.  FEMALE index = MALE index + 27.
# OTHER = "other services, except public administration" (027) + "public
# administration" (028), matching scripts/census.py's inverted_naics_sectors.
MALE_IDX = {
    "AGRICULTURE": [4], "MINING": [5], "CONSTRUCTION": [6], "MANUFACTURING": [7],
    "WHOLESALETRADE": [8], "RETAILTRADE": [9], "TRANSPORTATION": [11], "UTILITIES": [12],
    "INFORMATION": [13], "FINANCEINSURANCE": [15], "REALESTATERENTAL": [16],
    "SCIENTIFICTECHNICAL": [18], "ENTERPRISEMANAGEMENT": [19], "WASTEMANAGEMENT": [20],
    "EDUCATION": [22], "HEALTHCARE": [23], "ART": [25], "FOOD": [26],
    "OTHER": [27, 28],
}
# preserve the committed file's sector order
SECTOR_ORDER = [
    "AGRICULTURE", "MINING", "UTILITIES", "CONSTRUCTION", "MANUFACTURING",
    "WHOLESALETRADE", "RETAILTRADE", "TRANSPORTATION", "INFORMATION",
    "FINANCEINSURANCE", "REALESTATERENTAL", "SCIENTIFICTECHNICAL",
    "ENTERPRISEMANAGEMENT", "WASTEMANAGEMENT", "EDUCATION", "HEALTHCARE",
    "ART", "FOOD", "OTHER",
]


def fetch(fips, year):
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        sys.exit("[fetch_industry_c24030] CENSUS_API_KEY not in .env")
    idxs = sorted({i for lst in MALE_IDX.values() for i in lst} |
                  {i + 27 for lst in MALE_IDX.values() for i in lst})
    varstr = ",".join(f"C24030_{n:03d}E" for n in idxs)
    url = (f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{varstr}"
           f"&for=county:{fips[2:]}&in=state:{fips[:2]}&key={key}")
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        sys.exit(f"[fetch_industry_c24030] {fips}: HTTP {r.status_code} {r.reason}\n{r.text[:300]}")
    hdr, row = r.json()[0], r.json()[1]
    d = {}
    for h, v in zip(hdr, row):
        if h.startswith("C24030"):
            if v is None:
                sys.exit(f"[fetch_industry_c24030] {fips}: null value for {h}")
            d[h] = int(v)
    counts = {}
    for sector in SECTOR_ORDER:
        tot = 0
        for mi in MALE_IDX[sector]:
            tot += d[f"C24030_{mi:03d}E"] + d[f"C24030_{mi + 27:03d}E"]
        counts[sector] = tot
    n_zero = sum(1 for c in counts.values() if c == 0)
    if n_zero > 3:
        sys.exit(f"[fetch_industry_c24030] {fips}: {n_zero} sectors are 0 "
                 f"({[s for s, c in counts.items() if c == 0]}) — data looks wrong")
    return row[0], counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fips")
    ap.add_argument("--year", type=int, default=2021)
    a = ap.parse_args()

    name, counts = fetch(a.fips, a.year)
    out_dir = os.path.join(REPO_ROOT, "targeted_cf", "phase2", "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"agents_occupations_{a.fips}.csv")
    with open(out, "w") as f:
        f.write("Occupation,Number\n")
        for s in SECTOR_ORDER:
            f.write(f"{s},{counts[s]}\n")
    tot = sum(counts.values())
    print(f"[fetch_industry_c24030] {a.fips} ({name})  ACS{a.year} C24030  "
          f"total employed 16+ = {tot}")
    for s in sorted(SECTOR_ORDER, key=lambda x: -counts[x]):
        print(f"    {s:22s} {counts[s]:7d}  {100 * counts[s] / tot:5.1f}%")
    print(f"[fetch_industry_c24030] -> {out}")


if __name__ == "__main__":
    main()
