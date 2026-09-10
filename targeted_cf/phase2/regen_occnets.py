#!/usr/bin/env python3
"""Phase 2 / step 0c — rebuild ONLY the occupational contact networks
(occnets/<t>.pkl, t = 0..181) for the C24030-patched population.

Faithfully replicates the occupation-network loop of `networks/gen_mob_nw.py`
(same Google `workplaces_percent_change_from_baseline` handling, start date
2020-10-12, year rollover, `custom_watts_strogatz_graph` per sector, merge).
Household / school / random networks are occupation-independent and are NOT
rebuilt (step 0d symlinks them).

`random` and `np.random` are seeded to int(FIPS) for reproducibility.

Usage:  python targeted_cf/phase2/regen_occnets.py 01031
        NUM_STEPS=182 (default), NETWORKS_ESSNET=<scratch essnet root> (default from .env sibling)
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_DIR = os.path.join(REPO_ROOT, "networks")
sys.path.insert(0, NET_DIR)
from sim_gen_utils import custom_watts_strogatz_graph          # noqa: E402
from isolate_county_data import get_dataframes_dict             # noqa: E402

OCC_PARAMS = os.path.join(NET_DIR, "network_parameters", "occupation_network_parameters.csv")
START_DATE = "2020-10-12"


def die(m):
    sys.exit(f"[regen_occnets] {m}")


def essnet_root():
    v = os.environ.get("NETWORKS_ESSNET")
    if v:
        return v
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    causal = os.environ["NETWORKS_DIR"]                 # .../networks/covid_output_causal
    return os.path.join(os.path.dirname(causal), "covid_output_essnet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fips")
    a = ap.parse_args()
    fips = a.fips
    num_steps = int(os.environ.get("NUM_STEPS", "182"))

    random.seed(int(fips))
    np.random.seed(int(fips))

    pop_csv = os.path.join(REPO_ROOT, "targeted_cf", "phase2", "populations",
                           f"{fips}_population.csv")
    if not os.path.isfile(pop_csv):
        die(f"{pop_csv} missing — run patch_population.py {fips} first")
    individuals = pd.read_csv(pop_csv)

    occ_params = pd.read_csv(OCC_PARAMS, index_col="Occupation")
    occ_names = occ_params.index.to_list()
    occ_to_ix = {name: i for i, name in enumerate(occ_names)}

    individuals["OccupationID"] = individuals["Occupations"].map(occ_to_ix)
    with_occ = individuals.dropna(subset=["OccupationID"])
    occ_groups = {
        occ: with_occ[with_occ["OccupationID"] == occ]["ID"].tolist()
        for occ in occ_to_ix.values()
    }
    present = {occ_names[o]: len(ids) for o, ids in occ_groups.items() if ids}
    print(f"[regen_occnets] {fips}: {sum(present.values())} workers across "
          f"{len(present)} sectors: {present}")

    goog = get_dataframes_dict(fips)
    year = 2020
    curr = goog[year]
    if START_DATE not in curr["date"].values:
        die(f"{fips}: {START_DATE} not in 2020 Google mobility data")
    index = curr.index[curr["date"] == START_DATE][0]

    out_dir = os.path.join(essnet_root(), fips, "mobility_networks", "occnets")
    os.makedirs(out_dir, exist_ok=True)

    for t in range(num_steps):
        if index == curr.index[-1] + 1:
            year += 1
            if year not in goog:
                die(f"{fips}: ran out of Google mobility data at t={t} (year {year})")
            curr = goog[year]
            index = 0
        wpc = curr.at[index, "workplaces_percent_change_from_baseline"]
        occ_factor = 1.0 + (0.0 if pd.isna(wpc) else wpc / 100.0)
        index += 1

        G_all = nx.Graph()
        for occ, agents in occ_groups.items():
            n = len(agents)
            if n <= 1:
                continue
            mu = occ_params.loc[occ_names[occ], "mu"] * occ_factor
            rewire = occ_params.loc[occ_names[occ], "rewire"]
            avg_degree = min(int(np.round(mu)), n - 1)
            G_occ = custom_watts_strogatz_graph(n=n, k=avg_degree, p=[rewire, agents])
            G_all.add_nodes_from(G_occ.nodes(data=True))
            G_all.add_edges_from(G_occ.edges())
        with open(os.path.join(out_dir, f"{t}.pkl"), "wb") as f:
            pickle.dump(G_all, f)

    written = len([f for f in os.listdir(out_dir) if f.endswith(".pkl")])
    if written != num_steps:
        die(f"{fips}: wrote {written} occnet pkls, expected {num_steps}")
    print(f"[regen_occnets] {fips}: {num_steps} occnets -> {out_dir}")


if __name__ == "__main__":
    main()
