#!/usr/bin/env python3
"""Phase 2 / step 0d — assemble the isolated essnet mobility_networks dir.

occnets/  = the real regenerated files (step 0c).
HOUSEHOLD_NETWORK.pkl, SCHOOL_NETWORK.pkl, randnets/  = SYMLINKS to the sibling
committed covid_output_causal (occupation-independent, unchanged).
<FIPS>_residence_commute_data.txt = copied from causal if present (unused at
phase 0, harmless).

Usage:  python targeted_cf/phase2/assemble_essnet.py 01031
"""
import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def die(m):
    sys.exit(f"[assemble_essnet] {m}")


def roots():
    v = os.environ.get("NETWORKS_ESSNET")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    causal = os.environ["NETWORKS_DIR"]
    essnet = v or os.path.join(os.path.dirname(causal), "covid_output_essnet")
    return causal, essnet


def link(src, dst):
    if not os.path.exists(src):
        die(f"source missing: {src}")
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst) if (os.path.islink(dst) or os.path.isfile(dst)) else shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f"  {os.path.basename(dst):22s} -> {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fips")
    a = ap.parse_args()
    fips = a.fips
    causal, essnet = roots()

    src = os.path.join(causal, fips, "mobility_networks")
    dst = os.path.join(essnet, fips, "mobility_networks")
    if not os.path.isdir(src):
        die(f"committed causal nets not found: {src}")
    occ_dst = os.path.join(dst, "occnets")
    if not os.path.isdir(occ_dst) or len([f for f in os.listdir(occ_dst) if f.endswith(".pkl")]) < 1:
        die(f"{occ_dst} has no regenerated occnets — run regen_occnets.py {fips} first")

    print(f"[assemble_essnet] {fips}: {dst}")
    link(os.path.join(src, "HOUSEHOLD_NETWORK.pkl"), os.path.join(dst, "HOUSEHOLD_NETWORK.pkl"))
    link(os.path.join(src, "SCHOOL_NETWORK.pkl"), os.path.join(dst, "SCHOOL_NETWORK.pkl"))
    link(os.path.join(src, "randnets"), os.path.join(dst, "randnets"))

    rc = os.path.join(src, "occnets", f"{fips}_residence_commute_data.txt")
    if os.path.isfile(rc):
        shutil.copy(rc, os.path.join(occ_dst, f"{fips}_residence_commute_data.txt"))
        print(f"  copied {fips}_residence_commute_data.txt")

    # sanity: essnet occnets count == causal randnets count
    n_occ = len([f for f in os.listdir(occ_dst) if f.endswith(".pkl")])
    n_rand = len([f for f in os.listdir(os.path.join(src, "randnets")) if f.endswith(".pkl")])
    if n_occ != n_rand:
        die(f"{fips}: {n_occ} occnets != {n_rand} randnets")
    print(f"[assemble_essnet] {fips}: OK ({n_occ} occnets, symlinked household/school/randnets)")


if __name__ == "__main__":
    main()
