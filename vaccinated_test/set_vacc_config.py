#!/usr/bin/env python3
"""Patch the immunity-waning keys (incl. the VACCINATED-state keys) in
covid_abm/yamls/config.yaml in place.

Targeted line replacement, no YAML re-dump: every other line is left
byte-for-byte unchanged. Mirrors waning_tests/set_waning_config.py.

Usage:
    python3 vaccinated_test/set_vacc_config.py --mode none
    python3 vaccinated_test/set_vacc_config.py --mode fixed \
        --recovered-to-susceptible-time 80 --vaccinated-to-susceptible-time 120
    python3 vaccinated_test/set_vacc_config.py --mode stochastic \
        --recovered-waning-rate 0.02 --vaccinated-waning-rate 0.013333
"""
import argparse
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "covid_abm", "yamls", "config.yaml")


def _replace_key(text, key, value):
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.MULTILINE)
    if not pat.search(text):
        raise SystemExit(f"[set_vacc_config] key not found in config: {key}")
    return pat.sub(rf"\g<1>{key}: {value}", text, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["none", "fixed", "stochastic"])
    ap.add_argument("--recovered-to-susceptible-time", type=int, default=None)
    ap.add_argument("--vaccinated-to-susceptible-time", type=int, default=None)
    ap.add_argument("--recovered-waning-rate", type=float, default=None)
    ap.add_argument("--vaccinated-waning-rate", type=float, default=None)
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    with open(args.config) as f:
        text = f.read()

    text = _replace_key(text, "IMMUNITY_WANING_MODE", f'"{args.mode}"')
    if args.recovered_to_susceptible_time is not None:
        text = _replace_key(text, "RECOVERED_TO_SUSCEPTIBLE_TIME",
                            args.recovered_to_susceptible_time)
    if args.vaccinated_to_susceptible_time is not None:
        text = _replace_key(text, "VACCINATED_TO_SUSCEPTIBLE_TIME",
                            args.vaccinated_to_susceptible_time)
    if args.recovered_waning_rate is not None:
        text = _replace_key(text, "RECOVERED_WANING_RATE",
                            args.recovered_waning_rate)
        # keep the deprecated alias in sync so nothing reads a stale value
        text = _replace_key(text, "WANING_RATE", args.recovered_waning_rate)
    if args.vaccinated_waning_rate is not None:
        text = _replace_key(text, "VACCINATED_WANING_RATE",
                            args.vaccinated_waning_rate)

    with open(args.config, "w") as f:
        f.write(text)

    print(f"[set_vacc_config] mode={args.mode} "
          f"R_to_S={args.recovered_to_susceptible_time} "
          f"V_to_S={args.vaccinated_to_susceptible_time} "
          f"R_rate={args.recovered_waning_rate} "
          f"V_rate={args.vaccinated_waning_rate}")


if __name__ == "__main__":
    main()
