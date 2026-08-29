#!/usr/bin/env python3
"""Patch the immunity-waning keys in covid_abm/yamls/config.yaml in place.

This is the ONLY thing the waning tests change before running the existing
pipeline (`python3 main.py <fips>`); everything else in config.yaml is left
byte-for-byte unchanged (targeted line replacement, no YAML re-dump).

Usage:
    python3 waning_tests/set_waning_config.py --mode none
    python3 waning_tests/set_waning_config.py --mode fixed --recovered-to-susceptible-time 80
    python3 waning_tests/set_waning_config.py --mode stochastic --waning-rate 0.02
"""
import argparse
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "covid_abm", "yamls", "config.yaml")


def _replace_key(text, key, value):
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.MULTILINE)
    if not pat.search(text):
        raise SystemExit(f"[set_waning_config] key not found in config: {key}")
    return pat.sub(rf"\g<1>{key}: {value}", text, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["none", "fixed", "stochastic"])
    ap.add_argument("--recovered-to-susceptible-time", type=int, default=None)
    ap.add_argument("--waning-rate", type=float, default=None)
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    with open(args.config) as f:
        text = f.read()

    text = _replace_key(text, "IMMUNITY_WANING_MODE", f'"{args.mode}"')
    if args.recovered_to_susceptible_time is not None:
        text = _replace_key(text, "RECOVERED_TO_SUSCEPTIBLE_TIME",
                            args.recovered_to_susceptible_time)
    if args.waning_rate is not None:
        text = _replace_key(text, "WANING_RATE", args.waning_rate)

    with open(args.config, "w") as f:
        f.write(text)

    print(f"[set_waning_config] mode={args.mode} "
          f"RECOVERED_TO_SUSCEPTIBLE_TIME={args.recovered_to_susceptible_time} "
          f"WANING_RATE={args.waning_rate}")


if __name__ == "__main__":
    main()
